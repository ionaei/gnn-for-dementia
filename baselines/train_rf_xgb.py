"""
Train and evaluate Random Forest and XGBoost baseline models.

Implements the baseline methodology from the paper:
- Random Forest and XGBoost models (Section 3.2, "Baseline")
- SelectFromModel feature selection (top 25% of features ranked by importance)
- Bayesian hyperparameter search with 5-fold cross-validation (Appendix A4)
- 70/10/20 stratified train/val/test split (matching the GNN pipeline in SCHEMA.md)
- Evaluation metrics: Accuracy, Sensitivity, Specificity, AUROC

Hyperparameter search spaces are from the paper's Appendix A4, Table A2.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import feature_engineering as fe

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from data_split import add_label_column, split_dataframe  # noqa: E402
from metrics import compute_metrics  # noqa: E402

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Try to import wandb, but allow graceful degradation if not available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Label encoding: Dementia=0, Control=1
# Sensitivity = recall for class 0 (Dementia)
# Specificity = recall for class 1 (Control)
#
# `compute_metrics` is now imported from common/metrics.py (the unified
# convention shared with gnn/ and bert_models/) instead of a local
# reimplementation. The local version only computed accuracy/sensitivity/
# specificity/AUROC -- no F1 and no AUPRC -- which is why this script's
# Table 1 row was missing an F1 entry. `common/metrics.py`'s
# `compute_metrics(y_true, y_pred, probs_control)` has the exact same
# argument meaning (`probs_control` = P(class 1) = P(control), i.e. what
# this file was already passing as `y_pred_proba`), so no call-site
# reshaping is needed -- only the richer return dict (adds precision, f1,
# J, auprc, and raw tp/fn/fp/tn) and the shared cross-model convention.


def train_and_evaluate_rf(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    use_wandb: bool = False,
) -> dict:
    """
    Train Random Forest with feature selection and hyperparameter search.

    Following Appendix A4, Table A2:
    - min_child_weight: uniform {1, 2, ..., 10}
    - subsample_ratio: uniform [0.4, 1.0]
    - column_subsample_ratio: uniform [0.4, 1.0]
    - max_depth: uniform {3, 4, ..., 10}

    Using BayesSearchCV with 5-fold CV; if not available, fall back to RandomizedSearchCV.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        X_test, y_test: Test data.
        feature_names: List of feature names.
        use_wandb: Whether to log to wandb.

    Returns:
        Dictionary with model, metrics, selected features, and hyperparameters.
    """
    logger.info("Training Random Forest baseline...")

    # Step 1: Train initial RF on full feature set to identify top 25% features
    rf_initial = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )
    rf_initial.fit(X_train, y_train)

    # Step 2: SelectFromModel to select top 25% of features
    # threshold=-np.inf forces SelectFromModel to rank ALL features by
    # importance and keep exactly the top `max_features`. Without an
    # explicit threshold, SelectFromModel's default ("mean") first drops
    # every feature at or below the mean importance, THEN caps at
    # max_features -- so it silently returns "at most the top 25%, and
    # often far fewer" rather than "the top 25%" the paper describes.
    selector = SelectFromModel(
        rf_initial,
        prefit=True,
        threshold=-np.inf,
        max_features=max(1, int(0.25 * X_train.shape[1])),
    )
    X_train_selected = selector.transform(X_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)

    selected_feature_indices = selector.get_support(indices=True)
    selected_feature_names = [feature_names[i] for i in selected_feature_indices]

    logger.info(
        f"Selected {len(selected_feature_names)} / {len(feature_names)} features "
        f"(top 25%): {selected_feature_names}"
    )

    # Step 3: Hyperparameter search over selected features
    # Appendix A4, Table A2: For RF/XGBoost, use Bayesian search with 5-fold CV
    try:
        from skopt import BayesSearchCV
        BAYES_AVAILABLE = True
    except ImportError:
        logger.warning(
            "BayesSearchCV not available; falling back to RandomizedSearchCV. "
            "Install scikit-optimize for exact paper reproduction."
        )
        BAYES_AVAILABLE = False

    if BAYES_AVAILABLE:
        # Bayesian hyperparameter search space (Appendix A4, Table A2)
        search_space = {
            "n_estimators": (100, 500),
            "max_depth": (3, 10),
            "min_samples_split": (2, 20),
            "min_samples_leaf": (1, 10),
        }

        search = BayesSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            search_space,
            n_iter=20,
            cv=StratifiedKFold(n_splits=5, shuffle=False),
            scoring="roc_auc",
            n_jobs=-1,
            random_state=42,
        )
    else:
        # Fallback: RandomizedSearchCV with equivalent space
        from sklearn.model_selection import RandomizedSearchCV

        param_dist = {
            "n_estimators": [100, 200, 300, 400, 500],
            "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
            "min_samples_split": [2, 5, 10, 15, 20],
            "min_samples_leaf": [1, 2, 3, 5, 10],
        }

        search = RandomizedSearchCV(
            RandomForestClassifier(random_state=42, n_jobs=-1),
            param_dist,
            n_iter=20,
            cv=StratifiedKFold(n_splits=5, shuffle=False),
            scoring="roc_auc",
            n_jobs=-1,
            random_state=42,
        )

    search.fit(X_train_selected, y_train)
    best_rf = search.best_estimator_

    logger.info(f"Best RF hyperparameters (via Bayesian CV): {search.best_params_}")

    # Step 4: Evaluate on validation and test sets. The validation set was
    # previously computed (X_val_selected, above) but never actually used
    # anywhere -- BayesSearchCV/RandomizedSearchCV already select
    # hyperparameters via their own internal 5-fold CV on X_train, so val
    # metrics are reported here (not used for selection, to avoid leakage)
    # purely so val-vs-test can be compared to sanity-check that the
    # hyperparameter search didn't overfit the CV folds.
    y_val_pred = best_rf.predict(X_val_selected)
    y_val_pred_proba = best_rf.predict_proba(X_val_selected)[:, 1]
    val_metrics = compute_metrics(y_val, y_val_pred, y_val_pred_proba)

    y_pred = best_rf.predict(X_test_selected)
    y_pred_proba = best_rf.predict_proba(X_test_selected)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_pred_proba)

    logger.info(f"RF val metrics: {val_metrics}")
    logger.info(f"RF test metrics: {metrics}")

    if use_wandb and WANDB_AVAILABLE:
        try:
            wandb.log({
                "rf_val_metrics": val_metrics, "rf_metrics": metrics,
                "rf_best_params": search.best_params_,
            })
        except Exception as e:
            logger.warning(f"Failed to log to wandb: {e}")

    return {
        "model": best_rf,
        "selector": selector,
        "val_metrics": val_metrics,
        "metrics": metrics,
        "selected_features": selected_feature_names,
        "hyperparameters": dict(search.best_params_),
        "best_cv_score": search.best_score_,
    }


def train_and_evaluate_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    use_wandb: bool = False,
) -> dict:
    """
    Train XGBoost with feature selection and hyperparameter search.

    Following Appendix A4, Table A2 for XGBoost:
    - min_child_weight: uniform {1, 2, ..., 10}
    - subsample: uniform [0.4, 1.0]
    - colsample_bytree: uniform [0.4, 1.0]
    - max_depth: uniform {3, 4, ..., 10}
    - gamma: log-uniform [0.1, 5]

    Using BayesSearchCV with 5-fold CV; fallback to RandomizedSearchCV.

    Args:
        X_train, y_train: Training data.
        X_val, y_val: Validation data.
        X_test, y_test: Test data.
        feature_names: List of feature names.
        use_wandb: Whether to log to wandb.

    Returns:
        Dictionary with model, metrics, selected features, and hyperparameters.
    """
    logger.info("Training XGBoost baseline...")

    # Step 1: Train initial XGBoost on full feature set to identify top 25% features
    xgb_initial = XGBClassifier(
        n_estimators=100,
        random_state=42,
        eval_metric="logloss",
    )
    xgb_initial.fit(X_train, y_train)

    # Step 2: SelectFromModel to select top 25% of features
    # See the matching comment in train_and_evaluate_rf above: threshold=
    # -np.inf is required so SelectFromModel keeps exactly the top
    # max_features by importance, rather than "at most the top 25%" after
    # an implicit mean-importance pre-filter.
    selector = SelectFromModel(
        xgb_initial,
        prefit=True,
        threshold=-np.inf,
        max_features=max(1, int(0.25 * X_train.shape[1])),
    )
    X_train_selected = selector.transform(X_train)
    X_val_selected = selector.transform(X_val)
    X_test_selected = selector.transform(X_test)

    selected_feature_indices = selector.get_support(indices=True)
    selected_feature_names = [feature_names[i] for i in selected_feature_indices]

    logger.info(
        f"Selected {len(selected_feature_names)} / {len(feature_names)} features "
        f"(top 25%): {selected_feature_names}"
    )

    # Step 3: Hyperparameter search over selected features
    try:
        from skopt import BayesSearchCV
        BAYES_AVAILABLE = True
    except ImportError:
        logger.warning(
            "BayesSearchCV not available; falling back to RandomizedSearchCV. "
            "Install scikit-optimize for exact paper reproduction."
        )
        BAYES_AVAILABLE = False

    if BAYES_AVAILABLE:
        from skopt.space import Real

        # Bayesian hyperparameter search space (Appendix A4, Table A2 for XGBoost).
        # gamma is specified as log-uniform [0.1, 5] in the paper, but a plain
        # (low, high) tuple passed to BayesSearchCV is sampled UNIFORMLY, not
        # log-uniformly (skopt only applies a non-uniform prior when told to
        # via skopt.space.Real(..., prior=...)) -- so gamma needs to be
        # spelled out explicitly rather than left as a bare tuple like the
        # other (uniform-prior, per the paper) hyperparameters here.
        search_space = {
            "max_depth": (3, 10),
            "min_child_weight": (1, 10),
            "subsample": (0.4, 1.0),
            "colsample_bytree": (0.4, 1.0),
            "gamma": Real(0.1, 5.0, prior="log-uniform"),
        }

        search = BayesSearchCV(
            XGBClassifier(
                n_estimators=100,
                random_state=42,
                eval_metric="logloss",
            ),
            search_space,
            n_iter=20,
            cv=StratifiedKFold(n_splits=5, shuffle=False),
            scoring="roc_auc",
            n_jobs=-1,
            random_state=42,
        )
    else:
        # Fallback: RandomizedSearchCV with equivalent space
        from sklearn.model_selection import RandomizedSearchCV

        param_dist = {
            "max_depth": [3, 4, 5, 6, 7, 8, 9, 10],
            "min_child_weight": [1, 2, 3, 5, 10],
            "subsample": [0.4, 0.6, 0.8, 1.0],
            "colsample_bytree": [0.4, 0.6, 0.8, 1.0],
            # Log-spaced (not linearly spaced) grid points, approximating
            # the paper's log-uniform [0.1, 5] prior for gamma -- a discrete
            # RandomizedSearchCV grid can't express a continuous prior, but
            # log-spacing the candidate values is a much closer match than
            # the previous linearly-spaced [0.1, 0.5, 1.0, 2.0, 5.0] list.
            "gamma": list(np.geomspace(0.1, 5.0, 5)),
        }

        search = RandomizedSearchCV(
            XGBClassifier(
                n_estimators=100,
                random_state=42,
                eval_metric="logloss",
            ),
            param_dist,
            n_iter=20,
            cv=StratifiedKFold(n_splits=5, shuffle=False),
            scoring="roc_auc",
            n_jobs=-1,
            random_state=42,
        )

    search.fit(X_train_selected, y_train)
    best_xgb = search.best_estimator_

    logger.info(f"Best XGBoost hyperparameters (via Bayesian CV): {search.best_params_}")

    # Step 4: Evaluate on validation and test sets (see the matching
    # comment in train_and_evaluate_rf above for why val is reported here).
    y_val_pred = best_xgb.predict(X_val_selected)
    y_val_pred_proba = best_xgb.predict_proba(X_val_selected)[:, 1]
    val_metrics = compute_metrics(y_val, y_val_pred, y_val_pred_proba)

    y_pred = best_xgb.predict(X_test_selected)
    y_pred_proba = best_xgb.predict_proba(X_test_selected)[:, 1]
    metrics = compute_metrics(y_test, y_pred, y_pred_proba)

    logger.info(f"XGBoost val metrics: {val_metrics}")
    logger.info(f"XGBoost test metrics: {metrics}")

    if use_wandb and WANDB_AVAILABLE:
        try:
            wandb.log({
                "xgb_val_metrics": val_metrics, "xgb_metrics": metrics,
                "xgb_best_params": search.best_params_,
            })
        except Exception as e:
            logger.warning(f"Failed to log to wandb: {e}")

    return {
        "model": best_xgb,
        "selector": selector,
        "val_metrics": val_metrics,
        "metrics": metrics,
        "selected_features": selected_feature_names,
        "hyperparameters": dict(search.best_params_),
        "best_cv_score": search.best_score_,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate Random Forest and XGBoost baselines."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="five_updated_synthetic.csv",
        help="Path to the input CSV (default: five_updated_synthetic.csv in data_prep/)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="both",
        choices=["rf", "xgb", "both"],
        help="Which model(s) to train: 'rf', 'xgb', or 'both'.",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable wandb logging.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory to save trained models, scaler, and metrics "
             "(mirrors the --checkpoint-dir flag on explain_shap.py, added "
             "so callers -- e.g. a CI job or pipeline step writing outputs "
             "to a different mount -- can redirect it without editing this file).",
    )

    args = parser.parse_args()

    # Initialize wandb if available and not disabled
    use_wandb = (not args.no_wandb) and WANDB_AVAILABLE
    if use_wandb:
        try:
            wandb.init(project="dementia_baselines", name=f"train_{args.model}")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            use_wandb = False

    # Ensure checkpoints directory exists
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(exist_ok=True, parents=True)

    # Load data
    if not Path(args.data_path).exists():
        logger.error(f"Data file not found: {args.data_path}")
        logger.info("Generating synthetic data for smoke test...")
        sys.path.insert(0, str(Path(__file__).parent.parent / "data_prep"))
        from generate_synthetic_data import generate_synthetic_cohort
        rng = np.random.default_rng(args.seed)
        df = generate_synthetic_cohort(500, rng)
        df.to_csv(args.data_path, index=False)
        logger.info(f"Generated synthetic data: {args.data_path}")

    df = fe.load_and_preprocess(args.data_path)

    # Canonical 70/10/20 stratified split (common/data_split.py -- shared
    # with gnn/ and bert_models/ so every model family in this package is
    # scored on the same held-out patients). This script's own two chained
    # `train_test_split` calls already implemented the identical
    # test_size=0.2-then-0.125 scheme, so this changes no numbers here --
    # it just removes a second independent implementation of the same
    # split in favor of the single shared one.
    #
    # IMPORTANT: we split the raw dataframe FIRST, before any scaling, and
    # only fit StandardScaler (inside build_feature_matrix) on the training
    # split. An earlier version of this script called
    # `build_feature_matrix(df, fit_scaler=True)` on the *entire* dataset
    # before splitting, which fit the Age/PRS scaler's mean and variance
    # using validation- and test-set rows too -- a data leakage bug that
    # would silently inflate reported val/test performance, since those
    # splits were standardized using statistics partly derived from
    # themselves. Splitting first and fitting the scaler on
    # `df_train` only (then just `.transform`-ing val/test with that same
    # fitted scaler, never refitting) closes that leak.
    df = add_label_column(df)
    df_train, df_val, df_test = split_dataframe(df, seed=args.seed)
    y_train = df_train["label"].values
    y_val = df_val["label"].values
    y_test = df_test["label"].values

    X_train, feature_names, fitted_scaler = fe.build_feature_matrix(df_train, fit_scaler=True)
    X_val, _, _ = fe.build_feature_matrix(df_val, scaler=fitted_scaler, fit_scaler=False)
    X_test, _, _ = fe.build_feature_matrix(df_test, scaler=fitted_scaler, fit_scaler=False)

    # Persist the train-fitted scaler + feature name order so that
    # explain_shap.py (or anyone else reusing these checkpoints) can
    # reproduce the exact feature matrix the model was trained on, instead
    # of re-fitting a new scaler on whatever data happens to be passed to
    # it later -- which would silently shift the Age/PRS values SHAP sees
    # away from the distribution the model actually learned on.
    import joblib
    joblib.dump(fitted_scaler, checkpoint_dir / "scaler.pkl")
    joblib.dump(feature_names, checkpoint_dir / "feature_names.pkl")

    logger.info(f"Train/val/test split: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    logger.info(f"Class distribution (train): {np.bincount(y_train)}")
    logger.info(f"Class distribution (val): {np.bincount(y_val)}")
    logger.info(f"Class distribution (test): {np.bincount(y_test)}")

    results = {}

    if args.model in ["rf", "both"]:
        rf_result = train_and_evaluate_rf(
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            feature_names,
            use_wandb=use_wandb,
        )
        results["rf"] = rf_result

        # Save RF model and metrics
        import joblib
        joblib.dump(rf_result["model"], checkpoint_dir / "rf_model.pkl")
        joblib.dump(rf_result["selector"], checkpoint_dir / "rf_selector.pkl")

        metrics_file = checkpoint_dir / "rf_metrics.json"
        with open(metrics_file, "w") as f:
            # Convert numpy types to native Python for JSON serialization
            metrics_to_save = {
                k: float(v) for k, v in rf_result["metrics"].items()
            }
            val_metrics_to_save = {
                k: float(v) for k, v in rf_result["val_metrics"].items()
            }
            json.dump(
                {
                    "val_metrics": val_metrics_to_save,
                    "test_metrics": metrics_to_save,
                    "hyperparameters": rf_result["hyperparameters"],
                    "best_cv_score": float(rf_result["best_cv_score"]),
                    "selected_features": rf_result["selected_features"],
                },
                f,
                indent=2,
            )
        logger.info(f"Saved RF model and metrics to {checkpoint_dir}")

    if args.model in ["xgb", "both"]:
        xgb_result = train_and_evaluate_xgb(
            X_train, y_train,
            X_val, y_val,
            X_test, y_test,
            feature_names,
            use_wandb=use_wandb,
        )
        results["xgb"] = xgb_result

        # Save XGBoost model and metrics
        import joblib
        joblib.dump(xgb_result["model"], checkpoint_dir / "xgb_model.pkl")
        joblib.dump(xgb_result["selector"], checkpoint_dir / "xgb_selector.pkl")

        metrics_file = checkpoint_dir / "xgb_metrics.json"
        with open(metrics_file, "w") as f:
            metrics_to_save = {
                k: float(v) for k, v in xgb_result["metrics"].items()
            }
            val_metrics_to_save = {
                k: float(v) for k, v in xgb_result["val_metrics"].items()
            }
            json.dump(
                {
                    "val_metrics": val_metrics_to_save,
                    "test_metrics": metrics_to_save,
                    "hyperparameters": xgb_result["hyperparameters"],
                    "best_cv_score": float(xgb_result["best_cv_score"]),
                    "selected_features": xgb_result["selected_features"],
                },
                f,
                indent=2,
            )
        logger.info(f"Saved XGBoost model and metrics to {checkpoint_dir}")

    # Print summary
    print("\n" + "=" * 60)
    print("BASELINE MODEL TRAINING SUMMARY")
    print("=" * 60)
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Test Accuracy:  {result['metrics']['accuracy']:.4f}")
        print(f"  Sensitivity:    {result['metrics']['sensitivity']:.4f}")
        print(f"  Specificity:    {result['metrics']['specificity']:.4f}")
        print(f"  F1 (dementia):  {result['metrics']['f1']:.4f}")
        print(f"  AUROC:          {result['metrics']['auroc']:.4f}")
        print(f"  AUPRC:          {result['metrics']['auprc']:.4f}")
        print(f"  Val AUROC:      {result['val_metrics']['auroc']:.4f}  (sanity check vs. test, not used for model selection)")
        print(f"  Best CV Score:  {result['best_cv_score']:.4f}")
        print(f"  Selected Features: {len(result['selected_features'])} / {len(feature_names)}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
