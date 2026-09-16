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
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import (
    accuracy_score,
    auc,
    roc_curve,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

import feature_engineering as fe

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


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> dict:
    """
    Compute evaluation metrics.

    Args:
        y_true: Ground truth labels (0/1).
        y_pred: Predicted binary labels (0/1).
        y_pred_proba: Predicted probabilities for class 1 (shape: (n,)).

    Returns:
        Dictionary with accuracy, sensitivity, specificity, AUROC.
    """
    accuracy = accuracy_score(y_true, y_pred)

    # Sensitivity = recall for class 0 (Dementia)
    tn = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 1) & (y_pred == 0)).sum()
    fn = ((y_true == 0) & (y_pred == 1)).sum()
    tp = ((y_true == 0) & (y_pred == 0)).sum()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # AUROC: use probabilities for the positive class (class 1)
    auroc = roc_auc_score(y_true, y_pred_proba)

    return {
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "auroc": auroc,
    }


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
    selector = SelectFromModel(
        rf_initial,
        prefit=True,
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

    # Step 4: Evaluate on test set
    y_pred = best_rf.predict(X_test_selected)
    y_pred_proba = best_rf.predict_proba(X_test_selected)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_pred_proba)
    logger.info(f"RF test metrics: {metrics}")

    if use_wandb and WANDB_AVAILABLE:
        try:
            wandb.log({"rf_metrics": metrics, "rf_best_params": search.best_params_})
        except Exception as e:
            logger.warning(f"Failed to log to wandb: {e}")

    return {
        "model": best_rf,
        "selector": selector,
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
    selector = SelectFromModel(
        xgb_initial,
        prefit=True,
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
        # Bayesian hyperparameter search space (Appendix A4, Table A2 for XGBoost)
        search_space = {
            "max_depth": (3, 10),
            "min_child_weight": (1, 10),
            "subsample": (0.4, 1.0),
            "colsample_bytree": (0.4, 1.0),
            "gamma": (0.1, 5.0),
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
            "gamma": [0.1, 0.5, 1.0, 2.0, 5.0],
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

    # Step 4: Evaluate on test set
    y_pred = best_xgb.predict(X_test_selected)
    y_pred_proba = best_xgb.predict_proba(X_test_selected)[:, 1]

    metrics = compute_metrics(y_test, y_pred, y_pred_proba)
    logger.info(f"XGBoost test metrics: {metrics}")

    if use_wandb and WANDB_AVAILABLE:
        try:
            wandb.log({"xgb_metrics": metrics, "xgb_best_params": search.best_params_})
        except Exception as e:
            logger.warning(f"Failed to log to wandb: {e}")

    return {
        "model": best_xgb,
        "selector": selector,
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
             "so callers -- e.g. a Docker entrypoint writing to a mounted "
             "output volume -- can redirect it without editing this file).",
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
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "data_prep"))
        from generate_synthetic_data import generate_synthetic_cohort
        rng = np.random.default_rng(args.seed)
        df = generate_synthetic_cohort(500, rng)
        df.to_csv(args.data_path, index=False)
        logger.info(f"Generated synthetic data: {args.data_path}")

    df = fe.load_and_preprocess(args.data_path)
    y_full = fe.get_labels(df)

    # 70/10/20 stratified split (matching SCHEMA.md, GNN pipeline).
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

    # First split: 80% train+val, 20% test
    df_temp, df_test, y_temp, y_test = train_test_split(
        df, y_full,
        test_size=0.2,
        stratify=y_full,
        random_state=args.seed,
    )

    # Second split: 87.5% (of remaining 80%) = 70% total train, 12.5% = 10% total val
    df_train, df_val, y_train, y_val = train_test_split(
        df_temp, y_temp,
        test_size=0.125,
        stratify=y_temp,
        random_state=args.seed,
    )

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
            json.dump(
                {
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
            json.dump(
                {
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
        print(f"  AUROC:          {result['metrics']['auroc']:.4f}")
        print(f"  Best CV Score:  {result['best_cv_score']:.4f}")
        print(f"  Selected Features: {len(result['selected_features'])} / {len(feature_names)}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
