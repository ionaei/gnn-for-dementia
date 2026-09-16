"""
Generate SHAP explanations for trained baseline models.

Following the paper's methodology (Section 3.2, "Explainability"):
- Compute SHAP values for Random Forest and XGBoost models
- Generate summary plots (bar and beeswarm) for global feature importance
- Save mean |SHAP| per feature as CSV

SHAP values are computed using TreeExplainer (native to tree-based models),
allowing efficient computation and interpretability for baseline predictions.
"""

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt

import feature_engineering as fe

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def explain_model_shap(
    model,
    selector,
    X_test: np.ndarray,
    feature_names: list,
    model_name: str,
    output_dir: Path,
) -> pd.DataFrame:
    """
    Compute SHAP values for a trained model and generate visualizations.

    Args:
        model: Trained scikit-learn tree-based model (RandomForest or XGBoost).
        selector: SelectFromModel used to select features during training.
        X_test: Test feature matrix (full features before selection).
        feature_names: List of all feature names.
        model_name: Name for output files ('rf' or 'xgb').
        output_dir: Directory to save outputs.

    Returns:
        DataFrame with mean |SHAP| per feature, sorted by importance.
    """
    logger.info(f"Generating SHAP explanations for {model_name.upper()}...")

    # Apply feature selector to test set
    X_test_selected = selector.transform(X_test)

    # Get selected feature names
    selected_indices = selector.get_support(indices=True)
    selected_features = [feature_names[i] for i in selected_indices]

    # Compute SHAP values using TreeExplainer
    # For XGBoost, we may need to use a different method or check_additivity=False
    try:
        explainer = shap.TreeExplainer(model, check_additivity=False)
        shap_values = explainer.shap_values(X_test_selected)
    except Exception as e:
        logger.warning(f"TreeExplainer failed: {e}. Using KernelExplainer as fallback.")
        # Fallback: use KernelExplainer (slower but more robust)
        explainer = shap.KernelExplainer(model.predict_proba, shap.sample(X_test_selected, 50))
        shap_values = explainer.shap_values(X_test_selected)
        # KernelExplainer returns values for each class; we want class 0 (Dementia)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

    # Handle different SHAP value formats:
    # - List of arrays [class0, class1] for some models
    # - 3D array (samples, features, classes) for RandomForest
    # - 2D array (samples, features) for some models
    if isinstance(shap_values, list):
        # Format: list of arrays
        shap_values_dementia = np.array(shap_values[0])
    else:
        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            # Format: (samples, features, classes) - take class 0 (dementia)
            shap_values_dementia = shap_values[:, :, 0]
        elif shap_values.ndim == 2:
            # Format: (samples, features) - already in the right format
            shap_values_dementia = shap_values
        else:
            raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")

    # Compute mean |SHAP| per feature
    mean_abs_shap = np.abs(shap_values_dementia).mean(axis=0)

    # Create DataFrame for feature importances
    feature_importance_df = pd.DataFrame({
        "feature": selected_features,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    logger.info(f"Top 10 important features for {model_name.upper()}:")
    print(feature_importance_df.head(10))

    # Save feature importances as CSV
    output_dir.mkdir(exist_ok=True, parents=True)
    csv_path = output_dir / f"{model_name}_shap_feature_importance.csv"
    feature_importance_df.to_csv(csv_path, index=False)
    logger.info(f"Saved feature importances to {csv_path}")

    # Generate summary plot (bar + beeswarm combined)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values_dementia,
        X_test_selected,
        feature_names=selected_features,
        show=False,
    )
    plot_path = output_dir / f"{model_name}_shap_summary.png"
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved summary plot to {plot_path}")

    # Generate bar plot (mean absolute SHAP values)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values_dementia,
        X_test_selected,
        feature_names=selected_features,
        plot_type="bar",
        show=False,
    )
    bar_plot_path = output_dir / f"{model_name}_shap_bar.png"
    plt.tight_layout()
    plt.savefig(bar_plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Saved bar plot to {bar_plot_path}")

    return feature_importance_df


def main():
    parser = argparse.ArgumentParser(
        description="Generate SHAP explanations for baseline models."
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="five_updated_synthetic.csv",
        help="Path to the input CSV (same as used for training).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="both",
        choices=["rf", "xgb", "both"],
        help="Which model(s) to explain: 'rf', 'xgb', or 'both'.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory containing saved models and selectors.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="explainability_outputs",
        help="Directory to save SHAP explanations.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (must match training).",
    )

    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Load data (using same split as training)
    from sklearn.model_selection import train_test_split

    if not Path(args.data_path).exists():
        logger.error(f"Data file not found: {args.data_path}")
        return

    df = fe.load_and_preprocess(args.data_path)
    y_full = fe.get_labels(df)

    # Recreate the same 70/10/20 split as training, splitting the raw
    # dataframe first (not a pre-scaled feature matrix) so we can then
    # apply -- not re-fit -- the scaler that train_rf_xgb.py actually
    # trained the model with. An earlier version of this script called
    # `build_feature_matrix(df, fit_scaler=True)` on the whole dataset,
    # which (a) fit a brand-new StandardScaler on train+val+test combined
    # (the same leakage bug fixed in train_rf_xgb.py) and (b) meant SHAP
    # would see Age/PRS values standardized with different mean/scale than
    # what the model actually learned on, silently distorting every SHAP
    # value for those two features. Loading the persisted `scaler.pkl` /
    # `feature_names.pkl` from the training run closes both problems.
    scaler_path = checkpoint_dir / "scaler.pkl"
    feature_names_path = checkpoint_dir / "feature_names.pkl"
    if not scaler_path.exists() or not feature_names_path.exists():
        logger.error(
            f"{scaler_path} / {feature_names_path} not found. Have you run "
            f"train_rf_xgb.py (with this --checkpoint-dir) first?"
        )
        return
    fitted_scaler = joblib.load(scaler_path)
    feature_names = joblib.load(feature_names_path)

    df_temp, df_test, y_temp, y_test = train_test_split(
        df, y_full,
        test_size=0.2,
        stratify=y_full,
        random_state=args.seed,
    )
    df_train, df_val, y_train, y_val = train_test_split(
        df_temp, y_temp,
        test_size=0.125,
        stratify=y_temp,
        random_state=args.seed,
    )
    X_test, _, _ = fe.build_feature_matrix(df_test, scaler=fitted_scaler, fit_scaler=False)

    logger.info(f"Loaded test set: {X_test.shape[0]} samples, {X_test.shape[1]} features")

    results_summary = {}

    if args.model in ["rf", "both"]:
        # Load RF model and selector
        model_path = checkpoint_dir / "rf_model.pkl"
        selector_path = checkpoint_dir / "rf_selector.pkl"

        if not model_path.exists() or not selector_path.exists():
            logger.error(f"RF model not found. Have you run train_rf_xgb.py?")
        else:
            model = joblib.load(model_path)
            selector = joblib.load(selector_path)

            importance_df = explain_model_shap(
                model, selector,
                X_test, feature_names,
                "rf",
                output_dir,
            )
            results_summary["rf"] = {
                "n_features_explained": len(importance_df),
                "top_feature": importance_df.iloc[0]["feature"],
                "top_feature_shap": float(importance_df.iloc[0]["mean_abs_shap"]),
            }

    if args.model in ["xgb", "both"]:
        # Load XGBoost model and selector
        model_path = checkpoint_dir / "xgb_model.pkl"
        selector_path = checkpoint_dir / "xgb_selector.pkl"

        if not model_path.exists() or not selector_path.exists():
            logger.error(f"XGBoost model not found. Have you run train_rf_xgb.py?")
        else:
            model = joblib.load(model_path)
            selector = joblib.load(selector_path)

            importance_df = explain_model_shap(
                model, selector,
                X_test, feature_names,
                "xgb",
                output_dir,
            )
            results_summary["xgb"] = {
                "n_features_explained": len(importance_df),
                "top_feature": importance_df.iloc[0]["feature"],
                "top_feature_shap": float(importance_df.iloc[0]["mean_abs_shap"]),
            }

    # Save summary
    summary_path = output_dir / "shap_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    logger.info(f"Saved SHAP summary to {summary_path}")

    print("\n" + "=" * 60)
    print("SHAP EXPLANATION SUMMARY")
    print("=" * 60)
    for model_name, summary in results_summary.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Top Feature: {summary['top_feature']}")
        print(f"  Top Feature Mean |SHAP|: {summary['top_feature_shap']:.4f}")
        print(f"  Total Features Explained: {summary['n_features_explained']}")


if __name__ == "__main__":
    main()
