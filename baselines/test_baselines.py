"""
End-to-end test for baseline models.

This script validates that:
1. Synthetic data generation works
2. Feature engineering loads and preprocesses data correctly
3. Random Forest and XGBoost training complete successfully
4. SHAP explanations are generated without errors
5. All output files are created with valid content

"""

import json
import logging
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def test_synthetic_data_generation():
    """Test synthetic data generation."""
    logger.info("TEST 1: Synthetic data generation...")
    sys.path.insert(0, str(Path(__file__).parent.parent / "data_prep"))
    from generate_synthetic_data import generate_synthetic_cohort

    rng = np.random.default_rng(42)
    df = generate_synthetic_cohort(500, rng)

    assert len(df) == 500, f"Expected 500 patients, got {len(df)}"
    assert "eid" in df.columns, "Missing 'eid' column"
    assert "Class" in df.columns, "Missing 'Class' column"
    assert set(df["Class"].unique()).issubset({"Dementia", "Control"}), "Invalid class values"
    assert df["Class"].nunique() == 2, "Both classes should be present"

    logger.info(f"  PASS: Generated {len(df)} synthetic patients")
    return df


def test_feature_engineering(df):
    """Test feature engineering."""
    logger.info("TEST 2: Feature engineering...")
    import feature_engineering as fe

    # Test preprocessing
    df_processed = fe.load_and_preprocess(df)
    assert len(df_processed) == len(df), "Preprocessing changed number of rows"

    # Test feature matrix building
    X, feature_names, scaler = fe.build_feature_matrix(df_processed, fit_scaler=True)
    assert X.shape[0] == len(df), f"Feature matrix has wrong number of samples: {X.shape[0]}"
    assert X.shape[1] == len(feature_names), "Feature names mismatch with matrix shape"
    assert scaler is not None, "Scaler not returned"

    # Test label extraction
    y = fe.get_labels(df_processed)
    assert len(y) == len(df), "Labels have wrong length"
    assert set(y).issubset({0, 1}), "Invalid label values"

    logger.info(f"  PASS: Built feature matrix {X.shape[0]} x {X.shape[1]} with {len(feature_names)} feature names")
    return X, y, feature_names


def test_model_training(X, y):
    """Test that models can be trained."""
    logger.info("TEST 3: Model training (quick sanity check)...")
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.model_selection import train_test_split

    # Quick train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Train RF
    rf = RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    score_rf = rf.score(X_test, y_test)
    assert 0 <= score_rf <= 1, f"Invalid RF score: {score_rf}"
    logger.info(f"  RF quick test score: {score_rf:.3f}")

    # Train XGBoost
    xgb = XGBClassifier(n_estimators=10, random_state=42, eval_metric="logloss")
    xgb.fit(X_train, y_train)
    score_xgb = xgb.score(X_test, y_test)
    assert 0 <= score_xgb <= 1, f"Invalid XGBoost score: {score_xgb}"
    logger.info(f"  XGBoost quick test score: {score_xgb:.3f}")

    logger.info(f"  PASS: Both models trained successfully")


def test_checkpoint_files():
    """Test that checkpoint files exist and are valid."""
    logger.info("TEST 4: Checkpoint files...")
    checkpoint_dir = Path(__file__).parent / "checkpoints"

    required_files = [
        "rf_model.pkl",
        "rf_selector.pkl",
        "rf_metrics.json",
        "xgb_model.pkl",
        "xgb_selector.pkl",
        "xgb_metrics.json",
    ]

    for file in required_files:
        file_path = checkpoint_dir / file
        assert file_path.exists(), f"Missing checkpoint file: {file_path}"

    # Test that metrics JSON are valid
    for model_name in ["rf", "xgb"]:
        metrics_file = checkpoint_dir / f"{model_name}_metrics.json"
        with open(metrics_file) as f:
            metrics = json.load(f)

        assert "test_metrics" in metrics, f"Missing 'test_metrics' in {model_name} metrics"
        assert "hyperparameters" in metrics, f"Missing 'hyperparameters' in {model_name} metrics"
        assert "best_cv_score" in metrics, f"Missing 'best_cv_score' in {model_name} metrics"
        assert "selected_features" in metrics, f"Missing 'selected_features' in {model_name} metrics"

        # Check metric keys
        for metric_key in ["accuracy", "sensitivity", "specificity", "auroc"]:
            assert metric_key in metrics["test_metrics"], f"Missing {metric_key} in {model_name} metrics"

        logger.info(f"  {model_name.upper()} metrics valid")

    logger.info(f"  PASS: All checkpoint files valid")


def test_explainability_files():
    """Test that SHAP files exist and are valid."""
    logger.info("TEST 5: Explainability files...")
    output_dir = Path(__file__).parent / "explainability_outputs"

    required_files = [
        "rf_shap_feature_importance.csv",
        "rf_shap_summary.png",
        "rf_shap_bar.png",
        "xgb_shap_feature_importance.csv",
        "xgb_shap_summary.png",
        "xgb_shap_bar.png",
        "shap_summary.json",
    ]

    for file in required_files:
        file_path = output_dir / file
        assert file_path.exists(), f"Missing explainability file: {file_path}"

    # Test that SHAP summary JSON is valid
    summary_file = output_dir / "shap_summary.json"
    with open(summary_file) as f:
        summary = json.load(f)

    for model_name in ["rf", "xgb"]:
        assert model_name in summary, f"Missing {model_name} in SHAP summary"
        assert "top_feature" in summary[model_name], f"Missing 'top_feature' in {model_name} SHAP summary"
        assert "top_feature_shap" in summary[model_name], f"Missing 'top_feature_shap' in {model_name} SHAP summary"

        logger.info(f"  {model_name.upper()} top feature: {summary[model_name]['top_feature']}")

    # Test that feature importance CSVs are valid
    for model_name in ["rf", "xgb"]:
        csv_file = output_dir / f"{model_name}_shap_feature_importance.csv"
        df_importance = pd.read_csv(csv_file)
        assert len(df_importance) > 0, f"Empty feature importance CSV for {model_name}"
        assert "feature" in df_importance.columns, f"Missing 'feature' column in {model_name} CSV"
        assert "mean_abs_shap" in df_importance.columns, f"Missing 'mean_abs_shap' column in {model_name} CSV"
        logger.info(f"  {model_name.upper()} feature importance CSV has {len(df_importance)} features")

    logger.info(f"  PASS: All explainability files valid")


def main():
    """Run all tests."""
    logger.info("=" * 60)
    logger.info("BASELINE MODELS END-TO-END TEST SUITE")
    logger.info("=" * 60 + "\n")

    try:
        # Test 1: Synthetic data
        df = test_synthetic_data_generation()
        print()

        # Test 2: Feature engineering
        X, y, feature_names = test_feature_engineering(df)
        print()

        # Test 3: Model training
        test_model_training(X, y)
        print()

        # Test 4: Checkpoint files
        test_checkpoint_files()
        print()

        # Test 5: Explainability files
        test_explainability_files()
        print()

        logger.info("=" * 60)
        logger.info("ALL TESTS PASSED!")
        logger.info("=" * 60)
        return 0

    except AssertionError as e:
        logger.error(f"\nTEST FAILED: {e}")
        return 1
    except Exception as e:
        logger.error(f"\nUNEXPECTED ERROR: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
