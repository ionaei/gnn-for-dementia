# Baseline Models (Random Forest & XGBoost)

This directory implements the baseline models from the paper "Predicting Dementia Risk Using Longitudinal Electronic Health Records Data" (Imperial College London, NeurIPS 2025 Workshop).

## Overview

The baseline models capture the relationship between presence and timing of pre-existing diagnoses and dementia risk using tree-based classifiers. Two architectures are trained in parallel:

1. **Random Forest** - ensemble of decision trees
2. **XGBoost** - gradient-boosted decision trees

Both use the same feature engineering pipeline, data splits, and evaluation methodology as described in the paper (Section 3.2).

## Scripts

### `feature_engineering.py`
Loads the CSV (schema: `eid`, `Class`, `Sex`, `Age`, `Standard PRS for alzheimer's disease (AD)`, diagnosis block columns), preprocesses features, and applies standardization.

**Key decisions:**
- **NaN-filling for `{block}_time` columns**: When `{block}_present == 0` (diagnosis absent), the corresponding `_time` column is NaN. We fill these with 0 days, interpreting it as "no time elapsed because diagnosis never occurred." This is appropriate for tree-based models (which do not natively handle NaN) without inflating the signal—0 days = no diagnosis impact, semantically consistent with the binary indicator.
- **Standardization**: Applied to `Age` and `Standard PRS for alzheimer's disease (AD)` using `StandardScaler` fit on training data only. Binary and categorical features (`Sex`, diagnosis blocks `_present`) are not scaled.

### `train_rf_xgb.py`
CLI script that trains and evaluates baseline models end-to-end.

**Methodology (following the paper):**

1. **Data split**: 70% train / 10% validation / 20% test via stratified splits (matching the GNN pipeline in `SCHEMA.md`):
   - First: `train_test_split(test_size=0.2, stratify=label, random_state=42)` → 80% train+val, 20% test
   - Second: `train_test_split(test_size=0.125, stratify=label, random_state=42)` on train+val → 70% train, 10% val

2. **Feature selection (Section 3.2)**: 
   - Train an initial model on the full feature set
   - Use `SelectFromModel` to select the top 25% of features ranked by feature importance
   - All subsequent hyperparameter search and final models use only selected features

3. **Hyperparameter search (Appendix A4, Table A2)**:
   - Search space via Bayesian optimization (BayesSearchCV) with 5-fold cross-validation
   - Scoring metric: ROC-AUC
   - **Random Forest search space**: `n_estimators` [100–500], `max_depth` [3–10], `min_samples_split` [2–20], `min_samples_leaf` [1–10]
   - **XGBoost search space**: `max_depth` [3–10], `min_child_weight` [1–10], `subsample` [0.4–1.0], `colsample_bytree` [0.4–1.0], `gamma` [0.1–5.0]
   - If `scikit-optimize` is not available, falls back to `RandomizedSearchCV` with equivalent search space (documented in log output)

4. **Evaluation metrics (Table 1 of paper)**:
   - **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
   - **Sensitivity**: TP / (TP + FN) — recall for Dementia class (class 0)
   - **Specificity**: TN / (TN + FP) — recall for Control class (class 1)
   - **AUROC**: Area under the receiver operating characteristic curve

**Usage:**
```bash
# Train both models (auto-generates synthetic data if needed)
python train_rf_xgb.py --model both

# Train only Random Forest with specific data
python train_rf_xgb.py --model rf --data-path /path/to/data.csv

# Train only XGBoost, disable wandb
python train_rf_xgb.py --model xgb --no-wandb

# Train with custom seed
python train_rf_xgb.py --model both --seed 123
```

**Outputs:**
- `checkpoints/rf_model.pkl` - trained Random Forest (joblib)
- `checkpoints/xgb_model.pkl` - trained XGBoost (joblib)
- `checkpoints/rf_selector.pkl` - Random Forest feature selector (joblib)
- `checkpoints/xgb_selector.pkl` - XGBoost feature selector (joblib)
- `checkpoints/rf_metrics.json` - RF test metrics, hyperparameters, selected features
- `checkpoints/xgb_metrics.json` - XGBoost test metrics, hyperparameters, selected features
- `checkpoints/scaler.pkl` - `StandardScaler` fit on the **training split only** (Age + PRS), shared by both models since they're trained on the same split
- `checkpoints/feature_names.pkl` - ordered list of feature column names matching the columns of the (unselected) feature matrix the scaler/models expect

**No data leakage:** the 70/10/20 split is performed on the raw dataframe *before* any scaling, and `StandardScaler` is fit exclusively on the training split, then only `.transform()`-ed (never refit) onto validation and test. `explain_shap.py` loads this same persisted scaler rather than fitting a new one, so SHAP values are computed on the identical feature distribution the model was actually trained on.

### `explain_shap.py`
Generates SHAP (SHapley Additive exPlanations) feature importance explanations for trained models.

**Methodology (Section 3.2, "Explainability"):**
- Computes SHAP values using `shap.TreeExplainer` (native, efficient for tree-based models)
- Generates summary plots (combined bar and beeswarm) showing global feature importance
- Saves mean |SHAP| per feature as CSV, sorted by importance
- Focuses on class 0 (Dementia) for interpretability

**Usage:**
```bash
# Generate SHAP explanations for both models
python explain_shap.py --model both

# Explain only Random Forest
python explain_shap.py --model rf

# Use custom checkpoint/output directories
python explain_shap.py --checkpoint-dir ./my_checkpoints --output-dir ./my_shap_outputs
```

**Outputs:**
- `explainability_outputs/rf_shap_feature_importance.csv` - RF feature importance (mean |SHAP|)
- `explainability_outputs/xgb_shap_feature_importance.csv` - XGBoost feature importance
- `explainability_outputs/rf_shap_summary.png` - RF SHAP summary plot
- `explainability_outputs/xgb_shap_summary.png` - XGBoost SHAP summary plot
- `explainability_outputs/rf_shap_bar.png` - RF bar plot (top features)
- `explainability_outputs/xgb_shap_bar.png` - XGBoost bar plot (top features)
- `explainability_outputs/shap_summary.json` - Summary of top features and SHAP values per model

## End-to-End Example

```bash
cd /path/to/baselines

# Generate synthetic data for smoke testing
python ../data_prep/generate_synthetic_data.py --n-patients 500 --out five_updated_synthetic.csv

# Train baseline models
python train_rf_xgb.py --model both --data-path five_updated_synthetic.csv

# Generate SHAP explanations
python explain_shap.py --model both --data-path five_updated_synthetic.csv
```

## Dependencies

- `pandas`, `numpy`, `scikit-learn`
- `xgboost`
- `shap` (for explanations)
- `joblib` (for model serialization)
- `matplotlib` (for visualization)
- `wandb` (optional; for logging; safe to use without if not logged in or `--no-wandb` is set)
- `scikit-optimize` (optional; for exact Bayesian search; RandomizedSearchCV fallback available)

## Label Encoding

Throughout this module:
- **Class mapping**: `{"Dementia": 0, "Control": 1}`
- **Sensitivity** = recall for class 0 (Dementia)
- **Specificity** = recall for class 1 (Control)

## Comparison to Paper Results (Table 1)

The paper's Table 1 reports the following performance for **XGBoost** (on the actual UK Biobank data with true labels):
- F1: 0.708
- Sensitivity: 0.705
- Specificity: 0.711
- AUROC: 0.773

**NOTE**: The synthetic data generator produces fake patients and diagnoses with only weak synthetic signal. Test metrics on synthetic data are structurally valid but meaningless and will not match the paper's reported numbers. To reproduce the paper's results, you must use the real UK Biobank data (access-controlled under application 109607). The synthetic pipeline exists only to smoke-test code correctness.

