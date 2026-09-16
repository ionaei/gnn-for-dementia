"""
Feature engineering for baseline models.

Loads the CSV data (schema: eid, Class, Sex, Age, Standard PRS for alzheimer's disease (AD),
{diagnosis_block}_present, {diagnosis_block}_time columns), builds the flat feature matrix
for Random Forest and XGBoost, and applies standardization.

NaN-filling strategy for {diagnosis_block}_time columns:
- When {diagnosis_block}_present == 0, the _time column is NaN (diagnosis absent).
- We fill these with 0 days, interpreting it as "no time elapsed because diagnosis never occurred".
- This is consistent with the interpretation that _present is a binary indicator, and _time
  is only meaningful when _present == 1. Filling with 0 provides a valid numerical value
  for tree-based models (which do not natively handle NaN) without inflating the signal
  (0 days = no diagnosis impact, which is semantically appropriate).
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def load_and_preprocess(csv_path) -> pd.DataFrame:
    """
    Load the CSV and preprocess into the format expected by baseline models.

    Args:
        csv_path: Path to the CSV file (str or Path), or a DataFrame.

    Returns:
        DataFrame with preprocessed features.
    """
    if isinstance(csv_path, pd.DataFrame):
        df = csv_path.copy()
        logger.info(f"Using provided DataFrame with {len(df)} patients")
    else:
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} patients from {csv_path}")

    # Fill NaN values in _time columns (diagnosis absent) with 0 days
    time_cols = [col for col in df.columns if col.endswith("_time")]
    df[time_cols] = df[time_cols].fillna(0.0)

    return df


def build_feature_matrix(
    df: pd.DataFrame, scaler: StandardScaler = None, fit_scaler: bool = False
) -> Tuple[np.ndarray, list, StandardScaler]:
    """
    Build the flat feature matrix for baseline models.

    Features:
    - Sex (0/1)
    - Age (numeric)
    - Standard PRS for alzheimer's disease (AD) (numeric)
    - {diagnosis_block}_present (0.0/1.0) for each block
    - {diagnosis_block}_time (days) for each block

    Args:
        df: Input dataframe (from load_and_preprocess).
        scaler: Optional StandardScaler to apply (if not fitting a new one).
        fit_scaler: If True, fit a new StandardScaler on Age and PRS columns
            (should only be True for training data).

    Returns:
        (X, feature_names, scaler): Feature matrix, list of feature names, fitted/applied scaler.
    """
    # Collect all features at once to avoid pandas fragmentation warning
    feature_cols = ["Sex"]

    # Standardizable numeric features
    prs_col = "Standard PRS for alzheimer's disease (AD)"
    numeric_cols = ["Age", prs_col]

    if fit_scaler:
        scaler = StandardScaler()
        scaled_values = scaler.fit_transform(df[numeric_cols])
    else:
        if scaler is None:
            raise ValueError(
                "scaler must be provided if fit_scaler=False (e.g., for test data)"
            )
        scaled_values = scaler.transform(df[numeric_cols])

    # Diagnosis features: _present and _time columns
    diagnosis_cols = [col for col in df.columns if col.endswith("_present")]
    diagnosis_cols = sorted(diagnosis_cols)  # For determinism

    diagnosis_feature_cols = []
    for col in diagnosis_cols:
        block_name = col.replace("_present", "")
        time_col = f"{block_name}_time"
        diagnosis_feature_cols.append(col)
        if time_col in df.columns:
            diagnosis_feature_cols.append(time_col)

    # Build feature matrix all at once
    feature_data = {
        "Sex": df["Sex"].values,
    }
    for i, col in enumerate(numeric_cols):
        feature_data[col] = scaled_values[:, i]

    for col in diagnosis_feature_cols:
        feature_data[col] = df[col].values

    X_df = pd.DataFrame(feature_data)
    feature_names = X_df.columns.tolist()
    X_array = X_df.values

    logger.info(f"Built feature matrix: {X_array.shape[0]} samples x {X_array.shape[1]} features")

    return X_array, feature_names, scaler


def get_labels(df: pd.DataFrame) -> np.ndarray:
    """
    Extract and encode labels.

    Args:
        df: Input dataframe.

    Returns:
        Encoded labels (0 = Dementia, 1 = Control).
    """
    label_map = {"Dementia": 0, "Control": 1}
    y = df["Class"].map(label_map).values
    return y
