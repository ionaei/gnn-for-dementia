"""
Utility for building ICD-10 diagnosis sequences for BERT models.

This module provides a single source of truth for constructing the icd10_sequence
column from diagnosis features in the data CSV. All BERT-family models use this
sequence as their text input.

The sequence format is:
  "sex=<Male|Female>; age=<int>; polygenic_risk_score=<float>; <diagnosis> (<N>_months_before); ..."

Diagnoses are sorted by time descending (farthest in the past first).
"""

import pandas as pd
import numpy as np
from typing import Tuple, List


def build_icd10_sequence(
    row: pd.Series,
    include_temporal: bool = True,
    include_demographics: bool = True,
) -> str:
    """
    Build a semi-colon-separated text sequence of diagnoses for a single patient.

    Args:
        row: A pandas Series from the dataframe (one patient row).
        include_temporal: If True, append time-before-index info to each diagnosis.
                         If False (ablation), only diagnosis names without timing.
        include_demographics: If True, prepend sex, age, PRS to the sequence.
                            If False, skip demographics (text-only variant).

    Returns:
        A formatted string ready to be tokenized by BERT.

    Example:
        >>> row = df.iloc[0]
        >>> seq = build_icd10_sequence(row)
        >>> print(seq)
        "sex=Male; age=67; polygenic_risk_score=0.125; Hypertensive diseases (12_months_before); Arthropathies (48_months_before)"
    """
    # Detect diagnosis columns (all columns ending in '_present')
    present_cols = [c for c in row.index if c.endswith('_present')]
    time_cols_dict = {
        c.replace('_present', ''): c.replace('_present', '_time')
        for c in present_cols
    }

    # Build demographic prefix
    parts = []
    if include_demographics:
        sex = "Male" if row['Sex'] == 1 else "Female"
        age = int(row['Age'])
        prs = row.get(
            "Standard_PRS_for_alzheimer's_disease_(AD)",
            row.get("PRS", 0.0)
        )
        parts.append(f"sex={sex}")
        parts.append(f"age={age}")
        parts.append(f"polygenic_risk_score={prs}")

    # Build diagnosis list
    events = []
    for present_col in present_cols:
        if row[present_col] == 1.0 or row[present_col] == 1:
            diagnosis_name = present_col.replace('_present', '')
            if include_temporal:
                time_col = time_cols_dict[diagnosis_name]
                time_days = row[time_col]
                if pd.notna(time_days):
                    months = round(time_days / 30)
                    event_str = f"{diagnosis_name} ({months}_months_before)"
                    events.append((time_days, event_str))
            else:
                # No temporal info: just the diagnosis name
                events.append((0, diagnosis_name))

    # Sort by time descending (farthest in the past first)
    events.sort(reverse=True)

    # Concatenate diagnoses
    diagnosis_str = "; ".join([event[1] for event in events])

    # Build final sequence
    if parts and diagnosis_str:
        return "; ".join(parts) + "; " + diagnosis_str
    elif parts:
        return "; ".join(parts)
    else:
        return diagnosis_str


def apply_icd10_sequences(
    df: pd.DataFrame,
    include_temporal: bool = True,
    include_demographics: bool = True,
    column_name: str = "icd10_sequence",
) -> pd.DataFrame:
    """
    Apply build_icd10_sequence to all rows in a dataframe.

    Args:
        df: Input dataframe with diagnosis columns (_present, _time).
        include_temporal: Whether to include time annotations.
        include_demographics: Whether to include sex/age/PRS prefix.
        column_name: Name of the output column.

    Returns:
        A copy of df with the new sequence column added.
    """
    df_copy = df.copy()
    df_copy[column_name] = df.apply(
        lambda row: build_icd10_sequence(
            row,
            include_temporal=include_temporal,
            include_demographics=include_demographics,
        ),
        axis=1,
    )
    return df_copy
