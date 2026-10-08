"""
Canonical stratified train/val/test split, shared by gnn/, baselines/, and
bert_models/ so that every model in this package is scored on the same
held-out patients when pointed at the same CSV with the same seed.

Why this exists: an external code review found that `gnn/train.py` and
`baselines/train_rf_xgb.py` already used the identical split (two chained
`sklearn.train_test_split` calls, `test_size=0.2` then `test_size=0.125`,
both stratified on the label, both with `random_state=seed` -- i.e. 70%
train / 10% val / 20% test), but `bert_models/*.py` independently
implemented a *different* split (`test_size=0.1` then `test_size=0.15`,
which is actually ~76.5/13.5/10, not the "60/13.5/10" the code comments
claimed -- two chained percentages don't combine by simple subtraction).
Because the splits differed, the BERT family was evaluated on a different,
non-overlapping 10% test set from the one GNN/baselines used, so Table 1
rows from different model families were never comparable even before any
other bug. All three families now call `split_dataframe` below with the
same defaults.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

LABEL_MAP_IDX = {"Dementia": 0, "Control": 1}


def add_label_column(df: pd.DataFrame, class_col: str = "Class", label_col: str = "label") -> pd.DataFrame:
    """Attach the integer label column (0=Dementia, 1=Control) used for stratification."""
    df = df.copy()
    df[label_col] = df[class_col].map(LABEL_MAP_IDX)
    return df


def split_dataframe(df: pd.DataFrame, label_col: str = "label", seed: int = 42,
                     test_size: float = 0.2, val_size: float = 0.125):
    """
    Two-stage stratified split: first carve off `test_size` as the test set,
    then carve `val_size` of what remains off as the validation set.

    With the defaults (test_size=0.2, val_size=0.125) this produces a
    70/10/20 train/val/test split: 80% remains after the first split, and
    12.5% of that 80% is 10% of the original total.

    Returns:
        (df_train, df_val, df_test)
    """
    df_train_val, df_test = train_test_split(
        df, test_size=test_size, stratify=df[label_col], random_state=seed
    )
    df_train, df_val = train_test_split(
        df_train_val, test_size=val_size, stratify=df_train_val[label_col], random_state=seed
    )
    return df_train, df_val, df_test


def load_and_split_raw(data_path, seed: int = 42, class_col: str = "Class", label_col: str = "label",
                        test_size: float = 0.2, val_size: float = 0.125):
    """
    Load a CSV, attach the integer label column, and apply the canonical
    split. Returns (df_train, df_val, df_test) with no other
    feature-specific preprocessing (scaling, tokenization, graph-building,
    etc. are left to each model's own pipeline).
    """
    df = pd.read_csv(data_path)
    df = add_label_column(df, class_col=class_col, label_col=label_col)
    return split_dataframe(df, label_col=label_col, seed=seed, test_size=test_size, val_size=val_size)
