"""
Generate a small, fully synthetic dataset matching the schema described in
SCHEMA.md, so that every script in this repository can be run end-to-end as
a smoke test WITHOUT access to the real (access-controlled) UK Biobank data.

IMPORTANT: This produces fake patients, fake diagnoses, and a fake PRS. It is
NOT derived from and does not resemble any real individual's data. Model
performance on this synthetic dataset is meaningless -- its only purpose is
to prove that the preprocessing / baseline / BERT / GNN / risk-stratification
code runs without errors.

Usage:
    python generate_synthetic_data.py --n-patients 2000 --out five_updated_synthetic.csv
"""

import argparse

import numpy as np
import pandas as pd

from icd10_blocks import ICD10_BLOCKS

RNG_SEED = 42


def generate_synthetic_cohort(n_patients: int, rng: np.random.Generator) -> pd.DataFrame:
    eids = np.arange(1_000_000, 1_000_000 + n_patients)

    # Roughly balanced classes, matching the real cohort's near 50/50 split
    # (real data: Dementia=4795, Control=4742; see preprocessing_ft.ipynb output).
    labels = rng.choice(["Dementia", "Control"], size=n_patients, p=[0.503, 0.497])

    sex = rng.integers(0, 2, size=n_patients)  # 0 = female, 1 = male

    # Age at index date. Real cohort medians (Table A1 of the paper):
    # Control 70.5 [39.0, 81.0], Dementia 66.9 [38.0, 80.0].
    age = np.where(
        labels == "Dementia",
        np.clip(rng.normal(66.9, 7.0, size=n_patients), 38.0, 80.0),
        np.clip(rng.normal(70.5, 6.0, size=n_patients), 39.0, 81.0),
    )

    # Polygenic risk score for Alzheimer's disease: standard-normal-ish, with a
    # small mean shift for the Dementia class so the synthetic signal is not
    # pure noise (purely for smoke-testing purposes -- not a real effect size).
    prs = np.where(
        labels == "Dementia",
        rng.normal(0.15, 1.0, size=n_patients),
        rng.normal(-0.05, 1.0, size=n_patients),
    )

    base_cols = {
        "eid": eids,
        "Class": labels,
        "Sex": sex,
        "Age": age,
        "Standard PRS for alzheimer's disease (AD)": prs,
    }

    # Each patient is diagnosed with a random subset of ICD-10 blocks more than
    # 5 years before their index date. Dementia patients get a slightly higher
    # average diagnosis burden, again purely to give downstream models some
    # (meaningless) signal to chew on during smoke testing.
    diagnosis_cols = {}
    for block in ICD10_BLOCKS:
        base_rate = rng.uniform(0.02, 0.18)  # background prevalence per block
        dementia_bump = rng.uniform(0.0, 0.05)
        p_present = np.where(labels == "Dementia", base_rate + dementia_bump, base_rate)
        present = rng.binomial(1, p_present).astype(float)

        # Days between diagnosis and the 5-years-pre-index date. Only defined
        # when present == 1; NaN otherwise (per SCHEMA.md).
        days = rng.uniform(30, 3650, size=n_patients)
        days = np.where(present == 1.0, days, np.nan)

        diagnosis_cols[f"{block}_present"] = present
        diagnosis_cols[f"{block}_time"] = days

    df = pd.concat(
        [pd.DataFrame(base_cols), pd.DataFrame(diagnosis_cols)],
        axis=1,
    )

    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-patients", type=int, default=2000, help="Number of synthetic patients to generate")
    parser.add_argument(
        "--out",
        type=str,
        default="five_updated_synthetic.csv",
        help="Output CSV path (written relative to the current working directory)",
    )
    parser.add_argument("--seed", type=int, default=RNG_SEED)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    df = generate_synthetic_cohort(args.n_patients, rng)
    df.to_csv(args.out, index=False)

    n_dem = (df["Class"] == "Dementia").sum()
    n_ctrl = (df["Class"] == "Control").sum()
    print(f"Wrote {len(df)} synthetic patients to {args.out}")
    print(f"  Dementia: {n_dem}  Control: {n_ctrl}")
    print(f"  Columns: {len(df.columns)} ({len(ICD10_BLOCKS)} ICD-10 blocks x 2 + 5 base columns)")


if __name__ == "__main__":
    main()
