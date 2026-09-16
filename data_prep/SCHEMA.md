# Data schema

This project uses UK Biobank EHR data, which is access-controlled and **cannot be
redistributed or included in this repository** (see `README.md` at the repo root for
how to request access under application number 109607, as cited in the paper).

Every script in this repository expects a single CSV with the schema below. The
column names and encodings here were recovered directly from the original
preprocessing/modelling notebooks (`five.csv` / `five_updated.csv` in the original,
now-lost, working directory), so any real extract built the same way will drop in
without changes.

## Columns

| Column | Type | Description |
|---|---|---|
| `eid` | int | UK Biobank participant ID |
| `Class` | str | `"Dementia"` or `"Control"` |
| `Sex` | int (0/1) | 0 = female, 1 = male |
| `Age` | float | Age (years) at the 5-years-pre-diagnosis / index date |
| `Standard PRS for alzheimer's disease (AD)` | float | Polygenic risk score for Alzheimer's disease (note the literal apostrophe and spaces in the column name -- the original code does not rename it until the `_ft` preprocessing notebook, which replaces spaces with underscores) |
| `{diagnosis_block}_present` | float (0.0/1.0) | 1.0 if the participant had this ICD-10 block diagnosed more than 5 years before their dementia diagnosis / index date, else 0.0 |
| `{diagnosis_block}_time` | float (days), or NaN | Days between the diagnosis date and the 5-years-pre-index date. NaN (or absent) when `_present` is 0.0 |

`{diagnosis_block}` is an ICD-10 **block** description (e.g. `Hypertensive diseases`,
`Arthropathies`, `Diseases of oesophagus, stomach and duodenum`), not an individual
ICD-10 code. There are ~100-260 such blocks depending on how many appear in the
cohort; the original data had well over 100 `_present`/`_time` column pairs.

## Derived columns (computed by the scripts, not present in the raw CSV)

- `label`: `{"Dementia": 0, "Control": 1}` mapping of `Class`
- `PRS`: `Age`/PRS after `StandardScaler` (fit on train only)
- `icd10_sequence`: a semi-colon separated text string built from the present
  diagnoses and their time-before-index (in months), e.g.
  `"Hypertensive diseases (12_months_before); Arthropathies (48_months_before)"`,
  used as the input text for the BioClinicalBERT models.

## Splits

All notebooks use `sklearn.train_test_split` with `random_state=42`, stratified on
`label`. Two slightly different split ratios were used across the original notebooks
(preserved here per-model for faithfulness):

- BERT-only / multimodal BERT+MLP notebooks: 80/10/10 via two splits
  (`test_size=0.1` then `test_size=0.15` on the remainder) -> ~60/13.5/10 train/val/test.
- GNN notebook (`neurips_ad_graphs.ipynb`, the most complete/final pipeline, matching
  the paper's reported 80:20 train/test split): `test_size=0.2` then `test_size=0.125`
  on the remainder -> 70/10/20 train/val/test.

## Synthetic data

Because the real UK Biobank extract cannot be shared, `generate_synthetic_data.py`
fabricates a CSV with this exact schema (fake patients, fake diagnoses, fake PRS)
purely so that every script in this repository can be smoke-tested end-to-end. It is
clearly **not** real data and produces meaningless model performance -- it exists only
to prove the code runs, not to reproduce the paper's actual numbers.
