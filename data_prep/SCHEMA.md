# Data schema

This project uses UK Biobank EHR data, which is access-controlled and **cannot be
redistributed or included in this repository**. See `README.md` at the repo root for
how to request access (application number 109607, as cited in the paper).

Every script in this repository expects a single CSV with the schema below.

## Cohort construction (5-year prediction horizon)

The CSV was built from UK Biobank hospital inpatient records and participant data.

**Source fields.**
- ICD-10 diagnoses: Field 41270 (Diagnoses – ICD10), with dates from Field 41280
  (Date of first in-patient diagnosis – ICD10), linked by array index.
- Sex: Field 31
- Year of birth: Field 34
- Date of death: Field 40000
- Standard PRS for Alzheimer's disease (AD): Field 26206

**Case definition.** Cases have Alzheimer's disease: any ICD-10 code F00 or G30 in
their inpatient record. The case's **index date** is the date of their earliest
F00/G30 record. In the `Class` column, these cases are labelled `"Dementia"`.

**Control definition.** Controls have none of F00, F01 or G30, and none of the
following exclusion codes: F02, F03, F04, F05, F06.7, G31, G32 or Q90.

**Matching.** Each case was matched to up to 5 controls with the same sex and exact
year of birth, sampled without replacement (`random_state=42`). Controls had to be
alive on the case's index date, and no control was used for more than one case.
Each control takes its matched case's index date. Matching was done on the full
cohort, before the prediction-horizon filter below.

**5-year horizon.** The **cut-off date** is the index date minus 1,826 days (5 years).
Only diagnoses recorded on or before the cut-off date are kept, and participants with
no diagnoses before it drop out at this step. Participants with a missing PRS are
excluded.

**Diagnosis features.** ICD-10 codes are grouped into ICD-10 blocks (listed in
`icd10_blocks.py`), and each block becomes one `_present`/`_time` column pair (see
below).

## Columns

| Column | Type | Description |
|---|---|---|
| `eid` | int | UK Biobank participant ID |
| `Class` | str | `"Dementia"` (Alzheimer's disease case, as defined above) or `"Control"` |
| `Sex` | int (0/1) | 0 = female, 1 = male |
| `Age` | float | Age (years) at the cut-off date: index-date year minus year of birth minus 5 |
| `Standard PRS for alzheimer's disease (AD)` | float | Polygenic risk score for Alzheimer's disease (Field 26206). The column name contains a literal apostrophe and spaces. |
| `{diagnosis_block}_present` | float (0.0/1.0) | 1.0 if the participant had a diagnosis in this ICD-10 block before the cut-off date, else 0.0 |
| `{diagnosis_block}_time` | float (days), or NaN | Days between the diagnosis date and the cut-off date. NaN when `_present` is 0.0 |

`{diagnosis_block}` is an ICD-10 **block** description (e.g. `Hypertensive diseases`,
`Arthropathies`, `Diseases of oesophagus, stomach and duodenum`), not an individual
ICD-10 code. There is one `_present`/`_time` pair per block; 128 blocks are defined
in `icd10_blocks.py`.

## Derived columns (computed by the scripts, not present in the raw CSV)

- `label`: `{"Dementia": 0, "Control": 1}` mapping of `Class`.
- Standardised `Age` and PRS: the GNN and multimodal BERT pipelines standardise these
  with `StandardScaler`, fitted on the training set only. The tree-based baselines use
  the raw values.
- `icd10_sequence`: the input text for the BERT models. It starts with the
  participant's sex, age and PRS, followed by their diagnoses ordered from oldest to
  most recent, each with the number of months between the diagnosis and the cut-off
  date, e.g.
  `"sex=Male; age=67; polygenic_risk_score=0.125; Arthropathies (48_months_before); Hypertensive diseases (12_months_before)"`.
  The no-temporal ablation omits the months.

## Splits

All models use one stratified 70/10/20 train/validation/test split
(`common/data_split.py`, `random_state=42`, stratified on `label`).

## Synthetic data

`generate_synthetic_data.py` creates fake data with this exact schema so every script
can be tested end to end without UK Biobank access; because this is fake data, results on it are meaningless.
