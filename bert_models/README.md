# BERT-Family Models for Dementia Risk Prediction

This directory contains clean, reproducible implementations of all BERT-based methods from the paper "Predicting Dementia Risk Using Longitudinal Electronic Health Records Data" (Biggart & Fogel et al., NeurIPS 2025 TS4H Workshop).

## Scripts Overview

### Core Utilities

- **`sequence_builder.py`** — Shared utility for building ICD-10 diagnosis sequences from the CSV schema. This is the single source of truth for text preprocessing. Used by all BERT models. Provides:
  - `build_icd10_sequence()` — Build text for a single patient
  - `apply_icd10_sequences()` — Apply to entire dataframe
  - Supports ablation via `include_temporal` and `include_demographics` flags

- **`seed_utils.py`** — `set_seed(seed)`: seeds Python's `random`, NumPy, and PyTorch (CPU+CUDA) before training. Fixes a real bug: none of these 4 scripts seeded anything beyond the (already-fixed, `random_state=42`) train/val/test split, so the classification head `AutoModelForSequenceClassification.from_pretrained(..., num_labels=2)` randomly initializes, `Dropout`, and the training `DataLoader(..., shuffle=True)`'s batch order all drew from unseeded global RNGs — identical CLI args gave different metrics run to run. All 4 scripts now take a `--seed` argument (default `42`) and call `set_seed(seed)` before building the model or any `DataLoader`.

---

### Paper Methods

#### 1. **`multimodal_classifier.py`** — BioClinicalBERT + MLP (HEADLINE PAPER METHOD)

**Paper Reference:** "Multimodal BioClinical BERT+MLP" (Table 1, NeurIPS 2025 TS4H poster)  
**Published Table 1 Results (pre-stratification, real UKB data, n=9,537 sex-matched):** F1=0.704, Sensitivity=0.694, Specificity=0.714, J=0.408, AUCROC=0.776 — verified directly against the poster PDF (`../../NeurIPS_poster.pdf`).
**Ported from:** `bioclinical_multimodalclassifier.ipynb` (uses a separate `MultiModalDementiaClassifier` with its own age/sex/PRS MLP branch, concatenated with the BioClinicalBERT [CLS] embedding — see Architecture below).
**Note:** the last saved cell output *in that notebook* reports Accuracy=0.6551, Sensitivity=0.9146, Specificity=0.3924 — a different (and more class-imbalanced-looking) run than the one behind the published Table 1 row above. Notebooks were re-run multiple times during development, and the last-saved output is not necessarily the checkpoint/run used for the final paper table. Both numbers are given here for full transparency; treat the poster's Table 1 numbers as the citable paper result, and the notebook figure as a historical data point from one particular run.

**Architecture:**
- Encodes diagnosis text (with temporal info) → BioClinicalBERT [CLS] token (768-dim)
- Encodes demographics (age, sex, PRS) → 2-layer MLP (32 hidden, ReLU, dropout)
- Concatenates embeddings (800-dim total) → 2-layer classifier (64 hidden)

**Train Config:**
- Model: `emilyalsentzer/Bio_ClinicalBERT`
- LR: 2e-5 (Adam), Epochs: 50, Batch size: 16
- Split: 60/13.5/10 train/val/test
- Optimizer: Adam, Loss: CrossEntropyLoss
- Early stopping: monitors validation loss

**Usage:**
```bash
# Synthetic smoke test (1 epoch, minimal data)
python multimodal_classifier.py --no-wandb --epochs 1

# Real UK Biobank data (50 epochs as per paper)
python multimodal_classifier.py --data-path /path/to/five.csv --epochs 50
```

---

#### 2. **`text_only_classifier.py`** — BioClinicalBERT Pure (TEXT-ONLY BASELINE)

**Paper Reference:** "BioClinical BERT pure" (Table 1, NeurIPS 2025 TS4H poster)  
**Published Table 1 Results (pre-stratification, real UKB data, n=9,537 sex-matched):** F1=0.707, Sensitivity=0.726, Specificity=0.670, J=0.396, AUCROC=0.767 — verified directly against the poster PDF.
**Ported from:** `version_2.ipynb`, which loads `AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=2)` directly on the text sequence (no separate structured-feature MLP) and saves its checkpoint as `best_model_pure_bioclinical_v1.pt` — i.e. this notebook **is** the "BioClinical BERT pure" method in Table 1, not a duplicate of the multimodal method (see the corrected note in "Findings from Original Notebooks" below).
**Note:** the last saved cell output in `version_2.ipynb` reports Accuracy=0.6960, Sensitivity=0.6813, Specificity=0.7110, a different run from the one behind the published Table 1 row (see the multimodal section above for why these numbers needn't match exactly).

**Architecture:**
- Input: text sequence (demographics + diagnoses + temporal info)
- Uses `AutoModelForSequenceClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", num_labels=2)`
- No separate MLP for demographics; all info encoded in text

**Train Config:**
- Model: `emilyalsentzer/Bio_ClinicalBERT`
- LR: 2e-6 (AdamW), Epochs: 20, Batch size: 16
- Split: 60/13.5/10 train/val/test
- Gradient clipping: max_norm=1.0
- Scheduler: linear warmup with 0 warmup steps

**Usage:**
```bash
python text_only_classifier.py --no-wandb --epochs 1
python text_only_classifier.py --data-path /path/to/five.csv --epochs 20
```

---

### Ablation Studies

#### 3. **`bioclinical_no_temporal.py`** — ABLATION: WITHOUT Temporal Information

**Purpose:** Tests whether encoding diagnosis timing ("X_months_before") is necessary for good performance.

**Input Text Format (vs full method):**
```
Full:         "sex=Male; age=67; polygenic_risk_score=0.125; Hypertensive diseases (12_months_before); Arthropathies (48_months_before)"
No Temporal:  "sex=Male; age=67; polygenic_risk_score=0.125; Hypertensive diseases; Arthropathies"
```

**Model:** BioClinicalBERT (same as text_only, but with temporal info removed)

**Train Config:**
- LR: 2e-6, Epochs: 15, Batch size: 16
- Same as text_only_classifier

**Expected Outcome:** Should underperform text_only (0.707 F1, published) if temporal information is predictively important.
**Ported from:** `bioclinical_no_temporal.ipynb`. Last saved cell output in that notebook: Accuracy=0.6971, Sensitivity=0.5813, Specificity=0.8143 (not directly comparable to the text-only F1 above since the notebook only reports accuracy/sensitivity/specificity, not F1/AUCROC — this ablation isn't broken out as its own row in the paper's Table 1).

**Usage:**
```bash
python bioclinical_no_temporal.py --no-wandb --epochs 1
```

---

#### 4. **`roberta_classifier.py`** — BERT-VARIANT: RoBERTa vs BioClinicalBERT

**Purpose:** Tests whether BioClinicalBERT's clinical pre-training is essential, or if a general-purpose model (RoBERTa-large) achieves comparable performance.

**Model:** `roberta-large` (can be changed via `--model` flag)

**Architecture:** Same as text_only_classifier but with RoBERTa instead of BioClinicalBERT.

**Train Config:**
- LR: 2e-6, Epochs: 15, Batch size: 16
- Same input text (demographics + diagnoses + temporal info)
- Gradient clipping: max_norm=1.0

**Expected Outcome:** RoBERTa should underperform BioClinicalBERT on clinical/medical terminology.
**Ported from:** `roberta.ipynb`. Last saved cell output in that notebook: Accuracy=0.6761, Sensitivity=0.6021, Specificity=0.7511 (not broken out as its own row in the paper's Table 1; also not directly comparable to the text-only F1 above since only accuracy/sensitivity/specificity were recorded).

**Usage:**
```bash
# RoBERTa-large (default)
python roberta_classifier.py --no-wandb --epochs 1

# Other BERT variants
python roberta_classifier.py --model bert-base-uncased --no-wandb --epochs 1
```

---

## Data Requirements

All scripts expect a CSV file matching the **SCHEMA.md** specification:

| Column | Type | Notes |
|--------|------|-------|
| `eid` | int | UK Biobank ID |
| `Class` | str | "Dementia" or "Control" |
| `Sex` | int | 0=Female, 1=Male |
| `Age` | float | Age at 5-year-pre-diagnosis index date |
| `Standard_PRS_for_alzheimer's_disease_(AD)` | float | Polygenic risk score |
| `{diagnosis_block}_present` | float (0.0/1.0) | 1.0 if diagnosis in 5-year window |
| `{diagnosis_block}_time` | float (days) | Days before index date; NaN if absent |

**Note:** Column names have spaces replaced with underscores by all scripts (see `df.columns.str.replace(' ', '_')`).

---

## Smoke Testing (Synthetic Data)

All scripts can generate synthetic data if `--data-path` is omitted. This creates a minimal dataset for testing code runs without real data:

```bash
cd /path/to/reproducibility_package/bert_models
python multimodal_classifier.py --no-wandb --epochs 1
```

The synthetic generator is in `../data_prep/generate_synthetic_data.py` and creates ~60 fake patients with the correct schema.

---

## Key Hyperparameters & Differences

| Aspect | Multimodal | Text-Only | No-Temporal | RoBERTa |
|--------|-----------|-----------|-------------|---------|
| **Model** | BioClinicalBERT | BioClinicalBERT | BioClinicalBERT | RoBERTa-large |
| **Structured Features** | Age, Sex, PRS (MLP) | None (text only) | None (text only) | None (text only) |
| **Temporal Info** | Yes | Yes | **No** | Yes |
| **LR** | 2e-5 | 2e-6 | 2e-6 | 2e-6 |
| **Epochs** | 50 | 20 | 15 | 15 |
| **Gradient Clip** | None | 1.0 | 1.0 | 1.0 |

---

## Checkpoints & Output

All scripts save the best model checkpoint to `--checkpoint-dir` (default: `./checkpoints/`):

```
checkpoints/
  best_multimodal_classifier.pt
  best_text_only_classifier.pt
  best_no_temporal_classifier.pt
  best_roberta_large_classifier.pt
```

Test metrics (Accuracy, Sensitivity, Specificity, F1, AUCROC) are printed to console and optionally logged to Weights & Biases (WandB).

---

## WandB Integration

All scripts use Weights & Biases for experiment tracking by default:

```bash
# With WandB (requires wandb login)
python multimodal_classifier.py --data-path /path/to/data.csv

# Without WandB (smoke testing, CI/CD)
python multimodal_classifier.py --data-path /path/to/data.csv --no-wandb
```

WandB config includes learning rate, epochs, batch size, and logs per-epoch/per-step metrics.

---

## Training on Real Data

To train on real UK Biobank data (access-controlled; not included):

1. Obtain a CSV extract matching `SCHEMA.md` (see repository root README for UK Biobank application 109607)
2. Run any script with `--data-path /path/to/five.csv`:
   ```bash
   python multimodal_classifier.py --data-path /path/to/five.csv --epochs 50
   ```

Expected real results (published Table 1, pre-stratification, verified against `../../NeurIPS_poster.pdf`):
- **Multimodal BioClinicalBERT+MLP:** F1=0.704, Sens=0.694, Spec=0.714, AUC=0.776
- **Text-Only BioClinicalBERT ("BioClinical BERT pure"):** F1=0.707, Sens=0.726, Spec=0.670, AUC=0.767
- **No-Temporal ablation:** Accuracy=0.6971 (last notebook run; not a Table 1 row, no F1/AUC recorded)
- **RoBERTa ablation:** Accuracy=0.6761 (last notebook run; not a Table 1 row, no F1/AUC recorded)

Note: actual performance depends on real data size, cohort composition, and exact preprocessing/seed; the two headline rows are the published paper numbers, while the two ablations only ever had accuracy/sensitivity/specificity recorded in their source notebooks (see "Findings from Original Notebooks" above for the full picture, including why a notebook's last-saved output and the published number can legitimately differ).

---

## Assumptions & Caveats

1. **HuggingFace Model Download:** All scripts require internet access to download `emilyalsentzer/Bio_ClinicalBERT` and `roberta-large` from Hugging Face Hub on first run. This happens automatically; if network is unavailable, the scripts will fail gracefully.

2. **GPU Availability:** Training is significantly faster on GPU (NVIDIA A100 used in paper). CPU-only training is possible but slow.

3. **WandB Optional:** Weights & Biases is imported opportunistically; if not installed or `--no-wandb` is passed, training proceeds without logging.

4. **Synthetic Data:** Generated data is fake and produces meaningless metrics. It exists only to test that code runs without real data.

5. **Column Name Normalization:** Original data has spaces in column names (e.g., `Standard PRS for alzheimer's disease (AD)`). All scripts normalize to underscores. The `sequence_builder.py` handles both variants.

6. **Label Encoding:** Fixed mapping: `{"Dementia": 0, "Control": 1}`. Metrics (Sensitivity, Specificity) respect this encoding:
   - **Sensitivity** = Recall of class 0 (Dementia) = TP / (TP + FN)
   - **Specificity** = Recall of class 1 (Control) = TN / (TN + FP)

7. **Early Stopping:** Implemented via saving best checkpoint based on validation loss, not an explicit EarlyStopping callback. Training runs for full epoch count unless manually interrupted.

8. **Determinism (fixed):** All 4 scripts now accept `--seed` (default `42`) and call `seed_utils.set_seed()` before model/DataLoader construction, so repeated runs with identical CLI args give identical metrics on CPU. This was verified against the underlying mechanism (random classification-head init + `DataLoader` shuffle order + `Dropout`, all bit-identical across runs under the same seed) using a locally-constructed, un-pretrained BERT config — this environment had no network access to Hugging Face Hub to re-verify end-to-end against the real `Bio_ClinicalBERT`/`roberta-large` checkpoints, but the fix uses the exact same `torch.manual_seed`-based mechanism already verified end-to-end in `gnn/train.py` (see top-level README's "Reproducibility" section). GPU (CUDA) runs remain best-effort, not guaranteed bit-identical, per the note in `seed_utils.py`.

---

## References

- Paper: "Predicting Dementia Risk Using Longitudinal Electronic Health Records Data" (Imperial College London, UK DRI, NeurIPS 2025 TS4H)
- BioClinicalBERT: Alsentzer et al., "Publicly Available Clinical BERT Embeddings" (ACL 2019)
- RoBERTa: Liu et al., "RoBERTa: A Robustly Optimized BERT Pretraining Approach" (ICLR 2020)
- Data Schema: `../data_prep/SCHEMA.md`
- Synthetic Data Generator: `../data_prep/generate_synthetic_data.py`

---

## Questions or Issues?

- For real UK Biobank data access, contact the original authors (application 109607).
- For code issues, ensure dependencies are installed: `pip install torch transformers scikit-learn pandas wandb`
- All scripts print detailed progress; check console output for debugging.
