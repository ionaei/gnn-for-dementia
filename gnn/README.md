# Star-Graph GNN for Dementia Risk Prediction (Headline Method)

This directory implements `PatientICDGNN_BioBERT`, the paper's best-performing model (Table 1 rows "GNN+RV" and "GNN+BioClinical", and their "+MLP" variants) — a Graph Isomorphism Network with Edge features (GINEConv) trained on per-patient star graphs built from ICD-10 diagnosis history.

## Method (Section 3.1 of the paper)

Each patient is represented as a **complete bipartite star graph**:

- One central **patient node**, features `[age, sex, PRS]` (standardized).
- One leaf **diagnosis node** per ICD-10 block the patient has on record, with node features looked up from a shared code-embedding matrix (either randomly-initialized/trainable — "RV" in Table 1 — or precomputed BioClinicalBERT text embeddings — "BioClinical" in Table 1).
- Edges connect the patient node to each diagnosis leaf, carrying 3 temporal attributes: `[years_since_diagnosis, log1p(days_since_diagnosis), exp(-days_since_diagnosis/180)]`.

The model runs 2 layers of `GINEConv` message passing over this graph, pools node embeddings (mean/add/max, tuned), and classifies with either a single linear layer ("GNN+RV"/"GNN+BioClinical", the paper's best configuration) or a small MLP head ("GNN+MLP+RV"/"GNN+MLP+BioClinical" ablation).

## Files

- **`graph_construction.py`** — Builds the diagnosis-code vocabulary and converts each patient row into a `torch_geometric.data.Data` star graph. Recovered verbatim from `neurips_ad_graphs.ipynb` cells 1-8. Also provides `make_random_code_embeddings()` for the "RV" (randomly-initialized) embedding variant.
- **`code_embeddings_bert.py`** — Builds the "BioClinical" code-embedding variant by mean-pooling BioClinicalBERT token embeddings of each diagnosis block's text.
- **`model.py`** — `PatientICDGNN_BioBERT`, the GINEConv model itself, merging two near-duplicate class definitions found across `neurips_ad_graphs.ipynb` (linear head, tuned pooling — the paper's best config) and `traffic_light.ipynb` (MLP head, pooling hardcoded to `"add"` — the GNN+MLP ablation) into one class with a `head` argument.
- **`sweep_config.py`** — The WandB Bayesian hyperparameter search space, recovered verbatim from `neurips_ad_graphs.ipynb` cell 13 (matches the paper's Appendix A2).
- **`train.py`** — CLI training script: loads data, builds graphs, and either runs a single fixed-hyperparameter training run (`--no-wandb`, for smoke-testing) or launches the real WandB Bayesian sweep (`--sweep`). Preserves the original optimizer param groups (separate LR for decay/no-decay/embedding params), mixed-precision training, linear warmup+decay LR schedule, gradient clipping, and early-stopping-on-val-loss logic.
- **`evaluate_best_run.py`** — Loads a trained checkpoint (either the best run of a WandB sweep, or a local `--no-wandb` checkpoint) and evaluates it on the held-out test set. This fixes two bugs (a stray typo and an undefined function call) that meant the original notebook cell this was ported from (`neurips_ad_graphs.ipynb` cell 20) could never actually run.

## Usage

```bash
# 1. Generate synthetic data for smoke testing (real UK Biobank data is access-controlled)
python ../data_prep/generate_synthetic_data.py --n-patients 2000 --out ../data_prep/five_updated_synthetic.csv

# 2. Smoke-test training: no WandB, a few epochs, small hidden dims, CPU-friendly
python train.py --data-path ../data_prep/five_updated_synthetic.csv \
    --no-wandb --epochs 3 --hidden 32 --emb-dim 32 --batch-size 16 \
    --checkpoint-dir checkpoints_gnn

# 3. Evaluate the resulting local checkpoint
python evaluate_best_run.py --local-checkpoint checkpoints_gnn/best_local.pt \
    --data-path ../data_prep/five_updated_synthetic.csv --hidden 32 --emb-dim 32

# 4. Real Bayesian hyperparameter sweep (requires `wandb login` and real data)
python train.py --data-path /path/to/five_updated.csv --sweep --sweep-count 30
```

`checkpoints_gnn/best_local.pt` in this directory is a checkpoint produced by an actual `--no-wandb` smoke-test run on synthetic data during development of this package — it proves the training loop runs end-to-end, but (like all synthetic-data results in this package) its metrics are not meaningful and should not be compared against Table 1.

## Data split

70% train / 10% validation / 20% test, stratified on label, via two chained `sklearn.train_test_split` calls with `random_state=42` (`test_size=0.2` then `test_size=0.125` on the remainder) — matching the paper's reported 80:20 train:test split and consistent with `data_prep/SCHEMA.md`. `Age` and PRS are standardized with a `StandardScaler` fit on the training split only.

## Relationship to `explainability/` and `risk_stratification/`

- `explainability/` consumes `PatientICDGNN_BioBERT` checkpoints trained here to compute gradient-based and guided-backpropagation feature attributions (see `explainability/README.md`).
- `risk_stratification/run_stratification.py` loads a checkpoint from this directory (or a WandB run) and applies Youden's J / "traffic light" confidence-band analysis on top of its test-set predictions (see `risk_stratification/README.md`).
