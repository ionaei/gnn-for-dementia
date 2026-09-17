# Reproducibility Package: Predicting Dementia Risk Using Longitudinal Electronic Health Records Data

Iona Biggart, Antigone Fogel, Payam Barnaghi — Imperial College London / UK Dementia Research Institute
NeurIPS 2025 Workshop on Time Series for Health (TS4H)

Paper: [openreview.net/forum?id=2FfRHgVSx3](https://openreview.net/forum?id=2FfRHgVSx3)

This package reconstructs a clean, runnable, and documented version of the codebase behind the paper. 

## What's in this package

| Module | Paper method(s) | 
|---|---|
| `data_prep/` | Data schema + synthetic data generator |
| `baselines/` | XGBoost, Random Forest (Table 1 "XGBoost" row) | 
| `bert_models/` | BioClinical BERT pure, Multimodal BioClinical BERT+MLP, + 2 ablations (no-temporal, RoBERTa) | 
| `gnn/` | Star-graph GINEConv GNN (paper's best method: GNN+RV / GNN+BioClinical, ± MLP head) | 
| `explainability/` | Gradient saliency + Guided BackPropagation explainability for the GNN | 
| `risk_stratification/` | Youden's J / "traffic light" risk banding (Table 1 "after risk stratification" rows) |

Every module can be smoke-tested end-to-end on synthetic data with no external dependencies beyond what's in `requirements.txt` (BERT models additionally need internet access to download pretrained weights from Hugging Face Hub on first run).

## The elephant in the room: real data is not included

This project uses UK Biobank electronic health records, which are **access-controlled and cannot be redistributed** under the terms of the project's data access application (application number **109607**). Every script in this package therefore:

1. Expects a CSV in the schema documented in `data_prep/SCHEMA.md`.
2. Can be smoke-tested against `data_prep/generate_synthetic_data.py`'s fake-but-schema-matching output instead.
3. Will **not** reproduce the paper's actual Table 1 numbers on synthetic data — synthetic results are structurally valid (the code runs, the metrics are well-formed) but scientifically meaningless, since the synthetic cohort has no real biological signal.

To reproduce the paper's actual numbers, apply for UK Biobank access (application 109607) and build a CSV extract matching `data_prep/SCHEMA.md`, then point any script's `--data-path` at it.

## Repository structure

```
reproducibility_package/
├── README.md                      <- this file
├── requirements.txt                <- union of all module dependencies
├── Dockerfile, .dockerignore       <- containerized setup (see DOCKER.md)
├── entrypoint.sh                   <- Docker entrypoint: one subcommand per module
├── .devcontainer/devcontainer.json <- open this repo in VS Code via the same Dockerfile
├── DOCKER.md                       <- Docker prerequisites + step-by-step instructions
├── data_prep/                      <- schema docs + synthetic data generator
│   ├── SCHEMA.md
│   ├── icd10_blocks.py
│   ├── generate_synthetic_data.py
│   └── five_updated_synthetic.csv  <- pre-generated synthetic dataset (2000 patients)
├── baselines/                      <- RF/XGBoost 
│   ├── feature_engineering.py
│   ├── train_rf_xgb.py
│   ├── explain_shap.py
│   ├── test_baselines.py
│   └── checkpoints/, explainability_outputs/  <- produced by running the scripts
├── bert_models/                    <- 4 consolidated BERT-family scripts
│   ├── sequence_builder.py
│   ├── seed_utils.py               <- set_seed(): makes runs with the same --seed reproducible
│   ├── multimodal_classifier.py    <- headline method
│   ├── text_only_classifier.py     <- "BioClinical BERT pure"
│   ├── bioclinical_no_temporal.py  <- ablation
│   └── roberta_classifier.py       <- ablation
│       (per-notebook findings + provenance are in bert_models/README.md)
├── gnn/                             <- star-graph GINEConv GNN (paper's best method)
│   ├── graph_construction.py, model.py, sweep_config.py
│   ├── code_embeddings_bert.py     
│   ├── checkpoint_utils.py         <- shared checkpoint save/load, embeds pool/hidden/emb_dim/head
│   ├── seed_utils.py               <- set_seed(): makes runs with the same --seed reproducible
│   ├── train.py, evaluate_best_run.py
│   └── checkpoints_gnn/            <- produced by running train.py
├── explainability/                  <- gradient / guided-backprop GNN explainability
│   ├── gradient_explainer.py, guided_backprop_explainer.py, explain.py
│   └── test_explainers.py
├── risk_stratification/             <- Youden's J / traffic-light risk banding
│   ├── traffic_light_stratification.py
│   └── run_stratification.py
└── tools/
    └── export_test_graphs.py       <- pickles held-out test graphs for explain.py --graphs
```

Each module directory has its own `README.md` with method-specific detail, usage examples, exact hyperparameters, and known caveats. This top-level README is the map; the module READMEs are the territory.


## Quickstart

```bash
# 0. Clone repository
git clone https://github.com/ionaei/gnn-for-dementia
cd gnn-for-dementia

# 1. Create environment and install dependencies (see the "Environment note" below before doing this
# 1a. .venv environment (if conda not installed). If you run 1a, dont run 1b and vice versa.
python3 -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

# 1b.   if you want to use conda environment
conda create -n dementia_gnn python=3.11.9
conda activate dementia_gnn
pip install -r requirements.txt

# 2. Generate synthetic data for smoke-testing every module
cd data_prep
python generate_synthetic_data.py --n-patients 2000 --out five_updated_synthetic.csv
cd ..

# 3. Baselines (RF/XGBoost) + SHAP explanations
cd baselines
python train_rf_xgb.py --model both --no-wandb --data-path ../data_prep/five_updated_synthetic.csv
python explain_shap.py --model both --data-path ../data_prep/five_updated_synthetic.csv
python test_baselines.py   # full smoke-test suite, asserts on outputs
cd ..

# 4. GNN (paper's headline method)
cd gnn
python train.py --data-path ../data_prep/five_updated_synthetic.csv \
    --no-wandb --epochs 3 --hidden 32 --emb-dim 32 --batch-size 16
cd ..

# 5. GNN explainability
cd explainability
python test_explainers.py   # synthetic-graph smoke test (no real data needed)

# 5b. Explain the actual GNN checkpoint from step 4 against real held-out patients:
# tools/export_test_graphs.py reuses the same load_and_split + build_graphs path as
# training, so the exported graphs' code vocabulary matches the checkpoint.
python ../tools/export_test_graphs.py --data-path ../data_prep/five_updated_synthetic.csv \
    --out test_graphs.pkl
python explain.py --model ../gnn/checkpoints_gnn/best_local.pt \
    --graphs test_graphs.pkl --method gradient --output gradient_results.json
cd ..

# 6. Risk stratification on the GNN checkpoint from step 4.
# gnn/train.py embeds hidden/emb-dim/pool/head in the checkpoint itself (see
# gnn/checkpoint_utils.py), so run_stratification.py picks them up automatically --
# you don't need to (and shouldn't have to) repeat them here.
cd risk_stratification
python run_stratification.py --data-path ../data_prep/five_updated_synthetic.csv \
    --local-checkpoint ../gnn/checkpoints_gnn/best_local.pt
cd ..

# 7. BERT models (requires internet access to download pretrained weights)
cd bert_models
python multimodal_classifier.py --no-wandb --epochs 1
python text_only_classifier.py --no-wandb --epochs 1
cd ..
```

## Alternative setup: Docker

If you'd rather not set up a Python environment by hand, the whole
pipeline above (baselines, GNN, explainability, risk stratification, BERT
models) is also runnable from a single Docker image — build once, then one
`docker run` per module, with results landing in a folder on your host
machine:

```bash
docker build -t dementia-repro .
docker run --rm dementia-repro baselines-test      # self-contained smoke test
docker run --rm -v "$(pwd)/output":/output dementia-repro gnn-train --epochs 5
```

See **`DOCKER.md`** for prerequisites (is Docker Desktop needed? do you need
a GPU?), the full step-by-step guide, a command reference table, and an
honest account of what was and wasn't verified while preparing the image
(no Docker daemon was available in the environment this image was authored
in, so `docker build`/`docker run` themselves weren't run there — see
`DOCKER.md` for what was verified instead, and please do a first real build
and smoke test on your end).

If you use VS Code, `.devcontainer/devcontainer.json` builds this same
Dockerfile automatically — open the repo and choose "Reopen in Container"
(requires the [Dev
Containers](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
extension). See `DOCKER.md`'s "Running it in VS Code" section for the exact
steps.

## Environment note: `torch` / `transformers` version compatibility

This package's development environment is pinned to `torch==2.1.2` (aarch64 hardware / disk-space constraints). While preparing this package, we found — empirically, not just by reading changelogs — that recent `transformers` releases are **not** compatible with that `torch` version for the BERT models:

- `transformers>=4.41` and all of `transformers 5.x` fail at import or model-load time against `torch==2.1.2` (`AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'`, or, for 5.x specifically, an outright refusal to enable the PyTorch backend: `"requires the PyTorch library but it was not found in your environment"`, even with `torch` installed).
- `transformers==4.40.0` was confirmed (by installing it in this environment) to import cleanly and construct `AutoModel`/`AutoModelForSequenceClassification` against `torch==2.1.2` without error.

`requirements.txt` (and `bert_models/requirements.txt`) pin `transformers` to `>=4.30.0,<=4.40.0` accordingly. If you upgrade `torch` to `>=2.5.0` in your own environment, this ceiling can be lifted. `gnn/`, `explainability/`, and `risk_stratification/` (which depend on `torch_geometric`, not `transformers`) are unaffected by this and were fully re-run end-to-end against `torch==2.1.2` + `torch_geometric==2.8.0` while preparing this package (see the Quickstart above).

## Paper Results 

For reference — these are the paper's actual published numbers on real UK Biobank data, **not** reproducible from the synthetic smoke-test data in this package:

**Before risk stratification:**

| Model | F1 | Sensitivity | Specificity | J | AUCROC |
|---|---|---|---|---|---|
| GNN+MLP+RV | 0.709 | 0.696 | 0.735 | 0.428 | 0.780 |
| GNN+RV | 0.716 | 0.761 | 0.673 | 0.434 | 0.777 |
| GNN+BioClinical | 0.709 | 0.746 | 0.672 | 0.415 | 0.775 |
| GNN+MLP+BioClinical | 0.713 | 0.753 | 0.673 | 0.426 | 0.770 |
| BioClinical BERT pure | 0.707 | 0.726 | 0.670 | 0.396 | 0.767 |
| Multimodal BioClinical BERT+MLP | 0.704 | 0.694 | 0.714 | 0.408 | 0.776 |
| XGBoost | 0.708 | 0.705 | 0.711 | 0.416 | 0.773 |

**After risk stratification:**

| Model | F1 | Sensitivity | Specificity | J | AUCROC |
|---|---|---|---|---|---|
| GNN+RV | 0.816 | 0.834 | 0.758 | 0.592 | 0.838 |
| XGBoost | 0.810 | 0.821 | 0.780 | 0.601 | 0.844 |

Sensitivity = recall for the Dementia class (label 0); Specificity = recall for the Control class (label 1), throughout this package.

## Citation

If you use this code, please cite:

> Biggart, I., Fogel, A., Barnaghi, P. "Predicting Dementia Risk Using Longitudinal Electronic Health Records Data." NeurIPS 2025 Workshop on Time Series for Health (TS4H). https://openreview.net/forum?id=2FfRHgVSx3

## Contact

For questions about the paper or requesting collaboration on UK Biobank application 109607, contact the corresponding author.
