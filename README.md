# Reproducibility Package: Predicting Dementia Risk Using Longitudinal Electronic Health Records Data

Iona Biggart, Antigone Fogel, Payam Barnaghi — Imperial College London / UK Dementia Research Institute
NeurIPS 2025 Workshop on Time Series for Health (TS4H)

Paper: [openreview.net/forum?id=2FfRHgVSx3](https://openreview.net/forum?id=2FfRHgVSx3)

This package provides a clean, runnable, and documented version of the codebase behind the paper. 

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
│   ├── multimodal_classifier.py    <- BERT+MLP (ICD text + clinical features); see note below
│   ├── text_only_classifier.py     <- "BioClinical BERT pure"
│   ├── bioclinical_no_temporal.py  <- ablation
│   └── roberta_classifier.py       <- ablation
│       (per-model findings are in bert_models/README.md)
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

Each module directory has its own `README.md` with method-specific detail, usage examples, exact hyperparameters, and known caveats. 


## Quickstart  

### 1. Clone the repository

Alternatively, download the repository as a ZIP file and open the extracted directory in your terminal.

```bash
git clone https://github.com/ionaei/gnn-for-dementia
cd gnn-for-dementia
```

### 2. Create an environment and install dependencies

Read the **Environment note** below first. Choose **either venv or conda**.

#### Option A: venv

Requires **Python 3.9-3.11** for `torch` to resolve to this package's own tested version (`torch==2.1.2`); `requirements.txt` allows up through `<2.5.0` so Python 3.12+ will still install, just with a newer `torch` that hasn't been re-verified against this codebase (re-run a module's smoke test to confirm, or use conda below to match the tested setup exactly).

```bash
python3 -m venv .venv
source .venv/bin/activate      # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Option B: conda

See the conda installation link below if needed.

```bash
conda create -n dementia_gnn python=3.11.9
conda activate dementia_gnn
pip install -r requirements.txt
```

### 3. Generate synthetic data

Generate synthetic data for smoke-testing every module.

```bash
cd data_prep
python generate_synthetic_data.py --n-patients 2000 --out five_updated_synthetic.csv
cd ..
```

##### 4. Train and test the baselines

Train the RF and XGBoost models, generate SHAP explanations, and run the smoke-test suite.

```bash
cd baselines
python train_rf_xgb.py --model both --no-wandb --data-path ../data_prep/five_updated_synthetic.csv
python explain_shap.py --model both --data-path ../data_prep/five_updated_synthetic.csv
python test_baselines.py
cd ..
```

### 5. Train the GNN

Run the paper’s headline method.

```bash
cd gnn
python train.py --data-path ../data_prep/five_updated_synthetic.csv \
    --no-wandb --epochs 3 --hidden 32 --emb-dim 32 --batch-size 16
cd ..
```

### 6. Test GNN explainability

Run the synthetic-graph smoke test. No real data are needed.

```bash
cd explainability
python test_explainers.py
cd ..
```

### 7. Explain the trained GNN checkpoint

Export held-out patient graphs and explain the checkpoint from step 5. The export script uses the same data-loading and graph-building pipeline as training, keeping the code vocabulary consistent with the checkpoint.

```bash
cd explainability
python ../tools/export_test_graphs.py --data-path ../data_prep/five_updated_synthetic.csv \
    --out test_graphs.pkl
python explain.py --model ../gnn/checkpoints_gnn/best_local.pt \
    --graphs test_graphs.pkl --method gradient --output gradient_results.json
cd ..
```

### 8. Run risk stratification

Use the GNN checkpoint from step 5. Model settings (`hidden`, `emb-dim`, `pool`, and `head`) are loaded automatically from the checkpoint.

```bash
cd risk_stratification
python run_stratification.py --data-path ../data_prep/five_updated_synthetic.csv \
    --local-checkpoint ../gnn/checkpoints_gnn/best_local.pt
cd ..
```

### 9. Run the BERT models

Internet access is required to download pretrained weights.

```bash
cd bert_models
python multimodal_classifier.py --no-wandb --epochs 1
python text_only_classifier.py --no-wandb --epochs 1
cd ..
```

## Conda installation steps

Conda: [www.anaconda.com/docs/getting-started/concepts/anaconda-or-miniconda](https://www.anaconda.com/docs/getting-started/concepts/anaconda-or-miniconda)

If you would like to install the environement through conda, please install conda using the steps provided by anaconda in the link above. 


## Environment note: `torch` / `transformers` version compatibility

Recent `transformers` releases are **not** compatible with that `torch` version for the BERT models:

- `transformers>=4.41` and all of `transformers 5.x` fail at import or model-load time against `torch==2.1.2` (`AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'`, or, for 5.x specifically, an outright refusal to enable the PyTorch backend: `"requires the PyTorch library but it was not found in your environment"`, even with `torch` installed).
- `transformers==4.40.0` was confirmed (by installing it in this environment) to import cleanly and construct `AutoModel`/`AutoModelForSequenceClassification` against `torch==2.1.2` without error.

`requirements.txt` (and `bert_models/requirements.txt`) pin `transformers` to `==4.40.0` accordingly, and cap `torch` at `<2.5.0` so that ceiling stays valid. Both files leave `torch` itself as a range (`>=2.1.2,<2.5.0`) rather than an exact pin, because `torch==2.1.2` has no PyPI wheel for Python>=3.12 (see the venv note in the Quickstart above) — pip will resolve to the newest version in range with a wheel for your Python, which has not been separately re-verified against this codebase the way `torch==2.1.2` was (re-run a module's smoke test to check, or use conda with `python=3.11.9` to match the exact tested setup). If you upgrade `torch` to `>=2.5.0` in your own environment, the `transformers` ceiling can be lifted too. `gnn/`, `explainability/`, and `risk_stratification/` (which depend on `torch_geometric`, not `transformers`) are unaffected by the `transformers` compatibility issue and were fully re-run end-to-end against `torch==2.1.2` + `torch_geometric==2.8.0.post1` while preparing this package (see the Quickstart above).

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

A machine-readable citation (for GitHub's "Cite this repository" button and tools like `cffconvert`) is in [`CITATION.cff`](./CITATION.cff) at the repository root.

## Contact

For questions about the paper or requesting collaboration on UK Biobank application 109607, contact the corresponding author.
