# Running this package in Docker

This container exists so a reviewer can run the reproducibility pipeline
without setting up a Python environment by hand — one `docker build`, then
one `docker run` per module, and results land in a folder on your host
machine. It ships to three requirements:

1. **Everything in one image** — baselines, GNN, explainability, risk
   stratification, and the BERT-family models are all runnable from the
   same image via one entrypoint script.
2. **No pre-trained model weights baked in** — the image ships code only
   (plus the small synthetic smoke-test CSV, see "What's inside the image"
   below). Every checkpoint (RF/XGBoost models, GNN weights, BERT
   fine-tuned weights) is created fresh the first time you run a training
   command, and is written to a directory *you* mount in from the host, not
   into the image.
3. **The full pipeline is runnable end-to-end on synthetic data**, including
   the GNN (the paper's headline method) and its explainability step — not
   just the RF/XGBoost baselines.

## Prerequisites

- **Docker Desktop** (Mac/Windows) or **Docker Engine** (Linux) —
  either works; you just need `docker build` and `docker run` on your
  `PATH`. If `docker --version` prints something in a terminal, you're set.
- **~5-6 GB of free disk space** for the built image (`torch` +
  `torch_geometric` + `transformers` + `xgboost` + their dependencies
  account for most of this).
- **No GPU required.** Everything in this package runs on CPU; it'll just
  be slower for the GNN and especially the BERT models. If you do have an
  NVIDIA GPU and the [NVIDIA Container
  Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
  installed, add `--gpus all` to any `docker run` command below and PyTorch
  will use it automatically — no image changes needed.
- **Internet access inside the container is only needed for the BERT
  models** (`bert-train`), which download pretrained weights from Hugging
  Face Hub on first use. Everything else (baselines, GNN, explainability,
  risk stratification) runs fully offline once the image is built.

## Getting going

```bash
# 1. Clone the repository (skip if you already have it)
git clone https://github.com/ionaei/gnn-for-dementia
cd gnn-for-dementia

# 2. Build the image (one-time; ~5-10 minutes depending on your connection)
docker build -t dementia-repro .

# 3. Sanity-check the image itself -- self-contained, no mounts needed,
#    generates its own synthetic data, trains RF/XGBoost, explains, asserts
#    on outputs. If this passes, the image is good.
docker run --rm dementia-repro baselines-test

# 4. Make host folders for real input data and output results, then run the
#    headline method (the GNN) on the synthetic data that ships in the
#    image -- no --data-path needed, it's used automatically:
mkdir -p data output
docker run --rm -v "$(pwd)/output":/output dementia-repro gnn-train --epochs 5

# 5. Check the results landed on your host, not inside the container:
cat output/gnn/checkpoints_gnn/best_local.pt >/dev/null && echo "checkpoint written"
```

That's the whole loop: build once, then `docker run --rm -v "$(pwd)/output":/output dementia-repro <command>` for whichever module you want. Once you have a UK Biobank extract matching `data_prep/SCHEMA.md`, put it in the `data` folder you made and add `-v "$(pwd)/data":/data` plus `--data-path /data/your_file.csv` to point any command at it instead of the synthetic default:

```bash
docker run --rm \
    -v "$(pwd)/data":/data \
    -v "$(pwd)/output":/output \
    dementia-repro gnn-train --data-path /data/your_file.csv --epochs 5
```

### Running the full pipeline (GNN path, the paper's best method)

```bash
mkdir -p data output

# Train the GNN (omit --data-path to use the built-in synthetic data)
docker run --rm -v "$(pwd)/output":/output dementia-repro gnn-train --epochs 5

# Evaluate the checkpoint just produced (hidden/emb-dim/pool/head are read
# back automatically from the checkpoint -- nothing to repeat here)
docker run --rm -v "$(pwd)/output":/output dementia-repro gnn-evaluate

# Explain that checkpoint's predictions against real held-out patients
# (exports test-split graphs, then runs gradient-based explainability)
docker run --rm -v "$(pwd)/output":/output dementia-repro explain-run

# Youden's-J / traffic-light risk stratification on top of the same checkpoint
docker run --rm -v "$(pwd)/output":/output dementia-repro risk-stratify
```

Each of these reuses the previous command's output automatically (they all
share the same mounted `/output` volume), and each one auto-generates the
same synthetic CSV under `output/autogen_synthetic_data.csv` the first time
it's needed so the whole chain stays self-consistent, even without
`--data-path`.

### Running the RF/XGBoost baselines + SHAP

```bash
docker run --rm -v "$(pwd)/output":/output dementia-repro baselines-train --model both
docker run --rm -v "$(pwd)/output":/output dementia-repro baselines-explain --model both
```

### Running a BERT-family model (needs internet access)

```bash
# MODEL is one of: multimodal | text-only | bioclinical | roberta
docker run --rm -v "$(pwd)/output":/output dementia-repro bert-train text-only --epochs 1

# With a GPU:
docker run --rm --gpus all -v "$(pwd)/output":/output dementia-repro bert-train multimodal --epochs 1
```

### Dropping into a shell / running something not on the command list

```bash
docker run --rm -it -v "$(pwd)/output":/output dementia-repro shell
# or, passthrough for a one-off script:
docker run --rm dementia-repro python explainability/test_explainers.py
```

## Command reference

Run `docker run dementia-repro` (or `... dementia-repro help`) to print the
full command list with flags — it's kept in `entrypoint.sh` itself so it
can't drift out of sync with the actual code. In short:

| Command | What it does | Reads | Writes |
|---|---|---|---|
| `baselines-train` | RF/XGBoost: split → train → test → predict | `--data-path` CSV (or synthetic fallback) | `$OUTPUT_DIR/baselines/checkpoints/` |
| `baselines-explain` | SHAP explanations | checkpoints above | `$OUTPUT_DIR/baselines/explainability_outputs/` |
| `baselines-test` | Self-contained image smoke test | (generates its own data) | inside the container only |
| `gnn-train` | Train the star-graph GINEConv GNN | `--data-path` CSV (or synthetic fallback) | `$OUTPUT_DIR/gnn/checkpoints_gnn/` |
| `gnn-evaluate` | Evaluate a GNN checkpoint on the held-out test split | checkpoint above | stdout metrics |
| `risk-stratify` | Youden's-J traffic-light banding | GNN checkpoint | `$OUTPUT_DIR/risk_stratification/stratification_report.json` |
| `explain-test` | GNN explainability smoke test (synthetic in-memory graphs) | (generates its own data) | inside the container only |
| `explain-run` | Gradient / guided-backprop explanations on REAL held-out patients | GNN checkpoint + `--data-path` (or fallback) | `$OUTPUT_DIR/explainability/{test_graphs.pkl,results.json}` |
| `bert-train MODEL` | Fine-tune a BERT-family model (`multimodal`\|`text-only`\|`bioclinical`\|`roberta`) | `--data-path` CSV (optional; has its own internal synthetic fallback) | `$OUTPUT_DIR/bert_models/MODEL/` |
| `generate-data` | Write a synthetic CSV | — | `$OUTPUT_DIR/synthetic_data.csv` |
| `shell` | Interactive bash in the image | — | — |

Any command not in this list is executed as-is (passthrough), e.g.
`docker run dementia-repro python baselines/test_baselines.py`.

## What's inside the image

- All code in this repository, minus what `.dockerignore` excludes (see
  below).
- `data_prep/five_updated_synthetic.csv`, the same small (~290KB), fully
  synthetic, git-committed dataset the top-level README's non-Docker
  Quickstart uses — included so the Docker and non-Docker workflows have
  out-of-the-box parity. It is **not** real UK Biobank data and results on
  it are scientifically meaningless (see the top-level README's "elephant
  in the room" section) — it's there purely so every command works
  immediately after `docker build`, with no data of your own required.
- **Not** included, per the no-baked-in-weights requirement: any existing
  trained checkpoint (`baselines/checkpoints/`, `gnn/checkpoints_gnn/`),
  even though a couple of those are git-committed to this repo for
  documentation/reference purposes — see `.dockerignore` for the exact
  exclusion list. Every checkpoint you get from this image, you trained
  yourself, inside the container, on this run.
- **Not** included: any real patient data. UK Biobank data is
  access-controlled and cannot be redistributed — see the top-level README.


## Design notes

- `DATA_DIR` (default `/data`) and `OUTPUT_DIR` (default `/output`) are the
  two mount points; `APP_ROOT` (default `/app`) is the third environment
  variable the entrypoint reads, but you shouldn't need to touch it — it
  exists so the script can be dry-run tested outside a container (see
  above), not for normal use.
- If you omit `--data-path`, most commands auto-generate a small synthetic
  CSV under `$OUTPUT_DIR/autogen_synthetic_data.csv` on first use and reuse
  it on subsequent commands against the same output directory, so a
  `gnn-train` → `gnn-evaluate` → `explain-run` → `risk-stratify` chain
  without `--data-path` stays self-consistent. This is only useful for
  confirming the pipeline runs; results on it are meaningless (see
  `data_prep/README.md`).
- `bert-train` is the one exception: those four scripts already generate
  their own synthetic fallback internally, so the entrypoint doesn't
  duplicate that logic for them.
- `explain-run` is new in this version of the container (the previous
  Docker deliverable only exposed `explain-test`, the synthetic smoke
  test). It chains `tools/export_test_graphs.py` — which reuses the same
  data-loading/splitting/graph-building path as `gnn-train`, so the
  exported graphs' code vocabulary matches the checkpoint's trained
  embeddings — into `explainability/explain.py`.
- GNN and BERT training runs are reproducible for a given `--seed` (default
  `42`): both now call a shared `set_seed()` helper before model
  construction, seeding Python/NumPy/PyTorch RNGs, not just the
  train/val/test split. See the top-level README's "Reproducibility"
  section. This determinism is CPU-only best-effort on GPU (`--gpus all`).
