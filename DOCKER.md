# Running this package in Docker

This container exists so a reviewer can run the reproducibility pipeline
without setting up a Python environment by hand. It was built to three
requirements agreed for this submission:

1. **Everything in one image** — baselines, GNN, BERT models, explainability,
   and risk stratification are all runnable from the same image/entrypoint.
2. **No pre-trained model weights baked in** — the image ships code only.
   Every checkpoint (RF/XGBoost models, GNN weights, BERT fine-tuned weights)
   is created fresh the first time you run a training command, and is
   written to a directory *you* mount in from the host, not into the image.
3. **Random Forest / XGBoost is the fully-verified path**: given a data
   split, the container trains, evaluates on a held-out test split, and
   produces predictions, end to end.

## What was actually verified, and how

There is no Docker daemon available in the environment this container was
authored in, so `docker build` / `docker run` themselves were never
executed here. Instead, the exact filesystem layout Docker would produce was
reproduced by hand — a clean checkout of this repository with every
`.dockerignore`-excluded path removed (checkpoints, generated outputs,
`__pycache__`, `.git`) — and `entrypoint.sh`'s logic was run against it
directly with `DATA_DIR`/`OUTPUT_DIR` pointed at separate mock "mounted
volume" directories, exactly as `docker run -v host/data:/data -v
host/output:/output` would set them up. Concretely, this confirmed:

- `baselines-train` (RF and XGBoost): loads a CSV from the "mounted"
  data directory, performs the 70/10/20 stratified split, runs the Bayesian
  hyperparameter search, fits, evaluates (predicts) on the held-out test
  split, and writes models/scaler/metrics into the "mounted" output
  directory only — nothing lands inside the (simulated) image directory.
- `baselines-explain` correctly reads back those checkpoints and writes SHAP
  outputs to the output directory.
- `baselines-test` (the image's own self-contained smoke test) passes.
- `gnn-train` → `gnn-evaluate` → `risk-stratify` chain correctly, each
  reading the previous command's output from the mounted output directory,
  including the auto-generated-synthetic-data fallback when `--data-path`
  is omitted.
- `explain-test` (GNN explainability smoke test) passes.
- `bert-train` was run far enough to confirm the synthetic-data fallback
  now works (see "Bugs found and fixed" below) — it then fails to download
  pretrained weights from Hugging Face Hub, but that failure is a network
  restriction of the sandbox this container was authored in
  (`ProxyError: ... 403 Forbidden` contacting `huggingface.co`), not a bug
  in the code. **You should verify `bert-train` yourself** once you have a
  real Docker daemon with internet access — see "Please verify" below.

What was *not* done: an actual `docker build` (so the Dockerfile itself —
package installation, layer caching, base image compatibility — is
untested), and anything requiring a GPU or Hugging Face Hub access.

## Bugs found and fixed while preparing this container

Two real, pre-existing bugs surfaced while testing the exact code paths this
container exercises. Both are fixed in this codebase (not worked around in
`entrypoint.sh`), so they're fixed for anyone using these scripts directly
too, not just inside Docker:

1. **`baselines/train_rf_xgb.py` had no way to redirect its output
   directory.** It hardcoded `checkpoint_dir = Path("checkpoints")` relative
   to the current working directory, unlike `explain_shap.py` and every
   other training script in this package, which all expose a
   `--checkpoint-dir` flag. Fixed by adding the same `--checkpoint-dir` flag
   (default unchanged: `"checkpoints"`), so a Docker entrypoint — or anyone
   else — can point it at a mounted output volume without editing the file.

2. **All four `bert_models/*_classifier.py` scripts crashed when run without
   `--data-path`.** Their synthetic-data fallback called
   `from generate_synthetic_data import generate_synthetic_data` and then
   `generate_synthetic_data(path, n_patients=60)` — but no function with
   that name exists anywhere in this repository (`data_prep/` only exposes
   `generate_synthetic_cohort(n_patients, rng) -> DataFrame`), and the
   import would have failed regardless since `data_prep/` was never added to
   `sys.path`. This looks like it was never actually exercised end-to-end
   before (consistent with `bert_models/README.md`'s note that this code was
   reconstructed from notebooks, not recovered verbatim). Fixed in all four
   files to call the real `generate_synthetic_cohort` API with a proper
   `sys.path` insert, and confirmed by hand that it now produces a valid
   60-patient synthetic CSV instead of raising `ImportError`.

## Please verify yourself

Since no Docker daemon or GPU was available while preparing this container,
please run at least the following once you have Docker (this is also the
minimum bar the container was designed to meet — "given a data split, train,
test, and predict" for RF/XGBoost):

```bash
docker build -t dementia-repro .

# Sanity-check the image itself (self-contained, no mounts needed):
docker run --rm dementia-repro baselines-test

# The real path: your own data split in, predictions + metrics out.
docker run --rm \
    -v /path/to/your/data:/data \
    -v /path/to/your/output:/output \
    dementia-repro baselines-train --data-path /data/your_file.csv --model both

# Confirm nothing was baked into the image and everything landed in /output:
ls /path/to/your/output/baselines/checkpoints
cat /path/to/your/output/baselines/checkpoints/rf_metrics.json
```

If you have GPU + internet access and want to check the BERT path:

```bash
docker run --rm --gpus all \
    -v /path/to/your/data:/data -v /path/to/your/output:/output \
    dementia-repro bert-train text-only --data-path /data/your_file.csv --epochs 1
```

## Command reference

Run `docker run <image>` (or `docker run <image> help`) to print the full
command list with flags — it's kept in `entrypoint.sh` itself so it can't
drift out of sync with the actual code. In short:

| Command | What it does | Reads | Writes |
|---|---|---|---|
| `baselines-train` | RF/XGBoost: split → train → test → predict | `--data-path` CSV | `$OUTPUT_DIR/baselines/checkpoints/` |
| `baselines-explain` | SHAP explanations | checkpoints above | `$OUTPUT_DIR/baselines/explainability_outputs/` |
| `baselines-test` | Self-contained image smoke test | (generates its own data) | inside the container only |
| `gnn-train` | Train the star-graph GINEConv GNN | `--data-path` CSV | `$OUTPUT_DIR/gnn/checkpoints_gnn/` |
| `gnn-evaluate` | Evaluate a GNN checkpoint | checkpoint above | stdout metrics |
| `risk-stratify` | Youden's-J traffic-light banding | GNN checkpoint | `$OUTPUT_DIR/risk_stratification/stratification_report.json` |
| `explain-test` | GNN explainability smoke test | (generates its own data) | inside the container only |
| `bert-train MODEL` | Fine-tune a BERT-family model | `--data-path` CSV (optional) | `$OUTPUT_DIR/bert_models/MODEL/` |
| `generate-data` | Write a synthetic CSV | — | `$OUTPUT_DIR/synthetic_data.csv` |
| `shell` | Interactive bash in the image | — | — |

Any command not in this list is executed as-is (passthrough), e.g.
`docker run <image> python baselines/test_baselines.py`.

## Design notes

- `DATA_DIR` (default `/data`) and `OUTPUT_DIR` (default `/output`) are the
  two mount points. Nothing else needs mounting.
- If you omit `--data-path`, most commands auto-generate a small synthetic
  CSV under `$OUTPUT_DIR/autogen_synthetic_data.csv` on first use and reuse
  it on subsequent commands in the same output directory, so a `gnn-train`
  → `gnn-evaluate` → `risk-stratify` chain without `--data-path` stays
  self-consistent. This is only useful for confirming the pipeline runs;
  results on it are meaningless (see `data_prep/README.md`).
- `bert-train` is the one exception: those four scripts already generate
  their own synthetic fallback internally (now fixed — see above), so the
  entrypoint doesn't duplicate that logic for them.
- No real data, and no trained checkpoints, are copied into the image at
  build time — see `.dockerignore`.
