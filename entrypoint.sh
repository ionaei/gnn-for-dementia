#!/usr/bin/env bash
# Entrypoint for the dementia-risk reproducibility container.
#
# I/O convention (see DOCKER.md for full detail + examples):
#   - Real data goes in via a mounted volume, read with --data-path, e.g.:
#       docker run -v /host/data:/data -v /host/output:/output <image> \
#           gnn-train --data-path /data/five_updated.csv
#   - No trained checkpoints are baked into the image (see .dockerignore).
#     Every subcommand below writes fresh checkpoints/metrics under
#     $OUTPUT_DIR (default /output), which you mount from the host.
#   - The image DOES ship the repo's own pre-generated synthetic CSV
#     (data_prep/five_updated_synthetic.csv, 2000 fake-but-schema-matching
#     patients -- same file the top-level README's non-Docker Quickstart
#     uses), so every command below is runnable out of the box with no
#     mounts at all, as a smoke test. If you omit --data-path entirely,
#     most subcommands additionally auto-generate their own synthetic CSV
#     under $OUTPUT_DIR the first time they're needed. Results on any
#     synthetic data are meaningless either way -- see data_prep/README.md.
#
# Run `docker run <image>` (or `docker run <image> help`) to see this same
# summary.
#
# APP_ROOT defaults to /app (where the Dockerfile COPYs the repo). It's
# overridable so this exact script can be dry-run against a plain checkout
# without a container at all, e.g.:
#   APP_ROOT=$(pwd) DATA_DIR=/tmp/data OUTPUT_DIR=/tmp/output ./entrypoint.sh gnn-train
# This is how this script's command logic was actually verified while
# preparing this image (see DOCKER.md's "What was verified" section) --
# there is no Docker daemon in the environment this container was authored
# in, so `docker build`/`docker run` themselves were never executed here.

set -euo pipefail

APP_ROOT="${APP_ROOT:-/app}"
DATA_DIR="${DATA_DIR:-/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"

cmd="${1:-help}"
shift || true

print_help() {
    cat <<'EOF'
Usage: docker run [-v host/data:/data] [-v host/output:/output] <image> <command> [args...]

Commands:

  baselines-train   [--data-path PATH] [--model rf|xgb|both] [--seed N]
                     Runs the real pipeline: load CSV -> stratified 70/10/20
                     train/val/test split -> Bayesian hyperparameter search
                     -> fit -> evaluate (predict) on the held-out test split.
                     Writes models/scaler/metrics to $OUTPUT_DIR/baselines/checkpoints.

  baselines-explain [--data-path PATH] [--model rf|xgb|both]
                     SHAP explanations for a model already trained with
                     baselines-train. Reads checkpoints from, and writes
                     plots/CSVs to, $OUTPUT_DIR/baselines/.

  baselines-test     Self-contained smoke test (ignores mounted volumes):
                     generates synthetic data, trains, explains, and asserts
                     all outputs are well-formed, entirely inside the image's
                     own filesystem. Use this to sanity-check the image
                     itself right after `docker build`.

  gnn-train         [--data-path PATH] [--seed N] [any gnn/train.py flag]
                     Trains the star-graph GINEConv GNN with --no-wandb by
                     default. Checkpoints embed their own hidden/emb-dim/
                     pool/head config (see gnn/checkpoint_utils.py) and are
                     fully reproducible for a given --seed (see
                     gnn/seed_utils.py) -- rerunning this with the same
                     --seed reproduces the same weights. Writes to
                     $OUTPUT_DIR/gnn/checkpoints_gnn.

  gnn-evaluate      [--data-path PATH] [any gnn/evaluate_best_run.py flag]
                     Evaluates a checkpoint produced by gnn-train (auto-uses
                     $OUTPUT_DIR/gnn/checkpoints_gnn/best_local.pt unless
                     you pass --local-checkpoint yourself). Reads
                     hidden/emb-dim/pool/head back out of the checkpoint
                     automatically -- you don't need to (and shouldn't have
                     to) repeat them here.

  risk-stratify     [--data-path PATH] [any run_stratification.py flag]
                     Youden's-J traffic-light stratification on top of a
                     gnn-train checkpoint. Writes stratification_report.json
                     to $OUTPUT_DIR/risk_stratification/.

  explain-test       Self-contained GNN explainability smoke test
                     (gradient / guided-backprop explainers on synthetic
                     in-memory graphs -- no real data or checkpoint needed).

  explain-run       [--data-path PATH] [any explainability/explain.py flag]
                     Explains a gnn-train checkpoint against REAL held-out
                     patients: exports the test split's graphs (tools/
                     export_test_graphs.py, using the same --data-path so
                     the code vocabulary matches the checkpoint), then runs
                     explain.py against them. Writes
                     $OUTPUT_DIR/explainability/{test_graphs.pkl,results.json}.
                     Pass --method guided_bp to switch explainer method.

  bert-train MODEL  [--data-path PATH] [--seed N] [any classifier flag]
                     MODEL is one of: multimodal, text-only, bioclinical,
                     roberta. Requires downloading pretrained weights from
                     HuggingFace at runtime (network access required inside
                     the container) and is far slower without a GPU. Writes
                     checkpoints to $OUTPUT_DIR/bert_models/<MODEL>.

  generate-data     [--n-patients N] [--seed N]
                     Writes a fresh synthetic CSV to
                     $OUTPUT_DIR/synthetic_data.csv for use as --data-path
                     in any of the above.

  shell              Drop into an interactive bash shell in the image.

  help               Show this message.

Any other command is executed as-is (passthrough), e.g.:
  docker run <image> python baselines/test_baselines.py
EOF
}

mkdir -p "$OUTPUT_DIR"

# Returns 0 (true) if "--data-path" is already among the forwarded args.
has_data_path_arg() {
    for a in "$@"; do
        [ "$a" = "--data-path" ] && return 0
    done
    return 1
}

# If the caller didn't pass --data-path, generate (once) a small synthetic
# CSV under $OUTPUT_DIR and echo its path, so callers can splice
# `--data-path "$(default_data_path)"` in only when needed. Real usage
# should always pass --data-path pointing into $DATA_DIR instead.
default_data_path() {
    local auto_csv="$OUTPUT_DIR/autogen_synthetic_data.csv"
    if [ ! -f "$auto_csv" ]; then
        echo "[entrypoint] No --data-path given; generating a synthetic CSV at $auto_csv (results on this data are meaningless -- see data_prep/README.md)." >&2
        (cd "$APP_ROOT/data_prep" && python generate_synthetic_data.py --n-patients 500 --out "$auto_csv" --seed 42) >&2
    fi
    echo "$auto_csv"
}

# Runs a training/eval script, injecting --data-path with an auto-generated
# synthetic CSV only if the caller didn't already supply one.
run_with_data_fallback() {
    if has_data_path_arg "$@"; then
        exec "$@"
    else
        local auto
        auto="$(default_data_path)"
        # Argparse doesn't care about flag order, so it's simplest (and
        # safest -- no risk of splitting "python" from its script name) to
        # just append --data-path at the very end of the whole command.
        exec "$@" --data-path "$auto"
    fi
}

case "$cmd" in
    help|-h|--help|"")
        print_help
        ;;

    generate-data)
        cd "$APP_ROOT/data_prep"
        exec python generate_synthetic_data.py --out "$OUTPUT_DIR/synthetic_data.csv" "$@"
        ;;

    baselines-train)
        mkdir -p "$OUTPUT_DIR/baselines"
        cd "$APP_ROOT/baselines"
        run_with_data_fallback python train_rf_xgb.py --checkpoint-dir "$OUTPUT_DIR/baselines/checkpoints" "$@"
        ;;

    baselines-explain)
        mkdir -p "$OUTPUT_DIR/baselines"
        cd "$APP_ROOT/baselines"
        run_with_data_fallback python explain_shap.py \
            --checkpoint-dir "$OUTPUT_DIR/baselines/checkpoints" \
            --output-dir "$OUTPUT_DIR/baselines/explainability_outputs" \
            "$@"
        ;;

    baselines-test)
        # Entirely self-contained: ignores mounted volumes on purpose, so
        # this always works as an image sanity check even with nothing
        # mounted in. Uses the scripts' own default (relative) checkpoint
        # dirs, matching what test_baselines.py itself expects to find.
        cd "$APP_ROOT/baselines"
        python train_rf_xgb.py
        python explain_shap.py
        exec python test_baselines.py
        ;;

    gnn-train)
        mkdir -p "$OUTPUT_DIR/gnn"
        cd "$APP_ROOT/gnn"
        run_with_data_fallback python train.py --checkpoint-dir "$OUTPUT_DIR/gnn/checkpoints_gnn" --no-wandb "$@"
        ;;

    gnn-evaluate)
        mkdir -p "$OUTPUT_DIR/gnn"
        cd "$APP_ROOT/gnn"
        args=(python evaluate_best_run.py --checkpoint-dir "$OUTPUT_DIR/gnn/checkpoints_gnn")
        if ! printf '%s\n' "$@" | grep -qx -- '--local-checkpoint'; then
            args+=(--local-checkpoint "$OUTPUT_DIR/gnn/checkpoints_gnn/best_local.pt")
        fi
        run_with_data_fallback "${args[@]}" "$@"
        ;;

    risk-stratify)
        mkdir -p "$OUTPUT_DIR/risk_stratification"
        cd "$APP_ROOT/risk_stratification"
        args=(python run_stratification.py --out "$OUTPUT_DIR/risk_stratification/stratification_report.json")
        if ! printf '%s\n' "$@" | grep -qx -- '--local-checkpoint'; then
            args+=(--local-checkpoint "$OUTPUT_DIR/gnn/checkpoints_gnn/best_local.pt")
        fi
        run_with_data_fallback "${args[@]}" "$@"
        ;;

    explain-test)
        cd "$APP_ROOT/explainability"
        exec python test_explainers.py
        ;;

    explain-run)
        # tools/export_test_graphs.py reuses the same load_and_split +
        # build_graphs path as gnn-train, so the exported graphs' code
        # vocabulary matches the checkpoint's trained code_emb weights --
        # this is only guaranteed if the SAME --data-path (and --seed) is
        # used here as was used for gnn-train.
        mkdir -p "$OUTPUT_DIR/explainability"
        graphs_pkl="$OUTPUT_DIR/explainability/test_graphs.pkl"
        ckpt="$OUTPUT_DIR/gnn/checkpoints_gnn/best_local.pt"

        # --data-path (if given) is consumed by the export step only; every
        # other flag is forwarded to explain.py itself (e.g. --method,
        # --top_k, --target_class -- or --model/--graphs/--output to
        # override the defaults set below, since argparse takes the last
        # occurrence of a flag).
        data_path=""
        explain_args=()
        while [ $# -gt 0 ]; do
            case "$1" in
                --data-path) data_path="$2"; shift 2 ;;
                *) explain_args+=("$1"); shift ;;
            esac
        done
        [ -n "$data_path" ] || data_path="$(default_data_path)"

        cd "$APP_ROOT/tools"
        python export_test_graphs.py --data-path "$data_path" --out "$graphs_pkl"

        cd "$APP_ROOT/explainability"
        exec python explain.py \
            --model "$ckpt" \
            --graphs "$graphs_pkl" \
            --output "$OUTPUT_DIR/explainability/results.json" \
            "${explain_args[@]}"
        ;;

    bert-train)
        model="${1:-}"
        shift || true
        case "$model" in
            multimodal)  script="multimodal_classifier.py" ;;
            text-only)   script="text_only_classifier.py" ;;
            bioclinical) script="bioclinical_no_temporal.py" ;;
            roberta)     script="roberta_classifier.py" ;;
            "")
                echo "[entrypoint] bert-train requires a MODEL: multimodal | text-only | bioclinical | roberta" >&2
                exit 1
                ;;
            *)
                echo "[entrypoint] Unknown bert-train MODEL '$model'. Expected: multimodal | text-only | bioclinical | roberta" >&2
                exit 1
                ;;
        esac
        mkdir -p "$OUTPUT_DIR/bert_models/$model"
        cd "$APP_ROOT/bert_models"
        echo "[entrypoint] bert-train downloads pretrained weights from HuggingFace at runtime -- this container needs network access, and is far slower without a GPU. See DOCKER.md." >&2
        # NOTE: unlike the other subcommands, these scripts already default
        # to generating their own synthetic data internally when --data-path
        # is omitted (see the classifier scripts' train_and_evaluate()), so
        # no run_with_data_fallback wrapper is needed here.
        exec python "$script" --checkpoint-dir "$OUTPUT_DIR/bert_models/$model" "$@"
        ;;

    shell)
        exec /bin/bash
        ;;

    *)
        # Passthrough: run whatever was asked for verbatim from $APP_ROOT.
        cd "$APP_ROOT"
        exec "$cmd" "$@"
        ;;
esac
