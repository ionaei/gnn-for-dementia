#!/usr/bin/env bash
# Entrypoint for the dementia-risk reproducibility container.
#
# I/O convention (see DOCKER.md for full detail + examples):
#   - Real data goes in via a mounted volume, read with --data-path, e.g.:
#       docker run -v /host/data:/data -v /host/output:/output <image> \
#           baselines-train --data-path /data/five_updated_synthetic.csv
#   - Nothing is baked into the image at build time: no data, no trained
#     checkpoints. Every subcommand below writes fresh checkpoints/metrics
#     under $OUTPUT_DIR (default /output), which you mount from the host.
#   - If you omit --data-path, most subcommands auto-generate a small
#     synthetic dataset (same schema, fabricated values -- see
#     data_prep/generate_synthetic_data.py) so the image is runnable out of
#     the box as a smoke test, but results on it are meaningless.
#
# Run `docker run <image> help` (or no args) to see this same summary.

set -euo pipefail

DATA_DIR="${DATA_DIR:-/data}"
OUTPUT_DIR="${OUTPUT_DIR:-/output}"

cmd="${1:-help}"
shift || true

print_help() {
    cat <<'EOF'
Usage: docker run [-v host/data:/data] [-v host/output:/output] <image> <command> [args...]

Commands (RF/XGBoost baselines are the fully-verified train -> test ->
predict path; everything else is included and runnable but was authored
without access to a Docker daemon/GPU to test against -- see DOCKER.md):

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

  gnn-train         [--data-path PATH] [any gnn/train.py flag]
                     Trains the star-graph GINEConv GNN with --no-wandb by
                     default. Writes checkpoints to $OUTPUT_DIR/gnn/checkpoints_gnn.

  gnn-evaluate      [--data-path PATH] [any gnn/evaluate_best_run.py flag]
                     Evaluates a checkpoint produced by gnn-train (auto-uses
                     $OUTPUT_DIR/gnn/checkpoints_gnn/best_local.pt unless
                     you pass --local-checkpoint yourself).

  risk-stratify     [--data-path PATH] [any run_stratification.py flag]
                     Youden's-J traffic-light stratification on top of a
                     gnn-train checkpoint. Writes stratification_report.json
                     to $OUTPUT_DIR/risk_stratification/.

  explain-test       Self-contained GNN explainability smoke test
                     (gradient / guided-backprop explainers).

  bert-train MODEL  [--data-path PATH] [any classifier flag]
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
# `--data-path "$(default_data_path "$@")"` in only when needed. Real usage
# should always pass --data-path pointing into $DATA_DIR instead.
default_data_path() {
    local auto_csv="$OUTPUT_DIR/autogen_synthetic_data.csv"
    if [ ! -f "$auto_csv" ]; then
        echo "[entrypoint] No --data-path given; generating a synthetic CSV at $auto_csv (results on this data are meaningless -- see data_prep/README.md)." >&2
        (cd /app/data_prep && python generate_synthetic_data.py --n-patients 500 --out "$auto_csv" --seed 42) >&2
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
        cd /app/data_prep
        exec python generate_synthetic_data.py --out "$OUTPUT_DIR/synthetic_data.csv" "$@"
        ;;

    baselines-train)
        mkdir -p "$OUTPUT_DIR/baselines"
        cd /app/baselines
        run_with_data_fallback python train_rf_xgb.py --checkpoint-dir "$OUTPUT_DIR/baselines/checkpoints" "$@"
        ;;

    baselines-explain)
        mkdir -p "$OUTPUT_DIR/baselines"
        cd /app/baselines
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
        cd /app/baselines
        python train_rf_xgb.py
        python explain_shap.py
        exec python test_baselines.py
        ;;

    gnn-train)
        mkdir -p "$OUTPUT_DIR/gnn"
        cd /app/gnn
        run_with_data_fallback python train.py --checkpoint-dir "$OUTPUT_DIR/gnn/checkpoints_gnn" --no-wandb "$@"
        ;;

    gnn-evaluate)
        mkdir -p "$OUTPUT_DIR/gnn"
        cd /app/gnn
        args=(python evaluate_best_run.py --checkpoint-dir "$OUTPUT_DIR/gnn/checkpoints_gnn")
        if ! printf '%s\n' "$@" | grep -qx -- '--local-checkpoint'; then
            args+=(--local-checkpoint "$OUTPUT_DIR/gnn/checkpoints_gnn/best_local.pt")
        fi
        run_with_data_fallback "${args[@]}" "$@"
        ;;

    risk-stratify)
        mkdir -p "$OUTPUT_DIR/risk_stratification"
        cd /app/risk_stratification
        args=(python run_stratification.py --out "$OUTPUT_DIR/risk_stratification/stratification_report.json")
        if ! printf '%s\n' "$@" | grep -qx -- '--local-checkpoint'; then
            args+=(--local-checkpoint "$OUTPUT_DIR/gnn/checkpoints_gnn/best_local.pt")
        fi
        run_with_data_fallback "${args[@]}" "$@"
        ;;

    explain-test)
        cd /app/explainability
        exec python test_explainers.py
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
        cd /app/bert_models
        echo "[entrypoint] bert-train downloads pretrained weights from HuggingFace at runtime -- this container needs network access, and this path was not verified end-to-end (no Docker daemon / GPU / HF network access in the environment this image was authored in). See DOCKER.md." >&2
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
        # Passthrough: run whatever was asked for verbatim from /app.
        cd /app
        exec "$cmd" "$@"
        ;;
esac
