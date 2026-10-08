"""
End-to-end CLI: load a trained GNN checkpoint, evaluate it on both the
validation and test sets, and run the traffic-light risk stratification
analysis with the Youden's J threshold t* and confidence-band half-width w
tuned on validation only and reported on test only (see
`traffic_light_stratification.stratify`'s docstring for why -- tuning and
reporting on the same set leaks test-set information into the headline
numbers).

This wires together `gnn/evaluate_best_run.py` (model loading + evaluation)
and `traffic_light_stratification.py` (the Youden's J / confidence-band
statistics) into one runnable script. Run this from within the
`risk_stratification/` directory so the relative import of `gnn/` resolves;
the sibling scripts add `../gnn` to `sys.path`.

Usage:
    # Using a local (e.g. --no-wandb smoke-test) checkpoint from gnn/train.py.
    # Checkpoints saved by the current train.py embed their own hidden/emb-dim/
    # pool/head config, so you normally don't need to pass those explicitly:
    python run_stratification.py --data-path ../data_prep/five_updated_synthetic.csv \\
        --local-checkpoint ../gnn/checkpoints_gnn/best_local.pt

    # Older checkpoint with no embedded config (pre-dates this fix) -- you must
    # supply the training-time hyperparameters yourself, --pool especially:
    python run_stratification.py --data-path ../data_prep/five_updated_synthetic.csv \\
        --local-checkpoint ../gnn/checkpoints_gnn/best_local.pt \\
        --hidden 32 --emb-dim 32 --pool mean

    # Using the best run from a WandB sweep:
    python run_stratification.py --data-path /path/to/five_updated.csv \\
        --wandb-project NEURIPS_UPDATED_AD --checkpoint-dir ../gnn/checkpoints_gnn
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "gnn"))

import torch  # noqa: E402
from torch_geometric.loader import DataLoader  # noqa: E402

from checkpoint_utils import load_checkpoint  # noqa: E402
from evaluate_best_run import evaluate  # noqa: E402
from graph_construction import build_code_vocab, build_graphs, make_random_code_embeddings  # noqa: E402
from model import build_model_from_cfg  # noqa: E402
from train import load_and_split  # noqa: E402

from traffic_light_stratification import stratify  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=str, default="../data_prep/five_updated_synthetic.csv")
    parser.add_argument("--checkpoint-dir", type=str, default="../gnn/checkpoints_gnn")
    parser.add_argument("--wandb-project", type=str, default="NEURIPS_UPDATED_AD")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--local-checkpoint", type=str, default=None,
                         help="Path to a specific checkpoint .pt file (skips WandB entirely).")
    parser.add_argument("--hidden", type=int, default=None,
                         help="Only used with --local-checkpoint if the checkpoint doesn't carry its own "
                              "embedded config (e.g. an older bare-state-dict checkpoint) -- checkpoints "
                              "saved by the current gnn/train.py record this automatically. Falls back to 128.")
    parser.add_argument("--emb-dim", type=int, default=None, help="See --hidden; falls back to 128.")
    parser.add_argument("--dropout", type=float, default=None, help="See --hidden; falls back to 0.1.")
    parser.add_argument("--pool", type=str, default=None, choices=["mean", "add", "max"],
                         help="Overrides the checkpoint's (or WandB run's) embedded pooling strategy. "
                              "Pooling has no learnable weights, so for a checkpoint saved without "
                              "embedded config this CANNOT be inferred -- pass this explicitly if you "
                              "know the run's config, otherwise it falls back to 'mean' with a warning. "
                              "NOTE: this script used to default --pool to 'add' here while gnn/train.py "
                              "and evaluate_best_run.py defaulted to 'mean' -- that mismatch is exactly the "
                              "silent-wrong-predictions bug this checkpoint-config-embedding fix closes. "
                              "If you're evaluating an old checkpoint trained before this fix and you know "
                              "it used pool='add', pass --pool add explicitly.")
    parser.add_argument("--head", type=str, default=None, choices=["linear", "mlp"],
                         help="See --hidden; falls back to 'linear'.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-w", type=float, default=0.30)
    parser.add_argument("--step", type=float, default=0.002)
    parser.add_argument("--min-coverage", type=float, default=0.50)
    parser.add_argument("--out", type=str, default="stratification_report.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine the checkpoint path (and, for the WandB path, that run's own
    # config as a fallback for legacy checkpoints) *before* loading/splitting
    # data, since the checkpoint's embedded `seed` -- not always 42 -- picks
    # which held-out val/test patients this script must use. This mirrors the
    # same fix already applied to gnn/evaluate_best_run.py.
    wandb_cfg = {}
    if args.local_checkpoint:
        ckpt_path = args.local_checkpoint
    else:
        import wandb

        api = wandb.Api()
        runs = api.runs(args.wandb_project)
        best_run = next((r for r in runs if r.id == args.run_id), None) if args.run_id else None
        if best_run is None:
            best_run = min(runs, key=lambda r: r.summary.get("best_val_loss", float("inf")))
        print("Best run:", best_run.id, "val_loss=", best_run.summary.get("best_val_loss"))
        wandb_cfg = dict(best_run.config)
        ckpt_path = f"{args.checkpoint_dir}/best_{best_run.id}.pt"

    state, ckpt_cfg = load_checkpoint(
        ckpt_path, device=device,
        overrides=dict(hidden=args.hidden, emb_dim=args.emb_dim, dropout=args.dropout,
                       pool=args.pool, head=args.head),
    )

    # The checkpoint's own embedded config (already merged with any CLI
    # overrides by load_checkpoint() above) is the primary source of truth;
    # the WandB run's config is only a fallback for legacy checkpoints saved
    # without one; hardcoded literals are the last resort. Same precedence
    # as gnn/evaluate_best_run.py.
    cfg = dict(
        hidden=ckpt_cfg.get("hidden", wandb_cfg.get("hidden", 128)),
        dropout=ckpt_cfg.get("dropout", wandb_cfg.get("dropout", 0.1)),
        train_eps=ckpt_cfg.get("train_eps", wandb_cfg.get("train_eps", False)),
        pool=ckpt_cfg.get("pool", wandb_cfg.get("pool", "mean")),
        trainable=ckpt_cfg.get("trainable", wandb_cfg.get("trainable", True)),
    )
    if "pool" not in ckpt_cfg and "pool" not in wandb_cfg and args.pool is None:
        import warnings

        warnings.warn(
            "Neither the checkpoint nor the WandB run config records 'pool' and --pool was not "
            "given; defaulting to 'mean'. Pass --pool explicitly if you know the run's config. "
            "NOTE: this script used to default --pool to 'add' here while gnn/train.py and "
            "evaluate_best_run.py defaulted to 'mean' -- that mismatch is exactly the "
            "silent-wrong-predictions bug this checkpoint-config-embedding fix closes."
        )
    emb_dim = ckpt_cfg.get("emb_dim", wandb_cfg.get("EMB_DIM", wandb_cfg.get("emb_dim", 128)))
    head = ckpt_cfg.get("head", args.head or "linear")
    seed = ckpt_cfg.get("seed", 42)

    # Tuning (t*, w) needs a validation set distinct from the test set that
    # gets scored at those fixed values -- see traffic_light_stratification.
    # stratify()'s docstring for why. Build graphs/loaders for both splits.
    df_train, df_val, df_test = load_and_split(args.data_path, seed=seed)
    codes, code2idx, desc_to_time, code_texts = build_code_vocab(df_train)
    val_graphs = build_graphs(df_val, codes, code2idx, desc_to_time)
    test_graphs = build_graphs(df_test, codes, code2idx, desc_to_time)

    code_emb_matrix = make_random_code_embeddings(len(codes), emb_dim)
    model = build_model_from_cfg(cfg, code_emb_matrix, edge_dim=3, out_classes=2, head=head).to(device)

    # strict=True: the architecture above is rebuilt entirely from the
    # checkpoint's own config, so a key/shape mismatch here means something
    # is genuinely wrong and should raise loudly rather than silently loading
    # a partial/mismatched model (same reasoning as evaluate_best_run.py).
    model.load_state_dict(state, strict=True)

    val_loader = DataLoader(val_graphs, batch_size=args.batch_size)
    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)
    val_metrics, val_probs_control, val_labels = evaluate(model, val_loader, device)
    test_metrics, test_probs_control, test_labels = evaluate(model, test_loader, device)
    print("Val metrics:", val_metrics)
    print("Test metrics:", test_metrics)

    report = stratify(val_labels, val_probs_control, test_labels, test_probs_control,
                       max_w=args.max_w, step=args.step, min_coverage=args.min_coverage)
    report["val_metrics"] = val_metrics
    report["test_metrics"] = test_metrics

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"\nWrote stratification report to {args.out}")


if __name__ == "__main__":
    main()
