"""
End-to-end CLI: load a trained GNN checkpoint, evaluate it on the test set,
and run the traffic-light risk stratification analysis.

This wires together `gnn/evaluate_best_run.py` (model loading + evaluation)
and `traffic_light_stratification.py` (the Youden's J / confidence-band
statistics) into one runnable script, since the original
`traffic_light.ipynb` inlined both concerns together. Run this from within
the `risk_stratification/` directory so the relative import of `gnn/`
resolves; the sibling scripts add `../gnn` to `sys.path`.

Usage:
    # Using a local (e.g. --no-wandb smoke-test) checkpoint from gnn/train.py:
    python run_stratification.py --data-path ../data_prep/five_updated_synthetic.csv \\
        --local-checkpoint ../gnn/checkpoints_gnn/best_local.pt --hidden 32 --emb-dim 32

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
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--pool", type=str, default="add", choices=["mean", "add", "max"],
                         help="The original traffic_light.ipynb hardcoded pool='add' to match its "
                              "specific best run -- override if your run used a different pooling.")
    parser.add_argument("--head", type=str, default="linear", choices=["linear", "mlp"])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-w", type=float, default=0.30)
    parser.add_argument("--step", type=float, default=0.002)
    parser.add_argument("--min-coverage", type=float, default=0.50)
    parser.add_argument("--out", type=str, default="stratification_report.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    df_train, df_val, df_test = load_and_split(args.data_path)
    codes, code2idx, desc_to_time, code_texts = build_code_vocab(df_train)
    test_graphs = build_graphs(df_test, codes, code2idx, desc_to_time)

    if args.local_checkpoint:
        ckpt_path = args.local_checkpoint
        cfg = dict(hidden=args.hidden, dropout=args.dropout, train_eps=False, pool=args.pool, trainable=True)
        emb_dim = args.emb_dim
    else:
        import wandb

        api = wandb.Api()
        runs = api.runs(args.wandb_project)
        best_run = next((r for r in runs if r.id == args.run_id), None) if args.run_id else None
        if best_run is None:
            best_run = min(runs, key=lambda r: r.summary.get("best_val_loss", float("inf")))
        print("Best run:", best_run.id, "val_loss=", best_run.summary.get("best_val_loss"))
        cfg = dict(best_run.config)
        cfg.setdefault("pool", args.pool)
        emb_dim = cfg.get("EMB_DIM", cfg.get("emb_dim", 128))
        ckpt_path = f"{args.checkpoint_dir}/best_{best_run.id}.pt"

    code_emb_matrix = make_random_code_embeddings(len(codes), emb_dim)
    model = build_model_from_cfg(cfg, code_emb_matrix, edge_dim=3, out_classes=2, head=args.head).to(device)
    # Match the original notebook, which force-overrides pooling after
    # construction to be certain it matches the specific run being evaluated.
    from model import get_pool

    model.pool = get_pool(cfg.get("pool", args.pool))

    state = torch.load(ckpt_path, map_location=device)
    if "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=False)

    test_loader = DataLoader(test_graphs, batch_size=args.batch_size)
    metrics, probs_control, labels = evaluate(model, test_loader, device)
    print("Test metrics:", metrics)

    report = stratify(labels, probs_control, max_w=args.max_w, step=args.step, min_coverage=args.min_coverage)
    report["test_metrics"] = metrics

    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"\nWrote stratification report to {args.out}")


if __name__ == "__main__":
    main()
