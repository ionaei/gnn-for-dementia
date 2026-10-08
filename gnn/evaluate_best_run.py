"""
Load the best run from a WandB sweep and evaluate it on the held-out test
set.

This script locates the best run's checkpoint, rebuilds the corresponding
model via `model.build_model_from_cfg` (see `model.py`), and loads weights
from a checkpoint directory supplied as a CLI argument (rather than a
hardcoded path), so it works regardless of where `train.py` wrote its
checkpoints.

Unified metrics convention (common/metrics.py): Dementia (label 0) is the
positive class for F1/sensitivity/specificity; AUROC and AUPRC are computed
from probabilities with Dementia explicit as positive, never from hard 0/1
predictions. This replaces this script's previous local metrics, which used
macro-averaged F1 (not comparable to `train.py`'s or the baselines'/BERT's
Dementia-positive F1) and Control-positive AUPRC (`average_precision_score(
labels, probs, pos_label=1)`), and had no AUROC at all -- also why this
script's AUPRC never matched `train.py`'s AUPRC for the same checkpoint even
before the probabilities-vs-hard-predictions bug in `train.py` existed.

The checkpoint's own embedded `config` (see `checkpoint_utils.py`) is now
the primary source of truth for every architecture hyperparameter
(`hidden`, `dropout`, `train_eps`, `pool`, `trainable`, `emb_dim`, `head`)
*and* for `seed` -- which determines the train/val/test split via
`common/data_split.py` -- and `embedding_source`. A WandB run's own config
(`best_run.config`) is only consulted as a fallback for legacy checkpoints
saved without an embedded config; CLI overrides (`--hidden`, `--emb-dim`,
etc.) always win over both, same as `checkpoint_utils.load_checkpoint`'s
existing "CLI always wins" behavior.
"""

import argparse
import os
import sys
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from checkpoint_utils import load_checkpoint
from graph_construction import build_code_vocab, build_graphs, make_random_code_embeddings
from model import build_model_from_cfg
from train import load_and_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "common"))
from metrics import compute_metrics  # noqa: E402


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs_control = [], [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = torch.nn.functional.cross_entropy(logits, batch.y.view(-1))
        total_loss += loss.item() * batch.num_graphs

        probs = torch.softmax(logits, dim=1)[:, 1]  # P(control), i.e. class 1
        preds = logits.argmax(dim=1)

        correct += (preds == batch.y.view(-1)).sum().item()
        total += batch.num_graphs
        all_probs_control.extend(probs.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch.y.view(-1).cpu().tolist())

    acc = correct / max(total, 1)
    m = compute_metrics(all_labels, all_preds, all_probs_control)

    return dict(
        loss=total_loss / max(total, 1), accuracy=acc, f1=m["f1"],
        sensitivity=m["sensitivity"], specificity=m["specificity"],
        auprc=m["auprc"], auroc=m["auroc"],
    ), all_probs_control, all_labels


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=str, default="../data_prep/five_updated_synthetic.csv")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_gnn",
                         help="Directory containing best_<run_id>.pt files (must match --checkpoint-dir used in train.py)")
    parser.add_argument("--wandb-project", type=str, default="NEURIPS_UPDATED_AD")
    parser.add_argument("--run-id", type=str, default=None,
                         help="Specific WandB run id to evaluate. If omitted, queries the WandB API for the "
                              "run with the lowest best_val_loss across --wandb-project (requires `wandb login`).")
    parser.add_argument("--local-checkpoint", type=str, default=None,
                         help="Skip WandB entirely and just load this checkpoint file directly "
                              "(use this for the --no-wandb smoke-test checkpoints produced by train.py, "
                              "which are saved under run id 'local').")
    parser.add_argument("--hidden", type=int, default=None,
                         help="Only used with --local-checkpoint if the checkpoint doesn't carry its own "
                              "embedded config (e.g. an older bare-state-dict checkpoint) -- checkpoints "
                              "saved by the current train.py record this automatically. Falls back to 128.")
    parser.add_argument("--emb-dim", type=int, default=None, help="See --hidden; falls back to 128.")
    parser.add_argument("--dropout", type=float, default=None, help="See --hidden; falls back to 0.1.")
    parser.add_argument("--pool", type=str, default=None, choices=["mean", "add", "max"],
                         help="Overrides the checkpoint's embedded pooling strategy. Pooling has no "
                              "learnable weights, so for a checkpoint saved without embedded config this "
                              "CANNOT be inferred and matters a lot -- pass this explicitly if you know "
                              "the run's config, otherwise it falls back to 'mean' with a warning.")
    parser.add_argument("--head", type=str, default=None, choices=["linear", "mlp"],
                         help="See --hidden; falls back to 'linear'.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Determine the checkpoint path (and, for the WandB path, that run's own
    # config as a fallback for legacy checkpoints) *before* loading/splitting
    # data, since the checkpoint's embedded `seed` -- not always 42 -- picks
    # which held-out test patients this evaluation must use.
    wandb_cfg = {}
    if args.local_checkpoint:
        ckpt_path = args.local_checkpoint
    else:
        import wandb

        api = wandb.Api()
        runs = api.runs(args.wandb_project)
        if args.run_id:
            best_run = next(r for r in runs if r.id == args.run_id)
        else:
            best_run = min(runs, key=lambda r: r.summary.get("best_val_loss", float("inf")))
        print("Best run:", best_run.id, "val_loss=", best_run.summary.get("best_val_loss"))

        wandb_cfg = dict(best_run.config)
        ckpt_path = os.path.join(args.checkpoint_dir, f"best_{best_run.id}.pt")

        if not os.path.exists(ckpt_path):
            # Fall back to the WandB model artifact if the checkpoint directory
            # wasn't preserved locally (e.g. sweep ran on another machine).
            try:
                art = api.artifact(f"{args.wandb_project}/model-{best_run.id}:latest", type="model")
                art_dir = art.download(root=args.checkpoint_dir)
                for fn in os.listdir(art_dir):
                    if fn.endswith(".pt"):
                        ckpt_path = os.path.join(art_dir, fn)
                        break
            except Exception as e:
                raise FileNotFoundError(
                    f"Checkpoint not found at {ckpt_path} and no artifact available for run "
                    f"{best_run.id}. Error: {e}"
                )

    state, ckpt_cfg = load_checkpoint(
        ckpt_path, device=device,
        overrides=dict(hidden=args.hidden, emb_dim=args.emb_dim, dropout=args.dropout,
                       pool=args.pool, head=args.head),
    )

    # The checkpoint's own embedded config (already merged with any CLI
    # overrides by load_checkpoint() above) is the primary source of truth;
    # the WandB run's config is only a fallback for legacy checkpoints saved
    # without one; hardcoded literals are the last resort.
    cfg = dict(
        hidden=ckpt_cfg.get("hidden", wandb_cfg.get("hidden", 128)),
        dropout=ckpt_cfg.get("dropout", wandb_cfg.get("dropout", 0.1)),
        train_eps=ckpt_cfg.get("train_eps", wandb_cfg.get("train_eps", False)),
        pool=ckpt_cfg.get("pool", wandb_cfg.get("pool", "mean")),
        trainable=ckpt_cfg.get("trainable", wandb_cfg.get("trainable", True)),
        batch_size=wandb_cfg.get("batch_size", 64),
    )
    emb_dim = ckpt_cfg.get("emb_dim", wandb_cfg.get("EMB_DIM", wandb_cfg.get("emb_dim", 128)))
    head = ckpt_cfg.get("head", args.head or "linear")
    seed = ckpt_cfg.get("seed", 42)

    df_train, df_val, df_test = load_and_split(args.data_path, seed=seed)
    codes, code2idx, desc_to_time, code_texts = build_code_vocab(df_train)
    test_graphs = build_graphs(df_test, codes, code2idx, desc_to_time)

    # NOTE: the code embedding matrix must be reconstructed with the same
    # dimensionality used at train time (emb_dim, above), but its initial
    # *values* are irrelevant regardless of embedding_source ("random" or
    # "bioclinical"): `code_emb` is a trainable `nn.Embedding`/registered
    # buffer (see model.py), so it is part of `state_dict` and gets fully
    # overwritten by `load_state_dict` below. Only the embedding *lookup
    # indices* must line up with the same `codes` ordering used at train
    # time (guaranteed here since both are derived from the same
    # `build_code_vocab(df_train)` call on the same canonical split).
    code_emb_matrix = make_random_code_embeddings(len(codes), emb_dim)
    model = build_model_from_cfg(cfg, code_emb_matrix, edge_dim=3, out_classes=2, head=head).to(device)

    # strict=True: the architecture above is rebuilt entirely from the
    # checkpoint's own config, so a key/shape mismatch here means something
    # is genuinely wrong (e.g. an emb_dim/pool override that doesn't match
    # how the checkpoint was actually trained) and should raise loudly
    # rather than silently loading a partial/mismatched model.
    model.load_state_dict(state, strict=True)

    test_loader = DataLoader(test_graphs, batch_size=cfg.get("batch_size", 64))
    metrics, probs, labels = evaluate(model, test_loader, device)

    print(
        f"TEST | loss {metrics['loss']:.4f} acc {metrics['accuracy']:.4f} "
        f"f1 {metrics['f1']:.4f} sens {metrics['sensitivity']:.4f} "
        f"spec {metrics['specificity']:.4f} auprc {metrics['auprc']:.4f} "
        f"auroc {metrics['auroc']:.4f}"
    )
    return metrics


if __name__ == "__main__":
    main()
