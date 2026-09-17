"""
Load the best run from a WandB sweep and evaluate it on the held-out test
set.

Ported from `neurips_ad_graphs.ipynb` cell 20, which had two bugs that
prevented it from ever running (this was the last, unfinished cell in the
notebook):

  1. A stray typo `adfdf` on its own line, immediately after the imports --
     this alone would `NameError` and halt execution before anything else
     in the cell ran.
  2. A call to `build_model_from_cfg(cfg)`, a function that is never
     defined anywhere in the notebook.
  3. (Latent, would have surfaced next) `ckpt_path = f"checkpoints/best_{best_run.id}.pt"`
     -- but every run in this project actually saved to
     `checkpoints_trainable_GNN_simpleemb_pureGNN_updated_df/best_{run.id}.pt`
     (see `train.py` / the original `train_sweep()`), so this path would
     not have found the checkpoint even after fixing bugs 1 and 2.

This script fixes all three: it removes the typo, implements
`build_model_from_cfg` (now `model.build_model_from_cfg`, reused from
`model.py`), and takes the checkpoint directory as a CLI argument instead
of hardcoding a mismatched path.
"""

import argparse
import os

import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, recall_score
from torch_geometric.loader import DataLoader

from checkpoint_utils import load_checkpoint
from graph_construction import build_code_vocab, build_graphs, make_random_code_embeddings
from model import build_model_from_cfg
from train import load_and_split


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        loss = torch.nn.functional.cross_entropy(logits, batch.y.view(-1))
        total_loss += loss.item() * batch.num_graphs

        probs = torch.softmax(logits, dim=1)[:, 1]  # P(control), i.e. class 1
        preds = logits.argmax(dim=1)

        correct += (preds == batch.y.view(-1)).sum().item()
        total += batch.num_graphs
        all_probs.extend(probs.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())
        all_labels.extend(batch.y.view(-1).cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")
    sens = recall_score(all_labels, all_preds, pos_label=0)  # dementia recall
    spec = recall_score(all_labels, all_preds, pos_label=1)  # control recall
    aucpr = average_precision_score(all_labels, all_probs, pos_label=1)

    return dict(
        loss=total_loss / max(total, 1), accuracy=acc, f1_macro=f1,
        sensitivity=sens, specificity=spec, auprc=aucpr,
    ), all_probs, all_labels


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

    df_train, df_val, df_test = load_and_split(args.data_path)
    codes, code2idx, desc_to_time, code_texts = build_code_vocab(df_train)
    test_graphs = build_graphs(df_test, codes, code2idx, desc_to_time)

    state = None  # loaded here for --local-checkpoint; loaded further down for the WandB path
    if args.local_checkpoint:
        ckpt_path = args.local_checkpoint
        state, ckpt_cfg = load_checkpoint(
            ckpt_path, device=device,
            overrides=dict(hidden=args.hidden, emb_dim=args.emb_dim, dropout=args.dropout,
                           pool=args.pool, head=args.head),
        )
        cfg = dict(
            hidden=ckpt_cfg.get("hidden", 128), dropout=ckpt_cfg.get("dropout", 0.1),
            train_eps=ckpt_cfg.get("train_eps", False), pool=ckpt_cfg.get("pool", "mean"),
            trainable=ckpt_cfg.get("trainable", True),
        )
        emb_dim = ckpt_cfg.get("emb_dim", 128)
        head = ckpt_cfg.get("head", "linear")
    else:
        import wandb

        api = wandb.Api()
        runs = api.runs(args.wandb_project)
        if args.run_id:
            best_run = next(r for r in runs if r.id == args.run_id)
        else:
            best_run = min(runs, key=lambda r: r.summary.get("best_val_loss", float("inf")))
        print("Best run:", best_run.id, "val_loss=", best_run.summary.get("best_val_loss"))

        cfg = dict(best_run.config)
        emb_dim = cfg.get("EMB_DIM", cfg.get("emb_dim", 128))
        head = args.head or "linear"
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

    # NOTE: the code embedding matrix must be reconstructed with the same
    # dimensionality used at train time. Since the random Xavier init isn't
    # itself saved, exact numerical reproduction of a *specific* past run's
    # code embeddings isn't possible without having seeded and saved them --
    # this only matters for the *trainable* random-vector variant, since the
    # embedding weights are part of the loaded state_dict, but the embedding
    # *lookup indices* must still line up with the same `codes` ordering used
    # at train time (guaranteed here since both are derived from the same
    # `build_code_vocab(df_train)` call).
    code_emb_matrix = make_random_code_embeddings(len(codes), emb_dim)
    model = build_model_from_cfg(cfg, code_emb_matrix, edge_dim=3, out_classes=2, head=head).to(device)

    if state is None:
        # WandB path: state wasn't already loaded via load_checkpoint() above.
        state, _ = load_checkpoint(ckpt_path, device=device)
    model.load_state_dict(state, strict=False)

    test_loader = DataLoader(test_graphs, batch_size=cfg.get("batch_size", 64))
    metrics, probs, labels = evaluate(model, test_loader, device)

    print(
        f"TEST | loss {metrics['loss']:.4f} acc {metrics['accuracy']:.4f} "
        f"f1 {metrics['f1_macro']:.4f} sens {metrics['sensitivity']:.4f} "
        f"spec {metrics['specificity']:.4f} auprc {metrics['auprc']:.4f}"
    )
    return metrics


if __name__ == "__main__":
    main()
