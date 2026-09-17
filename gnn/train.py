"""
Train the star-graph GINEConv GNN (`PatientICDGNN_BioBERT`) for dementia
risk prediction, optionally as a WandB Bayesian hyperparameter sweep.

Ported from `neurips_ad_graphs.ipynb` cells 1-17 (`train_sweep()` and the
surrounding data-loading / graph-construction / sweep-launch code). The
training loop, optimizer param groups (decay / no-decay / embedding, each
with their own LR), mixed-precision (autocast + GradScaler), linear
warmup+decay schedule, gradient clipping, and early-stopping-on-val-loss
logic are unchanged from the original. What's new/cleaned up:

  - CLI interface (argparse) instead of hardcoded paths/notebook cells.
  - `--no-wandb` mode: runs a single training run with fixed hyperparameters
    (no sweep) and skips all `wandb.*` calls, so the pipeline can be
    smoke-tested without a WandB account/login.
  - `--data-path` defaults to a synthetic CSV (generate one first via
    `data_prep/generate_synthetic_data.py`) since real UK Biobank data is
    access-controlled and cannot ship with this repository.
  - Checkpoint directory is a CLI arg rather than a hardcoded folder name.
  - The LR schedule (`_linear_warmup_decay_scheduler` below) is a small
    hand-rolled `LambdaLR` replacement for `transformers.get_linear_schedule_with_warmup`.
    The original notebook imported that function directly; this package
    pins `torch==2.1.2` for aarch64/disk-space reasons (see gnn/README or
    top-level README), and current `transformers` releases gate their
    PyTorch-backed utilities behind `torch>=2.5`, so the import silently
    resolves to a dummy stub that raises `ImportError` at call time. Rather
    than pull in a second, older `transformers` pin just for one scheduler
    function, this reimplements the exact same linear-warmup-then-linear-decay
    schedule (identical formula to `transformers`' implementation) with zero
    extra dependencies. The schedule shape and hyperparameter semantics
    (`num_warmup_steps`, `num_training_steps`) are unchanged from the
    original.

Usage:
    # Smoke test on synthetic data, no WandB, a few epochs, CPU-friendly:
    python train.py --data-path ../data_prep/five_updated_synthetic.csv \\
        --no-wandb --epochs 3 --hidden 32 --emb-dim 32 --batch-size 16

    # Real Bayesian sweep (requires `wandb login` first):
    python train.py --data-path /path/to/five_updated.csv --sweep --sweep-count 30
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import auc, f1_score, precision_recall_curve, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from checkpoint_utils import load_checkpoint, wrap_checkpoint
from graph_construction import build_code_vocab, build_graphs, make_random_code_embeddings
from model import PatientICDGNN_BioBERT
from sweep_config import SWEEP_CONFIG


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """
    Drop-in replacement for `transformers.get_linear_schedule_with_warmup`
    (see module docstring for why this is hand-rolled instead of imported).
    Linearly increases LR from 0 to the base LR over `num_warmup_steps`,
    then linearly decays it to 0 over the remaining `num_training_steps -
    num_warmup_steps` steps -- the same formula `transformers` uses
    internally.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        remaining = num_training_steps - num_warmup_steps
        progress = float(current_step - num_warmup_steps) / float(max(1, remaining))
        return max(0.0, 1.0 - progress)

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


LABEL_MAP_IDX = {"Dementia": 0, "Control": 1}
PRS_RAW_COL = "Standard PRS for alzheimer's disease (AD)"


def load_and_split(data_path, seed=42):
    """
    Load the CSV, map labels, do the 70/10/20 stratified split (matching
    the paper's headline GNN results, per SCHEMA.md), and standardize
    Age/PRS with a scaler fit on train only.
    """
    df = pd.read_csv(data_path)
    df["label"] = df["Class"].map(LABEL_MAP_IDX)

    df_train_val, df_test = train_test_split(df, test_size=0.2, stratify=df["label"], random_state=seed)
    df_train, df_val = train_test_split(
        df_train_val, test_size=0.125, stratify=df_train_val["label"], random_state=seed
    )

    scaler = StandardScaler()
    scaler.fit(df_train[["Age", PRS_RAW_COL]])
    for d in (df_train, df_val, df_test):
        d[["Age", "PRS"]] = scaler.transform(d[["Age", PRS_RAW_COL]])

    return df_train, df_val, df_test


def run_epoch(model, loader, device, optimizer=None, scheduler=None, scaler=None, mixed_precision=False, use_wandb=False):
    """One train or eval epoch. If `optimizer` is given, trains; else evals."""
    if use_wandb:
        import wandb as wandb_mod  # local import so --no-wandb doesn't require wandb installed/logged in

    train = optimizer is not None
    model.train() if train else model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for batch in tqdm(loader, disable=False):
        batch = batch.to(device)

        if train:
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=mixed_precision):
                logits = model(batch)
                loss = criterion(logits, batch.y.view(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
        else:
            with torch.no_grad():
                logits = model(batch)
                loss = criterion(logits, batch.y.view(-1))

        total_loss += loss.item() * batch.num_graphs
        preds = logits.argmax(dim=1)
        correct += (preds == batch.y.view(-1)).sum().item()
        total += batch.num_graphs
        all_preds.extend(preds.detach().cpu().tolist())
        all_labels.extend(batch.y.view(-1).detach().cpu().tolist())

        if train and use_wandb:
            wandb_mod.log({"train/step_loss": loss.item(), "lr": scheduler.get_last_lr()[0]})

    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    f1 = f1_score(all_labels, all_preds, average="macro")
    sensitivity = recall_score(all_labels, all_preds, pos_label=0)  # dementia recall
    specificity = recall_score(all_labels, all_preds, pos_label=1)  # control recall
    precision, recall, _ = precision_recall_curve(all_labels, all_preds)
    aupr = auc(recall, precision)
    return avg_loss, acc, f1, sensitivity, specificity, aupr


def _checkpoint_config(cfg, emb_dim, num_codes, head="linear"):
    """
    Metadata embedded in every checkpoint this script saves (see
    `checkpoint_utils.py`), so downstream scripts (`evaluate_best_run.py`,
    `risk_stratification/run_stratification.py`,
    `explainability/explain.py`) can rebuild the exact same architecture
    instead of guessing it from their own CLI-arg defaults -- crucially
    including `pool`, which has no learnable parameters and therefore
    leaves no trace in the state_dict itself (the pooling-mismatch bug this
    module was added to fix).
    """
    return dict(
        hidden=cfg["hidden"], dropout=cfg["dropout"], train_eps=cfg["train_eps"],
        pool=cfg["pool"], head=head, emb_dim=emb_dim, num_codes=num_codes,
        trainable=cfg.get("trainable", True),
    )


def train_one_config(cfg, train_graphs, val_graphs, test_graphs, codes, device, mixed_precision,
                      ckpt_dir, use_wandb, run=None):
    """
    Core training loop for one hyperparameter configuration. This is the
    body of the original `train_sweep()`, factored out so it can be called
    either from a WandB sweep agent (`main(sweep=True)`) or directly with a
    fixed config (`main(sweep=False)`, e.g. for smoke tests).
    """
    os.makedirs(ckpt_dir, exist_ok=True)

    train_loader = DataLoader(train_graphs, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=cfg["batch_size"])
    test_loader = DataLoader(test_graphs, batch_size=cfg["batch_size"])

    num_codes = len(codes)
    emb_dim = cfg.get("EMB_DIM", cfg.get("emb_dim", 128))
    code_emb_matrix = make_random_code_embeddings(num_codes, emb_dim)

    model = PatientICDGNN_BioBERT(
        code_emb_matrix=code_emb_matrix.to(torch.float32),
        edge_dim=3,
        hidden=cfg["hidden"],
        out_classes=2,
        dropout=cfg["dropout"],
        train_eps=cfg["train_eps"],
        trainable=True,
        pool=cfg["pool"],
        head="linear",  # the paper's winning GNN configuration
    ).to(device)

    emb_params = [p for n, p in model.named_parameters() if "code_emb" in n]
    decay, nodecay = [], []
    for n, p in model.named_parameters():
        if "code_emb" in n:
            continue
        (nodecay if n.endswith("bias") else decay).append(p)

    optimizer = AdamW(
        [
            {"params": decay, "lr": cfg["lr_main"], "weight_decay": cfg["weight_decay"]},
            {"params": nodecay, "lr": cfg["lr_main"], "weight_decay": 0.0},
            {"params": emb_params, "lr": cfg["lr_emb"], "weight_decay": 0.0},
        ]
    )

    total_steps = len(train_loader) * cfg["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(cfg["warmup_pct"] * total_steps),
        num_training_steps=max(total_steps, 1),
    )
    scaler = GradScaler(enabled=mixed_precision)

    if use_wandb:
        import wandb

        wandb.watch(model, log="gradients", log_freq=200)

    run_id = run.id if run is not None else "local"
    ckpt_path = os.path.join(ckpt_dir, f"best_{run_id}.pt")
    best_val, patience, min_delta, no_improve = float("inf"), 10, 1e-3, 0

    for epoch in range(1, cfg["epochs"] + 1):
        tr_loss, tr_acc, tr_f1, tr_sens, tr_spec, tr_aupr = run_epoch(
            model, train_loader, device, optimizer, scheduler, scaler, mixed_precision, use_wandb
        )
        va_loss, va_acc, va_f1, va_sens, va_spec, va_aupr = run_epoch(
            model, val_loader, device, mixed_precision=mixed_precision, use_wandb=use_wandb
        )

        print(
            f"Epoch {epoch}: train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f} sens {va_sens:.4f} spec {va_spec:.4f}"
        )

        if use_wandb:
            import wandb

            wandb.log(
                {
                    "method": "gnn_basic_emb_trainable",
                    "epoch": epoch,
                    "train/loss": tr_loss, "train/acc": tr_acc, "train/f1_macro": tr_f1,
                    "train/sensitivity": tr_sens, "train/specificity": tr_spec, "train/auprc": tr_aupr,
                    "val/loss": va_loss, "val/acc": va_acc, "val/f1_macro": va_f1,
                    "val/sensitivity": va_sens, "val/specificity": va_spec, "val/auprc": va_aupr,
                }
            )

        improved = (best_val - va_loss) > min_delta
        if improved:
            best_val, no_improve = va_loss, 0
            torch.save(
                wrap_checkpoint(model.state_dict(), **_checkpoint_config(cfg, emb_dim, num_codes, head="linear")),
                ckpt_path,
            )
            if run is not None:
                run.summary["best_val_loss"] = best_val
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (best val loss={best_val:.4f})")
                break

    if os.path.exists(ckpt_path):
        state, _ = load_checkpoint(ckpt_path, device=device)
        model.load_state_dict(state)
    else:
        torch.save(
            wrap_checkpoint(model.state_dict(), **_checkpoint_config(cfg, emb_dim, num_codes, head="linear")),
            ckpt_path,
        )

    te_loss, te_acc, te_f1, te_sens, te_spec, te_auc = run_epoch(
        model, test_loader, device, mixed_precision=mixed_precision, use_wandb=use_wandb
    )
    print(
        f"TEST: loss {te_loss:.4f} acc {te_acc:.4f} f1 {te_f1:.4f} "
        f"sens {te_sens:.4f} spec {te_spec:.4f} auprc {te_auc:.4f}"
    )
    if use_wandb:
        import wandb

        wandb.log(
            {"test/loss": te_loss, "test/acc": te_acc, "test/f1_macro": te_f1,
             "test/sensitivity": te_sens, "test/specificity": te_spec, "test/auprc": te_auc}
        )
        art = wandb.Artifact(f"model-{run_id}", type="model")
        art.add_file(ckpt_path)
        wandb.log_artifact(art)

    return ckpt_path, dict(test_loss=te_loss, test_acc=te_acc, test_f1=te_f1,
                            test_sensitivity=te_sens, test_specificity=te_spec, test_auprc=te_auc)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-path", type=str, default="../data_prep/five_updated_synthetic.csv")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints_gnn")
    parser.add_argument("--no-wandb", action="store_true", help="Skip all WandB logging/sweeping")
    parser.add_argument("--sweep", action="store_true", help="Launch a real WandB Bayesian sweep")
    parser.add_argument("--sweep-count", type=int, default=30)
    parser.add_argument("--wandb-project", type=str, default="NEURIPS_UPDATED_AD")
    # Single-run hyperparameters (used when --sweep is not passed)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr-main", type=float, default=1e-3)
    parser.add_argument("--lr-emb", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--warmup-pct", type=float, default=0.06)
    parser.add_argument("--train-eps", action="store_true")
    parser.add_argument("--pool", type=str, default="mean", choices=["mean", "add", "max"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision = torch.cuda.is_available()
    print(f"Using device: {device} (mixed precision: {mixed_precision})")

    df_train, df_val, df_test = load_and_split(args.data_path, seed=args.seed)
    codes, code2idx, desc_to_time, code_texts = build_code_vocab(df_train)
    print(f"{len(codes)} diagnosis codes; train/val/test = {len(df_train)}/{len(df_val)}/{len(df_test)}")

    train_graphs = build_graphs(df_train, codes, code2idx, desc_to_time)
    val_graphs = build_graphs(df_val, codes, code2idx, desc_to_time)
    test_graphs = build_graphs(df_test, codes, code2idx, desc_to_time)

    use_wandb = not args.no_wandb

    if args.sweep:
        if not use_wandb:
            raise ValueError("--sweep requires WandB; remove --no-wandb")
        import wandb

        def _sweep_entry():
            with wandb.init() as run:
                cfg = dict(wandb.config)
                train_one_config(cfg, train_graphs, val_graphs, test_graphs, codes, device,
                                  mixed_precision, args.checkpoint_dir, use_wandb=True, run=run)

        sweep_id = wandb.sweep(SWEEP_CONFIG, project=args.wandb_project)
        wandb.agent(sweep_id, function=_sweep_entry, count=args.sweep_count)
        return

    cfg = dict(
        epochs=args.epochs, batch_size=args.batch_size, hidden=args.hidden, EMB_DIM=args.emb_dim,
        dropout=args.dropout, lr_main=args.lr_main, lr_emb=args.lr_emb, weight_decay=args.weight_decay,
        warmup_pct=args.warmup_pct, train_eps=args.train_eps, pool=args.pool, trainable=True,
    )

    run = None
    if use_wandb:
        import wandb

        run = wandb.init(project=args.wandb_project, config=cfg)

    ckpt_path, metrics = train_one_config(
        cfg, train_graphs, val_graphs, test_graphs, codes, device, mixed_precision,
        args.checkpoint_dir, use_wandb=use_wandb, run=run,
    )
    print(f"Done. Best checkpoint: {ckpt_path}")
    print(metrics)


if __name__ == "__main__":
    main()
