"""
Shared checkpoint (de)serialization helpers for the GNN pipeline.

Why this exists (the pooling-mismatch bug): `PatientICDGNN_BioBERT`'s graph
pooling ("mean" / "add" / "max") has no learnable parameters, so it leaves
no trace in a raw `state_dict` -- it cannot be recovered from the checkpoint
weights alone. Before this module existed, `gnn/train.py` saved a bare
`model.state_dict()` and every downstream script guessed the pooling from
its own `--pool` CLI default: `train.py`/`evaluate_best_run.py` defaulted to
`"mean"`, but `risk_stratification/run_stratification.py` defaulted to
`"add"` (hardcoded to match one specific historical run). A checkpoint
trained with one pooling and loaded with the wrong default produces no
error -- `load_state_dict` succeeds either way, since pooling isn't a
parameter -- it just silently computes the wrong predictions.

The fix: `gnn/train.py` now saves a *wrapped* checkpoint,
`{"state_dict": <model state dict>, "config": {...}}`, where `config`
records every hyperparameter needed to rebuild the exact architecture
(`hidden`, `emb_dim`, `num_codes`, `dropout`, `train_eps`, `pool`, `head`,
`trainable`). `load_checkpoint()` below is the single shared loader used by
`evaluate_best_run.py`, `risk_stratification/run_stratification.py`, and
(in spirit -- it has its own richer inference logic for legacy checkpoints)
`explainability/explain.py`, so all three read the same config the same way
instead of each guessing independently.

Old checkpoints saved before this fix (e.g. the currently-committed
`gnn/checkpoints_gnn/best_local.pt`, a bare state_dict with no embedded
config) are still loadable: `load_checkpoint()` falls back to whatever the
caller passes via `overrides`, and warns loudly if `pool` in particular
can't be determined either way, since that's the one hyperparameter that
can't be sanity-checked against the state_dict's tensor shapes.
"""

import warnings

import torch


def wrap_checkpoint(model_state_dict, **config):
    """
    Build the wrapped checkpoint payload saved by `gnn/train.py`.

    `config` should include at least: hidden, emb_dim, num_codes, dropout,
    train_eps, pool, head, trainable.
    """
    return {"state_dict": model_state_dict, "config": dict(config)}


def load_checkpoint(ckpt_path, device="cpu", overrides=None):
    """
    Load a checkpoint file and return `(state_dict, config)`.

    Args:
        ckpt_path: path to a `.pt` file, in any of three formats (newest
            first): (1) `{"state_dict": ..., "config": {...}}`, written by
            `gnn/train.py` from this fix onward; (2) `{"model_state_dict":
            ...}`, an older wrapped format some scripts defensively checked
            for but that `train.py` never actually produced (no embedded
            config either way); (3) a bare state_dict (`dict[str, Tensor]`),
            what `train.py` produced before this fix.
        device: passed through to `torch.load(..., map_location=device)`.
        overrides: dict of CLI-supplied hyperparameters (e.g.
            `{"pool": "add"}`). Any value that is not `None` always wins
            over whatever is embedded in the checkpoint -- matching
            `explainability/explain.py`'s existing "CLI always wins"
            behavior, since a human explicitly passing `--pool` presumably
            knows better than either the checkpoint or a guessed default.

    Returns:
        (state_dict, config) -- `config` is a plain dict; missing keys are
        the caller's responsibility to default (mirrors `dict.get(...,
        default)` throughout the rest of this package).
    """
    overrides = overrides or {}
    checkpoint = torch.load(ckpt_path, map_location=device)

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint and "config" in checkpoint:
        state = checkpoint["state_dict"]
        config = dict(checkpoint["config"])
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        config = dict(checkpoint.get("config", {}))
    else:
        state = checkpoint
        config = {}

    pool_recoverable = "pool" in config or overrides.get("pool") is not None

    for key, val in overrides.items():
        if val is not None:
            config[key] = val

    if not pool_recoverable:
        config.setdefault("pool", "mean")
        warnings.warn(
            f"Checkpoint at {ckpt_path!r} does not record its pooling strategy "
            "and --pool was not given; defaulting to 'mean'. Pooling has no "
            "learnable weights, so this CANNOT be verified from the checkpoint "
            "-- if the original run used pool='add' or pool='max', predictions "
            "will be silently wrong. Pass --pool explicitly if you know the "
            "run's config."
        )

    return state, config
