"""
Shared determinism helper for the GNN pipeline.

Bug this fixes: `gnn/train.py` seeded the train/val/test *split* (sklearn's
`train_test_split(..., random_state=seed)`, which uses its own isolated
RandomState and was already reproducible) but never seeded the global
PyTorch/NumPy/Python RNGs. Everything downstream of the split still drew
from those global RNGs with no seed at all:

  - Model weight initialisation (Xavier/Kaiming init on `PatientICDGNN_BioBERT`'s
    Linear/GINEConv layers).
  - `make_random_code_embeddings()`'s Xavier-uniform init of the code
    embedding matrix (its own `seed=` kwarg is optional and wasn't being
    passed from `train.py`).
  - `torch.nn.Dropout` at train time.
  - `torch_geometric.loader.DataLoader(..., shuffle=True)`'s default
    `RandomSampler`, which reshuffles training batches every epoch.

None of that is visible in a saved checkpoint (a state_dict is just the
final weights, not the RNG trace that produced them), so two runs of
`python train.py` with identical CLI args -- including identical
`--seed` -- silently diverged (e.g. test accuracy 0.6275 vs 0.6325,
sensitivity 0.635 vs 0.576 observed across two back-to-back runs).

Fix: call `set_seed(args.seed)` before anything that draws from those
RNGs. `train.py` calls it once in `main()` (defensive/cheap) and again at
the top of `train_one_config()` for every individual run -- including
each trial of a `--sweep`, so that re-running an identical sweep (same
sampled hyperparameters) reproduces identical per-trial results too,
rather than each trial's reproducibility depending on exactly how much
randomness every prior trial in the process happened to consume.
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Seed every RNG this package's training/eval code touches.

    Does NOT change dataset splitting: `train_test_split(..., random_state=seed)`
    already uses its own isolated `RandomState` instance and was already
    reproducible before this fix.

    Caveat: this makes CPU runs fully reproducible. GPU (CUDA) runs are
    "best effort" -- `torch.backends.cudnn.deterministic = True` forces
    cuDNN to pick deterministic (if slower) convolution algorithms instead
    of benchmarking and picking whichever is fastest, but it does not cover
    every non-deterministic GPU kernel. In particular, `torch_geometric`'s
    scatter/segment-reduce ops (used by `GINEConv`'s message aggregation)
    can still be non-deterministic on CUDA due to floating-point
    non-associativity in atomic adds, independent of any RNG seed. This is
    a known, accepted limitation of GPU training in general, not something
    this fix can close -- CPU runs (e.g. this package's `--no-wandb` smoke
    tests) are unaffected and are fully bit-reproducible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
