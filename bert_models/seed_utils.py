"""
Shared determinism helper for the BERT-family classifiers in this folder.

Mirrors `gnn/seed_utils.py` (same fix, same reasoning), duplicated here
rather than imported across folders because `bert_models/` and `gnn/` are
independently installable pieces of this reproducibility package (separate
`requirements.txt`, pinned separately) and neither currently has a shared
top-level utils module.

Bug this fixes: none of `text_only_classifier.py`, `bioclinical_no_temporal.py`,
`roberta_classifier.py`, or `multimodal_classifier.py` seeded the global
Python/NumPy/PyTorch RNGs (there was no `--seed` CLI argument at all). The
train/val/test split's `random_state=42` was hardcoded and reproducible,
but everything after the split was not:

  - `AutoModelForSequenceClassification.from_pretrained(..., num_labels=2)`
    attaches a freshly, randomly initialised classification head on top of
    the pretrained backbone (the backbone weights themselves are fixed by
    the pretrained checkpoint, but this head is not).
  - `torch.nn.Dropout` at train time.
  - `torch.utils.data.DataLoader(..., shuffle=True)`'s default
    `RandomSampler`, which reshuffles training batches every epoch.

So two runs of the same script with identical CLI args gave different
metrics, exactly as reported for `gnn/train.py` (test accuracy 0.6275 vs
0.6325 across two back-to-back runs) -- the same root cause, just not
independently reproduced against these scripts here (no cached pretrained
weights / network access in this environment to run
`AutoModelForSequenceClassification.from_pretrained` end-to-end and
confirm bit-for-bit, though the mechanism -- and the fix -- is identical).
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Seed every RNG this script's training code touches, so that repeated
    runs with identical CLI args (including the classification head's
    random init and the training DataLoader's shuffle order) produce
    identical metrics.

    Does NOT change dataset splitting: `train_test_split(..., random_state=42)`
    in `prepare_data()` already uses its own isolated `RandomState` and was
    already reproducible before this fix.

    Caveat: fully reproducible on CPU. On GPU (CUDA), `torch.backends.cudnn.deterministic
    = True` forces deterministic (if slower) conv/attention kernel
    selection where cuDNN supports it, but some GPU kernels are still
    non-deterministic by design regardless of seeding -- a known, accepted
    limitation of GPU training in general.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
