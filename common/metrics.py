"""
Unified evaluation-metrics convention, shared by gnn/, baselines/,
bert_models/, and risk_stratification/.

Why this exists: an external code review of this package found that every
module computed metrics its own way -- some used Control (label 1) as the
"positive" class for F1, others used Dementia (label 0); `baselines/`
computed no F1 at all; `gnn/train.py` computed "AUPRC" from hard 0/1
predictions (via `precision_recall_curve` on `all_preds`) while
`gnn/evaluate_best_run.py` computed it from probabilities
(`average_precision_score` on `all_probs`) -- two different numbers for
the same checkpoint, both called "auprc". None of this changes what the
models predict, but it means numbers from different scripts (or even the
same script before/after a refactor) were not comparable.

Convention used here, applied uniformly:
  - Dementia (label 0) is the positive class for sensitivity, specificity,
    precision, F1, and Youden's J -- matching risk_stratification's
    pre-existing convention, since that is the module whose whole purpose
    is threshold selection on P(dementia).
  - AUROC and AUPRC are computed from predicted PROBABILITIES, never from
    hard predictions. AUROC is invariant to which class is called
    "positive" (flipping both the label and the score preserves the
    ranking statistic), so `roc_auc_score(y_true, P(control))` is already
    a valid, convention-independent AUROC. AUPRC is NOT invariant to the
    positive-class choice, so it is computed explicitly with Dementia as
    positive: `average_precision_score(y_true == 0, P(dementia))`.

NOTE ON THE PAPER'S PUBLISHED TABLE 1: those numbers (see the top-level
README) were computed before this module existed, under each model's own
original, now-superseded convention. They are kept as published for
citation accuracy -- this module is for anyone re-running the code from
here on, and will not exactly reproduce those historical numbers even on
the real UK Biobank data, precisely because the convention itself changed
(e.g. F1's positive class, AUPRC computed from probabilities instead of
hard predictions). See the top-level README's "Metrics convention note"
for the full explanation.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_metrics(y_true, y_pred, probs_control):
    """
    Args:
        y_true: array-like of ground-truth labels, 0=Dementia, 1=Control.
        y_pred: array-like of hard 0/1 predictions (e.g. argmax of logits,
            or a thresholded probability).
        probs_control: array-like of P(Control) = P(label 1), e.g.
            `softmax(logits, dim=1)[:, 1]` -- used for AUROC/AUPRC only,
            never for the hard-prediction-based metrics below.

    Returns:
        dict with accuracy, sensitivity, specificity, precision, f1, J
        (all Dementia-positive, from `y_pred`), auroc and auprc (from
        `probs_control`), and the raw confusion-matrix counts
        tp/fn/fp/tn (Dementia-positive: tp = correctly predicted Dementia).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    probs_control = np.asarray(probs_control, dtype=float)
    p_dementia = 1.0 - probs_control
    y_dementia = (y_true == 0).astype(int)

    sensitivity = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    specificity = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    precision = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    J = sensitivity + specificity - 1.0

    # AUROC is direction-invariant to the positive-class choice, so scoring
    # with P(control) is equivalent to scoring with P(dementia).
    try:
        auroc = roc_auc_score(y_true, probs_control)
    except ValueError:
        # Only one class present in y_true (can happen on tiny smoke-test
        # batches) -- AUROC is undefined; report NaN rather than crash.
        auroc = float("nan")

    try:
        auprc = average_precision_score(y_dementia, p_dementia)
    except ValueError:
        auprc = float("nan")

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

    return dict(
        accuracy=float(accuracy),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        precision=float(precision),
        f1=float(f1),
        J=float(J),
        auroc=float(auroc),
        auprc=float(auprc),
        tp=int(tp),
        fn=int(fn),
        fp=int(fp),
        tn=int(tn),
    )
