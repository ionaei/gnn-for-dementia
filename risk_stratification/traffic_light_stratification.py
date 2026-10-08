"""
Risk stratification ("traffic light") for the GNN's dementia predictions.

Given a trained `PatientICDGNN_BioBERT` and a test set, this:

  1. Finds the Youden's J-optimal decision threshold t* on P(dementia)
     (`youden_threshold_for_dementia`), maximizing Sensitivity + Specificity - 1.
  2. Reports baseline metrics if every patient were classified at t*
     (`single_threshold_metrics`).
  3. Searches for a confidence band width w around t* (`pick_w_by_max_J` /
     `confident_metrics`) that maximizes J among only the "confident"
     predictions -- P(dementia) >= t*+w is called RED (high risk), P(dementia)
     <= t*-w is called GREEN (low risk), and anything in between is AMBER
     (ambiguous / not confidently classified either way). This is the
     "traffic light" risk-band scheme described in the paper.

This module factors the *statistics* (threshold search / traffic-light
banding) out from the *model loading*, so it can be driven either by a
freshly-trained local checkpoint (via `gnn/evaluate_best_run.py`) or a
WandB-hosted run, without duplicating the model definition. See
`run_stratification.py` in this same directory for a runnable end-to-end
CLI that wires the two together.
"""

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_curve


def youden_threshold_for_dementia(y, p_dem):
    """
    Pick t* maximizing J = Sensitivity + Specificity - 1 for dementia
    (class 0), using P(dementia) as the decision score.

    Args:
        y (array-like): true labels, 0=Dementia, 1=Control.
        p_dem (array-like): predicted P(dementia) per patient.

    Returns:
        (t_star, info): t_star is the optimal threshold; info is a dict with
        the J statistic and the sensitivity/specificity achieved at t_star.
    """
    y = np.asarray(y)
    p_dem = np.asarray(p_dem)
    y_dem = (y == 0).astype(int)  # 1 = dementia positive
    fpr, tpr, thr = roc_curve(y_dem, p_dem)  # scores = P(dementia)
    spec = 1.0 - fpr
    J = tpr + spec - 1.0
    i = int(np.nanargmax(J))
    return float(thr[i]), {"J": float(J[i]), "sensitivity": float(tpr[i]), "specificity": float(spec[i])}


def metrics_binary_pos0(y_true, y_pred):
    """Binary classification metrics with dementia (class 0) as the positive class."""
    sens = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    spec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    prec = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    J = sens + spec - 1.0
    return dict(
        sensitivity=sens, specificity=spec, precision=prec, f1=f1, accuracy=acc,
        tp=int(tp), fn=int(fn), fp=int(fp), tn=int(tn), J=J,
    )


def single_threshold_metrics(y, p_dem, t):
    """Metrics if every patient is classified dementia iff P(dementia) >= t."""
    y = np.asarray(y)
    p_dem = np.asarray(p_dem)
    y_pred = np.where(p_dem >= t, 0, 1)
    return metrics_binary_pos0(y, y_pred)


def confident_metrics(y, p_dem, t, w):
    """
    Traffic-light banding around threshold t with half-width w:
      - RED  (dementia):  P(dementia) >= t + w
      - GREEN (control):  P(dementia) <= t - w
      - AMBER (ambiguous): t - w < P(dementia) < t + w -- excluded from the
        returned metrics (only "confident" predictions are scored).

    Returns a metrics dict (as `metrics_binary_pos0`) plus `t`, `w`, the
    band edges `tl`/`tu`, the fraction of patients confidently classified
    (`coverage`), and the count `n`.
    """
    y = np.asarray(y)
    p_dem = np.asarray(p_dem)
    tl, tu = max(0.0, t - w), min(1.0, t + w)
    decisions = np.full_like(y, -1)
    decisions[p_dem >= tu] = 0  # RED: dementia
    decisions[p_dem <= tl] = 1  # GREEN: not dementia
    mask = decisions != -1
    if not mask.any():
        return dict(t=t, w=w, tl=tl, tu=tu, coverage=0.0, n=0)
    m = metrics_binary_pos0(y[mask], decisions[mask])
    m.update(dict(t=t, w=w, tl=tl, tu=tu, coverage=float(mask.mean()), n=int(mask.sum())))
    return m


def pick_w_by_max_J(y, p_dem, t, max_w=0.25, step=0.005, min_coverage=0.0):
    """
    Search band half-widths w in [0, max_w] (step `step`) for the one that
    maximizes J among confidently-classified patients, subject to a minimum
    coverage fraction `min_coverage`.
    """
    best = None
    for w in np.arange(0.0, max_w + 1e-12, step):
        m = confident_metrics(y, p_dem, t, w)
        if m["n"] == 0 or m["coverage"] < min_coverage:
            continue
        if best is None or m["J"] > best["J"]:
            best = m
    return best


def stratify(val_labels, val_probs_control, test_labels, test_probs_control,
             max_w=0.30, step=0.002, min_coverage=0.50):
    """
    End-to-end convenience wrapper, with threshold *tuning* and final
    *reporting* on deliberately separate data.

    Why two sets, not one: an earlier version of this function (and of
    `run_stratification.py`, which only ever built a test set) took a
    single `(labels, probs_control)` pair, searched it for both the
    Youden-J threshold `t*` (`youden_threshold_for_dementia`) AND the
    confidence-band half-width `w` that maximizes J among confident
    predictions (`pick_w_by_max_J`), and then reported `baseline`/
    `traffic_light` metrics computed on that *same* data. Tuning two free
    parameters (t*, w) to maximize a metric on a set and then reporting
    that same metric on that same set is standard train/test leakage -- it
    optimistically biases the reported J (and everything derived from it:
    sensitivity, specificity, coverage) relative to what the traffic-light
    scheme would actually achieve on unseen patients.

    The fix: `t*` and `w` are now selected using ONLY the validation set
    (`val_labels`/`val_probs_control`); those fixed values are then
    applied -- with no further search -- to the test set
    (`test_labels`/`test_probs_control`) to produce the metrics actually
    reported. This mirrors how every other model in this package already
    uses val (for early stopping / model selection) vs. test (for the
    number that gets reported) -- risk stratification just has two
    additional tuned parameters (t*, w) that need the same separation.

    Args:
        val_labels, val_probs_control: validation-set labels (0=Dementia,
            1=Control) and the model's P(control) on that set -- used only
            to pick t* and w, never scored themselves as the headline
            numbers.
        test_labels, test_probs_control: held-out test-set labels and
            P(control) -- scored at the (val-tuned) t*/w to produce
            `baseline` and `traffic_light` below.
        max_w, step, min_coverage: passed to `pick_w_by_max_J` (run on the
            validation set only).

    Returns:
        dict with keys: t_star, w, youden_info, val_tuned, baseline,
        traffic_light. `baseline`/`traffic_light` are test-set metrics;
        `youden_info`/`val_tuned` describe how t*/w performed on the
        validation set they were chosen from (expect these to look
        slightly better than the test-set numbers -- that gap is exactly
        the leakage this split stops from being reported as the headline
        number).
    """
    val_labels = np.asarray(val_labels)
    val_probs_control = np.asarray(val_probs_control)
    test_labels = np.asarray(test_labels)
    test_probs_control = np.asarray(test_probs_control)

    p_dem_val = 1.0 - val_probs_control
    p_dem_test = 1.0 - test_probs_control

    # Tune on validation only.
    t_star, info = youden_threshold_for_dementia(val_labels, p_dem_val)
    val_tuned = pick_w_by_max_J(val_labels, p_dem_val, t_star, max_w=max_w, step=step, min_coverage=min_coverage)
    w = val_tuned["w"] if val_tuned is not None else 0.0
    if val_tuned is None:
        print(
            f"WARNING: no confidence-band width in [0, {max_w}] reached "
            f"min_coverage={min_coverage} on the validation set; falling "
            "back to w=0 (every patient classified, no AMBER band)."
        )

    # Report on test only, at the already-fixed (t_star, w) -- no search here.
    baseline = single_threshold_metrics(test_labels, p_dem_test, t_star)
    traffic = confident_metrics(test_labels, p_dem_test, t_star, w)

    print("Youden t* (tuned on val):", t_star, info)
    print("w (tuned on val):", w, "-- val-set traffic-light metrics at this w:", val_tuned)
    print("Baseline (single threshold, everyone classified, TEST set):", baseline)
    print("Traffic-light (confident-only, TEST set):", traffic)

    if traffic is not None and traffic.get("n", 0) > 0:
        tl, tu = traffic["tl"], traffic["tu"]
        print(f"\nBaseline threshold on P(dementia): t* = {t_star:.3f}")
        print(f"Baseline threshold on P(control):  1 - t* = {1 - t_star:.3f}")
        print(f"Traffic-light thresholds on P(dementia): tl = {tl:.3f}, tu = {tu:.3f}")
        print(f"Traffic-light thresholds on P(control):  green >= {1 - tl:.3f}, red <= {1 - tu:.3f}")

    return dict(t_star=t_star, w=w, youden_info=info, val_tuned=val_tuned,
                baseline=baseline, traffic_light=traffic)
