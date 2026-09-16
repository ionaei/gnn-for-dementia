"""
Risk stratification ("traffic light") for the GNN's dementia predictions.

Ported verbatim (the statistical logic is unchanged) from
`traffic_light.ipynb` cells 7-11 in the original working directory. Given a
trained `PatientICDGNN_BioBERT` and a test set, this:

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

What's new here (the original notebook re-defined the model class and
re-loaded a specific WandB run inline): this module factors the *statistics*
(threshold search / traffic-light banding) out from the *model loading*, so
it can be driven either by a freshly-trained local checkpoint (via
`gnn/evaluate_best_run.py`) or a WandB-hosted run, without duplicating the
model definition. See `run_stratification.py` in this same directory for a
runnable end-to-end CLI that wires the two together.
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


def stratify(labels, probs_control, max_w=0.30, step=0.002, min_coverage=0.50):
    """
    End-to-end convenience wrapper matching the original notebook's usage
    (cells 10-11): given test-set labels and P(control) from the model,
    computes the Youden threshold, the baseline (single-threshold) metrics,
    and the traffic-light (confident-only) metrics, and prints the
    dementia-probability and control-probability views of both threshold
    schemes.

    Args:
        labels (array-like): true labels, 0=Dementia, 1=Control.
        probs_control (array-like): model's predicted P(control) (i.e. the
            softmax probability of class 1) per patient, as returned by
            `gnn/evaluate_best_run.evaluate`.
        max_w, step, min_coverage: passed to `pick_w_by_max_J`.

    Returns:
        dict with keys: t_star, youden_info, baseline, traffic_light.
    """
    labels = np.asarray(labels)
    probs_control = np.asarray(probs_control)
    p_dem = 1.0 - probs_control

    t_star, info = youden_threshold_for_dementia(labels, p_dem)
    baseline = single_threshold_metrics(labels, p_dem, t_star)
    traffic = pick_w_by_max_J(labels, p_dem, t_star, max_w=max_w, step=step, min_coverage=min_coverage)

    print("Youden t*:", t_star, info)
    print("Baseline (single threshold, everyone classified):", baseline)
    print("Traffic-light (confident-only):", traffic)

    if traffic is not None:
        tl, tu = traffic["tl"], traffic["tu"]
        print(f"\nBaseline threshold on P(dementia): t* = {t_star:.3f}")
        print(f"Baseline threshold on P(control):  1 - t* = {1 - t_star:.3f}")
        print(f"Traffic-light thresholds on P(dementia): tl = {tl:.3f}, tu = {tu:.3f}")
        print(f"Traffic-light thresholds on P(control):  green >= {1 - tl:.3f}, red <= {1 - tu:.3f}")

    return dict(t_star=t_star, youden_info=info, baseline=baseline, traffic_light=traffic)
