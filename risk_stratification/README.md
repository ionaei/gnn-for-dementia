# Risk Stratification ("Traffic Light" System)

This module implements the paper's post-hoc risk-stratification scheme, which takes a trained GNN's per-patient dementia probabilities and reports both a baseline single-threshold performance and a "confident-only" performance after excluding ambiguous predictions — corresponding to the paper's "after risk stratification" rows in Table 1 (e.g. GNN+RV: F1=0.816, Sensitivity=0.834, Specificity=0.758, J=0.592, AUCROC=0.838).

## Method

Both of this scheme's free parameters -- the Youden-J decision threshold `t*` and the traffic-light confidence half-width `w` -- are **tuned on the validation split and then applied, with no further search, to the held-out test split**. Scoring `baseline`/`traffic_light` metrics on the same data used to pick `t*`/`w` would be train/test leakage: it optimistically biases the reported J (and everything derived from it -- sensitivity, specificity, coverage) relative to what the scheme would actually achieve on unseen patients. This mirrors how every other model in this package already separates val (model/hyperparameter selection) from test (the number that gets reported); risk stratification just has two additional tuned parameters that need the same separation. See `stratify()`'s docstring in `traffic_light_stratification.py` for the full rationale.

1. **Youden's J-optimal threshold** (`youden_threshold_for_dementia`): finds the decision threshold `t*` on P(dementia) that maximizes `J = Sensitivity + Specificity - 1`, via `sklearn.metrics.roc_curve` — computed on the **validation** split only.
2. **Traffic-light band width** (`pick_w_by_max_J`): also computed on the **validation** split only, by searching confidence half-widths `w` around `t*` (subject to a minimum coverage fraction of confidently-classified patients) for the one that maximizes J among only the confidently-classified RED/GREEN validation patients:
   - **RED** (high risk / dementia): `P(dementia) >= t* + w`
   - **GREEN** (low risk / control): `P(dementia) <= t* - w`
   - **AMBER** (ambiguous, excluded from scored metrics): everything in between
3. **Baseline metrics** (`single_threshold_metrics`): the already-fixed `t*` (from step 1) applied to the **test** split — performance if every test patient is classified dementia iff `P(dementia) >= t*`.
4. **Traffic-light metrics** (`confident_metrics`): the already-fixed `t*`/`w` (from steps 1-2) applied to the **test** split — this is the mechanism behind the paper's reported jump in performance "after risk stratification"; it comes at the cost of not confidently classifying every patient (some fraction falls in the AMBER band).

## Files

- **`traffic_light_stratification.py`** — The pure statistics: threshold search, baseline metrics, and confidence-band search, plus `stratify()`, which wires them together with the val-tune / test-report separation described above. Model loading has been deliberately factored out of this file (see below) so the same statistics can be driven by either a local checkpoint or a WandB run without duplicating the model definition.
- **`run_stratification.py`** — An end-to-end CLI that wires together `gnn/evaluate_best_run.py`'s model loading and evaluation (run once on **both** the val and test splits) and `traffic_light_stratification.py`'s `stratify()` (the statistics above). Writes a JSON report with the val-tuned Youden threshold/band width, the test-set baseline and traffic-light metrics, and the underlying raw val/test metrics.

## Usage

```bash
# Using a local (--no-wandb smoke-test) checkpoint produced by gnn/train.py.
# Checkpoints saved by the current train.py embed their own hidden/emb-dim/
# pool/head config, so you normally don't need to pass any of those:
cd risk_stratification
python run_stratification.py \
    --data-path ../data_prep/five_updated_synthetic.csv \
    --local-checkpoint ../gnn/checkpoints_gnn/best_local.pt

# Using the best run of a real WandB sweep:
python run_stratification.py \
    --data-path /path/to/five_updated.csv \
    --wandb-project NEURIPS_UPDATED_AD --checkpoint-dir ../gnn/checkpoints_gnn
```

Must be run from within `risk_stratification/` (or otherwise have `../gnn` on `sys.path`) so the relative import of the GNN modules (`checkpoint_utils`, `evaluate_best_run`, `graph_construction`, `model`, `train`) resolves.


## Output

`run_stratification.py` writes a `stratification_report.json` (path configurable via `--out`) containing:

- `t_star`, `w`, `youden_info` — the Youden-optimal threshold, the traffic-light band half-width, and the J/sensitivity/specificity achieved at `t_star` — all selected on the **validation** set, not the test set
- `val_tuned` — the full traffic-light metrics (J, sensitivity, specificity, coverage, band edges, etc.) achieved by `(t_star, w)` on the **validation** set they were tuned on; expect this to look slightly better than `traffic_light` below -- that gap is exactly the leakage the val/test split prevents from being reported as the headline number
- `baseline` — metrics on the **test** set if every test patient were classified at the (val-tuned) `t_star`
- `traffic_light` — metrics on the **test** set, restricted to confidently-classified (RED/GREEN) patients only at the (val-tuned) `t_star`/`w`, plus `coverage` (fraction of test patients confidently classified) and the band edges
- `val_metrics`, `test_metrics` — the underlying GNN's raw validation-set and test-set accuracy/F1/sensitivity/specificity/AUROC/AUPRC from `gnn/evaluate_best_run.py`'s `evaluate()`, prior to any thresholding

## Caveats

- This module only stratifies predictions from a trained `gnn/` checkpoint; it does not itself perform any training. Train a GNN checkpoint first (see `../gnn/README.md`).
- As with every other module in this package, results on the synthetic smoke-test data are structurally valid (the code runs and produces a well-formed report) but not meaningful — reproducing the paper's "after risk stratification" Table 1 numbers requires the real UK Biobank data.
- `min_coverage` (default 0.50 in `stratify()`) trades off performance against how many patients get a confident RED/GREEN call; a stricter coverage floor will generally find a smaller `w` (more patients scored, closer to baseline J) while a looser one can find a larger `w` (fewer patients scored, higher J among the confident subset). This search happens entirely on the validation set (see Method above); the paper's reported post-stratification numbers reflect a specific choice of this trade-off. Adjust `--min-coverage`/`--max-w`/`--step` to explore others.
