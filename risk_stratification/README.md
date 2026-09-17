# Risk Stratification ("Traffic Light" System)

This module implements the paper's post-hoc risk-stratification scheme, which takes a trained GNN's per-patient dementia probabilities and reports both a baseline single-threshold performance and a "confident-only" performance after excluding ambiguous predictions — corresponding to the paper's "after risk stratification" rows in Table 1 (e.g. GNN+RV: F1=0.816, Sensitivity=0.834, Specificity=0.758, J=0.592, AUCROC=0.838).

## Method

1. **Youden's J-optimal threshold** (`youden_threshold_for_dementia`): finds the decision threshold `t*` on P(dementia) that maximizes `J = Sensitivity + Specificity - 1`, via `sklearn.metrics.roc_curve`.
2. **Baseline metrics** (`single_threshold_metrics`): performance if every patient is classified dementia iff `P(dementia) >= t*`.
3. **Traffic-light banding** (`confident_metrics` / `pick_w_by_max_J`): searches for a confidence half-width `w` around `t*` such that:
   - **RED** (high risk / dementia): `P(dementia) >= t* + w`
   - **GREEN** (low risk / control): `P(dementia) <= t* - w`
   - **AMBER** (ambiguous, excluded from scored metrics): everything in between

   `w` is chosen (subject to a minimum coverage fraction of confidently-classified patients) to maximize J among only the confidently-classified RED/GREEN patients. This is the mechanism behind the paper's reported jump in performance "after risk stratification" — it comes at the cost of not confidently classifying every patient (some fraction falls in the AMBER band).

## Files

- **`traffic_light_stratification.py`** — The pure statistics: threshold search, baseline metrics, and confidence-band search. Ported verbatim (statistical logic unchanged) from `traffic_light.ipynb` cells 7-11. Model loading has been deliberately factored out of this file (see below) so the same statistics can be driven by either a local checkpoint or a WandB run without duplicating the model definition.
- **`run_stratification.py`** — An end-to-end CLI that wires together `gnn/evaluate_best_run.py` (model loading + test-set evaluation) and `traffic_light_stratification.py` (the statistics above), since the original `traffic_light.ipynb` inlined both concerns together (including redefining the model class inline). Writes a JSON report with the Youden threshold, baseline metrics, traffic-light metrics, and the underlying test-set metrics.

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

- `t_star`, `youden_info` — the Youden-optimal threshold and the J/sensitivity/specificity achieved at it on the full test set
- `baseline` — metrics if every test patient were classified at `t_star`
- `traffic_light` — metrics restricted to confidently-classified (RED/GREEN) patients only, plus `coverage` (fraction of patients confidently classified) and the band edges
- `test_metrics` — the underlying GNN's raw test-set accuracy/F1/sensitivity/specificity/AUPRC from `gnn/evaluate_best_run.py`, prior to any thresholding

## Caveats

- This module only stratifies predictions from a trained `gnn/` checkpoint; it does not itself perform any training. Train a GNN checkpoint first (see `../gnn/README.md`).
- As with every other module in this package, results on the synthetic smoke-test data are structurally valid (the code runs and produces a well-formed report) but not meaningful — reproducing the paper's "after risk stratification" Table 1 numbers requires the real UK Biobank data.
- `min_coverage` (default 0.50 in `stratify()`) trades off performance against how many patients get a confident RED/GREEN call; a stricter coverage floor will generally find a smaller `w` (more patients scored, closer to baseline J) while a looser one can find a larger `w` (fewer patients scored, higher J among the confident subset). The paper's reported post-stratification numbers reflect a specific choice of this trade-off; adjust `--min-coverage`/`--max-w`/`--step` to explore others.
