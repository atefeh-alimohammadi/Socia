# Socia ML final evaluation pipeline

Usage notes only. This is NOT the research report/write-up — that comes
after these results are generated and reviewed, per your instructions.

## Requirements

- Python environment with: `torch`, `numpy`, `pandas`, `scikit-learn`,
  `scipy`, `matplotlib` (steps 2-6 need the first five; step 7 needs
  `matplotlib` only and does not need `torch`).
- The frozen project code, unmodified: `model.py`, `model_v2.py`,
  `model_v3a.py`, `model_v3c.py`, `dataset_v2.py` (all importable from one
  `--code-dir`).
- The frozen data directory: `train.jsonl`, `val.jsonl`, `test.jsonl`,
  `val_eval_metadata.jsonl`, `test_eval_metadata.jsonl`,
  `counterfactual_eval.jsonl`.
- The two checkpoints: `best_model_v3a.pt`, `best_model_v3c.pt`.

## Commands (run in order; each step only reads what earlier steps wrote)

```bash
cd research/ml/eval

python step2_final_evaluation.py \
    --data-dir /path/to/Socia/ml/dataset/scripts/data \
    --checkpoints-dir /path/to/Socia/ml/models/checkpoints \
    --code-dir /path/to/Socia/ml/models \
    --out-dir ./eval_outputs

python step3_subtype_analysis.py --data-dir /path/to/.../data --out-dir ./eval_outputs
python step4_counterfactual_analysis.py --data-dir /path/to/.../data --out-dir ./eval_outputs
python step5_recurrence_analysis.py --data-dir /path/to/.../data --out-dir ./eval_outputs
python step6_error_analysis.py --data-dir /path/to/.../data --out-dir ./eval_outputs

python step7_visualize.py --out-dir ./eval_outputs
```

Only step 2 imports `torch` / runs model inference. Steps 3-7 are pure
analysis over the files step 2 writes; none of them re-run a model.

## What step 2 does, precisely

1. Loads train/val/test/counterfactual data once.
2. Recomputes V2's `time_stats` from `train.jsonl` directly (via
   `dataset_v2.fit_time_normalization_v2`, unmodified) and compares
   against each checkpoint's own stored `time_stats` — this is the same
   check already done manually during the audit (which found a 0.0
   discrepancy against this exact dataset); it now runs automatically
   every time and will print a `WARNING` and continue rather than fail
   silently if `--data-dir` doesn't point at the right dataset.
3. Fits the timing+category baseline fresh on `train.jsonl` (same
   features/model as the earlier audit's `diagnostics.py`), giving a
   directly comparable, same-protocol baseline number.
4. For V3-A and V3-C: loads the checkpoint, runs inference on val/test/
   counterfactual, independently re-derives the validation-selected
   threshold by sweeping the SAME grid the training scripts used
   (0.10-0.90 step 0.05) and comparing against the checkpoint's stored
   threshold — again a `WARNING`, not a silent pass, if they disagree.
   **The threshold actually used for every test-set number is always the
   one freshly derived from validation predictions in this run, never a
   value pulled from test or hard-coded.**
5. Verifies baseline/V3-A/V3-C were scored on the identical set of test
   `sample_id`s.
6. Writes `predictions_{val,test,counterfactual}_{baseline,v3a,v3c}.jsonl`
   and `final_evaluation_report.json` / `.csv`.

## Output files (all under `--out-dir`)

- `predictions_val_{baseline,v3a,v3c}.jsonl`, `predictions_test_*.jsonl`,
  `predictions_counterfactual_*.jsonl` — raw per-sample probabilities.
- `final_evaluation_report.json` / `.csv` — ROC-AUC, PR-AUC, F1,
  precision, recall, confusion matrix, threshold, per model.
- `subtype_analysis.json` / `.csv` — per negative subtype (+ positive):
  N, FP count/rate, mean/median probability, % classified positive.
- `counterfactual_analysis.json` / `.csv` +
  `counterfactual_pairs_{v3a,v3c}.csv` — per transform: N, mean/median
  original vs. counterfactual probability, mean/median change, fraction
  moving in the expected (negative) direction.
- `recurrence_analysis.json` / `.csv` — per `n_occ_intended` bucket
  (2/3/4), positives only: N, mean/median probability, recall, plus a
  descriptive-only monotonicity flag and Spearman rho.
- `error_analysis.json` — FP share by negative subtype, FN share by
  recurrence bucket, per model.
- `figures/fig1_overall_comparison.png` … `fig5_error_breakdown.png`.

## Known, already-handled data quirk

`counterfactual_eval.jsonl` uses `label` on `role="base"` rows and
`expected_label` on `role="transformed"` rows (the latter is what a
correctly-behaving model *should* move toward, not a training label).
`dataset.py`'s `validate_record()` would reject transformed rows outright
(missing `"label"`). `common.load_counterfactual()` handles this
explicitly with its own minimal structural check instead of routing
these rows through that validator. Verified against the real file: 0
rows with negative or non-monotonic `delta_hours`, 300/300 base rows
have `label`, 900/900 transformed rows have `expected_label` — the file
itself is well-formed, this is purely a schema difference to account for
in the reader.
