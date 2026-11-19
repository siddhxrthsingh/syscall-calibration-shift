# Pre-specified analysis plan for the preliminary run (written BEFORE the run)

Status: **preliminary experiment**, not the final confirmatory study.

* Development seeds: 0-9 (generator, shift strength and attack difficulty tuned here; never reported as results).
* Preliminary run seeds: 1000-1029 (30 independent worlds). Config frozen in `config/default.json`; git-style
  hash of the config is stored next to the results (`*.meta.json`).
* Timeline (fixed a priori, not selected from a pilot): 10 pre-shift + 20 post-shift batches of 500 calibration
  traces; the shift takes effect at evaluation/calibration batch 11. Threshold computed from calibration batch k is
  used on evaluation batch k+1 (T[0] = T0 for evaluation batch 1).
* Summary windows (per seed, then aggregated over seeds): pre = batches 1-10; early-post = 11-15; late-post = 26-30.
* Primary metrics: benign FPR (nominal 0.05) and macro-average recall over the three attack families;
  secondary: per-family recall, threshold displacement from the oracle, excess FPR vs oracle, recall deficit vs oracle.
* Unit of analysis: seed. Reported: mean, SD over seeds, and 95% t-interval of the mean (n=30). Paired differences
  (same seed, same streams) for strategy contrasts. No null-hypothesis significance tests are reported for this
  preliminary run; intervals are descriptive.
* Nothing else is computed from the run before these summaries; no seeds are dropped or re-run.
