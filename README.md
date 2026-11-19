# Calibration Under Workload Shift in System-Call Anomaly Detection

Simulation code, configuration, tests and saved results for the paper

> S. Singh and S. Maslekar, *Calibration Under Workload Shift in System-Call Anomaly Detection: A Simulation
> Study of False-Alarm Control and Detection Sensitivity*, PES University, Bengaluru, India (manuscript, 2026).

## Overview

A system-call anomaly detector raises an alarm when an anomaly score exceeds a threshold. That threshold is
usually set once, as a high quantile of benign calibration scores. This repository measures, in a controlled
and fully seeded simulation, how two things affect the detector's **false-positive rate (FPR)** and **attack
recall**:

- **benign workload shift**: legitimate behaviour changes after calibration;
- **calibration contamination**: attack traces leak into the unlabeled recalibration data.

The comparison covers five threshold strategies:

| Strategy | Code name | Description |
|---|---|---|
| Fixed | `fixed` | `T_k = T_0` |
| Quantile | `quantile` | `T_k` = empirical α-quantile of calibration batch *k* |
| Oracle | `oracle` | α-quantile of 10 000 clean benign traces from the current phase (privileged, not deployable) |
| BACR (original) | `bacr_orig` | screen the batch at the previous threshold, re-estimate, clip the step to ±δ |
| BACR-B (exploratory) | `bacr_budget` | no screen; quantile at level α(1−ε̄), clip to ±δ |

The design crosses 3 workloads × 5 contamination levels (0, 1, 2.5, 5, 10 %) × 5 strategies, over 30
independent synthetic worlds (seeds 1000–1029). The detector is a frozen first-order Markov model with
additive smoothing. Everything is synthetic; see [Limitations](#limitations).

## Repository layout

| Path | Contents |
|---|---|
| `src/syscalib/generator.py` | Regime-switching benign trace generator (profiles, drift targets, novel profile, workload levels) |
| `src/syscalib/attacks.py` | Attack design: forbidden-transition definition and the rare, pattern and low-intensity families |
| `src/syscalib/detector.py` | Frozen first-order Markov scorer (smoothing κ, mean negative log-likelihood) |
| `src/syscalib/thresholds.py` | Threshold strategies: fixed, quantile, oracle, BACR, BACR-B |
| `src/syscalib/experiment.py` | One world (seed): training, calibration/evaluation timeline, contamination, all strategies |
| `src/syscalib/run_experiment.py` | Command-line runner (multiprocessing over seeds) |
| `src/syscalib/analyze.py` | Window means, oracle gaps, matched-FPR recall gap, summaries |
| `src/syscalib/make_report.py` | Command-line report: tables (CSV), `facts.json` and figures |
| `src/syscalib/rng.py` | Independent random streams keyed by (seed, stream, batch, sub-stream) |
| `src/syscalib/config.py` | Loads `config/default.json` |
| `config/default.json` | The frozen configuration used for the reported run |
| `tests/` | 25 automated tests (pytest) |
| `ANALYSIS_PLAN.md` | Analysis plan written before the reported run |
| `results/` | Saved results of the reported run (seeds 1000–1029), report tables, figures and diagnostic |
| `scripts/make_figures_type42.py` | Runs the unmodified report with TrueType (Type 42) PDF fonts, as used for the paper figures |
| `scripts/forbidden_in_benign.py` | Post-hoc diagnostic reported in the paper (forbidden transitions in benign traces) |
| `diagnose.py` | Development helper used only for tuning on development seeds 0–9 |
| `requirements.txt` | Dependencies (unpinned) |
| `requirements-tested.txt` | Exact package versions used in the reference Linux reproduction environment |
| `pytest.ini` | pytest settings (`pythonpath = src`, `testpaths = tests`) |

**Naming note.** The code calls the low-intensity attack family `mimic`: the config key is
`attacks.mimic_m` and the result column is `rec_mimic`. In the paper this family is called
**low-intensity**. It inserts only 2 forbidden transitions and is a weak version of the rare-transition
attack. **It is not a mimicry attack.** The identifier was kept so that the verified code and the saved
results stay byte-identical.

## Software requirements

- Python 3.10 or newer (verified with Python 3.13.18)
- NumPy, SciPy, pandas, Matplotlib and pytest (`requirements.txt`)
- No GPU, no internet access at run time, and no other external data

## Installation and environment setup

**Linux / macOS (bash):**

```bash
git clone https://github.com/siddhxrthsingh/syscall-calibration-shift
cd syscall-calibration-shift
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # or: pip install -r requirements-tested.txt
export PYTHONPATH=src
```

**Windows (PowerShell):**

```powershell
git clone https://github.com/siddhxrthsingh/syscall-calibration-shift
cd syscall-calibration-shift
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt          # or: pip install -r requirements-tested.txt
$env:PYTHONPATH = "src"
```

(Windows cmd.exe: `set PYTHONPATH=src`.) The commands below are written for bash. In PowerShell, use
`python` instead of `python3`; forward slashes in paths work as written.

`PYTHONPATH=src` is needed for `python -m syscalib...`. pytest finds `src` through `pytest.ini`, and the
scripts in `scripts/` set the path themselves.

## Running the tests

```bash
python3 -m pytest -q
```

This runs 25 tests in about 4 s. They check that:

- the scorer is frozen after training;
- the original BACR never raises its threshold;
- no threshold depends on later batches (no look-ahead);
- the contamination masks are nested across levels;
- the random streams are separate;
- the attacks have the intended structure.

## Reproducing the experiment (seeds 1000–1029)

```bash
python3 -m syscalib.run_experiment --seeds 1000-1029 --out results_repro/rerun_seeds1000-1029.csv --jobs 2
```

- **Arguments.** `--seeds` (required) takes `dev` (seeds 0–9), a range `a-b`, or a comma list such as
  `1,5,9`. `--out` (required) is the result CSV. `--jobs` is the number of worker processes (default 1).
  `--config` defaults to `config/default.json`.
- **Output.** The run writes three files next to `--out`:
  - `<name>.csv`: 67 500 result rows, one per seed × workload × contamination × strategy × batch;
  - `<name>_oracle_roc.csv`: 40 500 oracle ROC rows;
  - `<name>.meta.json`: the seeds, the full configuration and the elapsed time.
- **Runtime.** About 1 minute on 2 worker processes.

Write new runs to a separate directory such as `results_repro/`, as above, so the saved results in
`results/` are not overwritten. `results_repro/` is in `.gitignore`.

**Checking the reproduction:**

The committed CSV files are stored with LF line endings, while the CSV writer may emit CRLF
line endings on some platforms. A raw byte comparison can therefore fail even when every
CSV value is identical. On Linux/macOS, compare after normalising carriage returns:

```bash
cmp <(tr -d '\r' < results_repro/rerun_seeds1000-1029.csv) \
    <(tr -d '\r' < results/rerun_seeds1000-1029.csv)

cmp <(tr -d '\r' < results_repro/rerun_seeds1000-1029_oracle_roc.csv) \
    <(tr -d '\r' < results/rerun_seeds1000-1029_oracle_roc.csv)
```

In PowerShell, compare normalised text rather than raw file hashes:

```powershell
$a = (Get-Content results_repro/rerun_seeds1000-1029.csv -Raw) -replace "`r", ""
$b = (Get-Content results/rerun_seeds1000-1029.csv -Raw) -replace "`r", ""
$a -ceq $b
```

Repeat with the `_oracle_roc.csv` filenames for the oracle-ROC file. The reference Linux
reproduction was byte-identical before repository line-ending normalisation. In the independent
Windows rerun, the main CSV matched within a numerical tolerance of 1e-12, the oracle-ROC CSV
matched exactly, and all seven report outputs passed numerical or semantic comparisons. The
regenerated figures were visually consistent, but their rendered dimensions and PDF bytes were
not identical. Byte-for-byte identity is therefore not a cross-platform guarantee.


## Result analysis, tables and figures

```bash
python3 -m syscalib.make_report --results results_repro/rerun_seeds1000-1029 --outdir results_repro/report --figdir results_repro/figs
```

`--results` is the path **stem**, without `.csv`. It writes the following to `--outdir`:

- `seed_window_means.csv`
- `table_main_eps0.csv`, `table_family_eps0.csv`, `table_contamination_late_post.csv`,
  `table_bacr_ratchet.csv`, `table_interaction_exploratory.csv`
- `facts.json`

It also writes four figures, as PDF and PNG, to `--figdir`. If you omit `--outdir` and `--figdir`, the
defaults are `results/report` and `paper/figs`, which would overwrite the saved report. Always pass both.

The paper's figures use TrueType (Type 42) fonts in the PDFs. To regenerate them that way:

```bash
python3 scripts/make_figures_type42.py --results results_repro/rerun_seeds1000-1029 --outdir results_repro/report --figdir results_repro/figs
```

This produces the same tables and the same PNG figures; only the font embedding in the PDF figures differs.
The paper uses three of the four figures: `fig_trajectories`, `fig_bacr_ratchet` and `fig_operating_points`.

**Post-hoc diagnostic.** The paper also reports forbidden transitions in benign traces. This diagnostic was
defined after seeing the results and is not in `ANALYSIS_PLAN.md`. To regenerate it:

```bash
python3 scripts/forbidden_in_benign.py --out results_repro/forbidden_in_benign_per_seed.csv
```

### Saved results (`results/`)

| File | Contents |
|---|---|
| `rerun_seeds1000-1029.csv`, `_oracle_roc.csv`, `.meta.json` | Raw results of the reported run |
| `report/*.csv`, `report/facts.json` | Tables and quoted values used in the paper |
| `figs/*.pdf`, `figs/*.png` | Figures written by `syscalib.make_report` (the default Type 3 PDF fonts) |
| `diagnostics/forbidden_in_benign_per_seed.csv` | Per-seed values of the post-hoc diagnostic |

The saved run is a re-execution of the original run with the same code, configuration and seeds. The
reference Linux reproduction was byte-identical before Git line-ending normalisation. Cross-platform
comparison details and tolerances are given in the reproduction-check section above.

## Configuration (`config/default.json`)

| Block | Key settings |
|---|---|
| `world` | `V=24` symbols in `G=4` groups; `P=4` profiles; trace length `L=120`; regime switch probability `switch_q=0.04`; Gamma concentrations `sparse_alpha=0.08`, `dom_alpha=2.0`; dominant successors `k_shared=1`, `k_specific=2`; mixture weights `w0=(0.4,0.3,0.2,0.1)`, `w_alt=(0.05,0.15,0.3,0.5)` |
| `detector` | smoothing `kappa=0.1`; `n_train=3000` benign training traces |
| `shift` | stationary (μ,λ,π_new) = (0,0,0); moderate (0.5,0.01,0.02); severe (1.0,0.03,0.06) |
| `attacks` | rare: `rare_m=8` forbidden insertions; pattern: 4-symbol motif × 6; low-intensity: `mimic_m=2`; forbidden if max_p A_p(a,b) < `0.001` |
| `stream` | 10 pre-shift + 20 post-shift batches; 500 calibration traces per batch; 500 initial clean calibration traces; 2000 benign + 300 per attack family per evaluation batch; 10 000 oracle traces |
| `alpha` | 0.95 (nominal FPR 0.05) |
| `contamination` | 0, 0.01, 0.025, 0.05, 0.10 |
| `bacr` | step bound δ = 0.5 σ₀ (`delta_sigma`); BACR-B budget ε̄ = 0.025 (`eps_budget`) |
| `seeds.dev` | 0–9 (tuning only, never reported) |

The analysis windows (pre 1–10, early post-shift 11–15, late post-shift 26–30) are defined in
`src/syscalib/analyze.py` (`WINDOWS`), as specified in `ANALYSIS_PLAN.md`.

**Timeline.** A threshold computed from calibration batch *k* is first used on evaluation batch *k*+1,
and never on the data it was computed from. All empirical quantiles use NumPy's default `linear` method.
Calibration and evaluation streams are keyed by batch, not by workload, so all workloads share common
random numbers.

## Hardware and software environment

The reference Linux reproduction was verified on 19 Nov 2026 in this environment:

- Linux x86_64, 2 vCPU (Intel Xeon @ 2.10 GHz), 7 GB RAM
- Python 3.13.18; NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5, Matplotlib 3.11.2, pytest 9.1.1
- Run time: 58 s for seeds 1000–1029 with `--jobs 2`

The code is CPU-only; `--jobs` sets the number of worker processes.

## Limitations

- **Synthetic only.** The study says nothing direct about real systems, real workloads or real attacks.
  Shift and attack strengths were tuned on development seeds 0–9.
- **One scorer.** Only a first-order Markov scorer was evaluated; higher-order or frequency-based
  representations may behave differently.
- **Approximate shift.** The three shift mechanisms are unvalidated approximations of real workload change.
  Training-defined forbidden transitions also change meaning after the shift.
- **Simple contamination.** Contaminants are an equal three-family mixture injected independently per trace.
  They are not an adaptive or family-specific attacker, and the low-intensity family is not a mimicry attack.
- **Population-level theory.** The analytical results do not explain the finite-sample excess of the
  original BACR.
- **Exploratory analyses.** BACR-B, the matched-FPR comparison, the interaction contrast and the
  forbidden-transition diagnostic were defined after seeing results and need a confirmatory experiment.
- **Dependent comparisons.** Workloads share random numbers, which pairs the comparisons but makes them
  non-independent. The 30 worlds capture variability under one generator only.

## Citation

If you use this code, please cite the manuscript identified above. A formal bibliographic citation
will be added if and when the paper is published.

## Licence

No reuse licence is currently specified. Making this repository public allows people to view its
contents, but does not by itself grant permission to reuse, modify or redistribute the code.
Please contact the authors regarding permissions.
