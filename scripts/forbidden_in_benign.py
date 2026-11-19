"""Post-hoc diagnostic (not in ANALYSIS_PLAN.md; not part of the experiment).

How often do transitions that are 'forbidden' by the training-profile definition (max_p A_p(a,b) < 1e-3)
occur in BENIGN evaluation traces, before and after the shift? Uses the unmodified syscalib code and the
exact evaluation-benign streams of the experiment: seeds 1000-1029, evaluation batch 1 (pre-shift) and
evaluation batch 26 (post-shift). Pre-shift batches are identical across workloads (common random numbers),
so batch 1 is computed once (stationary).

Usage (from the repository root):
    python scripts/forbidden_in_benign.py --out results_repro/forbidden_in_benign_per_seed.csv
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
from syscalib.config import load_config
from syscalib.generator import World
from syscalib.attacks import AttackDesign
from syscalib.rng import stream

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=str(ROOT / "config" / "default.json"))
ap.add_argument("--out", required=True)
a = ap.parse_args()

cfg = load_config(a.config)
n = cfg["stream"]["eval_benign"]
rows = []
for s in range(1000, 1030):
    w = World(cfg, s)
    d = AttackDesign(w, cfg["attacks"])
    for wl in ("stationary", "moderate", "severe"):
        for j, ph in ((1, "pre"), (26, "post")):
            if ph == "pre" and wl != "stationary":
                continue
            x = w.sample(w.level_for(wl, ph), n, stream(s, "eval_benign", j))
            c = d.forb[x[:, :-1], x[:, 1:]].sum(axis=1)
            rows.append(dict(seed=s, workload=wl, phase=ph, mean_forb=c.mean(), frac_any=(c > 0).mean()))
df = pd.DataFrame(rows)
print(df.groupby(["phase", "workload"])[["mean_forb", "frac_any"]].agg(["mean", "std"]).round(4).to_string())
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
df.to_csv(a.out, index=False)
print("wrote", a.out)
