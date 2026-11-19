"""Development helper: print generator/attack/shift diagnostics for given seeds (used for tuning on dev seeds only)."""
import sys, numpy as np
sys.path.insert(0, "src")
from syscalib.config import load_config
from syscalib.experiment import run_seed
import pandas as pd
cfg = load_config()
seeds = [int(s) for s in sys.argv[1].split(",")] if len(sys.argv) > 1 else cfg["seeds"]["dev"][:4]
df = pd.concat([pd.DataFrame(run_seed(cfg, s), columns=__import__("syscalib.experiment", fromlist=["COLUMNS"]).COLUMNS) for s in seeds])
pd.set_option("display.width", 200)
c0 = df[df.eps == 0]
print("== benign mean score by workload/phase"); print(c0[c0.strategy == "fixed"].groupby(["workload", "phase"]).benign_mean_score.mean())
print("== FPR (eps=0) by strategy/workload/phase"); print(c0.groupby(["workload", "phase", "strategy"]).fpr.mean().unstack())
o = c0[c0.strategy == "oracle"]
print("== oracle recall by workload/phase"); print(o.groupby(["workload", "phase"])[["rec_rare", "rec_pattern", "rec_mimic", "fpr"]].mean())
print("== per-seed oracle recall, stationary pre"); print(o[(o.workload=="stationary")&(o.phase=="pre")].groupby("seed")[["rec_rare","rec_pattern","rec_mimic","fpr"]].mean())
