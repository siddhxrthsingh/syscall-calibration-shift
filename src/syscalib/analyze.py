"""Aggregate the per-batch CSV into seed-level summaries (seed = unit of analysis)."""
import numpy as np, pandas as pd
from scipy import stats

WINDOWS = {"pre": (1, 10), "early_post": (11, 15), "late_post": (26, 30)}
METRICS = ["fpr", "rec_rare", "rec_pattern", "rec_mimic", "rec_macro", "threshold"]


def load(path):
    df = pd.read_csv(path)
    df["thr_minus_oracle"] = df["threshold"] - df["oracle_threshold"]
    return df


def window_means(df):
    """Mean over the batches of each window, separately for every seed/cell."""
    out = []
    for name, (a, b) in WINDOWS.items():
        d = df[(df.batch >= a) & (df.batch <= b)]
        g = d.groupby(["seed", "workload", "eps", "strategy"])[METRICS + ["thr_minus_oracle", "realized_contam"]].mean().reset_index()
        g["window"] = name
        out.append(g)
    return pd.concat(out, ignore_index=True)


def add_oracle_gaps(w):
    o = w[w.strategy == "oracle"].set_index(["seed", "workload", "eps", "window"])
    w = w.copy()
    key = pd.MultiIndex.from_frame(w[["seed", "workload", "eps", "window"]])
    w["excess_fpr"] = w["fpr"].values - o["fpr"].reindex(key).values
    w["recall_deficit"] = o["rec_macro"].reindex(key).values - w["rec_macro"].values
    return w


def ci_summary(x):
    x = np.asarray(x, float)
    n = len(x)
    m, sd = x.mean(), x.std(ddof=1) if n > 1 else np.nan
    h = stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n) if n > 1 else np.nan
    return pd.Series({"n": n, "mean": m, "sd": sd, "lo": m - h, "hi": m + h})


def across_seeds(w, value, by=("workload", "eps", "strategy", "window")):
    rows = []
    for key, g in w.groupby(list(by)):
        s = ci_summary(g[value].values)
        rows.append({**dict(zip(by, key)), **s.to_dict()})
    return pd.DataFrame(rows)


def paired_diff(w, value, a, b, by=("workload", "eps", "window")):
    """Mean over seeds of (strategy a - strategy b), same seed/same streams."""
    p = w.pivot_table(index=["seed", *by], columns="strategy", values=value)
    d = (p[a] - p[b]).rename("diff").reset_index()
    return across_seeds(d.assign(strategy=f"{a}-{b}"), "diff", by=(*by, "strategy"))


def load_roc(path):
    return pd.read_csv(path)


def add_matched_gap(w, roc):
    """Recall deficit versus the oracle operated at the SAME benign FPR.

    For each (seed, workload, window) the oracle's (FPR, macro-recall) curve over many
    threshold levels is averaged over the window's batches; the oracle recall at a
    strategy's window-mean FPR is read off by linear interpolation (clamped at the ends
    of the curve). gap > 0: the strategy detects less than an oracle with equal FPR."""
    curves = {}
    for name, (a, b) in WINDOWS.items():
        d = roc[(roc.batch >= a) & (roc.batch <= b)]
        g = d.groupby(["seed", "workload", "level"])[["fpr", "rec_macro"]].mean().reset_index()
        for (s, wl), gg in g.groupby(["seed", "workload"]):
            gg = gg.sort_values("fpr")
            curves[(s, wl, name)] = (gg.fpr.values, gg.rec_macro.values)
    w = w.copy()
    ref = np.empty(len(w))
    for i, (s, wl, win, f) in enumerate(zip(w.seed, w.workload, w.window, w.fpr)):
        x, y = curves[(s, wl, win)]
        ref[i] = np.interp(f, x, y)
    w["oracle_recall_at_fpr"] = ref
    w["matched_gap"] = ref - w["rec_macro"].values
    return w
