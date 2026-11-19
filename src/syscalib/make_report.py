"""Build all tables (CSV + LaTeX fragments) and figures from saved experiment output only."""
import sys, json, argparse
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .analyze import (load, load_roc, window_means, add_oracle_gaps, add_matched_gap,
                      across_seeds, ci_summary, WINDOWS)

STRATS = ["fixed", "quantile", "oracle", "bacr_orig", "bacr_budget"]
LABEL = {"fixed": "Fixed", "quantile": "Quantile", "oracle": "Oracle", "bacr_orig": "BACR (original)",
         "bacr_budget": "BACR-B (budgeted)"}
STYLE = {  # colour (validated palette), linestyle, marker
    "fixed": ("#2a78d6", "-", "o"), "quantile": ("#eb6834", "--", "s"), "oracle": ("#222222", ":", None),
    "bacr_orig": ("#6250d6", "-.", "^"), "bacr_budget": ("#1baf7a", "-", "D")}
WL = ["stationary", "moderate", "severe"]
WL_T = {"stationary": "Stationary", "moderate": "Moderate shift", "severe": "Severe shift"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "legend.frameon": False,
                     "savefig.dpi": 300, "font.family": "DejaVu Sans"})


def save(fig, figdir, name):
    fig.savefig(figdir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(figdir / f"{name}.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


def fmt(m, s, d=3):
    return f"{m:.{d}f} ({s:.{d}f})"


# ---------------------------------------------------------------- figures
def fig_trajectories(df, figdir, n_pre):
    g = df[df.eps == 0].groupby(["workload", "strategy", "batch"])[["fpr", "threshold"]].mean().reset_index()
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True)
    for c, wl in enumerate(WL):
        for st in STRATS:
            col, ls, mk = STYLE[st]
            d = g[(g.workload == wl) & (g.strategy == st)]
            kw = dict(color=col, ls=ls, lw=1.8 if st == "oracle" else 1.4,
                      marker=mk, ms=3, markevery=3, label=LABEL[st])
            ax[0, c].plot(d.batch, d.fpr, **kw)
            ax[1, c].plot(d.batch, d.threshold, **kw)
        for r in (0, 1):
            ax[r, c].axvline(n_pre + 0.5, color=MUTED, lw=0.8, ls=(0, (1, 2)))
        ax[0, c].axhline(0.05, color=MUTED, lw=0.6)
        ax[0, c].set_title(WL_T[wl], fontsize=9, color=INK)
        ax[1, c].set_xlabel("evaluation batch")
        ax[0, c].set_ylim(-0.02, 1.02)
    ax[0, 0].set_ylabel("benign FPR")
    ax[1, 0].set_ylabel("threshold (mean NLL, nats)")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.04), fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, figdir, "fig_trajectories")


def fig_ratchet(df, figdir, alpha=0.95):
    d = df[(df.eps == 0) & (df.workload == "stationary")].groupby(["strategy", "batch"]).fpr.mean().reset_index()
    fig, ax = plt.subplots(figsize=(3.4, 2.9))
    j = np.arange(1, 31)
    ax.plot(j, 1 - alpha ** j, color=MUTED, lw=1.0, ls=(0, (4, 2)), label=r"theory $1-\alpha^{j}$ (no clipping)")
    for st in ("bacr_orig", "bacr_budget", "quantile"):
        col, ls, mk = STYLE[st]
        x = d[d.strategy == st]
        ax.plot(x.batch, x.fpr, color=col, ls=ls, lw=1.4, marker=mk, ms=3, markevery=3, label=LABEL[st])
    ax.axhline(0.05, color=MUTED, lw=0.6)
    ax.set_xlabel("evaluation batch $j$"); ax.set_ylabel("benign FPR"); ax.set_ylim(-0.02, 1.0)
    ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, columnspacing=1.0, handlelength=2.2)
    save(fig, figdir, "fig_bacr_ratchet")


def fig_contamination(w, figdir):
    lp = w[w.window == "late_post"]
    fig, ax = plt.subplots(2, 3, figsize=(7.2, 4.4), sharex=True)
    for r, (val, yl) in enumerate([("fpr", "benign FPR (late post)"), ("rec_macro", "macro recall (late post)")]):
        for c, wl in enumerate(WL):
            for st in ("fixed", "quantile", "bacr_budget", "oracle"):
                col, ls, mk = STYLE[st]
                s = across_seeds(lp[(lp.workload == wl) & (lp.strategy == st)], val, by=("eps",))
                x = s.eps * 100
                ax[r, c].errorbar(x, s["mean"], yerr=[s["mean"] - s.lo, s.hi - s["mean"]], color=col, ls=ls, lw=1.4,
                                  marker=mk, ms=3.5, capsize=1.5, elinewidth=0.7, label=LABEL[st])
            if r == 0:
                ax[r, c].set_title(WL_T[wl], fontsize=9, color=INK)
                ax[r, c].axhline(0.05, color=MUTED, lw=0.6)
            if r == 1:
                ax[r, c].set_xlabel("calibration contamination (%)")
        ax[r, 0].set_ylabel(yl)
    ax[0, 0].set_ylim(-0.01, 0.35); ax[0, 1].set_ylim(-0.01, 0.35); ax[0, 2].set_ylim(-0.01, 0.75)
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.04), fontsize=8)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    save(fig, figdir, "fig_contamination")


def fig_operating_points(w, roc, figdir):
    lp = w[w.window == "late_post"]
    rc = roc[(roc.batch >= 26)].groupby(["seed", "workload", "level"])[["fpr", "rec_macro"]].mean().reset_index()
    rc = rc.groupby(["workload", "level"])[["fpr", "rec_macro"]].mean().reset_index().sort_values("fpr")
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    for c, wl in enumerate(WL):
        d = rc[rc.workload == wl]
        ax[c].plot(d.fpr, d.rec_macro, color="#999999", lw=2.2, alpha=0.8, label="oracle ROC (all levels)", zorder=1)
        for st in ("fixed", "quantile", "bacr_budget", "bacr_orig"):
            col, ls, mk = STYLE[st]
            s = across_seeds(lp[(lp.workload == wl) & (lp.strategy == st)], "fpr", by=("eps",))
            t = across_seeds(lp[(lp.workload == wl) & (lp.strategy == st)], "rec_macro", by=("eps",))
            ax[c].scatter(s["mean"], t["mean"], s=14 + 70 * s.eps.values / 0.1 * 0.0 + 18, color=col, marker=mk,
                          zorder=3, label=LABEL[st], edgecolor="white", linewidth=0.5)
        o = across_seeds(lp[(lp.workload == wl) & (lp.strategy == "oracle")], "fpr", by=("eps",))["mean"].mean()
        orr = across_seeds(lp[(lp.workload == wl) & (lp.strategy == "oracle")], "rec_macro", by=("eps",))["mean"].mean()
        ax[c].scatter([o], [orr], color="#222222", marker="*", s=60, zorder=4, label="Oracle ($\\alpha$=0.95)")
        ax[c].set_title(WL_T[wl], fontsize=9, color=INK)
        ax[c].set_xlabel("benign FPR"); ax[c].set_xlim(-0.02, 1.02); ax[c].set_ylim(0, 1.03)
    ax[0].set_ylabel("macro recall")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.1), fontsize=7.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, figdir, "fig_operating_points")


# ---------------------------------------------------------------- tables
def table_main(w, out):
    rows = []
    for wl in WL:
        for st in STRATS:
            r = {"workload": wl, "strategy": st}
            for win, key in (("early_post", "fpr_early"), ("late_post", "fpr_late")):
                s = across_seeds(w[(w.workload == wl) & (w.strategy == st) & (w.eps == 0) & (w.window == win)], "fpr", by=("eps",)).iloc[0]
                r[key], r[key + "_sd"] = s["mean"], s["sd"]
            for val in ("rec_macro", "thr_minus_oracle"):
                s = across_seeds(w[(w.workload == wl) & (w.strategy == st) & (w.eps == 0) & (w.window == "late_post")], val, by=("eps",)).iloc[0]
                r[val], r[val + "_sd"] = s["mean"], s["sd"]
            rows.append(r)
    t = pd.DataFrame(rows); t.to_csv(out / "table_main_eps0.csv", index=False)
    return t


def table_family(w, out):
    rows = []
    for wl in WL:
        for fam, name in (("rec_rare", "Rare-transition"), ("rec_pattern", "Sequence-pattern"), ("rec_mimic", "Benign-mimicking")):
            r = {"workload": wl, "family": name}
            for st in ("oracle", "fixed", "quantile", "bacr_budget"):
                d = w[(w.workload == wl) & (w.strategy == st) & (w.eps == 0) & (w.window == "late_post")]
                s = ci_summary(d[fam].values)
                r[st], r[st + "_sd"] = s["mean"], s["sd"]
            rows.append(r)
    t = pd.DataFrame(rows); t.to_csv(out / "table_family_eps0.csv", index=False)
    return t


def table_contam(w, out):
    rows = []
    lp = w[w.window == "late_post"]
    for wl in WL:
        for eps in sorted(lp.eps.unique()):
            r = {"workload": wl, "eps": eps}
            for st in ("quantile", "bacr_budget", "oracle"):
                d = lp[(lp.workload == wl) & (lp.strategy == st) & (lp.eps == eps)]
                for val in ("fpr", "rec_macro", "matched_gap"):
                    s = ci_summary(d[val].values)
                    r[f"{st}_{val}"], r[f"{st}_{val}_sd"] = s["mean"], s["sd"]
                    r[f"{st}_{val}_lo"], r[f"{st}_{val}_hi"] = s["lo"], s["hi"]
            rows.append(r)
    t = pd.DataFrame(rows); t.to_csv(out / "table_contamination_late_post.csv", index=False)
    return t


def table_ratchet(df, out, alpha=0.95):
    d = df[(df.eps == 0) & (df.workload == "stationary") & (df.batch.isin([1, 5, 10, 15, 20, 30]))]
    rows = []
    for j in (1, 5, 10, 15, 20, 30):
        r = {"batch": j, "theory_no_clip": 1 - alpha ** j}
        for st in ("bacr_orig", "bacr_budget", "quantile"):
            x = d[(d.batch == j) & (d.strategy == st)].fpr
            r[st], r[st + "_sd"] = x.mean(), x.std(ddof=1)
        rows.append(r)
    t = pd.DataFrame(rows); t.to_csv(out / "table_bacr_ratchet.csv", index=False)
    return t


def table_interaction(w, out):
    """Exploratory contrast: absolute macro-recall loss of ordinary quantile recalibration when contamination goes 0 -> 5%,
    shifted minus stationary (paired by seed)."""
    lp = w[(w.window == "late_post") & (w.strategy == "quantile")]
    q = lp.pivot_table(index=["seed", "workload"], columns="eps", values="rec_macro")
    out_rows = []
    for e in (0.01, 0.025, 0.05, 0.10):
        loss = (q[0.0] - q[e]).unstack()
        rel = ((q[0.0] - q[e]) / q[0.0]).unstack()
        for wl in WL:
            s = ci_summary(loss[wl].values)
            r = {"eps": e, "workload": wl, "abs_loss": s["mean"], "abs_loss_sd": s["sd"], "rel_loss": rel[wl].mean()}
            if wl != "stationary":
                d = ci_summary((loss[wl] - loss["stationary"]).values)
                r.update({"did": d["mean"], "did_lo": d["lo"], "did_hi": d["hi"]})
            out_rows.append(r)
    t = pd.DataFrame(out_rows); t.to_csv(out / "table_interaction_exploratory.csv", index=False)
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="path stem, e.g. results/prelim_seeds1000-1029")
    ap.add_argument("--outdir", default="results/report")
    ap.add_argument("--figdir", default="paper/figs")
    a = ap.parse_args()
    out, figdir = Path(a.outdir), Path(a.figdir)
    out.mkdir(parents=True, exist_ok=True); figdir.mkdir(parents=True, exist_ok=True)
    df = load(a.results + ".csv")
    roc = load_roc(a.results + "_oracle_roc.csv")
    meta = json.loads(Path(a.results + ".meta.json").read_text())
    n_pre = meta["config"]["stream"]["n_pre"]
    w = add_matched_gap(add_oracle_gaps(window_means(df)), roc)
    w.to_csv(out / "seed_window_means.csv", index=False)
    fig_trajectories(df, figdir, n_pre); fig_ratchet(df, figdir); fig_contamination(w, figdir)
    fig_operating_points(w, roc, figdir)
    tabs = dict(main=table_main(w, out), family=table_family(w, out), contam=table_contam(w, out),
                ratchet=table_ratchet(df, out), inter=table_interaction(w, out))
    # extra facts used in the text
    facts = {}
    pre = w[(w.window == "pre") & (w.eps == 0)]
    facts["n_seeds"] = int(df.seed.nunique())
    facts["bacr_orig_pre_fpr"] = ci_summary(pre[pre.strategy == "bacr_orig"].groupby("seed").fpr.mean()).to_dict()
    facts["realized_contam"] = df[df.strategy == "fixed"].groupby("eps").realized_contam.agg(["mean", "std", "min", "max"]).to_dict()
    ev = df[(df.strategy == "fixed") & (df.eps == 0)].groupby(["workload", "phase"]).benign_mean_score.mean().to_dict()
    facts["benign_mean_score"] = {f"{k[0]}|{k[1]}": v for k, v in ev.items()}
    s0 = df[(df.strategy == "oracle") & (df.eps == 0)]
    facts["oracle_recall_by_family_late"] = {
        wl: {f: float(s0[(s0.workload == wl) & (s0.batch >= 26)][f].mean()) for f in ("rec_rare", "rec_pattern", "rec_mimic")} for wl in WL}
    (out / "facts.json").write_text(json.dumps(facts, indent=1, default=float))
    print("report written to", out, "and figures to", figdir)


if __name__ == "__main__":
    main()
