"""One seed = one independent world. Produces tidy per-batch records for every
(workload x contamination x strategy) cell, all computed from the SAME scores.

Dataset separation (each has its own keyed random stream, see rng.py):
  train | init_cal | cal_benign + cal_attack (online stream) | eval_benign |
  eval_attack (per family) | oracle reference
Thresholds are computed from calibration scores only; evaluation scores never
enter any threshold function.
"""
import numpy as np
from .generator import World
from .attacks import AttackDesign, FAMILIES, apply_family
from .detector import MarkovDetector
from .rng import stream
from . import thresholds as TH

WORKLOADS = ("stationary", "moderate", "severe")
STRATEGY_ORDER = ("fixed", "quantile", "oracle", "bacr_orig", "bacr_budget")


def _phase(j, n_pre):
    return "pre" if j <= n_pre else "post"


ROC_LEVELS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.926, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999)


def run_seed(cfg, seed, workloads=WORKLOADS, roc=False):
    sc, alpha = cfg["stream"], cfg["alpha"]
    n_pre, K = sc["n_pre"], sc["n_pre"] + sc["n_post"]
    bs, n_ev, n_att = sc["batch_size"], sc["eval_benign"], sc["eval_attack_per_family"]
    world = World(cfg, seed)
    design = AttackDesign(world, cfg["attacks"])
    pre = world.level_for("stationary", "pre")

    # ---- 1. train the detector (frozen afterwards) ---------------------------
    det = MarkovDetector(world.V, cfg["detector"]["kappa"]).fit(
        world.sample(pre, cfg["detector"]["n_train"], stream(seed, "train")))

    # ---- 2. initial clean calibration -> T0, delta ------------------------------
    S_init = det.score(world.sample(pre, sc["n_init_cal"], stream(seed, "init_cal")))
    T0 = TH.q(S_init, alpha)
    delta = cfg["bacr"]["delta_sigma"] * float(S_init.std(ddof=1))
    ref_pre = det.score(world.sample(pre, sc["oracle_ref"], stream(seed, "oracle", 0)))
    oracle_pre = TH.q(ref_pre, alpha)

    pre_cache = {}
    rows, roc_rows = [], []
    for wi, workload in enumerate(workloads):
        post_level = world.level_for(workload, "post")
        ref_post = det.score(world.sample(post_level, sc["oracle_ref"], stream(seed, "oracle", 1, wi)))
        oracle_post = TH.q(ref_post, alpha)

        # ---- 3. generate + score all batches once (shared by every strategy) ------
        Sb, Sa, U, Eb, Ea = {}, {}, {}, {}, {}
        for j in range(1, K + 1):
            ph = _phase(j, n_pre)
            if ph == "pre" and j in pre_cache:
                Sb[j], Sa[j], U[j], Eb[j], Ea[j] = pre_cache[j]
                continue
            level = pre if ph == "pre" else post_level
            # online calibration stream: benign slots + one potential contaminant per slot
            xb = world.sample(level, bs, stream(seed, "cal_benign", j))
            xa = world.sample(level, bs, stream(seed, "cal_attack_base", j))
            fam = stream(seed, "cal_family", j).integers(0, len(FAMILIES), size=bs)
            for fi, f in enumerate(FAMILIES):
                idx = np.flatnonzero(fam == fi)
                if idx.size:
                    xa[idx] = apply_family(design, f, xa[idx], stream(seed, "cal_attack_mod", j, fi))
            u = stream(seed, "cal_uniform", j).random(bs)
            # independent evaluation traces
            eb = det.score(world.sample(level, n_ev, stream(seed, "eval_benign", j)))
            ea = []
            for fi, f in enumerate(FAMILIES):
                base = world.sample(level, n_att, stream(seed, "eval_attack_base", j, fi))
                ea.append(det.score(apply_family(design, f, base, stream(seed, "eval_attack_mod", j, fi))))
            out = (det.score(xb), det.score(xa), u, eb, np.stack(ea))
            Sb[j], Sa[j], U[j], Eb[j], Ea[j] = out
            if ph == "pre":
                pre_cache[j] = out

        oracle_T = np.array([oracle_pre if _phase(j, n_pre) == "pre" else oracle_post for j in range(1, K + 1)])
        ben_mean = np.array([Eb[j].mean() for j in range(1, K + 1)])

        if roc:  # reference ROC of oracle thresholds at many levels (clean per-phase reference)
            for lv in ROC_LEVELS:
                t_pre, t_post = TH.q(ref_pre, lv), TH.q(ref_post, lv)
                for j in range(1, K + 1):
                    thr = t_pre if _phase(j, n_pre) == "pre" else t_post
                    rec = (Ea[j] > thr).mean(axis=1)
                    roc_rows.append((seed, workload, lv, j, float((Eb[j] > thr).mean()), float(rec.mean())))

        for eps in cfg["contamination"]:
            masks = {j: U[j] < eps for j in range(1, K + 1)}      # nested in eps via common U
            cal = {j: np.where(masks[j], Sa[j], Sb[j]) for j in range(1, K + 1)}
            realized = np.array([masks[j].mean() for j in range(1, K + 1)])
            cal_batches = [cal[j] for j in range(1, K)]            # batch K is never used
            kw = dict(alpha=alpha, delta=delta, eps_budget=cfg["bacr"]["eps_budget"])
            trajs = {name: TH.STRATEGIES[name](T0, cal_batches, **kw) for name in TH.STRATEGIES}
            trajs["oracle"] = np.concatenate([[oracle_T[0]], oracle_T[:-1]])  # placeholder, replaced below
            for name in STRATEGY_ORDER:
                for j in range(1, K + 1):
                    thr = oracle_T[j - 1] if name == "oracle" else trajs[name][j - 1]
                    fpr = float((Eb[j] > thr).mean())
                    rec = (Ea[j] > thr).mean(axis=1)
                    rows.append((seed, workload, eps, name, j, _phase(j, n_pre), float(thr), fpr,
                                 float(rec[0]), float(rec[1]), float(rec[2]), float(rec.mean()),
                                 float(realized[j - 1]), float(ben_mean[j - 1]), float(oracle_T[j - 1])))
    return (rows, roc_rows) if roc else rows


COLUMNS = ["seed", "workload", "eps", "strategy", "batch", "phase", "threshold", "fpr",
           "rec_rare", "rec_pattern", "rec_mimic", "rec_macro", "realized_contam",
           "benign_mean_score", "oracle_threshold"]

ROC_COLUMNS = ["seed", "workload", "level", "batch", "fpr", "rec_macro"]
