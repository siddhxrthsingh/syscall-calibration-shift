import numpy as np, pandas as pd
from syscalib.experiment import run_seed, COLUMNS


def _df(small_cfg, seed=11):
    return pd.DataFrame(run_seed(small_cfg, seed), columns=COLUMNS)


def test_row_count_and_batch_layout(small_cfg):
    df = _df(small_cfg)
    K = small_cfg["stream"]["n_pre"] + small_cfg["stream"]["n_post"]
    assert len(df) == 3 * len(small_cfg["contamination"]) * 5 * K
    assert set(df[df.batch <= 3].phase) == {"pre"} and set(df[df.batch > 3].phase) == {"post"}


def test_first_batch_uses_initial_threshold_for_all_non_oracle(small_cfg):
    df = _df(small_cfg)
    b1 = df[(df.batch == 1) & (df.strategy != "oracle")]
    assert b1.groupby(["workload", "eps"]).threshold.nunique().max() == 1


def test_evaluation_independent_of_contamination_for_fixed_and_oracle(small_cfg):
    """Evaluation data and oracle never see calibration streams."""
    df = _df(small_cfg)
    for s in ("fixed", "oracle"):
        d = df[df.strategy == s]
        piv = d.pivot_table(index=["workload", "batch"], columns="eps", values="fpr")
        assert np.allclose(piv.values, piv.values[:, [0]])


def test_pre_shift_rows_identical_across_workloads(small_cfg):
    df = _df(small_cfg)
    pre = df[df.phase == "pre"]
    piv = pre.pivot_table(index=["eps", "strategy", "batch"], columns="workload", values="fpr")
    assert np.allclose(piv["stationary"], piv["moderate"]) and np.allclose(piv["stationary"], piv["severe"])


def test_contamination_nested_and_realised_rate(small_cfg):
    df = _df(small_cfg)
    r = df[(df.strategy == "fixed") & (df.workload == "stationary")].pivot_table(
        index="batch", columns="eps", values="realized_contam")
    assert (r.diff(axis=1).iloc[:, 1:] >= 0).all().all()            # nested: higher eps => superset
    assert (r[0.0] == 0).all()


def test_threshold_at_batch_j_uses_only_earlier_calibration(small_cfg):
    """With an unreachable eps, changing later batches must not change earlier thresholds."""
    import copy
    c2 = copy.deepcopy(small_cfg)
    c2["stream"]["n_post"] = 6                                      # extend horizon only
    a = pd.DataFrame(run_seed(small_cfg, 12), columns=COLUMNS)
    b = pd.DataFrame(run_seed(c2, 12), columns=COLUMNS)
    K = small_cfg["stream"]["n_pre"] + small_cfg["stream"]["n_post"]
    for w in ("stationary",):
        for s in ("quantile", "bacr_orig", "bacr_budget"):
            ta = a[(a.workload == w) & (a.eps == 0.05) & (a.strategy == s)].set_index("batch").threshold
            tb = b[(b.workload == w) & (b.eps == 0.05) & (b.strategy == s)].set_index("batch").threshold
            assert np.allclose(ta.loc[1:K], tb.loc[1:K])


def test_seed_streams_do_not_overlap():
    from syscalib.rng import stream, STREAMS
    firsts = {n: stream(0, n, 1, 0).random() for n in STREAMS}
    assert len(set(firsts.values())) == len(STREAMS)
