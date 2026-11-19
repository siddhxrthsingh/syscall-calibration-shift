import numpy as np
from syscalib import thresholds as TH

ALPHA = 0.95


def _stream(rng, K=20, n=2000, shift=0.0, eps=0.0, attack_mu=8.0):
    out = []
    for k in range(K):
        s = rng.normal(shift, 1.0, n)
        if eps > 0:
            m = rng.random(n) < eps
            s[m] = rng.normal(attack_mu, 1.0, m.sum())
        out.append(s)
    return out


def test_original_bacr_can_never_raise_threshold():
    rng = np.random.default_rng(0)
    for shift in (0.0, 3.0, 10.0):
        cal = _stream(rng, shift=shift)
        T = TH.bacr_original(1.645, cal, alpha=ALPHA, delta=0.5)
        assert np.all(np.diff(T) <= 1e-12)


def test_original_bacr_ratchets_down_on_clean_stationary_data():
    """Coverage u_k = alpha^(k+1): FPR_k = 1 - alpha^(k+1) when the clip does not bind."""
    rng = np.random.default_rng(1)
    K = 15
    cal = _stream(rng, K=K, n=200000)
    T0 = float(np.quantile(rng.normal(size=400000), ALPHA))
    T = TH.bacr_original(T0, cal, alpha=ALPHA, delta=10.0)
    from scipy.stats import norm
    fpr = 1 - norm.cdf(T)
    for k in (1, 5, 10, 15):
        assert abs(fpr[k] - (1 - ALPHA ** (k + 1))) < 0.01, (k, fpr[k])
    assert fpr[-1] > 3 * fpr[0]


def test_quantile_under_contamination_matches_closed_form():
    from scipy.stats import norm
    rng = np.random.default_rng(2)
    for eps in (0.01, 0.025):
        cal = _stream(rng, K=3, n=400000, eps=eps)
        T = TH.quantile(0.0, cal, alpha=ALPHA)[-1]
        assert abs((1 - norm.cdf(T)) - (1 - ALPHA - eps) / (1 - eps)) < 0.004
    # at/above the breakdown point eps >= 1-alpha the threshold leaves the benign range
    cal = _stream(rng, K=3, n=400000, eps=0.10)
    T = TH.quantile(0.0, cal, alpha=ALPHA)[-1]
    assert T > 4.0


def test_budgeted_bacr_adapts_upward_but_rate_limited():
    rng = np.random.default_rng(3)
    cal = _stream(rng, K=12, n=20000, shift=3.0)
    delta = 0.5
    T = TH.bacr_budgeted(1.645, cal, alpha=ALPHA, delta=delta, eps_budget=0.025)
    assert np.all(np.abs(np.diff(T)) <= delta + 1e-12)       # movement bound respected
    assert T[1] - T[0] == delta                              # moves UP (original cannot)
    target = 3.0 + np.quantile(rng.normal(size=10**6), ALPHA * 0.975)
    assert abs(T[-1] - target) < 0.05                        # eventually reaches the new quantile


def test_budgeted_bacr_clean_and_contaminated_behaviour():
    from scipy.stats import norm
    rng = np.random.default_rng(4)
    T0 = float(np.quantile(rng.normal(size=10**6), ALPHA))
    clean = TH.bacr_budgeted(T0, _stream(rng, K=10, n=200000), alpha=ALPHA, delta=10, eps_budget=0.025)
    assert abs((1 - norm.cdf(clean[-1])) - (1 - ALPHA * 0.975)) < 0.004   # stable, no ratchet
    assert abs(clean[-1] - clean[-2]) < 0.02
    for eps, expect in ((0.025, 0.05), (0.05, 1 - ALPHA * 0.975 / 0.95)):
        T = TH.bacr_budgeted(T0, _stream(rng, K=6, n=400000, eps=eps), alpha=ALPHA, delta=10, eps_budget=0.025)
        assert abs((1 - norm.cdf(T[-1])) - expect) < 0.004


def test_fixed_is_constant():
    T = TH.fixed(2.0, [np.zeros(5)] * 7)
    assert np.all(T == 2.0) and len(T) == 8


def test_no_lookahead_in_trajectories():
    """T[k] depends only on calibration batches 1..k."""
    rng = np.random.default_rng(5)
    cal = _stream(rng, K=10, n=500)
    cal2 = [c.copy() for c in cal]
    cal2[6] = cal2[6] + 5.0                      # perturb calibration batch 7 (index 6)
    for name, fn in TH.STRATEGIES.items():
        a = fn(1.6, cal, alpha=ALPHA, delta=0.3, eps_budget=0.025)
        b = fn(1.6, cal2, alpha=ALPHA, delta=0.3, eps_budget=0.025)
        assert np.array_equal(a[:7], b[:7]), name   # T[0..6] unaffected
        if name != "fixed":
            assert a[7] != b[7], name
