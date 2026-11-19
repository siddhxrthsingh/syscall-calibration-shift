import numpy as np
from syscalib.generator import World
from syscalib.detector import MarkovDetector
from syscalib.rng import stream


def test_rows_stochastic_and_reproducible(cfg):
    w1, w2 = World(cfg, 123), World(cfg, 123)
    assert np.allclose(w1.A.sum(axis=2), 1) and np.allclose(w1.N.sum(axis=1), 1)
    assert np.array_equal(w1.A, w2.A)
    assert not np.array_equal(w1.A, World(cfg, 124).A)


def test_sample_shape_range_and_seed_determinism(cfg):
    w = World(cfg, 1)
    lv = w.level_for("stationary", "pre")
    a = w.sample(lv, 50, stream(1, "train"))
    b = w.sample(lv, 50, stream(1, "train"))
    assert a.shape == (50, cfg["world"]["L"]) and a.min() >= 0 and a.max() < cfg["world"]["V"]
    assert np.array_equal(a, b)


def test_generator_is_not_first_order_markov(cfg):
    """The detector must be misspecified: second-order structure should exist."""
    w = World(cfg, 2)
    X = w.sample(w.level_for("stationary", "pre"), 4000, stream(2, "train"))
    V = cfg["world"]["V"]
    c2 = np.zeros((V, V, V)); np.add.at(c2, (X[:, :-2].ravel(), X[:, 1:-1].ravel(), X[:, 2:].ravel()), 1)
    c1 = c2.sum(axis=0)  # not used; marginal over first symbol
    p2 = (c2 + 0.5) / (c2.sum(axis=2, keepdims=True) + 0.5 * V)       # P(c|a,b)
    c1b = c2.sum(axis=0); p1 = (c1b + 0.5) / (c1b.sum(axis=1, keepdims=True) + 0.5 * V)  # P(c|b)
    w_ab = c2.sum(axis=2) / c2.sum()
    kl = (w_ab[:, :, None] * (c2 / np.clip(c2.sum(axis=2, keepdims=True), 1, None)) *
          (np.log(p2) - np.log(p1[None]))).sum()
    assert kl > 0.02, kl


def test_shift_raises_benign_scores(cfg):
    w = World(cfg, 3)
    det = MarkovDetector(cfg["world"]["V"], cfg["detector"]["kappa"]).fit(
        w.sample(w.level_for("stationary", "pre"), 2000, stream(3, "train")))
    m = {k: det.score(w.sample(w.level_for(k, "post"), 1500, stream(3, "eval_benign", 1))).mean()
         for k in ("stationary", "moderate", "severe")}
    assert m["stationary"] < m["moderate"] < m["severe"]


def test_detector_frozen_and_rare_transitions_score_higher(cfg):
    V = cfg["world"]["V"]
    w = World(cfg, 4)
    det = MarkovDetector(V, cfg["detector"]["kappa"]).fit(
        w.sample(w.level_for("stationary", "pre"), 2000, stream(4, "train")))
    assert not det.logP.flags.writeable
    X = w.sample(w.level_for("stationary", "pre"), 100, stream(4, "eval_benign", 1))
    s_before = det.score(X).copy()
    det.score(X[::-1])  # scoring must not change state
    assert np.array_equal(s_before, det.score(X))
    # a trace made only of the least likely transitions scores above every benign trace
    worst = np.unravel_index(np.argmin(det.logP), det.logP.shape)
    bad = np.tile(np.array(worst, dtype=np.int16), (1, cfg["world"]["L"] // 2))
    assert det.score(bad)[0] > s_before.max()


def test_score_is_length_normalised(cfg):
    V = cfg["world"]["V"]
    w = World(cfg, 5)
    det = MarkovDetector(V, cfg["detector"]["kappa"]).fit(
        w.sample(w.level_for("stationary", "pre"), 2000, stream(5, "train")))
    X = w.sample(w.level_for("stationary", "pre"), 300, stream(5, "eval_benign", 1))
    assert abs(det.score(X).mean() - det.score(X[:, :60]).mean()) < 0.25
