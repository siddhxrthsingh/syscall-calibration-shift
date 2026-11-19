import numpy as np
from syscalib.generator import World
from syscalib.attacks import AttackDesign, FAMILIES, apply_family
from syscalib.rng import stream


def _setup(cfg, seed=7, n=200):
    w = World(cfg, seed)
    d = AttackDesign(w, cfg["attacks"])
    X = w.sample(w.level_for("stationary", "pre"), n, stream(seed, "eval_attack_base", 1))
    return w, d, X


def test_attacks_change_traces_but_preserve_length(cfg):
    w, d, X = _setup(cfg)
    for f in FAMILIES:
        Y = apply_family(d, f, X, stream(7, "eval_attack_mod", 1))
        assert Y.shape == X.shape and Y.min() >= 0 and Y.max() < w.V
        assert (Y != X).any(axis=1).mean() > 0.95, f   # almost every trace really modified


def test_base_not_mutated(cfg):
    w, d, X = _setup(cfg)
    X0 = X.copy()
    for f in FAMILIES:
        apply_family(d, f, X, stream(7, "eval_attack_mod", 1))
    assert np.array_equal(X, X0)


def test_rare_and_mimic_insert_forbidden_transitions(cfg):
    w, d, X = _setup(cfg)
    def n_forb(Z):
        return d.forb[Z[:, :-1], Z[:, 1:]].sum(axis=1)
    base = n_forb(X)
    rare = n_forb(apply_family(d, "rare", X, stream(7, "eval_attack_mod", 1)))
    mim = n_forb(apply_family(d, "mimic", X, stream(7, "eval_attack_mod", 1)))
    assert (rare - base).mean() >= cfg["attacks"]["rare_m"] * 0.8
    assert 1 <= (mim - base).mean() <= cfg["attacks"]["mimic_m"] * 1.2 + 1
    assert (rare - base).mean() > (mim - base).mean()


def test_pattern_inserts_contiguous_periodic_block(cfg):
    w, d, X = _setup(cfg)
    Y = apply_family(d, "pattern", X, stream(7, "eval_attack_mod", 1))
    block = np.tile(d.motif, cfg["attacks"]["pattern_repeats"])
    B = len(block)
    found = 0
    for y in Y:
        found += any(np.array_equal(y[s:s + B], block) for s in range(len(y) - B + 1))
    assert found == len(Y)


def test_mimic_is_not_just_a_benign_sample(cfg):
    """Mimic traces must be genuinely modified: forbidden transitions absent from clean traces."""
    w, d, X = _setup(cfg)
    clean = d.forb[X[:, :-1], X[:, 1:]].sum(axis=1).mean()
    mim = d.forb[apply_family(d, "mimic", X, stream(7, "eval_attack_mod", 1))[:, :-1],
                 apply_family(d, "mimic", X, stream(7, "eval_attack_mod", 1))[:, 1:]].sum(axis=1).mean()
    assert mim > clean + 1
