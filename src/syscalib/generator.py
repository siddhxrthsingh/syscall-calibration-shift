"""Synthetic system-call trace generator.

A trace is a length-L sequence over V abstract system-call symbols produced by a
*regime-switching* process: P program profiles (each with its own first-order
transition matrix) plus one profile that only appears after a workload shift.
The active profile persists for a geometric number of steps, then is re-drawn
from the workload mixture. Observed traces are therefore NOT first-order Markov,
so the first-order Markov detector is deliberately misspecified.

Workload shift has three components (all controlled by the 'shift' config):
  mu     : move the mixture weights from w0 toward w_alt
  lam    : drift every profile's transitions toward an independent target B_p
  pi_new : probability mass given to a benign profile never seen in training
"""
import numpy as np
from .rng import stream


def _entropy_rows(M):
    return float(np.mean(-(M * np.log(np.clip(M, 1e-300, None))).sum(axis=1)))


def make_profile(rng, V, G, sparse_alpha, dom_alpha, shared, k_specific):
    """Sparse row-stochastic matrix; a few dominant successors per row."""
    grp = np.arange(V) // (V // G)
    alpha = np.full((V, V), sparse_alpha)
    for a in range(V):
        alpha[a, shared[a]] = dom_alpha
        # profile-specific dominant successors, biased toward the same group
        same = np.flatnonzero(grp == grp[a])
        for _ in range(k_specific):
            pool = same if rng.random() < 0.7 else np.arange(V)
            alpha[a, rng.choice(pool)] = dom_alpha
    M = rng.gamma(alpha)
    M = np.clip(M, 1e-12, None)
    return M / M.sum(axis=1, keepdims=True)


class World:
    """One random 'world' (one seed): profile matrices and mixture weights."""

    def __init__(self, cfg, seed):
        wc = cfg["world"]
        self.cfg = cfg
        self.seed = seed
        self.V, self.L, self.q = wc["V"], wc["L"], wc["switch_q"]
        rng = stream(seed, "world")
        V, G, P = wc["V"], wc["G"], wc["P"]
        shared = rng.integers(0, V, size=(V, wc["k_shared"]))
        mk = lambda: make_profile(rng, V, G, wc["sparse_alpha"], wc["dom_alpha"], shared, wc["k_specific"])
        A = [mk() for _ in range(P)]
        order = np.argsort([_entropy_rows(m) for m in A])  # ascending entropy
        self.A = np.stack([A[i] for i in order])           # (P,V,V)
        self.B = np.stack([mk() for _ in range(P)])        # drift targets
        self.N = mk()                                      # novel (post-shift) profile
        self.w0 = np.asarray(wc["w0"], float)
        self.w_alt = np.asarray(wc["w_alt"], float)

    def regimes(self, level):
        """Return (matrices (R,V,V), weights (R,)) for a workload level dict."""
        mu, lam, pn = level["mu"], level["lam"], level["pi_new"]
        A = (1 - lam) * self.A + lam * self.B
        mats = np.concatenate([A, self.N[None]], axis=0)
        w = (1 - mu) * self.w0 + mu * self.w_alt
        w = np.concatenate([(1 - pn) * w, [pn]])
        return mats, w / w.sum()

    def level_for(self, workload, phase):
        """phase 'pre' is always the training workload; 'post' applies the shift."""
        if phase == "pre":
            return self.cfg["shift"]["stationary"]
        return self.cfg["shift"][workload]

    def sample(self, level, n, rng):
        mats, w = self.regimes(level)
        return sample_regime_switching(mats, w, n, self.L, self.q, rng)


def sample_regime_switching(mats, w, n, L, q, rng):
    R, V, _ = mats.shape
    cdf = np.cumsum(mats, axis=2)
    cdf[:, :, -1] = 1.0
    wc = np.cumsum(w)
    wc[-1] = 1.0
    x = np.empty((n, L), dtype=np.int16)
    x[:, 0] = rng.integers(0, V, size=n)
    r = np.searchsorted(wc, rng.random(n))
    for t in range(1, L):
        sw = rng.random(n) < q
        if sw.any():
            r[sw] = np.searchsorted(wc, rng.random(int(sw.sum())))
        rows = cdf[r, x[:, t - 1]]                       # (n,V)
        nxt = (rng.random(n)[:, None] > rows).sum(axis=1)
        x[:, t] = np.minimum(nxt, V - 1)
    return x
