"""Three attack families, each a genuine malicious modification of a benign trace.

All lengths are preserved, so trace length never carries signal.

  rare    : m scattered positions get a successor whose transition is (near-)forbidden
            in every training profile. Disturbs first-order statistics strongly.
  pattern : a short motif of individually plausible-but-uncommon transitions is
            repeated as one contiguous block (periodic, loop-like behaviour).
  mimic   : a benign trace with a small number (mimic_m, default 2) of forbidden-transition insertions.

The attack definitions use the generator's ground-truth matrices, not the
detector, so the attacker is not tuned against the detector under test.
"""
import numpy as np
from .rng import stream


class AttackDesign:
    def __init__(self, world, acfg):
        self.acfg = acfg
        A = world.A
        V = world.V
        pmax = A.max(axis=0)                              # best-case prob per pair
        forb = pmax < acfg["forbidden_thresh"]
        np.fill_diagonal(forb, forb.diagonal())           # keep diagonal as is
        # guarantee at least one forbidden successor per row
        for a in range(V):
            if not forb[a].any():
                forb[a, np.argmin(pmax[a])] = True
        self.forb = forb
        self.forb_cdf = np.cumsum(forb / forb.sum(axis=1, keepdims=True), axis=1)
        self.forb_cdf[:, -1] = 1.0
        rng = stream(world.seed, "attack_design")
        self.motif = self._make_motif(pmax, rng, acfg)

    def _make_motif(self, pmax, rng, acfg):
        V = pmax.shape[0]
        k = acfg["pattern_motif_len"]
        lo, hi = acfg["motif_pmin"], acfg["motif_pmax"]
        ok = (pmax >= lo) & (pmax <= hi)
        for _ in range(2000):
            m = [int(rng.integers(0, V))]
            good = True
            for _j in range(k - 1):
                cand = np.flatnonzero(ok[m[-1]])
                if len(cand) == 0:
                    good = False
                    break
                m.append(int(rng.choice(cand)))
            if good and ok[m[-1], m[0]]:                   # closed loop
                return np.array(m, dtype=np.int16)
        # fallback: relax closure
        m = [int(rng.integers(0, V))]
        for _j in range(k - 1):
            cand = np.flatnonzero(ok[m[-1]])
            if len(cand) == 0:
                cand = np.argsort(-pmax[m[-1]])[:3]
            m.append(int(rng.choice(cand)))
        return np.array(m, dtype=np.int16)

    # ---- attack constructors (operate on a copy) -------------------------
    def _insert_forbidden(self, x, m, rng):
        n, L = x.shape
        x = x.copy()
        pos = np.stack([rng.permutation(np.arange(1, L))[:m] for _ in range(n)])
        pos.sort(axis=1)
        for j in range(m):
            p = pos[:, j]
            prev = x[np.arange(n), p - 1]
            u = rng.random(n)
            new = (u[:, None] > self.forb_cdf[prev]).sum(axis=1)
            x[np.arange(n), p] = np.minimum(new, self.forb_cdf.shape[1] - 1)
        return x

    def rare(self, x, rng):
        return self._insert_forbidden(x, self.acfg["rare_m"], rng)

    def mimic(self, x, rng):
        return self._insert_forbidden(x, self.acfg["mimic_m"], rng)

    def pattern(self, x, rng):
        n, L = x.shape
        x = x.copy()
        block = np.tile(self.motif, self.acfg["pattern_repeats"])
        B = len(block)
        start = rng.integers(1, L - B + 1, size=n)
        idx = start[:, None] + np.arange(B)[None, :]
        x[np.arange(n)[:, None], idx] = block[None, :]
        return x


FAMILIES = ("rare", "pattern", "mimic")


def apply_family(design, family, x, rng):
    return getattr(design, family)(x, rng)
