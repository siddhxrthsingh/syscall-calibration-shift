"""First-order Markov anomaly detector (frozen after training).

Transition probabilities use additive smoothing:
    P(b|a) = (C[a,b] + kappa) / (sum_b' C[a,b'] + kappa * V)
Trace score = mean negative log-likelihood per scored transition:
    s(x) = -(1/(L-1)) * sum_{t=2..L} log P(x_t | x_{t-1})
Higher = more anomalous. The mean (not the sum) removes length from the score, and
all traces here have identical length L anyway.
"""
import numpy as np


class MarkovDetector:
    def __init__(self, V, kappa):
        self.V, self.kappa = V, kappa
        self.logP = None

    def fit(self, X):
        V = self.V
        C = np.zeros((V, V), dtype=np.float64)
        np.add.at(C, (X[:, :-1].ravel(), X[:, 1:].ravel()), 1.0)
        P = (C + self.kappa) / (C.sum(axis=1, keepdims=True) + self.kappa * V)
        self.logP = np.log(P)
        self.logP.setflags(write=False)  # frozen: no later mutation possible
        return self

    def score(self, X):
        assert self.logP is not None, "detector not trained"
        return -self.logP[X[:, :-1], X[:, 1:]].mean(axis=1)
