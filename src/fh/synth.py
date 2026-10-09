"""Gaussian committees with closed-form ground truth.

Y ~ Cat(pi). Member k observes a score z_k in R^d (d = C-1); jointly z | Y=y ~ N(mu_y, Sigma) with
mu_y = (a_1 M_1[y], ..., a_K M_K[y]) and block covariance Sigma_kj = r_kj s_k s_j I_d (correlated errors).
Member k reports P_k = softmax(tau_k * l_k(z_k) + b_k), where l_k is the exact log-posterior given z_k alone
(tau_k = 1, b_k = 0: calibrated). Because z_k -> l_k is injective, P_S carries exactly the information of z_S, so

    H(Y | P_S) = H(Y | z_S) = E[-log Pr(Y | z_S)]

is computed by Monte Carlo with the exact linear-discriminant posterior (standard error ~1e-3 for 2e5 samples).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _simplex(C, d, rng):
    """C class means in R^d (d = C-1), centred regular simplex, random rotation, unit average norm."""
    E = np.eye(C) - 1.0 / C
    U, s, Vt = np.linalg.svd(E)
    M = U[:, :d] * s[:d]
    Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
    M = M @ Q
    return M / np.sqrt((M ** 2).sum(1).mean())


@dataclass
class GaussianCommittee:
    C: int
    a: np.ndarray                     # (K,) signal strength
    s: np.ndarray                     # (K,) noise scale
    R: np.ndarray                     # (K, K) noise correlation
    tau: np.ndarray                   # (K,) temperature (1 = calibrated)
    b: np.ndarray                     # (K, C) class bias of the reported probabilities
    pi: np.ndarray                    # (C,) class prior
    M: np.ndarray = field(default=None)  # (K, C, d) class-mean configurations
    seed: int = 42

    def __post_init__(self):
        self.K = len(self.a)
        self.d = max(self.C - 1, 1)
        rng = np.random.default_rng(self.seed)
        if self.M is None:
            self.M = np.stack([_simplex(self.C, self.d, rng) for _ in range(self.K)])
        I = np.eye(self.d)
        self.Sigma = np.kron(self.R * np.outer(self.s, self.s), I)
        self.mu = np.stack([np.concatenate([self.a[k] * self.M[k, y] for k in range(self.K)]) for y in range(self.C)])

    def _idx(self, S):
        return np.concatenate([np.arange(k * self.d, (k + 1) * self.d) for k in S])

    def log_post(self, z, S):
        """Exact log Pr(Y | z_S) for rows z (n, K*d)."""
        ix = self._idx(S)
        Sig = self.Sigma[np.ix_(ix, ix)]
        mu = self.mu[:, ix]                                   # (C, |S|d)
        A = np.linalg.solve(Sig, mu.T)                        # (|S|d, C)
        lin = z[:, ix] @ A - 0.5 * np.einsum("cj,jc->c", mu, A) + np.log(self.pi)
        lin -= lin.max(1, keepdims=True)
        return lin - np.log(np.exp(lin).sum(1, keepdims=True))

    def sample(self, n, rng):
        y = rng.choice(self.C, n, p=self.pi)
        L = np.linalg.cholesky(self.Sigma + 1e-12 * np.eye(len(self.Sigma)))
        z = self.mu[y] + rng.standard_normal((n, len(self.Sigma))) @ L.T
        P = np.empty((n, self.K, self.C))
        for k in range(self.K):
            lg = self.tau[k] * self.log_post(z, [k]) + self.b[k]
            lg -= lg.max(1, keepdims=True)
            P[:, k] = np.exp(lg) / np.exp(lg).sum(1, keepdims=True)
        return P, y, z

    def truth(self, subsets, N=200_000, seed=0):
        """Monte-Carlo ground truth: H(Y), H(Y | z_S) for each subset, raw log-loss L_k of each member."""
        rng = np.random.default_rng(seed)
        P, y, z = self.sample(N, rng)
        out = {"H_y": float(-(self.pi * np.log(self.pi)).sum())}
        for S in subsets:
            S = tuple(S)
            out[S] = float(-self.log_post(z, S)[np.arange(N), y].mean()) if S else out["H_y"]
        out["L"] = np.array([-np.log(np.clip(P[np.arange(N), k, y], 1e-12, None)).mean() for k in range(self.K)])
        return out


def toy(kind, C=2, seed=42):
    """Two-member committees that separate diversity from complementarity.

    redundant : member 2 = member 1 plus small independent noise (high agreement, almost no complementarity)
    noise     : member 2 has no signal and independent noise (high disagreement, zero complementarity)
    synergy   : member 2 has a weak signal but its noise is correlated with member 1's (little accuracy of its
                own, yet large I(Y; P_2 | P_1): noise cancellation). A calibrated member with *no* signal reports
                the prior and cannot contribute (Lemma: I(Y;P_k)=0 and calibration imply P_k = pi a.s.)
    complement: two equally good members with independent errors
    """
    one = np.ones(2)
    pi = np.full(C, 1.0 / C)
    zb = np.zeros((2, C))
    M1 = _simplex(C, max(C - 1, 1), np.random.default_rng(seed))
    M = np.stack([M1, M1])  # both members share the class-mean configuration
    # (signal a, noise correlation R, temperature tau); a weak but non-zero signal keeps P_2 injective in z_2
    spec = {"redundant": ([1.5, 1.5], [[1, .98], [.98, 1]], [1, 1]),
            "noise": ([1.5, 0.05], [[1, 0], [0, 1]], [1, 20]),       # confident member without signal
            "synergy": ([1.5, 0.2], [[1, .8], [.8, 1]], [1, 1]),      # weak member whose noise cancels member 1's
            "complement": ([1.5, 1.5], [[1, 0], [0, 1]], [1, 1])}
    if kind not in spec:
        raise ValueError(kind)
    a, R, tau = spec[kind]
    return GaussianCommittee(C, np.array(a, float), one, np.array(R, float), np.array(tau, float), zb, pi, M=M,
                             seed=seed)


def random_committee(rng, K=None, C=None):
    """Random committee for estimator validation (varied signal, correlation, calibration)."""
    K = K or int(rng.integers(2, 9))
    C = C or int(rng.choice([2, 3, 5, 10]))
    a = np.where(rng.random(K) > 0.1, rng.uniform(0.1, 2.5, K), 0.02)  # a > 0 keeps outputs injective
    s = np.exp(rng.uniform(-0.3, 0.3, K))
    G = int(rng.integers(1, K + 1))
    grp = rng.integers(0, G, K)
    rho_in, rho_out = rng.uniform(0.3, 0.9), rng.uniform(0.0, 0.3)
    R = np.where(grp[:, None] == grp[None, :], rho_in, rho_out)
    np.fill_diagonal(R, 1.0)
    w, V = np.linalg.eigh(R)
    R = (V * np.clip(w, 1e-3, None)) @ V.T
    R = R / np.sqrt(np.outer(np.diag(R), np.diag(R)))
    tau = np.exp(rng.uniform(np.log(0.4), np.log(2.5), K))
    b = 0.5 * rng.standard_normal((K, C)) * (rng.random((K, 1)) < 0.3)
    pi = rng.dirichlet(np.full(C, 5.0))
    return GaussianCommittee(C, a, s, R, tau, b, pi, seed=int(rng.integers(1 << 30)))
