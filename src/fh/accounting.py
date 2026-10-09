"""Gain accounting on the OOF rows (internally consistent, Theorem 2).

For a rule phi with loss L_phi(n) (cross-validated for learned rules):

    L_best - L_phi(n) = headroom - A_phi - E_phi(n)
    A_phi    = L_phi(inf) - H(Y|P)      approximation gap (L_phi(inf) from the learning-curve asymptote)
    E_phi(n) = L_phi(n) - L_phi(inf)    estimation gap (= b_phi / m for a learning curve H + b/m)

Fixed rules (mean, geometric mean, best member) have E = 0. The predicted cross-over size at which a learned rule F
overtakes a fixed rule F0 is n* = b_F / (A_F0 - A_F) (in training rows of the recalibration fit).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import nnls
from sklearn.model_selection import StratifiedKFold

from .info import _folds, cond_entropy, cond_entropy_lc, nll

EPS = 1e-6


def _norm(P):
    P = np.clip(P, EPS, None)
    return P / P.sum(-1, keepdims=True)


def arith(P):
    return _norm(P.mean(1))


def geo(P):
    L = np.log(_norm(P)).mean(1)
    L -= L.max(1, keepdims=True)
    return _norm(np.exp(L))


def greedy_weights(P, y, iters=50):
    """Caruana et al. (2004) forward selection with replacement on log-loss; returns member counts."""
    K = P.shape[1]
    counts = np.zeros(K)
    cur = np.zeros_like(P[:, 0])
    for t in range(iters):
        best, bk = np.inf, 0
        for k in range(K):
            l = nll((cur * t + P[:, k]) / (t + 1), y)
            if l < best:
                best, bk = l, k
        counts[bk] += 1
        cur = (cur * t + P[:, bk]) / (t + 1)
    return counts / counts.sum()


def cv_loss_greedy(P, y, n_folds=5, seed=42):
    folds = _folds(y, n_folds, seed)
    Q = np.zeros((len(y), P.shape[2]))
    for a, b in folds:
        w = greedy_weights(P[a], y[a])
        Q[b] = _norm(np.einsum("k,nkc->nc", w, P[b]))
    return nll(Q, y)


def lc_curve(fn, P, y, fracs=(0.5, 0.75, 1.0), reps=3, n_folds=5, seed=42):
    """Learning curve of a CV loss fn(P, y) -> (asymptote, slope b, loss at full n)."""
    rng = np.random.default_rng(seed)
    ms, hs = [], []
    full = None
    for f in fracs:
        for r in range(1 if f >= 1.0 else reps):
            if f >= 1.0:
                idx = np.arange(len(y))
            else:
                idx = np.sort(np.concatenate([rng.choice(np.where(y == c)[0], max(2, int(round(f * (y == c).sum()))),
                                                         replace=False) for c in np.unique(y)]))
            h = fn(P[idx], y[idx])
            if f >= 1.0:
                full = h
            if np.isfinite(h):
                ms.append(len(idx) * (1 - 1 / n_folds)); hs.append(h)
    ms, hs = np.array(ms), np.array(hs)
    (h_inf, b), _ = nnls(np.stack([np.ones_like(ms), 1 / ms], 1), hs)
    return float(h_inf), float(b), float(full)


def account(P, y, n_folds=5, seed=42):
    """Per-rule gain accounting on the OOF rows of one committee instance."""
    n, K, C = P.shape
    L = np.array([nll(P[:, k], y) for k in range(K)])
    ks = int(np.argmin(L))
    H_lr = cond_entropy_lc(P, y, range(K), C, n_folds=n_folds, seed=seed)
    fixed = {"arith_mean": nll(arith(P), y), "geo_mean": nll(geo(P), y)}
    rules = {"stack_LR": lambda Pp, yy: cond_entropy(Pp, yy, range(K), C, n_folds=n_folds, seed=seed),
             "greedy_ES": lambda Pp, yy: cv_loss_greedy(Pp, yy, n_folds, seed)}
    curves = {name: lc_curve(fn, P, y, n_folds=n_folds, seed=seed) for name, fn in rules.items()}
    # every rule's asymptotic loss is an upper bound on H(Y|P): take the tightest (Theorem 2 requires A >= 0)
    H_all = min([H_lr] + [c[0] for c in curves.values()] + list(fixed.values()) + [L[ks]])
    head = L[ks] - H_all
    out = {"L_best": L[ks], "H_all": H_all, "H_lr": H_lr, "headroom": head,
           "H_source": ["lr"] + list(curves) + list(fixed) + ["best"]}
    out["H_source"] = out["H_source"][int(np.argmin([H_lr] + [c[0] for c in curves.values()] + list(fixed.values()) + [L[ks]]))]
    for name, l in fixed.items():
        out[f"{name}_loss"], out[f"{name}_A"], out[f"{name}_E"], out[f"{name}_b"] = l, l - H_all, 0.0, 0.0
    for name, (h_inf, b, full) in curves.items():
        out[f"{name}_loss"], out[f"{name}_A"], out[f"{name}_E"], out[f"{name}_b"] = full, h_inf - H_all, full - h_inf, b
    for r in ("arith_mean", "geo_mean", "stack_LR", "greedy_ES"):
        out[f"{r}_gain"] = L[ks] - out[f"{r}_loss"]
    # predicted cross-over training size of stacking vs. the arithmetic mean
    dA = out["arith_mean_A"] - out["stack_LR_A"]
    out["nstar_stack_vs_mean"] = out["stack_LR_b"] / dA if dA > 0 else np.inf
    return out
