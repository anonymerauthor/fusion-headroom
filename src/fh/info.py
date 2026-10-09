"""Variational estimators of the information terms of a committee.

For a subset S of members, the conditional entropy H(Y | P_S) is bounded from above by the expected log-loss of
any predictor q(y | P_S); we estimate it by the cross-validated log-loss of a recalibration model fitted on the
members' log-probabilities (multinomial logistic regression = log-linear pool with class biases, or gradient
boosting). Hence I(Y; P_S) = H(Y) - H(Y | P_S) is estimated from below, and the gap shrinks as n grows.

All quantities are in nats per row.
"""
from __future__ import annotations

import itertools
import math
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold

EPS = 1e-6
FEAT_EPS = 1e-3  # trees and kNN output exact zeros; clipping log-probabilities at 1e-3 keeps the recalibration
#                  features well-conditioned (chosen on development committees)
REGS = np.logspace(-3, 3, 7)


def logp(P, eps=None):
    P = np.clip(P, FEAT_EPS if eps is None else eps, None)
    return np.log(P / P.sum(-1, keepdims=True))


def nll(P, y):
    """Mean log-loss of probability matrix P (n, C)."""
    P = np.clip(P, EPS, None)
    P = P / P.sum(-1, keepdims=True)
    return float(-np.log(P[np.arange(len(y)), y]).mean())


def _folds(y, n_folds, seed):
    counts = np.bincount(y)
    k = int(min(n_folds, counts[counts > 0].min()))
    if k < 2:
        return None
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def entropy_y(y, C, n_folds=5, seed=42):
    """Cross-validated log-loss of the Laplace-smoothed class prior (estimate of H(Y))."""
    folds = _folds(y, n_folds, seed)
    if folds is None:
        p = (np.bincount(y, minlength=C) + 1) / (len(y) + C)
        return float(-np.log(p[y]).mean())
    tot = 0.0
    for a, b in folds:
        p = (np.bincount(y[a], minlength=C) + 1) / (len(a) + C)
        tot += -np.log(p[y[b]]).sum()
    return tot / len(y)


def _fit_predict(Xa, ya, Xb, C, model, reg):
    present = np.unique(ya)
    out = np.full((len(Xb), C), EPS)
    if len(present) == 1:
        out[:, present[0]] = 1.0
        return out / out.sum(1, keepdims=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if model == "lr":  # multinomial LR, regularisation chosen by inner 3-fold CV on log-loss
            k = int(min(3, np.bincount(ya)[present].min()))
            m = (LogisticRegressionCV(Cs=REGS, cv=k, scoring="neg_log_loss", max_iter=3000) if k >= 2
                 else LogisticRegression(C=1.0, max_iter=3000))
        elif model == "lr_fixed":
            m = LogisticRegression(C=reg, max_iter=3000)
        elif model == "hgb":
            m = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                               l2_regularization=1.0, random_state=0)
        else:
            raise ValueError(model)
        m.fit(Xa, ya)
        out[:, m.classes_] = m.predict_proba(Xb)
    return out / out.sum(1, keepdims=True)


def _logpool_fit(L, y, C, lam=1e-3):
    """Log-linear pool with one exponent per member and class intercepts: log q ∝ sum_k w_k L_k + b.
    L: (n, s, C) clipped log-probabilities. Small L2 penalty on (w - 1/s) and b. Returns (w, b)."""
    from scipy.optimize import minimize
    n, s, _ = L.shape
    Y = np.zeros((n, C)); Y[np.arange(n), y] = 1.0
    w0 = np.full(s, 1.0 / s)

    def f(theta):
        w, b = theta[:s], theta[s:]
        z = np.einsum("k,nkc->nc", w, L) + b
        z -= z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1))
        loss = (lse - (z * Y).sum(1)).mean() + lam * (((w - w0) ** 2).sum() + (b ** 2).sum())
        G = (np.exp(z - lse[:, None]) - Y) / n
        gw = np.einsum("nc,nkc->k", G, L) + 2 * lam * (w - w0)
        gb = G.sum(0) + 2 * lam * b
        return loss, np.concatenate([gw, gb])

    r = minimize(f, np.concatenate([w0, np.zeros(C)]), jac=True, method="L-BFGS-B", options={"maxiter": 500})
    return r.x[:s], r.x[s:]


def _logpool_predict(L, w, b):
    z = np.einsum("k,nkc->nc", w, L) + b
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def cond_entropy(P, y, S, C=None, model="lr", reg=1.0, n_folds=5, seed=42, return_pred=False):
    if model == "logpool":  # structured recalibration for many classes (K + C parameters)
        C = C or P.shape[-1]
        S = list(S)
        if not S:
            return entropy_y(y, C, n_folds, seed)
        L = logp(P[:, S])
        folds = _folds(y, n_folds, seed)
        if folds is None:
            return float("nan")
        Q = np.zeros((len(y), C))
        for a, b in folds:
            w, bb = _logpool_fit(L[a], y[a], C)
            Q[b] = _logpool_predict(L[b], w, bb)
        h = nll(Q, y)
        return (h, Q) if return_pred else h
    return _cond_entropy_generic(P, y, S, C, model, reg, n_folds, seed, return_pred)


def _cond_entropy_generic(P, y, S, C=None, model="lr", reg=1.0, n_folds=5, seed=42, return_pred=False):
    """Cross-validated estimate of H(Y | P_S). P: (n, K, C); S: iterable of member indices."""
    C = C or P.shape[-1]
    S = list(S)
    if not S:
        return entropy_y(y, C, n_folds, seed)
    X = logp(P[:, S]).reshape(len(y), -1)
    X = X - X.mean(0)
    sd = X.std(0)
    X = X / np.where(sd > 1e-8, sd, 1.0)
    folds = _folds(y, n_folds, seed)
    if folds is None:
        return float("nan")
    Q = np.zeros((len(y), C))
    for a, b in folds:
        Q[b] = _fit_predict(X[a], y[a], X[b], C, model, reg)
    h = nll(Q, y)
    return (h, Q) if return_pred else h


def cond_entropy_lc(P, y, S, C=None, model="lr", fracs=(0.5, 0.75, 1.0), reps=3, n_folds=5, seed=42,
                    return_curve=False):
    """Learning-curve corrected estimate of H(Y | P_S).

    The cross-validated loss of a recalibration model trained on m rows behaves like H + b/m (b >= 0: estimation
    gap of a regular parametric family). We evaluate it on stratified sub-samples of several sizes and extrapolate
    to m -> infinity by non-negative least squares on (1, 1/m)."""
    C = C or P.shape[-1]
    rng = np.random.default_rng(seed)
    ms, hs = [], []
    for f in fracs:
        for r in range(1 if f >= 1.0 else reps):
            if f >= 1.0:
                idx = np.arange(len(y))
            else:
                idx = np.sort(np.concatenate([rng.choice(np.where(y == c)[0], max(2, int(round(f * (y == c).sum()))),
                                                         replace=False) for c in np.unique(y)]))
            h = cond_entropy(P[idx], y[idx], S, C, model, n_folds=n_folds, seed=seed + r)
            if np.isfinite(h):
                ms.append(len(idx) * (1 - 1 / n_folds)); hs.append(h)
    ms, hs = np.array(ms), np.array(hs)
    A = np.stack([np.ones_like(ms), 1 / ms], 1)
    from scipy.optimize import nnls
    (h_inf, b), _ = nnls(A, hs)
    h_inf = float(h_inf)
    return (h_inf, ms, hs, b) if return_curve else h_inf


def decompose(P, y, model="lr", reg=1.0, n_folds=5, seed=42, lc=False, eps_grid=None):
    """eps_grid: clipping thresholds of the recalibration features; every threshold gives a valid variational bound,
    so the conditional entropies are the minimum over the grid (the tightest bound)."""
    global FEAT_EPS
    if eps_grid:
        old, outs = FEAT_EPS, []
        try:
            for e in eps_grid:
                FEAT_EPS = e
                outs.append(decompose(P, y, model, reg, n_folds, seed, lc))
        finally:
            FEAT_EPS = old
        d = dict(outs[0])
        d["H"] = np.min([o["H"] for o in outs], 0)
        d["H_all"] = min(o["H_all"] for o in outs)
        ks = d["k_star"]
        d.update({"headroom": d["L"][ks] - d["H_all"], "cal": d["L"][ks] - d["H"][ks], "ci": d["H"][ks] - d["H_all"],
                  "I_all": d["H_y"] - d["H_all"], "I": d["H_y"] - d["H"],
                  "eps_all": float(eps_grid[int(np.argmin([o["H_all"] for o in outs]))])})
        return d
    return _decompose(P, y, model, reg, n_folds, seed, lc)


def _decompose(P, y, model="lr", reg=1.0, n_folds=5, seed=42, lc=False):
    """Headroom decomposition of a committee.

    L_k      raw log-loss of member k
    H_k      H(Y | P_k)         (member k after recalibration)
    H_all    H(Y | P_1:K)       (Bayes fusion of all members, variational)
    k*       best raw member
    headroom = L_k* - H_all = CAL_k* + CI,  CAL_k* = L_k* - H_k*,  CI = I(Y; P_-k* | P_k*) = H_k* - H_all
    """
    n, K, C = P.shape
    H_y = entropy_y(y, C, n_folds, seed)
    L = np.array([nll(P[:, k], y) for k in range(K)])
    H = np.array([cond_entropy(P, y, [k], C, model, reg, n_folds, seed) for k in range(K)])
    H_all = cond_entropy(P, y, range(K), C, model, reg, n_folds, seed)
    ks = int(np.argmin(L))
    if lc:  # learning-curve correction of the two terms that enter the headroom
        H = H.copy()
        H[ks] = cond_entropy_lc(P, y, [ks], C, model, n_folds=n_folds, seed=seed)
        H_all = cond_entropy_lc(P, y, range(K), C, model, n_folds=n_folds, seed=seed)
    return {"H_y": H_y, "L": L, "H": H, "H_all": H_all, "k_star": ks,
            "headroom": L[ks] - H_all, "cal": L[ks] - H[ks], "ci": H[ks] - H_all,
            "I_all": H_y - H_all, "I": H_y - H}


def shapley_information(P, y, model="lr", reg=1.0, n_folds=5, seed=42, max_exact=8, n_perm=64):
    """Shapley values of the members in the cooperative game v(S) = I(Y; P_S) (they sum to I(Y; P_1:K))."""
    n, K, C = P.shape
    H_y = entropy_y(y, C, n_folds, seed)
    cache = {(): H_y}

    def v(S):
        S = tuple(sorted(S))
        if S not in cache:
            cache[S] = cond_entropy(P, y, S, C, model, reg, n_folds, seed)
        return H_y - cache[S]

    phi = np.zeros(K)
    if K <= max_exact:
        for k in range(K):
            others = [j for j in range(K) if j != k]
            for r in range(K):
                w = math.factorial(r) * math.factorial(K - r - 1) / math.factorial(K)
                for S in itertools.combinations(others, r):
                    phi[k] += w * (v(S + (k,)) - v(S))
    else:
        rng = np.random.default_rng(seed)
        for _ in range(n_perm):
            perm = rng.permutation(K)
            prev = ()
            for k in perm:
                cur = prev + (int(k),)
                phi[k] += v(cur) - v(prev)
                prev = cur
        phi /= n_perm
    return phi, {k: H_y - h for k, h in cache.items()}
