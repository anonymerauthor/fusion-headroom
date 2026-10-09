"""Committee pruning: information-based selection vs. classic criteria.

All selections use the OOF rows only; the pruned committees are evaluated on the test rows of the refitted members.
Orders (a ranking of all K members; the first m form the pruned committee of size m):
  info_fwd   forward selection maximising I(Y; P_S) (cross-validated, variational)
  shapley    members sorted by their Shapley value of I(Y; P_S)
  accuracy   members sorted by their own OOF log-loss
  kappa      start from the best member, add the member with the lowest mean pairwise kappa to the selected ones
             (Margineantu & Dietterich 1997, greedy variant)
  random     random order (averaged over seeds by the caller)
"""
from __future__ import annotations

import warnings

import numpy as np
from sklearn.linear_model import LogisticRegressionCV

from .info import entropy_y, logp, nll, shapley_information


def _kappa(a, b, C):
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in range(C))
    return (po - pe) / (1 - pe + 1e-12)


def orders(P, y, seed=42, reg=1.0, n_folds=3):
    n, K, C = P.shape
    phi, v = shapley_information(P, y, model="lr_fixed", reg=reg, n_folds=n_folds, seed=seed)
    H_y = entropy_y(y, C, n_folds, seed)

    from .info import cond_entropy

    def info(S):  # value from the Shapley game cache (complete for K <= 8), otherwise computed on demand
        S = tuple(sorted(S))
        if S not in v:
            v[S] = H_y - cond_entropy(P, y, S, C, "lr_fixed", reg, n_folds, seed)
        return v[S]

    sel, rest = [], list(range(K))
    while rest:
        k = max(rest, key=lambda j: info(sel + [j]))
        sel.append(k); rest.remove(k)
    L = np.array([nll(P[:, k], y) for k in range(K)])
    pred = P.argmax(-1)
    kap, rest = [int(np.argmin(L))], [k for k in range(K) if k != int(np.argmin(L))]
    while rest:
        k = min(rest, key=lambda j: np.mean([_kappa(pred[:, j], pred[:, s], C) for s in kap]))
        kap.append(k); rest.remove(k)
    # CMIM-style filter (Fleuret 2004) on members: needs only singletons and pairs; add the member whose worst-case
    # complementarity min_{j in S} I(Y; P_k | P_j) = v({j,k}) - v({j}) is largest
    cm = [int(np.argmax([info([k]) for k in range(K)]))]
    rest = [k for k in range(K) if k != cm[0]]
    while rest:
        k = max(rest, key=lambda j: min(info([s, j]) - info([s]) for s in cm))
        cm.append(k); rest.remove(k)
    # greedy ensemble selection (Caruana et al. 2004): members ordered by their selection weight, ties by accuracy
    from .accounting import greedy_weights
    gw = greedy_weights(P, y)
    ges = sorted(range(K), key=lambda k: (-gw[k], L[k]))
    # L1-regularised stacking: members ranked by the norm of their coefficients in a sparse
    # multinomial logistic regression on standardised log-probabilities (regularisation by inner CV); ties by accuracy
    X = logp(P).reshape(n, -1)
    X = (X - X.mean(0)) / np.where(X.std(0) > 1e-8, X.std(0), 1.0)
    kf = int(min(3, np.bincount(y)[np.unique(y)].min()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        l1 = LogisticRegressionCV(Cs=np.logspace(-3, 1, 6), cv=max(kf, 2), penalty="l1", solver="saga",
                                  scoring="neg_log_loss", max_iter=2000).fit(X, y)
    imp = np.linalg.norm(l1.coef_.reshape(l1.coef_.shape[0], K, C), axis=(0, 2))
    l1o = sorted(range(K), key=lambda k: (-imp[k], L[k]))
    rng = np.random.default_rng(seed)
    return {"info_fwd": sel, "cmim": cm, "l1_stack": l1o, "shapley": list(np.argsort(-phi)), "accuracy": list(np.argsort(L)),
            "greedy_es": ges, "kappa": kap, **{f"random{r}": list(rng.permutation(K)) for r in range(5)}}, phi


def eval_subset(P, y, Pt, yt, S):
    """Test log-loss of stacking (LR on log-probabilities, CV-regularised) and of the arithmetic mean on subset S."""
    S = list(S)
    Xa, Xb = logp(P[:, S]).reshape(len(y), -1), logp(Pt[:, S]).reshape(len(yt), -1)
    mu, sd = Xa.mean(0), Xa.std(0)
    sd = np.where(sd > 1e-8, sd, 1.0)
    C = P.shape[-1]
    k = int(min(3, np.bincount(y)[np.unique(y)].min()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LogisticRegressionCV(Cs=np.logspace(-3, 3, 7), cv=max(k, 2), scoring="neg_log_loss", max_iter=3000)
        m.fit((Xa - mu) / sd, y)
    Q = np.full((len(yt), C), 1e-6)
    Q[:, m.classes_] = m.predict_proba((Xb - mu) / sd)
    return {"stack": nll(Q, yt), "mean": nll(Pt[:, S].mean(1), yt)}
