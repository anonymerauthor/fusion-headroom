"""Classic ensemble diversity measures (Kuncheva & Whitaker 2003; Kuncheva 2014, ch. 8) and label-free
probabilistic diversity (ambiguity of the normalised geometric mean, Wood et al. 2023; Jensen-Shannon spread).

Oracle-output measures use the correctness indicators c_ik = 1[argmax P_ik = y_i]. Conventions follow Kuncheva
(2014): for Q, rho and double fault *lower* means more diverse; for the others *higher* means more diverse. We return
the raw values; analyses use them with their natural sign and also |correlation|.
"""
from __future__ import annotations

import itertools

import numpy as np

EPS = 1e-12


def _pair_table(a, b):
    n11 = np.mean(a & b); n00 = np.mean(~a & ~b); n10 = np.mean(a & ~b); n01 = np.mean(~a & b)
    return n11, n00, n10, n01


def oracle_measures(P, y):
    n, K, C = P.shape
    c = P.argmax(-1) == y[:, None]                    # (n, K)
    Q, R, DIS, DF = [], [], [], []
    for i, j in itertools.combinations(range(K), 2):
        n11, n00, n10, n01 = _pair_table(c[:, i], c[:, j])
        Q.append((n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10 + EPS))
        R.append((n11 * n00 - n01 * n10) / np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00) + EPS))
        DIS.append(n01 + n10)
        DF.append(n00)
    l = c.sum(1)                                      # number of correct members per row
    E = np.mean(np.minimum(l, K - l) / (K - np.ceil(K / 2)))
    KW = np.mean(l * (K - l)) / K ** 2
    p_bar = c.mean()
    kappa = 1 - (np.mean(l * (K - l)) / K) / ((K - 1) * p_bar * (1 - p_bar) + EPS)
    theta = np.var(l / K)                             # difficulty (lower = more diverse)
    f = np.bincount(K - l, minlength=K + 1) / n       # p_i: prob. that i randomly chosen members fail
    p1 = sum(i / K * f[i] for i in range(K + 1))
    p2 = sum(i * (i - 1) / (K * (K - 1)) * f[i] for i in range(K + 1))
    GD = 1 - p2 / (p1 + EPS)
    CFD = 0.0 if f[0] == 1 else sum((K - i) / (K - 1) * f[i] for i in range(1, K + 1)) / (1 - f[0])
    return {"Q": float(np.mean(Q)), "rho": float(np.mean(R)), "disagreement": float(np.mean(DIS)),
            "double_fault": float(np.mean(DF)), "entropy_E": float(E), "KW": float(KW), "kappa": float(kappa),
            "difficulty": float(theta), "GD": float(GD), "CFD": float(CFD)}


def label_free_measures(P):
    """Diversity from the outputs alone (no labels)."""
    Pc = np.clip(P, 1e-6, None); Pc /= Pc.sum(-1, keepdims=True)
    L = np.log(Pc)
    g = L.mean(1); g = g - np.log(np.exp(g).sum(-1, keepdims=True))     # normalised geometric mean
    amb_geo = np.mean(np.sum(np.exp(g)[:, None, :] * (g[:, None, :] - L), -1))  # mean_k KL(q_geo || p_k)
    m = Pc.mean(1)
    js = np.mean(np.sum(Pc * (L - np.log(m)[:, None, :]), -1))         # mean_k KL(p_k || mean)
    pred = P.argmax(-1)
    K = P.shape[1]
    dis_lf = np.mean([np.mean(pred[:, i] != pred[:, j]) for i, j in itertools.combinations(range(K), 2)])
    return {"ambiguity_geo": float(amb_geo), "js_spread": float(js), "pred_disagreement": float(dis_lf)}


def all_measures(P, y):
    return {**oracle_measures(P, y), **label_free_measures(P)}
