"""Pruning the 15-member deep pools (CIFAR-10/100).

For each data set, fusion size and repetition (stratified sub-sample of the fusion pool), rank the members by
  info_fwd   forward selection of I(Y; P_S) (log-pool recalibration, clipping 1e-6)
  cmim       CMIM-style filter on pairwise complementarity I(Y; P_k | P_j)
  accuracy   individual fusion-set log-loss
  greedy_es  greedy ensemble selection weights (Caruana et al. 2004)
  kappa      kappa-diversity pruning (Margineantu & Dietterich 1997)
  random     5 random orders
and evaluate the pruned committees of size m = 1..15 on the test set, fused by a fitted log pool and by the mean.
Output: results/deep_prune_<dataset>.csv (resumable). Usage: python scripts/deep_prune.py --dataset cifar10 --limit 1
"""
import argparse, json, os, sys, zlib
import numpy as np, pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import fh.info as info  # noqa: E402
from fh.accounting import greedy_weights  # noqa: E402
from fh.rules import BASELINES  # noqa: E402
from deep_analyze import load_pool, strat  # noqa: E402
from run_real import low_priority  # noqa: E402

SIZES = {"cifar10": (510, 10000), "cifar100": (2500, 10000)}
REPS = 5


def _kappa(a, b, C):
    po = np.mean(a == b)
    pe = sum(np.mean(a == c) * np.mean(b == c) for c in range(C))
    return (po - pe) / (1 - pe + 1e-12)


def orders(P, y, seed):
    info.FEAT_EPS = 1e-6  # deep nets: tightest bound (Section 4); no exact zeros in their outputs
    n, K, C = P.shape
    H_y = info.entropy_y(y, C, 3, seed)
    cache = {}

    def v(S):
        S = tuple(sorted(S))
        if S not in cache:
            cache[S] = H_y - info.cond_entropy(P, y, S, C, model="logpool", n_folds=3, seed=seed)
        return cache[S]

    sel, rest = [], list(range(K))
    while rest:
        k = max(rest, key=lambda j: v(sel + [j])); sel.append(k); rest.remove(k)
    cm = [int(np.argmax([v([k]) for k in range(K)]))]
    rest = [k for k in range(K) if k != cm[0]]
    while rest:
        k = max(rest, key=lambda j: min(v([s, j]) - v([s]) for s in cm)); cm.append(k); rest.remove(k)
    L = np.array([info.nll(P[:, k], y) for k in range(K)])
    gw = greedy_weights(P, y)
    pred = P.argmax(-1)
    kap, rest = [int(np.argmin(L))], [k for k in range(K) if k != int(np.argmin(L))]
    while rest:
        k = min(rest, key=lambda j: np.mean([_kappa(pred[:, j], pred[:, s], C) for s in kap])); kap.append(k); rest.remove(k)
    rng = np.random.default_rng(seed)
    return {"info_fwd": sel, "cmim": cm, "accuracy": list(np.argsort(L)),
            "greedy_es": sorted(range(K), key=lambda k: (-gw[k], L[k])), "kappa": kap,
            **{f"random{r}": list(rng.permutation(K)) for r in range(5)}}


def one(ds, n, rep, pool, y_pool, y_test, names):
    low_priority()
    seed = zlib.crc32(f"prune_{ds}_{n}_{rep}".encode())
    idx = strat(y_pool, n, seed) if n < len(y_pool) else np.random.default_rng(seed).permutation(len(y_pool))
    P = np.stack([pool[m][0][idx] for m in names], 1); y = y_pool[idx]
    Pt = np.stack([pool[m][1] for m in names], 1)
    ords = orders(P, y, seed % (2 ** 31))
    rows, ev = [], {}
    for crit, o in ords.items():
        for m in range(1, len(o) + 1):
            S = tuple(sorted(int(k) for k in o[:m]))
            if S not in ev:
                Ps, Pts = P[:, list(S)], Pt[:, list(S)]
                ev[S] = {"logpool": info.nll(BASELINES["LogPool_fit"](Ps, y, Pts), y_test),
                         "mean": info.nll(Pts.mean(1), y_test)}
            rows.append({"dataset": ds, "n": len(y), "rep": rep, "criterion": crit, "m": m,
                         "subset": json.dumps([names[k] for k in S]), **ev[S]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=5)
    a = ap.parse_args()
    pool, y_pool, y_test = load_pool(a.dataset)
    names = sorted(pool)
    out = os.path.join(ROOT, "results", f"deep_prune_{a.dataset}.csv")
    rows, done = [], set()
    if os.path.exists(out):
        old = pd.read_csv(out); rows = old.to_dict("records"); done = set(zip(old.n, old.rep))
    C = int(y_pool.max() + 1)
    actual = lambda n: len(y_pool) if n >= len(y_pool) else max(n // C, 1) * C
    todo = [(n, r) for n in SIZES[a.dataset] for r in range(REPS if n < len(y_pool) else 1)  # full pool: one run
            if (actual(n), r) not in done]
    todo = todo[:a.limit] if a.limit else todo
    print(f"{len(todo)} jobs -> {out}", flush=True)
    for res in Parallel(n_jobs=a.workers)(delayed(one)(a.dataset, n, r, pool, y_pool, y_test, names) for n, r in todo):
        rows += res
        pd.DataFrame(rows).to_csv(out, index=False)
    print("done", flush=True)


if __name__ == "__main__":
    main()
