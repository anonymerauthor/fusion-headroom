"""Deep committees: headroom decomposition, diversity and realised gains for committees drawn from the CIFAR pool.

Committees: random subsets of the 15-member pool (K in {3, 5, 8}, 40 each), the seed ensemble (5 x ResNet-18),
a heterogeneous committee (one member per architecture) and the full pool. Fusion rows: stratified sub-samples of the
10k fusion pool (sizes per data set below); test rows: the official test set.
Estimator: structured log-pool recalibration ("logpool"; K + C parameters), learning-curve corrected for small n.
Rules (fitted on the fusion rows, evaluated on the test rows): select best, arithmetic / geometric mean, greedy ensemble
selection, fitted log pool, fitted linear pool, temperature scaling + mean, and (C = 10 only) stacking with LR.
Output: results/deep_<dataset>.csv (resumable). Usage: python scripts/deep_analyze.py --dataset cifar10 --limit 3
"""
import argparse, glob, json, os, re, sys, zlib
import numpy as np, pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from fh.info import decompose, nll  # noqa: E402
from fh.diversity import all_measures  # noqa: E402
from fh.rules import BASELINES  # noqa: E402
from run_real import low_priority  # noqa: E402

SIZES = {"cifar10": (128, 512, 10000), "cifar100": (1000, 2500, 10000)}
RULES = ["single_best", "arith_mean", "geo_mean", "greedy_ES", "LogPool_fit", "LinPool_fit", "TS_avg", "stack_LR"]
HETERO = ["resnet18_s0", "wrn16_4_s0", "vgg11_s0", "mobilenetv2_s0", "densenet_s0", "simplecnn_s0"]


def load_pool(ds):
    pool = {}
    for f in sorted(glob.glob(os.path.join(ROOT, "runs", "deep", ds, "*.npz"))):
        if os.path.basename(f).startswith("_"):
            continue
        z = np.load(f)
        pool[os.path.basename(f)[:-4]] = (z["P_pool"].astype(np.float64), z["P_test"].astype(np.float64))
        y_pool, y_test = z["y_pool"].astype(int), z["y_test"].astype(int)
    return pool, y_pool, y_test


def committees(names, seed=42):
    rng = np.random.default_rng(seed)
    out = []
    for K in (3, 5, 8):
        for r in range(40):
            out.append((f"rand{K}_{r}", sorted(rng.choice(names, K, replace=False).tolist())))
    seeds = [n for n in names if re.fullmatch(r"resnet18_s\d+", n)]  # seed ensemble only (not resnet18_short)
    out += [("seeds", seeds), ("hetero", [n for n in HETERO if n in names]), ("all", sorted(names))]
    return out


def strat(y, n, seed):
    rng = np.random.default_rng(seed)
    if n >= len(y):
        return np.arange(len(y))
    C = y.max() + 1
    per = max(n // C, 1)
    return np.sort(np.concatenate([rng.choice(np.where(y == c)[0], per, replace=False) for c in range(C)]))


def one(ds, cname, members, n, pool, y_pool, y_test):
    low_priority()
    idx = strat(y_pool, n, seed=zlib.crc32(f"{cname}_{n}".encode()))  # deterministic across processes
    P = np.stack([pool[m][0][idx] for m in members], 1)
    y = y_pool[idx]
    Pt = np.stack([pool[m][1] for m in members], 1)
    C = P.shape[-1]
    d = decompose(P, y, model="logpool", lc=n < 10000, eps_grid=(1e-3, 1e-6))
    row = {"dataset": ds, "committee": cname, "members": json.dumps(members), "K": len(members), "n": len(y),
           "headroom": d["headroom"], "cal": d["cal"], "ci": d["ci"], "I_all": d["I_all"], "H_all": d["H_all"],
           "k_star": members[d["k_star"]], "L_best_fusion": float(d["L"][d["k_star"]]), "eps_all": d["eps_all"]}
    row.update({f"div_{k}": v for k, v in all_measures(P, y).items()})
    for r in RULES:
        if r == "stack_LR" and C > 10:
            continue
        try:
            Q = BASELINES[r](P, y, Pt)
            row[f"test_{r}"] = nll(Q, y_test)
        except Exception as e:  # noqa: BLE001
            row[f"test_{r}"] = np.nan
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cifar10")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    pool, y_pool, y_test = load_pool(a.dataset)
    names = sorted(pool)
    print(f"{len(names)} members:", names, flush=True)
    out = os.path.join(ROOT, "results", f"deep_{a.dataset}_v2.csv")
    rows, done = [], set()
    if os.path.exists(out):
        old = pd.read_csv(out).drop_duplicates(["committee", "n"], keep="last")
        rows = old.to_dict("records"); done = set(zip(old.committee, old.n))
    C = int(y_pool.max() + 1)
    actual = lambda n: len(y_pool) if n >= len(y_pool) else max(n // C, 1) * C  # size after stratified sampling
    todo = [(c, m, n) for c, m in committees(names) for n in SIZES[a.dataset] if (c, actual(n)) not in done]
    todo = todo[:a.limit] if a.limit else todo
    print(f"{len(todo)} jobs -> {out}", flush=True)
    for s in range(0, len(todo), 16):
        rows += Parallel(n_jobs=a.workers)(delayed(one)(a.dataset, c, m, n, pool, y_pool, y_test)
                                           for c, m, n in todo[s:s + 16])
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  {min(s + 16, len(todo))}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
