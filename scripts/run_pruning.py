"""Pruning experiment (resumable): for every instance, orders from each criterion and test losses of the pruned
committees of sizes 1..K. Usage: python scripts/run_pruning.py --split dev --regime full --limit 4"""
import argparse, json, os, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from fh.data import iter_folds, load_fold, split_of  # noqa: E402
from fh.pruning import orders, eval_subset  # noqa: E402
from run_real import low_priority  # noqa: E402


def one(meta, fold, path, regime, family=""):
    low_priority()
    z = load_fold(path)
    P, y, Pt, yt = z["P_oof"].astype(float), z["y_tr"].astype(int), z["P_te"].astype(float), z["y_te"].astype(int)
    names = [str(m) for m in z["models"]]
    ords, phi = orders(P, y)
    cache, rows = {}, []
    for crit, o in ords.items():
        for m in range(1, len(o) + 1):
            S = tuple(sorted(int(k) for k in o[:m]))
            if S not in cache:
                cache[S] = eval_subset(P, y, Pt, yt, S)
            rows.append({"regime": regime, "family": family, "dataset": meta["name"], "split": split_of(meta["name"]), "fold": fold,
                         "criterion": crit, "m": m, "subset": json.dumps([names[k] for k in S]), **cache[S]})
    rows.append({"regime": regime, "dataset": meta["name"], "split": split_of(meta["name"]), "fold": fold,
                 "criterion": "_shapley", "m": 0, "subset": json.dumps(dict(zip(names, np.round(phi, 6).tolist())))})
    return rows


ap = argparse.ArgumentParser()
ap.add_argument("--split", default="dev")
ap.add_argument("--regime", default="full", choices=["n128", "n512", "full"])
ap.add_argument("--family", default="")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()
low_priority()
out = os.path.join(ROOT, "results", f"prune{a.family}_{a.regime}.csv")
rows, done = [], set()
if os.path.exists(out):
    old = pd.read_csv(out); rows = old.to_dict("records"); done = set(zip(old.dataset, old.fold))
todo = [(m, f, p) for m, f, p in iter_folds(a.split, a.regime, a.family) if (m["name"], f) not in done]
todo = todo[:a.limit] if a.limit else todo
print(f"{len(todo)} folds -> {out}", flush=True)
for s in range(0, len(todo), 8):
    for rr in Parallel(n_jobs=a.workers)(delayed(one)(m, f, p, a.regime, a.family) for m, f, p in todo[s:s + 8]):
        rows += rr
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  {min(s + 8, len(todo))}/{len(todo)}", flush=True)
