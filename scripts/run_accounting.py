"""Gain accounting (Theorem 2) for every committee instance (resumable). Usage:
    python scripts/run_accounting.py --split all --regime n128 --workers 4
"""
import argparse, os, sys
import pandas as pd
from joblib import Parallel, delayed
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from fh.data import iter_folds, load_fold, split_of  # noqa: E402
from fh.accounting import account  # noqa: E402
from run_real import low_priority  # noqa: E402


def one(meta, fold, path, regime):
    low_priority()
    z = load_fold(path)
    o = account(z["P_oof"].astype(float), z["y_tr"].astype(int))
    return {"regime": regime, "dataset": meta["name"], "split": split_of(meta["name"]), "fold": fold,
            "n": len(z["y_tr"]), "K": z["P_oof"].shape[1], "C": z["P_oof"].shape[2], **o}


ap = argparse.ArgumentParser()
ap.add_argument("--split", default="dev")
ap.add_argument("--regime", default="full", choices=["n128", "n512", "full"])
ap.add_argument("--family", default="")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()
low_priority()
out = os.path.join(ROOT, "results", f"acct{a.family}_{a.regime}.csv")
rows, done = [], set()
if os.path.exists(out):
    old = pd.read_csv(out); rows = old.to_dict("records"); done = set(zip(old.dataset, old.fold))
todo = [(m, f, p) for m, f, p in iter_folds(a.split, a.regime, a.family) if (m["name"], f) not in done]
todo = todo[:a.limit] if a.limit else todo
print(f"{len(todo)} folds -> {out}", flush=True)
for s in range(0, len(todo), 8):
    rows += Parallel(n_jobs=a.workers)(delayed(one)(m, f, p, a.regime) for m, f, p in todo[s:s + 8])
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  {min(s + 8, len(todo))}/{len(todo)}", flush=True)
