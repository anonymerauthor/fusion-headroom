"""Cross-validated single-best selection for every instance (selection is a learned rule with its own E)."""
import os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from fh.data import iter_folds, load_fold
from fh.info import _folds, nll

for r in ("n128", "n512", "full"):
    f = os.path.join(ROOT, "results", f"acct_{r}.csv")
    d = pd.read_csv(f).set_index(["dataset", "fold"])
    vals = {}
    for meta, fold, path in iter_folds("all", r):
        z = load_fold(path); P, y = z["P_oof"].astype(float), z["y_tr"].astype(int)
        Q = np.zeros((len(y), P.shape[2]))
        for a, b in _folds(y, 5, 42):
            k = int(np.argmin([nll(P[a, j], y[a]) for j in range(P.shape[1])]))
            Q[b] = P[b, k]
        vals[(meta["name"], fold)] = nll(Q, y)
    d["select_cv_loss"] = pd.Series(vals)
    d["select_E"] = d.select_cv_loss - d.L_best
    d.reset_index().to_csv(f, index=False)
    print(r, "mean selection cost E (nats):", round(d.groupby(level=0).select_E.mean().mean(), 4))
