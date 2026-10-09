"""Robustness of the tabular analysis: per committee instance (full data, test data sets)
  H_all_hgb      H(Y|P) bound from gradient boosting on the members' log-probabilities (a family that no fusion rule uses)
  H_all_eps6     H(Y|P) bound from logistic recalibration with clipping 1e-6 instead of 1e-3
  oracle_acc     share of rows on which at least one member is correct (label-using baseline predictor)
  loss_spread    max - min raw member log-loss; L_best; mean member log-loss
Output: results/robust_full.csv (resumable)."""
import argparse, os, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "scripts"))
import fh.info as info  # noqa: E402
from fh.data import iter_folds, load_fold, split_of  # noqa: E402
from run_real import low_priority  # noqa: E402


def one(meta, fold, path):
    low_priority()
    z = load_fold(path); P, y = z["P_oof"].astype(float), z["y_tr"].astype(int)
    K, C = P.shape[1], P.shape[2]
    L = np.array([info.nll(P[:, k], y) for k in range(K)])
    h_hgb = info.cond_entropy(P, y, range(K), C, model="hgb")
    info.FEAT_EPS = 1e-6
    h6 = info.cond_entropy(P, y, range(K), C, model="lr")
    info.FEAT_EPS = 1e-3
    return {"dataset": meta["name"], "split": split_of(meta["name"]), "fold": fold, "n": len(y),
            "H_all_hgb": h_hgb, "H_all_eps6": h6, "oracle_acc": float((P.argmax(-1) == y[:, None]).any(1).mean()),
            "loss_spread": float(L.max() - L.min()), "L_best": float(L.min()), "L_mean": float(L.mean())}


ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); a = ap.parse_args()
low_priority()
out = os.path.join(ROOT, "results", "robust_full.csv")
rows, done = [], set()
if os.path.exists(out):
    old = pd.read_csv(out); rows = old.to_dict("records"); done = set(zip(old.dataset, old.fold))
todo = [(m, f, p) for m, f, p in iter_folds("test", "full") if (m["name"], f) not in done]
print(len(todo), "jobs", flush=True)
for s in range(0, len(todo), 8):
    rows += Parallel(n_jobs=a.workers)(delayed(one)(m, f, p) for m, f, p in todo[s:s + 8])
    pd.DataFrame(rows).to_csv(out, index=False); print(f"  {min(s + 8, len(todo))}/{len(todo)}", flush=True)
