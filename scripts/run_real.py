"""Headroom decomposition and diversity measures for every real committee instance (resumable).

One row per (regime, data set, fold): raw member losses, recalibrated entropies, headroom terms on the OOF rows,
classic and label-free diversity measures (OOF and test rows). Usage:

    python scripts/run_real.py --split dev --regime full --limit 10      # smoke test
    python scripts/run_real.py --split all --regime n128 --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from fh.data import iter_folds, load_fold, split_of  # noqa: E402
from fh.diversity import all_measures  # noqa: E402
from fh.info import decompose, nll  # noqa: E402


def low_priority():
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS if os.name == "nt" else 10)
    except Exception:
        pass


def one(meta, fold, path, regime, model, lc=False):
    low_priority()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    z = load_fold(path)
    P, y = z["P_oof"].astype(float), z["y_tr"].astype(int)
    Pt, yt = z["P_te"].astype(float), z["y_te"].astype(int)
    names = [str(m) for m in z["models"]]
    d = decompose(P, y, model=model, lc=lc)
    row = {"regime": regime, "dataset": meta["name"], "split": split_of(meta["name"]), "fold": fold,
           "n": len(y), "K": P.shape[1], "C": P.shape[2], "model": model + ("_lc" if lc else ""),
           "H_y": d["H_y"], "H_all": d["H_all"], "k_star": names[d["k_star"]],
           "headroom": d["headroom"], "cal": d["cal"], "ci": d["ci"], "I_all": d["I_all"],
           "L_json": json.dumps(dict(zip(names, np.round(d["L"], 6).tolist()))),
           "H_json": json.dumps(dict(zip(names, np.round(d["H"], 6).tolist()))),
           "L_test_json": json.dumps({m: round(nll(Pt[:, k], yt), 6) for k, m in enumerate(names)})}
    row.update({f"div_{k}": v for k, v in all_measures(P, y).items()})
    row.update({f"divte_{k}": v for k, v in all_measures(Pt, yt).items()})
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--regime", default="full", choices=["n128", "n512", "full"])
    ap.add_argument("--family", default="", help="'' (8 members), '_alt' or '_union'")
    ap.add_argument("--model", default="lr")
    ap.add_argument("--lc", action="store_true", help="learning-curve correction of H(Y|P_k*), H(Y|P)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    low_priority()
    out = os.path.join(ROOT, "results", f"real{a.family}_{a.regime}_{a.model}{'_lc' if a.lc else ''}.csv")
    done = set()
    rows = []
    if os.path.exists(out):
        old = pd.read_csv(out)
        rows = old.to_dict("records")
        done = set(zip(old.dataset, old.fold))
    todo = [(m, f, p) for m, f, p in iter_folds(a.split, a.regime, a.family) if (m["name"], f) not in done]
    if a.limit:
        todo = todo[:a.limit]
    print(f"{len(todo)} folds to do -> {out}", flush=True)
    chunk = max(a.workers * 2, 1)
    for s in range(0, len(todo), chunk):
        new = Parallel(n_jobs=a.workers)(delayed(one)(m, f, p, a.regime, a.model, a.lc) for m, f, p in todo[s:s + chunk])
        rows += new
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  {min(s + chunk, len(todo))}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
