"""Read-only access to the committees built by scripts/build_committees.py.

Each fold file holds out-of-fold (OOF) member probabilities on the training rows and probabilities of the
refitted members on the test rows: P_oof (n, K, C), P_te (m, K, C), y_tr, y_te, X_tr, X_te, models.
Data sets are split into development and test sets by a fixed md5 rule, so that all estimator design is done
on development data sets only.
The committee directory defaults to data/ and can be changed with the environment variable FH_COMMITTEES.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RULE_RESULTS = os.path.join(ROOT, "data", "rule_results")
COMMITTEES = os.environ.get("FH_COMMITTEES", os.path.join(ROOT, "data"))
REGIMES = {"n128": "committees_n128", "n512": "committees_n512", "full": "committees"}
# test data sets whose outer training folds have at most 512 rows (n=512 coincides with full data)
SMALL = {"kc2", "balance-scale", "ilpd", "wdbc", "dresses-sales", "climate-model-simulation-crashes", "cylinder-bands"}


def split_of(name: str) -> str:
    return "dev" if int(hashlib.md5(name.encode()).hexdigest(), 16) % 4 == 0 else "test"


def iter_folds(split="dev", regime="full", family=""):
    """Yield (meta, fold, path). family: '' (8 members), '_alt' (6 additional learners) or '_union' (14)."""
    sub = REGIMES[regime].replace("committees", "committees" + family)
    for d in sorted(glob.glob(os.path.join(COMMITTEES, sub, "*"))):
        mp = os.path.join(d, "meta.json")
        if not os.path.exists(mp):
            continue
        meta = json.load(open(mp))
        if split != "all" and split_of(meta["name"]) != split:
            continue
        if regime == "n512" and meta["name"] in SMALL:
            continue
        for f in sorted(glob.glob(os.path.join(d, "fold*.npz"))):
            yield meta, int(f[-5]), f


def load_fold(path):
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}
