"""Estimator validation (current estimator) with and without learning-curve correction, on random Gaussian
committees with closed-form truth. Writes results/validate_synth_v2.csv."""
import argparse, os, sys
import numpy as np, pandas as pd
from joblib import Parallel, delayed
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from fh.synth import random_committee
from fh.info import decompose


def run(c, ns):
    try:
        import psutil; psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass
    g = random_committee(np.random.default_rng([42, c]))
    full = tuple(range(g.K))
    t = g.truth([(k,) for k in range(g.K)] + [full], N=100_000)
    ks = int(np.argmin(t["L"]))
    tr = {"t_head": t["L"][ks] - t[full], "t_cal": t["L"][ks] - t[(ks,)], "t_ci": t[(ks,)] - t[full]}
    rows = []
    srng = np.random.default_rng([7, c])
    for n in ns:
        P, y, _ = g.sample(n, srng)
        for name, lc in (("lr", False), ("lr_lc", True)):
            d = decompose(P, y, model="lr", lc=lc)
            rows.append({"comm": c, "K": g.K, "C": g.C, "n": n, "est": name, **tr,
                         "e_head": d["headroom"], "e_cal": d["cal"], "e_ci": d["ci"]})
    return rows


ap = argparse.ArgumentParser()
ap.add_argument("--n-comm", type=int, default=20)
ap.add_argument("--ns", default="128,512,2000")
ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()
ns = list(map(int, a.ns.split(",")))
res = Parallel(n_jobs=a.workers)(delayed(run)(c, ns) for c in range(a.n_comm))
df = pd.DataFrame([r for rr in res for r in rr])
df.to_csv(os.path.join(ROOT, "results", "validate_synth_v2.csv"), index=False)
for col in ("head", "ci", "cal"):
    df[f"err_{col}"] = df[f"e_{col}"] - df[f"t_{col}"]
print(df.groupby(["est", "n"])[["err_head", "err_ci", "err_cal"]].agg(["mean", lambda x: np.abs(x).mean()]).round(3))
print("corr(true, est) headroom:", df.groupby(["est", "n"]).apply(lambda x: np.corrcoef(x.t_head, x.e_head)[0, 1]).round(3).to_dict())
