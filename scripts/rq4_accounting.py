"""Where does the fusion headroom go? Aggregates results/acct_*.csv.

1) Per regime and rule: mean headroom = gain + approximation gap A + estimation gap E (OOF, internally consistent).
2) Out-of-sample test of the cross-over prediction: from the n=512 learning curves predict n* (rows at which stacking
   overtakes the arithmetic mean) and check, per data set, whether stacking beats the mean on the *test* rows of the
   full-data committees (rule results) when the full training size exceeds n*.
"""
import glob, os, sys
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from fh.data import RULE_RESULTS

RULES = ["arith_mean", "geo_mean", "greedy_ES", "stack_LR"]
rows = []
acc = {}
for r in ("n128", "n512", "full"):
    f = os.path.join(ROOT, "results", f"acct_{r}.csv")
    if not os.path.exists(f):
        continue
    d = pd.read_csv(f)
    acc[r] = d
    m = d.groupby("dataset").mean(numeric_only=True)      # data-set level, then average
    for rule in RULES:
        rows.append({"regime": r, "rule": rule, "headroom": m.headroom.mean(), "gain": m[f"{rule}_gain"].mean(),
                     "A": m[f"{rule}_A"].mean(), "E": m[f"{rule}_E"].mean(), "n_sets": len(m)})
tab = pd.DataFrame(rows)
print("=== accounting (nats, mean over data sets): headroom = gain + A + E ===")
print(tab.round(4).to_string(index=False))
tab.to_csv(os.path.join(ROOT, "results", "rq4_accounting_table.csv"), index=False)

if "n512" in acc:
    d = acc["n512"].groupby("dataset").agg(b=("stack_LR_b", "median"),
                                           dA=("arith_mean_A", "median"), dS=("stack_LR_A", "median"))
    d["nstar"] = np.where(d.dA - d.dS > 1e-4, d.b / (d.dA - d.dS).clip(lower=1e-4), np.inf)
    fs = glob.glob(os.path.join(RULE_RESULTS, "*_cpu_committees.csv"))
    t = pd.concat([pd.read_csv(x) for x in fs]).pivot_table(index="dataset", columns="method", values="logloss")
    full_n = pd.concat([pd.read_csv(os.path.join(ROOT, "results", "real_full_lr.csv"))]).groupby("dataset").n.mean()
    j = d.join(t[["arith_mean", "stack_LR"]], how="inner").join(full_n.rename("n_full"), how="inner")
    j["m_full"] = j.n_full * 0.8
    j["pred_stack_wins"] = j.m_full > j.nstar
    j["obs_stack_wins"] = j.stack_LR < j.arith_mean
    acc_rate = (j.pred_stack_wins == j.obs_stack_wins).mean()
    base = max(j.obs_stack_wins.mean(), 1 - j.obs_stack_wins.mean())
    print(f"\n=== cross-over prediction from n=512 curves ({len(j)} data sets) ===")
    print(f"accuracy {acc_rate:.2f} (majority baseline {base:.2f}); predicted wins {j.pred_stack_wins.mean():.2f}, "
          f"observed wins {j.obs_stack_wins.mean():.2f}")
    print(pd.crosstab(j.pred_stack_wins, j.obs_stack_wins, rownames=["pred"], colnames=["obs"]))
    j.to_csv(os.path.join(ROOT, "results", "rq4_crossover.csv"))
