"""Choosing the fusion rule from the OOF rows. Candidates: select-best, mean, greedy ES, stacking.
Selectors: always-X, CV (lowest OOF CV loss), LC (learning-curve loss extrapolated to the deployment size n),
headroom threshold (stack if estimated headroom > tau else select-best; tau tuned on dev), oracle (best on test).
Evaluation: realised test log-loss of the chosen rule (rule results), relative to arithmetic mean."""
import glob, os, sys
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from fh.data import RULE_RESULTS
SUF = {"n128": "_n128", "n512": "_n512", "full": ""}
MAP = {"select": "single_best", "arith_mean": "arith_mean", "greedy_ES": "greedy_ES", "stack_LR": "stack_LR"}

for r in ("n128", "n512", "full"):
    a = pd.read_csv(os.path.join(ROOT, "results", f"acct_{r}.csv")).set_index(["dataset", "fold"])
    fs = [f for sp in ("dev", "test") for f in glob.glob(os.path.join(RULE_RESULTS, f"{sp}_cpu_committees{SUF[r]}.csv"))]
    t = pd.concat([pd.read_csv(f) for f in fs]).pivot_table(index=["dataset", "fold"], columns="method", values="logloss")
    j = a.join(t[list(MAP.values())].add_prefix("test_"), how="inner")
    cv = pd.DataFrame({"select": j.select_cv_loss, "arith_mean": j.arith_mean_loss, "greedy_ES": j.greedy_ES_loss,
                       "stack_LR": j.stack_LR_loss})
    m = j.n.astype(float)
    lc = pd.DataFrame({"select": j.select_cv_loss, "arith_mean": j.arith_mean_loss,
                       "greedy_ES": j.H_all + j.greedy_ES_A + j.greedy_ES_b / m,
                       "stack_LR": j.H_all + j.stack_LR_A + j.stack_LR_b / m})
    test = pd.DataFrame({k: j[f"test_{v}"] for k, v in MAP.items()})
    pick = {"CV": cv.idxmin(1), "LC": lc.idxmin(1)}
    dev = j.split == "dev"
    best_tau, best = 0, np.inf
    for tau in np.linspace(0, 0.1, 41):
        ch = np.where(j.headroom > tau, "stack_LR", "select")
        l = np.array([test.loc[i, c] for i, c in zip(j.index, ch)])[dev.values].mean()
        if l < best:
            best, best_tau = l, tau
    pick["headroom>tau"] = pd.Series(np.where(j.headroom > best_tau, "stack_LR", "select"), index=j.index)
    loss = {f"always_{k}": test[k] for k in MAP}
    for k, p in pick.items():
        loss[k] = pd.Series([test.loc[i, c] for i, c in zip(j.index, p)], index=j.index)
    loss["oracle"] = test.min(1)
    L = pd.DataFrame(loss)[~dev]
    rel = L.div(L.always_arith_mean, axis=0).groupby(level=0).mean()
    print(f"\n=== {r}: test data sets {rel.shape[0]}, tau*={best_tau:.3f} (dev) | mean rel. test log-loss (vs mean) ===")
    s = rel.mean().sort_values()
    for k, v in s.items():
        extra = ""
        if k in ("LC", "headroom>tau", "always_stack_LR", "always_greedy_ES"):
            d = L.groupby(level=0).mean()
            extra = f"  vs CV W/L {(d[k] < d.CV).sum()}/{(d[k] > d.CV).sum()} p={wilcoxon(d[k], d.CV).pvalue:.2g}" if k != "CV" else ""
        print(f"  {k:18s} {v:.4f}{extra}")
    print("  choice shares CV:", pick["CV"][~dev].value_counts(normalize=True).round(2).to_dict())
