"""Does the headroom add predictive power beyond each rule's own OOF CV gain?"""
import glob, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr, rankdata
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULE_RESULTS = os.path.join(ROOT, "data", "rule_results")
SUF = {"n128": "_n128", "n512": "_n512", "full": ""}
def realised(r):
    fs = [f for sp in ("dev", "test") for k in ("cpu", "gpu", "pool", "pretrained") for f in glob.glob(os.path.join(RULE_RESULTS, f"{sp}_{k}_committees{SUF[r]}.csv"))]
    w = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(["dataset", "fold", "method"], keep="last").pivot_table(index=["dataset", "fold"], columns="method", values="logloss")
    return pd.DataFrame({c: w["single_best"] - w[c] for c in w.columns if c != "single_best"})
def pcor(x, y, z):  # Spearman partial correlation of x and y given z
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz); ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return np.corrcoef(ex, ey)[0, 1]
for r in ("full", "n512", "n128"):
    a = pd.read_csv(os.path.join(ROOT, "results", f"acct_{r}.csv")).set_index(["dataset", "fold"])
    j = a.join(realised(r).add_prefix("t_"), how="inner"); j = j[j.split == "test"].groupby(level=0).mean(numeric_only=True)
    for rule, test in (("stack_LR", "t_stack_LR"), ("greedy_ES", "t_greedy_ES")):
        own = j[f"{rule}_gain"]; tg = j[test]
        print(f"{r:5s} {rule:9s} n={len(j)}: rho(own OOF CV gain, test gain)={spearmanr(own, tg).correlation:.2f} | "
              f"rho(headroom, test gain)={spearmanr(j.headroom, tg).correlation:.2f} | partial rho(headroom, test | own)={pcor(j.headroom, tg, own):.2f}")
    for t, lab in (("t_pretrained", "pretrained"), ("t_NE_stack", "NE")):
        if t in j:
            st = j["stack_LR_gain"]
            print(f"{r:5s} {lab:9s}: rho(stack OOF gain, test gain)={spearmanr(st, j[t], nan_policy='omit').correlation:.2f} | rho(headroom)={spearmanr(j.headroom, j[t], nan_policy='omit').correlation:.2f} | partial given stack OOF={pcor(j.headroom.values, j[t].fillna(j[t].median()).values, st.values):.2f}")
