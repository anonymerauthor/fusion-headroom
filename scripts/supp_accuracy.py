"""Supplement: does the log-loss headroom also predict gains in accuracy? (test data sets, unified estimates)"""
import glob, os
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RULE_RESULTS = os.path.join(ROOT, "data", "rule_results")
SUF = {"n128": "_n128", "n512": "_n512", "full": ""}
RULES = [("stack_LR", "Stacking"), ("greedy_ES", "Greedy ES"), ("NE_stack", "Neural ens."), ("pretrained", "Pretrained")]
LF = ["div_ambiguity_geo", "div_js_spread", "div_pred_disagreement"]
rows = []
for r, lab in (("full", "Full data"), ("n512", "$n=512$"), ("n128", "$n=128$")):
    fs = [f for sp in ("dev", "test") for k in ("cpu", "gpu", "pool", "pretrained") for f in glob.glob(os.path.join(RULE_RESULTS, f"{sp}_{k}_committees{SUF[r]}.csv"))]
    w = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(["dataset", "fold", "method"], keep="last").pivot_table(index=["dataset", "fold"], columns="method", values="acc")
    g = pd.DataFrame({c: w[c] - w["single_best"] for c, _ in RULES})
    u = pd.read_csv(os.path.join(ROOT, "results", f"unified_{r}.csv")).set_index(["dataset", "fold"])
    j = u.join(g, how="inner"); m = j[j.split == "test"].groupby(level=0).mean(numeric_only=True)
    divs = [c for c in m.columns if c.startswith("div_")]
    for c, cl in RULES:
        rh = spearmanr(m.headroom, m[c]).correlation
        bc = max([x for x in divs if x not in LF], key=lambda x: abs(spearmanr(m[x], m[c]).correlation))
        bl = max(LF, key=lambda x: abs(spearmanr(m[x], m[c]).correlation))
        rows.append(f"{lab} & {cl} & {100 * m[c].mean():+.2f} & {rh:.2f} & {abs(spearmanr(m[bc], m[c]).correlation):.2f} & {abs(spearmanr(m[bl], m[c]).correlation):.2f} \\\\")
T = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule", r"Regime & Rule & Mean acc.\ gain (pp) & Headroom & Best classic & Best label-free \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}"]
os.makedirs(os.path.join(ROOT, "outputs", "tables"), exist_ok=True)
open(os.path.join(ROOT, "outputs", "tables", "tab_supp_accuracy.tex"), "w").write("\n".join(T) + "\n")
print("\n".join(T))
