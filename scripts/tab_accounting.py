"""Accounting table (tab_accounting.tex), test data sets."""
import os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "tables")

# ---- accounting (all 63 / 56 data sets; OOF accounting is label-internal, no test leakage)
RULES = [("select", "Select best"), ("arith_mean", "Arithmetic mean"), ("geo_mean", "Geometric mean"),
         ("greedy_ES", "Greedy ensemble sel."), ("stack_LR", "Stacking (log. reg.)")]
L = [r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + "cccc" * 3 + "@{}}", r"\toprule"]
hdr, sub = [""], ["Rule"]
blocks = {}
for r, rl in (("n128", "$n=128$"), ("n512", "$n=512$"), ("full", "Full data")):
    d = pd.read_csv(os.path.join(ROOT, "results", f"acct_{r}.csv")).query("split == 'test'").groupby("dataset").mean(numeric_only=True)
    d["select_gain"], d["select_A"], d["select_E"] = d.L_best - d.select_cv_loss, d.headroom, d.select_cv_loss - d.L_best
    blocks[r] = (d, rl)
L.append(" & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{{rl} ($\Delta^*={d.headroom.mean():.3f}$)}}" for d, rl in blocks.values()) + r" \\")
L.append("".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(3)))
L.append("Rule" + " & Gain & $A$ & $E$ & \%" * 3 + r" \\")
L.append(r"\midrule")
for k, lab in RULES:
    cells = [lab]
    for d, _ in blocks.values():
        g, A, E, h = d[f"{k}_gain"].mean(), d[f"{k}_A"].mean(), d[f"{k}_E"].mean(), d.headroom.mean()
        cells += [f"${g:+.3f}$", f"{A:.3f}", f"{E:.3f}", f"{100 * g / h:.0f}"]
    L.append(" & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular*}"]
open(os.path.join(OUT, "tab_accounting.tex"), "w").write("\n".join(L) + "\n")
print("\n".join(L))

