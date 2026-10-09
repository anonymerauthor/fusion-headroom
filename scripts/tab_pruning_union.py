"""Table: pruning 14-member committees (test data sets, full data); geometric mean ratio to the full committee."""
import os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(os.path.join(ROOT, "results", "prune_union_full.csv"))
d = d[(d.criterion != "_shapley") & (d.split == "test")].copy()
d["crit"] = d.criterion.str.replace(r"random\d", "random", regex=True)
CR = [("info_fwd", "Information fwd."), ("shapley", "Shapley inform.$^\dagger$"), ("accuracy", "Indiv. accuracy"),
      ("kappa", r"$\kappa$ diversity"), ("random", "Random")]
MS = (3, 6)
tabs = {}
for ev in ("stack", "mean"):
    g = d.groupby(["dataset", "fold", "crit", "m"])[ev].mean().reset_index()
    ref = g[g.m == 14].groupby(["dataset", "fold"])[ev].mean().rename("ref")
    g = g.join(ref, on=["dataset", "fold"]); g["lr"] = np.log(g[ev] / g.ref)
    tabs[ev] = g.groupby(["dataset", "crit", "m"]).lr.mean().unstack(["crit", "m"])
L = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule",
     r" & \multicolumn{2}{c}{Stacking} & \multicolumn{2}{c}{Arithmetic mean} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
     "Criterion & " + " & ".join(f"$m={m}$" for m in MS * 2) + r" \\", r"\midrule"]
for k, lab in CR:
    cells = [lab]
    for ev in ("stack", "mean"):
        t = tabs[ev]
        for m in MS:
            s = f"{np.exp(t[(k, m)].mean()):.3f}"
            if k != "info_fwd":
                a, b = t[("info_fwd", m)], t[(k, m)]
                s += f" ({int((a < b).sum())}/{int((a > b).sum())}{'$^{*}$' if wilcoxon(a, b).pvalue < 0.05 else ''})"
            cells.append(s)
    L.append(" & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(ROOT, "outputs", "tables", "tab_pruning_union.tex"), "w").write("\n".join(L) + "\n")
print("\n".join(L)); print("data sets", d.dataset.nunique())
for ev in ("stack", "mean"):
    t = tabs[ev]; print(ev, {c: [round(float(np.exp(t[(c, m)].mean())), 3) for m in (2, 3, 4, 6, 8, 10)] for c, _ in CR})
