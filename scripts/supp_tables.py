"""Supplementary tables: deep pruning, headroom with/without LC correction and bootstrap CIs, rule selection."""
import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "tables")
rng = np.random.default_rng(42)

# --- deep pruning
CR = [("info_fwd", "Information fwd."), ("cmim", "CMIM filter"), ("greedy_es", "Greedy ES"), ("accuracy", "Indiv. accuracy"),
      ("kappa", "$\kappa$ diversity"), ("random", "Random")]
L = [r"\begin{tabular}{@{}llcccccc@{}}", r"\toprule", r"Data set & Fusion & \multicolumn{3}{c}{Log pool, $m=$} & \multicolumn{3}{c}{Mean, $m=$} \\",
     r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}", r" & criterion & 3 & 6 & 8 & 3 & 6 & 8 \\", r"\midrule"]
for ds, lab in (("cifar10", "CIFAR-10"), ("cifar100", "CIFAR-100")):
    d = pd.read_csv(os.path.join(ROOT, "results", f"deep_prune_{ds}.csv")); d["c"] = d.criterion.str.replace(r"random\d", "random", regex=True)
    for n in sorted(d.n.unique()):
        x = d[d.n == n]
        t = {}
        for ev in ("logpool", "mean"):
            ref = x[x.m == 15].groupby("rep")[ev].mean()
            t[ev] = x.assign(r=x[ev] / x.rep.map(ref)).groupby(["c", "m"]).r.mean()
        for i, (k, kl) in enumerate(CR):
            L.append(f"{lab + ' ($n=' + f'{n:,}'.replace(',', '{,}') + '$)' if i == 0 else ''} & {kl} & " + " & ".join(f"{t[ev][(k, m)]:.3f}" for ev in ("logpool", "mean") for m in (3, 6, 8)) + r" \\")
        L.append(r"\midrule")
L = L[:-1] + [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tab_supp_deep_prune.tex"), "w").write("\n".join(L) + "\n")

# --- headroom with / without LC, bootstrap CIs
L = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule", r"Regime & headroom (LC) & recal.\ (LC) & compl.\ (LC) & headroom (no LC) \\", r"\midrule"]
for r, lab in (("n128", "$n=128$"), ("n512", "$n=512$"), ("full", "Full data")):
    u = pd.read_csv(os.path.join(ROOT, "results", f"unified_{r}.csv")); m = u[u.split == "test"].groupby("dataset").mean(numeric_only=True)
    a = pd.read_csv(os.path.join(ROOT, "results", f"acct_{r}.csv")).set_index(["dataset", "fold"])
    p = pd.read_csv(os.path.join(ROOT, "results", f"real_{r}_lr.csv")).set_index(["dataset", "fold"])
    j = a.join(p[["H_all"]].rename(columns={"H_all": "H_plain"}), how="inner")
    j["hp"] = j.L_best - j[["H_plain", "stack_LR_loss", "greedy_ES_loss", "arith_mean_loss", "geo_mean_loss", "L_best"]].min(1)
    mp = j[j.split == "test"].groupby(level=0).hp.mean()
    def ci(v):
        b = [v.values[rng.integers(0, len(v), len(v))].mean() for _ in range(5000)]
        return f"{v.mean():.3f} [{np.percentile(b, 2.5):.3f}, {np.percentile(b, 97.5):.3f}]"
    L.append(f"{lab} & {ci(m.headroom)} & {ci(m.cal)} & {ci(m.ci)} & {ci(mp)} \\\\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tab_supp_lc.tex"), "w").write("\n".join(L) + "\n")
print(open(os.path.join(OUT, "tab_supp_lc.tex")).read())
print(open(os.path.join(OUT, "tab_supp_deep_prune.tex")).read()[:1500])
