"""Pruning tables after review (test data sets, full data, 8 members).

tab_pruning.tex: geometric mean over data sets of the test log-loss of the pruned committee / full committee for
m = 2, 3, 4 under stacking and under the arithmetic mean, for every criterion, and the W/L of each criterion against the
full committee with a Wilcoxon test (Holm over the criteria of a column).
Also regenerates figures/fig_pruning.pdf (outputs/figures) with the new criteria.
"""
import os
import numpy as np, pandas as pd
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CR = [("info_fwd", "Information fwd.\\ (wrapper)"), ("cmim", "CMIM filter"), ("l1_stack", "L1 stacking"),
      ("greedy_es", "Greedy ES"), ("accuracy", "Indiv.\\ accuracy"), ("kappa", "$\\kappa$ diversity"), ("random", "Random")]
MS = (2, 3, 4)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for i, k in enumerate(o):
        run = max(run, min(1, (len(p) - i) * p[k])); adj[k] = run
    return adj


d = pd.read_csv(os.path.join(ROOT, "results", "prune_full.csv"))
d = d[(d.criterion != "_shapley") & (d.split == "test")].copy()
d["crit"] = d.criterion.str.replace(r"random\d", "random", regex=True)
res = {}
for ev in ("stack", "mean"):
    g = d.groupby(["dataset", "fold", "crit", "m"])[ev].mean().reset_index()
    ref = g[g.m == 8].groupby(["dataset", "fold"])[ev].mean().rename("ref")
    g = g.join(ref, on=["dataset", "fold"]); g["lr"] = np.log(g[ev] / g.ref)
    res[ev] = g.groupby(["dataset", "crit", "m"]).lr.mean().unstack(["crit", "m"])
cells = {}
for ev in ("stack", "mean"):
    t = res[ev]
    for m in MS:
        ps = [wilcoxon(t[(k, m)]).pvalue for k, _ in CR]
        adj = holm(ps)
        for (k, _), p in zip(CR, adj):
            v = t[(k, m)]
            gm = np.exp(v.mean()); w, l = int((v < 0).sum()), int((v > 0).sum())
            cells[(ev, m, k)] = f"{gm:.3f} ({w}/{l}{'$^{*}$' if p < 0.05 else ''})"
L = [r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lcccccc@{}}", r"\toprule",
     r" & \multicolumn{3}{c}{Fused by stacking} & \multicolumn{3}{c}{Fused by the arithmetic mean} \\",
     r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
     "Criterion & " + " & ".join(f"$m={m}$" for m in MS * 2) + r" \\", r"\midrule"]
for k, lab in CR:
    L.append(lab + " & " + " & ".join(cells[(ev, m, k)] for ev in ("stack", "mean") for m in MS) + r" \\")
L += [r"\bottomrule", r"\end{tabular*}"]
open(os.path.join(ROOT, "outputs", "tables", "tab_pruning.tex"), "w").write("\n".join(L) + "\n")
print("\n".join(L))

# figure
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "pdf.fonttype": 42,
                     "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
COL = {"info_fwd": ("#2a78d6", "o"), "cmim": ("#4a3aa7", "D"), "l1_stack": ("#e34948", "v"), "greedy_es": ("#eda100", "P"),
       "accuracy": ("#1baf7a", "s"), "kappa": ("#eb6834", "^"), "random": ("#8a8f99", "x")}
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), gridspec_kw={"wspace": 0.28})
for ax, ev, title in zip(axes, ("stack", "mean"), ("(a) fused by stacking", "(b) fused by the arithmetic mean")):
    t = np.exp(res[ev].mean())
    for k, lab in CR:
        c, mk = COL[k]
        ax.plot(range(1, 9), [t[(k, m)] for m in range(1, 9)], color=c, marker=mk, ms=3.3,
                lw=1.4 if k in ("info_fwd", "cmim") else 1.0, label=lab.replace("\\ ", " ").replace("$\\kappa$", "κ"))
    ax.axhline(1, color="#1f2430", lw=0.6, ls="--")
    ax.set_xlabel("committee size $m$"); ax.set_xticks(range(1, 9))
    ax.set_ylim(0.8, 1.6) if ev == "stack" else ax.set_ylim(0.55, 1.4)
    ax.grid(color="#e3e6eb", lw=0.5); ax.set_axisbelow(True)
    ax.set_title(title, loc="left", fontsize=8.5, weight="bold")
axes[0].set_ylabel("test log-loss / full committee")
axes[1].legend(frameon=False, fontsize=6.3, loc="upper right", ncol=2)
for out in (os.path.join(ROOT, "outputs", "figures", "fig_pruning.pdf"), os.path.join(ROOT, "outputs", "figures", "fig_pruning.pdf")):
    fig.savefig(out, bbox_inches="tight")
fig.savefig(os.path.join(ROOT, "outputs", "figures", "fig_pruning.png"), dpi=200, bbox_inches="tight")
