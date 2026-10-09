"""Tables for the deep-committee section (outputs/tables): tab_deep.tex (prediction of realised gain and headroom
split over 120 random committees per data set and fusion size) and tab_deep_seeds.tex (seed / heterogeneous / full)."""
import os
import numpy as np, pandas as pd
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs", "tables")
RULES = ["arith_mean", "geo_mean", "greedy_ES", "LogPool_fit", "LinPool_fit", "TS_avg", "stack_LR"]
LF = ["div_ambiguity_geo", "div_js_spread", "div_pred_disagreement"]
NAMES = {"difficulty": "difficulty", "entropy_E": "entropy", "GD": "gen.\\ div.", "js_spread": "JS spread",
         "pred_disagreement": "pred.\\ disagr.", "ambiguity_geo": "ambiguity", "double_fault": "double fault",
         "KW": "KW", "kappa": "$\\kappa$", "Q": "$Q$", "rho": "$\\rho$", "disagreement": "disagr.", "CFD": "CFD"}


def load(ds):
    d = pd.read_csv(os.path.join(ROOT, "results", f"deep_{ds}_v2.csv")).drop_duplicates(["committee", "n"], keep="last")
    rules = [r for r in RULES if f"test_{r}" in d]
    for r in rules:
        d[f"g_{r}"] = d.test_single_best - d[f"test_{r}"]
    d["g_best"] = d[[f"g_{r}" for r in rules]].max(1)
    return d, rules


rows, seeds = [], []
for ds, lab in (("cifar10", "CIFAR-10"), ("cifar100", "CIFAR-100")):
    d, rules = load(ds)
    divs = [c for c in d.columns if c.startswith("div_")]
    first = True
    for n in sorted(d.n.unique()):
        x = d[(d.n == n) & d.committee.str.startswith("rand")]
        rho = lambda c: spearmanr(x[c], x.g_best).correlation
        bc = max([c for c in divs if c not in LF], key=lambda c: abs(rho(c)))
        bl = max(LF, key=lambda c: abs(rho(c)))
        vals = [rho("headroom"), rho("ci"), abs(rho(bc)), abs(rho(bl))]
        fmt = [f"\\textbf{{{v:.2f}}}" if v == max(vals) else f"{v:.2f}" for v in vals]
        rows.append(f"{lab if first else ''} & {n:,} & {x.headroom.mean():.3f} & {x.cal.mean():.3f} & {x.ci.mean():.3f} & "
                    f"{x.g_best.mean():.3f} & {fmt[0]} & {fmt[1]} & "
                    f"{fmt[2]} ({NAMES[bc[4:]]}) & {fmt[3]} ({NAMES[bl[4:]]}) \\\\")
        first = False
        if n == d.n.max():
            for c, cl in (("seeds", "5 seeds (ResNet-18)"), ("hetero", "6 architectures"), ("all", "all 15")):
                s = d[(d.n == n) & (d.committee == c)].iloc[0]
                seeds.append(f"{lab} & {cl} & {s.headroom:.3f} & {s.cal:.3f} & {s.ci:.3f} & "
                             f"{100 * s.ci / s.headroom:.0f}\\% & {s.g_LogPool_fit:+.3f} & {s.g_arith_mean:+.3f} & "
                             f"{1 - s.div_Q:.2f} \\\\")
    rows.append(r"\midrule")
rows = rows[:-1]
T = [r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}llcccccccc@{}}", r"\toprule",
     r" & & \multicolumn{4}{c}{Mean over 120 random committees (nats)} & \multicolumn{4}{c}{Spearman $\rho$ with realised gain of the best rule} \\",
     r"\cmidrule(lr){3-6}\cmidrule(lr){7-10}",
     r"Data set & $n$ & $\hat\Delta^*$ & recal. & compl. & best gain & headroom & compl. & best classic & best label-free \\",
     r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular*}"]
os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, "tab_deep.tex"), "w").write("\n".join(T) + "\n")
S = [r"\begin{tabular}{@{}llccccccc@{}}", r"\toprule",
     r"Data set & Committee & $\hat\Delta^*$ & recal. & compl. & compl.\ share & gain (log pool) & gain (mean) & $1-Q$ \\",
     r"\midrule"] + seeds + [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "tab_deep_seeds.tex"), "w").write("\n".join(S) + "\n")
print("\n".join(T)); print("\n".join(S))
