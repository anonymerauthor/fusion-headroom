"""Table: estimator validation on Gaussian committees with closed-form truth (results/validate_synth_v2.csv)."""
import os
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = pd.read_csv(os.path.join(ROOT, "results", "validate_synth_v2.csv"))
for c in ("head", "ci", "cal"):
    d[f"err_{c}"] = d[f"e_{c}"] - d[f"t_{c}"]
L = [r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
     r"$n$ & Estimator & Bias $\hat\Delta^*$ & MAE $\hat\Delta^*$ & Bias compl. & MAE compl. & Corr. \\", r"\midrule"]
for n in sorted(d.n.unique()):
    for est, lab in (("lr", "plain CV"), ("lr_lc", "learning curve")):
        x = d[(d.n == n) & (d.est == est)]
        L.append(f"{n if est == 'lr' else ''} & {lab} & ${x.err_head.mean():+.3f}$ & {x.err_head.abs().mean():.3f} & "
                 f"${x.err_ci.mean():+.3f}$ & {x.err_ci.abs().mean():.3f} & "
                 f"{np.corrcoef(x.t_head, x.e_head)[0, 1]:.2f} \\\\")
    if n != d.n.max():
        L.append(r"\addlinespace")
L += [r"\bottomrule", r"\end{tabular}"]
os.makedirs(os.path.join(ROOT, "outputs", "tables"), exist_ok=True)
open(os.path.join(ROOT, "outputs", "tables", "tab_synth.tex"), "w").write("\n".join(L) + "\n")
print("\n".join(L)); print("true headroom range:", round(d.t_head.min(), 3), round(d.t_head.max(), 3), round(d.t_head.median(), 3))
