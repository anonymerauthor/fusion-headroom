"""Figure: predicting realised fusion gain. (a) Spearman correlation (per data set) of each predictor with the realised
test gain of the best rule, full data and n=512 (LC-corrected). (b) Scatter, full data: estimated headroom vs realised
gain of the best rule. Output: figures/fig_rq3.pdf"""
import glob, os, sys
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from fh.data import RULE_RESULTS
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "pdf.fonttype": 42,
                     "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
INK, MUTED, GRID = "#1f2430", "#5b6170", "#e3e6eb"
SUF = {"n512": "_n512", "full": ""}
PRED = [("headroom", "Headroom $\\hat\\Delta^*$"), ("ci", "Complementarity"), ("cal", "Calibration"),
        ("div_js_spread", "JS spread*"), ("div_ambiguity_geo", "Ambiguity*"), ("div_pred_disagreement", "Pred. disagr.*"),
        ("div_difficulty", "Difficulty"), ("div_double_fault", "Double fault"), ("div_disagreement", "Disagreement"),
        ("div_KW", "Kohavi-Wolpert"), ("div_entropy_E", "Entropy E"), ("div_kappa", "Kappa"), ("div_rho", "Correlation"),
        ("div_Q", "Q statistic"), ("div_GD", "Gen. diversity"), ("div_CFD", "Coinc. failure")]


def realised(regime):
    fs = [f for sp in ("dev", "test") for k in ("cpu", "gpu", "pool", "pretrained")
          for f in glob.glob(os.path.join(RULE_RESULTS, f"{sp}_{k}_committees{SUF[regime]}.csv"))]
    df = pd.concat([pd.read_csv(f) for f in fs]).drop_duplicates(["dataset", "fold", "method"], keep="last")
    w = df.pivot_table(index=["dataset", "fold"], columns="method", values="logloss")
    rules = [c for c in w.columns if c != "single_best" and not c.startswith("TabPFN")]
    return (w["single_best"].values[:, None] - w[rules]).max(1).rename("best_gain")


data = {}
for r, est in (("full", "lr"), ("n512", "lr_lc")):
    d = pd.read_csv(os.path.join(ROOT, "results", f"real_{r}_{est}.csv")).set_index(["dataset", "fold"])
    u = pd.read_csv(os.path.join(ROOT, "results", f"unified_{r}.csv")).set_index(["dataset", "fold"])
    d[["headroom", "ci", "cal"]] = u[["headroom", "ci", "cal"]]  # consistent estimates (tightest bound)
    j = d.join(realised(r), how="inner")
    data[r] = j[j.split == "test"].groupby(level=0).mean(numeric_only=True)

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9), gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.42})
ax = axes[0]
y = np.arange(len(PRED))[::-1]
for j, (r, col, lab) in enumerate((("full", "#2a78d6", "full data"), ("n512", "#eda100", "$n=512$"))):
    rho = [spearmanr(data[r][p], data[r].best_gain).correlation for p, _ in PRED]
    ax.barh(y + (0.2 if j == 0 else -0.2), rho, 0.38, color=col, label=lab)
ax.axvline(0, color=INK, lw=0.6)
ax.axhspan(len(PRED) - 3.5, len(PRED) - 0.5, color="#eaf2fc", zorder=0)
ax.set_yticks(y, [lab for _, lab in PRED], fontsize=7)
ax.set_xlabel("Spearman $\\rho$ with realised gain of the best rule")
ax.legend(frameon=False, fontsize=7, loc="lower right")
ax.grid(axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.set_title("(a) information terms vs. diversity (*: label-free)", loc="left", fontsize=8.5, weight="bold")

ax = axes[1]
d = data["full"]
ax.scatter(d.headroom, d.best_gain, s=12, color="#2a78d6", alpha=0.75, lw=0)
lim = [min(d.headroom.min(), d.best_gain.min()) - 0.005, max(d.headroom.max(), d.best_gain.max()) + 0.01]
ax.plot(lim, lim, color=MUTED, lw=0.8, ls="--")
ax.text(lim[1] * 0.98, lim[1] * 0.9, "gain = headroom", ha="right", fontsize=6.5, color=MUTED, rotation=37)
ax.set_xlabel("estimated headroom (OOF, nats)"); ax.set_ylabel("realised test gain of best rule")
rho = spearmanr(d.headroom, d.best_gain).correlation
ax.text(0.04, 0.95, f"$\\rho$ = {rho:.2f}, {len(d)} test data sets", transform=ax.transAxes, va="top", fontsize=7)
ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.set_title("(b) full data", loc="left", fontsize=8.5, weight="bold")
fig.savefig(os.path.join(ROOT, "outputs", "figures", "fig_rq3.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(ROOT, "outputs", "figures", "fig_rq3.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(ROOT, "outputs", "figures", "fig_rq3.png"), dpi=200, bbox_inches="tight")
print("ok")
