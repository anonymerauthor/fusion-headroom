"""Figure 1: the diversity puzzle on four two-member Gaussian committees with closed-form truth.

Left: classic and label-free diversity (higher = more diverse after sign alignment). Right: true headroom split into
calibration and complementarity. Output: figures/fig_toys.pdf
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from fh.synth import toy
from fh.diversity import all_measures

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "pdf.fonttype": 42,
                     "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
INK, MUTED, GRID = "#1f2430", "#5b6170", "#e3e6eb"
KINDS = [("redundant", "Redundant"), ("noise", "Noisy"), ("complement", "Complementary"),
         ("synergy", "Synergistic")]
C = 3
rng = np.random.default_rng(42)
div, cal, ci, acc2 = {}, {}, {}, {}
for k, _ in KINDS:
    g = toy(k, C)
    t = g.truth([(0,), (1,), (0, 1)], N=200_000)
    ks = int(np.argmin(t["L"]))
    cal[k] = t["L"][ks] - t[(ks,)]
    ci[k] = t[(ks,)] - t[(0, 1)]
    P, y, _ = g.sample(20_000, rng)
    m = all_measures(P, y)
    acc2[k] = (P[:, 1].argmax(1) == y).mean()
    div[k] = {"1 - Q": 1 - m["Q"], "Disagreement": m["disagreement"], "1 - κ": 1 - m["kappa"],
              "JS spread": m["js_spread"]}

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3), gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.35})
ax = axes[0]
names = list(div[KINDS[0][0]].keys())
x = np.arange(len(KINDS)); w = 0.19
cols = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
for j, nm in enumerate(names):
    v = np.array([div[k][nm] for k, _ in KINDS])
    v = v / v.max()
    ax.bar(x + (j - 1.5) * w, v, w * 0.92, color=cols[j], label=nm)
ax.set_xticks(x, [f"{lab}\n(acc$_2$={acc2[k]:.2f})" for k, lab in KINDS])
ax.set_ylabel("diversity (scaled to max = 1)")
ax.legend(frameon=False, ncol=2, fontsize=6.8, loc="upper right", bbox_to_anchor=(1.0, 1.02))
ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.set_title("(a) diversity measures", loc="left", fontsize=8.5, weight="bold", pad=6)
ax.set_ylim(0, 1.32)

ax = axes[1]
c2 = np.array([ci[k] for k, _ in KINDS])
ax.bar(x, c2, 0.55, color="#2a78d6")
for i, (k, _) in enumerate(KINDS):
    ax.text(i, c2[i] + 0.006, f"{c2[i]:.3f}", ha="center", fontsize=7, color=INK)
ax.set_xticks(x, ["Redun-\ndant", "Noisy", "Comple-\nmentary", "Syner-\ngistic"])
ax.set_ylabel("nats")
ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.set_title(r"(b) true headroom $=I(Y;P_{-k^*}\mid P_{k^*})$", loc="left", fontsize=8.5, weight="bold", pad=6)
os.makedirs(os.path.join(ROOT, "outputs", "figures"), exist_ok=True)
fig.savefig(os.path.join(ROOT, "outputs", "figures", "fig_toys.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(ROOT, "outputs", "figures", "fig_toys.png"), dpi=200, bbox_inches="tight")
print({k: round(cal[k] + ci[k], 3) for k, _ in KINDS})
