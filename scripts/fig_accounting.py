"""Figure: where does the headroom go? For each regime and rule, headroom = gain + approximation gap A + estimation
gap E (OOF accounting, mean over data sets). Output: figures/fig_accounting.pdf"""
import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix", "pdf.fonttype": 42,
                     "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
INK, GRID = "#1f2430", "#e3e6eb"
RULES = [("select", "Select\nbest"), ("arith_mean", "Mean"), ("geo_mean", "Geo.\nmean"), ("greedy_ES", "Greedy\nES"),
         ("stack_LR", "Stack-\ning")]
REGS = [("n128", "$n=128$"), ("n512", "$n=512$"), ("full", "Full data")]
COL = {"gain": "#1baf7a", "A": "#eda100", "E": "#e34948"}

fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.4), sharey=True, gridspec_kw={"wspace": 0.08})
for ax, (r, rl) in zip(axes, REGS):
    d = pd.read_csv(os.path.join(ROOT, "results", f"acct_{r}.csv")).query("split == 'test'").groupby("dataset").mean(numeric_only=True)
    d["select_gain"] = d.L_best - d.select_cv_loss
    d["select_A"] = d.headroom  # selecting the best member leaves the whole headroom unused
    d["select_E"] = d.select_cv_loss - d.L_best
    head = d.headroom.mean()
    for i, (k, lab) in enumerate(RULES):
        g, A, E = d[f"{k}_gain"].mean(), d[f"{k}_A"].mean(), d[f"{k}_E"].mean()
        # stack the three non-negative-in-expectation losses from the headroom line downwards
        bottom = head
        for name, v in (("A", A), ("E", E)):
            ax.bar(i, -v, 0.62, bottom=bottom, color=COL[name], label={"A": "approximation gap $A$",
                   "E": "estimation gap $E(n)$"}[name] if (i == 0 and r == "n128") else None)
            bottom -= v
        ax.plot([i - 0.31, i + 0.31], [g, g], color=INK, lw=1.6)
        ax.text(i, g - 0.004, f"{g:+.3f}", ha="center", va="top", fontsize=6.3, color=INK,
                bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
    ax.axhline(head, color="#2a78d6", lw=1.0, ls="--")
    ax.axhline(0, color=INK, lw=0.6)
    ax.text(len(RULES) - 0.6, head + 0.004, f"headroom {head:.3f}", ha="right", fontsize=6.6, color="#2a78d6")
    ax.set_xticks(range(len(RULES)), [lab for _, lab in RULES], fontsize=6.6)
    ax.set_title(rl, fontsize=8.5, weight="bold", loc="left")
    ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
axes[0].set_ylabel("nats (relative to best member)")
axes[0].plot([], [], color=INK, lw=1.6, label="realised gain")
axes[0].set_ylim(-0.14, 0.095)
fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=7, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.07))
fig.savefig(os.path.join(ROOT, "outputs", "figures", "fig_accounting.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(ROOT, "outputs", "figures", "fig_accounting.png"), dpi=200, bbox_inches="tight")
print("ok")
