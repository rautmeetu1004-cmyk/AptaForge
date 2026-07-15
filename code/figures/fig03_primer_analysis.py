#!/usr/bin/env python
"""SELEX constant-region analysis figure, ACS column width (7.2 in), Arial.

Rebuilt from the cached summary (results/25_primer_analysis.json) and the corpus so it does
not require NUPACK. Panels: (A) recurrent 5'/3' flanks, (B) cumulative corpus coverage by
flank family, (C) folding energy of the largest primer family, full sequence vs its
primer-trimmed core (means from the cached NUPACK re-folding).
"""
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.rcParams.update({"font.size": 8, "font.family": "Arial", "figure.dpi": 300,
                     "axes.grid": True, "grid.alpha": 0.25})
BLUE, GREEN, RED, INK, GREY = "#3B6FB0", "#2f7d4f", "#C0473E", "#26303B", "#8A97A3"

J = json.load(open("results/25_primer_analysis.json"))
d = pd.read_csv("data/folding_dedup.csv")
FL = J["flank_len"]; frac = J["frac_corpus_with_primer"]
fam = J["primer_family"]; famn = J["family_n"]
full_m, core_m, shift = J["full_mfe_mean"], J["core_mfe_mean"], J["flank_mfe_shift_mean"]

fig = plt.figure(figsize=(7.2, 3.0)); gs = GridSpec(1, 3, figure=fig, wspace=0.5,
                                                    width_ratios=[1.15, 1, 1.05])

# ---- A: recurrent 5' and 3' flanks --------------------------------------
axA = fig.add_subplot(gs[0, 0])
p5 = J["top_5p_flanks"][:8]; p3 = J["top_3p_flanks"][:8]
y5 = np.arange(len(p5))[::-1] + 0.18
y3 = np.arange(len(p3))[::-1] - 0.18
axA.hlines(y5, 0, [c for _, c in p5], color=BLUE, lw=2, alpha=0.5)
axA.scatter([c for _, c in p5], y5, s=26, color=BLUE, zorder=3, label="5′ flank")
axA.hlines(y3, 0, [c for _, c in p3], color=GREEN, lw=2, alpha=0.5)
axA.scatter([c for _, c in p3], y3, s=26, color=GREEN, marker="s", zorder=3, label="3′ flank")
for i, (s, c) in enumerate(p5):
    axA.text(c + 1.5, np.arange(len(p5))[::-1][i] + 0.18, s, va="center", fontsize=4.6,
             family="monospace", color="#2c4a70")
axA.set_yticks([]); axA.set_xlabel("sequences sharing the flank")
axA.legend(fontsize=6, loc="lower right", frameon=True)
axA.set_title("A", loc="left", fontweight="bold", fontsize=9)

# ---- B: cumulative corpus coverage by 5' flank family -------------------
axB = fig.add_subplot(gs[0, 1])
flanks = d.sequence.str[:FL]
counts = flanks.value_counts()
counts = counts[counts >= 3]                       # a flank shared by >=3 sequences
cum = np.cumsum(counts.values) / len(d) * 100
axB.fill_between(np.arange(1, len(cum) + 1), cum, color=BLUE, alpha=0.2)
axB.plot(np.arange(1, len(cum) + 1), cum, color=BLUE, lw=2)
axB.axhline(frac * 100, color=GREY, ls="--", lw=1.2)
axB.text(len(cum) * 0.55, frac * 100 - 4, f"{frac*100:.0f}% carry a shared flank",
         fontsize=6.5, color=GREY, ha="center", va="top")
axB.set_xlabel("number of 5′ flank families (ranked)")
axB.set_ylabel("cumulative % of corpus")
axB.set_title("B", loc="left", fontweight="bold", fontsize=9)

# ---- C: largest primer family — full sequence vs trimmed core -----------
axC = fig.add_subplot(gs[0, 2])
famseqs = d[d.sequence.str[:FL] == fam]
fullvals = famseqs.mfe_kcal.values
rng = np.random.RandomState(0)
axC.scatter(rng.normal(1, 0.05, len(fullvals)), fullvals, s=20, color=RED, alpha=0.6,
            edgecolors="white", linewidth=0.4, zorder=3, label="full sequence")
axC.hlines(full_m, 0.7, 1.3, color=RED, lw=2)
axC.hlines(core_m, 1.7, 2.3, color=BLUE, lw=2)
axC.scatter([2], [core_m], s=45, color=BLUE, zorder=3, label="primer-trimmed core (mean)")
axC.annotate("", xy=(2, core_m), xytext=(1, full_m),
             arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
axC.text(1.5, (full_m + core_m) / 2, f"Δ = {abs(shift):.1f}\nkcal/mol", ha="center",
         va="center", fontsize=6.8, color=INK,
         bbox=dict(boxstyle="round", fc="white", ec=GREY, alpha=0.9))
axC.set_xlim(0.5, 2.5); axC.set_xticks([1, 2])
axC.set_xticklabels([f"full\n(n={len(fullvals)})", "core\n(mean)"])
axC.set_ylabel("MFE (kcal/mol)")
axC.legend(fontsize=5.6, loc="lower left", frameon=True)
axC.set_title("C", loc="left", fontweight="bold", fontsize=9)

fig.savefig("figures/Figure_03_primer_analysis.png", bbox_inches="tight",
            facecolor="white")
print(f"saved Figure_03_primer_analysis.png  family {fam} n={len(fullvals)} "
      f"full_mean={fullvals.mean():.2f} (cached {full_m}) core_mean={core_m}")
