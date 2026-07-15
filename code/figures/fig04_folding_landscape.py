#!/usr/bin/env python
"""Folding-energy landscape of the modelling corpus (n = 1,426 unique sequences).

Regenerated with explicit panel letters and Pearson correlations computed on the
deduplicated corpus, so the figure matches the values quoted in the text (-0.58, -0.11).
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import pearsonr

BLUE, RED, INK = "#3B6FB0", "#C0473E", "#26303B"
plt.rcParams.update({"font.size": 8, "font.family": "Arial", "figure.dpi": 300, "axes.grid": False,
                     "axes.spines.top": False, "axes.spines.right": False})

d = pd.read_csv("data/folding_dedup.csv")
n = len(d)
rL = pearsonr(d.mfe_kcal, d.L)[0]
rG = pearsonr(d.mfe_kcal, d.gc)[0]
med = np.median(d.mfe_kcal); nofold = (d.mfe_kcal == 0).mean()

fig, ax = plt.subplots(1, 3, figsize=(6.7, 2.5))
fig.subplots_adjust(wspace=0.5, top=0.74, bottom=0.2, left=0.07, right=0.97)

def tag(a, L, t=""):
    a.set_title(L, loc="left", fontweight="bold", fontsize=9, pad=4)

# A: folding-energy distribution
ax[0].hist(d.mfe_kcal, bins=45, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.3)
ax[0].axvline(med, color=RED, ls="--", lw=1.8, label=f"median = {med:.1f}")
ax[0].axvline(0, color="grey", ls=":", lw=1.4, label=f"no fold: {nofold*100:.1f}%")
ax[0].set_xlabel("minimum free energy (kcal/mol)"); ax[0].set_ylabel("count")
ax[0].legend(frameon=False, fontsize=6.5, loc="upper left")
tag(ax[0], "A", "Real folding-energy distribution")

# B: MFE vs length
hb = ax[1].hexbin(d.L, d.mfe_kcal, gridsize=30, cmap="viridis", mincnt=1)
fig.colorbar(hb, ax=ax[1], fraction=0.046, pad=0.04, label="count")
ax[1].set_xlabel("length (nt)"); ax[1].set_ylabel("MFE (kcal/mol)")
tag(ax[1], "B"); ax[1].text(0.96, 0.06, f"Pearson r = {rL:.2f}", transform=ax[1].transAxes, ha="right", va="bottom", fontsize=7)

# C: MFE vs GC
hb2 = ax[2].hexbin(d.gc, d.mfe_kcal, gridsize=30, cmap="magma", mincnt=1)
fig.colorbar(hb2, ax=ax[2], fraction=0.046, pad=0.04, label="count")
ax[2].set_xlabel("GC content"); ax[2].set_ylabel("MFE (kcal/mol)")
tag(ax[2], "C"); ax[2].text(0.96, 0.06, f"Pearson r = {rG:.2f}", transform=ax[2].transAxes, ha="right", va="bottom", fontsize=7)

fig.savefig("figures/Figure_04_folding_landscape.png",
            bbox_inches="tight", facecolor="white", dpi=300)
print(f"saved Figure_04_folding_landscape.png  n={n}  r(L)={rL:.3f}  r(GC)={rG:.3f}")
