#!/usr/bin/env python
"""Descriptive and inferential statistics of the aptamer corpus.

Four unique panels, with no content that is already shown elsewhere in the manuscript
(length-by-source lives in the composition figure; the length-energy relationship lives in
the folding-landscape figure, so both are omitted here to avoid duplication):
  (A) target-class composition,
  (B) sequences per target (corpus breadth),
  (C) mean folding energy by source with bootstrap 95% CIs and a Welch test,
  (D) a significance-annotated Spearman correlation matrix.
Everything is computed from the released data files.
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats

rng = np.random.default_rng(0)
BLUE, TAN, RED, GREEN, INK, GREY = "#3B6FB0", "#D6A15E", "#C0473E", "#4A9D5B", "#26303B", "#8A97A3"
plt.rcParams.update({"font.size": 8, "font.family": "Arial", "figure.dpi": 300, "axes.grid": False,
                     "axes.spines.top": False, "axes.spines.right": False})

# ---- assemble the modelling corpus with source / class metadata ----------
fold = pd.read_csv("data/folding_labels.csv").drop_duplicates("sequence")
meta = (pd.read_csv("data/aptamer_corpus.csv")
        .drop_duplicates("sequence")[["sequence", "source", "target_category"]])
d = fold.merge(meta, on="sequence", how="left")
d["source"] = d["source"].replace({"aptadb": "AptaDB", "utexas": "UT Austin", "pdb": "PDB"})
d = d[d.source.isin(["AptaDB", "UT Austin"])].copy()          # two principal sources
n = len(d)

def panel_tag(ax, letter):
    ax.set_title(letter, loc="left", fontweight="bold", fontsize=9, pad=4)

fig = plt.figure(figsize=(7.2, 5.4))
gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.42,
              left=0.10, right=0.95, top=0.95, bottom=0.09)

# ============ A: target-class composition (descriptive bar) ==============
axA = fig.add_subplot(gs[0, 0])
cc = (d.loc[d.target_category.notna() & (d.target_category != "Others"), "target_category"]
      .value_counts())
n_cls = int(cc.sum())
cols = [BLUE, TAN, GREEN, "#7E5AA2", RED, GREY][:len(cc)]
axA.barh(range(len(cc)), cc.values[::-1], color=cols[::-1], edgecolor="white")
axA.set_yticks(range(len(cc))); axA.set_yticklabels(cc.index[::-1], fontsize=9)
for i, v in enumerate(cc.values[::-1]):
    axA.text(v + n_cls*0.015, i, f"{v}  ({v/n_cls*100:.0f}%)", va="center", fontsize=6.5, color=INK)
axA.set_xlim(0, cc.values.max()*1.28); axA.set_xlabel("sequences")
panel_tag(axA, "A")

# ============ B: sequences per target (corpus breadth) ==================
axB = fig.add_subplot(gs[0, 1])
tc = d.target.value_counts().values
axB.hist(tc, bins=np.logspace(0, np.log10(tc.max()), 18), color=GREEN, alpha=0.85,
         edgecolor="white", linewidth=0.3)
axB.set_xscale("log")
axB.set_xlabel("sequences per target"); axB.set_ylabel("number of targets")
panel_tag(axB, "B")

# ============ C: mean folding energy by source, bootstrap 95% CI =========
axC = fig.add_subplot(gs[1, 0])
def boot_ci(a, B=5000):
    m = rng.choice(a, (B, len(a)), replace=True).mean(1)
    return a.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
stats_by = [(s, boot_ci(d.loc[d.source == s, "mfe_kcal"].values)) for s in ["AptaDB", "UT Austin"]]
for i, (s, (m, lo, hi)) in enumerate(stats_by):
    c = [BLUE, TAN][i]
    axC.bar(i, m, color=c, alpha=0.85, width=0.55, edgecolor="white")
    axC.errorbar(i, m, yerr=[[m-lo], [hi-m]], color=INK, capsize=6, lw=1.8)
    axC.text(i, m*0.5, f"{m:.2f}", ha="center", va="center",
             fontsize=9, color="white", fontweight="bold")
t, pt = stats.ttest_ind(d.loc[d.source=="AptaDB","mfe_kcal"], d.loc[d.source=="UT Austin","mfe_kcal"], equal_var=False)
axC.set_xticks([0, 1]); axC.set_xticklabels(["AptaDB", "UT Austin"])
axC.set_ylabel("mean MFE (kcal/mol)")
axC.set_ylim(min(s[1][1] for s in stats_by)*1.12, 0.4)
axC.text(0.5, 0.12, f"Welch t-test p = {pt:.1e}", ha="center", transform=axC.transAxes,
         fontsize=7, color=INK)
panel_tag(axC, "C")

# ============ D: significance-masked correlation matrix (inferential) ====
axD = fig.add_subplot(gs[1, 1])
cols_c = {"Length": d.L, "GC": d.gc, "MFE": d.mfe_kcal,
          "Entropy": d.structural_entropy, "Defect": d.ensemble_defect, "logQ": d.log_pfunc}
names = list(cols_c); M = np.zeros((len(names), len(names))); P = np.ones_like(M)
for i, a in enumerate(names):
    for j, b in enumerate(names):
        r, pv = stats.spearmanr(cols_c[a], cols_c[b])
        M[i, j] = r; P[i, j] = pv
im = axD.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
axD.set_xticks(range(len(names))); axD.set_xticklabels(names, rotation=45, ha="right", fontsize=7)
axD.set_yticks(range(len(names))); axD.set_yticklabels(names, fontsize=7)
for i in range(len(names)):
    for j in range(len(names)):
        star = "" if i == j else ("***" if P[i,j] < 1e-3 else "**" if P[i,j] < 1e-2 else "*" if P[i,j] < 0.05 else "ns")
        axD.text(j, i, f"{M[i,j]:.2f}\n{star}", ha="center", va="center", fontsize=5.2,
                 color="white" if abs(M[i,j]) > 0.55 else INK)
cb = fig.colorbar(im, ax=axD, fraction=0.046, pad=0.03); cb.set_label("Spearman ρ", fontsize=7)
axD.spines[:].set_visible(False)
panel_tag(axD, "D")

fig.savefig("figures/Figure_02_dataset_statistics.png",
            bbox_inches="tight", facecolor="white", dpi=300)
print(f"saved Figure_02_dataset_statistics.png  (n={n}); Welch p={pt:.2e}")
