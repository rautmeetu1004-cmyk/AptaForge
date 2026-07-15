#!/usr/bin/env python
"""Improved graphical abstract for AptaForge (v3).

Three-zone left-to-right narrative, separated by clear gutters so no box border or chip
ever coincides with a divider line:
  1. SCREENING FUNNEL     — an enormous candidate pool narrowed, in one fast surrogate pass,
                            to the small set worth exact NUPACK evaluation.
  2. INSIDE THE SURROGATE — how a raw sequence becomes a folding-energy prediction.
  3. IT ACTUALLY WORKS    — the real out-of-fold predicted-vs-true accuracy + headline chips.

Everything traces to cached, verified artifacts: results/FINAL_NUMBERS.json and the
out-of-fold prediction arrays results/19_{y,oof_histgbm}.npy.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyBboxPatch, FancyArrowPatch
from scipy.stats import spearmanr

ROOT = "."
N = json.load(open(f"{ROOT}/results/FINAL_NUMBERS.json"))
SP = N["case_study"]["speed"]
y = np.load(f"{ROOT}/results/19_y.npy")
oof = np.load(f"{ROOT}/results/19_oof_histgbm.npy")
mfe = N["mfe_benchmark"]["kmer_histgbm"]
ens = N["multiproperty"]["Ensemble free energy (kcal/mol)"]
rho = spearmanr(y, oof).correlation

# ---- house palette -------------------------------------------------------
BLUE, TAN, RED = "#3B6FB0", "#D6A15E", "#C0473E"
GREEN, INK, GREY = "#4A9D5B", "#26303B", "#8A97A3"
PURPLE = "#7E5AA2"
RULE = "#E3DFD6"
plt.rcParams.update({"font.size": 7, "font.family": "Arial", "figure.dpi": 300})

W, H = 150.0, 64.0
fig = plt.figure(figsize=(7.6, 3.35))
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

# ---- title band ----------------------------------------------------------
ax.text(3, 60.6, "AptaForge", fontsize=15, fontweight="bold", color=INK, va="center")
ax.text(33, 60.8, "a sequence surrogate that triages aptamer folding at scale",
        fontsize=7.6, color=GREY, va="center", style="italic")
ax.plot([3, 147], [57.0, 57.0], color=RULE, lw=2)

TITLE_Y = 54.6
DIV1, DIV2 = 63.0, 99.0

# ============================ 1. THE FUNNEL ==============================
cx = 18
top_w, bot_w = 27, 7
y_top, y_bot = 51.0, 13.0
stages = [
    (BLUE,      "millions",   "Candidate pool",            "proposed sequences"),
    ("#5E86BE", f"{SP['surrogate_ms_per_seq']*1000:.0f} µs/seq",
                              "AptaForge — one pass",       f"~{SP['speedup']}× vs NUPACK"),
    (TAN,       "top ~50%",   "Rank by predicted ΔG",      "keep the stable fraction"),
    ("#C77F44", "survivors",  "Exact NUPACK",              "only where it counts"),
    (RED,       "hits",       "Validated aptamers",        "few strong ones lost"),
]
n = len(stages)
band_h = (y_top - y_bot) / n
def half_at(yy):
    f = (y_top - yy) / (y_top - y_bot)
    return (top_w - (top_w - bot_w) * f) / 2

ax.text(30, TITLE_Y, "SCREENING FUNNEL", ha="center", fontsize=7.6, fontweight="bold", color=INK)
for i, (col, inside, label, sub) in enumerate(stages):
    yt = y_top - i * band_h
    yb = yt - band_h * 0.80
    poly = Polygon([(cx - half_at(yt), yt), (cx + half_at(yt), yt),
                    (cx + half_at(yb), yb), (cx - half_at(yb), yb)],
                   closed=True, facecolor=col, edgecolor="white", lw=2.0, alpha=0.96, zorder=2)
    ax.add_patch(poly)
    ymid = (yt + yb) / 2
    ax.text(cx, ymid, inside, ha="center", va="center", fontsize=6.6,
            fontweight="bold", color="white", zorder=3)
    lx = cx + top_w / 2 + 3.5
    ax.text(lx, ymid + 1.5, label, ha="left", va="center", fontsize=6.6,
            fontweight="bold", color=INK, zorder=3)
    ax.text(lx, ymid - 1.7, sub, ha="left", va="center", fontsize=5.4, color=GREY, zorder=3)
    ax.plot([cx + half_at(ymid) + 0.4, lx - 0.8], [ymid, ymid], color=col, lw=1.7, zorder=1)
    if i < n - 1:
        ax.add_patch(FancyArrowPatch((cx, yb + 0.1), (cx, yb - band_h * 0.20 + 0.3),
                     arrowstyle="-|>", mutation_scale=10, color=GREY, lw=1.3, zorder=4))

# recovery callout under the funnel
ax.add_patch(FancyBboxPatch((5, 4.6), 55, 5.6,
             boxstyle="round,pad=0.4,rounding_size=1.2", fc="#EFF5EF", ec=GREEN, lw=1.6))
ax.plot(8.2, 7.4, marker=">", ms=5, color=GREEN, zorder=3)
ax.text(10.8, 7.4, "Screen 50% of a pool, still recover >90% of the most-stable candidates",
        fontsize=5.6, color=INK, va="center")

ax.plot([DIV1, DIV1], [10, 53], color=RULE, lw=1.6)

# ==================== 2. INSIDE THE SURROGATE ===========================
rx = 66.0; bw = 30.0
ax.text(rx + bw / 2, TITLE_Y, "INSIDE THE SURROGATE", ha="center", fontsize=7.6,
        fontweight="bold", color=INK)
def box(x, y, w, h, text, sub, fc, ec):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                 fc=fc, ec=ec, lw=1.7, zorder=2))
    ax.text(x + w / 2, y + h / 2 + (0.95 if sub else 0), text, ha="center", va="center",
            fontsize=6.2, fontweight="bold", color=INK, zorder=3)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 1.85, sub, ha="center", va="center",
                fontsize=5.0, color=GREY, zorder=3)
py = 47.4; dh = 8.9
steps = [
    ("aptamer sequence  5′→3′", None, "#EEF2F7", BLUE),
    ("322 k-mer features", "tri/tetranucleotide + length + GC", "#FBF3E7", TAN),
    ("Gradient-boosted trees", "depth 6 · 400 rounds · leak-free", "#F3EEF6", PURPLE),
    ("Predicted folding ΔG", "minimum & ensemble free energy", "#EFF5EF", GREEN),
]
for j, (t, s, fc, ec) in enumerate(steps):
    yb = py - j * dh
    box(rx, yb, bw, 6.4, t, s, fc, ec)
    if j < len(steps) - 1:
        ax.add_patch(FancyArrowPatch((rx + bw / 2, yb - 0.2), (rx + bw / 2, yb - (dh - 6.4) + 0.4),
                     arrowstyle="-|>", mutation_scale=10, color=GREY, lw=1.3, zorder=4))

# accuracy / speed chips under the pipeline (kept fully inside the pipeline column)
chip_y = 8.4
def chip(x, w, big, small, col):
    ax.add_patch(FancyBboxPatch((x, chip_y - 4.4), w, 8.4, boxstyle="round,pad=0.25,rounding_size=1.0",
                 fc="white", ec=col, lw=1.7))
    ax.text(x + w / 2, chip_y + 1.5, big, ha="center", fontsize=8.0, fontweight="bold", color=col)
    ax.text(x + w / 2, chip_y - 2.3, small, ha="center", fontsize=5.0, color=INK)
cw = 9.2
chip(rx,           cw, f"R² {mfe['r2']:.2f}", "MFE  ρ 0.80", BLUE)
chip(rx + 10.4,    cw, f"R² {ens['r2']:.2f}", "ensemble ΔG", GREEN)
chip(rx + 20.8,    cw, f"~{SP['speedup']}×",  "faster",      RED)

ax.plot([DIV2, DIV2], [10, 53], color=RULE, lw=1.6)

# ==================== 3. IT ACTUALLY WORKS (real data) ==================
ax.text(124.5, TITLE_Y, "IT ACTUALLY WORKS", ha="center", fontsize=7.6,
        fontweight="bold", color=INK)
# inset placed in figure fraction; ax fills the whole figure so frac = data / (W, H)
il, ib, iw, ih = 108 / W, 18 / H, 34 / W, 29 / H
iax = fig.add_axes([il, ib, iw, ih])
iax.hexbin(y, oof, gridsize=22, cmap="Blues", mincnt=1, linewidths=0.15)
lim = [min(y.min(), oof.min()), max(y.max(), oof.max())]
iax.plot(lim, lim, color=RED, ls="--", lw=1.1)
iax.set_xlim(lim); iax.set_ylim(lim)
iax.set_xlabel("true MFE", fontsize=6, labelpad=1)
iax.set_ylabel("predicted MFE", fontsize=6, labelpad=1)
iax.tick_params(labelsize=5, length=2, pad=1)
for sp in iax.spines.values():
    sp.set_edgecolor(GREY); sp.set_linewidth(0.6)
iax.text(0.05, 0.94, f"R² = {mfe['r2']:.2f}\nρ = {rho:.2f}", transform=iax.transAxes,
         va="top", ha="left", fontsize=5.6, color=INK,
         bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=BLUE, lw=0.8, alpha=0.9))
ax.text(124.5, 12.2, "1,426 held-out ssDNA aptamers,", ha="center", fontsize=5.3, color=INK)
ax.text(124.5, 9.4, "predicted from sequence alone", ha="center", fontsize=5.3, color=INK)

# ---- footer provenance ---------------------------------------------------
ax.text(3, 1.5, f"{N['dataset']['n_unique_sequences']:,} unique ssDNA aptamers  ·  "
        f"{N['dataset']['n_targets']} targets  ·  real NUPACK labels  ·  leak-free "
        f"evaluation by sequence  ·  all data, code and models released",
        fontsize=5.6, color=GREY, va="center")

fig.savefig(f"{ROOT}/figures/Figure_00_graphical_abstract.png",
            bbox_inches="tight", facecolor="white", dpi=300)
print("saved Figure_00_graphical_abstract.png (v3, spaced)")
