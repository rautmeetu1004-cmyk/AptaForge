#!/usr/bin/env python
"""Comprehensive evaluation of the CNN surrogate on real NUPACK MFE, leak-free (dedup,
split by sequence). Uses the cached out-of-fold predictions (19_oof_cnn.npy) so nothing
is retrained here. Shows the CNN is a sound regressor across several complementary lenses:
accuracy, rank agreement, calibration, residual normality, homoscedasticity, and error
stratified by sequence properties.
"""
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

plt.rcParams.update({"figure.dpi": 300, "font.size": 8, "font.family": "Arial", "axes.grid": True, "grid.alpha": 0.25})
d = pd.read_csv("data/folding_dedup.csv")
y = np.load("results/19_y.npy")
oof = np.load("results/19_oof_cnn.npy")
folds = np.load("results/19_folds.npy")
L = d.L.values.astype(float); gc = d.gc.values.astype(float)
resid = y - oof

# metrics
r2 = r2_score(y, oof); rmse = np.sqrt(mean_squared_error(y, oof)); mae = mean_absolute_error(y, oof)
sp = stats.spearmanr(y, oof).correlation; pr = stats.pearsonr(y, oof)[0]
per_fold = [r2_score(y[folds == k], oof[folds == k]) for k in range(5)]
within1 = float(np.mean(np.abs(resid) <= 1.0))
metrics = dict(r2=round(float(r2), 4), spearman=round(float(sp), 4), pearson=round(float(pr), 4),
               rmse=round(float(rmse), 4), mae=round(float(mae), 4),
               within_1_kcal=round(within1, 4),
               per_fold_r2=[round(float(x), 4) for x in per_fold],
               per_fold_r2_mean=round(float(np.mean(per_fold)), 4),
               per_fold_r2_std=round(float(np.std(per_fold)), 4))
json.dump(metrics, open("results/28_cnn_evaluation.json", "w"), indent=2)
print(json.dumps(metrics, indent=2))

fig = plt.figure(figsize=(7.0, 4.7)); gs = GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.55)
def PL(ax, s): ax.set_title(s, loc="left", fontweight="bold", fontsize=9, pad=3)

# A: predicted vs true (density) with metric box
ax = fig.add_subplot(gs[0, 0])
hb = ax.hexbin(y, oof, gridsize=30, cmap="Blues", mincnt=1)
lims = [y.min(), y.max()]; ax.plot(lims, lims, "r--", lw=1.4, alpha=0.7)
ax.set(xlabel="true MFE (kcal/mol)", ylabel="predicted MFE"); PL(ax, "A")
ax.text(0.04, 0.96, f"R² = {r2:.2f}\nρ = {sp:.2f}\nr = {pr:.2f}\nRMSE = {rmse:.2f}\nMAE = {mae:.2f}",
        transform=ax.transAxes, va="top", fontsize=6,
        bbox=dict(boxstyle="round", fc="#eaf0f8", ec="#3B6FB0", alpha=0.9))
fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04).set_label("count")

# B: residual distribution vs normal
ax = fig.add_subplot(gs[0, 1])
ax.hist(resid, bins=45, density=True, color="#4C72B0", alpha=0.8, edgecolor="white", linewidth=0.3)
xs = np.linspace(resid.min(), resid.max(), 200)
ax.plot(xs, stats.norm.pdf(xs, resid.mean(), resid.std()), "r-", lw=2, label="normal fit")
ax.axvline(0, color="k", lw=1)
ax.set(xlabel="residual (true - predicted)", ylabel="density"); PL(ax, "B")
ax.legend(loc="upper right", fontsize=6)
ax.text(0.04, 0.96, f"mean = {resid.mean():+.2f}\nSD = {resid.std():.2f}", transform=ax.transAxes,
        va="top", fontsize=6, bbox=dict(boxstyle="round", fc="#f4f4f4", ec="grey", alpha=0.9))

# C: Q-Q plot
ax = fig.add_subplot(gs[0, 2])
(osm, osr), (slope, inter, r) = stats.probplot(resid, dist="norm")
ax.scatter(osm, osr, s=8, alpha=0.5, color="#4C72B0")
ax.plot(osm, slope*osm + inter, "r-", lw=1.5)
ax.set(xlabel="theoretical quantiles", ylabel="ordered residuals"); PL(ax, "C")

# D: homoscedasticity - |residual| vs predicted
ax = fig.add_subplot(gs[1, 0])
ax.scatter(oof, np.abs(resid), s=8, alpha=0.35, color="#55A868")
o = np.argsort(oof); w = 40
roll = np.convolve(np.abs(resid)[o], np.ones(w)/w, mode="valid")
ax.plot(oof[o][w-1:], roll, color="darkgreen", lw=2, label="rolling mean")
ax.set(xlabel="predicted MFE", ylabel="|residual|"); PL(ax, "D"); ax.legend(fontsize=6)

# E: error stratified by length and GC quantile (grouped bars, not line)
ax = fig.add_subplot(gs[1, 1])
qn = 5; wd = 0.38
for off, (vals, labq, col) in zip([-wd/2, wd/2], [(L, "length", "#4C72B0"), (gc*100, "GC%", "#C44E52")]):
    qs = np.quantile(vals, np.linspace(0, 1, qn+1)); cy = []
    for i in range(qn):
        m = (vals >= qs[i]) & (vals <= qs[i+1])
        cy.append(np.abs(resid[m]).mean() if m.sum() > 5 else np.nan)
    ax.bar(np.arange(qn)+off, cy, width=wd, color=col, alpha=0.8, label=labq)
ax.set_xticks(range(qn)); ax.set_xticklabels([f"Q{i+1}" for i in range(qn)], fontsize=7)
ax.set(xlabel="feature quantile (low to high)", ylabel="mean |residual|"); PL(ax, "E")
ax.legend(fontsize=6); ax.set_ylim(bottom=0)

# F: per-fold R2 stability
ax = fig.add_subplot(gs[1, 2])
xx = np.arange(5)
ax.bar(xx, per_fold, color="#8172B3", alpha=0.8, edgecolor="black", linewidth=0.6)
ax.axhline(np.mean(per_fold), color="crimson", ls="--", lw=1.6,
           label=f"mean {np.mean(per_fold):.2f} ± {np.std(per_fold):.2f}")
for i, v in enumerate(per_fold): ax.text(i, v+0.008, f"{v:.2f}", ha="center", fontsize=8)
ax.set_xticks(xx); ax.set_xticklabels([f"fold {k+1}" for k in range(5)])
ax.set(ylabel="held-out R²", ylim=(0, max(per_fold)*1.2)); PL(ax, "F"); ax.legend(fontsize=6)

fig.savefig("figures/Figure_08_cnn_evaluation.png", bbox_inches="tight")
print("saved 28_cnn_evaluation.png + .json")
