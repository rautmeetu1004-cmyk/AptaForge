#!/usr/bin/env python
"""Consolidate every verified number into one source-of-truth JSON and build the final
benchmark figure (all models, leak-free dedup protocol). Every value here traces to a
saved OOF prediction or prior verified JSON. Nothing hand-entered.
"""
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error

plt.rcParams.update({"figure.dpi": 300, "font.size": 8, "font.family": "Arial", "axes.grid": True, "grid.alpha": 0.25})
d = pd.read_csv("data/folding_dedup.csv")
y = np.load("results/19_y.npy")
oof = {"length + GC (Ridge)": np.load("results/19_oof_lengc.npy"),
       "k-mer (HistGBM)":     np.load("results/19_oof_histgbm.npy"),
       "CNN":                 np.load("results/19_oof_cnn.npy")}
def M(yp): return dict(r2=round(float(r2_score(y, yp)), 4),
                       spearman=round(float(stats.spearmanr(y, yp).correlation), 4),
                       mae=round(float(mean_absolute_error(y, yp)), 4))
tbl = {k: M(v) for k, v in oof.items()}

# pull the rest from verified JSONs
fast = json.load(open("results/19a_dedup_benchmark.json"))
multi = json.load(open("results/22_multiproperty.json"))
case = json.load(open("results/20_case_study.json"))
gcmr = json.load(open("results/21_kmer_analysis.json"))
pcon = json.load(open("results/23_per_target.json"))

consolidated = {
    "dataset": {"n_unique_sequences": int(len(d)), "n_targets": int(d.target.nunique()),
                "mfe_mean": round(float(y.mean()), 3), "mfe_std": round(float(y.std()), 3),
                "frac_no_fold": round(float((y == 0).mean()), 3)},
    "protocol": "deduplicated to unique sequences; 5-fold split BY SEQUENCE (leak-free for folding)",
    "mfe_benchmark": {"length_only": fast["length_only"], "length_gc": tbl["length + GC (Ridge)"],
                      "kmer_histgbm": tbl["k-mer (HistGBM)"], "cnn": tbl["CNN"]},
    "rigor": {"mean_predictor_r2": fast["mean_predictor_r2"],
              "histgbm_vs_lengc_wilcoxon_p": fast["histgbm_vs_lengc"]["wilcoxon_p"],
              "histgbm_vs_lengc_cliffs_delta": fast["histgbm_vs_lengc"]["cliffs_delta"],
              "histgbm_seed_std": fast["histgbm_seed_stability"]["std"]},
    "multiproperty": multi,
    "case_study": case,
    "kmer_interpretability": {"corr_gc_vs_stability": gcmr["corr_gc_vs_stability"],
                              "corr_grun_vs_stability": gcmr["corr_grun_vs_stability"],
                              "most_stabilizing": gcmr["most_stabilizing"][:6]},
    "thrombin_positive_control": pcon["thrombin_g4_control"],
}
# keep primer + sequence-feature analyses in the source-of-truth (added by scripts 25/26)
try:
    consolidated["primer_analysis"] = json.load(open("results/25_primer_analysis.json"))
    consolidated["sequence_features"] = json.load(open("results/26_sequence_features.json"))
except FileNotFoundError:
    pass
json.dump(consolidated, open("results/FINAL_NUMBERS.json", "w"), indent=2)
print("Consolidated -> FINAL_NUMBERS.json")
for k, v in consolidated["mfe_benchmark"].items():
    print(f"  {k:16s} R2={v['r2']:+.3f} Spearman={v['spearman']:+.3f} MAE={v['mae']:.3f}")

# ---- final benchmark figure
fig = plt.figure(figsize=(7.2, 3.0)); gs = GridSpec(1, 3, figure=fig, wspace=0.45)
order = ["length only", "length + GC (Ridge)", "k-mer (HistGBM)", "CNN"]
r2s = [fast["length_only"]["r2"], tbl["length + GC (Ridge)"]["r2"], tbl["k-mer (HistGBM)"]["r2"], tbl["CNN"]["r2"]]
sps = [fast["length_only"]["spearman"], tbl["length + GC (Ridge)"]["spearman"], tbl["k-mer (HistGBM)"]["spearman"], tbl["CNN"]["spearman"]]
ax = fig.add_subplot(gs[0, 0]); yy = np.arange(len(order))
ax.hlines(yy, 0, r2s, color="#ccc", lw=2, zorder=1)
ax.scatter(r2s, yy, s=130, color="#4C72B0", zorder=3, label="R²")
ax.scatter(sps, yy, s=95, color="#DD8452", marker="D", zorder=3, label="Spearman ρ")
ax.set_yticks(yy); ax.set_yticklabels(order); ax.set_xlabel("held-out score (leak-free)")
ax.set_title("A", loc="left", fontweight="bold", fontsize=9); ax.legend(loc="lower right", fontsize=7)
# best model hexbin
ax = fig.add_subplot(gs[0, 1]); yp = oof["k-mer (HistGBM)"]
hb = ax.hexbin(y, yp, gridsize=30, cmap="Blues", mincnt=1)
lims = [y.min(), y.max()]; ax.plot(lims, lims, "r--", lw=1.4, alpha=0.7)
ax.set(xlabel="true MFE (kcal/mol)", ylabel="predicted MFE")
ax.set_title("B", loc="left", fontweight="bold", fontsize=9)
ax.text(0.04, 0.96, f"R²={tbl['k-mer (HistGBM)']['r2']:.2f}\nρ={tbl['k-mer (HistGBM)']['spearman']:.2f}", transform=ax.transAxes, va="top", fontsize=7, bbox=dict(boxstyle="round", fc="white", ec="#4C72B0", alpha=0.9))
fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.04).set_label("count")
# per-model absolute-error distributions (box, not line)
ax = fig.add_subplot(gs[0, 2])
models = ["length + GC (Ridge)", "k-mer (HistGBM)", "CNN"]; cols3 = ["#55A868", "#4C72B0", "#8172B3"]
errs = [np.abs(y - oof[k]) for k in models]
bp = ax.boxplot(errs, positions=range(len(models)), widths=0.55, showfliers=False,
                patch_artist=True, medianprops=dict(color="black", lw=1.4))
for patch_, c in zip(bp["boxes"], cols3): patch_.set_facecolor(c); patch_.set_alpha(0.55)
ax.set_xticks(range(len(models))); ax.set_xticklabels(["len+GC", "k-mer", "CNN"], fontsize=7)
ax.set(ylabel="absolute error (kcal/mol)"); ax.set_title("C", loc="left", fontweight="bold", fontsize=9)
fig.savefig("figures/Figure_07_benchmark.png", bbox_inches="tight")
print("saved 24_final_benchmark.png")
