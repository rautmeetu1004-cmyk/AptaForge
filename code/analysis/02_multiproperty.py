#!/usr/bin/env python
"""Multi-property biophysical surrogate. Beyond MFE, the same k-mer model predicts
NUPACK structural entropy, ensemble defect and ensemble free energy, held-out by
sequence (identical leak-free folds as the MFE benchmark). Shows the surrogate is a
general folding oracle, not a single-number regressor.
"""
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from itertools import product
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

plt.rcParams.update({"figure.dpi": 300, "font.size": 8, "font.family": "Arial", "axes.grid": True, "grid.alpha": 0.25})
RT = 1.987204e-3 * 310.15   # kcal/mol at 37 C

base = pd.read_csv("data/folding_dedup.csv")            # sequence order + folds
bio = pd.read_csv("data/folding_labels.csv").drop_duplicates("sequence")
d = base.merge(bio[["sequence", "structural_entropy", "ensemble_defect", "log_pfunc"]],
               on="sequence", how="left").reset_index(drop=True)
d["ens_dG"] = -RT * d.log_pfunc                                      # ensemble free energy
folds = np.load("results/19_folds.npy")
assert len(folds) == len(d)
seqs = d.sequence.values
print(f"{len(d)} unique sequences; properties: MFE, entropy, defect, ensemble dG")

v3 = ["".join(p) for p in product("ATGC", repeat=3)]; i3 = {k: i for i, k in enumerate(v3)}
v4 = ["".join(p) for p in product("ATGC", repeat=4)]; i4 = {k: i for i, k in enumerate(v4)}
def kmer(s):
    f3 = np.zeros(64); f4 = np.zeros(256)
    for i in range(len(s)-2): f3[i3[s[i:i+3]]] += 1
    for i in range(len(s)-3): f4[i4[s[i:i+4]]] += 1
    if f3.sum(): f3 /= f3.sum()
    if f4.sum(): f4 /= f4.sum()
    return np.concatenate([f3, f4, [len(s)/100, (s.count("G")+s.count("C"))/len(s)]])
X = np.vstack([kmer(s) for s in seqs])

PROPS = {"MFE (kcal/mol)": "mfe_kcal", "Structural entropy (bits)": "structural_entropy",
         "Ensemble defect": "ensemble_defect", "Ensemble free energy (kcal/mol)": "ens_dG"}
def cv(y):
    yp = np.zeros_like(y, dtype=float)
    for k in range(5):
        te = folds == k; tr = ~te
        sc = StandardScaler().fit(X[tr])
        m = HistGradientBoostingRegressor(max_depth=6, max_iter=400, learning_rate=0.06,
                                          l2_regularization=1.0, random_state=0).fit(sc.transform(X[tr]), y[tr])
        yp[te] = m.predict(sc.transform(X[te]))
    return yp

res = {}
oofs = {}
for name, col in PROPS.items():
    y = d[col].values.astype(float); yp = cv(y); oofs[name] = (y, yp)
    res[name] = dict(r2=round(float(r2_score(y, yp)), 4),
                     spearman=round(float(stats.spearmanr(y, yp).correlation), 4),
                     mae=round(float(mean_absolute_error(y, yp)), 4))
    print(f"  {name:32s} R2={res[name]['r2']:+.3f} Spearman={res[name]['spearman']:+.3f}")
json.dump(res, open("results/22_multiproperty.json", "w"), indent=2)

# ---- figure: predicted vs true for the 4 properties (2x2, ACS column width)
fig = plt.figure(figsize=(7.0, 6.4)); gs = GridSpec(2, 2, figure=fig, wspace=0.32, hspace=0.5)
letters = ["A", "B", "C", "D"]
for ax_i, (name, (y, yp)) in enumerate(oofs.items()):
    ax = fig.add_subplot(gs[ax_i // 2, ax_i % 2])
    hb = ax.hexbin(y, yp, gridsize=26, cmap="viridis", mincnt=1)
    lo, hi = min(y.min(), yp.min()), max(y.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], "r--", lw=1.3, alpha=0.7)
    ax.set(xlabel=f"true {name.split(' (')[0]}", ylabel="predicted")
    ax.set_title(letters[ax_i], loc="left", fontweight="bold", fontsize=9)
    ax.text(0.04, 0.96, f"R²={res[name]['r2']:.2f}\nρ={res[name]['spearman']:.2f}", transform=ax.transAxes, va="top", fontsize=7, bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.85))
fig.savefig("figures/Figure_09_multiproperty.png", bbox_inches="tight")

# ---- correlation heatmap among the biophysical properties (data-level)
cols = ["mfe_kcal", "structural_entropy", "ensemble_defect", "ens_dG", "gc", "L"]
lab = ["MFE", "Entropy", "Defect", "Ens dG", "GC", "Length"]
C = d[cols].corr(method="spearman").values
fig2, ax = plt.subplots(figsize=(6.2, 5.4))
im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(lab))); ax.set_xticklabels(lab, rotation=45, ha="right")
ax.set_yticks(range(len(lab))); ax.set_yticklabels(lab)
for i in range(len(lab)):
    for j in range(len(lab)):
        ax.text(j, i, f"{C[i,j]:.2f}", ha="center", va="center",
                color="white" if abs(C[i, j]) > 0.6 else "black", fontsize=9)
fig2.colorbar(im, fraction=0.046, pad=0.04).set_label("Spearman ρ")
ax.set_title("Biophysical property correlations (real NUPACK)")
fig2.tight_layout(); fig2.savefig("results/22b_property_corr.png", bbox_inches="tight")
print(json.dumps(res, indent=2)); print("saved 22_multiproperty.png + 22b_property_corr.png")
