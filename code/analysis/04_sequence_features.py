#!/usr/bin/env python
"""Sequence-derived biophysical descriptors for DNA aptamers, computed from primary
sequence alone (no thermodynamic simulation). Formulas follow the DNA-aptamer feature
methodology. We relate each descriptor to real NUPACK folding energy, examine their
mutual structure by PCA and correlation, and test whether they sharpen the MFE surrogate.

Scope note: descriptors are used for FOLDING characterisation only. The target-class
classifier / de-novo design application from the source methodology is intentionally
not reproduced.
"""
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from itertools import product
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error

plt.rcParams.update({"figure.dpi": 300, "font.size": 8, "font.family": "Arial", "axes.grid": True, "grid.alpha": 0.25})
d = pd.read_csv("data/folding_dedup.csv")
y = d.mfe_kcal.values.astype(float); seqs = d.sequence.values
folds = np.load("results/19_folds.npy")
COMP = {"A": "T", "T": "A", "G": "C", "C": "G"}

def shannon_k1(s):
    L = len(s); return -sum((s.count(b)/L)*np.log2(s.count(b)/L) for b in "ATGC" if s.count(b))
def shannon_k3(s):
    ks = [s[i:i+3] for i in range(len(s)-2)]
    if not ks: return 0.0
    c = pd.Series(ks).value_counts()/len(ks)
    return float(-(c*np.log2(c)).sum())
def purine_ratio(s):
    pyr = s.count("C")+s.count("T"); return (s.count("A")+s.count("G"))/pyr if pyr else np.nan
def pi_stacking(s):
    return sum(s[i:i+2] in ("AG","GA","AA","GG") for i in range(len(s)-1))/max(len(s)-1,1)
def gqs_density(s):
    import re; return len(re.findall(r"G{3,}", s))/(len(s)/10)
def struct_density_adj(s):    # DNA doc: adjacent-identical fraction
    return sum(s[j]==s[j+1] for j in range(len(s)-1))/max(len(s)-1,1)
def rc_selfmatch(s):          # hairpin proxy: fraction of positions i whose base complements S[L-1-i]
    L = len(s); return sum(COMP.get(s[i],"?")==s[L-1-i] for i in range(L))/L
def gc(s): return (s.count("G")+s.count("C"))/len(s)
def scaffolding_index(s):     # DNA doc: (GC * L)/(H1 + eps)
    return (gc(s)*len(s))/(shannon_k1(s)+1e-6)

FEATS = {"GC content": gc, "Purine ratio": purine_ratio, "Shannon H (k=1)": shannon_k1,
         "Shannon H (k=3)": shannon_k3, "Pi-stacking": pi_stacking, "GQS density": gqs_density,
         "Struct. density": struct_density_adj, "RC self-match": rc_selfmatch,
         "Scaffolding index": scaffolding_index, "Length": lambda s: len(s)}
X = pd.DataFrame({name: [f(s) for s in seqs] for name, f in FEATS.items()})
X.to_csv("data/sequence_features.csv", index=False)
print(f"computed {X.shape[1]} descriptors for {len(X)} sequences")

# ---- 1. association of each descriptor with real MFE
assoc = {}
for c in X.columns:
    pr = stats.pearsonr(X[c], y)[0]; sp = stats.spearmanr(X[c], y)[0]
    assoc[c] = dict(pearson=round(float(pr), 3), spearman=round(float(sp), 3))
assoc_sorted = sorted(assoc.items(), key=lambda kv: kv[1]["spearman"])
print("descriptor vs MFE (sorted by Spearman):")
for c, v in assoc_sorted: print(f"  {c:20s} r={v['pearson']:+.3f}  rho={v['spearman']:+.3f}")

# ---- 2. does adding descriptors to the k-mer model help? (leak-free, same folds)
v3 = ["".join(p) for p in product("ATGC", repeat=3)]; i3 = {k: i for i, k in enumerate(v3)}
v4 = ["".join(p) for p in product("ATGC", repeat=4)]; i4 = {k: i for i, k in enumerate(v4)}
def kmer(s):
    f3 = np.zeros(64); f4 = np.zeros(256)
    for i in range(len(s)-2): f3[i3[s[i:i+3]]] += 1
    for i in range(len(s)-3): f4[i4[s[i:i+4]]] += 1
    if f3.sum(): f3 /= f3.sum()
    if f4.sum(): f4 /= f4.sum()
    return np.concatenate([f3, f4])
Xk = np.vstack([kmer(s) for s in seqs])
Xfeat = StandardScaler().fit_transform(X.values)
def cv(Xmat):
    yp = np.zeros_like(y)
    for k in range(5):
        te = folds == k; tr = ~te
        m = HistGradientBoostingRegressor(max_depth=6, max_iter=400, learning_rate=0.06,
                                          l2_regularization=1.0, random_state=0).fit(Xmat[tr], y[tr])
        yp[te] = m.predict(Xmat[te])
    return dict(r2=round(float(r2_score(y, yp)), 4), spearman=round(float(stats.spearmanr(y, yp).correlation), 4),
                mae=round(float(mean_absolute_error(y, yp)), 4))
res = {"descriptors_only": cv(Xfeat), "kmer_only": cv(Xk), "kmer_plus_descriptors": cv(np.hstack([Xk, Xfeat]))}
print("\nsurrogate with descriptors:")
for k, v in res.items(): print(f"  {k:24s} R2={v['r2']:+.3f} Spearman={v['spearman']:+.3f}")

# ---- 3. PCA of descriptor space
pca = PCA().fit(Xfeat); Z = pca.transform(Xfeat)
var = pca.explained_variance_ratio_
print(f"\nPCA: PC1={var[0]*100:.1f}% PC2={var[1]*100:.1f}% (cum {sum(var[:2])*100:.1f}%)")

json.dump({"descriptor_vs_mfe": assoc, "surrogate": res,
           "pca_var": [round(float(v), 4) for v in var[:5]]},
          open("results/26_sequence_features.json", "w"), indent=2)

from scipy.stats import gaussian_kde
def PL(ax, s): ax.set_title(s, loc="left", fontweight="bold", fontsize=9, pad=3)

# ================= FIGURE 5: descriptor–folding associations =================
fig = plt.figure(figsize=(6.6, 3.1)); gs = GridSpec(1, 2, figure=fig, wspace=0.55, width_ratios=[1.05, 1.2])
# A: descriptor vs MFE — diverging horizontal lollipop, sorted
ax = fig.add_subplot(gs[0, 0])
names = [c for c, _ in assoc_sorted]; sps = [assoc[c]["spearman"] for c in names]
yy = np.arange(len(names)); cols = ["#2166AC" if v < 0 else "#B2182B" for v in sps]
ax.hlines(yy, 0, sps, color=cols, lw=2.5, alpha=0.6)
ax.scatter(sps, yy, s=90, c=cols, zorder=3, edgecolors="white", linewidth=1)
for i, v in zip(yy, sps):
    ax.text(v + (0.03 if v >= 0 else -0.03), i, f"{v:+.2f}", va="center",
            ha="left" if v >= 0 else "right", fontsize=7, color=cols[i])
ax.axvline(0, color="k", lw=1); ax.set_yticks(yy); ax.set_yticklabels(names, fontsize=7.5)
ax.set_xlim(-0.92, 0.55); ax.set(xlabel="Spearman ρ with folding energy (MFE)"); PL(ax, "A")
# B: descriptor inter-correlation heatmap
ax = fig.add_subplot(gs[0, 1])
C = X.corr(method="spearman").values
im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(X.columns))); ax.set_xticklabels(X.columns, rotation=45, ha="right", fontsize=6.5)
ax.set_yticks(range(len(X.columns))); ax.set_yticklabels(X.columns, fontsize=6.5)
for i in range(len(X.columns)):
    for j in range(len(X.columns)):
        if abs(C[i, j]) >= 0.5 and i != j:
            ax.text(j, i, f"{C[i,j]:.1f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(C[i, j]) > 0.7 else "black")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Spearman ρ", fontsize=7)
PL(ax, "B")
fig.savefig("figures/Figure_05_descriptor_associations.png", bbox_inches="tight")
plt.close(fig)

# ================= FIGURE 6: descriptor structure (PCA) =====================
fig = plt.figure(figsize=(7.0, 2.9)); gs = GridSpec(1, 3, figure=fig, wspace=0.75, width_ratios=[0.85, 1.05, 0.95])
# A: scree as bars (per-PC variance) + cumulative marker line
ax = fig.add_subplot(gs[0, 0])
ax.bar(range(1, len(var)+1), var*100, color="#4C72B0", alpha=0.85, edgecolor="white")
ax.plot(range(1, len(var)+1), np.cumsum(var)*100, "o-", color="#C44E52", lw=1.3, ms=3, label="cumulative")
ax.axhline(66, color="grey", ls="--", lw=1)
ax.text(len(var), 70, "PC1+2 = 66%", ha="right", fontsize=6, color="grey")
ax.set(xlabel="principal component", ylabel="variance explained (%)", ylim=(0, 105))
ax.legend(fontsize=6, loc="center right", frameon=False); PL(ax, "A")
# B: PCA scatter coloured by MFE
ax = fig.add_subplot(gs[0, 1])
sc = ax.scatter(Z[:, 0], Z[:, 1], c=y, cmap="viridis", s=15, alpha=0.65, edgecolors="none")
xy = np.vstack([Z[:, 0], Z[:, 1]]); kde = gaussian_kde(xy)
xg, yg = np.mgrid[Z[:, 0].min():Z[:, 0].max():120j, Z[:, 1].min():Z[:, 1].max():120j]
zg = kde(np.vstack([xg.ravel(), yg.ravel()])).reshape(xg.shape)
ax.contour(xg, yg, zg, levels=6, colors="black", alpha=0.25, linewidths=0.7)
cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.05, pad=0.3, aspect=35); cb.set_label("MFE (kcal/mol)", fontsize=7)
ax.set(xlabel=f"PC1 ({var[0]*100:.0f}%)", ylabel=f"PC2 ({var[1]*100:.0f}%)"); PL(ax, "B")
# C: loadings heatmap
ax = fig.add_subplot(gs[0, 2])
Lmat = pca.components_[:4].T
im = ax.imshow(Lmat, cmap="PuOr", vmin=-0.6, vmax=0.6, aspect="auto")
ax.set_xticks(range(4)); ax.set_xticklabels([f"PC{i+1}" for i in range(4)], fontsize=7)
SHORT = ["GC", "Purine", "H (k=1)", "H (k=3)", "π-stack", "GQS", "Struct.", "RC-match", "Scaffold", "Length"]
ax.set_yticks(range(len(X.columns))); ax.set_yticklabels(SHORT, fontsize=6.5)
for i in range(len(X.columns)):
    for j in range(4):
        if abs(Lmat[i, j]) >= 0.3:
            ax.text(j, i, f"{Lmat[i,j]:.1f}", ha="center", va="center", fontsize=5.5,
                    color="white" if abs(Lmat[i, j]) > 0.45 else "black")
fig.colorbar(im, ax=ax, fraction=0.08, pad=0.04).set_label("loading", fontsize=7)
PL(ax, "C")
fig.savefig("figures/Figure_06_descriptor_pca.png", bbox_inches="tight")
plt.close(fig)
print(f"surrogate sharpening (in text): descriptors {res['descriptors_only']['r2']} -> "
      f"kmer {res['kmer_only']['r2']} -> combined {res['kmer_plus_descriptors']['r2']}")
print("saved 26_sequence_features.png + .json + data/sequence_features.csv")
