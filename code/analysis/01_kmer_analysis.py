#!/usr/bin/env python
"""Alignment-free sequence analysis of the folding dataset (honest alternative to a
meaningless global MSA of unrelated aptamers):
  1. k-mer enrichment vs a mononucleotide null (log2 observed/expected)
  2. k-mer <-> folding-stability association (which motifs drive low MFE)
  3. cross-check against permutation importance of the trained HistGBM surrogate
  4. biological validation: G-run content vs stability (G-quadruplex expectation)
"""
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from itertools import product
from scipy import stats
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold

plt.rcParams.update({"figure.dpi": 300, "font.size": 11, "axes.grid": True, "grid.alpha": 0.25})
d = pd.read_csv("data/folding_dedup.csv")
seqs = d.sequence.values; y = d.mfe_kcal.values.astype(float)
K = 4
kms = ["".join(p) for p in product("ATGC", repeat=K)]; kidx = {k: i for i, k in enumerate(kms)}
print(f"{len(seqs)} sequences, {len(kms)} {K}-mers")

# per-sequence k-mer counts + presence
counts = np.zeros((len(seqs), len(kms)))
for r, s in enumerate(seqs):
    for i in range(len(s)-K+1):
        counts[r, kidx[s[i:i+K]]] += 1
freq = counts / counts.sum(1, keepdims=True).clip(min=1)
presence = counts > 0

# 1. enrichment vs mononucleotide null (per-dataset base composition)
allbases = "".join(seqs); p = {b: allbases.count(b)/len(allbases) for b in "ATGC"}
exp = np.array([np.prod([p[c] for c in w]) for w in kms])
obs = counts.sum(0) / counts.sum()
enrich = np.log2((obs + 1e-9) / (exp + 1e-9))

# 2. k-mer <-> MFE association: Pearson corr of per-seq frequency with MFE
#    (negative corr => more of this k-mer -> lower MFE -> more stable)
corr = np.array([stats.pearsonr(freq[:, j], y)[0] if freq[:, j].std() > 0 else 0.0
                 for j in range(len(kms))])
mean_mfe_with = np.array([y[presence[:, j]].mean() if presence[:, j].sum() >= 10 else np.nan
                          for j in range(len(kms))])

# G-run content of each k-mer (max run of G) for biological validation
def grun(w):
    best = c = 0
    for ch in w:
        c = c+1 if ch == "G" else 0; best = max(best, c)
    return best
gr = np.array([grun(w) for w in kms]); gc_k = np.array([(w.count("G")+w.count("C"))/K for w in kms])

# 3. permutation importance of trained HistGBM (on a held-out fold, honest)
folds = np.load("results/19_folds.npy")
te = folds == 0; tr = ~te
sc = StandardScaler().fit(freq[tr])
m = HistGradientBoostingRegressor(max_depth=6, max_iter=400, learning_rate=0.06,
                                  l2_regularization=1.0, random_state=0).fit(sc.transform(freq[tr]), y[tr])
pi = permutation_importance(m, sc.transform(freq[te]), y[te], n_repeats=10, random_state=0, n_jobs=-1)
imp = pi.importances_mean

# rankings
stab = np.argsort(corr)[:12]           # most stabilizing (most negative corr)
destab = np.argsort(-corr)[:12]        # most destabilizing
top_imp = np.argsort(-imp)[:12]

summary = dict(
    k=K, n_seqs=int(len(seqs)),
    base_composition={b: round(p[b], 3) for b in "ATGC"},
    top_enriched=[(kms[j], round(float(enrich[j]), 2)) for j in np.argsort(-enrich)[:10]],
    most_stabilizing=[(kms[j], round(float(corr[j]), 3)) for j in stab],
    most_destabilizing=[(kms[j], round(float(corr[j]), 3)) for j in destab],
    top_model_importance=[(kms[j], round(float(imp[j]), 4)) for j in top_imp],
    corr_grun_vs_stability=round(float(stats.pearsonr(gr, corr)[0]), 3),
    corr_gc_vs_stability=round(float(stats.pearsonr(gc_k, corr)[0]), 3),
)
json.dump(summary, open("results/21_kmer_analysis.json", "w"), indent=2)
print(json.dumps({k: summary[k] for k in ["most_stabilizing", "corr_grun_vs_stability", "corr_gc_vs_stability"]}, indent=2))

# ---- figure
fig = plt.figure(figsize=(16, 9)); gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.28)
# A: enrichment volcano (enrichment vs stability assoc)
ax = fig.add_subplot(gs[0, 0])
scv = ax.scatter(enrich, corr, c=gr, cmap="viridis", s=28, alpha=0.8)
ax.axhline(0, color="grey", lw=1); ax.axvline(0, color="grey", lw=1)
ax.set(xlabel="k-mer enrichment (log2 obs/exp)", ylabel="assoc. with MFE (Pearson r)",
       title="k-mer landscape: enrichment vs folding effect")
fig.colorbar(scv, ax=ax, fraction=0.046, pad=0.04).set_label("max G-run in k-mer")
# B: most stabilizing k-mers (lollipop)
ax = fig.add_subplot(gs[0, 1])
js = stab[::-1]; yy = np.arange(len(js))
ax.hlines(yy, 0, corr[js], color="#4C72B0", lw=2)
ax.scatter(corr[js], yy, s=70, color="#C44E52", zorder=3)
ax.set_yticks(yy); ax.set_yticklabels([kms[j] for j in js], family="monospace")
ax.set(xlabel="Pearson r (k-mer freq vs MFE)  — more negative = more stabilizing",
       title="Top stabilizing 4-mers")
# C: GC fraction vs stability, coloured by G-run (continuous colorbar, no crowded legend)
ax = fig.add_subplot(gs[1, 0])
sc = ax.scatter(gc_k + np.random.RandomState(1).normal(0, 0.008, len(gc_k)), corr,
                c=gr, cmap="YlOrRd", s=28, alpha=0.8, edgecolors="grey", linewidth=0.2)
ax.axhline(0, color="grey", lw=1)
cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04); cb.set_label("max G-run length in k-mer")
ax.set(xlabel="GC fraction of k-mer", ylabel="assoc. with MFE (Pearson r)",
       title=f"GC content drives stability (r={summary['corr_gc_vs_stability']}); "
             f"G-runs do not (r={summary['corr_grun_vs_stability']})\n(standard NUPACK model omits G-quadruplexes)")
# D: model importance vs data association (do they agree?)
ax = fig.add_subplot(gs[1, 1])
ax.scatter(np.abs(corr), imp, s=24, alpha=0.7, color="#55A868")
rr = stats.spearmanr(np.abs(corr), imp).correlation
for j in top_imp[:6]:
    ax.annotate(kms[j], (abs(corr[j]), imp[j]), fontsize=8, family="monospace")
ax.set(xlabel="|data association| (|Pearson r|)", ylabel="surrogate permutation importance",
       title=f"Model uses the stabilizing motifs (Spearman={rr:.2f})")
fig.suptitle("Alignment-free motif analysis: what stabilizes aptamer folding, and what the surrogate learned",
             fontweight="bold", y=0.98)
fig.savefig("results/21_kmer_analysis.png", bbox_inches="tight")
print("saved 21_kmer_analysis.png + 21_kmer_analysis.json")
