#!/usr/bin/env python
"""Constant-region (SELEX primer) analysis of the aptamer corpus.

SELEX libraries carry fixed primer regions flanking a randomised core. Because those
constant regions are shared across many sequences and themselves fold, they matter for
any folding analysis. Here we (1) identify recurrent 5' and 3' flanks, (2) quantify how
much of the corpus carries an identifiable primer, and (3) estimate the folding-energy
contribution of the constant regions on a subset re-folded with NUPACK.
"""
import sys, json, time
sys.path.insert(0, "code")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from collections import Counter
from matplotlib.gridspec import GridSpec

plt.rcParams.update({"figure.dpi": 300, "font.size": 11, "axes.grid": True, "grid.alpha": 0.25})
raw = pd.read_csv("data/aptamer_corpus.csv")
d = pd.read_csv("data/folding_dedup.csv")
seqs = d.sequence.values
F = 12   # flank length to test

pre = Counter(s[:F] for s in seqs if len(s) >= 2*F)
suf = Counter(s[-F:] for s in seqs if len(s) >= 2*F)
top_pre = pre.most_common(10); top_suf = suf.most_common(10)

# a sequence is "primer-bearing" if its 5' or 3' flank is shared by >=3 sequences
shared_pre = {k for k, c in pre.items() if c >= 3}
shared_suf = {k for k, c in suf.items() if c >= 3}
has_primer = np.array([(s[:F] in shared_pre) or (s[-F:] in shared_suf) for s in seqs])
frac_primer = float(has_primer.mean())

# how many distinct library families (distinct shared prefixes)
n_families = len(shared_pre)
print(f"{len(seqs)} sequences; flank={F}")
print(f"distinct recurrent 5' flanks (>=3 seqs): {n_families}")
print(f"fraction of corpus carrying an identifiable primer flank: {frac_primer*100:.1f}%")
print("top 5' flanks:", top_pre[:5])
print("top 3' flanks:", top_suf[:5])

# ---- estimate primer folding contribution: refold constant vs core on a subset
from nupack_backend import compute_nupack_features
def mfe(s):
    try: return float(compute_nupack_features(s.upper().replace("U", "T"), temperature=37.0)["mfe_kcal"])
    except Exception: return np.nan

# take sequences from the single most common primer family, trim both flanks, refold core
fam_pre = top_pre[0][0]
fam = [s for s in seqs if s[:F] == fam_pre and len(s) > 3*F][:40]
rows = []
t0 = time.time()
for s in fam:
    core = s[F:-F]
    rows.append({"full_mfe": mfe(s), "core_mfe": mfe(core), "full_L": len(s), "core_L": len(core)})
fam_df = pd.DataFrame(rows).dropna()
dprimer = (fam_df.full_mfe - fam_df.core_mfe)   # how much the flanks change MFE
print(f"\nfamily '{fam_pre}...' n={len(fam_df)}: full MFE {fam_df.full_mfe.mean():.1f} vs core MFE {fam_df.core_mfe.mean():.1f}")
print(f"flanks shift MFE by mean {dprimer.mean():.2f} kcal/mol ({time.time()-t0:.0f}s)")

summary = dict(flank_len=F, n_sequences=int(len(seqs)),
               n_recurrent_5p_flanks=int(n_families),
               frac_corpus_with_primer=round(frac_primer, 3),
               top_5p_flanks=[(k, int(c)) for k, c in top_pre[:8]],
               top_3p_flanks=[(k, int(c)) for k, c in top_suf[:8]],
               primer_family=fam_pre, family_n=int(len(fam_df)),
               full_mfe_mean=round(float(fam_df.full_mfe.mean()), 2),
               core_mfe_mean=round(float(fam_df.core_mfe.mean()), 2),
               flank_mfe_shift_mean=round(float(dprimer.mean()), 2))
json.dump(summary, open("results/25_primer_analysis.json", "w"), indent=2)

# ---- figure (lollipops + cumulative-coverage + dumbbell; no plain bars)
from scipy.stats import gaussian_kde
fig = plt.figure(figsize=(17, 5.2)); gs = GridSpec(1, 3, figure=fig, wspace=0.42, width_ratios=[1.15, 1, 1.1])

# A: recurrent flanks as a two-colour lollipop (5' and 3' interleaved by rank)
ax = fig.add_subplot(gs[0, 0])
n = 8
pre8 = top_pre[:n]; suf8 = top_suf[:n]
yy = np.arange(n)
ax.hlines(yy+0.18, 0, [c for _, c in pre8][::-1], color="#4C72B0", lw=2, alpha=0.5)
ax.scatter([c for _, c in pre8][::-1], yy+0.18, s=70, color="#4C72B0", zorder=3, label="5' flank")
ax.hlines(yy-0.18, 0, [c for _, c in suf8][::-1], color="#55A868", lw=2, alpha=0.5)
ax.scatter([c for _, c in suf8][::-1], yy-0.18, s=70, color="#55A868", marker="s", zorder=3, label="3' flank")
for i, (k, c) in enumerate(pre8[::-1]): ax.text(c+1, i+0.18, k, va="center", fontsize=7, family="monospace", color="#2c4a70")
for i, (k, c) in enumerate(suf8[::-1]): ax.text(c+1, i-0.18, k, va="center", fontsize=7, family="monospace", color="#2f5d3a")
ax.set_yticks([]); ax.set_xlim(0, max(pre8[0][1], suf8[0][1])*1.35)
ax.set(xlabel="sequences sharing the flank", title=f"A  Recurrent SELEX constant regions ({F} nt)")
ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

# B: cumulative corpus coverage by number of primer families included
ax = fig.add_subplot(gs[0, 1])
fam_counts = sorted([c for _, c in pre.items()], reverse=True)
cov = np.cumsum(fam_counts)/len(seqs)*100
ax.fill_between(range(1, len(cov)+1), cov, alpha=0.18, color="#8172B3", step="mid")
ax.step(range(1, len(cov)+1), cov, where="mid", color="#8172B3", lw=2)
ax.axhline(frac_primer*100, color="grey", ls="--", lw=1)
ax.text(len(cov)*0.55, frac_primer*100+2, f"{frac_primer*100:.0f}% carry a shared flank", fontsize=8, color="grey")
ax.set(xlabel="number of 5' flank families (ranked)", ylabel="cumulative % of corpus",
       title="B  A few primer families cover much of the corpus", xlim=(1, min(60, len(cov))))

# C: dumbbell — every family sequence loses folding energy when the primer is trimmed
ax = fig.add_subplot(gs[0, 2])
fam_sorted = fam_df.sort_values("full_mfe").reset_index(drop=True)
yy = np.arange(len(fam_sorted))
ax.hlines(yy, fam_sorted.core_mfe, fam_sorted.full_mfe, color="#cccccc", lw=1.6, zorder=1)
ax.scatter(fam_sorted.full_mfe, yy, s=26, color="#C44E52", zorder=3, label="full sequence")
ax.scatter(fam_sorted.core_mfe, yy, s=26, color="#4C72B0", zorder=3, label="primer-trimmed core")
ax.set_yticks([])
ax.set(xlabel="MFE (kcal/mol)", ylabel=f"family sequences (n={len(fam_sorted)}, sorted)",
       title=f"C  Trimming primers costs {abs(dprimer.mean()):.1f} kcal/mol per sequence")
ax.legend(loc="lower left", framealpha=0.9, fontsize=9)

fig.suptitle(f"SELEX constant-region analysis: {frac_primer*100:.0f}% of the corpus carries an identifiable primer flank",
             fontweight="bold", y=1.02)
fig.savefig("results/25_primer_analysis.png", bbox_inches="tight")
print(json.dumps(summary, indent=2)); print("saved 25_primer_analysis.png + .json")
