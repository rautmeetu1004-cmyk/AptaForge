#!/usr/bin/env python
"""Per-target conservation, done honestly. A naive MSA of these sequences is confounded
by mixed SELEX libraries (different primer flanks), so instead we test the KNOWN biology:
thrombin aptamers fold as G-quadruplexes. We recover that signature statistically and
show enriched motifs per target relative to the dataset background.
"""
import json, re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from itertools import product
from scipy import stats

plt.rcParams.update({"figure.dpi": 300, "font.size": 8, "font.family": "Arial", "axes.grid": True, "grid.alpha": 0.25})
d = pd.read_csv("data/folding_dedup.csv")   # unique modelling corpus (n=1,426)
def gfrac(s): return s.count("G") / len(s)
def g_tracts(s): return len(re.findall(r"G{2,}", s))     # runs of >=2 G
d["gfrac"] = d.sequence.map(gfrac); d["gtracts"] = d.sequence.map(g_tracts)
d["is_g4"] = d.gtracts >= 4                                # canonical G-quadruplex needs >=4 G-tracts

TARGETS = d.target.value_counts().head(8).index.tolist()
LAB = {"POL_HV1H2": "HIV-1 Pol", "NRP1_HUMAN": "NRP1", "Ochratoxin A": "Ochratoxin A",
       "Thrombin, Human": "Thrombin", "Aflatoxin M1": "Aflatoxin M1", "SUH_HUMAN": "SUH",
       "Immunoglobulin E (IgE), Human": "IgE", "Escherichia coli (E. coli) core bacterial RNA polymeras": "E.coli RNAP"}
def lab(t): return LAB.get(t, t[:14])

# thrombin vs background stats
th = d[d.target == "Thrombin, Human"]; bg = d[d.target != "Thrombin, Human"]
mwu = stats.mannwhitneyu(th.gfrac, bg.gfrac, alternative="greater")
a, b = th.gtracts.values, bg.gtracts.values
delta = (np.sum(a[:, None] > b[None, :]) - np.sum(a[:, None] < b[None, :])) / (len(a)*len(b))
stat = dict(thrombin_n=int(len(th)), thrombin_gfrac_median=round(float(th.gfrac.median()), 3),
            bg_gfrac_median=round(float(bg.gfrac.median()), 3), gfrac_mwu_p=float(mwu.pvalue),
            thrombin_pct_g4=round(float(th.is_g4.mean()*100), 1), bg_pct_g4=round(float(bg.is_g4.mean()*100), 1),
            gtracts_cliffs_delta=round(float(delta), 3))

# per-target enriched 4-mers vs background
K = 4; kms = ["".join(p) for p in product("ATGC", repeat=K)]; kidx = {k: i for i, k in enumerate(kms)}
def kfreq(rows):
    c = np.zeros(len(kms))
    for s in rows:
        for i in range(len(s)-K+1): c[kidx[s[i:i+K]]] += 1
    return c / c.sum() if c.sum() else c
bg_f = kfreq(d.sequence) + 1e-6
enrich_tbl = {}
for t in TARGETS:
    tf = kfreq(d[d.target == t].sequence) + 1e-6
    e = np.log2(tf / bg_f)
    top = np.argsort(-e)[:5]
    enrich_tbl[lab(t)] = [(kms[j], round(float(e[j]), 2)) for j in top]

# ---- figure (violin + beeswarm-style dots + lollipop; no plain bars)
fig = plt.figure(figsize=(7.2, 3.4)); gs = GridSpec(1, 3, figure=fig, wspace=0.58)
# A: G-fraction thrombin vs background — violin with jittered points overlaid
ax = fig.add_subplot(gs[0, 0])
parts = ax.violinplot([bg.gfrac.values, th.gfrac.values], showmeans=True, showextrema=False)
for pc, c in zip(parts["bodies"], ["#B0B0B0", "#C44E52"]):
    pc.set_facecolor(c); pc.set_alpha(0.55)
rngg = np.random.RandomState(0)
ax.scatter(2 + rngg.normal(0, 0.05, len(th)), th.gfrac.values, s=22, color="#7a1f27",
           alpha=0.7, zorder=3, edgecolors="white", linewidth=0.4)
ax.set_xticks([1, 2]); ax.set_xticklabels([f"background\n(n={len(bg)})", f"thrombin\n(n={len(th)})"])
ax.set(ylabel="G fraction"); ax.set_title("A", loc="left", fontweight="bold", fontsize=9)
ax.text(0.03, 0.97, f"Mann–Whitney\np = {mwu.pvalue:.1e}", transform=ax.transAxes, ha="left", va="top", fontsize=6.5)
# B: G-quadruplex prevalence across targets — lollipop (dot + stem), coloured
ax = fig.add_subplot(gs[0, 1])
pct = [d[d.target == t].is_g4.mean()*100 for t in TARGETS]
names = [lab(t) for t in TARGETS]
order = np.argsort(pct); yy = np.arange(len(TARGETS))
vals = [pct[i] for i in order]; cols = ["#C44E52" if names[i] == "Thrombin" else "#4C72B0" for i in order]
ax.hlines(yy, bg.is_g4.mean()*100, vals, color=cols, lw=2, alpha=0.5)
ax.scatter(vals, yy, s=95, color=cols, zorder=3, edgecolors="white", linewidth=1)
ax.axvline(bg.is_g4.mean()*100, color="grey", ls="--", lw=1.5)
ax.text(bg.is_g4.mean()*100+1, 0.2, f"dataset mean {bg.is_g4.mean()*100:.0f}%", fontsize=8, color="grey", rotation=90, va="bottom")
ax.set_yticks(yy); ax.set_yticklabels([names[i] for i in order], fontsize=7)
ax.set(xlabel="% with G-quadruplex signature", xlim=(0, 105)); ax.set_title("B", loc="left", fontweight="bold", fontsize=9)
# C: thrombin enriched 4-mers — lollipop
ax = fig.add_subplot(gs[0, 2])
tf = kfreq(th.sequence) + 1e-6; e = np.log2(tf / bg_f); top = np.argsort(-e)[:10]
yy = np.arange(len(top))[::-1]
ax.hlines(yy, 0, e[top], color="#C44E52", lw=2, alpha=0.5)
ax.scatter(e[top], yy, s=85, color="#C44E52", zorder=3, edgecolors="white", linewidth=1)
ax.set_yticks(yy); ax.set_yticklabels([kms[j] for j in top], family="monospace", fontsize=7)
ax.set(xlabel="log₂ enrichment vs background"); ax.set_title("C", loc="left", fontweight="bold", fontsize=9)
fig.savefig("figures/Figure_10_thrombin_control.png", bbox_inches="tight")
json.dump({"thrombin_g4_control": stat, "per_target_top_4mers": enrich_tbl},
          open("results/23_per_target.json", "w"), indent=2)
print(json.dumps(stat, indent=2))
print("thrombin top enriched 4-mers:", enrich_tbl["Thrombin"])
print("saved 23_per_target.png + 23_per_target.json")
