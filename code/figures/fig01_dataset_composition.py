#!/usr/bin/env python
"""Dataset-contents / feature landscape, in the spirit of the earlier surrogate paper's
feature-engineering figure but with a single clean legend and no overplotting. We compare
the biophysical feature distributions of the two main sources of the corpus, AptaDB and the
University of Texas database, across the descriptors and the real folding energy. This
documents the composition of the corpus and shows the two sources are broadly comparable.
"""
import json, numpy as np, pandas as pd, matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import gaussian_kde, mannwhitneyu

plt.rcParams.update({"figure.dpi": 300, "font.size": 8, "font.family": "Arial", "axes.grid": True, "grid.alpha": 0.25})
raw = pd.read_csv("data/aptamer_corpus.csv")
raw["seq"] = raw.sequence.str.upper().str.replace("U", "T")
src = raw.drop_duplicates("seq").set_index("seq").source
d = pd.read_csv("data/folding_dedup.csv")
feat = pd.read_csv("data/sequence_features.csv")
bio = pd.read_csv("data/folding_labels.csv").drop_duplicates("sequence")
d = d.merge(bio[["sequence", "structural_entropy"]], on="sequence", how="left")
for c in feat.columns: d[c] = feat[c].values
d["source"] = d.sequence.map(src)
d = d[d.source.isin(["aptadb", "utexas"])].reset_index(drop=True)
COL = {"aptadb": "#3B6FB0", "utexas": "#C88A3B"}
NAME = {"aptadb": "AptaDB", "utexas": "UTexas DB"}
n = d.source.value_counts().to_dict()

PANELS = [("Length", "nt"), ("GC content", ""), ("mfe_kcal", "kcal/mol"),
          ("structural_entropy", "bits"), ("Purine ratio", ""), ("Scaffolding index", "")]
TITLES = {"Length": "Length", "GC content": "GC content", "mfe_kcal": "Folding energy (MFE)",
          "structural_entropy": "Structural entropy", "Purine ratio": "Purine ratio",
          "Scaffolding index": "Scaffolding index"}

fig = plt.figure(figsize=(7.2, 4.8)); gs = GridSpec(2, 3, figure=fig, hspace=0.62, wspace=0.3)
stats_out = {}
for idx, (col, unit) in enumerate(PANELS):
    ax = fig.add_subplot(gs[idx // 3, idx % 3])
    vals_all = d[col].values
    lo, hi = np.percentile(vals_all, 1), np.percentile(vals_all, 99)
    xs = np.linspace(lo, hi, 200)
    for s in ["aptadb", "utexas"]:
        v = d[d.source == s][col].values
        v = v[(v >= lo) & (v <= hi)]
        if len(v) > 5 and v.std() > 0:
            kde = gaussian_kde(v)
            ax.fill_between(xs, kde(xs), alpha=0.35, color=COL[s])
            ax.plot(xs, kde(xs), color=COL[s], lw=2)
            ax.axvline(np.median(d[d.source == s][col]), color=COL[s], ls="--", lw=1.3, alpha=0.8)
    u = mannwhitneyu(d[d.source == "aptadb"][col], d[d.source == "utexas"][col]).pvalue
    stats_out[col] = dict(aptadb_median=round(float(d[d.source=="aptadb"][col].median()), 3),
                          utexas_median=round(float(d[d.source=="utexas"][col].median()), 3),
                          mwu_p=float(u))
    ax.set(xlabel=f"{TITLES[col]}" + (f" ({unit})" if unit else ""), ylabel="density")
    ax.set_title(chr(65+idx), loc="left", fontweight="bold", fontsize=9)
    ax.set_yticks([])
# single shared legend
handles = [plt.Line2D([0], [0], color=COL[s], lw=6, alpha=0.6,
                      label=f"{NAME[s]} (n={n.get(s,0)})") for s in ["aptadb", "utexas"]]
fig.legend(handles=handles, loc="upper center", ncol=2, frameon=True, fontsize=7.5,
           bbox_to_anchor=(0.5, 1.02))
fig.savefig("figures/Figure_01_dataset_composition.png", bbox_inches="tight")
json.dump({"n_by_source": {k: int(v) for k, v in n.items()}, "feature_stats": stats_out},
          open("results/29_dataset_contents.json", "w"), indent=2)
print("saved 29_dataset_contents.png + .json")
print(json.dumps({"n": n}, indent=2))
