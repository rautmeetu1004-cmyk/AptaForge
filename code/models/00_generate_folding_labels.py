#!/usr/bin/env python
"""Compute NUPACK folding labels for every corpus sequence.

Writes folding_labels.csv with columns:
  sequence, target, mfe_kcal, structural_entropy, ensemble_defect, log_pfunc, L, gc
Backend is NUPACK 4 (material='dna', 37 C). Rows where NUPACK errors are dropped.
Requires NUPACK; the resulting labels are shipped in data/folding_labels.csv so
this step is optional for reproducing downstream results.
"""
import sys, time
sys.path.insert(0, "code")
import numpy as np, pandas as pd
from nupack_backend import compute_nupack_features, nupack_available

assert nupack_available(), "NUPACK not available"
df = pd.read_csv("data/aptamer_corpus.csv")
# Binding-affinity annotations from the source databases are not part of this
# release; the folding corpus is defined by the canonical-base and length filters
# below, and its labels are provided in data/folding_labels.csv.
df = df.copy()
df["L"] = df.sequence.str.len()
df = df[(df.L >= 15) & (df.L <= 100)].drop_duplicates(["sequence", "target"]).reset_index(drop=True)
print(f"computing real NUPACK MFE for {len(df)} sequences (dna, 37C)...")

rows, t0 = [], time.time()
for i, r in df.iterrows():
    s = r.sequence.upper().replace("U", "T")
    if set(s) - set("ACGT"):
        continue
    try:
        f = compute_nupack_features(s, temperature=37.0)
        rows.append(dict(sequence=s, target=r.target,
                         mfe_kcal=float(f["mfe_kcal"]),
                         structural_entropy=float(f["structural_entropy"]),
                         ensemble_defect=float(f["ensemble_defect"]),
                         log_pfunc=float(f["log_pfunc"]),
                         L=len(s), gc=(s.count("G")+s.count("C"))/len(s)))
    except Exception as e:
        if i < 5: print("  err", i, e)
    if (i + 1) % 300 == 0:
        print(f"  [{i+1}/{len(df)}]  {(time.time()-t0):.0f}s elapsed")

out = pd.DataFrame(rows)
out.to_csv("data/folding_labels.csv", index=False)
print(f"\nsaved {len(out)} rows -> data/folding_labels.csv  ({time.time()-t0:.0f}s)")
print(f"REAL MFE: min={out.mfe_kcal.min():.2f}  max={out.mfe_kcal.max():.2f}  "
      f"mean={out.mfe_kcal.mean():.2f}  std={out.mfe_kcal.std():.2f}")
print(f"  fraction MFE==0 (no stable fold): {(out.mfe_kcal==0).mean()*100:.1f}%")
print(f"REAL entropy: mean={out.structural_entropy.mean():.2f}  std={out.structural_entropy.std():.2f}")
print(f"corr(MFE, GC)={np.corrcoef(out.mfe_kcal, out.gc)[0,1]:.3f}  "
      f"corr(MFE, L)={np.corrcoef(out.mfe_kcal, out.L)[0,1]:.3f}")
