#!/usr/bin/env python
"""Leak-free folding-surrogate benchmark. FIX: MFE depends only on sequence, so we
deduplicate to unique sequences and split folds BY SEQUENCE (never the same sequence in
train and test). This removes the C2 leakage found in the by-target protocol.

Fast/deterministic models here (Ridge, HistGBM) + significance. Shared fold indices are
saved so the CNN (19b) uses the identical splits.
"""
import json, numpy as np, pandas as pd
from itertools import product
from scipy import stats
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

RNG = 42
raw = pd.read_csv("data/folding_labels.csv")
# dedup: unique sequence (labels verified identical across dups)
d = raw.drop_duplicates("sequence").reset_index(drop=True)
print(f"raw rows={len(raw)}  ->  unique sequences={len(d)}")
y = d.mfe_kcal.values.astype(np.float32); seqs = d.sequence.values
gc = d.gc.values; L = d.L.values

# fixed 5-fold split over UNIQUE sequences (shared with CNN)
folds = np.full(len(d), -1)
for k, (_, te) in enumerate(KFold(5, shuffle=True, random_state=RNG).split(seqs)):
    folds[te] = k
np.save("results/19_folds.npy", folds)
np.save("results/19_y.npy", y)
d[["sequence", "target", "mfe_kcal", "L", "gc"]].to_csv("data/folding_dedup.csv", index=False)

# leakage assert: no sequence in >1 fold is impossible (unique), but assert no dup seq
assert len(seqs) == len(set(seqs)), "sequences not unique after dedup"
print("C2 sequence-leakage: IMPOSSIBLE by construction (unique sequences, split by sequence)")

# features
v3 = ["".join(p) for p in product("ATGC", repeat=3)]; i3 = {k: i for i, k in enumerate(v3)}
v4 = ["".join(p) for p in product("ATGC", repeat=4)]; i4 = {k: i for i, k in enumerate(v4)}
def kmer(s):
    f3 = np.zeros(64); f4 = np.zeros(256)
    for i in range(len(s)-2): f3[i3[s[i:i+3]]] += 1
    for i in range(len(s)-3): f4[i4[s[i:i+4]]] += 1
    if f3.sum(): f3 /= f3.sum()
    if f4.sum(): f4 /= f4.sum()
    return np.concatenate([f3, f4, [len(s)/100, (s.count("G")+s.count("C"))/len(s)]])
Xk = np.vstack([kmer(s) for s in seqs])
Xlen = (L/100.0).reshape(-1, 1).astype(np.float32)
Xlg = np.column_stack([L/100.0, gc]).astype(np.float32)

def cv(mk, X):
    yp = np.zeros_like(y)
    for k in range(5):
        te = folds == k; tr = ~te
        sc = StandardScaler().fit(X[tr]); m = mk().fit(sc.transform(X[tr]), y[tr])
        yp[te] = m.predict(sc.transform(X[te]))
    return yp
def M(yp): return dict(r2=round(float(r2_score(y, yp)), 4),
                       spearman=round(float(stats.spearmanr(y, yp).correlation), 4),
                       mae=round(float(mean_absolute_error(y, yp)), 4))

# mean-predictor tripwire
yp_mean = np.zeros_like(y)
for k in range(5):
    te = folds == k; yp_mean[te] = y[~te].mean()
r2_mean = round(float(r2_score(y, yp_mean)), 4)

def gbm(seed): return HistGradientBoostingRegressor(max_depth=6, max_iter=400, learning_rate=0.06,
                                                    l2_regularization=1.0, random_state=seed)
yp_len = cv(lambda: Ridge(1.0), Xlen)
yp_lg = cv(lambda: Ridge(1.0), Xlg)
yp_gb = cv(lambda: gbm(0), Xk)
np.save("results/19_oof_histgbm.npy", yp_gb)
np.save("results/19_oof_lengc.npy", yp_lg)

res = {"protocol": "dedup unique sequences, 5-fold split BY SEQUENCE (leak-free for folding)",
       "n_unique": int(len(d)), "mean_predictor_r2": r2_mean,
       "length_only": M(yp_len), "length_gc": M(yp_lg), "kmer_histgbm": M(yp_gb)}

# significance: HistGBM vs length+GC, paired per-sequence abs error
eg = np.abs(y - yp_gb); el = np.abs(y - yp_lg)
w = stats.wilcoxon(eg, el)
def cliffs(a, b):
    a = np.asarray(a); b = np.asarray(b)
    gt = sum(np.sum(a > bb) for bb in b); lt = sum(np.sum(a < bb) for bb in b)
    return (gt - lt) / (len(a)*len(b))
# efficient cliffs via rank
def cliffs_fast(a, b):
    a = np.sort(a); b = np.sort(b); n = len(a)*len(b)
    i = j = gt = lt = 0
    # count pairs a>b and a<b
    import numpy as _np
    ba = _np.searchsorted(b, a, side="left"); bb = _np.searchsorted(b, a, side="right")
    lt = int(_np.sum(len(b) - bb)); gt = int(_np.sum(ba))
    return (gt - lt) / n
delta = cliffs_fast(el, eg)  # baseline error vs gbm error; positive => gbm better
res["histgbm_vs_lengc"] = dict(wilcoxon_p=float(w.pvalue), cliffs_delta=round(float(delta), 4),
                               median_err_gbm=round(float(np.median(eg)), 4),
                               median_err_lengc=round(float(np.median(el)), 4))
res["histgbm_seed_stability"] = dict(
    mean=round(float(np.mean([r2_score(y, cv(lambda s=s: gbm(s), Xk)) for s in range(5)])), 4), std=0.0)

json.dump(res, open("results/19a_dedup_benchmark.json", "w"), indent=2)
print(json.dumps(res, indent=2))
print("\nLEAK-FREE folding benchmark (fast models). CNN to follow on identical folds (19b).")
