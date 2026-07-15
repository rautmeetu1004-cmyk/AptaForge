#!/usr/bin/env python
"""CNN surrogate on the leak-free dedup benchmark, using the SAME fold indices as 19a.
Regularised, nested early-stopping (inner val carved from each fold's TRAIN only).
"""
import json, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error

RNG = 0; np.random.seed(RNG); torch.manual_seed(RNG)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
d = pd.read_csv("data/folding_dedup.csv")
y = d.mfe_kcal.values.astype(np.float32); seqs = d.sequence.values
L = d.L.values.astype(np.float32); gc = d.gc.values.astype(np.float32)
folds = np.load("results/19_folds.npy")
assert len(folds) == len(d)
BASES = "ATGC"; B2I = {b: i for i, b in enumerate(BASES)}; PAD = 4; MAXLEN = 100
Xtok = np.array([[B2I.get(c, PAD) for c in s[:MAXLEN]] + [PAD]*(MAXLEN-len(s)) for s in seqs])
aux = np.column_stack([L/100.0, gc]).astype(np.float32)
print(f"dedup CNN on {len(d)} unique sequences, shared folds")

class CNN(nn.Module):
    def __init__(s, dh=96, p=0.3):
        super().__init__()
        s.emb = nn.Embedding(5, 32, padding_idx=PAD)
        s.c1 = nn.Conv1d(32, dh, 7, padding=3); s.b1 = nn.BatchNorm1d(dh)
        s.c2 = nn.Conv1d(dh, dh, 5, padding=2); s.b2 = nn.BatchNorm1d(dh)
        s.c3 = nn.Conv1d(dh, dh, 3, padding=1); s.b3 = nn.BatchNorm1d(dh)
        s.drop = nn.Dropout(p)
        s.head = nn.Sequential(nn.Linear(dh*2+2, dh), nn.ReLU(), nn.Dropout(p), nn.Linear(dh, 1))
    def forward(s, x, a):
        m = (x != PAD).float().unsqueeze(1)
        h = s.emb(x).transpose(1, 2)
        h = F.relu(s.b1(s.c1(h))); h = F.relu(s.b2(s.c2(h))); h = F.relu(s.b3(s.c3(h)))
        h = h*m
        mean = h.sum(-1)/m.sum(-1).clamp(min=1); mx = (h+(m-1)*1e4).max(-1).values
        return s.head(s.drop(torch.cat([mean, mx, a], 1))).squeeze(1)

def run_fold(tr, te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    idx = np.where(tr)[0]; rng = np.random.default_rng(seed); rng.shuffle(idx)
    nval = max(16, int(0.15*len(idx))); vi = idx[:nval]; ti = idx[nval:]
    ym, ysd = y[ti].mean(), y[ti].std()
    def T(a): return torch.tensor(a, device=device)
    Xt, At, yt = T(Xtok[ti]), T(aux[ti]), T((y[ti]-ym)/ysd)
    Xv, Av, yv = T(Xtok[vi]), T(aux[vi]), y[vi]
    net = CNN().to(device); opt = torch.optim.AdamW(net.parameters(), 3e-3, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=12)
    best = {"v": 1e9, "s": None}; since = 0; BS = 64
    for ep in range(300):
        net.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), BS):
            b = perm[i:i+BS]; opt.zero_grad()
            F.mse_loss(net(Xt[b], At[b]), yt[b]).backward()
            nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
        net.eval()
        with torch.no_grad(): pv = net(Xv, Av).cpu().numpy()*ysd+ym
        v = float(np.mean((pv-yv)**2)); sch.step(v)
        if v < best["v"]-1e-4: best = {"v": v, "s": {k: vv.detach().cpu().clone() for k, vv in net.state_dict().items()}}; since = 0
        else: since += 1
        if since >= 30: break
    net.load_state_dict(best["s"]); net.eval()
    with torch.no_grad(): return net(T(Xtok[te]), T(aux[te])).cpu().numpy()*ysd+ym

t0 = time.time(); oof = np.zeros_like(y)
for k in range(5):
    te = folds == k
    oof[te] = run_fold(~te, te, seed=RNG)
    print(f"  fold {k}: R2={r2_score(y[te], oof[te]):+.3f}  ({time.time()-t0:.0f}s)")
np.save("results/19_oof_cnn.npy", oof)
res = dict(model="CNN (regularised, nested early-stop)", protocol="dedup, split-by-sequence, same folds as 19a",
           r2=round(float(r2_score(y, oof)), 4), spearman=round(float(stats.spearmanr(y, oof).correlation), 4),
           mae=round(float(mean_absolute_error(y, oof)), 4))
json.dump(res, open("results/19b_cnn_dedup.json", "w"), indent=2)
print("\nFINAL CNN (leak-free, comparable):", res)
print("reference HistGBM: R2=0.611 Spearman=0.798 MAE=1.805")
