#!/usr/bin/env python3
"""
train_gnn.py  --  graph neural network that predicts permeability from the pore network.

A GraphSAGE encoder reads each pore graph (nodes = pores with size, coordination, and
position; edges = throats), global-pools it, concatenates a few graph-level features that
carry absolute scale (mean pore size, graph size, mean coordination), and an MLP head
predicts log10(permeability).

It reports TWO evaluations, because they answer different questions:
  * random 5-fold CV            -- can the model predict permeability at all?
  * grouped-by-rock CV          -- can it generalise to an entirely unseen rock type?
Grouped CV is the strict test; random CV is the standard one.

Node features are log-scaled where appropriate so the model handles the wide range of
pore sizes and permeabilities across rock types.

INSTALL: pip install torch torch-geometric
RUN:     python train_gnn.py --graphs graphs/
         python train_gnn.py --graphs graphs/ --no-global   # ablation: drop global feats
Outputs: metrics for both splits, and fig_gnn_pred_vs_actual.png (grouped split).
"""

import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import SAGEConv, global_mean_pool
from sklearn.model_selection import GroupKFold, KFold

ACC_TOL = 0.5          # log10 units (~ factor of 3)
EPS = 1e-9
GLOBAL_DIM = 4


def load_records(graph_dir):
    recs = []
    for f in sorted(glob.glob(os.path.join(graph_dir, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        recs.append(dict(x=z["x"].astype(np.float64),
                         edge_index=z["edge_index"].astype(np.int64),
                         y=float(z["y"]), source=str(z["source"])))
    return recs


def transform_x(xraw):
    """log-scale pore diameter (column 0); keep coordination and position."""
    xt = xraw.copy()
    xt[:, 0] = np.log10(np.clip(xt[:, 0], EPS, None))
    return xt


def global_feats(xraw):
    """graph-level features carrying absolute scale the pooled embedding lacks."""
    d, deg = xraw[:, 0], xraw[:, 1]
    return np.array([np.log10(max(d.mean(), EPS)),
                     np.log10(max(len(d), 1)),
                     deg.mean(),
                     np.log10(max(np.median(d), EPS))], dtype=np.float64)


def make_data(rec, xm, xs, gm, gs, use_global=True):
    xt = (transform_x(rec["x"]) - xm) / xs
    g = (global_feats(rec["x"]) - gm) / gs if use_global else np.zeros(GLOBAL_DIM)
    return Data(x=torch.tensor(xt, dtype=torch.float),
                edge_index=torch.tensor(rec["edge_index"], dtype=torch.long),
                y=torch.tensor([rec["y"]], dtype=torch.float),
                g=torch.tensor(g[None, :], dtype=torch.float))


class GNN(torch.nn.Module):
    def __init__(self, in_dim, hidden=64, gdim=GLOBAL_DIM):
        super().__init__()
        self.c1 = SAGEConv(in_dim, hidden)
        self.c2 = SAGEConv(hidden, hidden)
        self.c3 = SAGEConv(hidden, hidden)
        self.lin1 = torch.nn.Linear(hidden + gdim, hidden)
        self.lin2 = torch.nn.Linear(hidden, 1)

    def forward(self, x, edge_index, batch, g):
        x = F.relu(self.c1(x, edge_index))
        x = F.relu(self.c2(x, edge_index))
        x = F.relu(self.c3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = torch.cat([x, g], dim=1)
        x = F.relu(self.lin1(x))
        return self.lin2(x).squeeze(-1)


def train_fold(train_recs, test_recs, use_global, epochs, device):
    Xtr = np.vstack([transform_x(r["x"]) for r in train_recs])
    xm, xs = Xtr.mean(0), Xtr.std(0); xs[xs == 0] = 1
    Gtr = np.vstack([global_feats(r["x"]) for r in train_recs])
    gm, gs = Gtr.mean(0), Gtr.std(0); gs[gs == 0] = 1

    train = [make_data(r, xm, xs, gm, gs, use_global) for r in train_recs]
    test = [make_data(r, xm, xs, gm, gs, use_global) for r in test_recs]
    tl = DataLoader(train, batch_size=16, shuffle=True)
    vl = DataLoader(test, batch_size=32)

    model = GNN(train[0].num_node_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for b in tl:
            b = b.to(device); opt.zero_grad()
            loss = F.mse_loss(model(b.x, b.edge_index, b.batch, b.g), b.y)
            loss.backward(); opt.step()

    model.eval(); preds, ys = [], []
    with torch.no_grad():
        for b in vl:
            b = b.to(device)
            preds.append(model(b.x, b.edge_index, b.batch, b.g).cpu().numpy())
            ys.append(b.y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(ys)


def run_cv(recs, splitter, use_global, epochs, device, name=""):
    pred = np.zeros(len(recs)); actual = np.zeros(len(recs))
    folds = list(splitter)
    for k, (tr, te) in enumerate(folds, 1):
        print(f"  [{name}] fold {k}/{len(folds)} training...", flush=True)
        p, a = train_fold([recs[i] for i in tr], [recs[i] for i in te], use_global, epochs, device)
        pred[te] = p; actual[te] = a
    r2 = 1 - np.sum((actual - pred) ** 2) / np.sum((actual - actual.mean()) ** 2)
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    acc = 100 * np.mean(np.abs(actual - pred) <= ACC_TOL)
    return r2, rmse, acc, pred, actual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="graphs/")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--no-global", action="store_true", help="ablation: drop graph-level features")
    ap.add_argument("--min-logk", type=float, default=-4.0,
                    help="drop samples with log10(K) below this (failed/disconnected sims)")
    args = ap.parse_args()
    use_global = not args.no_global

    recs = load_records(args.graphs)
    before = len(recs)
    recs = [r for r in recs if r["y"] >= args.min_logk]
    if len(recs) < before:
        print(f"[gnn] dropped {before - len(recs)} samples with log10(K) < {args.min_logk} (failed sims)")
    sources = np.array([r["source"] for r in recs])
    n_groups = len(set(sources))
    print(f"[gnn] {len(recs)} graphs from {n_groups} rocks: {sorted(set(sources))}")
    print(f"[gnn] global features: {'ON' if use_global else 'OFF (ablation)'}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) random 5-fold: can it predict at all?
    rk = KFold(n_splits=5, shuffle=True, random_state=0).split(recs)
    r2r, rmser, accr, _, _ = run_cv(recs, rk, use_global, args.epochs, device, name="random")
    print(f"\n[random 5-fold]  R2={r2r:.2f}  RMSE={rmser:.2f}  acc={accr:.0f}%")

    # 2) grouped by rock: can it generalise to an unseen rock type?
    if n_groups >= 2:
        gk = GroupKFold(n_splits=min(5, n_groups)).split(recs, groups=sources)
        r2g, rmseg, accg, predg, actg = run_cv(recs, gk, use_global, args.epochs, device, name="grouped")
        print(f"[grouped-by-rock] R2={r2g:.2f}  RMSE={rmseg:.2f}  acc={accg:.0f}%  (strict generalisation)")
    else:
        predg, actg, r2g, rmseg, accg = None, None, float("nan"), float("nan"), float("nan")

    if predg is not None:
        plt.figure(figsize=(5.5, 5.5))
        lo, hi = min(actg.min(), predg.min()) - 0.3, max(actg.max(), predg.max()) + 0.3
        plt.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect prediction")
        plt.scatter(actg, predg, s=45, edgecolor="k", linewidth=0.4)
        plt.xlim(lo, hi); plt.ylim(lo, hi)
        plt.xlabel("Actual log10 permeability (mD)"); plt.ylabel("Predicted log10 permeability (mD)")
        plt.title(f"GNN, grouped-by-rock CV\nR2={r2g:.2f}, RMSE={rmseg:.2f}, acc={accg:.0f}%")
        plt.legend(fontsize=8); plt.tight_layout()
        plt.savefig("fig_gnn_pred_vs_actual.png", dpi=150)
        print("wrote fig_gnn_pred_vs_actual.png")


if __name__ == "__main__":
    main()
