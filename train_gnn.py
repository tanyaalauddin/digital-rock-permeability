#!/usr/bin/env python3
"""
train_gnn.py  --  graph neural network that predicts permeability directly from the
pore network (not from hand-picked metrics).

A GraphSAGE encoder reads the pore graph (nodes = pores with size/coordination/position,
edges = throats) and a global pooling + MLP head predicts log10(permeability). It is
evaluated with LEAKAGE-AWARE cross-validation: folds are grouped by parent rock, so
subvolumes from the same scan never appear in both train and test. That is the honest
test of whether the model generalises to unseen rock.

INSTALL:
  pip install torch torch-geometric        # see pytorch.org / pyg docs for your system
RUN (after export_graphs.py has populated graphs/):
  python train_gnn.py --graphs graphs/

Outputs: cross-validated R2 / RMSE / accuracy printed, and fig_gnn_pred_vs_actual.png.
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

ACC_TOL = 0.5  # log10 units (~ factor of 3)


def load_records(graph_dir):
    recs = []
    for f in sorted(glob.glob(os.path.join(graph_dir, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        recs.append(dict(x=z["x"].astype(np.float32),
                         edge_index=z["edge_index"].astype(np.int64),
                         y=float(z["y"]),
                         source=str(z["source"])))
    return recs


def make_data(rec, mean, std):
    x = (rec["x"] - mean) / std
    return Data(x=torch.tensor(x, dtype=torch.float),
                edge_index=torch.tensor(rec["edge_index"], dtype=torch.long),
                y=torch.tensor([rec["y"]], dtype=torch.float))


class GNN(torch.nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.c1 = SAGEConv(in_dim, hidden)
        self.c2 = SAGEConv(hidden, hidden)
        self.c3 = SAGEConv(hidden, hidden)
        self.lin1 = torch.nn.Linear(hidden, hidden)
        self.lin2 = torch.nn.Linear(hidden, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.c1(x, edge_index))
        x = F.relu(self.c2(x, edge_index))
        x = F.relu(self.c3(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        return self.lin2(x).squeeze(-1)


def train_fold(train_recs, test_recs, epochs=200, lr=1e-3, device="cpu"):
    Xtr = np.vstack([r["x"] for r in train_recs])
    mean, std = Xtr.mean(0), Xtr.std(0)
    std[std == 0] = 1.0
    train = [make_data(r, mean, std) for r in train_recs]
    test = [make_data(r, mean, std) for r in test_recs]
    tl = DataLoader(train, batch_size=16, shuffle=True)
    vl = DataLoader(test, batch_size=32)

    model = GNN(train[0].num_node_features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    for _ in range(epochs):
        model.train()
        for b in tl:
            b = b.to(device)
            opt.zero_grad()
            pred = model(b.x, b.edge_index, b.batch)
            loss = F.mse_loss(pred, b.y)
            loss.backward()
            opt.step()

    model.eval()
    preds, ys = [], []
    with torch.no_grad():
        for b in vl:
            b = b.to(device)
            preds.append(model(b.x, b.edge_index, b.batch).cpu().numpy())
            ys.append(b.y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="graphs/")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    recs = load_records(args.graphs)
    if len(recs) < 8:
        print(f"[gnn] only {len(recs)} graphs found; results will be noisy. Export more first.")
    sources = np.array([r["source"] for r in recs])
    y_all = np.array([r["y"] for r in recs])
    n_groups = len(set(sources))
    print(f"[gnn] {len(recs)} graphs from {n_groups} parent rocks: {sorted(set(sources))}")

    # leakage-aware split when we have multiple parent rocks; else plain KFold
    if n_groups >= 2:
        k = min(args.folds, n_groups)
        splitter = GroupKFold(n_splits=k).split(recs, groups=sources)
        print(f"[gnn] using GroupKFold by parent rock (k={k}) -- no rock in both train and test")
    else:
        k = min(args.folds, len(recs))
        splitter = KFold(n_splits=k, shuffle=True, random_state=0).split(recs)
        print(f"[gnn] only one parent rock; falling back to plain {k}-fold (note the leakage caveat)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred = np.zeros(len(recs)); actual = np.zeros(len(recs))
    for tr_idx, te_idx in splitter:
        p, a = train_fold([recs[i] for i in tr_idx], [recs[i] for i in te_idx],
                          epochs=args.epochs, device=device)
        pred[te_idx] = p; actual[te_idx] = a

    ss_res = np.sum((actual - pred) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((actual - pred) ** 2))
    acc = 100 * np.mean(np.abs(actual - pred) <= ACC_TOL)
    print("\n=== GNN, leakage-aware cross-validation ===")
    print(f"R2={r2:.2f}  RMSE={rmse:.2f} log10-mD  accuracy(within x3)={acc:.0f}%")

    plt.figure(figsize=(5.5, 5.5))
    lo, hi = min(actual.min(), pred.min()) - 0.3, max(actual.max(), pred.max()) + 0.3
    plt.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect prediction")
    plt.scatter(actual, pred, s=45, edgecolor="k", linewidth=0.4)
    plt.xlim(lo, hi); plt.ylim(lo, hi)
    plt.xlabel("Actual log10 permeability (mD)")
    plt.ylabel("Predicted log10 permeability (mD)")
    plt.title(f"Graph neural network\nR2={r2:.2f}, RMSE={rmse:.2f}, acc={acc:.0f}%")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig("fig_gnn_pred_vs_actual.png", dpi=150)
    print("wrote fig_gnn_pred_vs_actual.png")


if __name__ == "__main__":
    main()
