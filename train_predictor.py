#!/usr/bin/env python3
"""
train_predictor.py  --  do graph metrics actually PREDICT permeability?

Trains a simple, cross-validated linear model that predicts log10(permeability) from
graph metrics, separately for the OpenPNM (pore-network) and Mapper (TDA) descriptors,
using the SAME rocks and SAME target for a fair head-to-head comparison.

Reports, for each method:
  - R^2            (fraction of variance explained on held-out data)
  - RMSE           (root-mean-square error, in log10 mD)
  - Accuracy       (% of held-out cubes predicted within a factor of ~3, i.e. 0.5 log units)
and saves a predicted-vs-actual figure (fig5_pred_vs_actual.png).

Uses only numpy + matplotlib (no sklearn needed).
RUN:  python train_predictor.py
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from metric_labels import label

# metrics used as predictors (present in BOTH tables; each from its own graph)
FEATURES = ["avg_node_size", "geom_spread", "avg_closeness",
            "avg_shortest_path", "avg_degree", "avg_clustering"]

ACC_TOL = 0.5   # log10 units ~ factor of 3


def load(path):
    rows = list(csv.DictReader(open(path)))
    return {r["name"]: r for r in rows}


def build_xy(table, names):
    X, y, keep = [], [], []
    for n in names:
        r = table[n]
        try:
            K = float(r["permeability_mD"])
            feat = [float(r[f]) for f in FEATURES]
        except (ValueError, KeyError):
            continue
        if K <= 0 or not np.all(np.isfinite(feat)):
            continue
        X.append(feat); y.append(np.log10(K)); keep.append(n)
    return np.array(X), np.array(y), keep


def _ridge_fit_predict(Xtr, ytr, Xte, alpha):
    """Standardize on train, ridge-regress (intercept unpenalized), predict test."""
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd[sd == 0] = 1
    Xtr = (Xtr - mu) / sd
    Xte = (Xte - mu) / sd
    b0 = ytr.mean()
    p = Xtr.shape[1]
    coef = np.linalg.solve(Xtr.T @ Xtr + alpha * np.eye(p), Xtr.T @ (ytr - b0))
    return b0 + Xte @ coef


def _cv_once(X, y, alpha, repeats, k, seed):
    rng = np.random.default_rng(seed)
    n = len(y)
    r2s, rmses, accs = [], [], []
    last_pred = np.zeros(n)
    for rep in range(repeats):
        idx = rng.permutation(n)
        folds = np.array_split(idx, k)
        pred = np.full(n, np.nan)
        for f in range(k):
            test = folds[f]
            train = np.concatenate([folds[j] for j in range(k) if j != f])
            pred[test] = _ridge_fit_predict(X[train], y[train], X[test], alpha)
        r2s.append(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
        rmses.append(np.sqrt(np.mean((y - pred) ** 2)))
        accs.append(100 * np.mean(np.abs(y - pred) <= ACC_TOL))
        last_pred = pred
    return np.mean(r2s), np.mean(rmses), np.mean(accs), last_pred


def cv_predict(X, y, repeats=30, k=5, seed=0):
    """Ridge-regularized repeated k-fold CV; picks the ridge strength that generalizes
    best (from a small grid) — the standard remedy for few-samples/many-features."""
    best = None
    for alpha in (0.3, 1, 3, 10, 30, 100):
        r2, rmse, acc, pred = _cv_once(X, y, alpha, repeats, k, seed)
        if best is None or r2 > best[0]:
            best = (r2, rmse, acc, pred, alpha)
    r2, rmse, acc, pred, alpha = best
    return (r2, rmse, acc), pred


def rock_of(n):
    return n.rsplit("_", 1)[0]


def main():
    op = load("all_rocks_openpnm.csv")
    mp = load("all_rocks_mapper.csv")
    common = [n for n in op if n in mp]     # same cubes in both packages

    Xo, yo, ko = build_xy(op, common)
    Xm, ym, km = build_xy(mp, common)
    # align to the intersection that is valid in BOTH
    both = [n for n in ko if n in km]
    def subset(X, y, keep):
        pos = {n: i for i, n in enumerate(keep)}
        i = [pos[n] for n in both]
        return X[i], y[i]
    Xo, yo = subset(Xo, yo, ko)
    Xm, ym = subset(Xm, ym, km)
    print(f"[data] {len(both)} cubes used in both models "
          f"({sum(rock_of(n)=='ketton' for n in both)} ketton, "
          f"{sum(rock_of(n)=='estaillades' for n in both)} estaillades)")
    print(f"[data] features: {FEATURES}")

    (r2o, rmseo, acco), predo = cv_predict(Xo, yo)
    (r2m, rmsem, accm), predm = cv_predict(Xm, ym)

    print("\n=== 5-fold cross-validated prediction of log10(permeability) ===")
    print(f"OpenPNM (pore network): R2={r2o:.2f}  RMSE={rmseo:.2f} log10-mD  "
          f"accuracy(within x3)={acco:.0f}%")
    print(f"Mapper  (TDA)         : R2={r2m:.2f}  RMSE={rmsem:.2f} log10-mD  "
          f"accuracy(within x3)={accm:.0f}%")

    # predicted vs actual figure
    cmap = {"ketton": "#8c564b", "estaillades": "#1f77b4"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, (yv, pv, title, r2, rmse, acc) in zip(
        axes,
        [(yo, predo, "OpenPNM (pore network)", r2o, rmseo, acco),
         (ym, predm, "Mapper (TDA)", r2m, rmsem, accm)]):
        for rk in ("ketton", "estaillades"):
            m = [i for i, n in enumerate(both) if rock_of(n) == rk]
            ax.scatter(yv[m], pv[m], s=45, color=cmap[rk], label=rk, edgecolor="k", linewidth=0.4)
        lo, hi = min(yv.min(), pv.min()) - 0.3, max(yv.max(), pv.max()) + 0.3
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect prediction")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Actual log10 permeability (mD)")
        ax.set_ylabel("Predicted log10 permeability (mD)")
        ax.set_title(f"{title}\nR2={r2:.2f}, RMSE={rmse:.2f}, acc={acc:.0f}%")
        ax.legend(fontsize=8)
    fig.suptitle("Predicted vs actual permeability (5-fold cross-validation)")
    fig.tight_layout()
    fig.savefig("fig5_pred_vs_actual.png", dpi=150)
    print("\nwrote fig5_pred_vs_actual.png")


if __name__ == "__main__":
    main()
