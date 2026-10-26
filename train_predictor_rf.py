#!/usr/bin/env python3
"""
train_predictor_rf.py  --  random-forest version of the permeability predictor.

Same idea as train_predictor.py, but uses a nonlinear Random Forest instead of ridge.
Compares OpenPNM vs Mapper (and the two combined) on the SAME subvolumes, with
cross-validation, and prints which metrics the forest relies on (feature importance).

INSTALL (once, in your venv):
    pip install scikit-learn
RUN:
    python train_predictor_rf.py

Outputs: metrics printed to screen + fig6_rf_pred_vs_actual.png
"""

import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RepeatedKFold
from metric_labels import label

FEATURES = ["avg_node_size", "geom_spread", "avg_closeness",
            "avg_shortest_path", "avg_degree", "avg_clustering",
            "n_cycles", "diameter", "avg_betweenness", "density"]
ACC_TOL = 0.5   # log10 units (~ factor of 3)


def load(path):
    return {r["name"]: r for r in csv.DictReader(open(path))}


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


def cv_rf(X, y, repeats=10, k=5, seed=0):
    """Repeated k-fold CV with a Random Forest. Returns metrics, pooled predictions,
    and mean feature importances."""
    rkf = RepeatedKFold(n_splits=k, n_repeats=repeats, random_state=seed)
    preds = np.zeros((repeats, len(y)))
    rep_counts = np.zeros(len(y), dtype=int)
    importances = []
    r = 0
    per_rep_pred = np.full(len(y), np.nan)
    fold_i = 0
    r2s, rmses, accs = [], [], []
    # iterate manually so we can pool predictions per repeat
    splits = list(rkf.split(X))
    folds_per_rep = k
    pred = np.full(len(y), np.nan)
    for i, (train, test) in enumerate(splits):
        rf = RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                   random_state=seed + i, n_jobs=-1)
        rf.fit(X[train], y[train])
        pred[test] = rf.predict(X[test])
        importances.append(rf.feature_importances_)
        if (i + 1) % folds_per_rep == 0:          # one full pass over the data done
            r2s.append(1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2))
            rmses.append(np.sqrt(np.mean((y - pred) ** 2)))
            accs.append(100 * np.mean(np.abs(y - pred) <= ACC_TOL))
            last_pred = pred.copy()
            pred = np.full(len(y), np.nan)
    return (np.mean(r2s), np.mean(rmses), np.mean(accs)), last_pred, np.mean(importances, 0)


def rock_of(n):
    return n.rsplit("_", 1)[0]


def main():
    op = load("all_rocks_openpnm.csv")
    mp = load("all_rocks_mapper.csv")
    common = [n for n in op if n in mp]

    Xo, yo, ko = build_xy(op, common)
    Xm, ym, km = build_xy(mp, common)
    both = [n for n in ko if n in km]

    def subset(X, y, keep):
        pos = {n: i for i, n in enumerate(keep)}
        i = [pos[n] for n in both]
        return X[i], y[i]
    Xo, yo = subset(Xo, yo, ko)
    Xm, ym = subset(Xm, ym, km)
    Xc = np.hstack([Xo, Xm])
    print(f"[data] {len(both)} cubes "
          f"({sum(rock_of(n)=='ketton' for n in both)} ketton, "
          f"{sum(rock_of(n)=='estaillades' for n in both)} estaillades)")

    results = {}
    for name, X, y in [("OpenPNM", Xo, yo), ("Mapper", Xm, ym), ("Combined", Xc, yo)]:
        (r2, rmse, acc), pred, imp = cv_rf(X, y)
        results[name] = (r2, rmse, acc, pred, imp)
        print(f"\n=== {name} (Random Forest, 5-fold CV) ===")
        print(f"  R2={r2:.2f}  RMSE={rmse:.2f} log10-mD  accuracy(within x3)={acc:.0f}%")
        feats = FEATURES if name != "Combined" else [f"OP:{f}" for f in FEATURES] + [f"MAP:{f}" for f in FEATURES]
        top = np.argsort(imp)[::-1][:5]
        print("  top features:", ", ".join(f"{feats[t]} ({imp[t]:.2f})" for t in top))

    # predicted vs actual for OpenPNM and Mapper
    cmap = {"ketton": "#8c564b", "estaillades": "#1f77b4"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, name in zip(axes, ["OpenPNM", "Mapper"]):
        r2, rmse, acc, pred, imp = results[name]
        yv = yo if name != "Mapper" else ym
        for rk in ("ketton", "estaillades"):
            m = [i for i, n in enumerate(both) if rock_of(n) == rk]
            ax.scatter(yv[m], pred[m], s=45, color=cmap[rk], label=rk, edgecolor="k", linewidth=0.4)
        lo, hi = min(yv.min(), pred.min()) - 0.3, max(yv.max(), pred.max()) + 0.3
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="perfect prediction")
        ax.set_xlabel("Actual log10 permeability (mD)")
        ax.set_ylabel("Predicted log10 permeability (mD)")
        ax.set_title(f"{name} (Random Forest)\nR2={r2:.2f}, RMSE={rmse:.2f}, acc={acc:.0f}%")
        ax.legend(fontsize=8)
    fig.suptitle("Random-forest predicted vs actual permeability (5-fold CV)")
    fig.tight_layout()
    fig.savefig("fig6_rf_pred_vs_actual.png", dpi=150)
    print("\nwrote fig6_rf_pred_vs_actual.png")


if __name__ == "__main__":
    main()
