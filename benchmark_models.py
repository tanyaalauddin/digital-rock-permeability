#!/usr/bin/env python3
"""
Graph-metric models on the same graphs used by the GNN.

For each pore-network graph in graphs/ (from export_graphs.py), this rebuilds the
network, computes graph metrics with NetworkX, and fits ridge, random forest, and
gradient boosting. It uses the same two cross-validation schemes as train_gnn.py
(random 5-fold and grouped by rock) so the metric models and the GNN can be compared
on identical data and splits.

Install: pip install scikit-learn networkx numpy matplotlib
Usage:   python benchmark_models.py --graphs graphs/
Outputs: a results table, random-forest feature importances, and
         fig_model_benchmark.png.
"""

import argparse, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold, GroupKFold

from rock_to_graph import graph_metrics
from metric_labels import label

ACC_TOL = 0.5


def graph_from_npz(z):
    x = z["x"].astype(float); ei = z["edge_index"].astype(int)
    G = nx.Graph()
    for i in range(len(x)):
        G.add_node(i, size=float(x[i, 0]), centroid=x[i, 2:5])
    for a, b in ei.T:
        G.add_edge(int(a), int(b))
    return G


def load_dataset(graph_dir):
    rows, ys, srcs = [], [], []
    for f in sorted(glob.glob(os.path.join(graph_dir, "*.npz"))):
        z = np.load(f, allow_pickle=True)
        m = graph_metrics(graph_from_npz(z))
        rows.append(m); ys.append(float(z["y"])); srcs.append(str(z["source"]))
    keys = [k for k in rows[0] if all(k in r for r in rows)]
    X = np.array([[float(r[k]) for k in keys] for r in rows])
    # fill nan/inf values (e.g. assortativity) with the column mean
    col_mean = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
    inds = np.where(~np.isfinite(X))
    X[inds] = np.take(col_mean, inds[1])
    # drop columns that are effectively constant. A tiny but nonzero variance
    # blows up under standardization and overflows RidgeCV's internal matmul, so
    # filter on variance relative to the column's magnitude, not an absolute floor.
    scale = np.maximum(np.abs(X.mean(axis=0)), 1e-12)
    keep_cols = (X.std(axis=0) / scale) > 1e-8
    X = X[:, keep_cols]
    keys = [k for k, kc in zip(keys, keep_cols) if kc]
    return X, np.array(ys), np.array(srcs), keys


def oof_predict(model_fn, X, y, splitter):
    pred = np.zeros(len(y))
    # RidgeCV's leave-one-out GCV does matmuls that can underflow/overflow on
    # collinear metric columns. It is harmless (the fitted model is unaffected),
    # so ignore those floating-point warnings rather than print them per fold.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for tr, te in splitter:
            m = model_fn(); m.fit(X[tr], y[tr]); pred[te] = m.predict(X[te])
    return pred


def score(y, pred):
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)
    rmse = np.sqrt(np.mean((y - pred) ** 2))
    acc = 100 * np.mean(np.abs(y - pred) <= ACC_TOL)
    return r2, rmse, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphs", default="graphs/")
    ap.add_argument("--min-logk", type=float, default=-4.0,
                    help="drop samples with log10(K) below this value")
    args = ap.parse_args()

    X, y, srcs, keys = load_dataset(args.graphs)
    keep = y >= args.min_logk
    if keep.sum() < len(y):
        print(f"[bench] dropped {len(y) - int(keep.sum())} samples with log10(K) < {args.min_logk}")
    X, y, srcs = X[keep], y[keep], srcs[keep]
    n_groups = len(set(srcs))
    print(f"[bench] {len(y)} graphs, {X.shape[1]} metrics, {n_groups} rocks: {sorted(set(srcs))}")

    models = {
        "Ridge": lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))),
        "RandomForest": lambda: RandomForestRegressor(n_estimators=400, min_samples_leaf=3, random_state=0, n_jobs=-1),
        "GradientBoosting": lambda: GradientBoostingRegressor(random_state=0),
    }
    splits = {"random 5-fold": list(KFold(5, shuffle=True, random_state=0).split(X))}
    if n_groups >= 2:
        splits["grouped-by-rock"] = list(GroupKFold(min(5, n_groups)).split(X, groups=srcs))

    results = {s: {} for s in splits}
    print(f"\n{'model':16s} {'split':16s}  R2     RMSE   acc")
    for sname, sp in splits.items():
        for mname, mfn in models.items():
            r2, rmse, acc = score(y, oof_predict(mfn, X, y, sp))
            results[sname][mname] = r2
            print(f"{mname:16s} {sname:16s}  {r2:+.2f}  {rmse:.2f}  {acc:.0f}%")

    # random-forest feature importances
    rf = models["RandomForest"](); rf.fit(X, y)
    order = np.argsort(rf.feature_importances_)[::-1][:6]
    print("\nMost informative graph metrics (random forest):")
    for i in order:
        print(f"  {label(keys[i]):24s} {rf.feature_importances_[i]:.3f}")

    # SHAP values: which metrics drive predictions and in which direction
    try:
        import shap
        names = [label(k) for k in keys]
        sv = shap.TreeExplainer(rf).shap_values(X)
        shap.summary_plot(sv, X, feature_names=names, show=False, plot_type="bar")
        plt.tight_layout(); plt.savefig("fig_shap.png", dpi=150, bbox_inches="tight"); plt.close()
        shap.summary_plot(sv, X, feature_names=names, show=False)
        plt.tight_layout(); plt.savefig("fig_shap_beeswarm.png", dpi=150, bbox_inches="tight"); plt.close()
        print("wrote fig_shap.png and fig_shap_beeswarm.png (SHAP interpretability)")
    except Exception as e:
        print(f"SHAP skipped ({e}); install with: pip install shap")

    # comparison bar chart
    plt.figure(figsize=(7, 4.2))
    mnames = list(models); x = np.arange(len(mnames)); w = 0.8 / len(splits)
    for j, s in enumerate(splits):
        plt.bar(x + j * w, [results[s][m] for m in mnames], w, label=s)
    plt.xticks(x + w * (len(splits) - 1) / 2, mnames)
    plt.ylabel("Cross-validated R2"); plt.axhline(0, color="k", lw=0.6)
    plt.title("Graph-metrics models: predicting log-permeability")
    plt.legend(); plt.tight_layout(); plt.savefig("fig_model_benchmark.png", dpi=150)
    print("\nwrote fig_model_benchmark.png")


if __name__ == "__main__":
    main()
