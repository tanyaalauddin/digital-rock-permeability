#!/usr/bin/env python3
"""
discriminate.py  --  can graph metrics tell REAL rocks from GENERATED ones?

Reads the table built by analyze_generated.py (rows labelled 'real' or 'generated')
and trains a cross-validated classifier to separate them. Reports the area under the
ROC curve (AUC): ~0.5 means the two are indistinguishable (a good generator), while
values well above 0.5 mean the metrics expose the fakes. Also lists which metrics
carry the signal, and compares graph metrics against a porosity-only baseline so you
can show the graph metrics catch what porosity alone does not.

INSTALL: pip install scikit-learn pandas numpy
RUN:     python discriminate.py --input real_vs_fake.csv
"""

import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RepeatedStratifiedKFold
from metric_labels import label

NON_FEATURES = ("name", "label", "source", "permeability_mD")


def auc_for(X, y, seed=0):
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=6, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    return scores.mean(), scores.std()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="real_vs_fake.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    df = df.dropna()
    y = (df["label"] == "generated").astype(int).to_numpy()
    n_real = int((y == 0).sum()); n_fake = int((y == 1).sum())
    print(f"[data] {len(df)} volumes: {n_real} real, {n_fake} generated")
    if n_real < 5 or n_fake < 5:
        print("[data] WARNING: need several of each label for a meaningful AUC.")

    metric_cols = [c for c in df.columns
                   if c not in NON_FEATURES and pd.api.types.is_numeric_dtype(df[c])]
    graph_cols = [c for c in metric_cols if c != "porosity"]

    # 1) full graph-metric classifier
    Xg = df[graph_cols].to_numpy()
    auc_g, sd_g = auc_for(Xg, y)
    print(f"\nGraph metrics:      AUC = {auc_g:.2f} +/- {sd_g:.2f}")

    # 2) porosity-only baseline
    if "porosity" in df.columns:
        Xp = df[["porosity"]].to_numpy()
        auc_p, sd_p = auc_for(Xp, y)
        print(f"Porosity only:      AUC = {auc_p:.2f} +/- {sd_p:.2f}")

    # 3) which metrics discriminate most
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=0, n_jobs=-1)
    clf.fit(Xg, y)
    order = np.argsort(clf.feature_importances_)[::-1][:6]
    print("\nMost discriminative graph metrics:")
    for i in order:
        print(f"  {label(graph_cols[i]):24s} importance={clf.feature_importances_[i]:.3f}")

    print("\nInterpretation: AUC near 0.5 = generated rocks look real on these metrics; "
          "higher AUC = the metrics expose them. If graph AUC > porosity AUC, the graph "
          "metrics catch flaws porosity misses.")


if __name__ == "__main__":
    main()
