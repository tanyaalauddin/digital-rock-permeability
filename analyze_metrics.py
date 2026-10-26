#!/usr/bin/env python3
"""
analyze_metrics.py  --  Step 3: do graph metrics predict porosity & permeability?

Reads the shared table you build up with:
    python rock_to_graph.py --input rockA ... --append all_metrics.csv --permeability 120
    python rock_to_graph.py --input rockB ... --append all_metrics.csv --permeability 540
    ... (one row per rock) ...

Then, for each target property it finds (porosity, permeability_mD), it:
  - ranks every graph metric by how strongly it correlates with that property
  - saves a correlation table (correlations.csv)
  - saves scatter plots of the strongest metrics vs the property

INSTALL (once):
    pip install pandas matplotlib scipy

RUN:
    python analyze_metrics.py --input all_metrics.csv
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # save plots to files without needing a display
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from metric_labels import label   # shared plain-English labels


# columns that are labels/targets, not predictor metrics
TARGETS = ["porosity", "permeability_mD"]
NON_METRICS = ["name"] + TARGETS


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="all_metrics.csv", help="the shared metrics table")
    ap.add_argument("--top", type=int, default=4, help="how many top metrics to plot per target")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    print(f"[analyze] loaded {len(df)} rocks from {args.input}")
    if len(df) < 3:
        print("[analyze] WARNING: correlations need several rocks to mean anything. "
              "Add more rows (aim for 8+).")

    # which targets are actually present and have real numbers?
    targets = [t for t in TARGETS if t in df.columns and df[t].notna().sum() >= 3]
    if not targets:
        print("[analyze] No target column (porosity / permeability_mD) with enough values. "
              "Re-run rock_to_graph.py with --permeability, or check porosity is present.")
        return

    # metric columns = everything numeric that isn't a label/target
    metric_cols = [c for c in df.columns
                   if c not in NON_METRICS and pd.api.types.is_numeric_dtype(df[c])]

    rows = []
    for target in targets:
        sub = df[df[target].notna()]
        y = sub[target].to_numpy()
        for m in metric_cols:
            x = sub[m].to_numpy()
            if np.all(np.isnan(x)) or np.nanstd(x) == 0:
                continue
            mask = ~np.isnan(x)
            if mask.sum() < 3:
                continue
            pr, pp = pearsonr(x[mask], y[mask])       # straight-line correlation
            sr, sp = spearmanr(x[mask], y[mask])      # rank correlation (robust)
            rows.append({
                "target": target, "metric": m,
                "pearson_r": round(pr, 3), "pearson_p": round(pp, 4),
                "spearman_r": round(sr, 3), "spearman_p": round(sp, 4),
                "abs_r": abs(pr),
            })

    corr = pd.DataFrame(rows).sort_values(["target", "abs_r"], ascending=[True, False])
    corr.drop(columns="abs_r").to_csv("correlations.csv", index=False)
    print("[analyze] wrote correlations.csv")

    # print and plot the strongest metrics for each target
    for target in targets:
        block = corr[corr.target == target].head(args.top)
        print(f"\n=== strongest predictors of {target} ===")
        for _, r in block.iterrows():
            print(f"  {r['metric']:20s}  r={r['pearson_r']:+.3f}  (rank r={r['spearman_r']:+.3f})")

        n = len(block)
        if n == 0:
            continue
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.6))
        if n == 1:
            axes = [axes]
        sub = df[df[target].notna()]
        for ax, (_, r) in zip(axes, block.iterrows()):
            m = r["metric"]
            ax.scatter(sub[m], sub[target], s=40)
            ax.set_xlabel(label(m))
            ax.set_ylabel(label(target))
            ax.set_title(f"{label(m)}\nr = {r['pearson_r']:+.2f}")
        fig.tight_layout()
        out = f"scatter_{target}.png"
        fig.savefig(out, dpi=130)
        print(f"[analyze] wrote {out}")


if __name__ == "__main__":
    main()
