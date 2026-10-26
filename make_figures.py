#!/usr/bin/env python3
"""
make_figures.py  --  publication-style figures for the abstract/poster.

Reads the two shared tables and produces several PNGs comparing the datasets
(Ketton vs Estaillades) and the two methods (OpenPNM vs Mapper).

INSTALL (once):  pip install matplotlib numpy
RUN:             python make_figures.py
  (defaults to all_rocks_openpnm.csv / all_rocks_mapper.csv; override with
   --openpnm and --mapper, e.g. --openpnm ketton_openpnm.csv)

FIGURES PRODUCED:
  fig1_perm_vs_porosity.png     permeability vs porosity, colored by rock
  fig2_openpnm_predictors.png   top OpenPNM metrics vs log-permeability
  fig3_mapper_predictors.png    top Mapper metrics vs log-permeability
  fig4_method_comparison.png    |correlation| of each metric with log-perm: OpenPNM vs Mapper
"""

import argparse, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

NON_METRICS = ("name", "porosity", "permeability_mD")

from metric_labels import label   # shared plain-English labels


def load(path):
    rows = list(csv.DictReader(open(path)))
    cols = {k: [] for k in rows[0]}
    for r in rows:
        for k, v in r.items():
            cols[k].append(v)
    return cols, len(rows)


def rock_of(name):
    """'ketton_12' -> 'ketton'."""
    return name.rsplit("_", 1)[0]


def num(c, k):
    return np.array([float(v) for v in c[k]])


def pearson(x, y):
    m = np.isfinite(x) & np.isfinite(y)          # ignore any nan rows pairwise
    if m.sum() < 3 or np.std(x[m]) == 0:
        return np.nan
    return float(np.corrcoef(x[m], y[m])[0, 1])


def logperm(c):
    K = num(c, "permeability_mD")
    K = np.where(K > 0, K, np.nan)   # drop 0/failed permeabilities instead of flooring
    return np.log10(K)


def color_map(names):
    rocks = sorted(set(rock_of(n) for n in names))
    palette = plt.cm.tab10(np.linspace(0, 1, max(3, len(rocks))))
    return {r: palette[i] for i, r in enumerate(rocks)}, rocks


# ----- Figure 1: permeability vs porosity, colored by rock -----
def fig_perm_vs_porosity(c, out="fig1_perm_vs_porosity.png"):
    names = c["name"]; cmap, rocks = color_map(names)
    por = num(c, "porosity"); K = num(c, "permeability_mD")
    plt.figure(figsize=(6, 4.5))
    for r in rocks:
        m = [i for i, n in enumerate(names) if rock_of(n) == r]
        plt.scatter(por[m], K[m], s=45, color=cmap[r], label=r, edgecolor="k", linewidth=0.4)
    plt.yscale("log")
    plt.xlabel("Porosity"); plt.ylabel("Simulated permeability (mD, log scale)")
    plt.title("Permeability vs porosity by carbonate")
    plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print("wrote", out)


# ----- top-N metric scatters vs log-permeability -----
def fig_predictors(c, title, out, n_top=3, metrics_list=None):
    names = c["name"]; cmap, rocks = color_map(names)
    y = logperm(c)
    if metrics_list is not None:
        # show a fixed set of metrics (so two figures can be compared directly)
        top = [m for m in metrics_list if m in c]
    else:
        metrics = [k for k in c if k not in NON_METRICS]
        scored = [(m, pearson(num(c, m), y)) for m in metrics]
        scored = [(m, r) for m, r in scored if not np.isnan(r)]
        scored.sort(key=lambda t: -abs(t[1]))
        top = [m for m, _ in scored[:n_top]]
    if not top:
        print("skip", out, "(no valid metrics)"); return
    fig, axes = plt.subplots(1, len(top), figsize=(4.2 * len(top), 3.8))
    if len(top) == 1:
        axes = [axes]
    for ax, mname in zip(axes, top):
        x = num(c, mname); r = pearson(x, y)
        for rk in rocks:
            idx = [i for i, n in enumerate(names) if rock_of(n) == rk]
            ax.scatter(x[idx], y[idx], s=35, color=cmap[rk], label=rk, edgecolor="k", linewidth=0.3)
        ax.set_xlabel(label(mname)); ax.set_ylabel("log10 permeability (mD)")
        ax.set_title(f"{label(mname)}\nr = {r:+.2f}")
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(); fig.savefig(out, dpi=150); plt.close(fig)
    print("wrote", out)


# ----- Figure 4: method comparison bar chart -----
def fig_method_comparison(co, cm, out="fig4_method_comparison.png"):
    yo, ym = logperm(co), logperm(cm)
    shared = [k for k in co if k not in NON_METRICS and k in cm]
    ro = {k: abs(pearson(num(co, k), yo)) for k in shared}
    rm = {k: abs(pearson(num(cm, k), ym)) for k in shared}
    order = sorted(shared, key=lambda k: -max(ro[k], rm[k]))
    x = np.arange(len(order)); w = 0.4
    plt.figure(figsize=(max(7, len(order) * 0.7), 4.5))
    plt.bar(x - w/2, [ro[k] for k in order], w, label="OpenPNM (pore network)")
    plt.bar(x + w/2, [rm[k] for k in order], w, label="Mapper (TDA)")
    plt.xticks(x, [label(k) for k in order], rotation=45, ha="right", fontsize=8)
    plt.ylabel("|correlation| with log-permeability")
    plt.title("Which metrics predict permeability: pore network vs TDA")
    plt.legend(); plt.tight_layout(); plt.savefig(out, dpi=150); plt.close()
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--openpnm", default="all_rocks_openpnm.csv")
    ap.add_argument("--mapper", default="all_rocks_mapper.csv")
    args = ap.parse_args()

    co, no = load(args.openpnm)
    cm, nm = load(args.mapper)
    print(f"OpenPNM table: {no} rocks | Mapper table: {nm} rocks")

    # same three metrics in both figures, so OpenPNM vs Mapper compare directly
    FEATURED = ["avg_closeness", "avg_shortest_path", "n_cycles"]
    fig_perm_vs_porosity(co)
    fig_predictors(co, "OpenPNM (pore network) vs permeability", "fig2_openpnm_predictors.png",
                   metrics_list=FEATURED)
    fig_predictors(cm, "Mapper (TDA) vs permeability", "fig3_mapper_predictors.png",
                   metrics_list=FEATURED)
    fig_method_comparison(co, cm)
    print("\nDONE. Four PNGs written.")


if __name__ == "__main__":
    main()
