#!/usr/bin/env python3
"""
build_ketton_dataset.py  --  turn ONE big Ketton scan into MANY samples.

It cuts several 200^3 cubes from different regions of the 1000^3 Ketton volume, and for
each cube runs BOTH pipelines:
  - OpenPNM  -> permeability + pore-network graph metrics   (-> ketton_openpnm.csv)
  - Mapper   -> topological graph metrics                    (-> ketton_mapper.csv)

Different regions have different pore structure, so you get a spread of porosity and
permeability -- exactly the variety you need for step 3 (correlation).

RUN (from the AGU Abstract folder, with venv active):
    python build_ketton_dataset.py

Then analyse:
    python analyze_metrics.py --input ketton_openpnm.csv
    python analyze_metrics.py --input ketton_mapper.csv
"""

import os, csv, itertools
import numpy as np

# reuse the already-working pieces from the two scripts
from rock_to_graph import to_pore_mask, crop_cube, volume_to_points, build_mapper_graph, mapper_to_networkx, graph_metrics
from pore_network_openpnm import extract_network, simulate_permeability, network_to_metrics

# ---- settings for this dataset ----
RAW = "Ketton_1000c_3p00006um/Ketton_1000c_3p00006um.raw"
SHAPE = (1000, 1000, 1000)
DTYPE = "uint8"
PORE_VALUE = 0            # 0 = pore in this file (confirmed)
VOXEL = 3.00006e-6        # metres per voxel
SIZE = 200               # cube edge length in voxels

# a 3x3x3 grid of corners (origins 0/400/800 on each axis) = 27 non-overlapping cubes.
# More samples -> more trustworthy correlations. ~2-3 minutes to run all 27.
ORIGINS = list(itertools.product([0, 400, 800], repeat=3))


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    # start fresh each run so re-running doesn't pile up duplicate rows
    for old in ("ketton_openpnm.csv", "ketton_mapper.csv"):
        if os.path.exists(old):
            os.remove(old)

    # memmap = look at the giant file on disk without loading all 1 GB into memory
    vol = np.memmap(RAW, dtype=np.uint8, mode="r", shape=SHAPE)

    for i, origin in enumerate(ORIGINS):
        name = f"ketton_{i}"
        sub_raw = np.array(crop_cube(vol, SIZE, origin))     # materialise just this cube
        pore = to_pore_mask(sub_raw, pore_value=PORE_VALUE)
        porosity = float(pore.mean())
        print(f"\n================  {name}  origin={origin}  porosity={porosity:.3f}  ================")

        # ---- OpenPNM route: permeability + metrics ----
        try:
            pn = extract_network(pore, VOXEL)
            K = simulate_permeability(pn, pore.shape, VOXEL, axis=0)
            m = network_to_metrics(pn)
            append_row("ketton_openpnm.csv",
                       {"name": name, "porosity": round(porosity, 4),
                        "permeability_mD": round(K, 3), **m})
            print(f"  [openpnm] K={K:.2f} mD  saved")
        except Exception as e:
            print(f"  [openpnm] FAILED on {name}: {e}")
            K = float("nan")

        # ---- Mapper route: topological metrics (carry K so both tables have the target) ----
        try:
            points = volume_to_points(pore, max_points=20000)
            _, graph = build_mapper_graph(points)
            G = mapper_to_networkx(graph, points)
            m2 = graph_metrics(G)
            append_row("ketton_mapper.csv",
                       {"name": name, "porosity": round(porosity, 4),
                        "permeability_mD": round(K, 3), **m2})
            print(f"  [mapper] {G.number_of_nodes()} nodes  saved")
        except Exception as e:
            print(f"  [mapper] FAILED on {name}: {e}")

    print("\nDONE. Now run:  python analyze_metrics.py --input ketton_openpnm.csv")


if __name__ == "__main__":
    main()
