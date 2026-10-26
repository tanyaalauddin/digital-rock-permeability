#!/usr/bin/env python3
"""
build_dataset.py  --  general version of the batch builder (works on ANY rock).

Cuts a grid of cubes from a big segmented scan and runs BOTH pipelines on each,
appending rows to shared tables. Run it once per rock with a different --prefix to
build a COMBINED multi-rock dataset (e.g. Ketton + Estaillades in one table).

EXAMPLES
  # rock 1 (start fresh):
  python build_dataset.py --input "Ketton_1000c_3p00006um/Ketton_1000c_3p00006um.raw" \
      --shape 1000 1000 1000 --pore-value 0 --voxel-size 3.00006e-6 --prefix ketton --fresh

  # rock 2 (append to the same tables):
  python build_dataset.py --input "Estaillades.../Estaillades....raw" \
      --shape 650 650 650 --pore-value 0 --voxel-size 3.31e-6 --prefix estaillades

  # then:
  python analyze_metrics.py --input all_rocks_openpnm.csv
  python analyze_metrics.py --input all_rocks_mapper.csv

OUTPUT: all_rocks_openpnm.csv  and  all_rocks_mapper.csv
"""

import os, csv, argparse, itertools
import numpy as np

from rock_to_graph import (to_pore_mask, crop_cube, pore_points_and_radii,
                           build_mapper_graph, mapper_to_networkx, graph_metrics)
from pore_network_openpnm import extract_network, simulate_permeability, network_to_metrics


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def grid_origins(shape, size, steps):
    """Evenly spaced cube corners along each axis (non-overlapping if steps*size<=dim)."""
    per_axis = []
    for dim in shape:
        if size >= dim:
            per_axis.append([0])
        else:
            per_axis.append([int(round(s)) for s in np.linspace(0, dim - size, steps)])
    return list(itertools.product(*per_axis))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--shape", nargs=3, type=int, required=True)
    p.add_argument("--dtype", default="uint8")
    p.add_argument("--pore-value", type=int, required=True)
    p.add_argument("--voxel-size", type=float, required=True, help="metres per voxel, e.g. 3.31e-6")
    p.add_argument("--size", type=int, default=200, help="cube edge in voxels")
    p.add_argument("--steps", type=int, default=3, help="cubes per axis (steps^3 total)")
    p.add_argument("--prefix", required=True, help="name prefix for this rock, e.g. estaillades")
    p.add_argument("--openpnm-out", default="all_rocks_openpnm.csv")
    p.add_argument("--mapper-out", default="all_rocks_mapper.csv")
    p.add_argument("--fresh", action="store_true", help="wipe the output tables before starting")
    args = p.parse_args()

    if args.fresh:
        for f in (args.openpnm_out, args.mapper_out):
            if os.path.exists(f):
                os.remove(f)

    vol = np.memmap(args.input, dtype=np.dtype(args.dtype), mode="r", shape=tuple(args.shape))
    origins = grid_origins(tuple(args.shape), args.size, args.steps)
    print(f"[build] {args.prefix}: {len(origins)} cubes of size {args.size}")

    for i, origin in enumerate(origins):
        name = f"{args.prefix}_{i}"
        sub = np.array(crop_cube(vol, args.size, origin))
        pore = to_pore_mask(sub, pore_value=args.pore_value)
        porosity = float(pore.mean())
        print(f"\n==== {name} origin={origin} porosity={porosity:.3f} ====")

        try:
            pn = extract_network(pore, args.voxel_size)
            K = simulate_permeability(pn, pore.shape, args.voxel_size, axis=0)
            append_row(args.openpnm_out, {"name": name, "porosity": round(porosity, 4),
                                          "permeability_mD": round(K, 3), **network_to_metrics(pn)})
            print(f"  [openpnm] K={K:.2f} mD")
        except Exception as e:
            print(f"  [openpnm] FAILED: {e}"); K = float("nan")

        try:
            pts, radii = pore_points_and_radii(pore, max_points=20000)   # distance-transform upgrade
            _, graph = build_mapper_graph(pts, lens_values=radii)
            append_row(args.mapper_out, {"name": name, "porosity": round(porosity, 4),
                                         "permeability_mD": round(K, 3),
                                         **graph_metrics(mapper_to_networkx(graph, pts, point_values=radii))})
            print(f"  [mapper] done")
        except Exception as e:
            print(f"  [mapper] FAILED: {e}")

    print(f"\nDONE. Tables: {args.openpnm_out}, {args.mapper_out}")


if __name__ == "__main__":
    main()
