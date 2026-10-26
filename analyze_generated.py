#!/usr/bin/env python3
"""
analyze_generated.py  --  run REAL or GENERATED rock volumes through the existing
SNOW -> OpenPNM -> graph-metrics + Stokes-flow pipeline, tagging each with a label so
the rows can later be used to train a real-vs-fake discriminator.

It reuses the same functions the AGU work used, so real and generated rocks are
measured identically.

INSTALL (already have most): pip install porespy openpnm networkx numpy tifffile h5py

USAGE
  # measure a folder of REAL rock subvolumes:
  python analyze_generated.py --input real_berea/ --label real --source berea \
      --pore-value 0 --voxel-size 5.6e-6 --crop 64 --out real_vs_fake.csv

  # measure a folder of GENERATED volumes from the GAN:
  python analyze_generated.py --input gan_berea/ --label generated --source berea_gan \
      --pore-value 0 --voxel-size 5.6e-6 --crop 64 --out real_vs_fake.csv

Then: python discriminate.py --input real_vs_fake.csv

Accepts .tif/.tiff, .npy, .raw (needs --shape), and .h5/.hdf5 volumes.
"""

import os, csv, glob, argparse
import numpy as np

from rock_to_graph import to_pore_mask, crop_cube
from pore_network_openpnm import extract_network, simulate_permeability, network_to_metrics


def load_any(path, shape=None, dtype="uint8"):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        return np.load(path)
    if ext in (".tif", ".tiff"):
        import tifffile
        return tifffile.imread(path)
    if ext in (".h5", ".hdf5"):
        import h5py
        with h5py.File(path, "r") as f:
            key = list(f.keys())[0]            # first dataset in the file
            return np.array(f[key])
    if ext == ".raw":
        if shape is None:
            raise ValueError("--shape is required for .raw files")
        return np.fromfile(path, dtype=np.dtype(dtype)).reshape(shape)
    raise ValueError(f"unsupported file type: {ext}")


def gather(inp):
    if os.path.isdir(inp):
        files = []
        for e in ("*.tif", "*.tiff", "*.npy", "*.raw", "*.h5", "*.hdf5"):
            files += glob.glob(os.path.join(inp, e))
        return sorted(files)
    return [inp]


def append_row(path, row):
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="a volume file OR a folder of them")
    p.add_argument("--label", required=True, choices=["real", "generated"])
    p.add_argument("--source", required=True, help="tag for the rock/model, e.g. berea or berea_gan")
    p.add_argument("--pore-value", type=int, default=0)
    p.add_argument("--voxel-size", type=float, default=1e-6)
    p.add_argument("--crop", type=int, default=None)
    p.add_argument("--shape", nargs=3, type=int, default=None)
    p.add_argument("--dtype", default="uint8")
    p.add_argument("--out", default="real_vs_fake.csv")
    args = p.parse_args()

    files = gather(args.input)
    print(f"[analyze] {len(files)} volumes labelled '{args.label}' (source={args.source})")

    for i, fp in enumerate(files):
        name = f"{args.source}_{args.label}_{i}"
        try:
            vol = load_any(fp, shape=args.shape, dtype=args.dtype)
            pore = to_pore_mask(vol, pore_value=args.pore_value)
            if args.crop:
                pore = crop_cube(pore, args.crop)
            porosity = float(pore.mean())
            pn = extract_network(pore, args.voxel_size)
            K = simulate_permeability(pn, pore.shape, args.voxel_size, axis=0)
            row = {"name": name, "label": args.label, "source": args.source,
                   "porosity": round(porosity, 4), "permeability_mD": round(K, 3),
                   **network_to_metrics(pn)}
            append_row(args.out, row)
            print(f"  {name}: porosity={porosity:.3f} K={K:.2f} mD  -> {args.out}")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    print(f"\nDONE. Rows appended to {args.out}. Next: python discriminate.py --input {args.out}")


if __name__ == "__main__":
    main()
