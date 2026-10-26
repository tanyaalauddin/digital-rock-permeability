#!/usr/bin/env python3
"""
export_graphs.py  --  save each subvolume's PORE NETWORK as a graph for the GNN.

For every subvolume it extracts the SNOW/OpenPNM pore network, simulates permeability
(the target), and writes a compact .npz holding:
    x           node features [pore_diameter, coordination, x, y, z(normalised)]
    edge_index  (2, E) throat connections, made undirected
    edge_attr   edge features [throat_diameter, throat_length]
    y           log10(permeability in mD)   -- the regression target
    meta        name, source (parent rock), label (real/generated)

These .npz files are the input to train_gnn.py. Export needs only the pore-network
stack (no PyTorch), so it runs in your main environment.

USAGE (mirrors build_dataset.py):
  python export_graphs.py --input "Ketton_1000c_3p00006um/Ketton_1000c_3p00006um.raw" \
      --shape 1000 1000 1000 --pore-value 0 --voxel-size 3.00006e-6 \
      --source ketton --label real --out graphs/

Run it once per rock (different --source) to build up a graph library in graphs/.
"""

import os, argparse, itertools
import numpy as np

from rock_to_graph import to_pore_mask, crop_cube
from pore_network_openpnm import extract_network, simulate_permeability


def grid_origins(shape, size, steps):
    per_axis = []
    for dim in shape:
        if size >= dim:
            per_axis.append([0])
        else:
            per_axis.append([int(round(s)) for s in np.linspace(0, dim - size, steps)])
    return list(itertools.product(*per_axis))


def network_to_graph_arrays(pn):
    """Build node/edge feature arrays from a solved OpenPNM network."""
    coords = np.asarray(pn["pore.coords"], dtype=float)
    diam = np.asarray(pn["pore.diameter"], dtype=float) if "pore.diameter" in pn else np.ones(pn.Np)
    conns = np.asarray(pn["throat.conns"], dtype=int)              # (E, 2)

    # coordination number (degree) per pore
    deg = np.bincount(conns.ravel(), minlength=pn.Np).astype(float)

    # normalise coordinates to [0, 1] per axis so position is scale-free
    span = coords.max(0) - coords.min(0)
    span[span == 0] = 1.0
    cn = (coords - coords.min(0)) / span

    x = np.column_stack([diam, deg, cn])                          # (N, 5)

    # undirected edges: add both directions
    edge_index = np.hstack([conns.T, conns.T[::-1]])              # (2, 2E)

    tdiam = np.asarray(pn["throat.diameter"], dtype=float) if "throat.diameter" in pn else np.ones(len(conns))
    if "throat.length" in pn:
        tlen = np.asarray(pn["throat.length"], dtype=float)
    elif "throat.total_length" in pn:
        tlen = np.asarray(pn["throat.total_length"], dtype=float)
    else:
        tlen = np.ones(len(conns))
    edge_attr_one = np.column_stack([tdiam, tlen])                # (E, 2)
    edge_attr = np.vstack([edge_attr_one, edge_attr_one])         # (2E, 2)

    return x.astype(np.float32), edge_index.astype(np.int64), edge_attr.astype(np.float32)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--shape", nargs=3, type=int, required=True)
    p.add_argument("--dtype", default="uint8")
    p.add_argument("--pore-value", type=int, required=True)
    p.add_argument("--voxel-size", type=float, required=True)
    p.add_argument("--size", type=int, default=200, help="subvolume edge in voxels")
    p.add_argument("--steps", type=int, default=3, help="subvolumes per axis (steps^3 total)")
    p.add_argument("--source", required=True, help="parent-rock tag, e.g. ketton")
    p.add_argument("--label", default="real", choices=["real", "generated"])
    p.add_argument("--out", default="graphs/")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    vol = np.memmap(args.input, dtype=np.dtype(args.dtype), mode="r", shape=tuple(args.shape))
    origins = grid_origins(tuple(args.shape), args.size, args.steps)
    print(f"[export] {args.source}: {len(origins)} subvolumes -> {args.out}")

    saved = 0
    for i, origin in enumerate(origins):
        name = f"{args.source}_{args.label}_{i}"
        try:
            sub = np.array(crop_cube(vol, args.size, origin))
            pore = to_pore_mask(sub, pore_value=args.pore_value)
            pn = extract_network(pore, args.voxel_size)
            K = simulate_permeability(pn, pore.shape, args.voxel_size, axis=0)   # also builds geometry
            if not np.isfinite(K) or K <= 0:
                print(f"  {name}: skipped (K={K})"); continue
            x, edge_index, edge_attr = network_to_graph_arrays(pn)
            np.savez(os.path.join(args.out, name + ".npz"),
                     x=x, edge_index=edge_index, edge_attr=edge_attr,
                     y=np.float32(np.log10(K)), name=name, source=args.source, label=args.label)
            saved += 1
            print(f"  {name}: {x.shape[0]} nodes, {edge_index.shape[1]//2} edges, K={K:.2f} mD")
        except Exception as e:
            print(f"  {name}: FAILED ({e})")

    print(f"\nDONE. {saved} graphs saved to {args.out}. Next: python train_gnn.py --graphs {args.out}")


if __name__ == "__main__":
    main()
