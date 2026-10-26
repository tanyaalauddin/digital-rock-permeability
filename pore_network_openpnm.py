#!/usr/bin/env python3
"""
pore_network_openpnm.py
------------------------------------------------------------------
The SECOND, physics-based route (runs alongside rock_to_graph.py):

    Digital rock (3D segmented image)
        -> REAL pore network (PoreSpy's SNOW algorithm: pores = nodes, throats = edges)
        -> SIMULATE absolute permeability by pushing fluid through it (OpenPNM)
        -> graph metrics on that same network (networkx)

Why this one matters: it gives you a *permeability value* by simulation, so you don't
have to hunt for lab-measured permeability. That simulated number becomes the "answer
key" (ground truth) you check your graph metrics against.

It reuses the file-loading helpers from rock_to_graph.py so both scripts read data the
same way.

------------------------------------------------------------------
INSTALL (once):
    pip install porespy openpnm

RUN on a synthetic test rock (no download needed):
    python pore_network_openpnm.py --synthetic

RUN on a real segmented volume:
    python pore_network_openpnm.py --input sample.tif --voxel-size 3.8e-6 --pore-value 0 --crop 200
    # add:  --append all_openpnm.csv   to collect many rocks into one table

OUTPUT (printed + saved):
    - permeability in milliDarcy (mD)
    - the same graph metrics as the Mapper script, for comparison
    - metrics_openpnm_<name>.csv   (one row per rock)
------------------------------------------------------------------
"""

import argparse
import os
import numpy as np

# borrow the loaders we already wrote + tested, so both scripts behave identically
from rock_to_graph import make_synthetic_rock, load_volume, to_pore_mask, center_crop, graph_metrics


# ----------------------------------------------------------------------
# 1. SEGMENTED ROCK -> REAL PORE NETWORK  (PoreSpy SNOW)
# ----------------------------------------------------------------------
def extract_network(pore_mask, voxel_size):
    """Run the SNOW algorithm: split the pore space into individual pores and find the
    throats connecting them. Returns an OpenPNM network object."""
    import porespy as ps
    import openpnm as op

    # snow2 wants True = pore space. boundary_width adds a thin layer of boundary pores
    # on each face so we can later push fluid in one side and out the other.
    snow = ps.networks.snow2(pore_mask, voxel_size=voxel_size, boundary_width=3)

    # turn PoreSpy's raw output into a proper OpenPNM network
    pn = op.io.network_from_porespy(snow.network)

    # remove any pores that ended up disconnected from the main network (they break the sim)
    health = op.utils.check_network_health(pn)
    bad = health["disconnected_pores"]
    if len(bad) > 0:
        op.topotools.trim(network=pn, pores=bad)
    print(f"[network] pores={pn.Np}  throats={pn.Nt}")
    return pn


# ----------------------------------------------------------------------
# 2. SIMULATE ABSOLUTE PERMEABILITY  (OpenPNM Stokes flow)
# ----------------------------------------------------------------------
def simulate_permeability(pn, im_shape, voxel_size, axis=0):
    """Push water through the network along one axis and measure the flow, then use
    Darcy's law to back out permeability. Returns permeability in milliDarcy."""
    import openpnm as op

    # give pores/throats physical sizes (spheres for pores, cylinders for throats)
    pn.add_model_collection(op.models.collections.geometry.spheres_and_cylinders)
    pn.regenerate_models()

    # define the fluid (water) and its flow physics
    water = op.phase.Water(network=pn)
    water.add_model_collection(op.models.collections.physics.standard)
    water.regenerate_models()

    # pick the inlet and outlet faces for the chosen flow direction
    face_in = {0: "xmin", 1: "ymin", 2: "zmin"}[axis]
    face_out = {0: "xmax", 1: "ymax", 2: "zmax"}[axis]
    inlet = pn.pores(face_in)
    outlet = pn.pores(face_out)

    # set up and solve steady-state flow: high pressure in, zero pressure out
    P_in, P_out = 1.0, 0.0          # Pascals (a relative 1 Pa drop is fine for absolute K)
    flow = op.algorithms.StokesFlow(network=pn, phase=water)
    flow.set_value_BC(pores=inlet, values=P_in)
    flow.set_value_BC(pores=outlet, values=P_out)
    flow.run()

    # total volumetric flow rate entering the inlet
    Q = float(flow.rate(pores=inlet, mode="group")[0])   # m^3/s
    mu = float(water["pore.viscosity"].mean())           # fluid viscosity, Pa*s

    # sample geometry: L = length along flow, A = cross-section area facing the flow
    dims = np.array(im_shape) * voxel_size               # physical size in metres
    L = dims[axis]
    A = np.prod(np.delete(dims, axis))
    dP = P_in - P_out

    K = Q * mu * L / (A * dP)          # Darcy's law -> permeability in m^2
    K_mD = K / 9.869233e-16            # convert m^2 to milliDarcy
    print(f"[flow] Q={Q:.3e} m^3/s   K={K:.3e} m^2   =  {K_mD:.2f} mD")
    return K_mD


# ----------------------------------------------------------------------
# 3. GRAPH METRICS ON THE PORE NETWORK
# ----------------------------------------------------------------------
def network_to_metrics(pn):
    """Build a networkx graph from the OpenPNM network and reuse the same metrics
    function as the Mapper script, so the two routes are directly comparable."""
    import networkx as nx
    G = nx.Graph()
    coords = pn["pore.coords"]
    diam = pn["pore.diameter"] if "pore.diameter" in pn else np.ones(pn.Np)
    for i in range(pn.Np):
        G.add_node(i, centroid=coords[i], size=float(diam[i]))
    for a, b in pn["throat.conns"]:
        G.add_edge(int(a), int(b))
    return graph_metrics(G)


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", help="segmented volume (.npy/.tif/.raw)")
    p.add_argument("--synthetic", action="store_true", help="use a generated test rock")
    p.add_argument("--shape", nargs=3, type=int, help="dims for raw files, e.g. 256 256 256")
    p.add_argument("--dtype", default="uint8", help="dtype for raw files")
    p.add_argument("--pore-value", type=int, default=None, help="voxel value meaning 'pore'")
    p.add_argument("--crop", type=int, default=None, help="centered NxNxN subcube, e.g. --crop 200")
    p.add_argument("--voxel-size", type=float, default=1e-6, help="metres per voxel (e.g. 3.8e-6)")
    p.add_argument("--append", default=None, help="append the row to a shared CSV")
    p.add_argument("--porosity", type=float, default=0.3, help="porosity for --synthetic rocks (0-1)")
    p.add_argument("--seed", type=int, default=0, help="random seed for --synthetic rocks")
    p.add_argument("--name", default=None, help="label for output files")
    args = p.parse_args()

    # --- get a boolean pore mask (True = pore) ---
    if args.synthetic:
        name = args.name or "synthetic"
        pore = make_synthetic_rock(porosity=args.porosity, seed=args.seed)
    elif args.input:
        name = args.name or os.path.splitext(os.path.basename(args.input))[0]
        vol = load_volume(args.input, shape=args.shape, dtype=args.dtype)
        pore = to_pore_mask(vol, pore_value=args.pore_value)
    else:
        p.error("give --synthetic or --input <file>")

    if args.crop:
        pore = center_crop(pore, args.crop)
        print(f"[main] cropped to {pore.shape}")

    porosity = float(pore.mean())
    print(f"[main] shape={pore.shape}  porosity={porosity:.3f}")

    # --- run the three stages ---
    pn = extract_network(pore, args.voxel_size)
    try:
        K_mD = simulate_permeability(pn, pore.shape, args.voxel_size, axis=0)
    except Exception as ex:
        print(f"[flow] permeability sim failed ({ex}); recording NaN. "
              f"Often a face-label issue -- print pn.labels() to see available faces.")
        K_mD = float("nan")
    metrics = network_to_metrics(pn)

    row = {"name": name, "porosity": round(porosity, 4),
           "permeability_mD": round(K_mD, 3), **metrics}

    # --- save ---
    import csv
    csv_path = f"metrics_openpnm_{name}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader(); w.writerow(row)
    print(f"[main] wrote {csv_path}")

    if args.append:
        new_file = not os.path.exists(args.append)
        with open(args.append, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new_file:
                w.writeheader()
            w.writerow(row)
        print(f"[main] appended row to {args.append}")

    print("\n=== OPENPNM RESULT ===")
    for k, v in row.items():
        print(f"{k:20s}: {v}")


if __name__ == "__main__":
    main()
