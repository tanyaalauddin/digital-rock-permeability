#!/usr/bin/env python3
"""
rock_to_graph.py
------------------------------------------------------------------
Starter pipeline for the AGU abstract project:

    Digital rock (3D segmented image)
        -> point cloud of the pore (or solid) phase
        -> Mapper graph (topological data analysis)
        -> geometric + topological graph metrics (networkx)

This is a scaffold, not a finished study. It is written so you can
run it TODAY on a synthetic rock (no download needed) to see the
whole pipeline work end to end, then swap in real digital-rock data
by pointing --input at a real segmented volume.

------------------------------------------------------------------
INSTALL (once):
    pip install numpy scipy scikit-learn networkx kmapper tifffile

RUN on a synthetic test rock (works immediately):
    python rock_to_graph.py --synthetic

RUN on a real segmented volume (.npy, .tif/.tiff, or raw .raw):
    python rock_to_graph.py --input mysample.npy
    python rock_to_graph.py --input sample.raw --shape 256 256 256 --dtype uint8

OUTPUT:
    - metrics_<name>.csv     one row of graph metrics per rock
    - mapper_<name>.html     interactive Mapper graph you can open in a browser
------------------------------------------------------------------
"""

import argparse
import os
import numpy as np


# ----------------------------------------------------------------------
# 1. LOAD / MAKE A SEGMENTED VOLUME
# ----------------------------------------------------------------------
# A "digital rock" here is a 3D array where each voxel is either
# pore (True) or solid (False). Real datasets come pre-segmented,
# or you segment a grayscale micro-CT scan with a threshold.

def make_synthetic_rock(n=64, porosity=0.3, blur=1.5, seed=0):
    """Generate a fake but topologically realistic porous volume.
    Random field -> smooth -> threshold. Good enough to test the pipeline."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    field = rng.random((n, n, n))
    field = gaussian_filter(field, sigma=blur)
    thresh = np.quantile(field, porosity)   # lowest `porosity` fraction = pore
    pore = field < thresh
    return pore


def load_volume(path, shape=None, dtype="uint8"):
    """Load a segmented volume from .npy, .tif/.tiff, or raw binary."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        vol = np.load(path)
    elif ext in (".tif", ".tiff"):
        import tifffile
        vol = tifffile.imread(path)
    elif ext == ".raw":
        if shape is None:
            raise ValueError("--shape is required for raw files, e.g. --shape 256 256 256")
        vol = np.fromfile(path, dtype=np.dtype(dtype)).reshape(shape)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return vol


def to_pore_mask(vol, pore_value=None):
    """Turn an arbitrary volume into a boolean pore mask.
    Many portal datasets store pore=0, solid=1  (or the reverse) -- CHECK YOURS."""
    vol = np.asarray(vol)
    if vol.dtype == bool:
        return vol
    if pore_value is not None:
        return vol == pore_value
    # heuristic: assume the minority phase is the pore space
    vals, counts = np.unique(vol, return_counts=True)
    minority = vals[np.argmin(counts)]
    print(f"[to_pore_mask] guessing pore value = {minority} "
          f"(override with --pore-value if wrong). Values present: {dict(zip(vals.tolist(), counts.tolist()))}")
    return vol == minority


# ----------------------------------------------------------------------
# 2. VOLUME -> POINT CLOUD
# ----------------------------------------------------------------------
# Mapper works on a point cloud. We take the coordinates of the pore
# voxels. Subsample if the rock is large (millions of voxels) so it runs fast.

def center_crop(vol, size):
    """Take a centered size x size x size subcube (speed + fits in memory).
    Real carbonate scans can be 1000^3+; cropping keeps a first run fast."""
    return crop_cube(vol, size, origin=None)


def crop_cube(vol, size, origin=None):
    """Take a size x size x size subcube. If origin (x,y,z) is given, start there;
    otherwise take the centered cube. Cutting cubes from different origins is how you
    turn ONE big scan into MANY samples with different porosity/permeability."""
    if size is None:
        return vol
    slices = []
    for i, dim in enumerate(vol.shape):
        if size >= dim:
            slices.append(slice(0, dim))                     # smaller than crop: keep all
        elif origin is not None:
            start = max(0, min(int(origin[i]), dim - size))  # clamp so we stay in bounds
            slices.append(slice(start, start + size))
        else:
            start = (dim - size) // 2                         # centered
            slices.append(slice(start, start + size))
    return vol[tuple(slices)]


def volume_to_points(pore_mask, max_points=20000, seed=0):
    coords = np.argwhere(pore_mask)          # (N, 3) voxel coordinates
    if len(coords) == 0:
        raise ValueError("No pore voxels found -- check your pore value / threshold.")
    if len(coords) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(coords), size=max_points, replace=False)
        coords = coords[idx]
    return coords.astype(float)


# ----------------------------------------------------------------------
# 3. POINT CLOUD -> MAPPER GRAPH
# ----------------------------------------------------------------------
# The Mapper algorithm (Chung et al. 2024 use exactly this for digital rocks):
#   filter/lens  -> project points to a low-dim value
#   cover        -> overlapping intervals over the lens range
#   cluster      -> DBSCAN within each interval
#   nerve        -> clusters sharing points become connected nodes
# We use a density + first-PCA-axis lens, a common robust choice.

def pore_points_and_radii(pore_mask, max_points=20000, seed=0):
    """Like volume_to_points, but also returns each point's LOCAL PORE RADIUS
    (its distance to the nearest grain, from the distance transform). Narrow throats
    get small values, big pores large ones -- the flow-relevant structure the plain
    coordinate cloud misses."""
    from scipy.ndimage import distance_transform_edt
    coords = np.argwhere(pore_mask)
    if len(coords) == 0:
        raise ValueError("No pore voxels found -- check your pore value / threshold.")
    if len(coords) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(coords), size=max_points, replace=False)
        coords = coords[idx]
    dt = distance_transform_edt(pore_mask)                 # pore radius at every voxel
    ci = coords.astype(int)
    radii = dt[ci[:, 0], ci[:, 1], ci[:, 2]]
    return coords.astype(float), radii


def build_mapper_graph(points, lens_values=None):
    import kmapper as km
    from sklearn.cluster import DBSCAN
    from sklearn.decomposition import PCA

    mapper = km.KeplerMapper(verbose=1)

    # Lens: one PCA axis (geometry) plus a second dimension.
    lens_geo = mapper.fit_transform(points, projection=PCA(n_components=1))
    if lens_values is not None:
        # distance-transform upgrade: use LOCAL PORE RADIUS as the second lens axis,
        # so the graph organises pore space by pore size / throat width (flow-relevant).
        second = np.asarray(lens_values, dtype=float).reshape(-1, 1)
    else:
        second = mapper.fit_transform(points, projection="l2norm")   # distance-to-mean proxy
    lens = np.c_[lens_geo, second]

    graph = mapper.map(
        lens,
        points,
        cover=km.Cover(n_cubes=12, perc_overlap=0.35),
        clusterer=DBSCAN(eps=3.0, min_samples=5),
    )
    return mapper, graph


def mapper_to_networkx(graph, points, point_values=None):
    """Convert kmapper dict to a networkx graph and attach node geometry. If
    point_values (e.g. pore radii) are given, each node's 'size' becomes its MEAN
    pore radius instead of a raw point count -- a physically meaningful size."""
    import networkx as nx
    pv = None if point_values is None else np.asarray(point_values, dtype=float)
    G = nx.Graph()
    for node_id, members in graph["nodes"].items():
        member_pts = points[members]
        size = float(pv[members].mean()) if pv is not None else len(members)
        G.add_node(node_id, size=size, centroid=member_pts.mean(axis=0))
    for a, bs in graph["links"].items():
        for b in bs:
            G.add_edge(a, b)
    return G


# ----------------------------------------------------------------------
# 4. GRAPH METRICS  (geometric + topological)
# ----------------------------------------------------------------------
# These are the numbers you correlate with rock properties, and use to
# compare real vs. generated rocks. networkx gives you all of them.

def graph_metrics(G):
    import networkx as nx
    m = {}
    n, e = G.number_of_nodes(), G.number_of_edges()
    m["n_nodes"] = n
    m["n_edges"] = e

    if n == 0:
        return m

    # --- topological ---
    m["n_components"] = nx.number_connected_components(G)          # Betti-0
    m["n_cycles"] = e - n + m["n_components"]                      # Betti-1 (loops)
    m["avg_degree"] = 2 * e / n
    m["density"] = nx.density(G)
    m["avg_clustering"] = nx.average_clustering(G)
    degs = [d for _, d in G.degree()]
    m["max_degree"] = max(degs)
    try:
        m["assortativity"] = nx.degree_assortativity_coefficient(G)
    except Exception:
        m["assortativity"] = float("nan")

    # path-based metrics on the largest connected component
    giant = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    m["giant_frac"] = giant.number_of_nodes() / n
    if giant.number_of_nodes() > 1:
        m["avg_shortest_path"] = nx.average_shortest_path_length(giant)
        m["diameter"] = nx.diameter(giant)
        m["avg_betweenness"] = np.mean(list(nx.betweenness_centrality(giant).values()))
        m["avg_closeness"] = np.mean(list(nx.closeness_centrality(giant).values()))

    # --- geometric (from node centroids) ---
    cents = np.array([G.nodes[nd]["centroid"] for nd in G.nodes])
    m["geom_spread"] = float(np.linalg.norm(cents.std(axis=0)))
    sizes = np.array([G.nodes[nd]["size"] for nd in G.nodes])
    m["avg_node_size"] = float(sizes.mean())
    return m


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", help="path to segmented volume (.npy/.tif/.raw)")
    p.add_argument("--synthetic", action="store_true", help="use a generated test rock")
    p.add_argument("--shape", nargs=3, type=int, help="dims for raw files, e.g. 256 256 256")
    p.add_argument("--dtype", default="uint8", help="dtype for raw files")
    p.add_argument("--pore-value", type=int, default=None, help="voxel value that means 'pore'")
    p.add_argument("--phase", choices=["pore", "solid"], default="pore")
    p.add_argument("--max-points", type=int, default=20000)
    p.add_argument("--crop", type=int, default=None, help="centered NxNxN subcube, e.g. --crop 256")
    p.add_argument("--append", default=None, help="also append this row to a shared CSV, e.g. --append all_metrics.csv")
    p.add_argument("--permeability", type=float, default=None, help="measured permeability (mD) for this rock, if known")
    p.add_argument("--porosity", type=float, default=0.3, help="porosity for --synthetic rocks (0-1)")
    p.add_argument("--seed", type=int, default=0, help="random seed for --synthetic rocks")
    p.add_argument("--name", default=None, help="label for output files")
    args = p.parse_args()

    if args.synthetic:
        name = args.name or "synthetic"
        vol = make_synthetic_rock(porosity=args.porosity, seed=args.seed)
        pore = vol
    elif args.input:
        name = args.name or os.path.splitext(os.path.basename(args.input))[0]
        vol = load_volume(args.input, shape=args.shape, dtype=args.dtype)
        pore = to_pore_mask(vol, pore_value=args.pore_value)
    else:
        p.error("give --synthetic or --input <file>")

    if args.crop:
        pore = center_crop(pore, args.crop)
        print(f"[main] cropped to {pore.shape}")

    if args.phase == "solid":
        pore = ~pore

    porosity = pore.mean()
    print(f"[main] volume shape={pore.shape}  porosity(phase)={porosity:.3f}")

    points = volume_to_points(pore, max_points=args.max_points)
    print(f"[main] point cloud: {len(points)} points")

    mapper, graph = build_mapper_graph(points)

    # save interactive Mapper visualization
    html_path = f"mapper_{name}.html"
    mapper.visualize(graph, path_html=html_path, title=f"Mapper graph: {name}")
    print(f"[main] wrote {html_path}")

    import networkx as nx
    G = mapper_to_networkx(graph, points)
    metrics = graph_metrics(G)
    head = {"name": name, "porosity": round(float(porosity), 4)}
    if args.permeability is not None:
        head["permeability_mD"] = args.permeability   # measured ground truth, for step 3
    metrics = {**head, **metrics}

    # write one-row CSV for this rock
    import csv
    csv_path = f"metrics_{name}.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        w.writeheader()
        w.writerow(metrics)
    print(f"[main] wrote {csv_path}")

    # optionally append to a shared table so many rocks accumulate in one file
    if args.append:
        new_file = not os.path.exists(args.append)
        with open(args.append, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            if new_file:
                w.writeheader()
            w.writerow(metrics)
        print(f"[main] appended row to {args.append}")

    print("\n=== GRAPH METRICS ===")
    for k, v in metrics.items():
        print(f"{k:20s}: {v}")


if __name__ == "__main__":
    main()
