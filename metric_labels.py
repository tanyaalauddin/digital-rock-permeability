"""Shared plain-English labels for graph metrics, used by make_figures.py and
analyze_metrics.py so every figure is labelled consistently. The scripts keep the
short internal column names; only the DISPLAY text changes."""

LABELS = {
    "n_nodes": "Number of pores",
    "n_edges": "Number of throats",
    "n_components": "Disconnected clusters",
    "n_cycles": "Number of loops",
    "avg_degree": "Avg. connections per pore",
    "density": "Network density",
    "avg_clustering": "Clustering coefficient",
    "max_degree": "Busiest-hub connections",
    "assortativity": "Degree assortativity",
    "giant_frac": "Fraction in largest cluster",
    "avg_shortest_path": "Avg. path length",
    "diameter": "Network diameter",
    "avg_betweenness": "Betweenness centrality",
    "avg_closeness": "Closeness centrality",
    "geom_spread": "Spatial spread of pores",
    "avg_node_size": "Mean pore size",
    "porosity": "Porosity",
    "permeability_mD": "Permeability (mD)",
}


def label(k):
    return LABELS.get(k, k)
