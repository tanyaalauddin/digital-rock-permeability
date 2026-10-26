# Graph-Based Permeability Prediction and Validation of Digital Rocks

Predicting the permeability of carbonate rocks from 3D micro-CT images by turning the
pore space into a graph, and using graph descriptors and graph neural networks as fast,
interpretable surrogates for expensive flow simulation. The same descriptors are then
used to test whether AI-generated ("synthetic") rocks are physically realistic.

**Author:** Tanya O. Alauddin · **Collaborator:** Rayan Kanfar (Upstream Advanced Research Center, Saudi Aramco)

> Research in progress toward a journal paper. A shortened version was submitted to the
> AGU 2026 Fall Meeting.

---

## Why this project

Permeability controls how fluids (water, hydrocarbons, CO2) move through rock, but it is
slow and expensive to measure or to compute by direct flow simulation. Two rocks with the
same porosity can differ in permeability by orders of magnitude because what matters is
how the pore space is *connected*. Graphs are a natural way to encode that connectivity,
so this project asks two questions:

1. Can graph descriptors of the pore network predict permeability well enough to act as a
   fast surrogate for simulation, and which descriptors carry the signal?
2. Can those same descriptors tell real rocks from AI-generated ones, providing a
   connectivity-aware validity check that goes beyond the porosity and two-point
   statistics used today?

## Method overview

Segmented micro-CT subvolumes are converted to a graph two ways: an explicit pore network
(pores as nodes, throats as edges) extracted with the SNOW algorithm in PoreSpy/OpenPNM,
and a topological graph from the Mapper algorithm. Geometric and topological metrics are
computed with NetworkX. Ground-truth permeability is obtained from a Stokes-flow
simulation in OpenPNM. Models (ridge, random forest, and a graph neural network) are
trained to predict log-permeability under leakage-aware cross-validation.

## Key results so far

On 41 subvolumes from two contrasting carbonates (Ketton, a well-connected oolitic
limestone, and Estaillades, microporous), spanning 0.07 to 255 mD:

- Pore-network graph metrics predict log-permeability with R2 ~ 0.39 (RMSE 0.65 log10-mD;
  65 percent of samples within a factor of three).
- The topological (Mapper) metrics are far less predictive; adding a distance-transform
  (pore-radius) filter lifts them from R2 ~ -0.9 to ~ 0 but they still do not match the
  pore network, showing that explicit pore-throat connectivity is what carries flow.

## Repository structure

```
.
├── rock_to_graph.py         # volume -> Mapper graph -> metrics
├── pore_network_openpnm.py  # volume -> SNOW pore network -> permeability + metrics
├── build_dataset.py         # cut subvolumes, run both pipelines into tables
├── build_ketton_dataset.py  # Ketton-only dataset builder
├── analyze_metrics.py       # correlations of metrics vs porosity/permeability
├── train_predictor.py       # ridge regression predictor (cross-validated)
├── train_predictor_rf.py    # random-forest predictor
├── export_graphs.py         # save pore-network graphs for the GNN
├── train_gnn.py             # graph neural network permeability predictor (in progress)
├── analyze_generated.py     # measure real vs generated rocks
├── discriminate.py          # real-vs-fake classifier (AUC)
├── make_figures.py          # results figures
├── metric_labels.py         # shared plain-English metric names
├── all_rocks_openpnm.csv    # pore-network metrics + simulated permeability per subvolume
├── all_rocks_mapper.csv     # Mapper-graph metrics per subvolume
├── figures/                 # results figures (see below)
├── requirements.txt
└── .gitignore
```

The two CSV tables let you rerun the analysis and models (for example
`python train_predictor_rf.py`) without downloading or processing any micro-CT data.

## Figures

- `fig1_perm_vs_porosity.png`: permeability vs porosity across subvolumes
- `fig2_openpnm_predictors.png`: pore-network metrics vs permeability
- `fig3_mapper_predictors.png`: Mapper-graph metrics vs permeability
- `fig4_method_comparison.png`: correlation of each metric with permeability, pore network vs Mapper
- `fig5_pred_vs_actual.png`: predicted vs actual permeability (ridge)
- `fig6_rf_pred_vs_actual.png`: predicted vs actual permeability (random forest)

## Getting started

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Quick end-to-end test on a synthetic rock (no download needed):

```bash
python rock_to_graph.py --synthetic
python pore_network_openpnm.py --synthetic
```

## Data

Segmented micro-CT volumes are large and are not committed to the repository. They are
openly available from:

- Imperial College London pore-scale collection (Ketton, Estaillades, Berea, and others)
- the Digital Rocks Portal / Digital Porous Media Portal

Download a segmented `.raw`/`.tif`, note its dimensions, voxel size, and pore value, then
point `build_dataset.py` at it.

## Reproducibility

All models use fixed random seeds and report cross-validated metrics with uncertainty.
Cross-validation is grouped by parent rock so that subvolumes from the same scan never
appear in both train and test, which is a strict test of generalization.

## Roadmap

- Graph neural network for direct permeability prediction from the pore network (in progress).
- Expanded rock library across sandstones and carbonates.
- Generative model (GAN, then diffusion) plus a learned real-vs-fake discriminator.
- Benchmarking against standard descriptors (two-point correlation, Minkowski functionals).

## Acknowledgements and tools

Built on OpenPNM and PoreSpy (Gostick et al.), NetworkX, scikit-learn, and PyTorch
Geometric. 
