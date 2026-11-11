#!/usr/bin/env bash
# Reproduce the full analysis end-to-end.
#
# Prerequisite: the segmented rock volumes (.raw) must already be downloaded and
# decompressed (they are too large to commit; see DATA_library_plan.md for sources and
# the per-rock dimensions / voxel sizes / pore value). This script assumes the four
# Imperial College rocks used in the report are present in the paths below.
#
# Usage:  bash run_all.sh
set -e
source venv/bin/activate 2>/dev/null || true

echo "==================================================================="
echo " 0. Sanity check on a synthetic rock (no data needed)"
echo "==================================================================="
python pore_network_openpnm.py --synthetic

echo "==================================================================="
echo " 1. Export pore-network graphs for each rock into graphs/"
echo "==================================================================="
python export_graphs.py --input "Ketton_1000c_3p00006um/Ketton_1000c_3p00006um.raw"       --shape 1000 1000 1000 --pore-value 0 --voxel-size 3.00006e-6 --source ketton      --label real --out graphs/
python export_graphs.py --input "Estaillades_1000c_3p31136um.raw"                           --shape 1000 1000 1000 --pore-value 0 --voxel-size 3.31136e-6 --source estaillades --label real --out graphs/
python export_graphs.py --input "Bentheimer_1000c_3p0035um/Bentheimer_1000c_3p0035um.raw"   --shape 1000 1000 1000 --pore-value 0 --voxel-size 3.0035e-6  --source bentheimer  --label real --out graphs/
python export_graphs.py --input "Doddington_1000c_2p6929um/Doddington_1000c_2p6929um.raw"   --shape 1000 1000 1000 --pore-value 0 --voxel-size 2.6929e-6  --source doddington  --label real --out graphs/

echo "==================================================================="
echo " 2. Scalar-metric tables + correlation/figures (per rock)"
echo "==================================================================="
# (optional; builds the CSV tables and correlation figures used in earlier sections)
# python build_dataset.py --input ... --prefix ...   # one per rock, see DATA_library_plan.md
python make_figures.py || echo "make_figures needs all_rocks_*.csv; skip if not built"

echo "==================================================================="
echo " 3. Benchmark graph-metric models (+ SHAP) under both CV schemes"
echo "==================================================================="
python benchmark_models.py --graphs graphs/

echo "==================================================================="
echo " 4. Train the graph neural network (random + grouped-by-rock CV)"
echo "==================================================================="
python train_gnn.py --graphs graphs/
python train_gnn.py --graphs graphs/ --no-global   # ablation

echo "==================================================================="
echo " DONE. Key outputs: fig_benchmark_all.png, fig_shap.png,"
echo "       fig_gnn_pred_vs_actual.png, and the printed metric tables."
echo "==================================================================="
