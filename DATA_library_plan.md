# Building the rock library (for the GNN and the generative-validation study)

Goal: a diverse set of segmented rock types so the models are tested across pore systems,
not just two carbonates. Aim for roughly 6 to 10 rock types; each 1000-cube scan yields
about 27 subvolumes at 200-cube (steps=3), so ~160 to 270 graphs total, which is a solid
dataset for a GNN and a real-vs-fake classifier.

## Where to get them

- **Digital Porous Media Portal** (digitalporousmedia.org): public, free datasets across
  sandstones, carbonates, and packs. Use the project list and search by rock name.
- **Imperial College London pore-scale collection**: Ketton, Estaillades, Berea,
  Bentheimer, bead packs, and others as segmented volumes with documentation (this is where
  Ketton and Estaillades came from).

Both give segmented volumes; check each dataset page (or its `.mhd` header) for the three
things every download needs: **dimensions**, **voxel size**, and **which value is pore**.

## Target list (adjust to what is actually available)

Diversity is the point: pick a spread of connected and poorly-connected systems.

- Sandstones: Bentheimer, Berea, Fontainebleau, Doddington, Castlegate
- Carbonates: Ketton (have), Estaillades (have), Savonnieres, Mount Gambier, Indiana limestone
- Reference packs: a sphere/bead pack and a sandpack (clean, well-understood end-members)

## Per-rock workflow (same as Ketton/Estaillades)

1. Download the **segmented** volume; decompress if needed (`gunzip file.raw.gz`).
2. Read its `.mhd` (or dataset page) for `DimSize`, `ElementSpacing` (voxel size in um),
   and confirm the pore value by sampling (pore is usually the minority value).
3. Export its pore-network graphs:
   ```
   python export_graphs.py --input <file.raw> --shape <nx ny nz> \
       --pore-value <0 or 1> --voxel-size <metres, e.g. 3.0e-6> \
       --source <rockname> --label real --out graphs/
   ```
4. Optionally also add scalar metrics + permeability to the combined tables:
   ```
   python build_dataset.py --input <file.raw> --shape <nx ny nz> \
       --pore-value <..> --voxel-size <..> --prefix <rockname>
   ```

Repeat for each rock type (different `--source` / `--prefix`), all into the same `graphs/`
folder and the same tables. Then:
```
python train_gnn.py --graphs graphs/          # GNN with leakage-aware CV grouped by rock
python analyze_metrics.py --input all_rocks_openpnm.csv
```

## Tracking table (fill as you go)

| Rock | Type | Source | Dimensions | Voxel (um) | Pore value | Subvolumes | Done |
|---|---|---|---|---|---|---|---|
| Ketton | carbonate | Imperial | 1000^3 | 3.00 | 0 | 27 | yes |
| Estaillades | carbonate | Imperial | 1000^3 | 3.31 | 0 | ~22 | yes |
| Berea | sandstone | DPM/Imperial | | | | | |
| Bentheimer | sandstone | | | | | | |
| Fontainebleau | sandstone | | | | | | |
| Savonnieres | carbonate | | | | | | |
| Bead pack | pack | | | | | | |

## Notes
- Keep subvolume size consistent across rocks (200-cube) so graphs are comparable; adjust
  `--size` only if a scan is smaller.
- The GNN's group-by-rock cross-validation needs several rock types to be meaningful, so
  gathering more than two is what unlocks that evaluation.
- For the generative study, you will also generate synthetic versions of a few of these
  rocks and export their graphs with `--label generated` (see SETUP_gan.md).
