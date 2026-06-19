# NDM-1 AD4Zn Ensemble Docking

This folder contains the portable docking utilities and data needed to run the
NDM-1 ensemble AutoDock4Zn workflow on the server.

## Contents

- `scripts/derive_dual_pseudocenters.py` derives two ligand-informed TZ pseudo-centers from a crystal complex.
- `scripts/run_ensemble_ad4zn.py` runs configurable ensemble docking for any input CSV with a SMILES column.
- `templates/7UP3.pdb` is the crystal template used to place the dual pseudo-centers.
- `receptors/` contains the original NDM-1 ensemble PDBQT receptors.
- `dual_receptors_7up3_active_site/` contains receptors patched with two 7UP3-derived TZ pseudo-centers.
- `ad4/AD4_parameters.dat` contains the patched AutoDock4 parameter file with TZ support.

## Environment

From the repository root:

```bash
uv sync
```

## Rebuild Dual Pseudo-Center Receptors

This uses active-site alignment on residues `120,122,124,189,208,250` and maps
the `NZ0` ligand donor atoms `N4,N3` from `7UP3`.

```bash
cd /scratch/ivanb/projects/Diplom

uv run python docking/scripts/derive_dual_pseudocenters.py \
  --template-pdb docking/templates/7UP3.pdb \
  --template-chain A \
  --target-chain A \
  --align-residues 120,122,124,189,208,250 \
  --ligand-resname NZ0 \
  --donor-names N4,N3 \
  --receptors-dir docking/receptors \
  --out-dir docking/dual_receptors_7up3_active_site \
  --replace-existing-tz
```

The coordinate report is written to:

```bash
docking/dual_receptors_7up3_active_site/dual_pseudocenters.csv
```

## Docking Smoke Test

Replace `/path/to/molecules.csv` and `smiles` when the input file is chosen.

```bash
cd /scratch/ivanb/projects/Diplom

uv run python docking/scripts/run_ensemble_ad4zn.py \
  --input-csv /path/to/molecules.csv \
  --smiles-col smiles \
  --output-dir /scratch/ivanb/projects/Diplom/ndm_dual_docking_smoke \
  --receptors-dir docking/dual_receptors_7up3_active_site \
  --ad4-bin /home/ivanb/vina_docking/autodock4 \
  --ag4-bin /home/ivanb/vina_docking/autogrid4 \
  --param-file docking/ad4/AD4_parameters.dat \
  --workers 4 \
  --limit 5 \
  --keep-tmp
```

## Full Docking Run

```bash
cd /scratch/ivanb/projects/Diplom

uv run python docking/scripts/run_ensemble_ad4zn.py \
  --input-csv /path/to/molecules.csv \
  --smiles-col smiles \
  --output-dir /scratch/ivanb/projects/Diplom/ndm_dual_docking_run01 \
  --receptors-dir docking/dual_receptors_7up3_active_site \
  --ad4-bin /home/ivanb/vina_docking/autodock4 \
  --ag4-bin /home/ivanb/vina_docking/autogrid4 \
  --param-file docking/ad4/AD4_parameters.dat \
  --workers 16
```

The output CSV will be written into the chosen `--output-dir`, and best poses
will be saved under `--output-dir/top_poses`.
