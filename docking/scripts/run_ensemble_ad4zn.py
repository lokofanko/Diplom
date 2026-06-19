#!/usr/bin/env python3
"""Run ensemble AutoDock4Zn docking for an input SMILES CSV.

This is a configurable version of the original ndm_hq_docking.py script.

Example:
    uv run python scripts/run_ensemble_ad4zn.py \
      --input-csv /scratch/ivanb/projects/Diplom/binding_core/data/generated/file.csv \
      --smiles-col smiles \
      --output-dir /scratch/ivanb/projects/Diplom/ndm_dual_docking_run01 \
      --receptors-dir /scratch/ivanb/projects/Diplom/ndm_hq_docking/dual_receptors \
      --ad4-bin /home/ivanb/vina_docking/autodock4 \
      --ag4-bin /home/ivanb/vina_docking/autogrid4 \
      --param-file /scratch/ivanb/projects/Diplom/ndm_hq_docking/AD4_parameters.dat \
      --workers 16
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from meeko import MoleculePreparation, PDBQTWriterLegacy
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem


RDLogger.DisableLog("rdApp.*")


def receptor_id(path: Path) -> str:
    stem = path.stem
    for part in reversed(stem.split("_")):
        if part.startswith("c") and part[1:].isdigit():
            return part
    return stem


def patch_parameter_file(src: Path, dst: Path) -> None:
    content = src.read_bytes().decode("utf-8", errors="ignore")
    content = content.replace("\r\n", "\n").replace("\r", "\n")
    if "atom_par TZ" not in content:
        content += (
            "\n# --- AutoDock4Zn Pseudo-Atom Parameters ---\n"
            "atom_par TZ 1.10 250.0 0.000 0.0 0.0 0.0 0 -1 -1 0\n"
            "atom_par SZ 1.10 250.0 0.000 0.0 0.0 0.0 0 -1 -1 0\n"
            "atom_par OZ 1.10 250.0 0.000 0.0 0.0 0.0 0 -1 -1 0\n"
        )
    dst.write_text(content, encoding="utf-8", newline="\n")


def prepare_receptors(receptors_dir: Path, output_dir: Path) -> dict[str, Path]:
    out_dir = output_dir / "receptors"
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Path] = {}
    for src in sorted(receptors_dir.glob("*.pdbqt")):
        rid = receptor_id(src)
        dst = out_dir / src.name
        lines = []
        with src.open() as handle:
            for line in handle:
                if line.startswith(("ATOM", "HETATM")) and ("ZN" in line or "TZ" in line):
                    val = 0.250 if "TZ" in line else 0.000
                    line = line[:70].ljust(70) + f"{val:>6.3f}" + line[76:]
                lines.append(line)
        dst.write_text("".join(lines))
        result[rid] = dst
    if not result:
        raise SystemExit(f"No .pdbqt receptors found in {receptors_dir}")
    return result


def auto_prepare_ligand(smiles: str, output_pdbqt: Path) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False

    targets = [
        ("[S;H1,H2;!$(S=O)]", 0),
        ("C(=O)[O;H1,H2]", 2),
        ("S(=O)(=O)[N;H1,H2]", 3),
        ("c1nnn[nH1]1", 4),
        ("c1nn[nH1]n1", 3),
        ("c1n[nH1]nn1", 2),
        ("c1[nH1]nnn1", 1),
    ]
    for smarts, atom_offset in targets:
        pattern = Chem.MolFromSmarts(smarts)
        if pattern is None:
            continue
        for match in mol.GetSubstructMatches(pattern):
            atom = mol.GetAtomWithIdx(match[atom_offset])
            if atom.GetTotalNumHs() > 0:
                atom.SetFormalCharge(-1)
                atom.SetNumExplicitHs(0)

    try:
        mol.UpdatePropertyCache(strict=False)
        Chem.SanitizeMol(mol)
    except Exception:
        return False

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    if AllChem.EmbedMolecule(mol, params) == -1:
        if AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42) == -1:
            return False
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=300)
    except Exception:
        pass

    try:
        setup = MoleculePreparation().prepare(mol)[0]
        pdbqt_lines = PDBQTWriterLegacy().write_string(setup)[0].splitlines()
    except Exception:
        return False

    final_lines = []
    for line in pdbqt_lines:
        if line.startswith(("ATOM", "HETATM")):
            if " S " in line:
                line = line[:77] + "SA"
            if " O " in line:
                line = line[:77] + "OA"
            if " N " in line:
                line = line[:77] + "NA"
        final_lines.append(line)
    output_pdbqt.write_text("\n".join(final_lines) + "\n")
    return True


def get_pdbqt_metadata(path: Path) -> tuple[list[str], int, np.ndarray]:
    atom_types: set[str] = set()
    root_coords = []
    torsions = 0
    in_root = False
    with path.open() as handle:
        for line in handle:
            if line.startswith("ROOT"):
                in_root = True
            elif line.startswith("ENDROOT"):
                in_root = False
            elif line.startswith("BRANCH"):
                torsions += 1
            elif line.startswith(("ATOM", "HETATM")):
                atom_type = line[77:80].strip() or line.split()[-1]
                atom_types.add(atom_type)
                if in_root:
                    root_coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    about = np.mean(root_coords, axis=0) if root_coords else np.array([0.0, 0.0, 0.0])
    return sorted(atom_types), torsions, about


def receptor_grid_center(path: Path) -> str:
    coords = []
    with path.open() as handle:
        for line in handle:
            if line.startswith(("ATOM", "HETATM")):
                atom_type = line[77:80].strip() or line.split()[-1]
                if atom_type.upper() in {"ZN", "TZ"}:
                    coords.append([float(line[30:38]), float(line[38:46]), float(line[46:54])])
    if not coords:
        raise ValueError(f"No ZN/TZ atoms found in {path}")
    center = np.mean(np.array(coords), axis=0)
    return f"{center[0]:.3f} {center[1]:.3f} {center[2]:.3f}"


def process_ligand(task: tuple[int, str, dict[str, str]]) -> tuple[int, dict[str, float], str]:
    idx, smiles, cfg = task
    output_dir = Path(cfg["output_dir"])
    receptors = {key: Path(value) for key, value in cfg["receptors"].items()}
    ligand_id = f"lig_{idx:05d}"
    work_dir = output_dir / f"tmp_{ligand_id}"
    poses_dir = output_dir / "top_poses"
    work_dir.mkdir(parents=True, exist_ok=True)
    scores = {rid: np.nan for rid in receptors}
    error = ""

    ligand_pdbqt = work_dir / f"{ligand_id}.pdbqt"
    try:
        if not auto_prepare_ligand(smiles, ligand_pdbqt):
            return idx, scores, "ligand_preparation_failed"

        ligand_types, torsions, about = get_pdbqt_metadata(ligand_pdbqt)
        env = os.environ.copy()
        env["AUTODOCK_PARAMETER_FILE"] = cfg["master_param"]

        for rid, receptor_src in receptors.items():
            try:
                base = receptor_src.stem
                receptor_dst = work_dir / receptor_src.name
                shutil.copy(receptor_src, receptor_dst)
                receptor_types, _, _ = get_pdbqt_metadata(receptor_dst)
                for atom_type in ["Zn", "ZN", "TZ"]:
                    if atom_type not in receptor_types:
                        receptor_types.append(atom_type)

                center = receptor_grid_center(receptor_dst)
                gpf_path = work_dir / f"{base}.gpf"
                gpf_lines = [
                    f"parameter_file {cfg['master_param']}",
                    f"npts {cfg['npts']} {cfg['npts']} {cfg['npts']}",
                    f"gridfld {base}.maps.fld",
                    f"spacing {cfg['spacing']}",
                    f"receptor_types {' '.join(sorted(receptor_types))}",
                    f"ligand_types {' '.join(ligand_types)}",
                    f"receptor {base}.pdbqt",
                    f"gridcenter {center}",
                    "smooth 0.5",
                    "nbp_r_eps 2.25 6.0000 12 6 SA Zn",
                    "nbp_r_eps 2.10 4.0000 12 6 OA Zn",
                    "nbp_r_eps 2.00 4.0000 12 6 NA Zn",
                    "nbp_r_eps 0.50 1.0000 12 6 SA TZ",
                    "nbp_r_eps 0.50 1.0000 12 6 OA TZ",
                    "nbp_r_eps 0.50 1.0000 12 6 NA TZ",
                    "nbp_r_eps 1.00 0.0000 12 6 HD Zn",
                ]
                gpf_lines.extend(f"map {base}.{atom_type}.map" for atom_type in ligand_types)
                gpf_lines.extend([f"elecmap {base}.e.map", f"dsolvmap {base}.d.map", "dielectric -0.1465"])
                gpf_path.write_text("\n".join(gpf_lines) + "\n")
                subprocess.run(
                    [cfg["ag4_bin"], "-p", str(gpf_path), "-l", f"{base}.glg"],
                    cwd=work_dir,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    env=env,
                    text=True,
                )

                dpf_path = work_dir / f"dock_{rid}.dpf"
                dlg_path = work_dir / f"dock_{rid}.dlg"
                dpf_lines = [
                    "autodock_parameter_version 4.2",
                    f"parameter_file {cfg['master_param']}",
                    "outlev adt",
                    "intelec",
                    "seed pid time",
                    f"ligand_types {' '.join(ligand_types)}",
                    f"fld {base}.maps.fld",
                ]
                dpf_lines.extend(f"map {base}.{atom_type}.map" for atom_type in ligand_types)
                dpf_lines.extend(
                    [
                        f"elecmap {base}.e.map",
                        f"desolvmap {base}.d.map",
                        f"move {ligand_id}.pdbqt",
                        f"about {about[0]:.4f} {about[1]:.4f} {about[2]:.4f}",
                        "tran0 random",
                        "quat0 random",
                        "dihe0 random",
                        f"torsdof {torsions}",
                        f"ga_pop_size {cfg['ga_pop_size']}",
                        f"ga_num_evals {cfg['ga_num_evals']}",
                        "ga_num_generations 27000",
                        "ga_elitism 1",
                        "ga_mutation_rate 0.02",
                        "ga_crossover_rate 0.8",
                        "ga_window_size 10",
                        "set_ga",
                        "sw_max_its 300",
                        "sw_max_succ 4",
                        "sw_max_fail 4",
                        "sw_rho 1.0",
                        "sw_lb_rho 0.01",
                        "ls_search_freq 0.06",
                        "set_psw1",
                        f"ga_run {cfg['ga_run']}",
                        "analysis",
                    ]
                )
                dpf_path.write_text("\n".join(dpf_lines) + "\n")
                subprocess.run(
                    [cfg["ad4_bin"], "-p", dpf_path.name, "-l", dlg_path.name],
                    cwd=work_dir,
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                    env=env,
                    text=True,
                )

                lines = []
                found_pose = False
                with dlg_path.open() as handle:
                    for line in handle:
                        if "Estimated Free Energy of Binding" in line and np.isnan(scores[rid]):
                            scores[rid] = float(line.split("=")[1].split()[0])
                        if "DOCKED: ATOM" in line or "DOCKED: HETATM" in line:
                            found_pose = True
                            lines.append(line[8:])
                        elif found_pose and "DOCKED: ENDMDL" in line:
                            break
                if lines:
                    (work_dir / f"result_{rid}.pdbqt").write_text("".join(lines))
            except Exception as exc:
                error += f"{rid}:{exc}; "
                continue

        valid_scores = {key: value for key, value in scores.items() if not np.isnan(value)}
        if valid_scores:
            best_rid = min(valid_scores, key=valid_scores.get)
            best_score = valid_scores[best_rid]
            best_pose = work_dir / f"result_{best_rid}.pdbqt"
            if best_pose.exists():
                shutil.copy(best_pose, poses_dir / f"{ligand_id}_score_{best_score:.2f}_{best_rid}.pdbqt")
    finally:
        if cfg["cleanup"] == "1":
            shutil.rmtree(work_dir, ignore_errors=True)

    return idx, scores, error


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--smiles-col", default="smiles")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--receptors-dir", required=True, type=Path)
    parser.add_argument("--ad4-bin", required=True)
    parser.add_argument("--ag4-bin", required=True)
    parser.add_argument("--param-file", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    parser.add_argument("--npts", type=int, default=90)
    parser.add_argument("--spacing", default="0.250")
    parser.add_argument("--ga-num-evals", default="5000000")
    parser.add_argument("--ga-run", default="5")
    parser.add_argument("--ga-pop-size", default="150")
    parser.add_argument("--keep-tmp", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "top_poses").mkdir(exist_ok=True)
    master_param = args.output_dir / "AD4_parameters.dat"
    patch_parameter_file(args.param_file, master_param)
    receptors = prepare_receptors(args.receptors_dir, args.output_dir)

    df = pd.read_csv(args.input_csv)
    if args.smiles_col not in df.columns:
        raise SystemExit(f"Missing SMILES column {args.smiles_col!r}. Columns: {', '.join(df.columns)}")
    if args.limit:
        df = df.head(args.limit).copy()

    for rid in receptors:
        df[f"score_{rid}"] = np.nan
    df["min_score"] = np.nan
    df["docking_error"] = ""

    cfg = {
        "output_dir": str(args.output_dir),
        "receptors": {key: str(value) for key, value in receptors.items()},
        "master_param": str(master_param),
        "ad4_bin": args.ad4_bin,
        "ag4_bin": args.ag4_bin,
        "npts": str(args.npts),
        "spacing": args.spacing,
        "ga_num_evals": args.ga_num_evals,
        "ga_run": args.ga_run,
        "ga_pop_size": args.ga_pop_size,
        "cleanup": "0" if args.keep_tmp else "1",
    }
    tasks = [(idx, str(row[args.smiles_col]), cfg) for idx, row in df.iterrows()]
    final_csv = args.output_dir / f"{args.input_csv.stem}_docked.csv"

    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(process_ligand, task) for task in tasks]
        for future in as_completed(futures):
            idx, scores, error = future.result()
            valid = []
            for rid, score in scores.items():
                df.at[idx, f"score_{rid}"] = score
                if not np.isnan(score):
                    valid.append(score)
            df.at[idx, "min_score"] = min(valid) if valid else np.nan
            df.at[idx, "docking_error"] = error
            completed += 1
            if completed % 10 == 0:
                df.to_csv(final_csv, index=False)
                print(f"Processed {completed}/{len(df)}")

    df.to_csv(final_csv, index=False)
    print(f"Wrote {final_csv}")
    print(f"Top poses: {args.output_dir / 'top_poses'}")


if __name__ == "__main__":
    main()
