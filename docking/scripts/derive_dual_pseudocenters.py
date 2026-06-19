#!/usr/bin/env python3
"""Derive dual zinc pseudo-centers from a crystal ligand pose.

Example:
    uv run python scripts/derive_dual_pseudocenters.py \
      --template-pdb /path/to/7UP3.pdb \
      --template-chain A \
      --ligand-resname NZ0 \
      --donor-names N4,N3 \
      --receptors-dir /scratch/ivanb/projects/Diplom/ndm_hq_docking/receptors \
      --out-dir /scratch/ivanb/projects/Diplom/ndm_hq_docking/dual_receptors \
      --replace-existing-tz
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Atom:
    record: str
    serial: int
    name: str
    resname: str
    chain: str
    resseq: int
    x: float
    y: float
    z: float
    element: str
    ad_type: str
    line: str

    @property
    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


def parse_atoms(path: Path) -> list[Atom]:
    atoms: list[Atom] = []
    with path.open() as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            try:
                atoms.append(
                    Atom(
                        record=line[:6].strip(),
                        serial=int(line[6:11]),
                        name=line[12:16].strip(),
                        resname=line[17:20].strip(),
                        chain=line[21].strip(),
                        resseq=int(line[22:26]),
                        x=float(line[30:38]),
                        y=float(line[38:46]),
                        z=float(line[46:54]),
                        element=line[76:78].strip() if len(line) >= 78 else "",
                        ad_type=(line[77:80].strip() if len(line) >= 80 else ""),
                        line=line.rstrip("\n"),
                    )
                )
            except ValueError:
                continue
    return atoms


def kabsch(template_points: np.ndarray, target_points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return rotation/translation mapping template_points onto target_points."""
    template_centroid = template_points.mean(axis=0)
    target_centroid = target_points.mean(axis=0)
    x = template_points - template_centroid
    y = target_points - target_centroid
    covariance = x.T @ y
    u, _s, vt = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[2, 2] = np.sign(np.linalg.det(u @ vt))
    rotation = u @ correction @ vt
    translation = target_centroid - template_centroid @ rotation
    aligned = template_points @ rotation + translation
    rmsd = float(np.sqrt(np.mean(np.sum((aligned - target_points) ** 2, axis=1))))
    return rotation, translation, rmsd


def distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def ca_map(atoms: list[Atom], chain: str | None) -> dict[tuple[int, str], np.ndarray]:
    result: dict[tuple[int, str], np.ndarray] = {}
    for atom in atoms:
        if atom.name != "CA":
            continue
        if chain and atom.chain != chain:
            continue
        result[(atom.resseq, atom.name)] = atom.xyz
    return result


def filter_alignment_keys(
    keys: list[tuple[int, str]],
    residues: set[int] | None,
) -> list[tuple[int, str]]:
    if not residues:
        return keys
    return [key for key in keys if key[0] in residues]


def find_donor_atoms(template_atoms: list[Atom], chain: str, ligand_resname: str, donor_names: list[str]) -> list[Atom]:
    ligand_atoms = [
        atom for atom in template_atoms if atom.chain == chain and atom.resname == ligand_resname
    ]
    if not ligand_atoms:
        raise SystemExit(f"No ligand atoms found for {ligand_resname} chain {chain}")

    by_name = {atom.name: atom for atom in ligand_atoms}
    donors = []
    for name in donor_names:
        if name not in by_name:
            available = ", ".join(sorted(by_name))
            raise SystemExit(f"Donor atom {name!r} not found. Available ligand atom names: {available}")
        donors.append(by_name[name])
    return donors


def receptor_zn_atoms(atoms: list[Atom]) -> list[Atom]:
    zincs = [
        atom for atom in atoms
        if atom.ad_type.upper() in {"ZN", "ZN"} or atom.resname.upper() == "ZN" or atom.name.upper() == "ZN"
    ]
    # The ensemble receptors have the catalytic zincs as the last two ZN atoms.
    return zincs[:2]


def pseudo_line(serial: int, name: str, xyz: np.ndarray, charge: float, ad_type: str) -> str:
    return (
        f"HETATM{serial:5d} {name:<4s} ZN Z 999    "
        f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}"
        f"  1.00  0.00    {charge:6.3f} {ad_type:<2s}"
    )


def patch_receptor(
    receptor_path: Path,
    out_path: Path,
    pseudo_points: list[tuple[str, np.ndarray]],
    replace_existing_tz: bool,
    charge: float,
    ad_type: str,
) -> None:
    lines = receptor_path.read_text().splitlines()
    patched = []
    max_serial = 0
    for line in lines:
        if line.startswith(("ATOM", "HETATM")):
            try:
                max_serial = max(max_serial, int(line[6:11]))
            except ValueError:
                pass
            current_type = line.split()[-1] if line.split() else ""
            if replace_existing_tz and current_type.upper() == "TZ":
                continue
        patched.append(line)

    for offset, (name, point) in enumerate(pseudo_points, start=1):
        patched.append(pseudo_line(max_serial + offset, name, point, charge, ad_type))
    out_path.write_text("\n".join(patched) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template-pdb", required=True, type=Path)
    parser.add_argument("--template-chain", default="A")
    parser.add_argument("--target-chain", default="A")
    parser.add_argument(
        "--align-residues",
        help="Optional comma-separated residue numbers for local alignment, e.g. 120,122,124,189,208,250",
    )
    parser.add_argument("--ligand-resname", required=True)
    parser.add_argument("--donor-names", required=True, help="Comma-separated ligand donor atom names, e.g. N4,N3")
    parser.add_argument("--receptors-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--replace-existing-tz", action="store_true")
    parser.add_argument("--pseudo-charge", type=float, default=0.250)
    parser.add_argument("--pseudo-type", default="TZ")
    parser.add_argument("--patch-receptors", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    template_atoms = parse_atoms(args.template_pdb)
    align_residues = (
        {int(item.strip()) for item in args.align_residues.split(",") if item.strip()}
        if args.align_residues
        else None
    )
    donors = find_donor_atoms(
        template_atoms,
        args.template_chain,
        args.ligand_resname,
        [item.strip() for item in args.donor_names.split(",") if item.strip()],
    )
    template_ca = ca_map(template_atoms, args.template_chain)
    if not template_ca:
        raise SystemExit("No template CA atoms found for alignment")

    rows = []
    receptor_paths = sorted(args.receptors_dir.glob("*.pdbqt"))
    if not receptor_paths:
        raise SystemExit(f"No .pdbqt receptors found in {args.receptors_dir}")

    for receptor_path in receptor_paths:
        receptor_atoms = parse_atoms(receptor_path)
        target_ca = ca_map(receptor_atoms, args.target_chain)
        common_keys = filter_alignment_keys(sorted(set(template_ca) & set(target_ca)), align_residues)
        if len(common_keys) < 20:
            if align_residues and len(common_keys) >= 3:
                pass
            else:
                raise SystemExit(f"Too few common CA atoms for {receptor_path.name}: {len(common_keys)}")

        template_points = np.array([template_ca[key] for key in common_keys])
        target_points = np.array([target_ca[key] for key in common_keys])
        rotation, translation, rmsd = kabsch(template_points, target_points)

        transformed = [(f"T{idx}", donor.xyz @ rotation + translation, donor) for idx, donor in enumerate(donors, 1)]
        zincs = receptor_zn_atoms(receptor_atoms)

        for name, point, donor in transformed:
            distances = {
                f"zn{i}_distance": distance(point, zinc.xyz)
                for i, zinc in enumerate(zincs, start=1)
            }
            rows.append(
                {
                    "receptor": receptor_path.name,
                    "alignment_ca": len(common_keys),
                    "alignment_rmsd": round(rmsd, 4),
                    "pseudo_name": name,
                    "template_ligand_atom": donor.name,
                    "x": round(float(point[0]), 4),
                    "y": round(float(point[1]), 4),
                    "z": round(float(point[2]), 4),
                    **{key: round(value, 4) for key, value in distances.items()},
                }
            )

        if args.patch_receptors:
            out_path = args.out_dir / receptor_path.name.replace(".pdbqt", "_dual.pdbqt")
            patch_receptor(
                receptor_path,
                out_path,
                [(name, point) for name, point, _donor in transformed],
                args.replace_existing_tz,
                args.pseudo_charge,
                args.pseudo_type,
            )

    csv_path = args.out_dir / "dual_pseudocenters.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {csv_path}")
    if args.patch_receptors:
        print(f"Wrote patched receptors to {args.out_dir}")


if __name__ == "__main__":
    main()
