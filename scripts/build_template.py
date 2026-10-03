"""Extract the peptide-HLA class I template backbone from PDB 1HHK.

1HHK is HLA-A*02:01 with a bound 9-mer. Residues 1-182 of its heavy chain are a 100% exact
match, position for position, to the alpha1/alpha2 sequence this dataset supplies for
HLA-A*02:01 -- so every allele's 182-residue groove sequence maps onto this backbone directly,
with no alignment step and no folding.

    python scripts/build_template.py
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np

PDB = Path("data/template/1HHK.pdb")
OUT = Path("data/template/complex_backbone.npz")
BACKBONE = ["N", "CA", "C", "O"]

THREE2ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
    "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# A = HLA heavy chain (alpha1/alpha2/alpha3), B = beta-2 microglobulin, C = the 9-mer peptide.
CHAINS = ["A", "B", "C"]


def parse(path: Path) -> dict[str, dict]:
    per_chain: dict[str, OrderedDict] = {c: OrderedDict() for c in CHAINS}
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM"):
            continue
        chain = line[21]
        if chain not in per_chain:
            continue
        altloc = line[16]
        if altloc not in (" ", "A"):
            continue
        atom = line[12:16].strip()
        if atom not in BACKBONE:
            continue
        resi = int(line[22:26])
        res3 = line[17:20].strip()
        entry = per_chain[chain].setdefault(resi, {"aa": THREE2ONE.get(res3, "X"), "xyz": {}})
        entry["xyz"][atom] = [float(line[30:38]), float(line[38:46]), float(line[46:54])]

    out = {}
    for chain, residues in per_chain.items():
        keep = [(r, d) for r, d in residues.items() if all(a in d["xyz"] for a in BACKBONE)]
        coords = np.array([[d["xyz"][a] for a in BACKBONE] for _, d in keep], dtype=np.float32)
        seq = "".join(d["aa"] for _, d in keep)
        out[chain] = {"coords": coords, "seq": seq, "resi": np.array([r for r, _ in keep])}
        print(f"chain {chain}: {len(seq)} residues with complete backbone")
    return out


def main() -> None:
    parsed = parse(PDB)

    groove = parsed["A"]["seq"][:182]
    peptide = parsed["C"]["seq"]
    print(f"\ngroove (heavy chain 1-182): {groove[:40]}...")
    print(f"peptide: {peptide}  (length {len(peptide)})")
    if len(peptide) != 9:
        raise SystemExit(f"template peptide is {len(peptide)}-mer; this dataset is all 9-mers")

    np.savez_compressed(
        OUT,
        hla_coords=parsed["A"]["coords"],
        hla_seq=parsed["A"]["seq"],
        b2m_coords=parsed["B"]["coords"],
        b2m_seq=parsed["B"]["seq"],
        pep_coords=parsed["C"]["coords"],
        pep_seq=parsed["C"]["seq"],
        source="1HHK",
    )
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
