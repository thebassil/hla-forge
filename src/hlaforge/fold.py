"""Per-allele groove structures, so R4 stops sharing one backbone across 75 alleles.

The inverse-folding reference threads every allele's sequence onto a single crystal template
(1HHK). That is a real approximation: C67S constructs are up to 92% below detection precisely
because their groove is deformed, and a shared backbone cannot represent that. PMGen reports
that fine-tuning ProteinMPNN on pMHC lifts sequence recovery from 0.19 to 0.40, which says the
generic model is being handed structures it is not calibrated for.

This predicts one structure per allele with ESMFold -- 75 folds, not 28,166 -- and swaps the
template backbone for the allele's own. The peptide backbone still comes from the template,
superposed into the predicted groove by Kabsch alignment on the conserved helices, because
ESMFold folds a single chain and will not place a ligand peptide for us.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path("artifacts/structures")
BACKBONE_ATOMS = 4  # N, CA, C, O


def fold_alleles(
    df: pd.DataFrame,
    device: str | None = None,
    chunk_size: int = 128,
    cache: bool = True,
) -> dict[str, np.ndarray]:
    """Predict one alpha1/alpha2 backbone per unique groove sequence.

    Returns {hla_seq: (182, 4, 3) backbone coordinates}.
    """
    sequences = sorted(df["hla_seq"].unique())
    path = CACHE_DIR / f"esmfold_grooves_{len(sequences)}.npz"
    if cache and path.exists():
        data = np.load(path, allow_pickle=False)
        return dict(zip(data["keys"].tolist(), data["coords"], strict=True))

    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained("facebook/esmfold_v1")
    net = EsmForProteinFolding.from_pretrained("facebook/esmfold_v1", low_cpu_mem_usage=True)
    net = net.to(device).eval()
    net.trunk.set_chunk_size(chunk_size)

    coords = []
    with torch.no_grad():
        for i, seq in enumerate(sequences):
            enc = tok([seq], return_tensors="pt", add_special_tokens=False).to(device)
            out = net(**enc)
            # positions: (recycles, batch, residues, atoms, 3) in atom14 ordering,
            # whose first four entries are N, CA, C, O -- the ProteinMPNN backbone.
            pos = out["positions"][-1, 0, :, :BACKBONE_ATOMS, :]
            coords.append(pos.float().cpu().numpy())
            if (i + 1) % 10 == 0:
                print(f"  folded {i + 1}/{len(sequences)}", flush=True)

    mat = np.stack(coords).astype(np.float32)
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, keys=np.array(sequences), coords=mat)
    return dict(zip(sequences, mat, strict=True))


def kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation that best superpose `mobile` onto `target` (both (N, 3))."""
    mc, tc = mobile.mean(0), target.mean(0)
    h = (mobile - mc).T @ (target - tc)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rot, tc - rot @ mc


def place_peptide(
    predicted_groove: np.ndarray, template_groove: np.ndarray, template_peptide: np.ndarray
) -> np.ndarray:
    """Move the template's bound peptide into a predicted groove's frame.

    Superposition uses the CA atoms of the two alpha helices that form the walls of the binding
    site (roughly residues 57-84 and 138-179), which are the conserved part of the fold and the
    part that actually contacts the peptide.
    """
    helix = np.r_[57:85, 138:180]
    rot, trans = kabsch(template_groove[helix, 1, :], predicted_groove[helix, 1, :])
    flat = template_peptide.reshape(-1, 3)
    moved = (rot @ flat.T).T + trans
    return moved.reshape(template_peptide.shape).astype(np.float32)


def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(((a - b) ** 2).sum(axis=-1).mean()))
