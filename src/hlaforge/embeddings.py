"""Frozen protein-language-model embeddings with an on-disk cache.

Only unique sequences are ever passed through the model (5.6k peptides, 75 HLA sequences in the
full dataset), so a complete embedding pass is cheap and every downstream experiment reads from
the cache instead of re-running the PLM.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path("artifacts/embeddings")

MODELS = {
    "esm2_t6": "facebook/esm2_t6_8M_UR50D",
    "esm2_t12": "facebook/esm2_t12_35M_UR50D",
    "esm2_t30": "facebook/esm2_t30_150M_UR50D",
    "esm2_t33": "facebook/esm2_t33_650M_UR50D",
}

POOLINGS = ["mean", "cls", "flatten", "max"]


def _device(prefer: str | None = None) -> str:
    import torch

    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _cache_path(model: str, pooling: str, seqs: list[str]) -> Path:
    digest = hashlib.sha1(("\n".join(seqs)).encode()).hexdigest()[:16]
    return CACHE_DIR / f"{model}__{pooling}__{digest}.npz"


def embed_sequences(
    seqs: list[str],
    model: str = "esm2_t12",
    pooling: str = "mean",
    batch_size: int = 64,
    device: str | None = None,
    cache: bool = True,
    layer: int = -1,
) -> dict[str, np.ndarray]:
    """Embed a list of unique sequences, returning {sequence: vector}. Cached on disk."""
    if model not in MODELS:
        raise ValueError(f"unknown model '{model}'. Known: {list(MODELS)}")
    if pooling not in POOLINGS:
        raise ValueError(f"unknown pooling '{pooling}'. Known: {POOLINGS}")

    uniq = sorted(set(seqs))
    path = _cache_path(f"{model}_L{layer}", pooling, uniq)
    if cache and path.exists():
        data = np.load(path, allow_pickle=False)
        return dict(zip(data["keys"].tolist(), data["vectors"], strict=True))

    import torch
    from transformers import AutoModel, AutoTokenizer

    dev = _device(device)
    tok = AutoTokenizer.from_pretrained(MODELS[model])
    net = AutoModel.from_pretrained(MODELS[model], output_hidden_states=True).to(dev).eval()

    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(uniq), batch_size):
            batch = uniq[start : start + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True).to(dev)
            out = net(**enc)
            hidden = out.hidden_states[layer]  # (B, T, D)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            # Drop BOS/EOS so pooling covers residues only.
            residue_mask = mask.clone()
            residue_mask[:, 0] = 0
            lengths = enc["attention_mask"].sum(1)
            for i, length in enumerate(lengths):
                residue_mask[i, length - 1] = 0

            if pooling == "mean":
                pooled = (hidden * residue_mask).sum(1) / residue_mask.sum(1).clamp(min=1)
            elif pooling == "cls":
                pooled = hidden[:, 0]
            elif pooling == "max":
                pooled = hidden.masked_fill(residue_mask == 0, -1e9).max(dim=1).values
            else:  # flatten: keep per-residue vectors (fixed-length inputs only)
                lens = residue_mask.sum(1).squeeze(-1)
                if len(torch.unique(lens)) != 1:
                    raise ValueError("pooling='flatten' requires equal-length sequences")
                n_res = int(lens[0].item())
                pooled = hidden[:, 1 : 1 + n_res].reshape(len(batch), -1)
            vectors.append(pooled.float().cpu().numpy())

    mat = np.concatenate(vectors, axis=0).astype(np.float32)
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, keys=np.array(uniq), vectors=mat)
    return dict(zip(uniq, mat, strict=True))


def encode_side_plm(
    df: pd.DataFrame,
    side: str,
    model: str = "esm2_t12",
    pooling: str = "mean",
    hla_field: str = "hla_pseudoseq",
    **kwargs,
) -> np.ndarray:
    """Return the per-row PLM embedding matrix for one side of the pair."""
    column = "peptide" if side == "peptide" else hla_field
    table = embed_sequences(df[column].tolist(), model=model, pooling=pooling, **kwargs)
    return np.stack([table[s] for s in df[column]]).astype(np.float32)


def encode_complex_plm(
    df: pd.DataFrame,
    model: str = "esm2_t12",
    pooling: str = "mean",
    linker: str = "GGGGSGGGGS",
    hla_field: str = "hla_pseudoseq",
    **kwargs,
) -> np.ndarray:
    """Embed peptide and HLA jointly as one chimeric sequence, so attention can mix the two."""
    joined = (df["peptide"] + linker + df[hla_field]).tolist()
    table = embed_sequences(joined, model=model, pooling=pooling, **kwargs)
    return np.stack([table[s] for s in joined]).astype(np.float32)
