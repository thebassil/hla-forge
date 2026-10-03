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
    # ProtT5 is an encoder-decoder trained on UniRef50; we use the encoder only. Its tokenizer
    # expects residues separated by spaces, and it has no masked-LM head, so it can serve as an
    # encoder here but not as a scorer.
    "prott5": "Rostlab/prot_t5_xl_uniref50",
    # ESM Cambrian. Shipped through EvolutionaryScale's own `esm` package rather than
    # transformers, so it takes a separate code path below.
    "esmc_300m": "esmc_300m",
    "esmc_600m": "esmc_600m",
}

#: Models that do not load through transformers' AutoModel.
SPECIAL_LOADERS = {"prott5", "esmc_300m", "esmc_600m"}

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

    dev = _device(device)
    if model.startswith("esmc"):
        mat = _embed_esmc(uniq, model, pooling, batch_size, dev)
        if cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, keys=np.array(uniq), vectors=mat)
        return dict(zip(uniq, mat, strict=True))

    if model == "prott5":
        from transformers import T5EncoderModel, T5Tokenizer

        tok = T5Tokenizer.from_pretrained(MODELS[model], do_lower_case=False, legacy=True)
        net = T5EncoderModel.from_pretrained(MODELS[model]).to(dev).eval()
    else:
        from transformers import AutoModel, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(MODELS[model])
        net = AutoModel.from_pretrained(MODELS[model], output_hidden_states=True).to(dev).eval()

    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(uniq), batch_size):
            batch = uniq[start : start + batch_size]
            if model == "prott5":
                # ProtT5 wants whitespace-separated residues and has no BOS token.
                spaced = [" ".join(seq) for seq in batch]
                enc = tok(spaced, return_tensors="pt", padding=True).to(dev)
                hidden = net(**enc).last_hidden_state
            else:
                enc = tok(batch, return_tensors="pt", padding=True).to(dev)
                hidden = net(**enc).hidden_states[layer]  # (B, T, D)
            mask = enc["attention_mask"].unsqueeze(-1).float()
            # Drop the special tokens so pooling covers residues only. ESM wraps sequences in
            # BOS and EOS; ProtT5 appends EOS with no BOS.
            residue_mask = mask.clone()
            if model != "prott5":
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
                offset = 0 if model == "prott5" else 1
                pooled = hidden[:, offset : offset + n_res].reshape(len(batch), -1)
            vectors.append(pooled.float().cpu().numpy())

    mat = np.concatenate(vectors, axis=0).astype(np.float32)
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, keys=np.array(uniq), vectors=mat)
    return dict(zip(uniq, mat, strict=True))


def _embed_esmc(
    uniq: list[str], model: str, pooling: str, batch_size: int, dev: str
) -> np.ndarray:
    """ESM Cambrian, via EvolutionaryScale's own package rather than transformers."""
    import torch
    from esm.models.esmc import ESMC
    from esm.sdk.api import ESMProtein, LogitsConfig

    net = ESMC.from_pretrained(MODELS[model]).to(dev).eval()
    cfg = LogitsConfig(sequence=True, return_embeddings=True)
    vectors = []
    with torch.no_grad():
        for seq in uniq:
            out = net.logits(net.encode(ESMProtein(sequence=seq)), cfg)
            h = out.embeddings[0]          # (T, D) including BOS/EOS
            residues = h[1:-1]
            if pooling == "mean":
                v = residues.mean(0)
            elif pooling == "max":
                v = residues.max(0).values
            elif pooling == "cls":
                v = h[0]
            else:
                v = residues.reshape(-1)
            vectors.append(v.float().cpu().numpy())
    return np.stack(vectors).astype(np.float32)


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
