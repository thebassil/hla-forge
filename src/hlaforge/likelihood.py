"""Zero-shot PLM scoring of a peptide in the context of its HLA groove.

Embeddings are only one way to use a protein language model. A masked language model also
assigns a probability to every residue, so we can ask: given the HLA pseudosequence as context,
how plausible does the model find this peptide? And crucially, how much does adding the HLA
context *change* that plausibility? That delta is a zero-shot interaction signal that requires
no training labels at all, which makes it a fair thing to report alongside supervised results.

Two scoring modes:
  wt      one forward pass per sequence; read off the log-prob of the observed residue.
          Fast (one pass per row), the standard "wild-type marginal".
  masked  mask each peptide position in turn and read its log-prob. L times more compute,
          generally the stronger signal.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from .embeddings import MODELS, _device

CACHE_DIR = Path("artifacts/embeddings")
LINKER = "GGGGSGGGGS"


def _cache_path(tag: str, keys: list[str]) -> Path:
    digest = hashlib.sha1("\n".join(keys).encode()).hexdigest()[:16]
    return CACHE_DIR / f"loglik__{tag}__{digest}.npz"


def _peptide_logprobs(
    seqs: list[str],
    peptide_len: int,
    model: str,
    mode: str,
    batch_size: int,
    device: str | None,
) -> np.ndarray:
    """Return (n, peptide_len) log-probabilities of the observed peptide residues.

    The peptide is assumed to occupy the first `peptide_len` residues of each sequence.
    """
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    dev = _device(device)
    tok = AutoTokenizer.from_pretrained(MODELS[model])
    net = AutoModelForMaskedLM.from_pretrained(MODELS[model]).to(dev).eval()
    mask_id = tok.mask_token_id

    out = np.zeros((len(seqs), peptide_len), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(seqs), batch_size):
            batch = seqs[start : start + batch_size]
            enc = tok(batch, return_tensors="pt", padding=True).to(dev)
            ids = enc["input_ids"]
            # Token 0 is BOS, so peptide residue p sits at token index p + 1.
            targets = ids[:, 1 : peptide_len + 1]

            if mode == "wt":
                logits = net(**enc).logits
                logp = torch.log_softmax(logits[:, 1 : peptide_len + 1], dim=-1)
                picked = logp.gather(2, targets.unsqueeze(-1)).squeeze(-1)
                out[start : start + len(batch)] = picked.float().cpu().numpy()
            elif mode == "masked":
                for pos in range(peptide_len):
                    masked_ids = ids.clone()
                    masked_ids[:, pos + 1] = mask_id
                    logits = net(input_ids=masked_ids, attention_mask=enc["attention_mask"]).logits
                    logp = torch.log_softmax(logits[:, pos + 1], dim=-1)
                    picked = logp.gather(1, targets[:, pos : pos + 1]).squeeze(-1)
                    out[start : start + len(batch), pos] = picked.float().cpu().numpy()
            else:
                raise ValueError(f"unknown mode '{mode}'. Use 'wt' or 'masked'.")
    return out


def score_pairs(
    df: pd.DataFrame,
    model: str = "esm2_t12",
    mode: str = "wt",
    hla_field: str = "hla_pseudoseq",
    linker: str = LINKER,
    batch_size: int = 64,
    device: str | None = None,
    cache: bool = True,
) -> np.ndarray:
    """Zero-shot likelihood features per row.

    Columns: 9 in-context per-position log-probs, their sum/mean/min, the peptide-alone sum,
    and the context delta (in-context minus alone). The delta is the part that actually encodes
    peptide-HLA compatibility rather than generic peptide plausibility.
    """
    if model not in MODELS:
        raise ValueError(f"unknown model '{model}'. Known: {list(MODELS)}")
    peptide_len = int(df["peptide"].str.len().max())

    chimeras = (df["peptide"] + linker + df[hla_field]).tolist()
    peptides = df["peptide"].tolist()

    tag = f"{model}_{mode}"
    path = _cache_path(tag, sorted(set(chimeras)))
    if cache and path.exists():
        data = np.load(path, allow_pickle=False)
        table = dict(zip(data["keys"].tolist(), data["vectors"], strict=True))
        return np.stack([table[c] for c in chimeras]).astype(np.float32)

    uniq_chimeras = sorted(set(chimeras))
    uniq_peptides = sorted(set(peptides))

    ctx = _peptide_logprobs(uniq_chimeras, peptide_len, model, mode, batch_size, device)
    alone = _peptide_logprobs(uniq_peptides, peptide_len, model, mode, batch_size, device)
    alone_sum = dict(zip(uniq_peptides, alone.sum(axis=1), strict=True))

    # Recover the peptide of each unique chimera to pair it with its context-free score.
    chimera_peptide = {c: c[:peptide_len] for c in uniq_chimeras}
    rows = []
    for i, c in enumerate(uniq_chimeras):
        per_pos = ctx[i]
        a = float(alone_sum[chimera_peptide[c]])
        rows.append(
            np.concatenate(
                [
                    per_pos,
                    [per_pos.sum(), per_pos.mean(), per_pos.min(), a, per_pos.sum() - a],
                ]
            )
        )
    mat = np.stack(rows).astype(np.float32)

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, keys=np.array(uniq_chimeras), vectors=mat)

    table = dict(zip(uniq_chimeras, mat, strict=True))
    return np.stack([table[c] for c in chimeras]).astype(np.float32)


def zero_shot_ranking(df: pd.DataFrame, **kwargs) -> np.ndarray:
    """A single untrained score per row: the context delta. Use as a no-training baseline."""
    return score_pairs(df, **kwargs)[:, -1]


FEATURE_NAMES = [f"logp_p{i + 1}" for i in range(9)] + [
    "logp_sum",
    "logp_mean",
    "logp_min",
    "logp_peptide_alone",
    "logp_context_delta",
]
