"""R4: inverse-folding scores for the peptide inside the HLA groove.

This reference asks a question no sequence model asks: given the three-dimensional backbone of
a class I binding groove, how likely is this particular 9-mer to be the sequence sitting in it?
That is much closer to "how long will it stay bound" than anything a protein language model
computes from sequence alone, which is why it is worth a reference system of its own.

Template, not prediction. PDB 1HHK is HLA-A*02:01 with a bound 9-mer, and residues 1-182 of its
heavy chain match this dataset's alpha1/alpha2 sequence exactly, position for position. So every
allele's groove sequence threads onto that backbone with no alignment and no folding step. Each
allele therefore shares one backbone and differs only in the residue identities ProteinMPNN
conditions on -- which is the approximation here, and it is a real one: allele-specific backbone
shifts in the B and F pockets are not modelled. Predicting a backbone per allele is the upgrade,
and it slots in behind the same interface.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MPNN_DIR = Path("external/ProteinMPNN")
TEMPLATE = Path("data/template/complex_backbone.npz")
CACHE_DIR = Path("artifacts/embeddings")

ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"
AA_TO_IDX = {a: i for i, a in enumerate(ALPHABET)}
CHAIN_GAP = 100  # ProteinMPNN's convention for a chain break in residue_idx

WEIGHTS = {
    "v_48_020": "vanilla_model_weights/v_48_020.pt",  # 0.20A backbone noise, the default
    "v_48_002": "vanilla_model_weights/v_48_002.pt",  # 0.02A, sharper
    "soluble_v_48_020": "soluble_model_weights/v_48_020.pt",
}

FEATURE_NAMES = [f"mpnn_p{i + 1}" for i in range(9)] + [
    "mpnn_sum", "mpnn_mean", "mpnn_min", "mpnn_entropy_mean"
]


def _load_mpnn(weights: str, device: str):
    import torch

    if str(MPNN_DIR) not in sys.path:
        sys.path.insert(0, str(MPNN_DIR))
    from protein_mpnn_utils import ProteinMPNN

    ckpt_path = MPNN_DIR / WEIGHTS[weights]
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"{ckpt_path} not found. Run: git clone --depth 1 "
            "https://github.com/dauparas/ProteinMPNN external/ProteinMPNN"
        )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ProteinMPNN(
        ca_only=False, num_letters=21, node_features=128, edge_features=128,
        hidden_dim=128, num_encoder_layers=3, num_decoder_layers=3,
        augment_eps=0.0, k_neighbors=ckpt["num_edges"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(device).eval()


def _template():
    if not TEMPLATE.exists():
        raise FileNotFoundError(
            f"{TEMPLATE} not found. Run: python scripts/build_template.py"
        )
    t = np.load(TEMPLATE, allow_pickle=False)
    return {
        "hla_coords": t["hla_coords"], "hla_seq": str(t["hla_seq"]),
        "b2m_coords": t["b2m_coords"], "b2m_seq": str(t["b2m_seq"]),
        "pep_coords": t["pep_coords"], "pep_seq": str(t["pep_seq"]),
    }


def _groove_in_template_frame(
    predicted: np.ndarray, tmpl: dict, n_groove: int = 182
) -> np.ndarray:
    """Put a predicted alpha1/alpha2 backbone into the template complex's coordinate frame.

    Superposing the prediction onto the template -- rather than the other way round -- lets the
    peptide, alpha3 domain and beta-2 microglobulin stay exactly where the crystal structure put
    them, so the only thing that changes between alleles is the groove itself.
    """
    from .fold import kabsch

    helix = np.r_[57:85, 138:180]
    rot, trans = kabsch(predicted[helix, 1, :], tmpl["hla_coords"][helix, 1, :])
    flat = predicted.reshape(-1, 3)
    moved = ((rot @ flat.T).T + trans).reshape(predicted.shape)
    out = tmpl["hla_coords"].copy()
    out[:n_groove] = moved[:n_groove]
    return out.astype(np.float32)


def _static_tensors(tmpl: dict, device: str, hla_coords: np.ndarray | None = None):
    """Coordinates, chain labels and residue indices for one complex."""
    import torch

    n_hla = len(tmpl["hla_seq"])
    n_b2m = len(tmpl["b2m_seq"])
    n_pep = len(tmpl["pep_seq"])
    coords = np.concatenate(
        [tmpl["hla_coords"] if hla_coords is None else hla_coords,
         tmpl["b2m_coords"], tmpl["pep_coords"]], axis=0
    )

    residue_idx = np.concatenate([
        np.arange(n_hla),
        np.arange(n_b2m) + n_hla + CHAIN_GAP,
        np.arange(n_pep) + n_hla + n_b2m + 2 * CHAIN_GAP,
    ])
    chain_encoding = np.concatenate([
        np.ones(n_hla), np.full(n_b2m, 2.0), np.full(n_pep, 3.0)
    ])
    # Only the peptide is "designed", so it is decoded last, conditioned on the whole groove.
    chain_M = np.concatenate([np.zeros(n_hla + n_b2m), np.ones(n_pep)])

    to = lambda a, dt: torch.tensor(a, dtype=dt, device=device)  # noqa: E731
    return {
        "X": to(coords, torch.float32),
        "residue_idx": to(residue_idx, torch.long),
        "chain_encoding": to(chain_encoding, torch.float32),
        "chain_M": to(chain_M, torch.float32),
        "n_hla": n_hla, "n_b2m": n_b2m, "n_pep": n_pep,
        "slice_pep": slice(n_hla + n_b2m, n_hla + n_b2m + n_pep),
    }


def _encode_rows(df: pd.DataFrame, tmpl: dict) -> np.ndarray:
    """Sequence indices for every row: allele groove + constant tail + b2m + peptide."""
    n_hla = len(tmpl["hla_seq"])
    tail = tmpl["hla_seq"][182:]  # alpha3 is near-invariant and not supplied by the dataset
    b2m = tmpl["b2m_seq"]

    rows = np.empty((len(df), n_hla + len(b2m) + 9), dtype=np.int64)
    for i, (groove, peptide) in enumerate(zip(df["hla_seq"], df["peptide"], strict=True)):
        if len(groove) != 182:
            raise ValueError(f"expected a 182-residue groove, got {len(groove)}")
        full = groove + tail + b2m + peptide
        rows[i] = [AA_TO_IDX.get(c, AA_TO_IDX["X"]) for c in full]
    return rows


def score_structure(
    df: pd.DataFrame,
    weights: str = "v_48_020",
    n_decoding_orders: int = 4,
    batch_size: int = 32,
    device: str | None = None,
    cache: bool = True,
    seed: int = 0,
    per_allele: bool = False,
) -> np.ndarray:
    """Per-row inverse-folding features for the peptide given the groove backbone.

    ProteinMPNN's score depends on the random order in which positions are decoded, so we
    average log-probabilities over several orders rather than trusting one draw.
    """
    keys = (df["hla_seq"] + "|" + df["peptide"]).tolist()
    uniq = sorted(set(keys))
    digest = hashlib.sha1("\n".join(uniq).encode()).hexdigest()[:16]
    tag = "mpnnpa" if per_allele else "mpnn"
    path = CACHE_DIR / f"{tag}__{weights}_o{n_decoding_orders}__{digest}.npz"
    if cache and path.exists():
        data = np.load(path, allow_pickle=False)
        table = dict(zip(data["keys"].tolist(), data["vectors"], strict=True))
        return np.stack([table[k] for k in keys]).astype(np.float32)

    # Imported only on a cache miss: torch and XGBoost each bring their own OpenMP runtime,
    # and loading both in one macOS process deadlocks the tree fit. Every downstream
    # experiment reads the cache, so torch never has to be present for them.
    import torch

    if device is None:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
    tmpl = _template()
    model = _load_mpnn(weights, device)

    uniq_df = pd.DataFrame(
        {"hla_seq": [k.split("|")[0] for k in uniq], "peptide": [k.split("|")[1] for k in uniq]}
    )
    S_all = _encode_rows(uniq_df, tmpl)
    generator = torch.Generator(device="cpu").manual_seed(seed)

    # Without per-allele folding every row shares one backbone, so the tensors are built once.
    # With it, each groove sequence gets its own structure and rows are grouped by allele.
    folded: dict[str, np.ndarray] = {}
    if per_allele:
        from .fold import fold_alleles

        folded = fold_alleles(df, device=device)
        groups = [
            (seq, np.flatnonzero(uniq_df["hla_seq"].to_numpy() == seq))
            for seq in uniq_df["hla_seq"].unique()
        ]
        print(f"  scoring against {len(groups)} predicted grooves", flush=True)
    else:
        groups = [(None, np.arange(len(uniq)))]

    out = np.zeros((len(uniq), len(FEATURE_NAMES)), dtype=np.float32)
    for groove_seq, members in groups:
        coords = (
            _groove_in_template_frame(folded[groove_seq], tmpl) if groove_seq is not None else None
        )
        static = _static_tensors(tmpl, device, hla_coords=coords)
        L = S_all.shape[1]
        pep_slice = static["slice_pep"]
        _score_group(
            model, S_all, members, static, L, pep_slice, n_decoding_orders,
            batch_size, device, generator, out,
        )

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, keys=np.array(uniq), vectors=out)
    table = dict(zip(uniq, out, strict=True))
    return np.stack([table[k] for k in keys]).astype(np.float32)


def _score_group(
    model, S_all, members, static, L, pep_slice, n_decoding_orders,
    batch_size, device, generator, out,
) -> None:
    """Score one set of rows that share a backbone."""
    import torch

    with torch.no_grad():
        for start in range(0, len(members), batch_size):
            idx = members[start : start + batch_size]
            S = torch.tensor(S_all[idx], device=device)
            b = S.shape[0]
            X = static["X"].unsqueeze(0).expand(b, -1, -1, -1).contiguous()
            mask = torch.ones(b, L, device=device)
            chain_M = static["chain_M"].unsqueeze(0).expand(b, -1).contiguous()
            residue_idx = static["residue_idx"].unsqueeze(0).expand(b, -1).contiguous()
            chain_enc = static["chain_encoding"].unsqueeze(0).expand(b, -1).contiguous()

            acc = torch.zeros(b, static["n_pep"], device=device)
            ent = torch.zeros(b, static["n_pep"], device=device)
            for _ in range(n_decoding_orders):
                randn = torch.randn(b, L, generator=generator).to(device)
                log_probs = model(X, S, mask, chain_M, residue_idx, chain_enc, randn)
                lp = log_probs[:, pep_slice, :]
                acc += lp.gather(2, S[:, pep_slice].unsqueeze(-1)).squeeze(-1)
                ent += -(lp.exp() * lp).sum(-1)
            per_pos = (acc / n_decoding_orders).float().cpu().numpy()
            entropy = (ent / n_decoding_orders).float().cpu().numpy()

            block = np.concatenate(
                [per_pos,
                 per_pos.sum(1, keepdims=True),
                 per_pos.mean(1, keepdims=True),
                 per_pos.min(1, keepdims=True),
                 entropy.mean(1, keepdims=True)],
                axis=1,
            )
            out[idx] = block


def zero_shot_ranking(df: pd.DataFrame, **kwargs) -> np.ndarray:
    """Untrained score: summed log-likelihood of the peptide given the groove."""
    return score_structure(df, **kwargs)[:, FEATURE_NAMES.index("mpnn_sum")]
