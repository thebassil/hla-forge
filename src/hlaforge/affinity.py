"""R6: binding affinity as a transfer signal.

The literature's clearest finding on this task is that stability prediction improves when you
come at it through binding affinity, for which orders of magnitude more data exist. MINT reports
ESM-2 going 0.57 -> 0.75 purely from a binding-affinity-to-stability curriculum, and TLStab
found that simply handing a model a binding-affinity prediction as a feature was often as good
as their full transfer machinery.

So this is the cheap form of that idea: run MHCflurry, a pretrained affinity and presentation
predictor, over every peptide-allele pair and hand its outputs to the same gradient-boosted
trees everything else uses. It tests the transfer hypothesis without a transfer pipeline.

One caveat stated up front: MHCflurry is trained on IEDB binding-affinity and eluted-ligand
data, not on this stability dataset, so it is not leakage in the usual sense. But peptides
assayed for stability were often selected because they were predicted binders, so the two
corpora are not independent. The honest reading is "does a binding predictor carry stability
information a sequence encoding does not", not "we discovered new signal".
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

CACHE_DIR = Path("artifacts/embeddings")

FEATURE_NAMES = [
    "mhcflurry_log50k",            # log-transformed predicted IC50, the usual pMHC scale
    "mhcflurry_percentile",        # allele-normalised rank, so alleles are comparable
    "mhcflurry_ci_width_log",      # ensemble disagreement: how sure is the affinity predictor
    "mhcflurry_log50k_low",
    "mhcflurry_log50k_high",
]


def _normalise_allele(allele: str) -> str:
    """Strip the engineered-construct suffix: HLA-B*14:01(C67S) -> HLA-B*14:01.

    The three mutants have no MHCflurry model of their own. Falling back to the parent allele
    is the only option, and it is a real limitation: C67S destabilises the groove, which is
    exactly why those constructs are up to 92% below detection.
    """
    return allele.split("(")[0].strip()


def predict_affinity(
    df: pd.DataFrame, cache: bool = True, batch_size: int = 20000
) -> np.ndarray:
    """MHCflurry outputs per row: affinity, percentile ranks, processing and presentation."""
    keys = (df["allele"] + "|" + df["peptide"]).tolist()
    uniq = sorted(set(keys))
    digest = hashlib.sha1("\n".join(uniq).encode()).hexdigest()[:16]
    path = CACHE_DIR / f"mhcflurry__{digest}.npz"
    if cache and path.exists():
        data = np.load(path, allow_pickle=False)
        table = dict(zip(data["keys"].tolist(), data["vectors"], strict=True))
        return np.stack([table[k] for k in keys]).astype(np.float32)

    from mhcflurry import Class1AffinityPredictor

    predictor = Class1AffinityPredictor.load()
    supported = set(predictor.supported_alleles)

    alleles = [_normalise_allele(k.split("|")[0]) for k in uniq]
    peptides = [k.split("|")[1] for k in uniq]
    unsupported = sorted({a for a in alleles if a not in supported})
    if unsupported:
        print(f"  {len(unsupported)} alleles not supported by MHCflurry: {unsupported[:6]}")

    out = np.full((len(uniq), len(FEATURE_NAMES)), np.nan, dtype=np.float32)
    usable = [i for i, a in enumerate(alleles) if a in supported]
    log50k = np.log(50000.0)
    for start in range(0, len(usable), batch_size):
        idx = usable[start : start + batch_size]
        res = predictor.predict_to_dataframe(
            peptides=[peptides[i] for i in idx],
            alleles=[alleles[i] for i in idx],
            include_percentile_ranks=True,
            include_confidence_intervals=True,
            throw=False,
        )
        lo = np.log(res["prediction_low"].to_numpy()) / log50k
        hi = np.log(res["prediction_high"].to_numpy()) / log50k
        block = np.stack(
            [
                np.log(res["prediction"].to_numpy()) / log50k,
                res["prediction_percentile"].to_numpy(),
                hi - lo,
                lo,
                hi,
            ],
            axis=1,
        ).astype(np.float32)
        out[idx] = block

    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, keys=np.array(uniq), vectors=out)
    table = dict(zip(uniq, out, strict=True))
    return np.stack([table[k] for k in keys]).astype(np.float32)
