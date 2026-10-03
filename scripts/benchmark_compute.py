"""Measure real PLM throughput on this machine so the compute plan is grounded, not guessed.

Times a small batch for each ESM2 size and extrapolates to the full dataset's workload:
5,633 unique peptides, 75 unique HLA sequences, 28,166 unique peptide-HLA chimeras.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from hlaforge.data import load_raw
from hlaforge.embeddings import MODELS, _device

N_PROBE = 128
WORKLOAD = {"peptides": 5633, "hla": 75, "chimeras": 28166}
OUT = Path("artifacts/compute_budget.json")


def time_forward(model_key: str, seqs: list[str], batch_size: int, device: str) -> dict:
    import torch
    from transformers import AutoModel, AutoTokenizer

    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(MODELS[model_key])
    net = AutoModel.from_pretrained(MODELS[model_key]).to(device).eval()
    load_s = time.time() - t_load
    n_params = sum(p.numel() for p in net.parameters())

    with torch.no_grad():  # warm up: first MPS call compiles kernels
        enc = tok(seqs[:8], return_tensors="pt", padding=True).to(device)
        net(**enc)

    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(seqs), batch_size):
            enc = tok(seqs[start : start + batch_size], return_tensors="pt", padding=True).to(
                device
            )
            net(**enc)
    elapsed = time.time() - t0
    del net
    return {
        "params_m": round(n_params / 1e6, 1),
        "load_s": round(load_s, 1),
        "seqs_per_s": round(len(seqs) / elapsed, 1),
    }


def main() -> None:
    device = _device()
    df = load_raw()
    peptides = df["peptide"].drop_duplicates().tolist()[:N_PROBE]
    chimeras = (df["peptide"] + "GGGGSGGGGS" + df["hla_pseudoseq"]).drop_duplicates().tolist()[
        :N_PROBE
    ]

    results = {"device": device, "n_probe": N_PROBE, "models": {}}
    for key in ["esm2_t6", "esm2_t12", "esm2_t30", "esm2_t33"]:
        print(f"\n=== {key} ({MODELS[key]}) on {device}")
        try:
            short = time_forward(key, peptides, 64, device)
            long = time_forward(key, chimeras, 32, device)
        except Exception as exc:  # out of memory, download failure, unsupported op
            print(f"  skipped: {type(exc).__name__}: {exc}")
            results["models"][key] = {"error": f"{type(exc).__name__}: {exc}"}
            continue

        # Embedding pass: every unique peptide and HLA sequence, once.
        embed_s = WORKLOAD["peptides"] / short["seqs_per_s"] + WORKLOAD["hla"] / short["seqs_per_s"]
        # Zero-shot wt likelihood: one pass over every unique chimera.
        wt_s = WORKLOAD["chimeras"] / long["seqs_per_s"]
        # Masked marginals: nine masked passes per chimera.
        masked_s = 9 * wt_s

        entry = {
            "params_m": short["params_m"],
            "load_s": short["load_s"],
            "short_seqs_per_s": short["seqs_per_s"],
            "long_seqs_per_s": long["seqs_per_s"],
            "full_embed_min": round(embed_s / 60, 1),
            "zeroshot_wt_min": round(wt_s / 60, 1),
            "masked_marginal_min": round(masked_s / 60, 1),
        }
        results["models"][key] = entry
        print(
            f"  {entry['params_m']}M params | peptides {short['seqs_per_s']}/s | "
            f"chimeras {long['seqs_per_s']}/s"
        )
        print(
            f"  full embed {entry['full_embed_min']} min | "
            f"zero-shot {entry['zeroshot_wt_min']} min | "
            f"masked marginals {entry['masked_marginal_min']} min"
        )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\n-> {OUT}")


if __name__ == "__main__":
    main()
