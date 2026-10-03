"""Modal GPU backend for the expensive embedding passes.

Local runs cover everything up to ESM2-150M on CPU/MPS. Use this for ESM2-650M, residue-level
flattening on the full dataset, or any fine-tuning. Embeddings come back into the same
artifacts/embeddings cache the local CLI reads.

    modal run modal_app.py::embed --model esm2_t33 --pooling mean
"""

from __future__ import annotations

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.2", "transformers>=4.44", "numpy>=1.26", "pandas>=2.2",
        "scipy>=1.13", "scikit-learn>=1.5", "xgboost>=2.1", "pyyaml>=6.0",
        "typer>=0.12", "rich>=13.7", "tqdm>=4.66",
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("configs", remote_path="/root/configs")
    .add_local_file("data/raw/rasmussen_stability.csv", "/root/data/raw/rasmussen_stability.csv")
)

volume = modal.Volume.from_name("hlaforge-artifacts", create_if_missing=True)
app = modal.App("hlaforge")


@app.function(image=image, gpu="A10G", timeout=3600, volumes={"/root/artifacts": volume})
def _embed_remote(model: str, pooling: str, hla_field: str) -> list[str]:
    import sys

    sys.path.insert(0, "/root/src")
    import os

    os.chdir("/root")
    from hlaforge.data import load_raw
    from hlaforge.embeddings import embed_sequences

    df = load_raw()
    for column in ["peptide", hla_field]:
        embed_sequences(
            df[column].tolist(), model=model, pooling=pooling, batch_size=128, device="cuda"
        )
    volume.commit()
    import pathlib

    return [p.name for p in pathlib.Path("/root/artifacts/embeddings").glob("*.npz")]


@app.local_entrypoint()
def embed(model: str = "esm2_t33", pooling: str = "mean", hla_field: str = "hla_pseudoseq"):
    files = _embed_remote.remote(model, pooling, hla_field)
    print("cached on the hlaforge-artifacts volume:")
    for f in files:
        print(" ", f)
    print("\nPull them locally with:  modal volume get hlaforge-artifacts / artifacts/embeddings")
