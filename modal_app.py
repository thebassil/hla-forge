"""Modal backend: fan the ablation queue out across containers, and run the GPU-only jobs.

The bottleneck in this project is not the protein language model -- embedding every unique
sequence costs under two minutes even at 650M parameters. The bottleneck is fitting tree
ensembles on dense embedding features, which is CPU-bound and embarrassingly parallel across
experiments. So the queue runs as one container per experiment and the wall clock collapses
from the sum of the experiments to the duration of the slowest one.

    modal volume create hlaforge-artifacts            # once
    modal volume put hlaforge-artifacts artifacts/embeddings /embeddings   # once, ~150MB
    modal run modal_app.py::queue                     # the whole ablation queue
    modal run modal_app.py::embed --model esm2_t33    # GPU embedding pass
"""

from __future__ import annotations

import json

import modal

DEPS = [
    "torch>=2.2", "transformers>=4.44", "numpy>=1.26", "pandas>=2.2,<3",
    "scipy>=1.13", "scikit-learn>=1.5", "xgboost>=2.1", "pyyaml>=6.0",
    "typer>=0.12", "rich>=13.7", "tqdm>=4.66",
]

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(*DEPS)
    .apt_install("git")
    .env({"PYTHONPATH": "/root/src", "HF_HOME": "/cache/hf"})
    .run_commands("git clone --depth 1 https://github.com/dauparas/ProteinMPNN "
                  "/root/external/ProteinMPNN")
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("scripts", remote_path="/root/scripts")
    .add_local_dir("data/template", remote_path="/root/data/template")
    .add_local_file("data/raw/rasmussen_stability.csv", "/root/data/raw/rasmussen_stability.csv")
)

artifacts = modal.Volume.from_name("hlaforge-artifacts", create_if_missing=True)
hf_cache = modal.Volume.from_name("hlaforge-hf-cache", create_if_missing=True)
app = modal.App("hlaforge")

VOLUMES = {"/root/artifacts": artifacts, "/cache": hf_cache}


def _chdir() -> None:
    import os
    import sys

    sys.path.insert(0, "/root/src")
    os.chdir("/root")


@app.function(image=image, cpu=16.0, memory=16384, timeout=3600, volumes=VOLUMES)
def _run_one(job: tuple[str, str, str], limit: int, folds: int, splits: list[str]) -> dict:
    """Run a single experiment. `job` is (family, name, json-encoded spec)."""
    import time

    _chdir()
    from hlaforge.data import load_raw
    from hlaforge.experiment import ExperimentConfig, run_experiment

    family, name, spec_json = job
    spec = json.loads(spec_json)
    df = load_raw(limit=limit)

    t0 = time.time()
    try:
        cfg = ExperimentConfig(
            name=name, splits=splits, n_splits=folds, seed=0, notes=family, **spec
        )
        records = run_experiment(df, cfg, save=False, verbose=False)
    except Exception as exc:
        return {
            "family": family, "name": name, "error": f"{type(exc).__name__}: {exc}",
            "seconds": time.time() - t0,
        }
    return {
        "family": family,
        "name": name,
        "seconds": time.time() - t0,
        "n_features": records[0]["n_features"],
        "by_split": {r["split"]: r["metrics"] for r in records},
    }


@app.function(image=image, gpu="A10G", timeout=7200, volumes=VOLUMES)
def _embed_remote(model: str, pooling: str, hla_field: str, batch_size: int) -> list[str]:
    import pathlib

    _chdir()
    from hlaforge.data import load_raw
    from hlaforge.embeddings import embed_sequences

    df = load_raw()
    for column in ["peptide", hla_field]:
        embed_sequences(
            df[column].tolist(), model=model, pooling=pooling,
            batch_size=batch_size, device="cuda",
        )
    artifacts.commit()
    hf_cache.commit()
    return [p.name for p in pathlib.Path("/root/artifacts/embeddings").glob("*.npz")]


@app.function(image=image, gpu="A10G", timeout=14400, volumes=VOLUMES)
def _score_remote(model: str, mode: str, limit: int | None) -> int:
    """Zero-shot likelihood scoring. `masked` mode is 9x the passes, hence the GPU."""
    _chdir()
    from hlaforge.data import load_raw
    from hlaforge.likelihood import score_pairs

    df = load_raw(limit=limit)
    mat = score_pairs(df, model=model, mode=mode, batch_size=128, device="cuda")
    artifacts.commit()
    hf_cache.commit()
    return int(mat.shape[0])


@app.local_entrypoint()
def queue(
    families: str = "R1,R2,R3",
    limit: int = 10000,
    folds: int = 3,
    splits: str = "peptide,allele,strict",
    out: str = "artifacts/growth_modal.json",
):
    """Fan the whole ablation queue out, one container per experiment."""
    import pathlib
    import sys
    import time

    sys.path.insert(0, "src")
    from hlaforge.queues import all_jobs

    fams = [f.strip() for f in families.split(",") if f.strip()]
    split_list = [s.strip() for s in splits.split(",") if s.strip()]
    jobs = [(fam, name, json.dumps(spec)) for fam, name, spec in all_jobs(fams)]
    print(f"dispatching {len(jobs)} experiments across {len(fams)} reference systems")

    t0 = time.time()
    results = list(_run_one.map(jobs, kwargs={"limit": limit, "folds": folds,
                                              "splits": split_list}))
    wall = time.time() - t0

    ok = [r for r in results if "error" not in r]
    bad = [r for r in results if "error" in r]
    container_seconds = sum(r["seconds"] for r in results)
    print(f"\n{len(ok)} succeeded, {len(bad)} failed")
    print(f"wall clock {wall / 60:.1f} min | container time {container_seconds / 60:.1f} min "
          f"| speedup {container_seconds / max(wall, 1e-9):.1f}x")
    for r in bad:
        print(f"  FAILED {r['name']}: {r['error']}")

    pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out).write_text(json.dumps(
        {"results": results, "wall_seconds": wall,
         "container_seconds": container_seconds,
         "settings": {"families": fams, "limit": limit, "folds": folds,
                      "splits": split_list}},
        indent=2, default=float))
    print(f"-> {out}\n   summarise with: python scripts/growth_report.py {out}")


@app.function(image=image, gpu="A10G", timeout=14400, volumes=VOLUMES)
def _mpnn_remote(weights: str, orders: int, batch_size: int, limit: int | None) -> int:
    """R4: inverse-folding scores. 1.8 rows/s on a laptop CPU, far faster here."""
    _chdir()
    from hlaforge.data import load_raw
    from hlaforge.structure import score_structure

    df = load_raw(limit=limit)
    mat = score_structure(
        df, weights=weights, n_decoding_orders=orders,
        batch_size=batch_size, device="cuda",
    )
    artifacts.commit()
    return int(mat.shape[0])


@app.function(image=image, cpu=16.0, memory=65536, timeout=7200, volumes=VOLUMES)
def _curve_remote(folds: int, seed: int, arms: str = "", caps: str = "") -> dict:
    """The data-starvation crossover curve. Needs a big box: the 650M flattened
    representation is 12,800 columns over 28,166 rows and OOMs a 24GB laptop."""
    import subprocess

    _chdir()
    cmd = ["python", "/root/scripts/data_curve.py", "--folds", str(folds),
           "--seed", str(seed)]
    if arms:
        cmd += ["--arms", *arms.split(",")]
    if caps:
        cmd += ["--caps", *caps.split(",")]
    out = subprocess.run(cmd, capture_output=True, text=True)
    artifacts.commit()
    return {"stdout": out.stdout[-8000:], "stderr": out.stderr[-3000:],
            "returncode": out.returncode}


@app.local_entrypoint()
def mpnn(weights: str = "v_48_020", orders: int = 4, batch_size: int = 64, limit: int = 0):
    n = _mpnn_remote.remote(weights, orders, batch_size, limit or None)
    print(f"scored {n} peptide-groove pairs with ProteinMPNN/{weights}")
    print("pull locally: modal volume get hlaforge-artifacts /embeddings artifacts/")


@app.local_entrypoint()
def curve(folds: int = 3, seed: int = 0, arms: str = "", caps: str = ""):
    res = _curve_remote.remote(folds, seed, arms, caps)
    print(res["stdout"])
    if res["returncode"] != 0:
        print("STDERR:", res["stderr"])


@app.local_entrypoint()
def embed(model: str = "esm2_t33", pooling: str = "mean",
          hla_field: str = "hla_pseudoseq", batch_size: int = 128):
    files = _embed_remote.remote(model, pooling, hla_field, batch_size)
    print("on volume hlaforge-artifacts:")
    for f in sorted(files):
        print(" ", f)
    print("\npull locally: modal volume get hlaforge-artifacts /embeddings artifacts/")


@app.local_entrypoint()
def score(model: str = "esm2_t12", mode: str = "masked", limit: int = 0):
    n = _score_remote.remote(model, mode, limit or None)
    print(f"scored {n} pairs with {model}/{mode}")
    print("pull locally: modal volume get hlaforge-artifacts /embeddings artifacts/")
