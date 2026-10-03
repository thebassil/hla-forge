"""Experiment orchestration: representation -> interaction -> predictor -> metrics.

One experiment is one YAML config. Every axis can be varied independently while everything
else is held fixed, which is what makes the resulting comparisons attributable to a single
design decision rather than to a bundle of them.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from . import features as cheap
from . import splits as split_mod
from .evaluate import aggregate_folds, compute_metrics
from .models import build_model

RESULTS_DIR = Path("artifacts/results")
INTERACTIONS = ["concat", "product", "absdiff", "all", "peptide_only", "hla_only"]


@dataclass
class ExperimentConfig:
    name: str = "unnamed"
    representation: dict[str, Any] = field(default_factory=lambda: {"kind": "cheap"})
    interaction: str = "concat"
    model: dict[str, Any] = field(default_factory=lambda: {"kind": "ridge"})
    target: str = "log1p"
    splits: list[str] = field(default_factory=lambda: ["random", "peptide", "allele"])
    n_splits: int = 5
    seed: int = 0
    limit: int | None = None
    notes: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExperimentConfig:
        raw = yaml.safe_load(Path(path).read_text()) or {}
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "nogit"


def build_sides(df: pd.DataFrame, rep: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return (peptide_matrix, hla_matrix) for the configured representation."""
    kind = rep.get("kind", "cheap")

    if kind == "cheap":
        pep = cheap.encode_side(df, "peptide", rep.get("peptide", ["blosum"]))
        hla = cheap.encode_side(df, "hla", rep.get("hla", ["blosum"]))
        return pep, hla

    if kind in ("plm", "hybrid"):
        from .embeddings import encode_side_plm

        plm = rep.get("plm", {})
        kw = dict(
            model=plm.get("model", "esm2_t12"),
            hla_field=plm.get("hla_field", "hla_pseudoseq"),
            layer=plm.get("layer", -1),
            batch_size=plm.get("batch_size", 64),
            device=plm.get("device"),
        )
        # Pooling can differ per side. Flattening a 9-mer costs 9 x D columns, but flattening a
        # 34-residue pseudosequence costs 34 x D -- so the useful setting is often "keep the
        # peptide position-resolved, pool the groove".
        default_pool = plm.get("pooling", "mean")
        pep = encode_side_plm(
            df, "peptide", pooling=plm.get("pooling_peptide", default_pool), **kw
        )
        hla = encode_side_plm(df, "hla", pooling=plm.get("pooling_hla", default_pool), **kw)
        if kind == "hybrid":
            pep = np.concatenate(
                [pep, cheap.encode_side(df, "peptide", rep.get("peptide", ["blosum"]))], axis=1
            )
            hla = np.concatenate(
                [hla, cheap.encode_side(df, "hla", rep.get("hla", ["blosum"]))], axis=1
            )
        return pep, hla

    raise ValueError(f"unknown representation kind '{kind}'")


def combine(pep: np.ndarray, hla: np.ndarray, interaction: str) -> np.ndarray:
    if interaction == "peptide_only":
        return pep
    if interaction == "hla_only":
        return hla
    if interaction == "concat":
        return np.concatenate([pep, hla], axis=1)
    if interaction in ("product", "absdiff", "all"):
        if pep.shape[1] != hla.shape[1]:
            raise ValueError(
                f"interaction '{interaction}' needs matching widths, got "
                f"{pep.shape[1]} vs {hla.shape[1]}. Use a PLM representation or 'concat'."
            )
        prod = pep * hla
        diff = np.abs(pep - hla)
        if interaction == "product":
            return np.concatenate([pep, hla, prod], axis=1)
        if interaction == "absdiff":
            return np.concatenate([pep, hla, diff], axis=1)
        return np.concatenate([pep, hla, prod, diff], axis=1)
    raise ValueError(f"unknown interaction '{interaction}'. Known: {INTERACTIONS}")


def build_features(df: pd.DataFrame, cfg: ExperimentConfig) -> np.ndarray:
    rep = cfg.representation
    blocks: list[np.ndarray] = []

    if rep.get("kind", "cheap") != "likelihood":
        pep, hla = build_sides(df, rep)
        blocks.append(combine(pep, hla, cfg.interaction))

    # Zero-shot PLM likelihood features can stand alone or be bolted onto any representation.
    lik = rep.get("likelihood")
    if lik or rep.get("kind") == "likelihood":
        from .likelihood import score_pairs

        lik = dict(lik or {})
        blocks.append(score_pairs(df, **lik))

    if not blocks:
        raise ValueError("representation produced no features")
    X = blocks[0] if len(blocks) == 1 else np.concatenate(blocks, axis=1)
    return np.ascontiguousarray(X, dtype=np.float32)


def _target(df: pd.DataFrame, target: str) -> np.ndarray:
    if target == "log1p":
        return df["y"].to_numpy(dtype=np.float32)
    if target == "raw":
        return df["thalf_hours"].to_numpy(dtype=np.float32)
    if target == "binary":
        return df["y_binary"].to_numpy(dtype=np.float32)
    raise ValueError(f"unknown target '{target}'")


def run_experiment(
    df: pd.DataFrame, cfg: ExperimentConfig, save: bool = True, verbose: bool = True
) -> list[dict[str, Any]]:
    """Run one config across every configured split regime. Returns one record per split."""
    t0 = time.time()
    X = build_features(df, cfg)
    y = _target(df, cfg.target)
    alleles = df["allele"].to_numpy()
    y_bin = df["y_binary"].to_numpy()
    feat_time = time.time() - t0

    records = []
    for split_name in cfg.splits:
        folds = split_mod.make_folds(df, split_name, n_splits=cfg.n_splits, seed=cfg.seed)
        fold_metrics = []
        t_split = time.time()
        for train_idx, test_idx in folds:
            model = build_model(
                cfg.model.get("kind", "ridge"),
                cfg.model.get("params"),
                cfg.seed,
                reduce=cfg.model.get("reduce"),
            )
            model.fit(X[train_idx], y[train_idx])
            pred = np.asarray(model.predict(X[test_idx]), dtype=float)
            fold_metrics.append(
                compute_metrics(
                    y[test_idx].astype(float),
                    pred,
                    alleles=alleles[test_idx],
                    y_binary=y_bin[test_idx],
                )
            )
        record = {
            "name": cfg.name,
            "split": split_name,
            "metrics": aggregate_folds(fold_metrics),
            "per_fold": fold_metrics,
            "n_features": int(X.shape[1]),
            "n_rows": int(len(df)),
            "n_folds": len(folds),
            "feature_seconds": round(feat_time, 2),
            "fit_seconds": round(time.time() - t_split, 2),
            "config": cfg.to_dict(),
            "git_sha": _git_sha(),
        }
        records.append(record)
        if verbose:
            m = record["metrics"]
            print(
                f"  {cfg.name:28s} {split_name:16s} "
                f"rho={m.get('spearman', float('nan')):.3f}"
                f"±{m.get('spearman_std', 0):.3f}  "
                f"rho_allele={m.get('spearman_per_allele_mean', float('nan')):.3f}  "
                f"auc={m.get('auc', float('nan')):.3f}  "
                f"rmse={m.get('rmse', float('nan')):.3f}  "
                f"[{record['fit_seconds']}s]"
            )

    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = str(int(t0))
        out = RESULTS_DIR / f"{cfg.name}__{stamp}.json"
        out.write_text(json.dumps(records, indent=2, default=float))
        if verbose:
            print(f"  -> {out}")
    return records


def load_results(results_dir: str | Path = RESULTS_DIR) -> pd.DataFrame:
    rows = []
    for path in sorted(Path(results_dir).glob("*.json")):
        for rec in json.loads(path.read_text()):
            row = {
                "name": rec["name"],
                "split": rec["split"],
                "n_features": rec.get("n_features"),
                "fit_seconds": rec.get("fit_seconds"),
                "file": path.name,
            }
            row.update(rec["metrics"])
            rows.append(row)
    return pd.DataFrame(rows)
