"""Metrics.

Overall Spearman is the headline number, but per-allele Spearman is the one that matters
biologically: a model can score well overall purely by learning which alleles are sticky, while
being useless at ranking peptides within the allele a clinician actually cares about.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


def _safe_spearman(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) < 3 or np.std(y) == 0 or np.std(p) == 0:
        return float("nan")
    return float(spearmanr(y, p).statistic)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alleles: np.ndarray | None = None,
    y_binary: np.ndarray | None = None,
    min_allele_n: int = 20,
) -> dict[str, float]:
    m: dict[str, float] = {}
    m["spearman"] = _safe_spearman(y_true, y_pred)
    m["pearson"] = (
        float(pearsonr(y_true, y_pred).statistic)
        if len(y_true) > 2 and np.std(y_true) > 0 and np.std(y_pred) > 0
        else float("nan")
    )
    m["rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    m["mae"] = float(np.mean(np.abs(y_true - y_pred)))
    m["n"] = float(len(y_true))

    if y_binary is not None and len(np.unique(y_binary)) == 2:
        m["auc"] = float(roc_auc_score(y_binary, y_pred))
        m["ap"] = float(average_precision_score(y_binary, y_pred))

    if alleles is not None:
        per = []
        per_auc = []
        for allele in np.unique(alleles):
            mask = alleles == allele
            if mask.sum() < min_allele_n:
                continue
            per.append(_safe_spearman(y_true[mask], y_pred[mask]))
            if y_binary is not None and len(np.unique(y_binary[mask])) == 2:
                per_auc.append(float(roc_auc_score(y_binary[mask], y_pred[mask])))
        per = [v for v in per if not np.isnan(v)]
        if per:
            m["spearman_per_allele_mean"] = float(np.mean(per))
            m["spearman_per_allele_median"] = float(np.median(per))
            m["spearman_per_allele_min"] = float(np.min(per))
            m["n_alleles_scored"] = float(len(per))
        if per_auc:
            m["auc_per_allele_mean"] = float(np.mean(per_auc))
    return m


def aggregate_folds(fold_metrics: list[dict[str, float]]) -> dict[str, float]:
    """Mean and standard deviation across folds, so single-fold luck is visible."""
    keys = sorted({k for fm in fold_metrics for k in fm})
    out: dict[str, float] = {}
    for k in keys:
        vals = np.array([fm.get(k, np.nan) for fm in fold_metrics], dtype=float)
        vals = vals[~np.isnan(vals)]
        if len(vals) == 0:
            continue
        out[k] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    return out


def results_table(records: list[dict]) -> pd.DataFrame:
    rows = []
    for r in records:
        row = {"name": r["name"], "split": r["split"]}
        row.update({k: v for k, v in r["metrics"].items()})
        rows.append(row)
    return pd.DataFrame(rows)
