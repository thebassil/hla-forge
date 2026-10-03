"""Downstream predictors. Every model is a plain sklearn-style regressor over a feature matrix."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _xgboost(params: dict[str, Any], seed: int):
    from xgboost import XGBRegressor

    defaults = dict(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.06,
        subsample=0.8,
        colsample_bytree=0.6,
        min_child_weight=4,
        reg_lambda=1.0,
        n_jobs=-1,
        tree_method="hist",
        random_state=seed,
    )
    return XGBRegressor(**{**defaults, **params})


def build_model(
    kind: str,
    params: dict[str, Any] | None = None,
    seed: int = 0,
    reduce: int | None = None,
):
    """Construct a predictor. `reduce` prepends a truncated SVD to the pipeline.

    The SVD is part of the pipeline, so it is fit on the training fold only. Frozen PLM
    embeddings are 480-1280 dense dimensions per side and tree ensembles scale badly in that
    regime; projecting to a few hundred components cuts fit time several-fold. Because it is
    unsupervised and fold-local, it cannot leak the target.
    """
    est = _build_estimator(kind, params, seed)
    if reduce:
        return Pipeline([("svd", TruncatedSVD(n_components=reduce, random_state=seed)), ("est", est)])
    return est


def _build_estimator(kind: str, params: dict[str, Any] | None = None, seed: int = 0):
    params = dict(params or {})
    if kind == "ridge":
        return Pipeline(
            [("scale", StandardScaler()), ("est", Ridge(**{"alpha": 10.0, **params}))]
        )
    if kind == "rf":
        defaults = dict(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=seed)
        return RandomForestRegressor(**{**defaults, **params})
    if kind == "hgb":
        defaults = dict(max_iter=400, learning_rate=0.06, random_state=seed)
        return HistGradientBoostingRegressor(**{**defaults, **params})
    if kind == "xgboost":
        return _xgboost(params, seed)
    if kind == "mlp":
        defaults = dict(
            hidden_layer_sizes=(512, 128),
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=120,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=seed,
        )
        return Pipeline(
            [("scale", StandardScaler()), ("est", MLPRegressor(**{**defaults, **params}))]
        )
    if kind == "mean":
        return _MeanBaseline()
    raise ValueError(f"unknown model kind '{kind}'")


class _MeanBaseline:
    """Predicts the training mean. Any model that cannot beat this is not learning."""

    def fit(self, X, y):
        self._mu = float(np.mean(y))
        return self

    def predict(self, X):
        return np.full(len(X), self._mu)
