"""Quantile forecasts without scikit-learn on the agents: gradient-boosted trees exported to gzipped JSON by
tools/export_models.py and evaluated here with numpy only (the Jetsons and the Mac need no ML library).

    m = TreeModel.load("models/forecast_2025_26/home_load.json.gz")
    p10, p50, p90 = m.predict(rows)            # rows: list of dicts or a 2-D array in m.features order
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np


class _Trees:
    """All trees of one model evaluated together: their nodes are laid end to end, every row walks every tree at
    once (one numpy step per level of depth, not per tree), and the leaves reached are summed."""

    def __init__(self, d: dict):
        self.baseline = float(d["baseline"])
        trees = [{k: np.asarray(v) for k, v in t.items()} for t in d["trees"]]
        bitsets = [np.asarray(b, dtype=np.uint64).reshape(-1, 8) for b in d["bitsets"]]
        node_off = np.cumsum([0] + [len(t["f"]) for t in trees])[:-1]
        bit_off = np.cumsum([0] + [len(b) for b in bitsets])[:-1]
        cat = lambda k, dt: np.concatenate([t[k] for t in trees]).astype(dt)      # noqa: E731
        self.f = cat("f", np.int64)
        self.thr = cat("thr", float)
        self.leaf = cat("leaf", bool)
        self.v = cat("v", float)
        self.miss_left = cat("miss_left", bool)
        self.cat = cat("cat", bool)
        self.l = np.concatenate([t["l"] + o for t, o in zip(trees, node_off)]).astype(np.int64)
        self.r = np.concatenate([t["r"] + o for t, o in zip(trees, node_off)]).astype(np.int64)
        self.bitset = np.concatenate([t["bitset"] + o for t, o in zip(trees, bit_off)]).astype(np.int64)
        self.bits = np.concatenate(bitsets) if bitsets else np.zeros((1, 8), dtype=np.uint64)
        self.roots = node_off.astype(np.int64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        n = X.shape[0]
        node = np.broadcast_to(self.roots, (n, len(self.roots))).copy()      # rows x trees
        rows = np.arange(n)[:, None]
        for _ in range(64):
            leaf = self.leaf[node]
            if leaf.all():
                break
            x = X[rows, self.f[node]]
            nan = np.isnan(x)
            left = np.where(nan, self.miss_left[node], x <= self.thr[node])
            cat = self.cat[node] & ~nan & ~leaf
            if cat.any():
                xi = x[cat].astype(np.int64)
                words = self.bits[self.bitset[node][cat], xi // 32]
                left[cat] = ((words >> (xi % 32).astype(np.uint64)) & np.uint64(1)) == 1
            node = np.where(leaf, node, np.where(left, self.l[node], self.r[node]))
        return self.baseline + self.v[node].sum(axis=1)


class TreeModel:
    def __init__(self, d: dict):
        self.features: list[str] = d["features"]
        self.quantiles = sorted(float(q) for q in d["quantiles"])
        self._q = {float(q): _Trees(v) for q, v in d["quantiles"].items()}
        self._extra = {k: _Trees(v) for k, v in d.get("extra", {}).items()}   # e.g. "mean" (squared-error model)
        self.widen = float(d.get("widen", 0.0))         # calibration: how much to widen the 10-90 % range (target units)
        self.meta = d.get("meta", {})
        # scikit-learn encodes categorical inputs (value -> its position in the category list) and puts them first
        self.categories = {k: {float(v): i for i, v in enumerate(vals)} for k, vals in d.get("categories", {}).items()}
        self.order = d.get("order", self.features)
        self._cols = [self.features.index(c) for c in self.order]

    @classmethod
    def load(cls, path: str | Path) -> "TreeModel":
        with gzip.open(path, "rt") as f:
            return cls(json.load(f))

    def matrix(self, rows) -> np.ndarray:
        """Rows in self.features order (dicts or an array) -> the encoded, reordered matrix the trees read."""
        if isinstance(rows, np.ndarray):
            X = rows.astype(float)
        else:
            X = np.array([[np.nan if r.get(c) is None else float(r[c]) for c in self.features] for r in rows], dtype=float)
        X = X[:, self._cols].copy()
        for j, c in enumerate(self.order):
            if c in self.categories:
                m = self.categories[c]
                X[:, j] = [m.get(float(v), np.nan) if np.isfinite(v) else np.nan for v in X[:, j]]
        return X

    def has(self, name: str) -> bool:
        return name in self._extra

    def predict_extra(self, name: str, rows) -> np.ndarray:
        """An extra model stored with the quantiles, e.g. "mean": the expected value (what sums of forecasts need)."""
        return self._extra[name].predict(self.matrix(rows))

    def predict(self, rows, calibrated: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        X = self.matrix(rows)
        p = np.sort(np.column_stack([self._q[q].predict(X) for q in self.quantiles]), axis=1)
        lo, mid, hi = p[:, 0], p[:, len(self.quantiles) // 2], p[:, -1]
        if calibrated and self.widen:
            lo, hi = lo - self.widen, hi + self.widen
        return lo, mid, hi


def _trees_of(est) -> dict:
    trees, bitsets = [], []
    for (tp,) in est._predictors:
        n = tp.nodes
        trees.append({"f": n["feature_idx"].astype(int).tolist(), "thr": [float(v) for v in n["num_threshold"]],
                      "l": n["left"].astype(int).tolist(), "r": n["right"].astype(int).tolist(),
                      "leaf": n["is_leaf"].astype(int).tolist(), "v": [float(v) for v in n["value"]],
                      "miss_left": n["missing_go_to_left"].astype(int).tolist(),
                      "cat": n["is_categorical"].astype(int).tolist(), "bitset": n["bitset_idx"].astype(int).tolist()})
        bitsets.append(tp.raw_left_cat_bitsets.astype(int).tolist() if len(tp.raw_left_cat_bitsets) else [[0] * 8])
    return {"baseline": float(np.ravel(est._baseline_prediction)[0]), "trees": trees, "bitsets": bitsets}


def export_sklearn(models: dict, features: list[str], widen: float = 0.0, meta: dict | None = None,
                   extra: dict | None = None) -> dict:
    """{quantile: HistGradientBoostingRegressor} (+ extra {name: regressor on the same features}) -> the JSON
    structure TreeModel reads."""
    out = {"features": list(features), "quantiles": {}, "widen": float(widen), "meta": meta or {}}
    est0 = next(iter(models.values()))
    pre = getattr(est0, "_preprocessor", None)
    if pre is not None:
        cat_mask = np.asarray(est0.is_categorical_ if est0.is_categorical_ is not None else [False] * len(features))
        enc = pre.named_transformers_["encoder"]
        cats = [f for f, c in zip(features, cat_mask) if c]
        out["categories"] = {f: [float(v) for v in vals] for f, vals in zip(cats, enc.categories_)}
        out["order"] = cats + [f for f, c in zip(features, cat_mask) if not c]
    for q, est in models.items():
        out["quantiles"][str(q)] = _trees_of(est)
    for name, est in (extra or {}).items():
        out.setdefault("extra", {})[name] = _trees_of(est)
    return out
