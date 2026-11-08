"""What a building agent runs: tomorrow's forecast from the gzipped JSON trees, with numpy only. stage4/code/buildings_agent.py, brought into the package.

    f = BuildingForecaster("university")
    f.forecast("2026-07-02", [(h, fc_temp[h]) for h in range(24)])     # (local hour, forecast as issued, in C)
    -> {hour: {"total": (p10, p50, p90), "cooling": ..., "base": ..., "fans": kW, "lighting": kW, "rest": kW}}

The rows need only the date and the temperature forecast. The published calendar travels in each file's meta and stops on 30 Sep 2026: next
year's dates must be added before it is used on them. Total load beats the rule at sale hours; cooling on hot evenings only ties it.
Before agents rely on it, stage3/nodes must be rebuilt from the rebuilt series (stage4 reports/check.json): until then the forecast runs 5 to 8 % off.
"""
from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np

from .treemodel import TreeModel

MODELS = Path(__file__).resolve().parents[1] / "models" / "forecast_2026" / "buildings"
_LOADED: dict[tuple[str, str], "BuildingForecaster"] = {}
LAST_DAY = "2026-09-30"              # the published calendar stops here
TEMP_C = (-40.0, 50.0)               # the trees answer anything: a missing or absurd forecast must stop here, not become a lower load


def for_node(name: str) -> "BuildingForecaster":
    """bld-office's models, from the folder the dataset names: a date-limited copy, for a day it must not have seen, works as it does for homes."""
    from . import config
    meta = config.dataset_meta()
    folder = config.REPO_DIR / meta["models_2026"] / "buildings" if meta.get("models_2026") else MODELS
    k = (name, str(folder))
    if k not in _LOADED:
        _LOADED[k] = BuildingForecaster(name.split("-", 1)[1], folder)
    return _LOADED[k]


class BuildingForecaster:
    def __init__(self, key: str, models: Path = MODELS):
        self.key = key
        self.total = TreeModel.load(models / f"{key}_total.json.gz")
        self.cooling = TreeModel.load(models / f"{key}_cooling.json.gz")
        self.base = TreeModel.load(models / f"{key}_base.json.gz")
        self.fix = TreeModel.load(models / "boston_temp_fix.json.gz")
        card = self.base.meta["card"]
        self.cal, self.shares = card["calendar"], card["shares"]
        self.holidays = set(self.cal["holidays"])

    def corrected(self, hours: list[int], temps: list[float]) -> np.ndarray:
        """The forecast as issued, with its usual error for this hour and temperature taken off."""
        return np.round(np.asarray(temps, dtype=float) + self.fix.predict([{"hour": h, "temp": t} for h, t in zip(hours, temps)])[1], 2)

    def row(self, day: str, hour: int, temp_fixed: float) -> dict:
        d = date.fromisoformat(day)
        dow = d.weekday()
        r = {"hour": hour, "temp_fixed": temp_fixed, "weekend": int(dow >= 5), "daytype": 2 if day in self.holidays or dow == 6 else int(dow == 5)}
        if "sessions" in self.cal:
            r["session"] = next((c for a, b, c in self.cal["sessions"] if a <= day <= b), 0)
        if "closed_days" in self.cal:
            o, a, b = self.cal["open_hours"], *self.cal["christmas_season"]
            r["closed_day"] = int(day in self.cal["closed_days"])
            r["xmas_season"] = int(a <= day <= b)
            r["open_now"] = int(o["from"] <= hour < (o["sunday_to"] if dow == 6 else o["to"]) and not r["closed_day"])
        return r

    def predict(self, rows: list[dict], calibrated: bool = True) -> dict:
        return {"total": self.total.predict(rows, calibrated=calibrated), "cooling": self.cooling.predict(rows, calibrated=calibrated),
                "base": self.base.predict(rows, calibrated=calibrated)}

    def forecast(self, day: str, hours: list[tuple[int, float]]) -> dict:
        if date.fromisoformat(day) > date.fromisoformat(LAST_DAY):
            raise ValueError(f"no calendar for {day}: it stops on {LAST_DAY}")
        for h, t in hours:
            if not 0 <= h <= 23 or not math.isfinite(t) or not TEMP_C[0] <= t <= TEMP_C[1]:
                raise ValueError(f"no forecast for hour {h} at {t} C")
        fixed = self.corrected([h for h, _ in hours], [t for _, t in hours])
        p = self.predict([self.row(day, h, float(t)) for (h, _), t in zip(hours, fixed)])
        out = {}
        for i, (h, _) in enumerate(hours):
            fans, light = self.shares["vent"] * p["base"][1][i], self.shares["light"] * p["base"][1][i]
            f = {k: tuple(max(0.0, float(v[j][i])) for j in range(3)) for k, v in p.items()}
            out[h] = f | {"fans": float(fans), "lighting": float(light),
                          "rest": float(p["total"][1][i] - max(0.0, p["cooling"][1][i]) - fans - light)}
        return out
