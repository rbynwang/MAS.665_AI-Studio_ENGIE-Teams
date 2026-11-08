"""A home's air conditioning as something FlexGrid can move: the house's own one-node physics, relative to its day.

Every Massachusetts home is a simulated house (training_data/07_synthetic, homesim). Its cooling and indoor
temperature for every hour are in the dataset (tools/make_ma_hvac.py -> data_ma/..._hvac.csv.gz), and so are the
house's constants: UA (kW per C lost to the outdoors), C (kWh per C stored in the house), the cooling capacity.

FlexGrid works with the difference from that day, x = indoor temperature minus what it would have been. Removing
dq kW less heat than the house's thermostat would have (dq < 0: cooling eased off; dq > 0: pre-cooling) moves x by
the same physics homesim uses:

    C dx/dt = -UA x - dq        so over a step of dt hours, with a = exp(-dt / tau) and tau = C / UA:
    x(t + dt) = a x(t) - (dq / UA) (1 - a)

The electricity changes by dq / COP (homesim's cooling efficiency for that hour). With nobody steering it (dq the
thermostat's own choice) the house goes back to its day as fast as the cooling can: that is the rebound.

Winter (Stage 2): a heat pump is the same lever, turned around. x is then how far the house has moved in the less
comfortable direction: colder, not warmer. cool_kw in the data holds the heat pump's electricity and `mode` is -1, so
the physics, the owner's limits and the plan stay as they are. Only the reported indoor temperature changes sign.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import threading
from typing import Any

from .config import data_path

_CACHE: dict[str, dict[str, Any]] = {}
_PARAMS: dict[str, Any] | None = None
_LOCK = threading.Lock()


def params_for(name: str) -> dict[str, Any] | None:
    """The house constants of a FlexGrid home (data_ma/hvac_params.json), None where the dataset has none."""
    global _PARAMS
    with _LOCK:
        if _PARAMS is None:
            try:
                _PARAMS = json.loads(data_path("hvac_params.json").read_text()).get("homes", {})
            except (FileNotFoundError, OSError, ValueError):
                _PARAMS = {}
    return _PARAMS.get(name)


class _Series:
    """One house's own day, compact: numpy columns over a time index shared by every home with the same steps
    (100 homes x a season as Python tuples: about 320 MB; as arrays: 27 MB, measured 1 Nov 2026)."""
    __slots__ = ("index", "cool", "cop", "tin", "occ", "mode")

    def __init__(self, index: dict[str, int], cool, cop, tin, occ, mode=None):
        self.index, self.cool, self.cop, self.tin, self.occ, self.mode = index, cool, cop, tin, occ, mode

    def get(self, ts: str, default: tuple[float, float, float, int]) -> tuple[float, float, float, int]:
        i = self.index.get(ts)
        if i is None:
            return default
        return float(self.cool[i]), float(self.cop[i]), float(self.tin[i]), int(self.occ[i])

    def __bool__(self) -> bool:
        return len(self.cool) > 0


_INDEXES: dict[tuple[str, str, int], dict[str, int]] = {}       # (first ts, last ts, steps) -> ts -> row


def _series(profile: str) -> "_Series | None":
    """The house's own day (cool_kw, cop, t_in_c, occ) for a replay profile, None if the dataset has no cooling file."""
    import numpy as np
    with _LOCK:
        if profile in _CACHE:
            return _CACHE[profile]
        stem = profile if "/" in profile else f"home_{profile}"
        try:
            ts, cool, cop, tin, occ, mode = [], [], [], [], [], []
            with gzip.open(data_path(f"{stem}_hvac.csv.gz"), "rt") as f:
                for r in csv.DictReader(f):
                    ts.append(r["ts_local"]); cool.append(float(r["cool_kw"])); cop.append(float(r["cop"]))
                    tin.append(float(r["t_in_c"])); occ.append(int(r["occ"])); mode.append(int(r.get("mode") or 1))
            key = (ts[0], ts[-1], len(ts)) if ts else ("", "", 0)
            index = _INDEXES.get(key)
            if index is None:
                index = {t: k for k, t in enumerate(ts)}
                _INDEXES[key] = index
            rows = _Series(index, np.asarray(cool, dtype=np.float32), np.asarray(cop, dtype=np.float32),
                           np.asarray(tin, dtype=np.float32), np.asarray(occ, dtype=np.int8), np.asarray(mode, dtype=np.int8))
        except (FileNotFoundError, OSError):
            rows = None
        _CACHE[profile] = rows
        return rows


class House:
    """One home's cooling lever. `at(ts)` is the house's own day; the controller and the home move x around it."""

    def __init__(self, params: dict[str, Any], profile: str):
        self.p = params
        self.ua = float(params["ua_kw_per_c"])
        self.cap = float(params["cap_kwh_per_c"])
        self.tau = self.cap / self.ua
        self.qmax = float(params["cool_cap_kw"])                 # kW of heat it can remove at most
        self.rows = _series(profile)

    @classmethod
    def of(cls, name: str | None, profile: str) -> "House | None":
        p = params_for(name) if name else None
        if not p or not _series(profile):
            return None
        return cls(p, profile)

    def at(self, ts: str) -> tuple[float, float, float, int]:
        """(cooling kW electric, COP, indoor C, someone home) of the house's own day at this step."""
        if self.rows is None:
            return (0.0, 4.0, 24.0, 1)
        return self.rows.get(ts, (0.0, 4.0, 24.0, 1))

    def sign(self, ts: str) -> int:
        """+1 while the house is cooled, -1 while its heat pump heats it."""
        if self.rows is None or self.rows.mode is None:
            return 1
        i = self.rows.index.get(ts)
        return 1 if i is None else int(self.rows.mode[i])

    def a(self, dt_h: float) -> float:
        return math.exp(-dt_h / self.tau)

    def step(self, x: float, dq_kw_th: float, dt_h: float) -> float:
        """x after dt hours of dq kW more heat removed than the house's own thermostat removes."""
        a = self.a(dt_h)
        return a * x - (dq_kw_th / self.ua) * (1.0 - a)

    def dq_bounds(self, ts: str) -> tuple[float, float]:
        """dq can switch the cooling off (down to -q) or run it at full capacity (up to qmax - q), kW of heat."""
        cool, cop, _, _ = self.at(ts)
        q = cool * cop
        return -q, max(0.0, self.qmax - q)

    def thermostat(self, x: float, ts: str, dt_h: float) -> float:
        """The house's own thermostat with nobody steering: back to its day as fast as the capacity allows (an
        eased-off house cools back down: the rebound; a pre-cooled one coasts up)."""
        a = self.a(dt_h)
        lo, hi = self.dq_bounds(ts)
        want = self.ua * a * x / (1.0 - a) if a < 1.0 else 0.0     # dq that brings x to 0 in this step
        return max(lo, min(hi, want))

    def kw(self, dq_kw_th: float, ts: str) -> float:
        """Electricity of a change in heat removed, kW."""
        return dq_kw_th / max(1.0, self.at(ts)[1])
