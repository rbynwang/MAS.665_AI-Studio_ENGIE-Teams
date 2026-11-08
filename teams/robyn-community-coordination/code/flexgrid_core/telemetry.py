"""Telemetry health checks, run on observed readings only (never the dataset's truth columns).

Rules mirror the three fault types in the team dataset:
  frozen  - SOC stuck and the battery reads 0 kW, yet the power balance says the battery is moving:
            |grid - (solar + battery - home - EV - curtailed)| > 0.3 kW. (A truly idle battery passes;
            like the dataset's "unobservable" faults, a freeze while idle cannot be seen and does no harm.)
  jump    - SOC moves more in one 15-minute reading than the battery's power rating allows
  stale   - reading timestamp did not advance for >= 3 readings
"""
from __future__ import annotations

from typing import Any

BALANCE_TOL_KW = 0.3


def balance_residual(r: dict[str, Any]) -> float:
    expected = r.get("pv_kw", 0) + r.get("batt_kw", 0) - r.get("load_kw", 0) - r.get("ev_kw", 0) - r.get("curtailed_kw", 0)
    return r.get("grid_kw", 0) - expected


def check(history: list[dict[str, Any]], limits: dict[str, Any] | None = None) -> dict[str, Any]:
    if not history:
        return {"status": "no_data", "confidence_factor": 0.0, "detail": "no readings yet"}
    h = history[-4:]
    last = h[-1]
    res = balance_residual(last)
    if len(h) >= 2:
        prev = h[-2]
        max_step = 10.0                               # the most SOC can physically move in 15 min, plus margin
        if limits and limits.get("battery_kwh", 0) > 0:     # (no battery: SOC never moves, any step is a jump)
            max_step = max(10.0, 100 * limits["battery_kw"] * 0.25 / (limits.get("eta", 0.95) * limits["battery_kwh"]) + 2)
        if abs(last.get("soc_pct", 0) - prev.get("soc_pct", 0)) > max_step:
            return {"status": "jump", "confidence_factor": 0.5,
                    "detail": f"SOC moved {prev['soc_pct']:.1f} -> {last['soc_pct']:.1f} in one step"}
    flat = len(h) >= 2 and len({r.get("soc_pct") for r in h[-2:]}) == 1 and not (limits and limits.get("battery_kwh", 1) <= 0)
    if flat and abs(last.get("batt_kw", 0)) < 1e-6 and any(abs(balance_residual(r)) > BALANCE_TOL_KW for r in h[-2:]):
        return {"status": "frozen", "confidence_factor": 0.0,
                "detail": f"SOC stuck at {last['soc_pct']:.1f}% and battery reads 0 kW, but the power balance is off by "
                          f"{res:+.2f} kW: the battery is moving and its sensor is not"}
    if len(h) >= 3 and len({r.get("ts") for r in h[-3:]}) == 1:
        return {"status": "stale", "confidence_factor": 0.3, "detail": "timestamp not advancing"}
    if abs(res) > BALANCE_TOL_KW:
        return {"status": "inconsistent", "confidence_factor": 0.5, "detail": f"power balance off by {res:+.2f} kW"}
    return {"status": "ok", "confidence_factor": 1.0, "detail": "readings consistent (power balance closes)"}
