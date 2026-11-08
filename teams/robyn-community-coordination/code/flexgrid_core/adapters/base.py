"""The one interface every device adapter implements.

Sign convention everywhere: battery kW > 0 = discharging; grid kW > 0 = exporting.
"""
from __future__ import annotations

from typing import Any, Protocol


class Adapter(Protocol):
    name: str

    def telemetry(self) -> dict[str, Any]:
        """Latest reading: ts, pv_kw, load_kw, ev_kw, batt_kw, soc_pct, grid_kw, mode, ev fields."""

    def limits(self) -> dict[str, Any]:
        """Hard physical limits: battery kWh/kW, reserve, SOC max, EV kW, export limit."""

    def apply(self, delta_kw: float, minutes: float, dispatch_id: str) -> dict[str, Any]:
        """Shift net grid power by delta_kw (+ = more export / less import) for `minutes`.
        Returns {applied_kw, battery_target_kw, ev_action, readback}. Must clamp to limits."""

    def release(self) -> None:
        """Drop any override and return to the default (self-consumption) mode."""
