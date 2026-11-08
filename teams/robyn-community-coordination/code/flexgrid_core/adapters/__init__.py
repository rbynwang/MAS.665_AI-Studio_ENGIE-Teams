"""Device adapters. Every adapter speaks the same small interface (see base.Adapter)."""
from __future__ import annotations

from typing import Any


def make_adapter(cfg: dict[str, Any], replay=None):
    kind = cfg.get("adapter", "simulated")
    if kind == "simulated":
        from .simulated import SimulatedHome
        return SimulatedHome(cfg, replay)
    if kind == "building":                    # a building node (building.py): cooling and lighting, no battery or car
        from ..building import SimulatedBuilding
        return SimulatedBuilding(cfg, replay)
    if kind == "tesla":
        from .tesla import TeslaAdapter
        return TeslaAdapter(cfg)
    if kind == "enphase":
        from .enphase import EnphaseAdapter
        return EnphaseAdapter(cfg)
    raise ValueError(f"unknown adapter {kind!r}")
