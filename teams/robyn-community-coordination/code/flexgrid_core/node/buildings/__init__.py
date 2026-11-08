"""The zone maps: each building's real tool targets, its kW share of each end use, and who is where. stage3/zones3.py writes them.

A space is one of the schedule's own: classrooms, a bank of floors, the common areas. A target is a group of spaces, or a plant, a battery, a set of ports.

plan() is the plan made the day before, the only thing about the future a node may read. true() is the true day: it stands in for the sensors, so
code reads it for the hour that has come and never ahead.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).parent
EVENT = .15        # sensors reading this far above the plan: an added event or a late meeting, not noise
BUSY = .10         # at or above this share in use a space counts as booked: stage 1's own line
KEYS = {"bld-university": "university", "bld-mall": "shopping_center", "bld-office": "office"}


@lru_cache(maxsize=1)
def _schedules() -> dict[str, Any]:
    z = np.load(HERE / "schedules.npz")
    return {"row": {k: i for i, k in enumerate(z["keys"].tolist())}, "day": {d: i for i, d in enumerate(z["days"].tolist())},
            "plan": z["plan"], "true": z["true"]}


class ZoneMap:
    def __init__(self, key: str):
        self.key = key
        self.m = json.loads((HERE / f"{key}.json").read_text())
        self.spaces: dict[str, dict[str, Any]] = self.m["spaces"]
        self.plants: dict[str, dict[str, Any]] = self.m["plants"]
        self.targets: dict[str, list[dict[str, Any]]] = self.m["targets"]
        self.limits: dict[str, Any] = self.m.get("limits") or {}
        self.surprises: list[dict[str, Any]] = []        # test hook, tools/sim_event.py: {from, to, space (None: all), share} on the true day

    def _at(self, kind: str, space: str, ts: str) -> float:
        s = _schedules()
        r = s["row"].get(f'{self.m["schedule"]}/{self.spaces[space]["schedule"]}')
        d = s["day"].get(ts[:10])
        return 1.0 if r is None or d is None else float(s[kind][r, d, int(ts[11:13])])     # no record reads as in use: the guard's "unknown counts as occupied"

    def plan(self, space: str, ts: str) -> float:
        """The share of this space in use at this hour, by the plan made the day before."""
        return self._at("plan", space, ts)

    def true(self, space: str, ts: str) -> float:
        """The share in use now, as the sensors read it. Never call this for a time still to come."""
        for e in self.surprises:
            if e["from"] <= ts < e["to"] and e.get("space") in (None, space):
                return float(e["share"])
        return self._at("true", space, ts)

    def use(self, space: str, now: str, steps: list[str]) -> float:
        """The share in use over these steps: the plan, or what the sensors read now if that is more. A clear miss reads as an event, which may last."""
        if not steps:
            return self.plan(space, now)
        read = self.true(space, now)
        event = read - self.plan(space, now) >= EVENT
        return sum(max(self.plan(space, t), read if event or t[:13] == now[:13] else 0.0) for t in steps) / len(steps)

    def state(self, space: str, ts: str) -> str:
        """free or booked at this hour, by the plan. A shopping center is open or closed."""
        use = self.plan(space, ts)
        if not self.m.get("sale_free"):
            return "open" if use > 0 else "closed"
        return "booked" if use >= BUSY else "free"

    def avoid(self, space: str, ts: str) -> str | None:
        """Why this space is off limits today: exams at the campus, Black Friday to Christmas at the mall."""
        return next((a["why"] for a in self.m.get("avoid", []) if a["from"] <= ts[:10] <= a["to"] and space in a["spaces"]), None)


@lru_cache(maxsize=None)
def _load(key: str) -> ZoneMap:
    return ZoneMap(key)


def load(key: str) -> ZoneMap:
    """A building's map, by its shelf name or its node name."""
    return _load(KEYS.get(key, key))
