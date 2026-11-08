"""Dataset replay clock: maps wall-clock time onto 15-minute steps of the active dataset (config.DATASET).

A profile is either a Jetson home (`tesla_calhoun` -> data/home_tesla_calhoun.csv.gz + data/ev_tesla_calhoun.csv)
or a fleet home (`homes/county_COM_03` -> data/homes/county_COM_03.csv.gz + _ev.csv). Both are built by
tools/make_homes.py from the team dataset: PV per PV-equipped home, residential load per household, one
vehicle's daily charging need (see DATA_SOURCES.md).

Clock modes, identical on every machine with an NTP-synced clock:
  * looping (loop_days > 0): step k = wall_seconds // step_seconds, wrapped every loop_days
  * event run (loop_days = 0, `at` + `at_wall`): dataset time `at` is shown at unix time `at_wall`,
    then one step every step_seconds. Every agent given the same three numbers plays the same step.
  * held (`hold`): dataset time `at` is shown until the clock is set again. The coordinator's conductor holds every clock at
    each step until every agent due there has answered, then sets the next (coordinator.py, cfg clock_waits).
"""
from __future__ import annotations

import csv
import gzip
import math
import threading
import time
from datetime import datetime
from typing import Any

from .config import data_path, dataset_meta

FMT = "%Y-%m-%d %H:%M"
_CACHE: dict[str, tuple[list, dict, list]] = {}
_LOCK = threading.Lock()

# What-if weather (the Control Room's What if panel, POST /demo/scenario on each home process): from dataset time
# `from` on, every home's solar is scaled by `pv` and its household load by `load`. It changes the physical truth of
# this process's homes only (actual and no-event twin alike: the same homes in the same weather); forecasts made
# earlier do not know it, as with a real surprise. Off (None) by default and after every demo reset.
WEATHER: dict[str, Any] = {"from": None, "pv": 1.0, "load": 1.0, "dT": 0.0, "to": None}

# Manual daily weather: the dataset's weather_days.json, one entry per date, set by hand before a run:
#   {"2026-01-15": {"dT": -6, "pv": 0.3, "load": 1.0, "known": false, "note": "cold snap"}}
# dT moves the temperature, and the home's heating or cooling moves with it (dkw_dT, kW per degree, a column of the
# year dataset). pv scales the sun, load the rest of the house. known = true means the forecast saw it coming too.
WEATHER_DAYS: dict[str, dict[str, Any]] = {}


def load_weather_days() -> dict[str, dict[str, Any]]:
    """Read the dataset's manual weather table again (empty when the file is absent)."""
    import json
    WEATHER_DAYS.clear()
    try:
        WEATHER_DAYS.update(json.loads(data_path("weather_days.json").read_text()))
    except (FileNotFoundError, OSError, ValueError):
        pass
    return dict(WEATHER_DAYS)


def set_weather(pv: float | None = None, load: float | None = None, frm: str | None = None, to: str | None = None,
                dT: float | None = None) -> dict[str, Any]:
    """Change the what-if weather from dataset time `frm` on (until `to`, if given); pv = load = 1 and dT = 0 switch
    it off. dT: degrees C warmer (cooler if negative) than measured."""
    pv = WEATHER["pv"] if pv is None else max(0.0, min(1.5, float(pv)))
    load = WEATHER["load"] if load is None else max(0.5, min(2.0, float(load)))
    dT = WEATHER.get("dT", 0.0) if dT is None else max(-15.0, min(15.0, float(dT)))
    on = abs(pv - 1.0) > 1e-6 or abs(load - 1.0) > 1e-6 or abs(dT) > 1e-6
    WEATHER.update({"pv": pv, "load": load, "dT": dT, "from": (frm or WEATHER["from"]) if on else None,
                    "to": (to if to is not None else WEATHER.get("to")) if on else None})
    return dict(WEATHER)


def set_price(factor: float | None, frm: str | None = None, to: str | None = None) -> dict[str, Any]:
    """A what-if real-time price: the measured price times `factor` from `frm` until `to` (None: off)."""
    if not factor or abs(float(factor) - 1.0) < 1e-6:
        for k in ("price", "price_from", "price_to"):
            WEATHER.pop(k, None)
    else:
        WEATHER.update({"price": max(0.0, min(20.0, float(factor))), "price_from": frm, "price_to": to})
    return dict(WEATHER)


def _load(profile: str):
    with _LOCK:
        if profile in _CACHE:
            return _CACHE[profile]
        name = dataset_meta().get("profile_alias", {}).get(profile, profile)   # a Jetson's profile in another dataset
        stem, ev_name = (name, f"{name}_ev.csv") if "/" in name else (f"home_{name}", f"ev_{name}.csv")
        with gzip.open(data_path(f"{stem}.csv.gz"), "rt") as f:
            rows = [dict(r) for r in csv.DictReader(f)]
        for r in rows:
            for k in list(r):
                if k != "ts_local":                     # every other column is a number (fc_* = forecast as issued)
                    r[k] = float(r[k]) if r.get(k) not in (None, "") else 0.0
        index: dict[str, int] = {}
        for i, r in enumerate(rows):
            index.setdefault(r["ts_local"], i)        # the repeated hour when clocks fall back keeps its first copy
        sessions = []
        with open(data_path(ev_name)) as f:
            for r in csv.DictReader(f):
                sessions.append({"arrival": r["arrival_local"], "deadline": r["deadline_local"],
                                 "kwh": float(r["kwh"]), "max_kw": float(r["max_kw"]), "plug_in": r.get("plug_in_local") or None})
        _CACHE[profile] = (rows, index, sessions)
        return _CACHE[profile]


class Replay:
    def __init__(self, profile: str, start: str, step_seconds: float, loop_days: int = 1):
        self.profile = profile
        self.rows, self.index, self.sessions = _load(profile)
        self.arrivals = {s["arrival"]: s for s in self.sessions}          # a session starts charging: at the plug-in or at a timer
        self.plugins = {s["plug_in"]: s for s in self.sessions if s.get("plug_in")}   # empty on a dataset without plug-in times
        self.at_wall: float | None = None
        # this clock's own what-if weather (an offline replay of a live run: each home under the weather its signed
        # offer declared, tools/sim_event.py); None: the process's WEATHER
        self.weather: dict[str, Any] | None = None
        self.configure(start, step_seconds, loop_days)

    def configure(self, start: str, step_seconds: float, loop_days: int = 1, at: str | None = None,
                  at_wall: float | None = None, hold: bool = False) -> None:
        """`at`: jump so that the current step shows that dataset time. With `at_wall` (unix seconds) and
        loop_days = 0 the jump is absolute, so agents configured at different moments still agree. hold: stay on that
        step until the clock is set again."""
        if start not in self.index:
            raise ValueError(f"replay start {start!r} not in dataset")
        step, loop = float(step_seconds), int(loop_days)
        i0, aw = self.index[start], None
        if at:
            if at not in self.index:
                raise ValueError(f"replay time {at!r} not in dataset")
            if at_wall is not None and loop == 0:
                i0, aw = self.index[at], float(at_wall)
            else:
                k = int(time.time() // step)
                if loop > 0:
                    k %= loop * 96
                i0 = self.index[at] - k
            start = self.ts(max(0, i0))
        # Looping replays are anchored to the wall clock (not to process start), so every machine with
        # an NTP-synced clock and the same settings plays the same dataset step at the same moment.
        wall0 = aw if aw is not None else (0.0 if loop > 0 else time.time())
        # one assignment, so a thread reading the clock while it changes (a change of speed mid-run) sees the old
        # line or the new one, never half of each
        self._line = (wall0, step, i0, loop, i0 if hold else None)
        self.start, self.step_seconds, self.loop_days, self.i0, self.at_wall, self.wall0 = start, step, loop, i0, aw, wall0

    @property
    def held(self) -> bool:
        return self._line[4] is not None

    def current(self) -> int:
        wall0, step, i0, loop, held = self._line
        if held is not None:
            return max(0, min(held, len(self.rows) - 1))
        k = int((time.time() - wall0) // step)
        if loop > 0:
            k %= loop * 96
        return max(0, min(i0 + k, len(self.rows) - 1))

    def retime_plan(self, new: float, now: float | None = None) -> dict[str, Any]:
        """How to change this running clock's speed (an anchored live run) on every agent at once, none stepping back.
        Each agent's clock is a straight line from wall time to steps; the change draws a new line through a step
        boundary T of the current one. Speeding up, the agents are told less than one new step before T: told early,
        an agent still shows the current step until T; told late, it is behind for a moment. Slowing down, they are
        told up to 2 s before T: an agent still on the fast line after T would play on and then step back.
        Returns {"send": wall time to tell them, "start", "at", "at_wall": T, "step_seconds": new, "loop_days": 0}."""
        now = time.time() if now is None else now
        wall0, old, i0, _, _ = self._line
        lead = 0.7 * new if new < old else min(2.0, 0.9 * new)
        T = wall0 + (math.floor((now - wall0) / old) + 1) * old
        while T - lead < now + 0.05:
            T += old
        at = self.ts(i0 + int(round((T - wall0) / old)))
        return {"send": T - lead, "start": at, "at": at, "at_wall": T, "step_seconds": float(new), "loop_days": 0}

    def clock(self) -> dict[str, Any]:
        return {"start": self.start, "step_seconds": self.step_seconds, "loop_days": self.loop_days,
                "at_wall": self.at_wall, "now": self.ts(self.current()), "held": self.held}

    def row(self, i: int) -> dict[str, Any]:
        r = self.rows[max(0, min(i, len(self.rows) - 1))]
        w = self.weather if self.weather is not None else WEATHER
        ts = r["ts_local"]
        d = WEATHER_DAYS.get(ts[:10])
        if d:
            dT = float(d.get("dT", 0.0))
            r = {**r, "pv_kw": r["pv_kw"] * float(d.get("pv", 1.0)), "ghi_wm2": r.get("ghi_wm2", 0.0) * float(d.get("pv", 1.0)), "temp_c": r.get("temp_c", 0.0) + dT,
                 "base_kw": max(0.0, r["base_kw"] * float(d.get("load", 1.0)) + r.get("dkw_dT", 0.0) * dT)}
            if d.get("known"):
                r["fc_temp_c"] = r.get("fc_temp_c", 0.0) + dT
        if w.get("from") and ts >= w["from"] and (not w.get("to") or ts < w["to"]):
            r = {**r, "pv_kw": r["pv_kw"] * w.get("pv", 1.0), "base_kw": r["base_kw"] * w.get("load", 1.0)}
            if w.get("dT"):
                r["temp_c"] = r.get("temp_c", 0.0) + w["dT"]
        if w.get("price_from") and w["price_from"] <= ts < (w.get("price_to") or "9999"):
            r = {**r, "price_usd_mwh": r["price_usd_mwh"] * w.get("price", 1.0)}
        return r

    def ts(self, i: int) -> str:
        return self.row(i)["ts_local"]

    def idx_of(self, ts: str) -> int | None:
        return self.index.get(ts)

    def session_starting(self, i: int) -> dict[str, Any] | None:
        return self.arrivals.get(self.ts(i))

    def session_plugging(self, i: int) -> dict[str, Any] | None:
        """The session whose car is plugged in at this step. A timer home's car waits here until its session starts."""
        return self.plugins.get(self.ts(i))

    def columns(self, names: tuple[str, ...]) -> dict[str, Any]:
        """The year as numpy columns, for the agents' forecasts: the same values row() gives under the manual weather table.
        A one-clock what-if (self.weather, WEATHER) is left out: it starts after any day-ahead forecast is made."""
        import numpy as np
        out = {k: np.fromiter((r[k] for r in self.rows), dtype=float, count=len(self.rows)) if k in self.rows[0] else np.full(len(self.rows), np.nan) for k in names}
        if WEATHER_DAYS:
            for i, r in enumerate(self.rows):
                if r["ts_local"][:10] in WEATHER_DAYS:
                    x = self.row(i)
                    for k in names:
                        out[k][i] = x.get(k, np.nan)
        return out

    @staticmethod
    def minutes_between(a: str, b: str) -> float:
        return (datetime.strptime(b, FMT) - datetime.strptime(a, FMT)).total_seconds() / 60.0
