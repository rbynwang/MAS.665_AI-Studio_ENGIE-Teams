"""The day-before plan, past data only: what a home can really write down at 10:00 for tomorrow.

stage2/plans2.py stays as signed off, and its files hold things no agent can know at 10:00: the usual week comes from the whole year,
snow closures come the night before, planned kWh read the whole of the day before, and a car row exists only when a session really ended.
So agents read this adapter instead. For day D it reads the home's own record up to D-2, the calendar, and the trip calendar for D.

The trip calendar stays what the family tells the plan, so trips are right by construction and the plan's gain is an upper limit.
`miss_trips` makes the plan wrong on purpose: that share of trip days never reaches it, chosen by a seeded draw per day.

Numpy only, and every node's plan has the same shape: space, start, end, state, confidence, source (MULTI_AGENT_DESIGN, section 7).
"""
from __future__ import annotations

import zlib
from typing import Any

import numpy as np

from .home_features import Clock

SLEEP = np.r_[0:26, 92:96]                 # slots counted as asleep when someone is home, as in plans2.py
STATES = {0: "empty", 1: "occupied", 2: "asleep"}


def hm(q: int) -> str:
    return f"{q // 4:02d}:{(q % 4) * 15:02d}" if q < 96 else "24:00"


def day_kinds(ck: Clock, breaks: list[list[str]], family: bool) -> np.ndarray:
    """plans2.py's day kind: 0 a normal weekday, 1 anything else. School breaks count for families."""
    out = np.zeros(ck.ND, bool)
    for a, b in breaks:
        out |= np.array([a <= d <= b for d in ck.days])
    wk = (ck.day_dow < 5) & ~ck.day_holiday
    return np.where(wk & ~(out & family), 0, 1)


def school_out(ck: Clock, breaks: list[list[str]]) -> np.ndarray:
    out = np.zeros(ck.ND, bool)
    for a, b in breaks:
        out |= np.array([a <= d <= b for d in ck.days])
    return out


class PastPlan:
    def __init__(self, ck: Clock, occ: np.ndarray, away: np.ndarray, breaks: list[list[str]], family: bool, miss_trips: float = 0.0, seed: str = ""):
        """occ: the year's 15-minute record of someone home. away: the trip calendar, one flag a day."""
        self.ck = ck
        self.grid_true = ck.grid((np.asarray(occ) > 0).astype(float))
        self.away = np.asarray(away, dtype=bool)
        self.told = self.away.copy()                           # the trips the plan was told about
        if miss_trips > 0:
            for d in np.flatnonzero(self.away):
                if zlib.crc32(f"{seed}{d}".encode()) % 1000 < miss_trips * 1000:
                    self.told[d] = False
        self.kind = day_kinds(ck, breaks, family)
        self.school = school_out(ck, breaks)
        self.c, self.cnt = {}, {}
        for k in (0, 1):                                       # running sums: position d holds days 0 .. d only
            sel = (self.kind == k) & ~self.away
            self.c[k] = np.cumsum(np.where(sel[:, None], self.grid_true, 0.0), axis=0)
            self.cnt[k] = np.cumsum(sel)
        self._occ: dict[int, np.ndarray] = {}

    def occupied(self, D: int) -> np.ndarray:
        """96 flags: someone home at each slot of day D, as planned at 10:00 on D-1. Reads the record up to D-2."""
        if D not in self._occ:
            k = int(self.kind[D])
            out = np.zeros(96, dtype=bool)
            if not self.told[D] and D >= 3 and self.cnt[k][D - 2] > 0:
                out = np.round(self.c[k][D - 2] / self.cnt[k][D - 2]) > 0
            self._occ[D] = out
        return self._occ[D]

    def _hours(self, D: int) -> np.ndarray:
        return self.occupied(D).astype("float32").reshape(24, 4).mean(axis=1)

    def _empty(self, d: int) -> float:
        return float(not self.occupied(d).any())

    def features(self, t: np.ndarray) -> dict[str, np.ndarray]:
        """The plan's inputs to the home models, for hours t of one day (stage4 homes_features.plan_hours)."""
        ck = self.ck
        D = int(ck.day_i[t[0]])
        hr = ck.hour[t]
        lag = np.stack([self._hours(max(D - k, 0))[hr] for k in (2, 3, 4)]).mean(axis=0)
        a14 = sum(self._empty(d) for d in range(max(D - 15, 0), max(D - 1, 0))) / 14
        a3 = sum(self._empty(d) for d in range(max(D - 4, 0), max(D - 1, 0))) / 3
        n = len(t)
        return {"plan_occ": self._hours(D)[hr].astype(float), "plan_occ_day": np.full(n, float(self.occupied(D).astype("float32").mean())), "plan_away": np.full(n, self._empty(D)),
                "plan_occ_lag": lag.astype(float), "plan_away14": np.full(n, a14), "plan_away3": np.full(n, a3), "school_out": np.full(n, float(self.school[D]))}

    def confidence(self, D: int) -> np.ndarray:
        """Share of the past days of this kind each slot's plan would have been right, from days over at 10:00 the day before."""
        k = int(self.kind[D])
        past = [d for d in range(max(0, D - 29), D - 1) if self.kind[d] == k and not self.away[d]]
        if not past:
            return np.full(96, .5)
        return ((self.grid_true[past] > 0) == self.occupied(D)).mean(axis=0)

    def blocks(self, D: int) -> list[dict[str, Any]]:
        """The plan for day D as the design's shape, in blocks of one state."""
        s = self.occupied(D).astype(int)
        s[SLEEP] *= 2
        conf = self.confidence(D)
        e = np.flatnonzero(np.diff(s)) + 1
        a, b = np.r_[0, e], np.r_[e, 96]
        return [{"space": "whole house", "start": hm(int(x)), "end": hm(int(y)), "state": STATES[int(s[x])], "confidence": round(float(conf[x:y].mean()), 2), "source": "own record to D-2 and the trip calendar"}
                for x, y in zip(a, b)]


def runs(on: np.ndarray, x: np.ndarray) -> list[tuple[int, int, float]]:
    """One day's device record: its runs as (first slot, slot after, kWh)."""
    s = on.astype(int)
    e = np.flatnonzero(np.diff(s)) + 1
    return [(int(a), int(b), float(x[a:b].sum()) * .25) for a, b in zip(np.r_[0, e], np.r_[e, 96]) if s[a]]


def planned_run(plan: PastPlan, x96: np.ndarray, D: int, days: int = 28, min_share: float = .4) -> dict[str, Any] | None:
    """A dryer or dishwasher tomorrow, from the same kind of day on the last `days` days over at 10:00 (plans2.py with a cut at D-2).
    x96: days x 96 of the device's kW. None when it runs on under 40 % of those days."""
    k = int(plan.kind[D])
    past = [d for d in range(max(0, D - 1 - days), D - 1) if plan.kind[d] == k and not plan.away[d]]
    if len(past) < 3 or plan.told[D]:
        return None
    seen = [runs(x96[d] > 0, x96[d]) for d in past]
    share = sum(1 for r in seen if r) / len(past)
    if share < min_share:
        return None
    rs = [r for day in seen for r in day]
    a = int(np.median([r[0] for r in rs]))
    n = int(np.median([r[1] - r[0] for r in rs]))
    return {"start": hm(a), "end": hm(min(96, a + n)), "first_slot": a, "last_slot": min(96, a + n), "kwh": float(np.median([r[2] for r in rs])), "chance": round(share, 2)}
