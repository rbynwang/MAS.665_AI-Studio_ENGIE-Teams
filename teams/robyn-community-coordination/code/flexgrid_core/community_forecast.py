"""Tomorrow's community demand, by hour, from the trained community model: the forecast a buyer agent (the utility's
demand-response desk) calls as a tool before it orders flexibility (MCP tool forecast_community_demand), and the coordinator
calls itself when a sale opens (market.py, outlook(): the buyer's side moves to a separate project, so the seller can't lean on its tool).

The utility owns the meters, so it has the community's demand history: the sum of the homes' base load (all but the
EVs, which the homes shift themselves) from the dataset's replay. Each hour's 10/50/90 % comes from that history itself, by
own_fit(): the community's last 42 days against the cooling and heating degrees in the weather forecast as issued about
48 h ahead (the replay's fc_* columns), and whether the day is a day off. Only for a dataset that has both (Massachusetts 2025-26).

models/forecast_2025_26/community_load stands in until a dataset has 21 such days, fed exactly as in training. It's
gradient-boosted trees trained on National Grid's R-1 average Massachusetts home, a community's load relative to its last
14 days. It can't lead here: this community isn't that home. Over 336 evenings its 10-90 % range held the truth on 14 %, and
the truth sat above it on 86 %. Given the model's forecast as one more input, the fit puts almost no weight on it, a median
of 0.08 against 1 for a forecast that was right, and does worse with it than without.

Two rules keep this honest. It reads only days already over at the issue time. And it reads no day the benchmark keeps: the list is
the one the fleet table leaves out (models/forecast_2026/fleet_errors.json), so no fit has seen a kept day. Without that list it
doesn't fit at all. A forecast's other inputs may read a kept day as history, as any agent's would.

    from flexgrid_core.community_forecast import forecast
    forecast("2026-07-02")  -> {"hours": [{"hour", "p10_kw", "p50_kw", "p90_kw"}...], "peak_window": {...}, ...}
"""
from __future__ import annotations

import json
from datetime import date, timedelta
from functools import lru_cache
from typing import Any

import numpy as np

from . import beckn, config
from .forecaster import HomeForecaster
from .replay import _load

FIT_DAYS = 42                  # the days each hour's fit reads
FIT_MIN = 21                   # with fewer, community_load stands in
SPREAD = 1.25                  # the fit's own days understate a new day's miss: chosen on evenings up to 29 Jun, the date-limited copies' cut-off
REACH = 1.25                   # a forecast stays within this factor of what its days metered. A few warm days in spring can't set a slope, and one ran to 14,000 times the truth
COOL_C, HEAT_C = 20.0, 10.0    # above this the homes cool, below this they heat


class _Community:
    """A replay-shaped view of the whole community: base_kw and pv_kw summed over the homes, weather from the first."""

    def __init__(self, profiles: list[str]):
        first_rows, index, _ = _load(profiles[0])
        rows = [dict(r) for r in first_rows]
        for r in rows:
            r["base_kw"], r["pv_kw"] = 0.0, 0.0
        for p in profiles:
            prow, pidx, _ = _load(p)
            for r in rows:
                j = pidx.get(r["ts_local"])
                if j is not None:
                    r["base_kw"] += prow[j]["base_kw"]
                    r["pv_kw"] += prow[j]["pv_kw"]
        self.rows, self.index = rows, index
        self.sessions: list = []

    def row(self, i: int) -> dict[str, Any]:
        return self.rows[max(0, min(i, len(self.rows) - 1))]

    def ts(self, i: int) -> str:
        return self.row(i)["ts_local"]

    def idx_of(self, ts: str) -> int | None:
        return self.index.get(ts)


@lru_cache(maxsize=4)
def _setup(members: tuple[str, ...] = ()):
    """members: the homes on the desk, by name. Empty means the whole fleet. The model works on a community's load relative to its own
    last 14 days, so it takes any group."""
    meta = config.dataset_meta()
    fleet = json.loads(config.data_path("fleet_v2.json").read_text())["homes"]
    if members:
        fleet = [h for h in fleet if h["name"] in set(members)]
    comm = _Community([h["profile"] for h in fleet])
    if "fc_temp_c" not in comm.rows[0]:
        raise ValueError(f"dataset {meta['name']} has no issued weather forecasts: no community forecast")
    fc = HomeForecaster("community", comm, {"lat": meta.get("community", {}).get("lat", 42.26),
                                           "lon": meta.get("community", {}).get("lon", -71.80), "pv_kw_dc": 0.0},
                        correct_pv=False)
    from .forecaster import _model
    fc.load_m = _model(str(config.REPO_DIR / meta.get("models", "models/forecast_2025_26") / "community_load.json.gz"))
    return comm, fc, len(fleet)


def _kept_out() -> set[str]:
    """The benchmark's kept days, as the fleet table lists them. A table of past errors holds none of them, and neither does a fit."""
    meta = config.dataset_meta()
    return set(json.loads((config.REPO_DIR / meta.get("models_2026", "models/forecast_2026") / "fleet_errors.json").read_text())["kept_out"])


def fit_days(comm: _Community, issue: str, kept: set[str]) -> list[str]:
    """The last FIT_DAYS whole days that were over at the issue time, newest first, none of them kept."""
    out, d, first = [], date.fromisoformat(issue[:10]), date.fromisoformat(comm.ts(0)[:10])
    while len(out) < FIT_DAYS and d > first:
        d -= timedelta(days=1)
        if d.isoformat() not in kept and all(comm.idx_of(f"{d} {h:02d}:00") is not None for h in range(24)):
            out.append(d.isoformat())
    return out


def _inputs(comm: _Community, holidays: set[date], day: str, hour: int) -> list[float]:
    """An hour's inputs, all known the day before: cooling and heating degrees in the issued forecast, and whether it's a day off."""
    t = comm.row(comm.idx_of(f"{day} {hour:02d}:00"))["fc_temp_c"]
    d = date.fromisoformat(day)
    return [1.0, max(t - COOL_C, 0.0), max(HEAT_C - t, 0.0), float(d.weekday() >= 5 or d in holidays)]


def _metered_kw(comm: _Community, day: str, hour: int) -> float:
    i = comm.idx_of(f"{day} {hour:02d}:00")
    return sum(comm.row(i + k)["base_kw"] for k in range(4)) / 4


def own_fit(comm: _Community, holidays: set[date], day: str, days: list[str]) -> list[dict[str, Any]]:
    """Each hour of `day` from the community's own meters: log kW on _inputs() over `days`, then the range from how far those days fell from the line, widened by SPREAD."""
    hours = []
    for h in range(24):
        x = np.array([_inputs(comm, holidays, d, h) for d in days])
        y = np.log([max(_metered_kw(comm, d, h), 1e-6) for d in days])
        b = np.linalg.lstsq(x, y, rcond=None)[0]
        lo, mid, hi = np.quantile(y - x @ b, (.1, .5, .9))
        at = float(np.clip(np.array(_inputs(comm, holidays, day, h)) @ b + mid, y.min() - np.log(REACH), y.max() + np.log(REACH)))
        hours.append({"hour": h, "p10_kw": round(float(np.exp(at + SPREAD * (lo - mid))), 1), "p50_kw": round(float(np.exp(at)), 1), "p90_kw": round(float(np.exp(at + SPREAD * (hi - mid))), 1)})
    return hours


def forecast(day: str, issued_at: str | None = None, members: list[str] | None = None) -> dict[str, Any]:
    """Hourly 10/50/90 % of the community's base demand (kW, all homes, or just `members`) on `day`, as forecast at `issued_at`
    (default 10:00 the day before, the call for offers). days_read says how many days its own fit read: 0 means community_load stood in."""
    comm, fc, n = _setup(tuple(sorted(members or ())))
    issue = issued_at or f"{beckn.add_minutes(f'{day} 00:00', -1440)[:10]} 10:00"
    now = comm.idx_of(issue)
    ts = [f"{day} {h:02d}:00" for h in range(24)]
    if now is None or any(comm.idx_of(t) is None for t in ts):
        meta = config.dataset_meta()
        raise ValueError(f"{day} (issued {issue}) is outside the data ({meta['range'][0]} to {meta['range'][1]})")
    if issue >= ts[0]:
        raise ValueError("the forecast must be issued before the day starts")
    out = fc.forecast(now, ts)
    hours = [{"hour": int(t[11:13]), "p10_kw": round(float(out[t]["load"][0]), 1), "p50_kw": round(float(out[t]["load"][1]), 1),
              "p90_kw": round(float(out[t]["load"][2]), 1)} for t in ts if t in out]
    try:
        days = fit_days(comm, issue, _kept_out())
    except (OSError, ValueError, KeyError):
        days = []                                         # no list of kept days, so no fit
    fitted = len(days) >= FIT_MIN
    if fitted:
        hours = own_fit(comm, fc.holidays, day, days)
    best = max(range(15, 19), key=lambda h0: sum(x["p50_kw"] for x in hours if h0 <= x["hour"] < h0 + 3))
    win = [x for x in hours if best <= x["hour"] < best + 3]
    return {"day": day, "issued_at": issue, "homes": n, "unit": "kW, the community's base demand (all homes, EVs excluded)",
            "hours": hours,
            "peak_window": {"start": f"{best:02d}:00", "end": f"{best + 3:02d}:00",
                            "mean_p50_kw": round(float(sum(x["p50_kw"] for x in win)) / 3, 1),
                            "why": "the 3 hours between 15:00 and 21:00 with the highest forecast demand"},
            "days_read": len(days) if fitted else 0,
            "model": (f"the community's own meters: each hour, its last {len(days)} days against the issued forecast's cooling and heating degrees and days off; "
                      f"the range from how far those days missed, widened {SPREAD:g} times" if fitted else
                      "community_load: gradient-boosted trees (10/50/90 % quantiles) trained on National Grid's R-1 "
                      "average Massachusetts home, May 2025 - Jun 2026; range widened by its rolling-test errors"),
            "information": "the community's metered demand up to the issue time and the ECMWF weather forecast as "
                           "issued about 48 h ahead; nothing from the day itself"}
