"""Tomorrow's community demand, by hour, from the trained community model: the forecast a buyer agent (the utility's
demand-response desk) calls as a tool before it orders flexibility (MCP tool forecast_community_demand), and the coordinator
calls itself when a sale opens (market.py, outlook(): the buyer's side moves to a separate project, so the seller can't lean on its tool).

The utility owns the meters, so it has the community's demand history: the sum of the homes' base load (all but the
EVs, which the homes shift themselves) from the dataset's replay. The model is models/forecast_2025_26/community_load
(gradient-boosted trees trained on National Grid's R-1 average Massachusetts home, a community's load relative to its
last 14 days), fed exactly as in training: the history known at the issue time (10:00 the day before) and the weather
forecast as issued about 48 h ahead (the replay's fc_* columns). Only for a dataset that has both (Massachusetts 2025-26).

    from flexgrid_core.community_forecast import forecast
    forecast("2026-07-02")  -> {"hours": [{"hour", "p10_kw", "p50_kw", "p90_kw"}...], "peak_window": {...}, ...}
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

from . import beckn, config
from .forecaster import HomeForecaster
from .replay import _load


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


def forecast(day: str, issued_at: str | None = None, members: list[str] | None = None) -> dict[str, Any]:
    """Hourly 10/50/90 % of the community's base demand (kW, all homes, or just `members`) on `day`, as forecast at `issued_at`
    (default 10:00 the day before, the call for offers)."""
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
    best = max(range(15, 19), key=lambda h0: sum(x["p50_kw"] for x in hours if h0 <= x["hour"] < h0 + 3))
    win = [x for x in hours if best <= x["hour"] < best + 3]
    return {"day": day, "issued_at": issue, "homes": n, "unit": "kW, the community's base demand (all homes, EVs excluded)",
            "hours": hours,
            "peak_window": {"start": f"{best:02d}:00", "end": f"{best + 3:02d}:00",
                            "mean_p50_kw": round(float(sum(x["p50_kw"] for x in win)) / 3, 1),
                            "why": "the 3 hours between 15:00 and 21:00 with the highest forecast demand"},
            "model": "community_load: gradient-boosted trees (10/50/90 % quantiles) trained on National Grid's R-1 "
                     "average Massachusetts home, May 2025 - Jun 2026; range widened by its rolling-test errors",
            "information": "the community's metered demand up to the issue time and the ECMWF weather forecast as "
                           "issued about 48 h ahead; nothing from the day itself"}
