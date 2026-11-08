"""What a node can sell: forecasts in, kW by device out, for a sale window. Code does the arithmetic, and the model only reads the answer.

  home_window(sim, fc, start, end)   the day before: kW by device, expected, 10 % and 90 %, firm and tentative
  can_give(sim, fc, ts, sale)        before a sale: what each device can hold from now on, by see_home's device keys
  overlay(facts, sim, fc)            the same numbers written over see_home's can_give_kw
  fleet_range(rows, day, start, end) firm and tentative for a whole fleet
  offer_app(app, sell)               the devices' part of a home's offer, from these forecasts

Firm and tentative. One home's 10 % figures don't add up: on hot evenings they sum to 241 kW against 432 kW true, because homes
don't all have a bad evening together. So a home's own firm figure is the sum of its 10 %s, safe for that home alone, and a firm
sale for a fleet comes from fleet_range: the fleet's expected kW times how far the fleet's forecasts really missed on days already
over, each device on its own and summed day by day, so errors that move together stay together.

Every number reads the past only: forecasts made at 10:00 the day before, and device records of days over by now.
"""
from __future__ import annotations

import json
from functools import lru_cache
from typing import Any

import numpy as np

from .. import config
from ..home_features import cop as heat_per_kw
from .see import POOL_EXP, POOL_FLOOR, POOL_SPEED

V2H_MAX_KW = 9.6                 # the two-way kit's rating, as see_home uses it
SAME_KIND_DAYS = 14              # a device's record: this many days of the same kind
SEE_KEY = {"battery": "battery", "car_hold": "charger", "v2h": "car", "cooling": "cool", "water": "water", "dish": "dish", "dryer": "dryer", "pool": "pool"}
TABLE_KEY = {"cooling": "cooling", "water": "water", "car_hold": "car_hold", "v2h": "v2h", "dish": "appliance", "dryer": "appliance", "pool": "pool"}
FLEET_DAYS = 45                  # the fleet range reads this many past days of the same window
MIN_FLEET_DAYS = 10
RATIO_MAX = 3.0                  # one odd day must not set the range: a device's truth over forecast is clipped here
FALLBACK_SHARE = .7              # too little history: only this share of what isn't physics counts as firm
TAKES_CALL = {"charger": "charging", "car": "idle"}             # the status in which each of the car's two tools can act


def _row(expected: float, p10: float, p90: float, source: str, firm: float | None = None, **more: Any) -> dict[str, Any]:
    e = round(max(0.0, expected), 3)
    lo = round(max(0.0, min(p10, e)), 3)
    f = round(max(0.0, min(lo if firm is None else firm, e)), 3)
    return {"expected_kw": e, "p10_kw": lo, "p90_kw": round(max(p90, e), 3), "firm_kw": f, "tentative_kw": round(e - f, 3), "source": source, **more}


def _steps(sim: Any, start: str, end: str) -> list[int]:
    rp = sim.replay
    i0, i1 = rp.idx_of(start), rp.idx_of(end)
    if i0 is None or i1 is None or i1 <= i0:
        raise ValueError(f"no such window: {start} to {end}")
    return list(range(i0, i1))


def _on(d: dict[str, Any], idx: list[int], a: np.ndarray) -> np.ndarray:
    """A day's hourly values on the window's 15-minute steps."""
    return np.asarray(a)[[i // 4 - int(d["t"][0]) for i in idx]]


def past_window(sim: Any, fc: Any, key: str, i0: int, i1: int, now: int, days: int = SAME_KIND_DAYS) -> list[float]:
    """A device's mean kW in this clock window, on the last `days` days of the same kind already over at `now`."""
    ck = fc.ck
    off = bool(ck.off[i0 // 4])
    a, out = sim.dev_arr[key], []
    for d in range(1, 60):
        j0, j1 = i0 - 96 * d, i1 - 96 * d
        if j0 < 0 or len(out) >= days:
            break
        if j1 <= now and bool(ck.off[j0 // 4]) == off:
            out.append(float(np.asarray(a[sim.dev_row, j0:j1]).mean()))
    return out


def _battery(sim: Any, hours: float, live: bool) -> dict[str, Any] | None:
    if sim.E <= 0:
        return None
    if sim.P <= 0:
        return _row(0.0, 0.0, 0.0, "tripped")
    stored = (max(0.0, sim.soc - sim.R) if live else sim.SM - sim.R) * sim.E      # the day before it counts on being refilled
    kw = min(sim.P, stored * sim.eta / hours)
    return _row(kw, kw, kw, "physics", firm=kw)


def _car(sim: Any, c: dict[str, Any] | None, idx: list[int], hours: float, start: str, end: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """(hold the charging, give from the car). Expected kWh and the plugged share go up as they are: one home's 10 % is near zero."""
    if not sim.has_ev or c is None:
        return None, None
    h0, h1 = int(start[11:13]), int(end[11:13])
    whole = start[14:16] == end[14:16] == "00"                  # a window model's kWh are for its whole hours, not for part of them
    hour_of = [int(sim.replay.ts(i)[11:13]) for i in idx]
    kwh = float(sum(c["hour_kwh"][h] for h in hour_of)) * .25
    share = float(np.mean([c["hour_plugged"][h] for h in hour_of]))
    lo_kwh = hi_kwh = None
    p_charge, s_lo, s_hi = None, 0.0, 1.0
    for w in c["window"].values():
        if whole and w["hours"] == (h0, h1):
            kwh, share, lo_kwh, hi_kwh, p_charge, s_lo, s_hi = w["mean_kwh"], w["plugged_share"], w["p10_kwh"], w["p90_kwh"], w["p_charging"], w["plugged_share_p10"], w["plugged_share_p90"]
    hold = _row(kwh / hours, (lo_kwh or 0.0) / hours, (hi_kwh if hi_kwh is not None else kwh * 2) / hours, "car_window" if lo_kwh is not None else "car_hour_kwh", p_charging=p_charge)
    give = None
    if "v2h" in (sim.app or {}):
        spec = sim.app["v2h"]
        allowed = float(spec.get("event_kwh", 15.0)) * float(spec.get("eff", .9))    # the owner's kWh leave the car, and this share reaches the home
        full = min(V2H_MAX_KW, sim.ev_max, allowed / hours)
        give = _row(full * share, full * s_lo, full * s_hi, "plugged share x kit limit", plugged_share=round(share, 3))
    return hold, give


def _cooling(sim: Any, d: dict[str, Any], idx: list[int], hours: float, live: bool) -> dict[str, Any] | None:
    """Cooling eased off: the kW it runs at, for the part of the window the house can hold inside the owner's limit. In heating the same, turned around."""
    h = sim.house
    if h is None or "drift" not in d:
        return None
    ts = [sim.replay.ts(i) for i in idx]
    cop = heat_per_kw([float(sim.replay.rows[i].get("fc_temp_c", 20.0)) for i in idx], int(ts[0][5:7]) in (5, 6, 7, 8, 9))
    lim = sim.ctl.limits
    occ = float(_on(d, idx, d["plan"]["plan_occ"]).mean()) >= .5
    limit_c = (lim.max_rise_c if occ else lim.max_rise_away_c) + (0.0 if live else lim.precool_c)    # the day before, there is time to pre-cool
    out = []
    for q in (d["drift"][0], d["drift_mean"], d["drift"][2]):
        c = _on(d, idx, q)
        run = float((c * h.cap / cop).mean())
        out.append(run * min(1.0, limit_c / max(1e-6, float(c.mean()) * hours)))
    return _row(out[1], min(out), max(out), d["drift_source"], run_kw=round(float((_on(d, idx, d["drift_mean"]) * h.cap / cop).mean()), 3), cop=round(float(np.mean(cop)), 2), limit_c=round(limit_c, 2))


def _past_device(sim: Any, fc: Any, key: str, i0: int, i1: int, now: int, source: str, factor: float = 1.0) -> dict[str, Any] | None:
    v = past_window(sim, fc, key, i0, i1, now)
    if not v:
        return None
    a = np.array(v) * factor * float(sim.app[key].get("share", 1.0))
    return _row(float(a.mean()), float(np.quantile(a, .1)), float(np.quantile(a, .9)), source)


def home_window(sim: Any, fc: Any, start: str, end: str, now_idx: int | None = None, shelf: list[str] | None = None, live: bool = False) -> dict[str, Any]:
    """kW by device for one window. Raises if the day isn't forecast yet at `now_idx`. shelf: the tools this home holds (None: all it has)."""
    idx = _steps(sim, start, end)
    now = sim.replay.current() if now_idx is None else now_idx
    D = fc.d_of(start)
    if fc.d_of(sim.replay.ts(idx[-1])) != D:
        raise ValueError(f"a window stays inside one day: {start} to {end}")
    if D is None or not fc.known(now, D):
        raise ValueError(f"no forecast yet for {start} at step {now}")
    d = fc.day(D)
    if d is None:
        raise ValueError(f"not enough history for {start[:10]}")
    hours = len(idx) * .25
    has = (lambda t: True) if shelf is None else (lambda t: t in shelf)
    dev: dict[str, dict[str, Any]] = {}
    if has("set_battery"):
        dev["battery"] = _battery(sim, hours, live)
    hold, give = _car(sim, fc.cars(D), idx, hours, start, end)
    if has("limit_ev_charging"):
        dev["car_hold"] = hold
    if has("discharge_car"):
        dev["v2h"] = give
    if has("set_thermostat") or has("set_room_ac"):
        dev["cooling"] = _cooling(sim, d, idx, hours, live)
    if has("set_water_heater") and "water_kind4" in d and "hpwh" in (sim.app or {}):
        dev["water"] = _row(float(_on(d, idx, d["water_kind4"]).mean()), float(_on(d, idx, d["water"][0]).mean()), float(_on(d, idx, d["water"][2]).mean()), "plain 4-day average", model_mean_kw=round(float(_on(d, idx, d["water_mean"]).mean()), 3))
    if has("delay_appliance"):
        for k in ("dish", "dryer"):
            if k in (sim.app or {}):
                dev[k] = _past_device(sim, fc, k, idx[0], idx[-1] + 1, now, "same kind of day, last 14")
    if has("set_pool_pump") and "pool" in (sim.app or {}):
        dev["pool"] = _past_device(sim, fc, "pool", idx[0], idx[-1] + 1, now, "same kind of day, last 14", 1 - (POOL_FLOOR / POOL_SPEED) ** POOL_EXP)
    dev = {k: v for k, v in dev.items() if v is not None}
    # the meter is the last limit: sale kW can't take a home past its export limit
    load = float(_on(d, idx, d["load_mean"]).mean())
    pv = float(_on(d, idx, d["pv"][1]).mean())
    draw = dev["car_hold"]["expected_kw"] if "car_hold" in dev else 0.0
    cap = max(0.0, float(sim.export_limit) + load + draw - pv)
    total = sum(v["expected_kw"] for v in dev.values())
    if total > cap:
        over = total - cap
        for k in ("v2h", "battery"):                            # what's stored gives way first: shed load saves the meter either way
            if k in dev and over > 0:
                cut = min(over, dev[k]["expected_kw"])
                v = dev[k]
                dev[k] = _row(v["expected_kw"] - cut, min(v["p10_kw"], v["expected_kw"] - cut), v["p90_kw"], v["source"], firm=min(v["firm_kw"], v["expected_kw"] - cut), capped_by="export limit")
                over -= cut
    return {"window": [start, end], "hours": hours, "devices": dev, "load_kw": round(load, 3), "pv_kw": round(pv, 3), "export_cap_kw": round(cap, 3),
            "expected_kw": round(sum(v["expected_kw"] for v in dev.values()), 3), "firm_kw": round(sum(v["firm_kw"] for v in dev.values()), 3),
            "tentative_kw": round(sum(v["tentative_kw"] for v in dev.values()), 3), "made_at": sim.replay.ts(now)}


def can_give(sim: Any, fc: Any, ts: str, sale: dict[str, Any], shelf: list[str] | None = None) -> dict[str, float]:
    """What each device can hold from now to the end of the sale, by see_home's keys. State now where a device has one, forecasts for the rest."""
    t0 = max(ts, sale["start"])
    w = home_window(sim, fc, t0, sale["end"], sim.replay.idx_of(ts), shelf, live=True)
    dev = w["devices"]
    out = {SEE_KEY[k]: v["expected_kw"] for k, v in dev.items()}
    hours = w["hours"]
    plugged = bool(sim.has_ev and sim.ev_deadline)
    if "charger" in out and plugged:                            # the car is here, so what it draws is known
        kwh = sim.ev_left
        if sim.ev_wait:                                         # on a timer: only what falls inside the sale
            a = max(t0, sim.ev_wait["arrival"])
            kwh = min(sim.ev_wait["kwh"], sim.ev_max * sim.replay.minutes_between(a, sale["end"]) / 60.0) if a < sale["end"] else 0.0
        out["charger"] = round(min(sim.ev_max, kwh / hours), 3)
    if "car" in out and plugged:
        spec = sim.app["v2h"]
        left = max(0.0, float(spec.get("event_kwh", 15.0)) - (sim.v2h_used if sim.v2h_day == ts[:10] else 0.0))
        out["car_max_kwh"] = round(min(left, sim.v2h_room(t0)), 3)      # kWh leaving the car, as the card counts them
        out["car"] = round(min(V2H_MAX_KW, sim.ev_max, out["car_max_kwh"] * float(spec.get("eff", .9)) / hours), 3)
    if sim.overrides.get("ev_now"):                             # the owner's now: the car is out of the sale
        out.update({k: 0.0 for k in ("car", "car_max_kwh", "charger") if k in out})
    # what the home can take, as in home_window: the car and the battery together stay inside the export limit
    over = sum(out[k] for k in SEE_KEY.values() if k in out) - max(0.0, float(sim.export_limit) + w["load_kw"] + out.get("charger", 0.0) - w["pv_kw"])
    for k in ("car", "battery"):
        if k in out and over > 0:
            cut = min(over, out[k])
            out[k] = round(out[k] - cut, 3)
            over -= cut
    if "cooling" in dev:                                        # the guard's sum needs these too, or it reads the true future through them
        out["cool_run_kw"], out["cool_cop"] = dev["cooling"]["run_kw"], dev["cooling"]["cop"]
    return out


def overlay(facts: dict[str, Any], sim: Any, fc: Any) -> dict[str, Any]:
    """see_home reads the true future for cooling, the dishwasher and the dryer. This writes the forecast's numbers over them, with the ones the guard's sums read."""
    if not facts.get("sale"):
        return facts
    try:
        give = can_give(sim, fc, facts["ts"], facts["sale"], facts.get("shelf"))
    except ValueError:                                           # no forecast for this day: a sale must not stop for that
        return facts
    floor2 = lambda x: float(np.floor(x * 100 + 1e-9) / 100)     # noqa: E731
    dev = facts["devices"]
    for k, d in dev.items():
        if k in give and (k not in TAKES_CALL or d["status"] == TAKES_CALL[k]):    # a car that can't take this call keeps see_home's number, whatever tonight's forecast says
            d["can_give_kw"] = floor2(give[k])
    if "car" in dev and "car_max_kwh" in give and dev["car"]["status"] == TAKES_CALL["car"]:    # see_home's room leaves out a waiting car's own session
        dev["car"]["max_kwh"] = floor2(give["car_max_kwh"])
    if "cool" in dev and "cool_run_kw" in give:
        dev["cool"].update(run_kw=give["cool_run_kw"], cop=give["cool_cop"], running=give["cool_run_kw"] > .02)
    return facts


# --------------------------------------------------------------------------------------------------- the fleet
@lru_cache(maxsize=1)
def error_table() -> dict[str, Any]:
    """Fleet forecasts and truths by day and device, out of sample. stage2/fleet_errors.py writes it, without the benchmark's kept days."""
    meta = config.dataset_meta()
    return json.loads((config.REPO_DIR / meta.get("models_2026", "models/forecast_2026") / "fleet_errors.json").read_text())


def _weekend(day: str) -> bool:
    import datetime as dt
    return dt.date.fromisoformat(day).weekday() >= 5


def fleet_range(rows: list[dict[str, Any]], day: str, start: str, end: str, table: dict[str, Any] | None = None) -> dict[str, Any]:
    """Firm and tentative for a fleet. rows: home_window results for the same window. Reads days up to D-2 only."""
    import datetime as dt
    t = table or error_table()
    exp: dict[str, float] = {}
    for r in rows:
        for k, v in r["devices"].items():
            kw = float(v.get("expected_kw") or 0.0)
            if (k == "battery" or k in TABLE_KEY) and np.isfinite(kw) and kw > 0:      # these rows ride on offers: one odd row must not stop or poison the sum
                exp[k] = exp.get(k, 0.0) + kw
    total = sum(exp.values())
    certain = exp.get("battery", 0.0)                           # physics, not a forecast
    rest = {k: v for k, v in exp.items() if k != "battery"}
    last = str(dt.date.fromisoformat(day) - dt.timedelta(days=2))
    w = t["windows"].get(f"{start[11:16]}-{end[11:16]}")
    dates = [] if w is None else [x for x in w["dates"] if x <= last and _weekend(x) == _weekend(day)][-FLEET_DAYS:]
    out = {"day": day, "window": [start, end], "homes": len(rows), "expected_kw": round(total, 1), "by_device_kw": {k: round(v, 1) for k, v in exp.items()}, "sum_of_home_firm_kw": round(sum(r["firm_kw"] for r in rows), 1), "days_read": len(dates)}
    if len(dates) < MIN_FLEET_DAYS:
        firm = certain + FALLBACK_SHARE * sum(rest.values())
        return {**out, "p10_kw": round(firm, 1), "p90_kw": round(total, 1), "firm_kw": round(firm, 1), "tentative_kw": round(total - firm, 1), "firm_share": round(firm / max(total, 1e-9), 3), "note": "too little history: a fixed share"}
    pos = {x: i for i, x in enumerate(w["dates"])}
    sums = []
    for x in dates:
        s = certain
        for k, kw in rest.items():
            fc, tr = w[TABLE_KEY[k]][pos[x]]
            s += kw * (min(RATIO_MAX, tr / fc) if fc > 1e-6 else 1.0)
        sums.append(s)
    p10, p50, p90 = (float(np.quantile(sums, q)) for q in (.1, .5, .9))
    firm = min(p10, total)
    return {**out, "p10_kw": round(p10, 1), "p50_kw": round(p50, 1), "p90_kw": round(p90, 1), "firm_kw": round(firm, 1), "tentative_kw": round(max(0.0, total - firm), 1), "firm_share": round(firm / max(total, 1e-9), 3)}


def offer_app(app: dict[str, Any], sell: dict[str, Any], confidence: float = 0.8) -> dict[str, Any]:
    """The devices' part of a home's offer on these forecasts: rt_home.app_offer's shape, with the two-way car at what its owner's kWh allow."""
    dev = sell["devices"]
    by = {k: dev[k]["expected_kw"] for k in ("v2h", "water", "dish", "dryer", "pool") if k in dev}
    if "water" in by:
        by["hpwh"] = by.pop("water")
    m = sum(by.values())
    return {**app, "kw": round(m * confidence, 3), "mean_kw": round(m, 3), "by_device": {k: round(v, 3) for k, v in by.items()}, "from": "sellable"}
