"""What a home's agent is told at a decision point. Code does all the arithmetic here, so the model gets ready numbers.

see_home() reads the simulated home the way a real node reads its devices. render() writes it in the short sentences the
tool tests used on the real model, so the wording stays the wording that was tested.
"""
from __future__ import annotations

import math
from typing import Any

from . import tools as T
from ..adapters.simulated import DONE_BY
from ..rt_control import person_limit

COLD_F = 40.0                        # at or below this, the pool pump's freeze protection comes first
POOL_SPEED, POOL_FLOOR = 60, 33      # percent: the running speed and the lowest filter speed
POOL_EXP = 2.57                      # kW goes as speed to this power (each pump's two ENERGY STAR test points)
MACHINE_TARGET = {"dish": "dishwasher", "dryer": "dryer"}
MACHINE_NAME = {"dish": "Dishwasher", "dryer": "Dryer"}


def hhmm(ts: str) -> str:
    return ts[11:16]


def floor2(x: float) -> float:
    return math.floor(max(0.0, x) * 100 + 1e-9) / 100


def window_mean(sim: Any, key: str, start: str, end: str, days: int = 14) -> float:
    """The device's mean kW in this clock window on the last `days` days: only what has already happened."""
    rp = sim.replay
    w0, w1 = rp.idx_of(start), rp.idx_of(end)
    if w0 is None or w1 is None or key not in sim.dev_arr:
        return 0.0
    a = sim.dev_arr[key]
    v = [float(a[sim.dev_row, w0 - 96 * d:w1 - 96 * d].mean()) for d in range(1, days + 1) if w0 - 96 * d >= 0]
    return sum(v) / len(v) if v else 0.0


def see_home(sim: Any, ts: str, sale: dict[str, Any] | None, shelf: list[str]) -> dict[str, Any]:
    rp = sim.replay
    i = rp.idx_of(ts)
    row = rp.row(i)
    start = sale["start"] if sale else ts
    end = sale["end"] if sale else ts
    t0 = max(ts, start)
    hours = max(0.25, rp.minutes_between(t0, end) / 60.0) if sale else 0.25
    lev = sim.levers or {}
    dev: dict[str, dict[str, Any]] = {}
    lowered: dict[str, str] = {}             # tool -> what a person lowered, in words: the prompt, the tool cards and the site status all say it

    if "set_battery" in shelf and sim.E > 0:
        stored = max(0.0, sim.soc - sim.R) * sim.E
        tripped = sim.P <= 0
        give = 0.0 if tripped else floor2(min(sim.P, stored * sim.eta / hours))
        dev["battery"] = {"tool": "set_battery", "target": "home-battery", "can_give_kw": give, "tripped": tripped,
                          "soc_pct": round(sim.soc * 100), "full": sim.soc >= sim.SM - 0.02, "at_reserve": stored <= 0.01,
                          "charge_limit_kw": 0.0 if tripped else floor2(min(sim.P, sim.P_ch)),
                          "giving": float(lev.get("battery_kw", 0.0)) > 0}

    plugged = bool(sim.has_ev and sim.ev_deadline)
    if plugged and sim.ev_left > 0.01:
        car = "charging"
    elif plugged:
        car = "idle"
    else:
        car = "away"
    if lev.get("v2h_kw"):
        car = "giving"
    elif lev.get("ev") and plugged:
        car = "capped" if lev.get("ev_kw") else "idle"          # the card: a car paused by this sale shows idle, and an idle car can give
    if sim.overrides.get("ev_now"):
        car = "reserved"
    leaves = hhmm(sim.ev_deadline) if plugged else None
    charge_kw = min(sim.ev_max, sim.ev_left / 0.25) if car == "charging" else 0.0
    keep_h = sim.ctl.limits.ev_keep_h
    to_leave_h = rp.minutes_between(t0, sim.ev_deadline) / 60.0 if plugged else 0.0

    if "limit_ev_charging" in shelf and sim.has_ev:
        need_h = sim.ev_left / max(0.1, sim.ev_max)
        room_h = to_leave_h - keep_h - need_h                     # hours it can wait and still be ready
        dev["charger"] = {"tool": "limit_ev_charging", "target": "car-charger", "status": car, "charge_kw": round(charge_kw, 2),
                          "can_give_kw": floor2(charge_kw), "leaves": leaves, "wait_h": round(room_h, 2),
                          "max_amps": int(round(sim.ev_max * 1000 / 240))}

    if "discharge_car" in shelf and "v2h" in sim.app:
        spec = sim.app["v2h"]
        eff = float(spec.get("eff", 0.9))
        used = sim.v2h_used if sim.v2h_day == ts[:10] else 0.0
        left = max(0.0, float(spec.get("event_kwh", 15.0)) - used)
        room = max(0.0, sim.ev_max * eff * max(0.0, to_leave_h - keep_h) - sim.ev_left) * eff
        max_kwh = floor2(min(left, room))
        give = floor2(min(9.6, sim.ev_max, max_kwh / hours)) if plugged else 0.0
        dev["car"] = {"tool": "discharge_car", "target": "car", "status": car, "can_give_kw": give, "max_kwh": max_kwh, "leaves": leaves}

    tool = "set_thermostat" if "set_thermostat" in shelf else "set_room_ac" if "set_room_ac" in shelf else None
    if tool and sim.house:
        h = sim.house
        steps = [rp.ts(k) for k in range(rp.idx_of(t0), rp.idx_of(end))] if sale else [ts]
        at = [h.at(x) for x in steps] or [h.at(ts)]
        run = sum(a[0] for a in at) / len(at)
        cop = sum(a[1] for a in at) / len(at)
        occ = any(a[3] for a in at)                               # anyone home at any point in the window: a house warmed for an empty home was over the limit when they came back
        key = "max_rise_c" if occ else "max_rise_away_c"
        said = person_limit(sim, key)
        limit_c = round(said, 2)
        if said < getattr(sim.ctl.limits, key):
            lowered[tool] = f"The owner's limit is {limit_c:.2f} C now, not {getattr(sim.ctl.limits, key):.2f}."
        share = min(1.0, limit_c * h.cap / max(1e-6, run * cop * hours))
        dev["cool"] = {"tool": tool, "target": "main-thermostat" if tool == "set_thermostat" else "all-room-acs",
                       "mode": "cooling" if h.sign(t0) > 0 else "heating", "running": run > 0.02, "run_kw": round(run, 3), "cop": round(cop, 2),
                       "can_give_kw": floor2(run * share), "limit_c": limit_c, "cap": h.cap, "hours": hours}

    left_out = sim.dev_off | sim.kept_out                         # the offer left these out, or a person did: the controller won't touch them, so a call would give nothing
    if "set_water_heater" in shelf and "hpwh" in sim.app:
        give = floor2(window_mean(sim, "hpwh", start, end)) if sale and "hpwh" not in left_out else 0.0
        dev["water"] = {"tool": "set_water_heater", "target": "water-heater", "can_give_kw": give, "valve": True, "tank_f": 128, "normal_f": 128, "hot_ok": True, "left_out": "hpwh" in left_out}

    if "delay_appliance" in shelf:
        j = rp.idx_of(t0)
        w1 = rp.idx_of(end) if sale else j
        for k in ("dish", "dryer"):
            if k in sim.app and j is not None and w1 is not None and w1 > j:
                a = sim.dev_arr[k][sim.dev_row, j:w1] * float(sim.app[k].get("share", 1.0))
                soon = float(sim.dev_arr[k][sim.dev_row, j:j + 2].max()) > 0     # loaded and armed: it reads waiting now
                if soon and k not in sim.dev_off and k not in sim.kept_out:      # not the agent's levers: a closed machine is what its call opens
                    dev[k] = {"tool": "delay_appliance", "target": MACHINE_TARGET[k], "name": MACHINE_NAME[k], "status": "waiting",
                              "can_give_kw": floor2(float(a.sum()) * 0.25 / hours), "done_by": DONE_BY[k], "cycle_min": 75 if k == "dryer" else 150}

    if "set_pool_pump" in shelf and "pool" in sim.app:
        kw = window_mean(sim, "pool", start, end) if sale else 0.0
        air_f = float(row.get("temp_c", 20.0)) * 9 / 5 + 32
        dev["pool"] = {"tool": "set_pool_pump", "target": "pool-pump", "can_give_kw": 0.0 if "pool" in left_out else floor2(kw * (1 - (POOL_FLOOR / POOL_SPEED) ** POOL_EXP)), "run_kw": round(kw, 3),
                       "running": kw > 0.02, "speed": POOL_SPEED, "floor": POOL_FLOOR, "air_f": round(air_f), "cold": air_f <= COLD_F, "filter_ok": True, "left_out": "pool" in left_out}

    temp = float(row.get("temp_c", 20.0))
    wk = "weekend" if rp.row(i)["ts_local"][:10] and _weekend(ts) else "weekday"
    label = ("Hot " if temp >= 27 else "Cold " if temp <= 5 else "Mild ") + wk
    return {"node": getattr(sim, "cfg", {}).get("name"), "side": "home", "ts": ts, "now": hhmm(ts), "sale": sale, "owed_kw": 0.0 if not sale else float(sale["owed_kw"]),
            "day": label, "alarm": sim.overrides.get("alarm"), "stopped": bool(sim.overrides.get("stop")), "devices": dev, "shelf": list(shelf), "lowered": lowered}


def _weekend(ts: str) -> bool:
    import datetime as dt
    return dt.date.fromisoformat(ts[:10]).weekday() >= 5


def line(k: str, d: dict[str, Any]) -> str:
    """One device in the tested wording."""
    if k == "battery":
        if d["tripped"]:
            return "home-battery shut off and can give 0 kW."
        if d["at_reserve"]:
            return "home-battery is at the owner's reserve and can give 0 kW."
        how = "full" if d["full"] else f'{d["soc_pct"]}% full'
        return f'home-battery is {how} and {"discharging" if d["giving"] else "idle"} and can give {d["can_give_kw"]:.2f} kW.'
    if k == "charger":
        if d["status"] == "charging":
            return f'Car status: charging at {d["charge_kw"]:.1f} kW. can_give_kw {d["can_give_kw"]:.1f}. It leaves at {d["leaves"]}.'
        return f'Car status: {d["status"]}, 0 kW.' + (f' It leaves at {d["leaves"]}.' if d["leaves"] else "")
    if k == "car":
        if d["status"] == "idle":
            return f'Car status: idle, 0 kW. can_give_kw {d["can_give_kw"]:.1f}. max_kwh up to {d["max_kwh"]:.1f}.' + (f' It leaves at {d["leaves"]}.' if d["leaves"] else "")
        return ""                                   # the charger's line already says it
    if k == "cool":
        if not d["running"]:
            return f'{d["target"]} is off and can give 0 kW.'
        return f'{d["target"]} {"is" if d["tool"] == "set_thermostat" else "are"} {d["mode"]} and can give {d["can_give_kw"]:.2f} kW. This home\'s limit is {d["limit_c"]:.2f} C.'
    if k == "water":
        return f'Water heater: valve confirmed, can_give_kw {d["can_give_kw"]:.2f}, tank hot at {d["tank_f"]} °F, hot water {"OK" if d["hot_ok"] else "low"}.'
    if k in ("dish", "dryer"):
        s = f'{d["name"]}: waiting to start, remote start on,'
        if k == "dryer":
            s += f' {d["cycle_min"]}-minute cycle,'
        return s + f' can_give_kw {d["can_give_kw"]:.2f}. Owner\'s done-by time {d["done_by"]}.'
    if k == "pool":
        air = "air reads cold, " if d["cold"] else ""
        return f'Pool pump: running at {d["speed"]} percent, can_give_kw {d["can_give_kw"]:.2f}, air {d["air_f"]} °F, {air}filter hours OK.'
    return ""


def render(f: dict[str, Any], owed: float, done: list[str], why: str | None, block: str | None, used: set[str]) -> str:
    sale = f["sale"]
    out = [f'{f["day"]}, {f["now"]}.']
    if sale:
        out.append(f'Sale window {hhmm(sale["start"])} to {hhmm(sale["end"])}.')
    out.append(f"This home still owes {owed:.1f} kW." if owed > 0.05 else "0 kW are owed.")
    if why:
        out.append(f"What changed: {why}")
    out += list((f.get("lowered") or {}).values())
    out += done
    rank = sorted(f["devices"].items(), key=lambda kv: -kv[1].get("can_give_kw", 0.0))
    for k, d in rank:
        if d["tool"] not in used and can_act(f, d):
            s = line(k, d)
            if s:
                out.append(s)
    out.append("No alarms." if not f.get("alarm") else f'Alarm: {f["alarm"]}.')
    if block:
        out.append(block)
    return " ".join(out)


def can_act(f: dict[str, Any], d: dict[str, Any]) -> bool:
    """Worth showing now: it has kW to give."""
    return d.get("can_give_kw", 0.0) > 0


def shortlist(f: dict[str, Any], used: set[str], reads: bool = True, limit: int = 8) -> list[dict[str, Any]]:
    """At most 8 device tools: what this node has, that can act now, largest kW first, then read_device and no_action."""
    rank = sorted(f["devices"].values(), key=lambda d: -d.get("can_give_kw", 0.0))
    seen: dict[str, list[str]] = {}
    for d in rank:
        if d["tool"] not in used and can_act(f, d):
            seen.setdefault(d["tool"], []).append(d["target"])
    fits: dict[str, dict[str, list[str]]] = {}
    w = f["devices"].get("water")
    if w and (w["tank_f"] >= w["normal_f"] or f["sale"]):     # load_up only fits a below-normal tank before a sale
        fits["set_water_heater"] = {"action": ["hold"]}
    out = [T.spec(f["side"], t, tg, fits.get(t)) for t, tg in list(seen.items())[:limit]]
    out.append(T.spec(f["side"], "no_action"))
    if reads:
        out.append(T.spec(f["side"], "read_device", [d["target"] for d in f["devices"].values()] or None))
    return out
