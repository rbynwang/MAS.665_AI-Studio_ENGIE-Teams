"""What a building's agent is told at a decision point. Code does all the arithmetic here, so the model gets ready numbers.

see_building() reads the simulated building and its zone map the way a real node reads its BMS and calendar. render() writes it in the short sentences
the card tests used on the real model. shortlist() is SPEC section 5: only what can act now, largest kW first.

Each tool has one whole-building target and, where a space can differ, one per space. The agent sees the whole one unless a space must be left out.

Count kW once (SPEC section 11): where a plant tool can act in the window, plant-served zones show 0 kW. Rooftop and tenant units still show theirs.
Zones at 0 kW stay on the list while a plant waits for them, because a cap on zones that can't coast would only warm the water.
"""
from __future__ import annotations

import datetime as dt
import math
from typing import Any

from . import tools as T
from .building_guard import EMPTY_C, OPEN_C, RECOVER_MIN, zone_kw
from .see import floor2, hhmm, window_mean

SESSION_KW = 5.4          # one EV session at the shopping center: 10.8 kW over two
EASED_SHARE = .5          # zones count as set to coast once this share of the cooling has an offset
PRE_RAN_C = .3            # zones this far under normal: pre-cool ran, which a stop needs (its card's before_the_call)
HOT_C = 27.0          # from this forecast evening temperature the plant isn't stopped: see_building


def steps_of(sim: Any, t0: str, end: str) -> list[str]:
    rp = sim.replay
    a, b = rp.idx_of(t0), rp.idx_of(end)
    return [rp.ts(k) for k in range(a, b)] if a is not None and b is not None else []


def own_cooling(sim: Any, ts: str, steps: list[str]) -> float:
    """Mean cooling kW the building would run in these steps: the controller's own forecast, from what it knows now."""
    if not steps:
        return float(sim.house.at(ts)[0])
    i = sim.replay.idx_of(ts)
    seq = [ts] + [t for t in steps if t != ts]
    cool = sim.ctl._cooling(i, seq, [1e9] * len(seq))[0]
    part = cool[1:] if steps[0] != ts else cool
    return sum(part) / max(1, len(part))


def free_shares(zm: Any, ts: str, steps: list[str]) -> dict[str, float]:
    """The share of each space with nobody in it over these steps. A shopping center's spaces count as free only when it is closed."""
    out = {}
    for s, sp in zm.spaces.items():
        use = zm.use(s, ts, steps)
        if zm.m["sale_free"] == 0:
            out[s] = 0.0 if any(zm.plan(s, t) > 0 for t in (steps or [ts])) else 1.0
        else:
            out[s] = 1.0 if sp.get("closed") else max(0.0, 1.0 - use)
    return out


def rise_allowed(free: float, offset_c: float) -> float:
    """Rooms with no booking take the offset, the rest +1.5 C at most."""
    return free * min(offset_c, EMPTY_C) + (1 - free) * min(offset_c, OPEN_C)


def comfort_c(zm: Any, zones: dict[str, float], free: dict[str, float]) -> float:
    """The mean rise the building may reach: its one thermal node stands for all its zones, each by its share of the cooling."""
    return sum(zm.spaces[s]["share"]["cool"] * rise_allowed(free[s], c) for s, c in zones.items() if c > 0)


def see_building(sim: Any, ts: str, sale: dict[str, Any] | None, shelf: list[str], zm: Any, owed: float | None = None) -> dict[str, Any]:
    rp = sim.replay
    row = rp.row(rp.idx_of(ts))
    start, end = (sale["start"], sale["end"]) if sale else (ts, ts)
    t0 = max(ts, start)
    steps = steps_of(sim, t0, end) if sale else []
    hours = max(0.25, rp.minutes_between(t0, end) / 60.0) if sale else 0.25
    lev = sim.levers or {}
    zones, standby, plants = lev.get("zones") or {}, lev.get("standby") or set(), lev.get("plants") or {}
    free = free_shares(zm, ts, steps)
    temp = float(row.get("temp_c", 20.0))
    h = sim.house
    heating = any(h.sign(t) < 0 for t in (steps[:1] + steps[-1:] or [ts]))     # where heat leads, no cooling line
    C = 0.0 if heating else own_cooling(sim, ts, steps)
    cooling = C > 1.0
    pre_c = max(0.0, -sim.x)
    # A stop can't be turned down: on a hot evening it gave 683 kW for an hour against 427 owed, the plant came back and the office ended short. Eased zones under a cap held 427 flat
    hot = max([float(rp.row(rp.idx_of(s)).get("fc_temp_c", temp)) for s in steps], default=temp) >= HOT_C
    base_c = float(h.at(ts)[2])
    sale_free = float(zm.m.get("sale_free", 0.0))
    cool_eased = sum(zm.spaces[s]["share"]["cool"] for s, c in zones.items() if c > 0)
    coast = cool_eased >= EASED_SHARE or any(p.get("stop") for p in plants.values())
    flags = {s: _exception(zm, s, ts) for s in zm.spaces}
    dev: dict[str, dict[str, Any]] = {}

    def share(spaces: list[str], use: str) -> float:
        return sum(zm.spaces[s]["share"][use] for s in spaces)

    def free_of(spaces: list[str], use: str = "cool") -> float:
        w = share(spaces, use) or 1.0
        return sum(zm.spaces[s]["share"][use] * free[s] for s in spaces) / w

    def scale_of(spaces: list[str]) -> float:
        """A sale counts free rooms: fewer free than the sale assumed means less to give, as in the winter sheet."""
        return 1.0 if sale_free <= 0 else min(1.0, free_of(spaces, "light") / sale_free)

    def shown(tool: str) -> dict[str, bool]:
        """Which targets show: the whole-building one unless a space must be left out. Calls already made hide what they cover."""
        items = zm.targets.get(tool, [])
        held = set(_held(tool, lev))
        has_whole = any(x.get("whole") for x in items)
        bad = {s for s, v in flags.items() if v and not (tool == "dim_lights" and set(v) == {"smoke_ok"})}     # a fan's smoke role doesn't touch the lights
        if tool == "set_zone_standby":
            bad |= set(lev.get("events") or ())                 # an event is on there, so nobody asks that zone to stand by
        clean = has_whole and not held and not bad & {s for x in items for s in x.get("spaces", [])}
        return {x["target"]: bool(x.get("whole")) if clean else (not x.get("whole") and not (held | bad) & set(x["spaces"])) for x in items}

    # ---------------------------------------------------------------- the plant
    mix_zone = {"cool_kw": C, "cap": h.cap, "hours": hours, "pre_c": pre_c, "free_share": free_of(list(zm.spaces))}
    rise = max(zones.values(), default=0.0) or (EMPTY_C if sale_free else OPEN_C)
    mix = zone_kw(mix_zone, rise) / C if cooling else 0.0
    info: dict[str, dict[str, Any]] = {}
    for t, p in zm.plants.items():
        kw_cool, lab = C * p["cool_share"], float(p.get("lab_share", 0.0))
        plant_kw = kw_cool / (1 - lab)
        now = min(100.0, 100.0 * plant_kw / p["full_kw"])
        lab_floor = math.ceil(now * lab * (1 + float(p.get("lab_margin", .10)))) if lab else 0
        best = max([c["floor"] for c in p["chillers"] if c.get("running", True)] + [lab_floor])
        info[t] = {"p": p, "plant_kw": plant_kw, "now": now, "best": best, "sold": kw_cool * mix, "lab": lab, "called": t in plants,
                   "stop_ok": "stop_plant_early" in shelf and bool(p.get("stop")) and bool(p.get("loop_list")) and cooling and pre_c >= PRE_RAN_C and not hot,
                   "cap_kw": max(0.0, min(plant_kw * (1 - best / max(now, 1e-6)), kw_cool * mix)) if cooling else 0.0}
    covers = sum(i["p"]["cool_share"] for i in info.values() if i["called"] or i["stop_ok"] or i["cap_kw"] > 0)
    covered = min(1.0, covers / (sum(p["cool_share"] for p in zm.plants.values()) or 1.0))
    for x in zm.targets.get("stop_plant_early", []):
        t = x["target"]
        i = info[t]
        p = i["p"]
        dev[f"stop_plant_early:{t}"] = {
            "tool": "stop_plant_early", "target": t, "running": cooling, "plant_kw": i["plant_kw"], "chillers": p["chillers"], "cleared": bool(p.get("loop_list")), "normal_off": p.get("normal_off"), "pre_ran": pre_c >= PRE_RAN_C,
            "labs": i["lab"] > 0, "shared": not p.get("stop"), "can_give_kw": 0.0 if i["called"] or not i["stop_ok"] else floor2(i["sold"]), "show": i["stop_ok"] and not i["called"]}
    for x in zm.targets.get("set_plant_demand_limit", []):
        t = x["target"]
        i = info[t]
        give = 0.0 if i["called"] or i["stop_ok"] else i["cap_kw"]
        want = give if owed is None else min(give, max(0.0, owed))
        now = i["now"]
        pct = max(i["best"], math.ceil(now * (1 - want / max(i["plant_kw"], 1e-6)) - 1e-9))
        dev[f"set_plant_demand_limit:{t}"] = {
            "tool": "set_plant_demand_limit", "target": t, "running": cooling, "plant_kw": i["plant_kw"], "percent_now": round(now, 1), "chillers": i["p"]["chillers"], "lab_share": i["lab"],
            "lab_margin": i["p"].get("lab_margin", .10), "lab_live": True, "zones_coast": coast, "cap_percent": pct, "gives_kw": floor2(i["plant_kw"] * (1 - pct / max(now, 1e-6))),
            "can_give_kw": floor2(give), "floor_percent": i["best"], "ease_with": _ease(zm, shown("set_zone_cooling")), "show": give > 0 and coast and pct < now}
    plant_waits = any(d["tool"] == "set_plant_demand_limit" and d["can_give_kw"] > 0 and not coast for d in dev.values())

    # ---------------------------------------------------------------- zones
    if "set_zone_cooling" in shelf:
        show = shown("set_zone_cooling")
        for x in zm.targets["set_zone_cooling"]:
            sp = x["spaces"]
            t = x["target"]
            w = share(sp, "cool")
            fr = free_of(sp)
            state = "free" if fr >= .999 else "open" if sale_free <= 0 else "booked" if fr <= .001 else "part"
            off = _off_plant(zm, sp, w)
            d = {"tool": "set_zone_cooling", "target": t, "spaces": sp, "points": sp, "running": cooling, "state": state, "free_share": round(fr, 3), "cool_kw": C * w, "cap": h.cap * sum(zm.spaces[s]["area"] for s in sp),
                 "hours": hours, "pre_c": pre_c, "plant_covers": covered > 0, "plant_part": (1 - off) * covered, "base_c": base_c, "top_c": zm.limits.get("top_c"),
                 "top_says": "Store and mall zones stop" if sale_free <= 0 else "Occupied zones stop", "precool_c": zm.limits.get("precool_c", 1.0),
                 "tenant_fan_kw": _tenant_fans(sim, zm, sp, start, end), "next_booking": _next_booking(zm, sp, steps, end) if state == "free" else None, **_flag(flags, sp),
                 **_hall(zm, sp, ts, base_c)}
            limit = EMPTY_C if state in ("free", "part") else OPEN_C
            kw = zone_kw(d, limit) * (1 - d["plant_part"]) + d["tenant_fan_kw"] if cooling else 0.0
            d.update({"can_give_kw": floor2(kw), "unlocks": plant_waits, "show": bool(show.get(t)) and cooling})
            dev[f"set_zone_cooling:{t}"] = d

    # ---------------------------------------------------------------- fans and standby
    fans_kw = window_mean(sim, "fans", start, end) if sale else 0.0
    if "limit_air_handler" in shelf:
        for x in zm.targets["limit_air_handler"]:
            sp = x.get("spaces") or list(zm.spaces)
            sc = scale_of(sp)
            kw_now = fans_kw * x["kw_share"]
            now, floor = x["speed_now"], x["floor_percent"]
            done = bool((lev.get("fans") or {}).get(x["target"]))
            eased = not (cooling or "set_zone_standby" in shelf) or bool(set(sp) & (set(zones) | set(standby)))
            dev[f'limit_air_handler:{x["target"]}'] = {
                "tool": "limit_air_handler", "target": x["target"], "spaces": sp, "kw_now": kw_now * sc, "kw_shown": kw_now, "speed_now": now, "floor_percent": floor, "smoke_ok": x.get("smoke_ok", True),
                "can_give_kw": 0.0 if done else floor2(kw_now * sc * (1 - (floor / now) ** 3)), "eased": eased, "ease_with": _ease_fans(zm, shown("set_zone_standby"), shown("set_zone_cooling") if cooling else {}), "show": not done and not any(flags[s] for s in sp),
                "floor_why": "Below this speed booked zones lose their air.", **_flag(flags, sp)}
    if "set_zone_standby" in shelf:
        show = shown("set_zone_standby")
        fans_live = any(d["tool"] == "limit_air_handler" and d["show"] for d in dev.values())
        fans_wait = any(d["tool"] == "limit_air_handler" and d["show"] and not d["eased"] for d in dev.values())
        for x in zm.targets["set_zone_standby"]:
            sp = x["spaces"]
            closed = all(zm.spaces[s].get("closed") for s in sp)
            events = set(sp) & set(lev.get("events") or ())
            hit = events & set(standby) if not x.get("whole") else set()    # an event in a zone on standby: it can go back
            listed = bool(show.get(x["target"]))
            fr = free_of(sp)
            dev[f'set_zone_standby:{x["target"]}'] = {
                "tool": "set_zone_standby", "target": x["target"], "spaces": sp, "points": sp, "can_give_kw": 0.0, "free_share": round(fr, 3), "closed": closed, "sensors": True, "standby_ok": True,
                "next_booking": _next_booking(zm, sp, steps, end) if fr >= .999 and sale else None, "booked": fr <= .001, "event_here": bool(events), "event": bool(hit), "unlocks": (listed and fans_wait) or bool(hit),
                "show": ((listed and fans_live) or bool(hit)) and not (fr <= .001 and not hit), **_flag(flags, sp)}

    # ---------------------------------------------------------------- rooms
    if "dim_lights" in shelf:
        show = shown("dim_lights")
        light_kw = sum(h.light_kw(t) for t in steps) / len(steps) if steps else h.light_kw(ts)
        lim = float(zm.limits.get("light_pct", 50))
        for x in zm.targets["dim_lights"]:
            sp = x["spaces"]
            kw = light_kw * share(sp, "light")
            sc = scale_of(sp)
            dev[f'dim_lights:{x["target"]}'] = {
                "tool": "dim_lights", "target": x["target"], "spaces": sp, "points": sp, "running": kw > .1, "running_kw": kw, "scale": sc, "limit_pct": lim,
                "limit_why": f"{lim:g}% is the most open common areas allow." if sale_free <= 0 else f"{lim:g}% is the most the guard allows.", "can_give_kw": floor2(kw * sc * lim / 100),
                "all_empty": False, "show": bool(show.get(x["target"])), **{k: v for k, v in _flag(flags, sp).items() if k != "smoke_ok"}}
    if "switch_plugs" in shelf:
        plugs = window_mean(sim, "plugs", start, end) if sale else 0.0
        sold = float(getattr(sim, "sell", {}).get("plugs", 0.0))
        for x in zm.targets["switch_plugs"]:
            sp = x["spaces"]
            sc = 1.0 if x.get("any_time") else scale_of(sp)
            done = x["target"] in (lev.get("plugs") or {})
            dev[f'switch_plugs:{x["target"]}'] = {
                "tool": "switch_plugs", "target": x["target"], "spaces": sp, "holds": x.get("holds", ""), "any_time": bool(x.get("any_time")), "free": free_of(sp, "light") > .01,
                "can_give_kw": 0.0 if done else floor2(plugs * sold * sc), "show": not done}

    # ---------------------------------------------------------------- site
    if "set_battery" in shelf and sim.E > 0:
        x = zm.targets["set_battery"][0]
        stored, reserve = sim.soc * sim.E * sim.eta, sim.R * sim.E * sim.eta
        tripped = sim.P <= 0
        give = 0.0 if tripped else floor2(min(sim.P, max(0.0, stored - reserve) / hours))
        done = float(lev.get("battery_kw", 0.0)) > 0
        dev["set_battery:" + x["target"]] = {
            "tool": "set_battery", "target": x["target"], "can_give_kw": 0.0 if done else give, "stored_kwh": round(stored, 1), "reserve_kwh": round(reserve, 1), "rating_kw": x["rating_kw"], "tripped": tripped,
            "import_kw": float(row["base_kw"]) * x["service_share"], "trip_kw": x["trip_kw"], "charge_limit_kw": x["charge_limit_kw"], "discharge_limit_kw": x["discharge_limit_kw"],
            "award_kwh": 0.0, "show": not done}
    if "limit_ev_charging" in shelf and "ev" in getattr(sim, "app", {}):
        x = zm.targets["limit_ev_charging"][0]
        kw = window_mean(sim, "ev", start, end) if sale else 0.0
        n = max(1, round(kw / SESSION_KW)) if kw > .01 else 0
        amps = kw * 1000 / x["volts"] / n if n else 0.0
        done = float((lev.get("dev") or {}).get("ev", 0.0)) > 0
        dev["limit_ev_charging:" + x["target"]] = {
            "tool": "limit_ev_charging", "target": x["target"], "volts": x["volts"], "can_give_kw": 0.0 if done else floor2(kw), "leaves": x["leaves"], "kw_now": kw,
            "sessions": [{"amps": amps, "leaves": x["leaves"], "need_kwh": 0.0} for _ in range(n)], "show": not done}

    wk = "weekend" if dt.date.fromisoformat(ts[:10]).weekday() >= 5 else "weekday"
    label = ("Hot " if temp >= 27 else "Cold " if temp <= 5 else "Mild ") + wk
    return {"node": getattr(sim, "cfg", {}).get("name"), "side": "building", "ts": ts, "now": hhmm(ts), "sale": sale, "owed_kw": 0.0 if not sale else float(sale["owed_kw"]), "day": label,
            "place": zm.m["place"], "alarm": sim.overrides.get("alarm"), "note": sim.overrides.get("note"), "devices": dev, "shelf": list(shelf), "free": free, "cool_kw": C, "fans_kw": fans_kw,
            "instead": zm.m.get("instead") or {}, "never": zm.m.get("never") or [], "hours": hours}


def _exception(zm: Any, s: str, ts: str) -> dict[str, Any]:
    """What takes a space out of a whole-building call today."""
    if zm.spaces[s].get("smoke_ok") is False:
        return {"smoke_ok": False}
    why = zm.avoid(s, ts)
    if not why:
        return {}
    return {"exam": True} if why == "exams" else {"avoid": why}


def _flag(flags: dict[str, dict[str, Any]], spaces: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for s in spaces:
        out.update(flags[s])
    return out


def _held(tool: str, lev: dict[str, Any]) -> Any:
    """The spaces this tool already has a call on: only the others can still take one."""
    return {"set_zone_cooling": lev.get("zones"), "set_zone_standby": lev.get("standby"), "dim_lights": lev.get("lights")}.get(tool) or ()


def _off_plant(zm: Any, spaces: list[str], w: float) -> float:
    """The share of this group's cooling that no plant tool reaches: rooftop and tenant units."""
    if any(zm.spaces[s].get("off_plant") for s in zm.spaces):
        return sum(zm.spaces[s]["share"]["cool"] for s in spaces if zm.spaces[s].get("off_plant")) / (w or 1.0)
    return float(zm.m.get("off_plant_share", 0.0))


def _tenant_fans(sim: Any, zm: Any, spaces: list[str], start: str, end: str) -> float:
    """At the shopping center the store fans' part of the sale comes through the store zones."""
    k = sum(float(zm.spaces[s].get("tenant_fans", 0.0)) for s in spaces)
    return k * window_mean(sim, "fans", start, end) if k else 0.0


def _next_booking(zm: Any, spaces: list[str], steps: list[str], end: str | None = None) -> str | None:
    """The next time anyone is due, from the plan. It looks on past the sale, because a zone must be back at temperature before then."""
    from ..beckn import add_minutes
    after = [add_minutes(end, k) for k in range(0, RECOVER_MIN + 1, 15)] if end and steps else []
    return next((hhmm(t) for t in steps + after if any(zm.state(s, t) in ("booked", "open") for s in spaces)), None)


def _hall(zm: Any, spaces: list[str], ts: str, base_c: float) -> dict[str, Any]:
    """Residence halls, 15 Sep to 31 May: heating at 20 C or more by day, so the 2.8 C gap sets a floor under cooling."""
    md = ts[5:10]
    if any(zm.spaces[s].get("hall") for s in spaces) and (md >= "09-15" or md <= "05-31"):
        return {"heat_floor_c": 20.0, "cool_c": base_c}
    return {}


def _ease(zm: Any, zones: dict[str, bool]) -> dict[str, Any]:
    """The call that sets zones to coast, on a target that can take it now."""
    t = next((k for k, v in zones.items() if v), None)
    return {"tool": "set_zone_cooling", "target": t, "offset_c": EMPTY_C if zm.m.get("sale_free") else OPEN_C} if t else {}


def _ease_fans(zm: Any, standby: dict[str, bool], zones: dict[str, bool]) -> dict[str, Any]:
    t = next((k for k, v in standby.items() if v), None)
    return {"tool": "set_zone_standby", "target": t, "mode": "on"} if t else _ease(zm, zones)


# ---------------------------------------------------------------------------------------------------------- the words
def line(d: dict[str, Any], f: dict[str, Any]) -> str:
    """One device in the wording the card tests used."""
    tool, t, kw = d["tool"], d["target"], d.get("can_give_kw", 0.0)
    if tool == "set_battery":
        return f'{t} holds {d["stored_kwh"]:,.0f} kWh stored, reserve {d["reserve_kwh"]:,.0f} kWh, and can give {kw:,.0f} kW for the whole window.'
    if tool == "stop_plant_early":
        return f'{t} is cleared to stop: the signed loop list is on file and closet cooling is on. Stopping it gives {kw:,.0f} kW.'
    if tool == "set_plant_demand_limit":
        lab = " Live lab load reads normal." if d.get("lab_share") else ""
        return f'{t} must keep running. Its chillers run at {d["percent_now"]:.0f} percent. A cap of {d["cap_percent"]:.0f} percent gives {d["gives_kw"]:,.0f} kW. Zones are set to coast.{lab}'
    if tool == "set_zone_cooling":
        # one ready number, as the other lines give: with two in the sentence the model picked +3 on one run and +1.5 on the next, and the campus sale moved by 500 kW
        top = EMPTY_C if d["state"] in ("free", "part") else OPEN_C
        who = f'{d["free_share"] * 100:.0f}% of its rooms have no booking, and code holds the rest at +1.5 C' if d["state"] == "part" else "it is free" if d["state"] == "free" else "it is open"
        tail = f"An offset of +{top:g} C gives {kw:,.0f} kW." if kw > 0 else f"An offset of +{top:g} C gives no kW of its own."
        need = " The plant can't cap until these zones are eased." if d.get("unlocks") else ""
        return f"{t}: cooling is on, {who}. {tail}{need}"
    if tool == "limit_air_handler":
        ease = "Its zones are eased." if d.get("eased") else "Its zones aren't eased yet: call set_zone_standby first." if "set_zone_standby" in f["shelf"] else "Its zones aren't eased yet."
        return (f'{t}: fans run at {d["speed_now"]:.0f} percent, {d["kw_shown"]:,.0f} kW. They hold outdoor airflow and have no smoke duty. '
                f'A limit of {d["floor_percent"]:.0f} percent gives {kw:,.0f} kW. {ease}')
    if tool == "set_zone_standby":
        if d.get("event"):
            return f'{t}: an event was added there, so standby must end. Call set_zone_standby with mode off.'
        return f'{t}: {d["free_share"] * 100:.0f}% of it is free and every room has a working sensor. Standby gives no kW of its own, and lets the fans slow.'
    if tool == "dim_lights":
        return f'{t}: lights draw {d["running_kw"]:,.0f} kW. A cut of {d["limit_pct"]:.0f} percent gives {kw:,.0f} kW.'
    if tool == "switch_plugs":
        return f'{t}: the group holds {d["holds"]}. Off gives {kw:,.0f} kW.'
    if tool == "limit_ev_charging":
        n = len(d["sessions"])
        return f'{t}: {n} session{"s draw" if n != 1 else " draws"} {d["kw_now"]:.1f} kW. No driver leaves before {d["leaves"]}. A pause gives {kw:.1f} kW.'
    return ""


def render(f: dict[str, Any], owed: float, done: list[str], why: str | None, block: str | None, used: set[str]) -> str:
    sale = f["sale"]
    out = [f'{f["day"]}, {f["now"]} {f["place"]}.']
    if sale:
        out.append(f'Sale window {hhmm(sale["start"])} to {hhmm(sale["end"])}.')
    out.append(f"This building still owes {owed:,.0f} kW." if owed > 0.05 else "0 kW are owed.")
    if why:
        out.append(f"What changed: {why}")
    if f.get("note"):
        out.append(f'Manager\'s note: "{f["note"]}"')
    out += done
    for d in sorted(f["devices"].values(), key=lambda d: -d.get("can_give_kw", 0.0)):
        if key(d) not in used and can_act(f, d):
            out.append(line(d, f))
    out.append("No alarms." if not f.get("alarm") else f'Alarm: {f["alarm"]}.')
    if block:
        out.append(block)
    return " ".join(x for x in out if x)


def key(d: dict[str, Any]) -> str:
    return f'{d["tool"]}:{d["target"]}'


def can_act(f: dict[str, Any], d: dict[str, Any]) -> bool:
    """Worth showing now: it has kW to give, or something with kW waits for it."""
    return bool(d.get("show", True)) and (d.get("can_give_kw", 0.0) > 0 or bool(d.get("unlocks")))


def shortlist(f: dict[str, Any], used: set[str], reads: bool = True, limit: int = 8) -> list[dict[str, Any]]:
    """At most 8 device tools, then read_device and no_action. set_zone_standby and limit_air_handler always show together."""
    seen: dict[str, list[str]] = {}
    for d in sorted(f["devices"].values(), key=lambda d: -d.get("can_give_kw", 0.0)):
        if key(d) not in used and can_act(f, d):
            seen.setdefault(d["tool"], []).append(d["target"])
    for a, b in (("set_zone_standby", "limit_air_handler"), ("limit_air_handler", "set_zone_standby")):
        if a in seen and b not in seen:
            rest = [d["target"] for d in f["devices"].values() if d["tool"] == b and key(d) not in used and d.get("show", True)]
            if rest:
                seen[b] = rest
    out = [T.spec(f["side"], t, tg) for t, tg in list(seen.items())[:limit]]
    out.append(T.spec(f["side"], "no_action"))
    if reads:
        out.append(T.spec(f["side"], "read_device", sorted({t for tg in seen.values() for t in tg}) or None))
    return out
