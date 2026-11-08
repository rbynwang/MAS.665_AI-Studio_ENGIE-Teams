"""The guard for the nine building tools: plain code, in the order of SPEC section 6, from the cards in device_tools/cards/bld_*.json.

  1 safety state   2 allowed   3 range   4 timing   5 service   6 one writer per point

It reads only `f`, the facts the step gives it, so it runs the same on the simulated building and on a real one. tests/test_node_building_cards.py
walks every example and test on the building cards through it. A reading that isn't in the facts stops the call: d["humid"] raises, and check() answers that it's
missing. `.get` with a default is only for what a target may simply not have, like a booking. building_see.NO_SENSOR lists what the simulation can't read.

Rulings it follows (SPEC section 11): one writer per point, not per device. A writer may end or shorten its own call, and calls that don't overlap may be
queued. Count kW once: where a plant tool covers a window, zone calls show 0 kW, except rooftop and tenant units.
"""
from __future__ import annotations

import math
import re
from datetime import date
from typing import Any

from . import owner
from .guard import STOPPED, Guard, answer, blocked, num, off_list

VOLTS = 240.0
OPEN_C, EMPTY_C = 1.5, 3.0       # warmer than normal: while people may be in, and in empty zones
SETPOINT_C = (13.0, 29.0)        # the code's range for a cooling setpoint
GAP_C = 2.8                      # between heating and cooling setpoints
RECOVER_MIN = 60                 # an empty zone is back at temperature this long before anyone is due
RAMP_MIN = 30                    # the ramp and the fade start this long before the sale
COP = 4.4
OCCUPIED_ADD = .10               # occupied zones add 10% of the rest (the sale's rule)
FAN_MIN = 50.0                   # percent while the building is open: fans slow, and only closed wings stop
MAX_DIM = 50.0
AMPS = (6.0, 80.0)
POINT = {"set_zone_cooling": "zone", "set_zone_standby": "mode", "limit_air_handler": "fan", "set_plant_demand_limit": "plant", "stop_plant_early": "plant",
         "dim_lights": "light", "switch_plugs": "plug", "set_battery": "battery", "limit_ev_charging": "ev"}   # a cap and a stop on one plant are one writer
CONFIRM_MIN = {"set_zone_cooling": 30, "set_zone_standby": 5, "limit_air_handler": 10, "set_plant_demand_limit": 10, "stop_plant_early": 10,
               "dim_lights": 35, "switch_plugs": 5, "set_battery": 5, "limit_ev_charging": 5}
WRAP = {"set_battery", "limit_ev_charging"}                 # for these an end earlier than start means the next morning
MAX_WRITE_MIN = 1440             # every write ends by itself, and applied only says HH:MM, so a call can't reach past a day
CLIMATE = {"set_zone_cooling", "set_zone_standby", "limit_air_handler", "set_plant_demand_limit", "stop_plant_early"}   # what "leave the house as it is" means here
TIME = re.compile(r"\d{4}-\d\d-\d\d[ T]([01]\d|2[0-3]):[0-5]\d(:\d\d)?|([01]\d|2[0-3]):[0-5]\d")


def mins(x: str) -> int:
    """Minutes since midnight."""
    x = x[11:16] if len(x) > 5 else x
    return int(x[:2]) * 60 + int(x[3:5])


def clock(m: float) -> str:
    m = int(m) % 1440
    return f"{m // 60:02d}:{m % 60:02d}"


def good(x: Any) -> bool:
    """A real local time, HH:MM or with its date."""
    if not isinstance(x, str) or not TIME.fullmatch(x):
        return False
    try:
        return len(x) == 5 or bool(date.fromisoformat(x[:10]))
    except ValueError:
        return False


def day_of(call: dict[str, Any], f: dict[str, Any]) -> str:
    """The call's day: its own date, or the step's when it gives only HH:MM."""
    s = str(call["start"])
    return s[:10] if len(s) > 5 else str(f.get("ts", ""))[:10]


def span(call: dict[str, Any]) -> tuple[int, int]:
    """Start and end in minutes. Full timestamps say their own day. For HH:MM, an end earlier than start means the next morning where the card says so."""
    s, e = str(call["start"]), str(call["end"])
    a, b = mins(s), mins(e)
    if len(s) > 5 and len(e) > 5:
        return a, b + 1440 * (date.fromisoformat(e[:10]) - date.fromisoformat(s[:10])).days
    return a, b + 1440 if b <= a and call["tool"] in WRAP else b


def hours_of(call: dict[str, Any]) -> float:
    a, b = span(call)
    return (b - a) / 60.0


def held_share(rise_c: float, pre_c: float, cap: float, cool_kw: float, hours: float) -> float:
    """The share of the cooling the building's stored cold can carry: the card's formula, 1 at most."""
    return max(0.0, min(1.0, (rise_c + pre_c) * cap / max(1e-6, cool_kw * COP * hours)))


def zone_kw(d: dict[str, Any], offset_c: float) -> float:
    """kW a zone group gives at this offset: rooms with no booking at the offset, the rest at +1.5 C at most, with the 10% occupied add."""
    if d.get("cool_kw", 0.0) <= 0:
        return 0.0
    free = float(d.get("free_share", 1.0 if d.get("state") == "free" else 0.0))
    args = (d.get("pre_c", 0.0), d["cap"], d["cool_kw"], d.get("hours", 3.0))
    s_free = held_share(offset_c, *args)
    s_open = held_share(min(offset_c, float(d.get("open_c", OPEN_C))), *args)
    s_open += OCCUPIED_ADD * (1 - s_open)
    return d["cool_kw"] * (free * s_free + (1 - free) * s_open)


def with_times(try_: dict[str, Any], call: dict[str, Any], start: str | None = None, end: str | None = None) -> dict[str, Any]:
    keep = {k: clock(mins(call[k])) for k in ("start", "end") if good(call.get(k))}
    return {**try_, **keep, **({"start": start} if start else {}), **({"end": end} if end else {})}


class BuildingGuard(Guard):
    """One per node. writers holds each point's running calls, day_stops the day's one plant stop."""

    def __init__(self, setup: dict[str, Any] | None = None):
        super().__init__(setup)
        self.writers: dict[str, list[dict[str, Any]]] = {}
        self.day_stops: dict[str, str] = {}
        self.mode_writes: dict[str, int] = {}
        self.mode_day: str | None = None                  # the day mode_writes counts. A box's stored writes are by the day, so a new sale or a restart mustn't start them again

    def clear(self) -> None:
        self.writers.clear()

    def kept(self) -> set[str]:
        """The tools a manager's wish stops. owner.py names a home's tools, so keep_house_as_is reaches the climate tools here."""
        out = owner.blocked_tools(self.wishes)
        return out | CLIMATE if any(w["wish"] == "keep_house_as_is" for w in self.wishes) else out

    def check(self, call: dict[str, Any], f: dict[str, Any]) -> dict[str, Any]:
        tool, a = call["tool"], call.get("args")
        a = dict(a) if isinstance(a, dict) else {}
        target = str(a.get("target", ""))
        if f.get("alarm"):                                                      # step 1
            return blocked(target, "An alarm is on, so nothing is sent and every agent limit releases.")
        if f.get("stopped"):
            return blocked(target, STOPPED)
        if tool not in f["shelf"]:                                              # step 2
            return blocked(target, f"This node has no {tool}.")
        d = next((x for x in f["devices"].values() if x["tool"] == tool and x["target"] == target), None)
        if d is None:
            return self._no_target(tool, target, a, f, call)
        sale = f.get("sale") or {}
        if any(call.get(k) and not good(call[k]) for k in ("start", "end")):
            return blocked(target, "start and end must be local HH:MM.")
        if not call.get("start") or not call.get("end"):                        # step 3: every write ends by itself
            fill = {"start": clock(mins(call.get("start") or sale["start"])), "end": clock(mins(sale["end"]))} if sale else {}
            return blocked(target, "Every write needs a start and an end.", {"tool": tool, **a, **fill})
        s, e = span(call)
        if e <= s:
            return blocked(target, "end must come after start.")
        if e - s > MAX_WRITE_MIN:
            return blocked(target, "A write ends by itself within a day.")
        keeps = tool == "set_battery" and a.get("action") in ("hold", "charge")
        if tool in self.kept() and not keeps:                                   # step 5: the owner first
            return blocked(target, owner.why(tool, self.wishes))
        hidden = off_list(f, tool, target)                                      # step 2: a writer giving back its own call needs no list
        if hidden and not (tool == "set_zone_standby" and a.get("mode") == "off") and not (tool == "switch_plugs" and a.get("state") == "auto"):
            return blocked(target, hidden)
        try:
            ans = getattr(self, "_" + tool)(a, d, f, call)                      # steps 3 to 5
        except (KeyError, TypeError, ValueError, AttributeError) as exc:        # no reading, no write
            return blocked(target, f"A reading this call needs is missing ({exc}), so nothing was sent.")
        if ans.pop("released", False):                                          # a writer ending its own call
            ans["confirm_by"] = clock(s + CONFIRM_MIN[tool])
            return ans
        if ans["status"] in ("done", "adjusted"):                               # step 6
            clash = self._clash(tool, d, s, e)
            if clash:
                later = {"tool": tool, **a, "start": clock(clash["end"]), "end": clock(e)} if clash["end"] < e else None
                return blocked(target, f'{clash["says"]} runs until {clock(clash["end"])}, and the later call wins only after it ends.', later)
            give = float(ans["applied"].get("kw", 0.0)) if ans["applied"].get("action") == "discharge" else 0.0
            for p in self._points(tool, d):
                self.writers.setdefault(p, []).append({"start": s, "end": e, "tool": tool, "target": target, "says": self._says(tool), "kw": give})
            ans["confirm_by"] = clock(s + (5 if tool == "dim_lights" and ans["applied"].get("cut_percent") == 100 else CONFIRM_MIN[tool]))
            if tool == "stop_plant_early":
                self.day_stops[target] = day_of(call, f)
            if tool == "set_zone_standby" and d.get("write_limit"):             # only a write that goes out counts against the box
                self.mode_writes[target] = self.mode_writes.get(target, 0) + 2
        return ans

    # ------------------------------------------------------------------ steps 2 and 6
    def _no_target(self, tool, target, a, f, call):
        alt = (f.get("instead") or {}).get(target)
        if alt:
            return blocked(target, alt["why"], with_times({"tool": alt.get("tool", tool), **{**a, **alt.get("args", {})}, "target": alt["target"]}, call))
        if any(w in target for w in f.get("never", ())):
            return blocked(target, f"{target} is on the never-touched list.")
        names = [x["target"] for x in f["devices"].values() if x["tool"] == tool]
        return blocked(target, f"{target or 'That target'} can't take this call now.", {"tool": tool, "target": names[0]} if names else None)

    @staticmethod
    def _points(tool: str, d: dict[str, Any]) -> list[str]:
        return [f"{POINT[tool]}:{p}" for p in (d.get("points") or [d["target"]])]

    @staticmethod
    def _says(tool: str) -> str:
        return {"plant": "A call on this plant"}.get(POINT[tool], "An earlier call here")

    def _clash(self, tool, d, s, e):
        for p in self._points(tool, d):
            for w in self.writers.get(p, []):
                if w["start"] < e and s < w["end"]:
                    return w
        return None

    def _end(self, tool: str, d: dict[str, Any], at: int) -> list[dict[str, Any]]:
        """A writer ends its own running call at this minute."""
        out = []
        for p in self._points(tool, d):
            for w in self.writers.get(p, []):
                if w["start"] <= at < w["end"]:
                    w["end"] = at
                    out.append(w)
        return out

    # ------------------------------------------------------------------ what stops any zone tool
    @staticmethod
    def _back_by(d: dict[str, Any], call: dict[str, Any]) -> int | None:
        """The minute this zone must be back at temperature, on the call's own clock. A booking earlier than the start is tomorrow's."""
        nb = d.get("next_booking")
        if not nb:
            return None
        m = mins(nb)
        return (m + 1440 if m < span(call)[0] else m) - RECOVER_MIN

    @staticmethod
    def _zone_stop(d: dict[str, Any]) -> str | None:
        if d.get("exam"):
            return "A room in this group is in an exam, so it is never touched."
        if d.get("avoid"):
            return f'This is a time to avoid: {d["avoid"]}.'
        if not d["smoke_ok"]:
            return "Rule out a smoke role before any write to this unit."
        if d["closet"]:
            return "A unit in this group also cools a closet, and the driver can't separate them."
        if d["override"]:
            return "An occupant override is running, and it beats the agent."
        if d["humid"]:
            return "Humidity ends the event: dew point is past 15 C or RH is past 65%."
        return None

    # ------------------------------------------------------------------ zones
    def _set_zone_cooling(self, a, d, f, call):
        t, c = d["target"], num(a.get("offset_c"))
        stop = self._zone_stop(d)
        if stop:
            return blocked(t, stop)
        if not d["running"]:
            return blocked(t, "No cooling is running in this zone, so an offset moves nothing.")
        if c is None:
            return blocked(t, "offset_c must be a number.")
        if c == 0:
            return blocked(t, "An offset of 0 moves nothing.")
        times = {k: clock(mins(call[k])) for k in ("start", "end")}
        if c < 0:
            sale = f.get("sale")
            if sale and span(call)[0] < mins(sale["end"]) and span(call)[1] > mins(sale["start"]):
                return blocked(t, "A pre-cool adds load, so it can't run into the sale.")
            return self._precool(t, c, d, call, times)
        lead = clock(mins((f.get("sale") or call)["start"]) - RAMP_MIN)
        state = d.get("state", "open")                          # not known means people may be in
        lim = d.get("limit_c")
        if lim is not None and lim <= 0:
            return blocked(t, d.get("limit_why", "This zone may not run warmer now."))
        if state in ("open", "booked") and c > OPEN_C:
            why = "A booked floor stays within +1.5 C." if state == "booked" else "+3 C is for empty zones, and this one is open."
            return blocked(t, why, {"tool": "set_zone_cooling", "target": t, "offset_c": OPEN_C if lim is None else min(OPEN_C, lim), "start": lead, "end": times["end"]})
        back = self._back_by(d, call)
        if back is not None and c > OPEN_C and span(call)[1] > back:
            try_ = {"offset_c": c, "start": times["start"], "end": clock(back)} if back > span(call)[0] else {"offset_c": OPEN_C, **times}
            return blocked(t, f'It must be back at temperature {RECOVER_MIN} minutes before its {clock(mins(d["next_booking"]))} booking.', {"tool": "set_zone_cooling", "target": t, **try_})
        applied, status, why = c, "done", ""
        if c > EMPTY_C:
            applied, status, why = EMPTY_C, "adjusted", "+3 C is the most any zone may take."
        if lim is not None and applied > lim:
            applied, status, why = lim, "adjusted", d.get("limit_why", f"This zone's limit is +{lim:g} C.")
        top, base = d.get("top_c"), d.get("base_c")
        if top is not None and base is not None and state in ("open", "booked") and base + applied > top + 1e-9:
            applied = math.floor((top - base) * 10 + 1e-9) / 10   # down, or the rounding itself passes the top
            if applied <= 0:
                return blocked(t, f'{d.get("top_says", "Occupied zones stop")} at {top:g} C at 50% RH, and this {base:g} C base is already there.')
            status, why = "adjusted", f'{d.get("top_says", "Occupied zones stop")} at {top:g} C at 50% RH, so a {base:g} C base takes +{applied:g} C.'
        if base is not None and base + applied > SETPOINT_C[1]:
            return blocked(t, "A cooling setpoint stays between 13 and 29 C.")
        out = {"offset_c": applied, **times}
        if state == "part" and applied > float(d.get("open_c", OPEN_C)):
            out["booked_rooms_offset_c"] = float(d.get("open_c", OPEN_C))       # the controller holds rooms with a booking there
        kw = zone_kw(d, applied)
        if d.get("plant_covers"):                                # count once: the plant carries this part
            kw *= 1 - float(d.get("plant_part", 1.0))
            why = why or "The plant carries this kW, so it counts once, under the plant."
        return answer(status, t, out, kw + float(d.get("tenant_fan_kw", 0.0)) * min(1.0, applied / OPEN_C), why)

    def _precool(self, t, c, d, call, times):
        depth = float(d.get("precool_c", 1.0))
        applied, status, why = c, "done", ""
        if c < -depth:
            applied, status, why = -depth, "adjusted", f"Pre-cool stops at -{depth:g} C here."
        base = d.get("base_c")
        if base is not None and base + applied < SETPOINT_C[0]:
            return blocked(t, "A cooling setpoint stays between 13 and 29 C.")
        floor, cool = d.get("heat_floor_c"), d.get("cool_c")
        if floor is not None and cool is not None and cool + applied < floor + GAP_C - 1e-9:
            best = round(floor + GAP_C - cool, 1)
            return blocked(t, f"Heating can't go under {floor:g} C by day, and the {GAP_C:g} C gap stays, so cooling stops at {floor + GAP_C:g} C.",
                           {"tool": "set_zone_cooling", "target": t, "offset_c": best, **times})
        return answer(status, t, {"offset_c": applied, **times}, 0.0, why)

    def _set_zone_standby(self, a, d, f, call):
        t, mode = d["target"], a.get("mode")
        times = {k: clock(mins(call[k])) for k in ("start", "end")}
        if mode not in ("on", "off"):
            return blocked(t, "mode must be on or off.")
        if mode == "off":                                       # keeps the zone on its schedule, and ends any standby this node holds
            ended = self._end("set_zone_standby", d, mins(call["start"]))
            stop = None if ended else self._zone_stop(d)
            if stop:                                            # nothing of ours to give back, so nothing is written there
                return blocked(t, stop)
            for w in ended:                                     # a stop never holds a zone in standby: this is how it gets out
                w["ended"] = True
            return {**answer("done", t, {"mode": "off", **times}), "released": True}
        stop = self._zone_stop(d)
        if stop:
            return blocked(t, stop)
        if d.get("booked") or d.get("event_here"):
            return blocked(t, "This zone holds a booking or an event, so it stays on its schedule.")
        if d["mall_open"]:
            return blocked(t, "No mall zone counts as free while it is open.")
        if d["doas_open"]:
            return blocked(t, "An outdoor air unit stays on until every zone it serves is closed.")
        if d.get("closed"):
            return answer("done", t, {"mode": "on", **times}, 0.0, "Closed wing: the BMS schedule stops its air.")
        if not d["sensors"]:
            alt = d.get("instead")
            return blocked(t, "Every room needs a working occupant sensor, and CO2 is only a second check.",
                           {"tool": "set_zone_standby", "target": alt, "mode": "on", **times} if alt else None)
        if not d["standby_ok"]:
            return blocked(t, d.get("standby_why", "Its standby would stop outdoor air in an open building."),
                           {"tool": "set_zone_cooling", "target": t, "offset_c": EMPTY_C, "start": clock(mins((f.get("sale") or call)["start"])), "end": times["end"]})
        back = self._back_by(d, call)
        if back is not None and span(call)[1] > back:
            return blocked(t, f'It must be back at temperature {RECOVER_MIN} minutes before its {clock(mins(d["next_booking"]))} booking.',
                           {"tool": "set_zone_standby", "target": t, "mode": "on", "start": times["start"], "end": clock(back)} if back > span(call)[0] else None)
        limit = d.get("write_limit")
        if limit:                                               # Daikin boxes store each write: one long window, not short ones
            sale = f.get("sale")
            if sale and span(call)[1] - span(call)[0] < mins(sale["end"]) - mins(sale["start"]):
                return blocked(t, "Each change is a stored write, so one long window replaces short ones.",
                               {"tool": "set_zone_standby", "target": t, "mode": "on", "start": clock(mins(sale["start"])), "end": clock(mins(sale["end"]))})
            if self.mode_day and self.mode_day != day_of(call, f):
                self.mode_writes.clear()
            self.mode_day = day_of(call, f)
            if self.mode_writes.get(t, 0) + 2 > limit:
                return blocked(t, "Today's mode writes on these boxes are used up.")
        return answer("done", t, {"mode": "on", **times}, 0.0, "The saving shows under limit_air_handler and set_zone_cooling.")

    # ------------------------------------------------------------------ air handlers
    def _limit_air_handler(self, a, d, f, call):
        t, sp = d["target"], num(a.get("speed_percent"))
        times = {k: clock(mins(call[k])) for k in ("start", "end")}
        stop = self._zone_stop(d)
        if stop:
            return blocked(t, stop)
        if d["doas"]:
            return blocked(t, "A dedicated outdoor air unit holds its flow.")
        if not d["oa_ok"]:
            return blocked(t, "The unit has no outdoor airflow reading, and a fixed damper loses air as the fan slows.")
        if not d["co2_ok"]:
            return blocked(t, "A failed CO2 sensor ends any fan cap on that zone, and design airflow returns.")
        if d["hood"]:
            return blocked(t, "Hood make-up air takes no fan limit during service.")
        if sp is None:
            return blocked(t, "speed_percent must be a number.")
        floor = float(d.get("floor_percent", FAN_MIN))
        if sp < FAN_MIN:
            return blocked(t, "No agent stop while the area is open, so fans slow to about half flow.",
                           {"tool": "limit_air_handler", "target": t, "speed_percent": max(FAN_MIN, floor), **times})
        now = float(d.get("speed_now", 100.0))
        if sp >= now or floor >= now:                           # lifting the ask to the floor would raise a fan that runs under it
            return blocked(t, "It already runs at or below that speed, and a cap never raises a fan.")
        if not d["eased"]:
            e = d.get("ease_with") or {}
            return blocked(t, "Its zones aren't eased yet, so boxes would starve. Close the boxes first.", with_times(e, call, end=times["end"]) if e else None)
        applied, status, why = sp, "done", ""
        if sp < floor:
            applied, status, why = floor, "adjusted", d.get("floor_why", f"This unit stays at {floor:g} percent or more.")
        return answer(status, t, {"speed_percent": applied, **times}, float(d["kw_now"]) * (1 - (applied / now) ** 3), why)

    # ------------------------------------------------------------------ the plant
    def _set_plant_demand_limit(self, a, d, f, call):
        t, pct = d["target"], num(a.get("limit_percent"))
        times = {k: clock(mins(call[k])) for k in ("start", "end")}
        if not d["running"]:
            return blocked(t, "The chillers are off, so there is nothing to cap.")
        if d["computer_room"]:
            return blocked(t, "This plant cools a computer room, and that cooling is never touched.")
        lab = float(d.get("lab_share", 0.0))
        if lab > 0 and not d["lab_live"]:
            return blocked(t, "No live lab load reading, so no cap on this loop.")
        if not d["zones_coast"]:
            z = d.get("ease_with") or {}
            return blocked(t, "Zones on this plant aren't set to coast, so the cap would only warm the water.", with_times(z, call) if z else None)
        if pct is None:
            return blocked(t, "limit_percent must be a number.")
        now = float(d["percent_now"])
        room, trimmed = d.get("room_percent"), ""
        if room is not None and pct < room:                     # a person lowered the comfort limit: past this the zones reach it early, and the plant comes back mid-sale
            pct, trimmed = room, d["limit_why"]
        if pct >= now:
            return blocked(t, trimmed or "The chillers already run under that percent, so a cap would move nothing.")
        lab_floor = math.ceil(now * lab * (1 + float(d.get("lab_margin", .10))) - 1e-9) if lab > 0 else 0
        chillers = d["chillers"]
        lowest = min(c["floor"] for c in chillers)
        need = max(lab_floor, lowest)
        if pct < need:
            c0 = min(chillers, key=lambda c: c["floor"])
            why = f"Labs set a floor of {lab_floor} percent on this loop." if lab_floor > lowest else f'{c0["product"]} stays at {lowest:g} percent or more.'
            return blocked(t, why, {"tool": "set_plant_demand_limit", "target": t, "limit_percent": need, **times})
        per = {c["name"]: max(pct, c["floor"]) for c in chillers}
        run = [c for c in chillers if c.get("running", True)] or chillers
        eff = sum(per[c["name"]] * c.get("count", 1) for c in run) / sum(c.get("count", 1) for c in run)
        if eff >= now:                                          # the running chillers' floors sit above where they run
            return blocked(t, "The chillers already run under their floors, so a cap would move nothing.")
        kw = float(d["plant_kw"]) * (1 - eff / now)
        high = [c for c in chillers if c["floor"] > pct]
        if high:
            c0 = high[0]
            return answer("adjusted", t, {**{f'{c["name"]} ({c["product"]})': {"limit_percent": per[c["name"]]} for c in chillers}, **times}, kw,
                          f'{c0["product"]} stays at {c0["floor"]:g} percent.')
        return answer("adjusted" if trimmed else "done", t, {"limit_percent": pct, **times}, kw, trimmed)

    def _stop_plant_early(self, a, d, f, call):
        t = d["target"]
        times = {k: clock(mins(call[k])) for k in ("start", "end")}
        if not d["whole"]:
            return blocked(t, "Whole plants only: taking one chiller off is staging, and that stays in the BMS.", {"tool": "stop_plant_early", "target": d.get("plant", t), **times})
        if d.get("labs"):
            return blocked(t, "Labs share this water, so it can't stop.")
        if d.get("shared"):
            return blocked(t, "This plant is shared with another building.")
        if not d["running"]:
            return blocked(t, "The chillers are off, so there is nothing to stop.")
        if d["hot"] or d.get("limit_lowered"):                  # the list hides it too, but a rule that keeps people cool can't rest on what a model was shown
            return blocked(t, f'A stop can\'t be turned down, so {"on a hot evening" if d["hot"] else "with the comfort limit lowered"} the plant keeps running. Cap it instead.',
                           {"tool": "set_plant_demand_limit", "target": t, "limit_percent": max(c["floor"] for c in d.get("chillers", [{"floor": 70}])), **times})
        if not d["cleared"]:
            return blocked(t, "No signed list shows that nothing critical hangs on this loop.",
                           {"tool": "set_plant_demand_limit", "target": t, "limit_percent": max(c["floor"] for c in d.get("chillers", [{"floor": 70}])), **times})
        if not d["pre_ran"]:
            return blocked(t, "Pre-cool didn't run today, so there is no stored cold to coast on.",
                           {"tool": "set_plant_demand_limit", "target": t, "limit_percent": max(c["floor"] for c in d.get("chillers", [{"floor": 70}])), **times})
        if d.get("stops_today", 0) >= 1 or self.day_stops.get(t) == day_of(call, f):
            return blocked(t, "One stop and one start per chiller per day, and today's stop is used.")
        off = d.get("normal_off")
        if off and mins(call["end"]) < mins(off):
            return blocked(t, f"Ending before the plant's normal off time, {clock(mins(off))}, would restart it: a second start.",
                           {"tool": "stop_plant_early", "target": t, "start": times["start"], "end": clock(mins(off))})
        return answer("done", t, times, float(d["can_give_kw"]))

    # ------------------------------------------------------------------ rooms
    def _dim_lights(self, a, d, f, call):
        t, cut = d["target"], num(a.get("cut_percent"))
        times = {k: clock(mins(call[k])) for k in ("start", "end")}
        if d.get("exam"):
            return blocked(t, "A room in this group is in an exam, so it is never touched.")
        if d.get("avoid"):
            return blocked(t, f'This is a time to avoid: {d["avoid"]}.')
        if d["utility"]:
            return blocked(t, "A utility event holds this target, and utility commands outrank the agent.")
        if d["override"]:
            return blocked(t, "A person's override is running, and the agent never cancels one.")
        if not d["shed_ok"]:
            alt = d.get("instead")
            return blocked(t, "An area here reads Load shed Allowed No, and the agent can't change that point.",
                           {"tool": "dim_lights", "target": alt, "cut_percent": cut, **times} if alt else None)
        if not d["running"]:
            return blocked(t, "The lights are already off, so there is no kW to move.")
        if cut is None or cut < 1 or cut > 100:
            return blocked(t, "cut_percent must be 1 to 50, or 100 for off.")
        kw, lim = float(d["running_kw"]) * float(d.get("scale", 1.0)), min(MAX_DIM, float(d.get("limit_pct", MAX_DIM)))     # a map can lower the 50%, never raise it
        if lim <= 0:
            return blocked(t, d.get("limit_why", "No dim is allowed here now."))
        fc = float(d.get("egress_fc") or 0.0)
        if d["egress"] and cut != 100:                          # 1 fc stays at the darkest point: 1.6 fc measured gives 37%, and no measure gives no dim
            if fc <= 1.0:
                return blocked(t, "This zone is on an exit route with no measured light to spare, so it takes no dim.")
            lim = min(lim, float(math.floor(100 * (1 - 1 / fc))))
        if cut == 100:
            if d.get("limit_lowered"):                          # a person's limit binds off as well
                return blocked(t, d.get("limit_why", "A person lowered this limit."), {"tool": "dim_lights", "target": t, "cut_percent": lim, **times})
            if not d.get("all_empty") or d["egress"]:
                return blocked(t, "Off needs every area empty, and away from exit routes.", {"tool": "dim_lights", "target": t, "cut_percent": lim, **times})
            return answer("done", t, {"cut_percent": 100, **times}, kw)
        if cut > lim:
            return answer("adjusted", t, {"cut_percent": lim, **times}, kw * lim / 100, d.get("limit_why", f"{lim:g}% is the most the guard allows."))
        return answer("done", t, {"cut_percent": cut, **times}, kw * cut / 100)

    def _switch_plugs(self, a, d, f, call):
        t, st = d["target"], a.get("state")
        times = {k: clock(mins(call[k])) for k in ("start", "end")}
        if st == "on":
            return blocked(t, "The agent can't hold controlled receptacles on.", {"tool": "switch_plugs", "target": t, "state": "auto", **times})
        if st not in ("off", "auto"):
            return blocked(t, "state must be off or auto.")
        if st == "auto":
            ended = self._end("switch_plugs", d, mins(call["start"]))
            if not ended:
                return blocked(t, "This node holds nothing here to give back.")
            for w in ended:
                w["handed_back"] = True
            return {**answer("done", t, {"state": "auto", **times}, 0.0, "The plugs go back to their sensors."), "released": True}
        if any(w.get("handed_back") for p in self._points("switch_plugs", d) for w in self.writers.get(p, [])):
            return blocked(t, "One off and one hand-back per group per window, and this group has had both.")
        if d["override"]:
            return blocked(t, "An occupant override is running, and the agent never removes one.")
        if not d["loads_ok"]:
            alt = d.get("instead")
            return blocked(t, d.get("loads_why", "The group holds a load that can't be switched."), {"tool": "switch_plugs", "target": alt, "state": "off", **times} if alt else None)
        if not d.get("any_time") and not d["free"]:
            return blocked(t, "Only free zones: empty, with no booking. Unknown counts as occupied.")
        return answer("done", t, {"state": "off", **times}, float(d["can_give_kw"]))

    # ------------------------------------------------------------------ site
    def _set_battery(self, a, d, f, call):
        t, act, kw = d["target"], a.get("action"), num(a.get("kw"))
        times = {k: clock(mins(call[k])) for k in ("start", "end")}
        if act not in ("charge", "discharge", "hold"):
            return blocked(t, "action must be charge, discharge or hold.")
        if d.get("tripped"):
            return blocked(t, "The battery tripped in the last 10 minutes. It stays off and the agent doesn't retry.")
        for k, why in (("utility", "A utility command is active, so the plan stands down and reports."), ("generator", "A generator is running, so the plan stands down."),
                       ("other_program", "Another program holds these hours for this battery.")):
            if d[k]:
                return blocked(t, why)
        if act == "hold":
            if kw:
                return answer("adjusted", t, {"action": "hold", "kw": 0, **times}, 0.0, "With hold, kw is 0.")
            return answer("done", t, {"action": "hold", "kw": 0, **times})
        if kw is None or kw <= 0:
            return blocked(t, "A charge or discharge needs more than 0 kW.", {"tool": "set_battery", "target": t, "action": "hold", "kw": 0, **times})
        rating = float(d["rating_kw"])
        if kw > rating:
            return blocked(t, f"{rating:g} kW is the planning rating, and the overload isn't for planning.", {"tool": "set_battery", "target": t, "action": act, "kw": rating, **times})
        s, e = span(call)
        sale = f.get("sale")
        if act == "charge":
            if sale and s < mins(sale["end"]) and e > mins(sale["start"]):
                return blocked(t, "A charge adds load, so it can't run into the sale.", {"tool": "set_battery", "target": t, "action": "hold", "kw": 0, **times})
            limits = d.get("charge_limit_kw") or {}
            low = [(w, v) for w, v in limits.items() if _touches(w, s, e) and kw > v]
            if low:
                edge = min(mins(w[:5]) for w, _ in low)
                return blocked(t, "The charge limit for part of this call is below that.",
                               {"tool": "set_battery", "target": t, "action": "charge", "kw": kw, "start": clock(edge - (e - s)), "end": clock(edge)})
            return answer("done", t, {"action": "charge", "kw": kw, **times}, -kw)
        hours = (e - s) / 60.0
        above = float(d["stored_kwh"]) - float(d["reserve_kwh"]) - self._promised(d, f)
        if kw * hours > above + 1e-6:
            best = math.floor(max(0.0, above) / hours)
            return blocked(t, f"The call needs {kw * hours:,.0f} kWh and only {above:,.0f} kWh sits above the reserve.",
                           {"tool": "set_battery", "target": t, "action": "discharge", "kw": best, **times} if best > 0 else None)
        need = float(d.get("award_kwh", 0.0))                    # a discharge outside an award can't use what an award needs
        if need and sale and not (mins(sale["start"]) <= s and e <= mins(sale["end"])) and above - kw * hours < need:
            return blocked(t, "That would use energy the award needs.")
        export = max(0.0, kw - float(d["import_kw"]))
        if any(_touches(w, s, e) and export > v for w, v in (d.get("discharge_limit_kw") or {}).items()):
            return blocked(t, "Export would pass the discharge limit for part of this call.")
        room = float(d["import_kw"]) - float(d["trip_kw"])
        if kw >= room:
            best = max(0.0, math.ceil(room) - 1)
            if best <= 0:
                return blocked(t, f'Import is already at the {d["trip_kw"]:g} kW trip point.')
            return answer("adjusted", t, {"action": "discharge", "kw": best, **times}, best,
                          f'Import must stay above the {d["trip_kw"]:g} kW trip point, and {best + 1:g} kW would leave it at {d["trip_kw"]:g} kW.')
        return answer("done", t, {"action": "discharge", "kw": kw, **times}, kw)

    def _promised(self, d: dict[str, Any], f: dict[str, Any]) -> float:
        """kWh that calls already running or queued will still take from this battery. Each call alone can pass and the sum still reach under the reserve."""
        now = mins(f["now"]) if f.get("now") else 0
        return sum(w["kw"] * max(0, w["end"] - max(w["start"], now)) / 60.0 for p in self._points("set_battery", d) for w in self.writers.get(p, []))

    def _limit_ev_charging(self, a, d, f, call):
        t, amps = d["target"], num(a.get("amps"))
        times = {k: clock(mins(call[k])) for k in ("start", "end")}
        if not d["control"]:
            alt = d.get("instead")
            return blocked(t, "Those ports have no control path.", {"tool": "limit_ev_charging", "target": alt, "amps": a.get("amps"), **times} if alt else None)
        if amps is None or amps < 0 or 0 < amps < AMPS[0] or amps > AMPS[1]:
            return blocked(t, "A port takes 0, or 6 to 80 A, with nothing between.", {"tool": "limit_ev_charging", "target": t, "amps": AMPS[0], **times})
        sessions = [x for x in d.get("sessions", []) if x.get("amps", 0) > 0]
        if not sessions:
            return blocked(t, "No session is drawing power here, so it moves 0 kW.")
        if amps > 0 and amps >= max(x["amps"] for x in sessions):
            return blocked(t, "Every car here draws less than that, so it moves 0 kW, and a limit never raises a port.")
        s, e = span(call)
        volts = float(d.get("volts", VOLTS))
        day = str(call["start"])[:10] if len(str(call["start"])) > 5 else None
        lost = [x for x in sessions if not self._car_ok(x, amps, s, e, volts, day)]
        if lost:
            alt = d.get("instead")
            if alt:
                return blocked(t, "A car that must leave stays out of the limit.", {"tool": "limit_ev_charging", "target": alt, "amps": a.get("amps"), **times})
            if amps < AMPS[0] and all(self._car_ok(x, AMPS[0], s, e, volts, day) for x in sessions):
                return blocked(t, "A pause would leave a car short when it leaves.", {"tool": "limit_ev_charging", "target": t, "amps": AMPS[0], **times})
            return blocked(t, "A car must leave and still needs its charge.")
        kw = sum(max(0.0, volts * (x["amps"] - amps) / 1000) for x in sessions)
        return answer("done", t, {"amps": amps, **times}, kw, d.get("note", ""))

    @staticmethod
    def _car_ok(x, amps, s, e, volts, day=None):
        """Does the car still get what it needs by the time it leaves? A clock time before the call is tomorrow morning."""
        left = str(x.get("leaves") or "23:59")
        leaves = mins(left)
        if len(left) > 5 and day:
            leaves += 1440 * (date.fromisoformat(left[:10]) - date.fromisoformat(day)).days
        elif leaves < s:
            leaves += 1440
        need = float(x.get("need_kwh", 0.0))
        if need <= 0:
            return True
        rate = volts * x["amps"] / 1000
        got = volts * amps / 1000 * max(0, min(e, leaves) - s) / 60 + rate * max(0, leaves - e) / 60
        return got >= need - 1e-6

    # ------------------------------------------------------------------ the reading that proves it
    def confirm(self, tool: str, call: dict[str, Any], ans: dict[str, Any], reading: dict[str, Any]) -> dict[str, Any]:
        """At confirm_by: `failed` when the device didn't do what was sent. Sent values stay in `applied`, so the record shows both."""
        t = ans["target"]
        late = reading.get("failed")
        if not late:
            return ans
        return {**ans, "status": "failed", "expected_kw": round(float(reading.get("kw", 0.0)), 2), "why": reading.get("why", f"{late} did not respond.")}


def _touches(window: str, s: int, e: int) -> bool:
    """Does the call touch this HH:MM-HH:MM window of the agreement's schedule?"""
    a, b = int(window[:2]) * 60 + int(window[3:5]), int(window[6:8]) * 60 + int(window[9:11])
    b += 1440 if b <= a else 0
    return any(s + k < b and e + k > a for k in (-1440, 0, 1440))
