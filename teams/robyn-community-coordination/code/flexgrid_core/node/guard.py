"""The safety guard for device calls: plain code, in the order of SPEC section 6, the same for every agent and every arm.

  1 safety state   2 allowed   3 range   4 timing   5 service   6 one writer per point

Every call gets the same answer: done, adjusted, blocked or failed, with why and try_instead, so the agent can correct itself.
Limits that protect a service, a panel or a person also sit in a listed device: this isn't the only protection.
"""
from __future__ import annotations

import math
import re
from typing import Any

from . import owner
from .see import hhmm

VOLTS = 240.0
STOPPED = "A person took this node out of the sale, so nothing is sent."


def answer(status: str, target: str, applied: dict[str, Any] | None = None, expected_kw: float = 0.0, why: str = "",
           try_instead: dict[str, Any] | None = None, confirm_by: str = "") -> dict[str, Any]:
    return {"status": status, "target": target, "applied": applied or {}, "expected_kw": round(expected_kw, 2),
            "confirm_by": confirm_by, "why": why, "try_instead": try_instead or {}}


def blocked(target: str, why: str, try_instead: dict[str, Any] | None = None) -> dict[str, Any]:
    return answer("blocked", target, why=why, try_instead=try_instead)


def off_list(f: dict[str, Any], tool: str, target: str) -> str | None:
    """The turn's list is a limit too: some rules live only in what it hides, like the plant on a hot evening. Why it's refused, or None. The list is in the facts, so a replay reads the same one."""
    m = f.get("menu")
    if m is None or (tool in m and (not m[tool] or target in m[tool])):
        return None
    return f"{tool} on {target or 'that target'} isn't on this step's list, so nothing was sent."


def num(x: Any) -> float | None:
    """A real number, or None. NaN and infinity count as no number: comparisons with them all come out false."""
    if isinstance(x, bool):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


class Guard:
    """One per node. It remembers who writes what, so a later call can't take a running point from an earlier one."""

    def __init__(self, setup: dict[str, Any] | None = None):
        self.setup = setup or {}
        self.writers: dict[str, dict[str, str]] = {}          # target -> {start, end}
        self.wishes: list[dict[str, Any]] = []                # the owner's, read from a note: step 5

    def clear(self) -> None:
        self.writers.clear()

    def check(self, call: dict[str, Any], f: dict[str, Any]) -> dict[str, Any]:
        tool, a = call["tool"], dict(call["args"])
        target = str(a.get("target", ""))
        if f.get("alarm"):                                                      # step 1
            return blocked(target, "An alarm is on, so nothing is sent and every agent limit releases.")
        if f.get("stopped"):
            return blocked(target, STOPPED)
        d = next((x for x in f["devices"].values() if x["tool"] == tool and x["target"] == target), None)
        if tool not in f["shelf"]:                                              # step 2
            return blocked(target, f"This node has no {tool}.")
        if d is None:
            names = [x["target"] for x in f["devices"].values() if x["tool"] == tool]
            return blocked(target, f"{target or 'That target'} can't take this call now.", {"tool": tool, "target": names[0]} if names else None)
        keeps = tool == "set_battery" and a.get("action") in ("hold", "charge")
        if tool in owner.blocked_tools(self.wishes) and not keeps:              # step 5: the owner first
            return blocked(target, owner.why(tool, self.wishes))
        hidden = off_list(f, tool, target)                                      # step 2: the list comes after the owner, whose words say more
        if hidden:
            return blocked(target, hidden)
        ans = getattr(self, "_" + tool)(a, d, f, call)                          # steps 3 to 5
        if ans["status"] in ("done", "adjusted"):                               # step 6
            w = self.writers.get(target)
            if w and w["start"] < call["end"] and call["start"] < w["end"]:
                return blocked(target, f'{target} already runs a call until {hhmm(w["end"])}.')
            self.writers[target] = {"start": call["start"], "end": call["end"]}
            ans["confirm_by"] = hhmm(call["start"])
        return ans

    # ------------------------------------------------------------------ battery
    def _set_battery(self, a, d, f, call):
        t, act, kw = d["target"], a.get("action"), num(a.get("kw"))
        if act not in ("charge", "discharge", "hold"):
            return blocked(t, "action must be charge, discharge or hold.")
        if d["tripped"]:
            return blocked(t, "The battery shut off under 10 minutes ago.")
        if act == "hold":
            return answer("done", t, {"action": "hold", "kw": 0})
        if kw is None or kw <= 0:
            return blocked(t, "A charge or discharge needs more than 0 kW.", {"tool": "set_battery", "target": t, "action": "hold", "kw": 0})
        sale = f["sale"]
        if act == "charge":
            if sale and call["end"] > sale["start"]:
                return blocked(t, "A charge can't run into the sale window.", {"tool": "set_battery", "target": t, "action": "hold", "kw": 0})
            limit = float(d.get("charge_limit_kw", 0.0))
            if limit <= 0:
                return blocked(t, "The battery can't take a charge now.")
            if kw > limit:
                return answer("adjusted", t, {"action": "charge", "kw": limit}, -limit, f"Lowered to the charge limit, {limit:.2f} kW.")
            return answer("done", t, {"action": "charge", "kw": kw}, -kw)
        give = d["can_give_kw"]
        if give <= 0:
            return blocked(t, "A discharge would pass the owner's reserve.")
        if kw > give:
            return answer("adjusted", t, {"action": "discharge", "kw": give}, give, f"Lowered to {give:.2f} kW, which keeps the owner's reserve.")
        return answer("done", t, {"action": "discharge", "kw": kw}, kw)

    # ------------------------------------------------------------------ car
    def _limit_ev_charging(self, a, d, f, call):
        t, amps = d["target"], num(a.get("amps"))
        if d["status"] != "charging":
            return blocked(t, f'The car reads {d["status"]}, so there is nothing to limit.')
        if amps is None or amps < 0:
            return blocked(t, "amps must be 0, or 6 and up.")
        if 0 < amps < 6:
            amps = 6.0
        amps = min(amps, float(d["max_amps"]))
        hours = max(0.25, (_mins(call["end"]) - _mins(call["start"])) / 60.0)
        if d["wait_h"] < hours and amps == 0:
            return blocked(t, f'The car must be ready by {d["leaves"]}, so it can\'t wait that long.')
        kw = max(0.0, d["charge_kw"] - VOLTS * amps / 1000.0)
        return answer("done", t, {"amps": amps}, kw)

    def _discharge_car(self, a, d, f, call):
        t, kw, mx = d["target"], num(a.get("kw")), num(a.get("max_kwh"))
        if d["status"] == "charging":
            return blocked(t, "The car is charging. Pause the charger first.", {"tool": "limit_ev_charging", "target": "car-charger", "amps": 0})
        if d["status"] != "idle":
            return blocked(t, f'The car reads {d["status"]}.')
        if kw is None or kw <= 0:
            return blocked(t, "kw must be above 0.")
        if a.get("max_kwh") is not None and (mx is None or mx <= 0):
            return blocked(t, "max_kwh must be above 0.")
        if d["can_give_kw"] <= 0:
            return blocked(t, "The car has nothing to give and still be ready to leave.")
        applied = {"kw": min(kw, d["can_give_kw"]), "max_kwh": min(mx if mx else d["max_kwh"], d["max_kwh"])}
        if kw > d["can_give_kw"] or (mx and mx > d["max_kwh"]):
            return answer("adjusted", t, applied, applied["kw"], "Lowered to what the car can give and still be ready.")
        return answer("done", t, applied, applied["kw"])

    # ------------------------------------------------------------------ comfort
    def _comfort(self, a, d, f, call):
        t, c = d["target"], num(a.get("less_comfort_c"))
        if c is None:
            return blocked(t, "less_comfort_c must be a number.")
        if not d["running"]:
            return blocked(t, "It isn't running, so there is nothing to ease.")
        if c <= 0:
            return blocked(t, "A pre-move is the controller's job in this step.")
        applied, why, status = c, "", "done"
        if c > d["limit_c"]:
            applied, status, why = d["limit_c"], "adjusted", f'Lowered to this home\'s limit, {d["limit_c"]:.2f} C.'
        share = min(1.0, applied * d["cap"] / max(1e-6, d["run_kw"] * d["cop"] * d["hours"]))
        return answer(status, t, {"less_comfort_c": applied}, d["run_kw"] * share, why)

    _set_thermostat = _comfort
    _set_room_ac = _comfort

    # ------------------------------------------------------------------ water, machines, pool
    def _set_water_heater(self, a, d, f, call):
        t, act = d["target"], a.get("action")
        if not d["valve"]:
            return blocked(t, "No mixing valve is on record for this home.")
        if act == "load_up":
            if f["sale"] and call["start"] >= f["sale"]["start"]:
                return blocked(t, "load_up can't run during a sale.", {"tool": "set_water_heater", "target": t, "action": "hold"})
            if d["tank_f"] >= d["normal_f"]:
                return blocked(t, "The tank is already at normal.")
            return answer("done", t, {"action": "load_up"}, 0.0)
        if act != "hold":
            return blocked(t, "action must be hold or load_up.")
        if not d["hot_ok"]:
            return blocked(t, "Hot water reads low.")
        return answer("done", t, {"action": "hold"}, d["can_give_kw"])

    def _delay_appliance(self, a, d, f, call):
        t, by = d["target"], str(a.get("finish_by") or d["done_by"])
        if not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", by):
            by = d["done_by"]
        if d["status"] != "waiting":
            return blocked(t, "It doesn't read waiting.")
        end = hhmm(call["end"])
        if _mins(end) + d["cycle_min"] > _mins(d["done_by"]):
            return blocked(t, f'It would miss the done-by time, {d["done_by"]}.')
        if by > d["done_by"]:
            by = d["done_by"]
        return answer("done", t, {"finish_by": by}, d["can_give_kw"])

    def _set_pool_pump(self, a, d, f, call):
        t, s = d["target"], num(a.get("speed_percent"))
        if d["cold"]:
            return blocked(t, "At 40 °F or below nothing is sent: freeze protection comes first.")
        if not d["filter_ok"]:
            return blocked(t, "Filter hours read short.")
        if not d["running"]:
            return blocked(t, "The pump isn't running.")
        if s is None:
            return blocked(t, "speed_percent must be a number.")
        if s >= d["speed"]:
            return blocked(t, f'This tool slows the pump, and it already runs at {d["speed"]} percent.')
        if s < d["floor"]:
            return answer("adjusted", t, {"speed_percent": d["floor"]}, d["can_give_kw"], f'Raised to this pump\'s floor, {d["floor"]} percent.')
        return answer("done", t, {"speed_percent": s}, d["can_give_kw"])


def _mins(x: str) -> int:
    x = x[11:16] if len(x) > 5 else x
    m = int(x[:2]) * 60 + int(x[3:5])
    return m + 1440 if m < 360 else m           # a time after midnight belongs to the same evening
