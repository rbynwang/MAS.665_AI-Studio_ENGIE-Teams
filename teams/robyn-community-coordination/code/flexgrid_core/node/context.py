"""A node's Context MCP: one tool server per node, and the only way its agent reaches anything (MULTI_AGENT_DESIGN, sections 6 and 10).

  See      get_site_status, get_free_windows, get_device_groups
  Predict  forecast_site, estimate_sellable
  Think    simulate_plan, lookup_device_card
  Trade    read_request, submit_offer, answer_commitment, report_shortfall
  Act      the node's own device tools, read_device and no_action, and for an exception set_plan and release_call

Code runs the workflow and enters a step. Each step shows only its own one to three tools, and a step that can act shows the device
shortlist too. Once a tool has answered, the step hides it: on the real model, simulate_plan on the sale step took all six turns of a
step and no device was opened. So the sale step shows the shortlist alone, the list that was tested. Every call goes: the step's
menu, the arguments, policy, then the tool. A device call goes on to the guard, the seal and the driver. The model reads only `view`,
a short line, and code does all the arithmetic.

An exception is one decision over the whole node (NodeAgent.plan): plan_tools() shows set_plan, whose letters stand for what can open and
what runs, with the reading tools this exception makes worth having. A simulation always says it is one, and a release goes through the same door as a call.

The server serves homes and buildings alike. It needs what NodeAgent already has: name, shelf, guard, view(), menu(), apply(),
close(), and for the market side HomeV2, which serves both. agent_mcp.serve_context() puts it on MCP's stdio transport.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import threading
from datetime import date, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .. import rt_control
from . import drivers as D
from . import owner
from . import policy as P
from . import record as R
from . import tools as T
from .guard import answer, blocked

HERE = Path(__file__).parent
MAX_SIMULATE = 4                 # calls in one simulate_plan
VIEW_BLOCKS = 24                 # schedule blocks one view shows, and it says so if there are more
EVENING = ("18:00", "21:00")     # the window a forecast reads when no request or sale names one
TEXT_MAX = 200                   # a model's free text, in characters
SANE = {"kW": (0.0, 1e5), "kWh": (0.0, 1e6), "A": (0.0, 1000.0), "percent": (0.0, 100.0), "C": (-10.0, 10.0)}     # by the unit a card gives
GROUP_TITLE = {"energy_storage": "Energy Storage", "ev_charging": "EV Charging", "central_plant": "Central Cooling Plant", "hvac_zones": "HVAC and Zone Control",
               "lighting_and_shading": "Lighting and Shading", "plug_loads": "Plug Loads and Circuits", "water_heating": "Water Heating",
               "pools_and_water_features": "Pools and Water Features", "appliances": "Major Appliances"}


def _param(name: str, kind: str, says: str, allowed: str = "", required: bool = True, unit: str = "") -> dict[str, Any]:
    return {"name": name, "type": kind, "says": says, "allowed": allowed, "required": required, "unit": unit}


REASON = _param("reason", "string", "why, in a few words")
# What Qwen reads. The wording follows SPEC section 10: what it does, Use when, Not when, 40 words at most, and one example.
CONTEXT: dict[str, dict[str, Any]] = {
    "get_site_status": {"job": ("read", P.SITE), "example": "get_site_status()", "parameters": [],
                        "description": "Reads the node now: what is owed, what each device group can give, any alarm. Use when something changed and the step gave no numbers. Not when the step already gave them."},
    "get_free_windows": {"job": ("read", P.SITE), "example": 'get_free_windows(day="tomorrow")', "parameters": [_param("day", "enum", "which day", "today | tomorrow")],
                         "description": "Lists when this node's spaces are busy or free on one day. Use when planning what can move. Not for names or reasons: it holds neither."},
    "get_device_groups": {"job": ("read", P.SITE), "example": "get_device_groups()", "parameters": [],
                          "description": "Lists this node's device groups and what each can give. Use when choosing where a cut could come from. Not to change anything: it only reads."},
    "forecast_site": {"job": ("read", P.SITE), "example": 'forecast_site(day="tomorrow")', "parameters": [_param("day", "enum", "which day", "today | tomorrow")],
                      "description": "Forecasts load, solar and car charging for the sale window. Use when judging a window still to come. Not for the next few minutes: the controller handles those."},
    "estimate_sellable": {"job": ("read", P.SITE), "example": "estimate_sellable()", "parameters": [],
                          "description": "Works out firm and tentative kW for the window, by device. Use when deciding how much to offer. Not after the offer has gone."},
    "simulate_plan": {"job": ("read", P.SITE), "example": 'simulate_plan(calls=[{"tool": "set_battery", "target": "home-battery", "action": "discharge", "kw": 3}])',
                      "parameters": [_param("calls", "calls", "the device calls to test, up to 4")],
                      "description": "A test: says what device calls would give and leave owed. Nothing is sent, and no device moves. Use when unsure a choice covers a shortfall. Not to act: its answer is never a call that ran."},
    "lookup_device_card": {"job": ("read", P.SITE), "example": 'lookup_device_card(tool="set_battery")', "parameters": [_param("tool", "enum", "the tool to read about", "this node's tools")],
                           "description": "Reads one tool's card: what it does, how fast, what follows, its limits. Use when a device's limits are unclear. Not for its reading now: read_device gives that."},
    "read_request": {"job": ("read", P.MARKET), "example": "read_request()", "parameters": [],
                     "description": "Reads what the market is asking this node now. Use when a step says a request or commitment is waiting. Not when no request is waiting."},
    "submit_offer": {"job": ("trade", P.MARKET), "example": 'submit_offer(share_percent=80, reason="storm tomorrow")',
                     "parameters": [_param("share_percent", "number", "percent of the safe offer, 0 to 100", unit="percent"), REASON],
                     "description": "Sends this node's offer as a share of the safe offer. Use when the share is chosen. Not to offer more: code keeps it at or under the safe offer."},
    "answer_commitment": {"job": ("trade", P.MARKET), "example": 'answer_commitment(decision="accept", reason="fits the plan")',
                          "parameters": [_param("decision", "enum", "accept or decline", "accept | decline"), REASON],
                          "description": "Answers a commitment that code has checked. Use when one is waiting. Not to change its kW. Code can't be overruled: a no stays a no."},
    "report_shortfall": {"job": ("trade", P.MARKET), "example": 'report_shortfall(reason="forecast_changed")',
                         "parameters": [_param("reason", "enum", "what caused it. device_out only for a device that reads out, owner_need only for a wish or a person's word", "device_out | schedule_changed | owner_need | forecast_changed | other")],
                         "description": "Tells the coordinator this node will fall short, and code fills the kW. Use when no device can cover the gap. Not when a device can still cover it."},
    "release_call": {"job": ("trade", P.MARKET), "example": 'release_call(target="floor-12-lights", reason="people came in")',
                     "parameters": [_param("target", "string", "the device a running call holds"), REASON],
                     "description": "Lets one running call go: the device goes back to its own day and its kW stop counting. Use when it presses on something that changed. Not to change its size."},
    "set_plan": {"job": ("trade", P.MARKET), "example": 'set_plan(open=["A", "C"], end_call="none", report="none", because="none", reason="covers it with no one feeling it")',
                 "parameters": [_param("open", "ids", "letters of what to open now"), _param("end_call", "ids", "the one running call to end, giving up its kW, or none"),
                                _param("report", "enum", "tell the desk why kW stay uncovered, or none", "none | device_out | schedule_changed | owner_need | forecast_changed | other"),
                                _param("because", "string", "the fact behind leaving kW uncovered or letting a call go: its id, or none"), REASON],
                 "description": "The whole plan for this exception, in one call: what opens, what is let go, what the desk hears. Use when you have decided. Not for a test: simulate_plan is the test."},
}
STEPS: dict[str, tuple[str, ...]] = {
    "1 read schedule": ("get_free_windows",),
    "2 forecast": ("forecast_site",),
    "3 check devices": ("get_device_groups", "lookup_device_card"),
    "4 estimate sellable": ("estimate_sellable", "simulate_plan"),
    "5 offer": ("read_request", "estimate_sellable", "submit_offer"),
    "6 commitment": ("read_request", "simulate_plan", "answer_commitment"),
    "6 prepare": (),
    "7 watch": (),
    "8 exception": ("get_site_status", "simulate_plan", "report_shortfall"),
    "9 report": ("get_site_status",),
}
PLAN_READS = ("simulate_plan", "get_free_windows", "forecast_site", "estimate_sellable", "lookup_device_card")     # shown in an exception only when it makes them worth having: NodeAgent.plan_reads
PLAN_TOOLS = ("set_plan", "release_call")       # an exception's own, through plan_tools: FG_WALK=model keeps the step's old list
ACT_STEPS = ("6 prepare", "7 watch", "8 exception")
ONCE_PER_REQUEST = ("submit_offer", "answer_commitment")
SENT = ("done", "adjusted")
TEST_LABEL = "A test only: nothing has opened yet."      # a simulation's answer goes into the next prompt: on the real model "X: done, 8.64 kW" read as a call that had run,
# and on 7 Nov "TEST ONLY, nothing was sent" read as the answer itself: plans copied it as their reason and opened nothing
WOULD = {"done": "would be taken", "adjusted": "would be lowered", "blocked": "would be blocked", "failed": "would fail"}
# What the model may say about a shortfall, as the desk reads it. These words are in rt_market.KNOWN_REASONS: anything else reaches the pool as "not given"
SHORT_WORDS = {"device_out": "a device is out", "schedule_changed": "the schedule changed", "owner_need": "an owner's need came first", "forecast_changed": "the forecast changed", "other": "another cause"}
HELD_BY_CONTROLLER = {"set_water_heater": ("hpwh",), "set_pool_pump": ("pool",), "delay_appliance": ("dish", "dryer")}     # devices a home's controller holds by itself


def spec(name: str, shelf: list[str] | None = None) -> dict[str, Any]:
    """One tool as Qwen sees it: the same shape as tools.spec()."""
    c = CONTEXT[name]
    props: dict[str, Any] = {}
    for p in c["parameters"]:
        if p["type"] == "calls":
            props[p["name"]] = {"type": "array", "items": {"type": "object"}, "description": p["says"]}
            continue
        if p["type"] == "ids":
            props[p["name"]] = {"type": "array", "items": {"type": "string"}, "description": p["says"]}
            continue
        d: dict[str, Any] = {"type": "number" if p["type"] == "number" else "string", "description": p["says"]}
        if p["type"] == "enum":
            d["enum"] = list(shelf or []) if p["name"] == "tool" else [x.strip() for x in p["allowed"].split("|")]
        props[p["name"]] = d
    return {"type": "function", "function": {"name": name, "description": c["description"] + " Example: " + c["example"],
                                             "parameters": {"type": "object", "properties": props, "required": [p["name"] for p in c["parameters"] if p["required"]]}}}


# --------------------------------------------------------------------------------------------------- the arguments
def clean_args(params: list[dict[str, Any]], args: Any) -> tuple[dict[str, Any], str]:
    """(clean values, why not). Stops what the guard shouldn't have to see: bad types, NaN, unknown names, text where a number goes.
    start and end are dropped: code fills them. A number outside its unit's sane range stops here too, and the guard's range check
    for the installed product still follows."""
    if args is None:
        args = {}
    if not isinstance(args, dict):
        return {}, "Arguments must be named values."
    known = {p["name"]: p for p in params if p["name"] not in ("start", "end")}
    out: dict[str, Any] = {}
    for k, v in args.items():
        if k in ("start", "end") or v is None:
            continue
        p = known.get(k)
        if p is None:
            return {}, f"{str(k)[:40]} isn't a parameter of this tool."
        kind = p["type"]
        if kind == "number":
            x = _number(v)
            lo, hi = SANE.get(p.get("unit", ""), (-1e6, 1e6))
            if x is None or not lo <= x <= hi:
                return {}, f"{k} must be a number from {lo:g} to {hi:g}."
            out[k] = x
        elif kind == "enum" and k != "target":
            allowed = [x.strip() for x in str(p.get("allowed", "")).split("|")]
            if p["name"] == "tool" and allowed == ["this node's tools"]:
                if not isinstance(v, str) or not re.fullmatch(r"[a-z_]{1,40}", v):
                    return {}, "That isn't a tool's name."
            elif v not in allowed:
                return {}, f"{k} must be one of: {', '.join(allowed)}."
            out[k] = v
        elif kind == "time":
            if not isinstance(v, str) or not re.fullmatch(r"([01]\d|2[0-3]):[0-5]\d", v):
                return {}, f"{k} must be a time like 18:30."
            out[k] = v
        elif kind == "calls":
            if not isinstance(v, list) or not 1 <= len(v) <= MAX_SIMULATE or not all(isinstance(c, dict) for c in v):
                return {}, f"{k} must list 1 to {MAX_SIMULATE} device calls."
            out[k] = v
        elif kind == "ids":
            ids = [v] if isinstance(v, str) else v
            if not isinstance(ids, list) or len(ids) > 16 or not all(isinstance(c, str) and re.fullmatch(r"[A-Z][0-9]{0,2}|none", c) for c in ids):
                return {}, f"{k} must list ids like A or R1."
            out[k] = ids
        else:
            if not isinstance(v, str) or len(v) > (64 if k == "target" else TEXT_MAX) or any(ord(c) < 32 for c in v):
                return {}, f"{k} must be plain text, {64 if k == 'target' else TEXT_MAX} characters at most."
            out[k] = v
    return out, ""


def _number(v: Any) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        return None
    try:
        x = float(v)
    except (ValueError, OverflowError):           # text, or an integer too long to be a number here
        return None
    return x if math.isfinite(x) else None


def device_params(side: str, tool: str) -> list[dict[str, Any]]:
    b = T.bundle()
    return (b.get(side, {}).get(tool) or b["common"].get(tool) or {"model_view": {"parameters": []}})["model_view"]["parameters"]


def run_call(tool: str, args: Any, start: str, end: str, f: dict[str, Any], side: str, step: str | None, who: P.Who, policy: P.Policy, gate: D.Gate) -> dict[str, Any]:
    """One device call, from the step's menu to the driver. The live path and the replay both run exactly this."""
    target = str(args.get("target", ""))[:64] if isinstance(args, dict) else ""
    out: dict[str, Any] = {"clean": None, "policy": None, "answer": None, "command": None, "driver": None}
    if step not in ACT_STEPS:
        out["answer"] = blocked(target, "This step can't change a device.")
        return out
    if tool not in f["shelf"]:
        out["answer"] = blocked(target, f"This node has no {str(tool)[:40]}.")
        return out
    clean, why = clean_args(device_params(side, tool), args)
    if why:
        out["answer"] = blocked(target, why)
        return out
    out["clean"] = clean
    v = policy.check(who, "call", P.group_of(tool))
    out["policy"] = v
    if not v["allow"]:
        out["answer"] = blocked(target, v["why"])
        return out
    out["answer"] = gate.run({"tool": tool, "args": clean, "start": start, "end": end}, f)
    out["command"], out["driver"] = gate.last.get("command"), gate.last.get("driver")
    return out


def listed(tools: list[dict[str, Any]]) -> dict[str, list[str] | None]:
    """What a turn's tools allow: each tool and the targets it lists, or None for a tool with no target list."""
    return {t["function"]["name"]: (t["function"]["parameters"]["properties"].get("target") or {}).get("enum") for t in tools}


def guard_state(guard: Any) -> dict[str, Any]:
    """What the guard remembers, so a replay can start it from the same place."""
    return R.clean({k: v for k, v in vars(guard).items() if k != "setup" and isinstance(v, (dict, list, set))})


def rerun(row: dict[str, Any], make_guard: Callable[[], Any], policy: P.Policy) -> dict[str, Any]:
    """One recorded device call again: the same inputs and guard state, no model, no device. record.replay() uses this."""
    g = make_guard()
    for k, v in copy.deepcopy(row["guard_before"]).items():
        setattr(g, k, set(v) if isinstance(getattr(g, k, None), set) else v)

    class Playback(D.Driver):
        def _write(self, ticket: D.Ticket) -> dict[str, Any]:
            return row.get("driver") or {"status": "done", "why": ""}

    who = P.Who(row["who"]["role"], row["who"]["via"], row["who"]["proven"])
    then = P.Policy(row["side"], policy.rules)
    then.closed = dict(row.get("policy_closed") or {})     # what people had closed at that moment
    return run_call(row["tool"], row["args"], row["start"], row["end"], row["inputs"], row["side"], row["step"], who, then, D.Gate(row["node"], g, Playback()))


# --------------------------------------------------------------------------------------------------- the schedule service
BLOCK = ("space", "start", "end", "state", "confidence", "source")
STATES = ("empty", "occupied", "asleep", "home", "away", "open", "closed", "busy", "event")
SOURCES = ("calendar", "timetable", "sensor", "history")


def clean_block(b: dict[str, Any], spaces: set[str]) -> dict[str, Any] | None:
    """The one shape every node gives, and nothing else: titles, names and notes never get past this. A value that isn't on a fixed
    list is replaced, so text can't ride in on a field either."""
    if not isinstance(b, dict):
        return None
    t = re.compile(r"([01]\d|2[0-3]):[0-5]\d|24:00")
    start, end = str(b.get("start", "")), str(b.get("end", ""))
    if not t.fullmatch(start) or not t.fullmatch(end):
        return None
    c = _number(b.get("confidence"))
    return {"space": b.get("space") if b.get("space") in spaces else "a space", "start": start, "end": end, "state": b.get("state") if b.get("state") in STATES else "busy",
            "confidence": round(min(1.0, max(0.0, c or 0.0)), 2), "source": b.get("source") if b.get("source") in SOURCES else "calendar"}


@lru_cache(maxsize=1)
def _home_plans() -> dict[str, Any]:
    z = np.load(HERE / "home_plans_past.npz")
    return {"row": {k: i for i, k in enumerate(z["keys"].tolist())}, "day": {d: i for i, d in enumerate(z["days"].tolist())},
            "state": z["state"], "conf": z["conf"], "leave": z["leave"]}


def _hm(q: int) -> str:
    return f"{q // 4:02d}:{(q % 4) * 15:02d}" if q < 96 else "24:00"


def home_plan(node: str, day: str) -> tuple[list[dict[str, Any]], set[str]]:
    """The day-before plan only: tools/export_home_plans.py leaves the true day out of this file."""
    z = _home_plans()
    h, d = z["row"].get(T.home_id(node)), z["day"].get(day)
    spaces = {"whole house", "car"}
    if h is None or d is None:
        return [], spaces
    a, c = z["state"][h, d], z["conf"][h, d]
    cut = [0, *(np.flatnonzero(np.diff(a)) + 1).tolist(), 96]
    names = {0: "empty", 1: "occupied", 2: "asleep"}
    rows = [{"space": "whole house", "start": _hm(s), "end": _hm(e), "state": names[int(a[s])], "confidence": float(c[s:e].mean()) / 100, "source": "calendar"} for s, e in zip(cut, cut[1:])]
    leave = float(z["leave"][h, d])
    if math.isfinite(leave):
        rows.append({"space": "car", "start": "00:00", "end": _hm(int(round(leave * 4))), "state": "home", "confidence": .9, "source": "calendar"})
    return rows, spaces


EVENING_RIGHT = {"university": .95, "shopping_center": .99, "office": .9}       # stage1/year/schedule_summary.json
EMPTY_BELOW, BUSY_FROM = .1, .5                                                 # share of a space in use


def building_plan(node: str, day: str) -> tuple[list[dict[str, Any]], set[str]]:
    """Each space by the hour, from the plan made the day before. Below a tenth in use reads as empty, as Stage 1 counted it."""
    from . import buildings
    z = buildings.load(node)
    rows: list[dict[str, Any]] = []
    spaces = {s["names"] for s in z.spaces.values()}
    for k, s in z.spaces.items():
        share = [z.plan(k, f"{day} {h:02d}:00") for h in range(24)]
        names = ["closed" if s.get("closed") else "empty" if x < EMPTY_BELOW else "busy" if x >= BUSY_FROM else "occupied" for x in share]
        h0 = 0
        for h in range(1, 25):
            if h == 24 or names[h] != names[h0]:
                rows.append({"space": s["names"], "start": f"{h0:02d}:00", "end": f"{h:02d}:00" if h < 24 else "24:00", "state": names[h0],
                             "confidence": EVENING_RIGHT.get(z.key, .9), "source": "timetable"})
                h0 = h
    return rows, spaces


class Schedule:
    """Calendars stop here: the model gets busy and free blocks, which protects privacy and blocks injection in one step."""

    def __init__(self, side: str, source: Callable[[str, str], tuple[list[dict[str, Any]], set[str]]] | None = None):
        self.source = source or (building_plan if side == "building" else home_plan)

    def blocks(self, node: str, day: str) -> list[dict[str, Any]]:
        rows, spaces = self.source(node, day)
        return [b for b in (clean_block(r, spaces) for r in rows) if b]


# --------------------------------------------------------------------------------------------------- cards
@lru_cache(maxsize=1)
def _card_views() -> dict[str, dict[str, str]]:
    try:
        return json.loads((HERE / "card_views.json").read_text())
    except (OSError, ValueError):
        return {}


def card_view(side: str, tool: str) -> str:
    """The short part of a card, from tools/export_card_views.py. The full cards stay with code and people, and don't go to a Jetson."""
    return _card_views().get(side, {}).get(tool) or T.bundle()[side][tool]["model_view"]["description"]


# --------------------------------------------------------------------------------------------------- the market side
class Desk:
    """The node's market side, for homes and buildings alike: HomeV2 serves both. A message with no transaction is held for the
    caller: HomeV2.handle sends what brain.decide_offer and decide_accept return."""

    def report(self, labelled: dict[str, Any]) -> dict[str, Any]:
        """Step 9's labelled answer goes up to the coordinator, signed. It keeps it beside the sale, and no model reads it there."""
        from .. import beckn
        v2 = self.v2
        body = beckn.make("on_status", sender=v2.name, receiver=v2.coordinator, transaction_id=f"report-{v2.name}", message={"node_report": labelled}, key=v2.key, sender_role="bpp")
        return {"sent": v2.send(v2.coordinator, beckn.kind("on_status"), body, v2.mid(v2.coordinator, "on_status", "report", labelled.get("sale"), labelled.get("ts")))}

    def __init__(self, v2: Any):
        self.v2 = v2

    def verify(self, kind: str, sender: str, body: dict[str, Any]) -> str | None:
        return self.v2.gate(kind, sender, body)

    def check_contract(self, sender: str, body: dict[str, Any]) -> dict[str, Any]:
        """Code's own verdict on a commitment."""
        return self.v2.verify_contract(sender, body)

    def safe_offer(self, event_id: str) -> dict[str, Any] | None:
        return self.v2.offers.get(event_id)

    def scale(self, safe: dict[str, Any], share: float) -> dict[str, Any]:
        """This share of the whole offer, every part alike. scale_offer does the battery and the evening's rebound."""
        from .. import beckn
        from ..brain import scale_offer
        out = scale_offer(safe, share, True, share > 0)
        for s, s0 in zip(out["slots"], safe["slots"]):
            for k in ("ev_pause_kw", "ac_kw", "app_kw", "light_kw"):
                if k in s:
                    s[k] = round(float(s0[k]) * share, 3)
            s["kw_offered"] = round(float(s0["kw_offered"]) * share, 2)
        out["energy_kwh"] = round(sum(s["kw_offered"] for s in out["slots"]) * beckn.SLOT_MINUTES / 60, 2)
        return out

    def send_offer(self, request: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
        self.v2.offers[final["event_id"]] = final
        return self.v2.send_offer(request["txn"], final) if request.get("txn") else {"held": True}

    def answer(self, request: dict[str, Any], verdict: dict[str, Any]) -> dict[str, Any]:
        if not request.get("txn"):
            return {"held": True}
        self.v2.install_plan(request["contract"], verdict)
        return self.v2.answer(request["txn"], request["contract"], verdict)

    def shortfall(self, short: dict[str, float], reason: str) -> dict[str, Any]:
        from .. import beckn, ids
        v2 = self.v2
        msg = {"short": short, "reason": reason}
        body = beckn.make("rt_capacity", sender=v2.name, receiver=v2.coordinator, transaction_id=f"cap-{v2.name}", message=msg, key=v2.key, sender_role="bpp")
        return {"sent": v2.send(v2.coordinator, beckn.kind("rt_capacity"), body, v2.mid(v2.coordinator, "rt_capacity", v2.now_ts(), ids.digest(msg)))}


def offer_request(need: dict[str, Any], safe: dict[str, Any], txn: str | None = None) -> dict[str, Any]:
    """Step 5's request: what the buyer asks and what the planner found safe."""
    sl = need.get("slots") or []
    return {"kind": "offer", "id": need.get("event_id"), "window": (need["window_start"], need["window_end"]), "price": float(sl[0].get("price") or 0.0) if sl else 0.0,
            "asked_kw": [s.get("kw_requested") for s in sl], "reply_by": need.get("commit_by"), "safe": safe, "txn": txn}


def commitment_request(c: dict[str, Any], verdict: dict[str, Any], txn: str | None = None) -> dict[str, Any]:
    """Step 6's request: the commitment and code's own verdict on it."""
    return {"kind": "commitment", "id": c.get("id"), "window": (c.get("window_start"), c.get("window_end")), "contract": c, "verdict": verdict, "txn": txn,
            "kw": [s.get("kw") for s in c.get("commitments", [])], "reply_by": c.get("opt_out_by")}


# --------------------------------------------------------------------------------------------------- the server
def device_write(node: Any) -> Callable[[dict[str, Any], dict[str, Any], dict[str, Any]], None]:
    """What a call changes on this simulated node. A building's needs the device and the facts too."""
    if getattr(node, "zm", None) is None:
        return lambda call, ans, f: node.apply(call, ans)
    from .building_agent import apply_call

    def write(call: dict[str, Any], ans: dict[str, Any], f: dict[str, Any]) -> None:
        d = next((x for x in f["devices"].values() if x["tool"] == call["tool"] and x["target"] == ans["target"]), None)
        apply_call(node.sim, node.zm, call, ans, d, f)
    return write


class ContextServer:
    def __init__(self, node: Any, *, side: str | None = None, policy: P.Policy | None = None, log: R.Log | None = None, driver: D.Driver | None = None,
                 schedule: Schedule | None = None, desk: Any = None, mode: str | None = None):
        self.node = node
        self.v2 = getattr(node, "home", None)
        self.side = side or getattr(node, "side", None) or ("building" if getattr(self.v2, "is_building", False) else "home")
        self.policy = policy or P.Policy(self.side)
        self.log = log or R.Log(node.name)
        self.steps = R.Steps(self.log)
        self.schedule = schedule or Schedule(self.side)
        self.desk = desk or (Desk(self.v2) if self.v2 is not None else None)
        self._write = device_write(node)
        self.driver = driver or D.make(self.side, mode or os.environ.get("FG_DRIVERS", "simulated"), P.group_of, write=self._sim_write, release=self._sim_release,
                                       read=self._sim_read, limit=self._sim_limit)
        self.gate = D.Gate(node.name, node.guard, self.driver)
        self._lock = threading.RLock()            # the workflow and an MCP client may ask at once
        self._f: dict[str, Any] = {}
        self.step: str | None = None
        self.ts = ""
        self.seen: set[str] = set()               # tools that have answered in this step
        self.reads: set[str] = set()              # readings this exception made worth having
        self.sale: dict[str, Any] | None = None
        self.request: dict[str, Any] | None = None
        self.chosen: dict[str, Any] = {}          # what was decided, by request id, for a caller holding the message
        self.stopped: str | None = None           # an alarm's stop: nothing is sent until a person clears it, whatever the signal does
        self.acted: set[str] = set()              # calls this process has already answered, so a repeat isn't counted twice
        self.handed_back = False                  # the model went, so the controller is running this sale alone
        self.shown: dict[str, list[str] | None] | None = None     # what this step last offered the model: the guard refuses a device call outside it
        self._restore()

    def _restore(self) -> None:
        """What a restart must not forget, from the record: what people said, a stop after an alarm, the request still waiting. It comes back here, not when
        a channel opens, because a node can start with no channel. A new run starts clean, and what people said through the channel still stands.

        The guard's other memory stays behind on purpose: a restart resets the levers too, so a point it remembered would only block the node's own new call."""
        stop, wishes = None, []
        self.notes_read: set[str] = set()         # the notes on file this node has read: a date in one means what it meant the first time
        self.noted: str | None = None             # the last door note it read, so a restart doesn't read it twice
        for r in list(self.log.rows):
            k = r["kind"]
            if k == "new_run":
                self.request, stop, wishes = None, None, []
                self.notes_read.clear()
                self.noted = None
            elif k == "alarm":
                stop = r.get("why") or "alarm"
            elif k == "request":
                self.request = r.get("request")
            elif k == "decision" and r.get("why") == "owner's note" and r.get("node", self.node.name) == self.node.name:     # a loan's row is on the lender's log, and it's the borrower's wish
                wishes += [w for w in r.get("answer") or [] if isinstance(w, dict) and w.get("wish") in owner.BLOCKS]
                self.notes_read.add(str(r.get("text")))
                self.noted = r.get("note_key") or self.noted
            elif k == "wish_ended":
                wishes = [w for w in wishes if (w.get("wish"), w.get("from", "")) != (r.get("wish"), r.get("since"))]
            elif k == "owner_channel" and (r.get("result") or {}).get("status") == "done":
                if r["verb"] in ("close_group", "need_now"):
                    self.policy.closed[r["group"]] = r["role"]
                elif r["verb"] == "open_group":
                    self.policy.closed.pop(r["group"], None)
                elif r["verb"] == "clear_alarm":
                    stop = None
                try:
                    if r["verb"] != "clear_alarm":
                        self._person(r["verb"], r["group"], r.get("name") or "", r.get("value"))
                except Exception as exc:          # a word that can't be put back mustn't stop the node coming back
                    self.log.add("restore_failed", verb=r["verb"], group=r["group"], why=type(exc).__name__)
        if wishes:
            self.node.guard.wishes = wishes
        if stop:
            self.stopped = stop
            self.gate.release_all()
            self._controller("hold", stop)

    # ------------------------------------------------------------------ the workflow
    def enter(self, step: str, ts: str, sale: dict[str, Any] | None = None, request: dict[str, Any] | None = None) -> None:
        """Code moves the node to a step. This is what decides which tools exist right now. The step before ends here."""
        if step not in STEPS:
            raise ValueError(f"no step {step!r}")
        self.leave()
        self.step, self.ts, self.sale = step, ts, sale
        self.seen, self.shown = set(), None
        self.handed_back = self.handed_back and step != "7 watch"      # a new sale opens its own levers
        self.reads = set()
        if request is not None:
            self.request = request
        self.steps.enter(step, ts, sale=(sale or {}).get("start"), request=(self.request or {}).get("id"))

    def leave(self, outcome: str = "done") -> None:
        if self.step:
            self.steps.leave(self.step, self.ts, outcome)
        self.step = None

    def new_run(self, keep_alarm: bool = False) -> None:
        """A run over the same days again on the same log, as fleet.py's demo does. Without this a repeat would answer done and send nothing.
        keep_alarm: fleet.py's door has no signature, so an alarm stays on through it, and on the record for a restart."""
        held = self.stopped if keep_alarm else None
        self.leave("new run")
        self.steps.new_run()
        if self.stopped and not held:
            self._controller("clear_stop")        # the restore's stop was the last run's
        self.seen, self.chosen, self.request, self.acted, self.stopped = set(), {}, None, set(), held
        if held:
            self.log.add("alarm", ts=self.ts, why=held, release={"status": "kept through a reset"})
        self.node.guard.wishes = []
        self.node.guard.clear()
        self.notes_read.clear()
        self.noted = None
        if hasattr(self.node, "noted"):
            self.node.noted = None

    def facts(self) -> dict[str, Any]:
        """What the node reads now. With no sale running, the window ahead stands in, with nothing owed."""
        if self.sale:
            return self._held(self.node.view(self.ts, self.sale))
        start, end = self._window()
        return self._held(self.node.view(self.ts, {"start": start, "end": end, "owed_kw": 0.0, "by_ts": {}}))

    def _held(self, f: dict[str, Any]) -> dict[str, Any]:
        """A stop after an alarm reads as the alarm, so the guard's first step answers it and a replay sees the same."""
        return {**f, "alarm": f.get("alarm") or self.stopped} if self.stopped else f

    def plan_tools(self, st: dict[str, Any], reads: list[str]) -> list[dict[str, Any]]:
        """What the model sees in an exception: set_plan, with the letters of this moment as the only values it can send, then the readings that matter here, then no_action.
        With nothing to open or let go it can only say why kW stay uncovered."""
        self.reads = {t for t in reads if t in PLAN_READS}
        out: list[dict[str, Any]] = []
        if st["options"] or st["running"]:
            s = spec("set_plan")
            props = s["function"]["parameters"]["properties"]
            if st["options"]:
                props["open"]["items"]["enum"] = [x["id"] for x in st["options"]]
            else:
                del props["open"]                 # with nothing there a letter could only be made up
            if st["running"]:
                props["end_call"] = {"type": "string", "description": "the one running call to end, giving up its kW, or none. Ending a call never covers anything", "enum": ["none"] + [x["id"] for x in st["running"]]}
            else:
                del props["end_call"]
            props["because"]["enum"] = ["none"] + [x["id"] for x in st["facts"]]
            s["function"]["parameters"]["required"] = list(props)
            out.append(s)
        else:
            out.append(spec("report_shortfall"))
        for t in reads:
            if t in self.reads and t not in self.seen:
                s = spec(t, self.node.shelf)
                if t == "simulate_plan":
                    s["function"]["parameters"] = {"type": "object", "required": ["open"], "properties": {"open": {"type": "array", "description": "the letters to test", "items": {"type": "string", "enum": [o["id"] for o in st["options"]]}}}}
                    s["function"]["description"] = CONTEXT[t]["description"] + ' Example: simulate_plan(open=["A", "C"])'
                if t in ("get_free_windows", "forecast_site"):
                    s["function"]["parameters"]["properties"]["day"]["enum"] = ["today"]
                out.append(s)
        return out + [T.spec(self.side, "no_action")]

    def tools(self) -> list[dict[str, Any]]:
        """This step's own tools, less those that have already answered. A step that can act adds the device shortlist through menu()."""
        return [spec(t, self.node.shelf) for t in STEPS.get(self.step or "", ()) if t not in self.seen]

    def menu(self, f: dict[str, Any], used: set[str], reads: bool = True) -> list[dict[str, Any]]:
        """What the model sees this turn: the node's shortlist, at most 8 device tools and read_device and no_action, then the step's own."""
        shown = self.node.menu(f, used, reads) if self.step in ACT_STEPS else []
        out = [t for t in shown if P.group_of(t["function"]["name"]) not in self.policy.closed] + self.tools()      # SPEC section 5: nothing blocked
        for t in out:                                                     # the card says what a person lowered, not only the step's words
            fn = t["function"]
            if fn["name"] in (f.get("lowered") or {}):
                fn["description"] += f' Now: {f["lowered"][fn["name"]]}'
        self.shown = listed(out)
        return out

    def show(self, tools: list[dict[str, Any]]) -> None:
        """A step that builds its own list says so here. The morning's pre-cool does."""
        self.shown = listed(tools)

    def on_list(self, f: dict[str, Any]) -> dict[str, list[str] | None]:
        """What a device call may name now: what the model was offered, or with nothing offered yet, as with an MCP client that hasn't listed, what this step would offer."""
        return self.shown if self.shown is not None else listed(self.node.menu(f, set(), True))

    # ------------------------------------------------------------------ calling a tool
    def act(self, tool: str, args: Any, *, start: str, end: str, facts: dict[str, Any] | None = None, who: P.Who = P.AGENT) -> dict[str, Any]:
        """One device call. Repeating it in the same step at the same moment sends nothing. Here the repeat comes back blocked, as the guard's one writer
        per point would say, so the agent doesn't count its kW twice. After a restart the first answer comes back: that memory is gone, and the call is running."""
        f = self._held(facts) if facts is not None else self.facts() if self.step in ACT_STEPS else {}
        target = str(args.get("target", ""))[:64] if isinstance(args, dict) else ""
        if self.step in ACT_STEPS:
            f = {**f, "menu": self.on_list(f)}

        def first() -> dict[str, Any]:
            before = guard_state(self.node.guard)
            self._f = f
            out = run_call(tool, args, start, end, f, self.side, self.step, who, self.policy, self.gate)
            self.log.add("act", step=self.step, ts=self.ts, side=self.side, tool=tool, args=args, start=start, end=end, who={"role": who.role, "via": who.via, "proven": who.proven},
                         inputs=f, guard_before=before, policy_version=self.policy.rules.get("version"), policy_closed=self.policy.closed, **out)
            return out["answer"]
        with self._lock:
            key = ("act", self.step, self.ts, tool, args, start, end)
            k = R.sha(key)
            a = self.steps.once(key, first, lambda x: x["status"] in SENT)
            if a["status"] == "unknown":          # cut off by a crash: it may have reached the device, and it isn't sent again
                return answer("failed", target, why=a["why"])
            if k in self.acted:
                return blocked(a["target"], f'{a["target"]} already took this call in this step, so nothing more was sent.')
            if a["status"] in SENT:
                self.acted.add(k)
            return a

    def call(self, name: str, args: Any = None, who: P.Who = P.AGENT) -> dict[str, Any]:
        """Any tool by name, the way the model or MCP asks. Code fills start and end for a device tool."""
        with self._lock:
            return self._call(name, args, who)

    def _call(self, name: str, args: Any, who: P.Who) -> dict[str, Any]:
        if name in ("no_action", "read_device") or name in (self.node.shelf or []):
            if self.step not in ACT_STEPS:        # the shortlist exists only in a step that can act
                return self._no(f"{name} isn't on this step's list.", name, args, who)
            return self._device(name, args, who)
        c = CONTEXT.get(name) if isinstance(name, str) else None
        if c is None or (name not in STEPS.get(self.step or "", ()) and name not in self.reads and not (name in PLAN_TOOLS and self.step == "8 exception")):
            return self._no(f"{str(name)[:40]} isn't on this step's list.", name, args, who)
        if name in self.seen and c["job"][0] == "read":      # a trade tool gives its first answer again, further down
            return self._no(f"{name} has answered in this step. Use what it said.", name, args, who)
        clean, why = clean_args(c["parameters"], args)
        if why:
            return self._no(why, name, args, who)
        v = self.policy.check(who, *c["job"])
        if not v["allow"]:
            return self._no(v["why"], name, args, who)
        try:
            run = getattr(self, "_" + name)
            out = self.steps.once(("trade", name, (self.request or {}).get("id")), lambda: run(**clean), lambda a: a["status"] == "done") if name in ONCE_PER_REQUEST else run(**clean)
        except Exception as exc:                  # a tool that breaks tells the model no more than that
            out = {"status": "failed", "why": f"{name} couldn't run: {type(exc).__name__}.", "view": ""}
        if out.get("status") == "done":
            self.seen.add(name)
        self.log.add("tool", step=self.step, ts=self.ts, tool=name, args=clean, who=who.role, result=out)
        return out

    def _no(self, why: str, name: Any, args: Any, who: P.Who) -> dict[str, Any]:
        out = {"status": "blocked", "why": why, "view": why}
        self.log.add("tool", step=self.step, ts=self.ts, tool=str(name)[:40], args=args, who=who.role, result=out)
        return out

    def _device(self, name: str, args: Any, who: P.Who) -> dict[str, Any]:
        if name == "no_action":
            clean, why = clean_args(device_params(self.side, name), args)
            self.log.add("tool", step=self.step, ts=self.ts, tool=name, args=clean, who=who.role, result={"status": "done"})
            return {"status": "done", "why": why, "view": "Nothing was done."}
        if name == "read_device":
            return self._read_device(args, who)
        if not self.sale:
            return self._no("No sale window is open, so no device call can run.", name, args, who)
        start, end = self._times(name, args)
        out = self.act(name, args, start=start, end=end, who=who)
        return {**out, "view": f'{out["status"]}: {out["target"]}, {out["expected_kw"]:.2f} kW. {out["why"]}'.strip()}

    def _times(self, tool: str, args: Any) -> tuple[str, str]:
        """Code fills start and end: from the window, or from the card where the node knows better."""
        times = getattr(self.node, "times", None)
        if times is None or not isinstance(args, dict):
            return max(self.ts, self.sale["start"]), self.sale["end"]
        f = self.facts()
        d = next((x for x in f["devices"].values() if x["tool"] == tool and x["target"] == args.get("target")), None)
        return times(tool, args, d, self.sale, self.ts)

    def _read_device(self, args: Any, who: P.Who) -> dict[str, Any]:
        clean, why = clean_args(device_params(self.side, "read_device"), args)
        if why:
            return self._no(why, "read_device", args, who)
        f = self.facts()
        d = next((x for x in f["devices"].values() if x["target"] == clean.get("target")), None)
        if d is None:
            return self._no("That isn't one of this node's devices.", "read_device", args, who)
        v = self.policy.check(who, "read", P.group_of(d["tool"]))
        if not v["allow"]:
            return self._no(v["why"], "read_device", args, who)
        r = self.driver.read(d["target"])
        return {**r, "view": r.get("view") or r.get("why", "")}

    # ------------------------------------------------------------------ see
    def _groups(self, f: dict[str, Any]) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        kept = self._kept()                       # a wish or a closed group gives nothing, so it isn't counted
        for d in f["devices"].values():
            if d["tool"] in kept:
                continue
            g = out.setdefault(P.group_of(d["tool"]) or "other", {"kw": 0.0, "targets": [], "tools": []})
            g["kw"] += float(d.get("can_give_kw", 0.0))
            g["targets"].append(d["target"])
            if d["tool"] not in g["tools"]:
                g["tools"].append(d["tool"])
        return out

    def _owed(self, f: dict[str, Any]) -> float:
        """What is still owed, after the calls already running."""
        fn = getattr(self.node, "owed_now", None)
        return float(fn(self.sale)) if fn is not None and self.sale else float(f.get("owed_kw", 0.0))

    def _get_site_status(self) -> dict[str, Any]:
        f = self.facts()
        groups = self._groups(f)
        owed = self._owed(f)
        words = [f'{f.get("day", "")}, {f.get("now", "")}.'.lstrip(", ")]
        if f.get("sale"):
            words.append(f'Sale window {f["sale"]["start"][11:16]} to {f["sale"]["end"][11:16]}.')
        words.append(f"{owed:.1f} kW are owed.")
        words += [f'{GROUP_TITLE.get(k, k)} can give {v["kw"]:.2f} kW.' for k, v in sorted(groups.items(), key=lambda x: -x[1]["kw"])]
        words += list((f.get("lowered") or {}).values())
        words.append("No alarms." if not f.get("alarm") else f'Alarm: {f["alarm"]}.')
        return {"status": "done", "view": " ".join(words), "owed_kw": round(owed, 2), "groups": {k: round(v["kw"], 2) for k, v in groups.items()}, "alarm": f.get("alarm")}

    def _day(self, day: str) -> str:
        return self.ts[:10] if day == "today" else str(date.fromisoformat(self.ts[:10]) + timedelta(days=1))

    def _get_free_windows(self, day: str = "tomorrow") -> dict[str, Any]:
        rows = self.schedule.blocks(self.node.name, self._day(day))
        by: dict[str, list[str]] = {}
        for b in rows[:VIEW_BLOCKS]:
            by.setdefault(b["space"], []).append(f'{b["state"]} {b["start"]} to {b["end"]}')
        view = " ".join(f'{k}: {", ".join(v)}.' for k, v in by.items()) or "No schedule for that day."
        if len(rows) > VIEW_BLOCKS:
            view += f" {len(rows) - VIEW_BLOCKS} more blocks aren't shown."
        return {"status": "done", "view": view, "blocks": rows}

    def _get_device_groups(self) -> dict[str, Any]:
        groups = self._groups(self.facts())
        rows = [{"group": k, "title": GROUP_TITLE.get(k, k), "can_give_kw": round(v["kw"], 2), "devices": v["targets"], "tools": v["tools"]} for k, v in sorted(groups.items(), key=lambda x: -x[1]["kw"])]
        view = " ".join(f'{r["title"]}: {r["can_give_kw"]:.2f} kW from {", ".join(r["devices"][:4])}.' for r in rows) or "No device can act now."
        return {"status": "done", "view": view, "groups": rows}

    # ------------------------------------------------------------------ predict
    def _window(self, day: str = "today") -> tuple[str, str]:
        """The window being traded in steps 5 and 6, then the sale running, then a request still waiting, then the evening."""
        w = (self.request or {}).get("window") or ()
        ask = (w[0], w[1]) if len(w) == 2 and w[0] else None
        if ask and self.step in ("5 offer", "6 commitment"):
            return ask
        if self.sale:
            return self.sale["start"], self.sale["end"]
        if ask:
            return ask
        d = self._day(day)
        return f"{d} {EVENING[0]}", f"{d} {EVENING[1]}"

    def _forecast_site(self, day: str = "tomorrow") -> dict[str, Any]:
        from .. import beckn
        start, end = self._window(day)
        d = self._day(day)
        start, end = d + start[10:], d + end[10:]          # the same hours, on the day asked for
        steps, t, k = [], start, 0
        while t < end:
            steps.append({"id": k, "start": t})
            t, k = beckn.add_minutes(t, beckn.SLOT_MINUTES), k + 1
        fc = self.v2.forecast_window({"event_id": "forecast_site", "window_start": start, "window_end": end, "slots": steps})
        n = max(1, len(fc["steps"]))
        load, pv, car = (sum(float(s[k]) for s in fc["steps"]) / n for k in ("load_p50", "pv_p50", "ev_kw"))
        view = f"From {start[11:16]} to {end[11:16]}: load {load:.2f} kW, solar {pv:.2f} kW, car {car:.2f} kW."
        return {"status": "done", "view": view, "load_kw": round(load, 3), "solar_kw": round(pv, 3), "car_kw": round(car, 3), "method": fc.get("method")}

    def _estimate_sellable(self) -> dict[str, Any]:
        start, end = self._window()
        w = self._sellable(start, end)
        view = f'From {start[11:16]} to {end[11:16]}: {w["firm_kw"]:.2f} kW firm and {w["tentative_kw"]:.2f} kW tentative.'
        return {"status": "done", "view": view, **{k: w.get(k) for k in ("firm_kw", "tentative_kw", "expected_kw", "devices", "source")}}

    def _sellable(self, start: str, end: str) -> dict[str, Any]:
        """The node's own estimate where it has one, then the forecasts, then what the devices show now, with sellable.py's share as firm."""
        from . import sellable
        own = getattr(self.node, "sellable", None)
        if own is not None:
            return own(start, end)
        v2 = self.v2
        kept = self._kept()                                # a wish or a closed group sells nothing
        if v2 is not None and v2.cfg.get("forecast") == "trained_2026" and not v2.is_building:
            try:
                return {"source": "forecasts", **sellable.home_window(self.node.sim, v2._models(), start, end, v2.now_idx(), [t for t in self.node.shelf if t not in kept])}
            except ValueError:                             # that day isn't forecast yet
                pass
        total = sum(float(d.get("can_give_kw", 0.0)) for d in self.facts()["devices"].values() if d["tool"] not in kept)
        firm = round(total * sellable.FALLBACK_SHARE, 3)
        return {"source": "devices now", "expected_kw": round(total, 3), "firm_kw": firm, "tentative_kw": round(total - firm, 3), "devices": {}}

    # ------------------------------------------------------------------ think
    def _simulate_plan(self, calls: list[dict[str, Any]]) -> dict[str, Any]:
        """A copy of the guard answers, so no point is taken and nothing is sent."""
        f = self.facts()
        if self.step in ACT_STEPS:                # a test says what the real call would say
            f = {**f, "menu": self.on_list(f)}
        g = copy.copy(self.node.guard)
        for k, v in vars(g).items():
            if isinstance(v, (dict, list, set)) and k != "setup":
                setattr(g, k, copy.deepcopy(v))
        start, end = (max(self.ts, self.sale["start"]), self.sale["end"]) if self.sale else self._window()
        rows, kw = [], 0.0
        node, seen = self.node, set()
        pre_of = getattr(node, "prerequisite", None)
        after = getattr(node, "after_test", lambda f, d, a: f)

        def one(tool: str, args: dict[str, Any], f: dict[str, Any], who: P.Who) -> tuple[dict[str, Any], dict[str, Any]]:
            clean, why = clean_args(device_params(self.side, tool), args)
            v = self.policy.check(who, "call", P.group_of(tool))             # a test says what the real call would say
            why = why or ("" if v["allow"] else v["why"])
            return clean, blocked(str(args.get("target", ""))[:64], why) if why else g.check({"tool": tool, "args": clean, "start": start, "end": end}, f)

        for c in calls:
            tool = c.get("tool")
            args = {k: v for k, v in c.items() if k != "tool"}
            if tool not in f["shelf"]:
                rows.append({"tool": str(tool)[:40], "status": "blocked", "kw": 0.0, "why": "This node has no such tool."})
                continue
            d = next((x for x in f["devices"].values() if x["tool"] == tool and x["target"] == args.get("target")), None)
            if d is not None and f'{tool}:{d["target"]}' in seen:
                rows.append({"tool": tool, "target": d["target"], "status": "done", "kw": 0.0, "why": "Set already, as what an earlier call waited for."})
                continue                            # as a plan carries it out: no second write
            pre = pre_of(d, f) if pre_of and d is not None else None
            if pre is not None and f'{pre["tool"]}:{pre["target"]}' not in seen:      # what it waits for goes first, as a plan carries it out (first_needs): code's own call, on its own list
                pa, ap = one(pre["tool"], node.ready(pre, f, self._owed(f)), {k: x for k, x in f.items() if k != "menu"}, P.CODE)
                rows.append({"tool": pre["tool"], "target": ap["target"], "status": ap["status"], "kw": ap["expected_kw"], "why": " ".join(x for x in (f'Set first: {d["target"]} waits for it.', ap["why"]) if x)})
                if ap["status"] in SENT:
                    kw += ap["expected_kw"]
                    f = after(f, pre, {**pa, **(ap.get("applied") or {})})
                    seen.add(f'{pre["tool"]}:{pre["target"]}')
            clean, a = one(tool, args, f, P.AGENT)
            rows.append({"tool": tool, "target": a["target"], "status": a["status"], "kw": a["expected_kw"], "why": a["why"]})
            if a["status"] in SENT:
                kw += a["expected_kw"]
                if d is not None:
                    f = after(f, d, {**clean, **(a.get("applied") or {})})          # zones eased here ease the fans for the calls after
                    seen.add(f'{tool}:{d["target"]}')
        for r in getattr(node, "test_unlocked", lambda f, seen: [])(f, seen):     # what code opens after the calls, which the letters already count
            rows.append(r)
            kw += r["kw"]
        owed = self._owed(f)
        left = max(0.0, owed - kw)
        view = (f"{TEST_LABEL} " + " ".join(f'{r.get("target") or r["tool"]}: {WOULD.get(r["status"], r["status"])}, {r["kw"]:.2f} kW.' + (f' {r["why"]}' if r["why"] else "") for r in rows)
                + f" If these ran, {left:.2f} kW would still be owed. Right now {owed:.2f} kW are owed. Nothing opens until you call set_plan.")
        return {"status": "done", "simulated": True, "view": view, "plan": rows, "total_kw": round(kw, 2), "left_kw": round(left, 2), "owed_kw": round(owed, 2)}

    def _test_guard(self) -> Any:
        g = copy.copy(self.node.guard)
        for k, v in vars(g).items():
            if isinstance(v, (dict, list, set)) and k != "setup":
                setattr(g, k, copy.deepcopy(v))
        return g

    def dry(self, tool: str, args: dict[str, Any], start: str, end: str, f: dict[str, Any]) -> dict[str, Any]:
        """What the real call would hear now, with no point taken and nothing sent: a plan offers only what would go through, and says why the rest can't."""
        target = str(args.get("target", ""))[:64]
        clean, why = clean_args(device_params(self.side, tool), args)
        v = self.policy.check(P.AGENT, "call", P.group_of(tool))
        why = why or ("" if v["allow"] else v["why"])
        if why:
            return blocked(target, why)
        try:
            return self._test_guard().check({"tool": tool, "args": clean, "start": start, "end": end}, f)
        except Exception as exc:                  # a guard that breaks has said no
            return blocked(target, f"The guard couldn't check this ({type(exc).__name__}).")

    def _lookup_device_card(self, tool: str) -> dict[str, Any]:
        if tool not in self.node.shelf:
            return {"status": "blocked", "why": "This node has no such tool.", "view": "This node has no such tool."}
        now = (self.facts().get("lowered") or {}).get(tool)
        return {"status": "done", "view": card_view(self.side, tool) + (f" Now: {now}" if now else "")}

    # ------------------------------------------------------------------ trade
    def _read_request(self) -> dict[str, Any]:
        r = self.request
        if not r:
            return {"status": "done", "view": "No request is waiting."}
        w = r.get("window") or ("", "")
        if r["kind"] == "offer":
            safe = [s.get("kw_offered", 0.0) for s in (r.get("safe") or {}).get("slots", [])]
            view = f'The buyer asks for {str(w[0])[11:16]} to {str(w[1])[11:16]} on {str(w[0])[:10]} at up to ${r["price"]:.2f} per kWh. The safe offer is {max(safe, default=0.0):.2f} kW.'
        else:
            view = f'A commitment of {max([float(k) for k in r.get("kw") or []], default=0.0):.2f} kW for {str(w[0])[11:16]} to {str(w[1])[11:16]}. Code says {r["verdict"]["decision"]}.'
        return {"status": "done", "view": view, "kind": r["kind"], "id": r.get("id")}

    def _submit_offer(self, share_percent: float, reason: str) -> dict[str, Any]:
        r = self.request
        if not r or r.get("kind") != "offer" or not r.get("safe"):
            return {"status": "blocked", "why": "No request for an offer is waiting.", "view": "No request for an offer is waiting."}
        final = self.desk.scale(r["safe"], share_percent / 100.0)
        if any(float(a.get("kw_offered", 0.0)) > float(b.get("kw_offered", 0.0)) + 1e-9 for a, b in zip(final["slots"], r["safe"]["slots"])):
            return {"status": "blocked", "why": "That offer is above the safe offer.", "view": "That offer is above the safe offer."}
        final = {**final, "agent": {"decided_by": "llm", "reason": reason, "take_part": share_percent > 0, "battery_share": share_percent / 100.0, "pause_ev": True, "devices_off": [],
                                    "share_percent": share_percent}}
        self.chosen[r["id"]] = final
        sent = self.desk.send_offer(r, final)
        kw = max((float(s.get("kw_offered", 0.0)) for s in final["slots"]), default=0.0)
        return {"status": "done", "view": f"Offer of {kw:.2f} kW sent.", "kw": [s.get("kw_offered") for s in final["slots"]], "sent": sent}

    def _answer_commitment(self, decision: str, reason: str) -> dict[str, Any]:
        r = self.request
        if not r or r.get("kind") != "commitment":
            return {"status": "blocked", "why": "No commitment is waiting.", "view": "No commitment is waiting."}
        code = r["verdict"]
        v = dict(code)                                       # the guard said no: the agent is not asked, and a yes changes nothing
        if code.get("decision") == "ACCEPTED" and decision == "decline":
            v = {**code, "decision": "DECLINED", "reason": reason}
        self.chosen[r["id"]] = v
        sent = self.desk.answer(r, v)
        return {"status": "done", "view": f'Answered {v["decision"]}.', "decision": v["decision"], "sent": sent}

    def _report_shortfall(self, reason: str) -> dict[str, Any]:
        """Code fills the kW from the controller's last plan, or from what is owed and not covered. The reason is a claim, and code holds what it can read:
        device_out was true in 2 of 16 reports on the real model, because it is the card's example. A claim that fails goes up in code's own words, and the record keeps both.
        A report with devices still closed is refused: on 9 Jul the office filed four with 105 kW unopened, and a report ended the step."""
        short = self._short()
        if not short:
            return {"status": "blocked", "why": "The plan covers what is owed.", "view": "The plan covers what is owed."}
        can = self._still_open()
        if can > 0.05 and can >= max(short.values()):
            why = f"Devices still closed can give {can:.2f} kW, which covers this."
            return {"status": "blocked", "why": why, "view": why}
        holds, words = self._reason_holds(reason)
        self.node.short_reason = words
        once = self.steps.once(("shortfall", self.ts[:10], short, reason), lambda: {"sent": self.desk.shortfall(short, words)})
        view = f"Reported {max(short.values()):.2f} kW short, because {words}." if holds else f"Reported {max(short.values()):.2f} kW short. {reason} doesn't fit what this node reads, so it went up as: {words}."
        return {"status": "done", "view": view, "short": short, "reason": words, "claim": reason, "claim_holds": holds, **once}

    def _still_open(self) -> float:
        """kW the devices nobody has opened could still give."""
        fn = getattr(self.node, "still_open", None)
        return float(fn(self.facts())) if fn is not None else 0.0

    def _reason_holds(self, reason: str) -> tuple[bool, str]:
        """(does the node's own reading back this up, the words to send). What code can't read stands as said."""
        sim = getattr(self.node, "sim", None)
        ov = getattr(sim, "overrides", None) or {}
        lev = getattr(sim, "levers", None) or {}
        tripped = bool(sim is not None and getattr(sim, "E", 0) > 0 and getattr(sim, "P", 1) <= 0)
        wishes = bool(getattr(self.node.guard, "wishes", None) or self.policy.closed or ov.get("ev_now") or ov.get("note"))
        holds = sim is None or {"device_out": tripped, "owner_need": wishes, "schedule_changed": bool(lev.get("events") or ov.get("zone_event") or ov.get("closed"))}.get(reason, True)
        if holds:
            return True, SHORT_WORDS.get(reason, SHORT_WORDS["other"])
        own = getattr(self.v2, "_rt_reason", None)
        return False, own() if own is not None else SHORT_WORDS["other"]

    def _release_call(self, target: str, reason: str) -> dict[str, Any]:
        """A running call ends now, through the same driver a person's "need it now" uses. Its kW stop counting, so the model sees them as owed again. Ending a limit is always allowed:
        the device goes back to its own day."""
        calls = [c for c in getattr(self.node, "calls", []) if c["target"] == target]
        if not calls:
            why = f"No call of this node runs on {target}."
            return {"status": "blocked", "why": why, "view": why}
        kw = sum(float(c["expected_kw"]) for c in calls)
        out = self.driver.release(target)
        if out.get("status") != "done":
            return {"status": "failed", "why": out.get("why", ""), "view": f"{target} couldn't be let go: {out.get('why', '')}"}
        return {"status": "done", "view": f"{target} was let go, so {kw:.2f} kW no longer count.", "kw": round(kw, 2), "reason": reason}

    def _set_plan(self, open: list[str] | None = None, end_call: list[str] | None = None, report: str = "none", because: str = "none", reason: str = "") -> dict[str, Any]:
        """The same plan through MCP: NodeAgent.plan carries out the one the model gives in its own step."""
        take = getattr(self.node, "take_plan", None)
        if take is None or not self.sale:
            return {"status": "blocked", "why": "No sale is running, so there is no plan to set.", "view": "No sale is running, so there is no plan to set."}
        return take({"open": open or [], "end_call": end_call or [], "report": report, "because": because, "reason": reason})

    def _short(self) -> dict[str, float]:
        ctl = getattr(getattr(self.node, "sim", None), "ctl", None)
        got = dict(((getattr(ctl, "last", None) or {}).get("short_by_ts")) or {})
        ahead = {t: round(float(v), 3) for t, v in got.items() if t >= self.ts}
        if ahead:
            return ahead
        if self.sale:
            left = self._owed(self.facts())
            if left > 0.05:
                return {max(self.ts, self.sale["start"]): round(left, 3)}
        return {}

    # ------------------------------------------------------------------ messages from outside
    def receive(self, kind: str, sender: str, body: dict[str, Any]) -> dict[str, Any]:
        """A market message: checked first, and held for read_request. Repeating one changes nothing."""
        if not isinstance(body, dict):
            return {"status": "blocked", "why": "Not a message."}
        ctx, msg = body.get("context"), body.get("message")
        ctx, msg = ctx if isinstance(ctx, dict) else {}, msg if isinstance(msg, dict) else {}

        def first() -> dict[str, Any]:
            try:
                v = self.policy.check(P.Who("coordinator", "signed", True), "request", P.MARKET)
                if not v["allow"]:
                    return {"status": "blocked", "why": v["why"]}
                txn = ctx.get("transactionId")
                if "intent" in msg:
                    need = msg["intent"]["need"]
                    self.request = offer_request(need, self.desk.safe_offer(need.get("event_id")), txn)
                elif "contract" in msg:
                    self.request = commitment_request({**msg["contract"], "transaction_id": txn}, self.desk.check_contract(sender, body), txn)
                if "intent" in msg or "contract" in msg:
                    self.log.add("request", request=self.request)      # a repeat after a restart changes nothing, so this is what brings it back
            except (KeyError, TypeError, AttributeError, ValueError) as exc:
                return {"status": "blocked", "why": f"That message can't be read ({type(exc).__name__})."}
            return {"status": "done", "why": ""}
        with self._lock:
            try:
                why = self.desk.verify(kind, sender, body)      # first, so a stranger reusing an id doesn't get the old answer
            except (KeyError, TypeError, AttributeError, ValueError) as exc:
                return {"status": "blocked", "why": f"That message can't be read ({type(exc).__name__})."}
            if why:
                return {"status": "blocked", "why": why}
            return self.steps.once(("message", ctx.get("messageId") or R.sha(body)), first, lambda a: a["status"] == "done")

    def alarm(self, why: str, ts: str | None = None) -> None:
        """Life safety: nothing more is sent, and every agent limit releases. It holds when the signal ends and across a restart, until a person clears it
        through the signed channel. The first call does that. Each step calls this again, and that only says the stop again to the controller: the guard reads
        self.stopped and the controller reads its own overrides, and whatever takes one away, they must not come to disagree. ts: when, since the last step
        entered can be hours back."""
        with self._lock:
            if not self.stopped:
                self.stopped = str(why)[:80]
                out = self.gate.release_all()
                self.log.add("alarm", ts=ts or self.ts, why=self.stopped, release=out)
            self._controller("hold", self.stopped)

    def closed_tools(self) -> set[str]:
        """This node's tools in a group a person has closed."""
        return {t for t in self.node.shelf if P.group_of(t) in self.policy.closed}

    def hand_back(self) -> None:
        """The model is gone, so the controller may use every device, except what the owner's wish or a person's closed group keeps out. With no levers at all
        the controller would take those too, and the guard would never be asked."""
        self.handed_back = True
        keep = self._kept()
        if keep and getattr(self.node, "zm", None) is not None:
            if self.node.sim.levers is None:      # no agent opened anything here, as at a building on a loan: the controller runs it all, so the wish has to hold there
                self.node.hold_out(keep)
            return                                # a building's levers can't open all but a few: what the agent opened stays and the rest stays closed
        self.node.close()
        if keep:
            self._controller("hand_back", keep)

    def _kept(self) -> set[str]:
        g = self.node.guard
        return (g.kept() if hasattr(g, "kept") else owner.blocked_tools(g.wishes)) | self.closed_tools()

    def _controller(self, what: str, *args: Any) -> None:
        """The simulated controller's side of what people say. The real controller isn't built, so on real drivers nothing moves here."""
        if isinstance(self.driver, D.SimulatedDriver):
            getattr(self, "_sim_" + what)(*args)

    def owner_channel(self, keys: dict[str, str], seq: dict[str, int] | None = None, save: Callable[[dict[str, int]], Any] | None = None) -> P.OwnerChannel:
        """A person's channel. What people said is back already (_restore): this takes no old message again, including one the node refused, which used its number all the same.
        The log's owner_seq row is written before the word does anything. Older code wrote only what finished, so those rows count too. seq, save: fleet.py's file for the node."""
        def person(*a: Any) -> dict[str, Any]:
            with self._lock:
                return self._person(*a)

        def kept(numbers: dict[str, int]) -> None:
            self.log.add("owner_seq", seq=numbers)
            if save:
                save(numbers)
        ch = P.OwnerChannel(self.node.name, self.policy, keys, person, self.log.add, seq, kept)
        for r in self.log.all("owner_seq"):
            for k, n in r["seq"].items():
                ch.seq[k] = max(ch.seq.get(k, 0), n)
        for r in self.log.all("owner_channel"):
            k = keys.get(r["role"]) if isinstance(r.get("role"), str) else None
            if k and (r.get("result") or {}).get("status") == "done":
                ch.seq[k] = max(ch.seq.get(k, 0), r["seq"])
        return ch

    def _person(self, verb: str, group: str, name: str, value: float | None) -> dict[str, Any]:
        if verb == "set_limit":
            return self.driver.set_limit(group, name, float(value))
        if verb == "clear_alarm":
            ov = getattr(getattr(self.node, "sim", None), "overrides", {})
            if ov.get("alarm"):
                return {"status": "blocked", "why": "The alarm is still on."}
            if not self.stopped and "alarm_hold" not in ov:   # a signal over before the next step holds the controller and never reached self.stopped
                return {"status": "blocked", "why": "No alarm is holding this node."}
            self.stopped = None
            self.log.add("alarm_cleared", ts=self.ts)
            self._controller("clear_stop")
            if self._kept():
                self.hand_back()                  # the alarm closed the levers, and with none the controller has every device: a wish or a closed group would stop being kept
            return {"status": "done"}
        if verb == "need_now":
            for c in [c for c in getattr(self.node, "calls", []) if P.group_of(c["tool"]) == group]:
                self.driver.release(c["target"])
        if verb in ("need_now", "close_group", "open_group"):
            self._controller("keep_out", group, verb != "open_group")
        if verb in ("need_now", "close_group") and self.handed_back:
            self._controller("hand_back", self._kept())     # no agent is running the sale, so its levers won't do this
        return {"status": "done"}

    # ------------------------------------------------------------------ the simulated device
    def _sim_write(self, call: dict[str, Any], ans: dict[str, Any]) -> None:
        self._write(call, ans, self._f)

    def _sim_release(self, target: str | None = None) -> None:
        """One device leaves the sale, or all of them. With levers open again the controller could use every device, so when all are let go it stands
        down too: writes stop, and the node leaves the sale until a person clears it. A building's agent can't yet release a single group, so a building lets go of everything."""
        node = self.node
        if target is not None and hasattr(node, "let_go"):
            return node.let_go(target)            # NodeAgent and BuildingAgent let one call go. What follows is for a node that can't
        if target is None or getattr(node, "zm", None) is not None:
            node.close()
            node.calls = []
            if target is None and getattr(node, "sim", None) is not None:
                node.sim.rt_override("stop", True)
            elif target is not None:
                node.open()                       # with no levers the controller would take back the group the person just asked for: what's left stays shut until the agent opens it
            return
        tools = {c["tool"] for c in node.calls if c["target"] == target}
        node.calls = [c for c in node.calls if c["target"] != target]
        node.guard.writers.pop(target, None)
        if node.sim.levers is not None and tools:
            node.reopen(tools)

    def _sim_read(self, target: str) -> dict[str, Any]:
        d = next((x for x in self.facts()["devices"].values() if x["target"] == target), {})
        keep = {k: v for k, v in d.items() if k not in ("tool", "target") and isinstance(v, (str, int, float, bool))}
        return {"status": "done", "why": "", "reading": keep, "view": f"{target}: " + ", ".join(f"{k} {v}" for k, v in list(keep.items())[:8]) + "."}

    def _sim_limit(self, group: str, name: str, value: float) -> None:
        rt_control.write_limit(self.node.sim, group, name, value)

    def _sim_keep_out(self, group: str, out: bool) -> None:
        """A person's "not this group" reaches the controller too, on a node with no agent call running. It stays through offers until the person takes it back."""
        sim = getattr(self.node, "sim", None)
        if not hasattr(sim, "kept_out"):
            return
        keys = {k for t in self.node.shelf if P.group_of(t) == group for k in HELD_BY_CONTROLLER.get(t, ())}
        sim.kept_out = (sim.kept_out | keys) if out else (sim.kept_out - keys)

    def _sim_hand_back(self, keep: set[str]) -> None:
        node = self.node
        if getattr(node, "zm", None) is not None:
            if node.sim.levers is None:           # with no levers the whole building is the controller's, so a person's word takes it out of the sale
                node.open()
            node.reopen(keep)
            return
        node.all_open(keep)                       # every device open, the device's own rating still limiting each, and these closed again

    def _sim_hold(self, why: str) -> None:
        sim = getattr(self.node, "sim", None)
        if hasattr(sim, "hold_alarm"):
            sim.hold_alarm(why)

    def _sim_clear_stop(self) -> None:
        sim = getattr(self.node, "sim", None)
        if hasattr(sim, "rt_override"):
            sim.overrides.pop("alarm_hold", None)
            sim.rt_override("stop", False)

    # ------------------------------------------------------------------ MCP
    def mcp_tools(self) -> list[dict[str, Any]]:
        """What agent_mcp lists: the step's tools, plus the device shortlist in a step that can act."""
        tools = self.menu(self.facts(), set()) if self.step in ACT_STEPS else self.tools()
        return [{"name": t["function"]["name"], "description": t["function"]["description"], "parameters": t["function"]["parameters"]} for t in tools]

    def mcp_call(self, name: str, args: Any) -> dict[str, Any]:
        """Only the short view leaves. A call from MCP is the agent's, whatever it says about itself."""
        out = self.call(name, args)
        return {"status": out.get("status"), "view": out.get("view") or out.get("why") or ""}
