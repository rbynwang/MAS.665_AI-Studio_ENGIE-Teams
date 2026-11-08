"""A building's agent: NodeAgent with a building's tools, words, guard and levers.

The loop is NodeAgent's, with what a building adds:
  many targets per tool, so what is used is a target, not a tool
  what depends on what: fans wait for their zones and a plant for its zones to coast, so code walks those in order and a plan's letter brings its prerequisite with it
  times from the card: code fills start and end per tool, and a guard's try_instead keeps its own
  the schedule watch: sensors far above the plan wake the agent, and the rise the zones may take falls at once
  the plant's release: zones at their limit end a cap or a stop, and the agent is told
  on any alarm every agent limit releases at once

What a call changes on the simulated building is in apply_call, the controller's side. sim.levers holds it all.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Callable

from . import buildings, context, owner
from . import policy as P
from . import tools as T
from .agent import CUSHION, LEAD_MIN, Brain, NodeAgent, fact, write_comfort
from .building_guard import CLIMATE, RAMP_MIN, BuildingGuard, mins
from .building_see import can_act, comfort_c, free_shares, key, render, see_building, shortlist, steps_of
from .agent import NO_REASON, WALK_BY_CODE
from . import arm
PRECOOL_AGAIN = "If you don't pre-cool, call no_action with your reason for the manager, in plain words."
PRECOOL_LESS_AGAIN = ("That pre-cools by less than the {depth:g} C the playbook allows. Pre-cool the full {depth:g} C, or keep your figure and give your reason "
                      "for the manager in one plain sentence in your reply.")
from .building_guard import EMPTY_C, OPEN_C
from .building_see import line

MAX_ASKS = 12          # model answers in one step: a campus sale takes about eight device calls
STANDBY_LEAD_MIN = 15
PRE_COOL = {"university": ("10:00", "13:00"), "shopping_center": ("08:00", "11:00"), "office": ("07:00", "13:00")}   # the playbook, MULTI_AGENT_DESIGN section 8
PRE_LEAD_MIN = 10
PRE_RAMP_MIN = 120     # how long the pull-down takes: well inside the card's limit, 0.25 C a step
HELD_DEVICES = {"cooling": "fans", "car": "ev", "devices": "plugs"}      # the building's own devices each part takes with it
# what the card tests' sentences say that a plan doesn't need: orders the letters carry out by themselves, and constants that read as sensors
re_order = re.compile(r" ?(Its zones aren't eased yet(: call set_zone_standby first)?\.|The plant can't cap until these zones are eased\.|Its zones are eased\.|Zones are set to coast\.|Call set_zone_standby with mode off\.|Live lab load reads normal\.)| and every room has a working sensor")


# --------------------------------------------------------------------------------------------------- the controller's side
def comfort(sim: Any, rise: float) -> dict[str, float]:
    """The plan may let the building rise this far, and may not cool it below where it is: extra cooling is a call of its own, not a closed lever."""
    cap = getattr(sim, "person_rise", None)     # a person's tighter limit holds through every step's rewrite
    rise = rise if cap is None else min(rise, cap)
    return {"max_rise_c": rise, "max_rise_away_c": rise, "precool_c": max(0.0, -sim.x)}


def open_levers(sim: Any) -> None:
    """From now on the controller uses only what the agent's calls open."""
    sim.levers = {"battery_kw": 0.0, "ev": False, "ev_kw": None, "v2h_kw": 0.0, "light_share": 0.0, "dev": {}, "zones": {}, "standby": set(), "plants": {}, "lights": {},
                  "fans": {}, "plugs": {}, "released": None, "until": {}}
    write_comfort(sim, comfort(sim, 0.0))
    sim.house.cut = 0.0
    sim.hold_cold = False


def close_levers(sim: Any) -> None:
    """Every write ends by itself, and the building goes back to its own day."""
    sim.levers = None
    write_comfort(sim)
    sim.house.cut = 0.0


def sync_comfort(sim: Any, zm: Any, free: dict[str, float]) -> float:
    """The rise the building may reach now: from its zones' offsets, or a stopped plant's limits, and who is in the rooms."""
    lev = sim.levers
    zones = dict(lev["zones"])
    if any(p.get("stop") for p in lev["plants"].values()):
        zones = {s: 3.0 for s in zm.spaces}
    c = comfort_c(zm, zones, free)
    write_comfort(sim, comfort(sim, c))
    return c


def sync_plant(sim: Any, f: dict[str, Any]) -> None:
    """Caps and stops force cooling down from their start to their end: the share of the building's own cooling they hold off."""
    live = [p for p in sim.levers["plants"].values() if not p.get("released")]
    sim.house.cut_span = (min((p["from"] for p in live), default=""), max((p["to"] for p in live), default="9"))
    if any(p.get("stop") for p in live):
        sim.house.cut = 1.0
        return
    sim.house.cut = min(1.0, sum(p["kw"] for p in live) / max(1e-6, f["cool_kw"]))


def _add(sim: Any, device: str, kw: float, total: float) -> None:
    dev = sim.levers["dev"]
    dev[device] = min(1.0, dev.get(device, 0.0) + kw / max(1e-6, total))


def _take(sim: Any, device: str, kw: float, total: float) -> None:
    dev = sim.levers["dev"]
    dev[device] = max(0.0, dev.get(device, 0.0) - kw / max(1e-6, total))


def apply_call(sim: Any, zm: Any, call: dict[str, Any], ans: dict[str, Any], d: dict[str, Any], f: dict[str, Any]) -> None:
    tool, v, lev, kw = call["tool"], ans["applied"], sim.levers, float(ans["expected_kw"])
    t = ans["target"]
    if lev is None:
        return                                      # the morning's pre-cool: BuildingAgent._pre_cool plays it
    if tool == "set_battery":
        lev["battery_kw"] = float(v.get("kw", 0.0)) if v.get("action") == "discharge" else 0.0
    elif tool == "limit_ev_charging":
        lev["dev"]["ev"] = 1.0 if not v.get("amps") else min(1.0, kw / max(1e-6, d["kw_now"]))
    elif tool == "set_plant_demand_limit":
        lev["plants"][t] = {"kw": kw, "from": call["start"], "to": call["end"]}
        sync_plant(sim, f)
    elif tool == "stop_plant_early":
        lev["plants"][t] = {"kw": kw, "stop": True, "from": call["start"], "to": call["end"]}
        sync_plant(sim, f)
        sync_comfort(sim, zm, f["free"])
    elif tool == "set_zone_cooling":
        c = float(v.get("offset_c", 0.0))
        if c > 0:
            lev["zones"].update({s: c for s in d["spaces"]})
            _until(lev, "zones", d["spaces"], call["end"], f)
            sync_comfort(sim, zm, f["free"])
            if d.get("tenant_fan_kw"):
                _add(sim, "fans", d["tenant_fan_kw"] * min(1.0, c / 1.5), f["fans_kw"])
    elif tool == "set_zone_standby":
        lev["standby"] = (lev["standby"] | set(d["spaces"])) if v.get("mode") == "on" else (lev["standby"] - set(d["spaces"]))
        if v.get("mode") == "on":
            _until(lev, "standby", d["spaces"], call["end"], f)
    elif tool == "limit_air_handler":
        lev["fans"][t] = kw
        _add(sim, "fans", kw, f["fans_kw"])
    elif tool == "dim_lights":
        lev["lights"].update({s: float(v.get("cut_percent", 0.0)) for s in d["spaces"]})
        lev["light_share"] = min(1.0, lev["light_share"] + kw / max(1e-6, sum(sim.house.light_kw(x) for x in f["steps"]) / max(1, len(f["steps"]))))
    elif tool == "switch_plugs":
        if v.get("state") == "off":
            lev["plugs"][t] = kw
            _add(sim, "plugs", kw, _plugs_kw(sim, f))
        else:
            _add(sim, "plugs", -lev["plugs"].pop(t, 0.0), _plugs_kw(sim, f))


def _until(lev: dict[str, Any], kind: str, spaces: list[str], end: str, f: dict[str, Any]) -> None:
    """A write ends by itself. Most end with the sale, when the levers close. A booking can end a zone's sooner, and that end is kept here."""
    if end < f["sale"]["end"]:
        lev["until"].update({(kind, s): end for s in spaces})


def _plugs_kw(sim: Any, f: dict[str, Any]) -> float:
    from .see import window_mean
    sale = f["sale"]
    return window_mean(sim, "plugs", sale["start"], sale["end"])


# --------------------------------------------------------------------------------------------------- the agent
class BuildingAgent(NodeAgent):
    word = "building"

    def __init__(self, home: Any, brain: Brain, record: Callable[[dict[str, Any]], None] | None = None, log: Any = None):
        self.zm = buildings.load(home.name)
        self.retry: dict[tuple[str, str], tuple[str, str]] = {}
        self.noticed: set[str] = set()
        self.seen_release: str | None = None
        self.prepared: str | None = None
        self.pre: dict[str, Any] | None = None
        self.held_devices: set[str] = set()         # what hold_out took out of the sale, so it can give back that and no more
        super().__init__(home, 0, brain, record, log)
        if WALK_BY_CODE and not arm.all_agents():
            self.prepare = self.prepare_by_code     # prepare() stays below as it was, for FG_WALK=model. Every node an agent: the model's pre-cool, the playbook as its fallback

    # ------------------------------------------------------------------ what differs by node
    def tools(self) -> list[str]:
        return T.building_shelf(self.zm.key)

    def make_guard(self) -> BuildingGuard:
        return BuildingGuard({"zones": self.zm.key})

    def view(self, ts: str, sale: dict[str, Any], owed: float | None = None) -> dict[str, Any]:
        f = see_building(self.sim, ts, sale, self.shelf, self.zm, self.owed_now(sale) if owed is None else owed)
        f["steps"] = steps_of(self.sim, max(ts, sale["start"]), sale["end"])
        self.view_plan(f)
        return f

    def words(self, f: dict[str, Any], owed: float, done: list[str], why: str | None, block: str | None, used: set[str]) -> str:
        return render(f, owed, done, why, block, used)

    def menu(self, f: dict[str, Any], used: set[str], reads: bool) -> list[dict[str, Any]]:
        return shortlist(f, used, reads)

    def limits_now(self) -> dict[str, float]:
        return {**super().limits_now(), "lights": self.sim.light_limit()}

    def shrink(self, f: dict[str, Any]) -> None:
        """The lights share one limit, so each call keeps its part of it. The rest give what their device can now give."""
        lev = self.sim.levers
        share, limit = float(lev.get("light_share", 0.0)), self.sim.light_limit()
        k = min(1.0, limit / share) if share > 0 else 1.0
        lev["light_share"] = min(share, limit)
        for c in self.calls:
            d = next((x for x in f["devices"].values() if x["tool"] == c["tool"] and x["target"] == c["target"]), None)
            if c["tool"] == "dim_lights":
                self.cap_call(c, c["expected_kw"] * k)
            elif d is not None and c["tool"] in ("set_zone_cooling", "limit_air_handler", "switch_plugs"):
                self.cap_call(c, float(d.get("gives_kw", d.get("can_give_kw", 0.0))))

    def gives(self, target: str, kw: float) -> str:
        return f"{target} gives {kw:,.0f} kW."

    def open(self) -> None:
        open_levers(self.sim)
        self.guard.clear()                          # an alarm or a lost model ends a sale without the usual close
        self.noticed.clear()
        self.retry.clear()
        self.seen_release = None

    def close(self) -> None:
        self.hold_out(set())
        close_levers(self.sim)

    def hold_out(self, keep: set[str]) -> None:
        """What a wish keeps out, with no levers open: the building is the controller's, and these limits hold there. A note at a building with no agent was read, recorded and then did nothing:
        the lights dimmed 252.4 kW with it and without it. keep: the tools kept out, or nothing to hand the building back."""
        sim = self.sim
        parts = {owner.PART_OF.get(t) for t in keep}
        off = {d for p, d in HELD_DEVICES.items() if p in parts} & set(getattr(sim, "app", None) or ())
        sim.dev_off = (set(sim.dev_off) - self.held_devices) | off
        self.held_devices = off
        sim.held_out = parts
        if self.sim.levers is None:
            write_comfort(sim, comfort(sim, 0.0) if "cooling" in parts else None)

    def all_open(self, keep: set[str]) -> None:
        """For could_give: with no levers the plan may use every device. A wish or a closed group can't open all but a few, so then the sum stays with what's open."""
        if not keep:
            self.sim.levers = None

    def replan(self) -> None:
        """A building has more points than a home, and a stop or a cap can't be run twice: what's running stays, and the model picks from the rest."""

    def reopen(self, blocked: set[str]) -> None:
        """Take back what a manager's wish stops, at once. owner.py names a home's tools: a building's levers sit elsewhere."""
        super().reopen(blocked)
        lev = self.sim.levers
        if "dim_lights" in blocked:                 # keep_lights_on: every dim ends at once
            lev["lights"].clear()
            lev["light_share"] = 0.0
        if "limit_ev_charging" in blocked:
            lev["dev"].pop("ev", None)
        if "switch_plugs" in blocked:               # keep_machines_on
            lev["plugs"].clear()
            lev["dev"].pop("plugs", None)
        if blocked & {"set_thermostat", "set_room_ac"}:         # keep_house_as_is: the climate tools leave the sale
            lev["zones"].clear()
            lev["standby"].clear()
            lev["plants"].clear()
            lev["until"].clear()
            self.sim.house.cut = 0.0
            taken = sum(lev["fans"].values())
            lev["fans"].clear()
            if taken:
                lev["dev"]["fans"] = 0.0
            self.calls = [c for c in self.calls if c["tool"] not in CLIMATE]
            write_comfort(self.sim, comfort(self.sim, 0.0))

    def amount(self, tool: str, args: Any, f: dict[str, Any], owed: float) -> Any:
        """The list gives one ready number, which can be far more than is left to sell: with 3 kW owed the model took a light cut worth 88 kW, and the office warmed
        for kW the sale doesn't pay for. Code only lowers it here: the model picks the device, and what is left is arithmetic."""
        d = next((x for x in f["devices"].values() if x["tool"] == tool and x["target"] == args.get("target")), None)
        want = owed * CUSHION
        if d is None or want <= 0:
            return args
        if tool == "set_battery" and args.get("action") == "discharge" and isinstance(args.get("kw"), (int, float)):
            return {**args, "kw": round(min(float(args["kw"]), want), 1)}
        if tool == "dim_lights" and isinstance(args.get("cut_percent"), (int, float)) and args["cut_percent"] != 100:
            full = float(d.get("running_kw", 0.0)) * float(d.get("scale", 1.0))
            return {**args, "cut_percent": max(1, min(float(args["cut_percent"]), math.ceil(100 * want / max(1e-6, full))))} if full > 0 else args
        return args

    def expire(self, ts: str) -> None:
        """A call that ends before the sale does ends here. The next step played is the one after ts."""
        from ..beckn import add_minutes
        lev, nxt = self.sim.levers, add_minutes(ts, 15)
        gone = [k for k, e in lev["until"].items() if e <= nxt]
        for kind, s in gone:
            del lev["until"][(kind, s)]
            if kind == "zones":
                lev["zones"].pop(s, None)
            else:
                lev["standby"].discard(s)
        if gone:
            self.calls = [c for c in self.calls if c["tool"] not in ("set_zone_cooling", "set_zone_standby") or not any(s in c.get("spaces", ()) for _, s in gone)]

    # ------------------------------------------------------------------ times, from the cards
    def times(self, tool: str, a: dict[str, Any], d: dict[str, Any] | None, sale: dict[str, Any], ts: str) -> tuple[str, str]:
        """In a sale step code fills start and end. Where people may be in, the ramp starts 30 minutes early, so it ends as the window opens."""
        from ..beckn import add_minutes
        k = (tool, str(a.get("target", "")))
        if k in self.retry:
            return self.retry.pop(k)
        s, e = sale["start"], sale["end"]
        t0 = max(ts, s)
        lead = max(ts, add_minutes(s, -RAMP_MIN))
        if tool == "dim_lights":
            return (ts if a.get("cut_percent") == 100 else lead), e
        if tool == "set_zone_cooling" and (d or {}).get("state") != "free":
            return lead, e
        if tool == "set_zone_standby":
            return max(ts, add_minutes(s, -STANDBY_LEAD_MIN)), e
        return t0, e

    # ------------------------------------------------------------------ one step
    def decide(self, ts: str, sale: dict[str, Any], why: str | None = None) -> None:
        if self.sim.levers is None or self.sim.overrides.get("stop"):
            return                                  # an alarm, a stop or a lost model took the node out of this sale
        kept = self.ctx._kept()
        used = {f'{c["tool"]}:{c["target"]}' for c in self.calls}
        done = [c["says"] for c in self.calls] + [owner.SAYS[w["wish"]] for w in self.guard.wishes]
        block, retried, reads = None, False, True
        tries: dict[str, int] = {}
        for _ in range(MAX_ASKS):
            f = self.view(ts, sale)
            used |= {key(d) for d in f["devices"].values() if d["tool"] in kept}    # a wish takes the tool off the list as well as out of the guard
            owed = self.owed_now(sale)
            if owed <= 0.05 and not why:
                break                               # nothing owed: code doesn't call the model at all
            live = [d for d in f["devices"].values() if key(d) not in used and d["tool"] not in kept and can_act(f, d)]
            if not live:
                self.nothing_left(ts, why, owed)
                break                               # nothing left that could give anything
            text = self.words(f, owed, done, why, block, used)
            tools = self.ctx.menu(f, used, reads)
            a = self.brain.ask(T.SYSTEM, tools, text)
            rec = {"node": self.name, "tier": 0, "ts": ts, "why": why, "text": text, "shortlist": [x["function"]["name"] for x in tools], "answer": a["calls"], "said": a["text"],
                   "ms": a["ms"], "model": self.brain.model, "prompt": T.PROMPT_VERSION, "owed_kw": round(owed, 2), "verdicts": []}
            block, stop = None, False
            if a.get("error"):
                rec["fallback"] = "the model did not answer, so the controller may use every device"
                self.ctx.hand_back()
                self.record(rec)
                return
            if not a["calls"]:
                self.record(rec)
                if retried:
                    return
                retried, block = True, "Call one tool."
                continue
            for c in a["calls"]:
                if c["name"] == "no_action":
                    stop = True
                    continue
                if c["name"] == "read_device":
                    reads = False                   # it answers once, and the answer goes into the next prompt
                    done.append("Reading: " + str(self.ctx.call("read_device", c["args"]).get("view") or "nothing to read."))
                    continue
                if c["name"] in context.CONTEXT:
                    done.append(self.ctx.call(c["name"], c["args"])["view"])
                    continue                        # a report doesn't end the step: on 9 Jul it did, with 105 kW still closed
                c = c if isinstance(c.get("args"), dict) else {**c, "args": {}}
                f = self.view(ts, sale)             # an earlier call in this answer may have changed what's true
                d = next((x for x in f["devices"].values() if x["tool"] == c["name"] and x["target"] == c["args"].get("target")), None)
                c = {**c, "args": self.amount(c["name"], c["args"], f, self.owed_now(sale))}
                t0, t1 = self.times(c["name"], c["args"], d, sale, ts)
                call = {"tool": c["name"], "args": c["args"], "start": t0, "end": t1}
                ans = self.ctx.act(c["name"], c["args"], start=t0, end=t1, facts=f)
                rec["verdicts"].append({"call": call, "answer": ans})
                self.log_calls.append({"target": ans["target"], "status": ans["status"], "expected_kw": ans["expected_kw"], "why": ans["why"]})
                if ans["status"] in ("done", "adjusted"):
                    says = f'{ans["target"]} gives {ans["expected_kw"]:,.0f} kW.' if ans["expected_kw"] > 0 else f'{ans["target"]} is set.'
                    if ans["applied"].get("state") == "auto":       # plugs handed back: their kW no longer count against what is owed
                        self.calls = [x for x in self.calls if (x["tool"], x["target"]) != (c["name"], ans["target"])]
                    self.calls.append({"tool": c["name"], "target": ans["target"], "expected_kw": ans["expected_kw"], "says": says, "spaces": (d or {}).get("spaces", [])})
                    used.add(f'{c["name"]}:{ans["target"]}')
                    done.append(says)
                else:
                    k = f'{c["name"]}:{c["args"].get("target")}'
                    tries[k] = tries.get(k, 0) + 1
                    if tries[k] > 1:
                        used.add(k)                 # one retry a step
                    ti = dict(ans.get("try_instead") or {})
                    if ti.get("start") and ti.get("end") and ti.get("tool"):
                        day = ts[:10]
                        self.retry[(ti["tool"], str(ti.get("target", "")))] = (f'{day} {ti["start"]}', f'{day} {ti["end"]}')
                    args = ", ".join(f"{k}={json.dumps(v)}" for k, v in ti.items() if k not in ("tool", "start", "end"))
                    block = f'The guard answered: status blocked. Why: {ans["why"]}' + (f' try_instead: {ti.get("tool")}({args}).' if ti else "")
            self.record(rec)
            why = None
            if stop:
                return

    # ------------------------------------------------------------------ the morning of a sale
    def prepare(self, ts: str, sale: dict[str, Any]) -> None:
        """Before a cooling sale: pre-cool, so the stored cold is there when the sale opens. One model answer, through the same guard."""
        if self.sim.overrides.get("stop"):
            return                                  # a closed building or a person's stop: no sale to cool for
        day = ts[:10]
        t0, t1 = (f"{day} {x}" for x in PRE_COOL[self.zm.key])
        f = see_building(self.sim, ts, sale, self.shelf, self.zm, 0.0)
        whole = next((x for x in self.zm.targets["set_zone_cooling"] if x.get("whole")), None)
        d = f["devices"].get(f'set_zone_cooling:{whole["target"]}') if whole else None
        if not d or not d["running"] or d.get("exam") or d.get("avoid") or d.get("smoke_ok") is False:
            return
        depth = float(d["precool_c"])
        text = (f'{f["day"]}, {ts[11:16]} {f["place"]}. A sale runs {sale["start"][11:16]} to {sale["end"][11:16]} today and cooling will be running then. '
                f'Pre-cool runs {t0[11:16]} to {t1[11:16]}. {d["target"]} may go {-depth:g} C. No alarms.')
        tools = [T.spec("building", "set_zone_cooling", [d["target"]]), T.spec("building", "no_action")]
        self.ctx.show(tools)
        a = self.brain.ask(T.SYSTEM, tools, text)
        if arm.all_agents() and not a.get("error") and self.precool_without_reason(a):       # no pre-cool and no word of why: asked once more
            a = self.brain.ask(T.SYSTEM, tools, text + " " + PRECOOL_AGAIN)
            if not a.get("error") and self.precool_without_reason(a):
                return self.prepare_by_code(ts, sale, flags=[NO_REASON])
        if arm.all_agents() and not a.get("error") and self.shallower_without_reason(a, depth):     # less than the playbook's depth, and no word of why: the same
            a = self.brain.ask(T.SYSTEM, tools, text + " " + PRECOOL_LESS_AGAIN.format(depth=depth))
            if not a.get("error") and (self.precool_without_reason(a) or self.shallower_without_reason(a, depth)):
                return self.prepare_by_code(ts, sale, flags=[NO_REASON])
        if a.get("error") and arm.all_agents():     # every node an agent: the model failed, so the playbook's pre-cool stands for this point, counted
            self.fallbacks["pre-cool"] += 1
            return self.prepare_by_code(ts, sale, fallback=f'the model did not answer: {str(a["error"])[:120]}')
        rec = {"node": self.name, "tier": 0, "ts": ts, "why": "pre-cool", "text": text, "shortlist": [x["function"]["name"] for x in tools], "answer": a["calls"], "said": a["text"], "ms": a["ms"],
               "model": self.brain.model, "prompt": T.PROMPT_VERSION, "owed_kw": 0.0, "verdicts": [], "by": "model"}
        for c in a["calls"]:
            if c["name"] != "set_zone_cooling":
                continue
            call = {"tool": c["name"], "args": c["args"], "start": t0, "end": t1}
            ans = self.ctx.act(c["name"], c["args"], start=t0, end=t1, facts={**f, "sale": None})
            rec["verdicts"].append({"call": call, "answer": ans})
            v = float(ans["applied"].get("offset_c", 0.0))
            if ans["status"] in ("done", "adjusted") and v < 0:
                self.sim.hold_cold = True
                self.pre = {"c": v, "from": t0, "until": t1}
        if arm.all_agents():
            chose = (self.pre or {}).get("c", 0.0) if self.pre and self.pre.get("from") == t0 else 0.0
            said = next((str((c.get("args") or {}).get("reason") or "") for c in a["calls"] if c["name"] == "no_action" and isinstance(c.get("args"), dict)), "") \
                or str(a.get("text") or "").strip()[:300]          # a shallower pre-cool gives its reason beside the call
            asked = [float((c.get("args") or {}).get("offset_c") or 0.0) for c in a["calls"] if c["name"] == "set_zone_cooling" and isinstance(c.get("args"), dict)]
            want = min(asked) if asked else 0.0             # what the agent asked for: a guard that trims it is the guard's, and its verdict says so
            less = want > -depth + 1e-9
            told = (f"I didn't pre-cool the building this morning. {said}" if want == 0.0 else
                    f"I pre-cooled the building by {abs(want):g} C this morning, not the {depth:g} C the playbook allows. {said}") if less and said else None
            if told:                                        # the receipt keeps the words the manager heard
                self.tell(told, "pre-cool", ts)
            self.receipt("pre-cool", ts, asked={"sale": [sale["start"], sale["end"]], "may_go_c": -depth}, rule={"offset_c": -depth, "from": t0, "until": t1},
                         chose={"offset_c": want, "applied_c": chose}, reason=said, facts=[], flags=[], by="agent", less=less, told=told)
        self.record(rec)
        self.guard.clear()

    @staticmethod
    def shallower_without_reason(a: dict[str, Any], depth: float) -> bool:
        """The model pre-cools, but less than the playbook's depth, and says no word of why: no reason beside the call, and no no_action reason."""
        asked = [float((c.get("args") or {}).get("offset_c") or 0.0) for c in a["calls"] if c["name"] == "set_zone_cooling" and isinstance(c.get("args"), dict)]
        if not asked or min(asked) <= -depth + 1e-9:
            return False
        said = str(a.get("text") or "").strip() or any(str((c.get("args") or {}).get("reason") or "").strip() for c in a["calls"]
                                                        if c["name"] == "no_action" and isinstance(c.get("args"), dict))
        return not said

    def precool_without_reason(self, a: dict[str, Any]) -> bool:
        """The model answered with no pre-cool, and no word of why."""
        if any(c["name"] == "set_zone_cooling" for c in a["calls"]):
            return False
        return not any(str((c.get("args") or {}).get("reason") or "").strip() for c in a["calls"] if c["name"] == "no_action" and isinstance(c.get("args"), dict))

    def _pre_cool(self, ts: str) -> None:
        """The controller's side of a pre-cool: the setpoint comes down slowly, or the pull-down is a new morning peak. At its end the write ends
        by itself, and the stored cold stays in the mass."""
        p = self.pre
        if not p or ts < p["from"]:
            return
        if self.sim.overrides.get("stop"):          # a person took the node out: this write ends now
            self.pre = None
            self.sim.hold_cold = False
            write_comfort(self.sim)
            return
        if self.sim.levers is not None:             # the sale's levers are open: its offsets decide now
            self.pre = None
            return
        from ..beckn import add_minutes
        if add_minutes(ts, 15) >= p["until"]:       # the next step played is the one after ts
            self.pre = None
            write_comfort(self.sim)
            return
        c = p["c"] * min(1.0, (self.sim.replay.minutes_between(p["from"], ts) + 15) / PRE_RAMP_MIN)
        write_comfort(self.sim, {"max_rise_c": c, "max_rise_away_c": c})

    # ------------------------------------------------------------------ when to wake
    def watch(self, ts: str, sale: dict[str, Any]) -> str | None:
        """The schedule watch: sensors far above the plan mean an event, so the rise falls now, and the agent hears why."""
        out = None
        for s, sp in self.zm.spaces.items():
            if s not in self.noticed and self.zm.true(s, ts) - self.zm.plan(s, ts) >= buildings.EVENT:
                self.noticed.add(s)
                self.sim.levers.setdefault("events", set()).add(s)
                out = f'{sp["names"]} read {self.zm.true(s, ts) * 100:.0f}% in use, and the plan said {self.zm.plan(s, ts) * 100:.0f}%.'
        return out

    def step(self, ts: str) -> None:
        alarm = self.sim.overrides.get("alarm") or self.sim.overrides.get("alarm_hold") or self.ctx.stopped
        if alarm:
            self.ctx.alarm(alarm, ts)               # nothing is sent and every agent limit releases
            self.calls = []
            self.guard.clear()
            self.pre = None                         # the morning's pre-cool is an agent limit too
            self.sim.hold_cold = False
            write_comfort(self.sim)
            return
        from ..beckn import add_minutes
        sale = self.sale(ts)
        self._pre_cool(ts)
        if sale and self.prepared != sale["start"] and ts < add_minutes(sale["start"], -LEAD_MIN):
            at = add_minutes(f'{ts[:10]} {PRE_COOL[self.zm.key][0]}', -PRE_LEAD_MIN)
            if at <= ts < f'{ts[:10]} {PRE_COOL[self.zm.key][1]}':
                self.prepared = sale["start"]
                self.steps.append(("6 prepare", ts))
                self.ctx.enter("6 prepare", ts, sale)
                self.prepare(ts, sale)
                self.ctx.leave()
        running = self.sim.levers is not None
        super().step(ts)
        sale = self.sale(ts)
        if not running or self.sim.levers is None or not sale or ts >= sale["end"] or ts < sale["start"]:
            self.wake_zones(ts)
            return
        why = self.watch(ts, sale)
        free = free_shares(self.zm, ts, steps_of(self.sim, ts, sale["end"]))
        had = bool(self.sim.levers["zones"])
        self.expire(ts)
        if had or any(p.get("stop") for p in self.sim.levers["plants"].values()):
            sync_comfort(self.sim, self.zm, free)
        released = self.sim.levers.get("released")
        if released and released != self.seen_release:
            self.seen_release = released
            for t, p in self.sim.levers["plants"].items():
                p["released"] = True                # called and used up: it doesn't come back in this sale
                self.log_calls.append({"target": t, "status": "released", "expected_kw": 0.0, "why": f"Zones reached their limit at {ts[11:16]}, so the plant came back."})
            self.calls = [c for c in self.calls if c["tool"] not in ("set_plant_demand_limit", "stop_plant_early")]
            why = "The plant was released: zones reached their limit."
        if why:
            self.last_wake = ts
            self.steps.append(("8 exception", ts))
            self.ctx.enter("8 exception", ts, sale)
            self.decide(ts, sale, why)
            self.ctx.leave()
        self.wake_zones(ts)

    def wake_zones(self, ts: str) -> None:
        """Standby stops a zone's outdoor air, so whatever the agent decides, it ends once the zone's sensors read people. The plan said free, and on 1 Jul a campus zone filled to 23% with standby on."""
        lev = self.sim.levers
        if lev is not None and lev.get("standby"):
            lev["standby"] = {x for x in lev["standby"] if self.zm.true(x, ts) < buildings.BUSY}

    # ------------------------------------------------------------------ what a call needs, and what it costs
    def can(self, f: dict[str, Any], d: dict[str, Any]) -> bool:
        return can_act(f, d)

    def may_decide(self) -> bool:
        return self.sim.levers is not None and not self.sim.overrides.get("stop")     # an alarm, a stop or a lost model took the node out of this sale

    def ready(self, d: dict[str, Any], f: dict[str, Any], owed: float) -> dict[str, Any]:
        tool, t = d["tool"], d["target"]
        if tool == "set_battery":
            return {"target": t, "action": "discharge", "kw": d["can_give_kw"]}
        if tool == "set_plant_demand_limit":
            return {"target": t, "limit_percent": d["cap_percent"]}
        if tool == "set_zone_cooling":
            return {"target": t, "offset_c": EMPTY_C if d["state"] in ("free", "part") else OPEN_C}
        if tool == "limit_air_handler":
            return {"target": t, "speed_percent": d["floor_percent"]}
        if tool == "set_zone_standby":
            return {"target": t, "mode": "off" if d.get("event") else "on"}
        if tool == "dim_lights":
            return {"target": t, "cut_percent": d["limit_pct"]}
        if tool == "switch_plugs":
            return {"target": t, "state": "off"}
        if tool == "limit_ev_charging":
            return {"target": t, "amps": 0}
        return {"target": t}

    def took(self, tool: str, ans: dict[str, Any], d: dict[str, Any] | None) -> None:
        says = f'{ans["target"]} gives {ans["expected_kw"]:,.0f} kW.' if ans["expected_kw"] > 0 else f'{ans["target"]} is set.'
        if ans["applied"].get("state") == "auto":       # plugs handed back: their kW no longer count against what is owed
            self.calls = [x for x in self.calls if (x["tool"], x["target"]) != (tool, ans["target"])]
        self.calls.append({"tool": tool, "target": ans["target"], "expected_kw": ans["expected_kw"], "says": says, "spaces": (d or {}).get("spaces", [])})

    def refused(self, ans: dict[str, Any], ts: str) -> None:
        ti = dict(ans.get("try_instead") or {})
        if ti.get("start") and ti.get("end") and ti.get("tool"):
            self.retry[(ti["tool"], str(ti.get("target", "")))] = (f'{ts[:10]} {ti["start"]}', f'{ts[:10]} {ti["end"]}')

    def next_device(self, f: dict[str, Any], live: list[dict[str, Any]], used: set[str]) -> dict[str, Any] | None:
        """The top of the list, except that what a device waits for goes first. Zones at 0 kW sort last, so they are eased only when nothing else covers what is owed: then the plant opens."""
        top = live[0] if live else None
        first = self.prerequisite(top, f) if top else None
        return first if first is not None and self.key(first) not in used else top       # tried and still waiting: the guard says so, and the walk goes on

    def prerequisite(self, d: dict[str, Any] | None, f: dict[str, Any]) -> dict[str, Any] | None:
        e = d.get("ease_with") if d and d["tool"] == "limit_air_handler" and not d.get("eased") else None
        return f["devices"].get(f'{e["tool"]}:{e["target"]}') if e else None

    def after_test(self, f: dict[str, Any], d: dict[str, Any] | None, args: dict[str, Any]) -> dict[str, Any]:
        """Zones eased in a test ease the air handlers on their spaces for the calls after it, as the view reads them once the call has run (building_see: eased)."""
        if not d or not ((d["tool"] == "set_zone_cooling" and float(args.get("offset_c") or 0.0) > 0) or (d["tool"] == "set_zone_standby" and args.get("mode") == "on")):
            return f
        sp = set(d.get("spaces") or ())
        return {**f, "devices": {k: {**x, "eased": True} if x["tool"] == "limit_air_handler" and sp & set(x.get("spaces") or ()) else x for k, x in f["devices"].items()}}

    def test_unlocked(self, f: dict[str, Any], done: set[str]) -> list[dict[str, Any]]:
        """The plant a test's zones unlock, which code caps after the plan's calls (walk_unlocked). The zone's letter counts its kW, so the test does too: on 14 Jul a test of
        code's own plan said 175 kW would stay owed where the plan left 92."""
        kw = sum(float(f["devices"][k].get("behind_kw") or 0.0) for k in done if k in f["devices"] and f["devices"][k]["tool"] == "set_zone_cooling")
        if kw <= 0:
            return []
        plants = [d["target"] for d in f["devices"].values() if d["tool"] == "set_plant_demand_limit" and not d.get("show")]
        return [{"tool": "set_plant_demand_limit", "target": ", ".join(plants) or "the plant", "status": "done", "kw": round(kw, 2), "why": "Code caps it after the plan's calls, once these zones coast."}]

    def would_refuse(self, d: dict[str, Any], f: dict[str, Any], owed: float, ts: str, sale: dict[str, Any]) -> str:
        """A device that waits for its zones is asked as it will be when a plan opens it: code sets what it waits for first (first_needs), so that is asked first, and this one
        as the view reads it once that went through. If the guard would refuse what it waits for, it can't open either. Asked alone, the office's air handlers came back blocked."""
        held = dict(self.retry)                     # times() takes a try_instead's times once: a test must not use them up
        try:
            pre = self.prerequisite(d, f)
            if pre is None:
                return super().would_refuse(d, f, owed, ts, sale)
            why = super().would_refuse(pre, f, owed, ts, sale)
            if why:
                return f'It needs {pre["target"]} first, and the guard would refuse that: {why}'
            return super().would_refuse(d, self.after_test(f, pre, self.ready(pre, f, owed)), owed, ts, sale)
        finally:
            self.retry = held

    def first_needs(self, d: dict[str, Any], f: dict[str, Any], done: set[str]) -> dict[str, Any] | None:
        x = self.prerequisite(d, f)
        if x is None or self.key(x) in done:
            return None
        done.add(self.key(x))
        return {"tool": x["tool"], **self.ready(x, f, self.owed_now(f["sale"]))}

    def walk_unlocked(self, ts: str, sale: dict[str, Any], done: set[str], why: str) -> None:
        """Zones eased in a plan open the plant behind them, and that follows by itself: the letter already counted its kW."""
        kept = self.kept_now()
        for _ in range(4):
            f = self.view(ts, sale)
            owed = self.owed_now(sale)
            d = next((x for x in self.live(f, done, kept) if x["tool"] == "set_plant_demand_limit" and x.get("can_give_kw", 0.0) > 0), None)
            if owed <= 0.05 or d is None:
                return
            self.code_call(d, f, owed, ts, sale, why, "Code opened the plant the plan's zones unlocked.")
            done.add(self.key(d))

    def view_plan(self, f: dict[str, Any]) -> None:
        """Zones that wait on nothing but open a plant carry the plant's kW on their letter."""
        behind = sum(float(d.get("can_give_kw", 0.0)) for d in f["devices"].values() if d["tool"] == "set_plant_demand_limit" and not d.get("show"))
        for d in f["devices"].values():
            if d["tool"] == "set_zone_cooling" and d.get("unlocks"):
                d["behind_kw"] = behind
                behind = 0.0

    def cost_of(self, d: dict[str, Any], f: dict[str, Any]) -> str:
        tool = d["tool"]
        if tool == "set_battery":
            return "stored energy runs down toward the reserve"
        if tool == "dim_lights":
            return "people in these rooms see dimmer lights"
        if tool == "set_zone_cooling":
            return f'rooms warm up to +{EMPTY_C if d["state"] in ("free", "part") else OPEN_C:g} C, and booked rooms stay at +{OPEN_C:g} C at most'
        if tool == "limit_air_handler":
            return "fans slow, and booked rooms keep their air"
        if tool == "set_zone_standby":
            return "free rooms stop getting air until people come"
        if tool == "switch_plugs":
            return f'{d.get("holds") or "these outlets"} go off'
        if tool == "limit_ev_charging":
            return "drivers charge later"
        if tool == "set_plant_demand_limit":
            return "zones coast warmer"
        if tool == "stop_plant_early":
            return "no cooling from this plant until the sale ends"
        return "none known"

    def weighs(self, d: dict[str, Any]) -> bool:
        return d["tool"] in ("set_zone_cooling", "set_plant_demand_limit", "stop_plant_early", "dim_lights", "switch_plugs", "limit_air_handler")

    def option_line(self, d: dict[str, Any], f: dict[str, Any]) -> str:
        """The card tests' sentence, without its orders: a plan's letters carry those. What opens more behind it says so."""
        s = re_order.sub("", line(d, f)).strip()
        return s + (f' It also opens about {d["behind_kw"]:,.0f} kW at the plant.' if d.get("behind_kw") else "")

    def still_full(self) -> list[str]:
        """The spaces the schedule watch caught that still read far above the plan. One that has emptied again is no fact, however it read an hour ago."""
        now = self.ctx.ts
        return [s for s in sorted((self.sim.levers or {}).get("events") or ()) if self.zm.true(s, now) - self.zm.plan(s, now) >= buildings.EVENT]

    def touched(self, c: dict[str, Any], f: dict[str, Any]) -> bool:
        """People are in a zone this call holds: the lights, the air, the warmth."""
        return bool(set(self.still_full()) & set(c.get("spaces") or ()))

    def facts_at_hand(self, f: dict[str, Any], why: str) -> list[dict[str, Any]]:
        out = super().facts_at_hand(f, why)
        now = self.ctx.ts
        for s in self.still_full():                 # about these rooms and no others
            out.append(fact(f'{self.zm.spaces[s]["names"]} read {self.zm.true(s, now) * 100:.0f}% in use, and the plan said {self.zm.plan(s, now) * 100:.0f}%.', f"space:{s}", spaces=[s]))
        if float(self.sim.x) >= 0.8 * OPEN_C:
            out.append(fact(f'The building already runs {float(self.sim.x):.2f} C above its own day, and open rooms stop at +{OPEN_C:g} C.', "warm", ("cooling",)))
        return out

    def let_go(self, target: str) -> None:
        """One call ends, and only what that call opened closes. The controller's side of apply_call, backwards."""
        lev, sim = self.sim.levers, self.sim
        gone = [c for c in self.calls if c["target"] == target]
        self.calls = [c for c in self.calls if c["target"] != target]
        for points in self.guard.writers.values():
            points[:] = [w for w in points if w.get("target") != target]
        if lev is None or not gone:
            return
        f = self.ctx.facts() if self.ctx.ts else {"free": {s: 0.0 for s in self.zm.spaces}, "fans_kw": 0.0, "steps": [], "cool_kw": 0.0}     # a person's word can come with no step running
        for c in gone:
            tool, spaces = c["tool"], c.get("spaces") or []
            if tool == "set_battery":
                lev["battery_kw"] = 0.0
            elif tool == "limit_ev_charging":
                lev["dev"].pop("ev", None)
            elif tool in ("set_plant_demand_limit", "stop_plant_early"):
                lev["plants"].pop(target, None)
                sync_plant(sim, f)
                sync_comfort(sim, self.zm, f["free"])
            elif tool == "set_zone_cooling":
                for s in spaces:
                    lev["zones"].pop(s, None)
                    lev["until"].pop(("zones", s), None)
                sync_comfort(sim, self.zm, f["free"])
            elif tool == "set_zone_standby":
                lev["standby"] -= set(spaces)
                for s in spaces:
                    lev["until"].pop(("standby", s), None)
            elif tool == "limit_air_handler":
                _take(sim, "fans", lev["fans"].pop(target, 0.0), f["fans_kw"])
            elif tool == "dim_lights":
                for s in spaces:
                    lev["lights"].pop(s, None)
                still = any(x["tool"] == "dim_lights" for x in self.calls)
                on = sum(sim.house.light_kw(x) for x in f["steps"]) / max(1, len(f["steps"]))
                lev["light_share"] = max(0.0, lev["light_share"] - float(c["expected_kw"]) / max(1e-6, on)) if still else 0.0
            elif tool == "switch_plugs":
                _take(sim, "plugs", lev["plugs"].pop(target, 0.0), _plugs_kw(sim, f))

    # ------------------------------------------------------------------ one step
    def prepare_by_code(self, ts: str, sale: dict[str, Any], fallback: str | None = None, flags: list[str] | None = None) -> None:
        """The morning's pre-cool: one number and no choice in it, so no model answer. 93 of 93 answers copied the number shown. The same call, guard and record."""
        if self.sim.overrides.get("stop"):
            return
        day = ts[:10]
        t0, t1 = (f"{day} {x}" for x in PRE_COOL[self.zm.key])
        f = see_building(self.sim, ts, sale, self.shelf, self.zm, 0.0)
        whole = next((x for x in self.zm.targets["set_zone_cooling"] if x.get("whole")), None)
        d = f["devices"].get(f'set_zone_cooling:{whole["target"]}') if whole else None
        if not d or not d["running"] or d.get("exam") or d.get("avoid") or d.get("smoke_ok") is False:
            return
        args = {"target": d["target"], "offset_c": -float(d["precool_c"])}
        if hasattr(self.ctx, "show"):
            self.ctx.show([T.spec("building", "set_zone_cooling", [d["target"]])])
        rec = {**self.blank(ts, "pre-cool", 0.0, text=f'Code: pre-cool {t0[11:16]} to {t1[11:16]}, {d["target"]} to {args["offset_c"]:g} C.'), "tier": 0, "answer": [{"name": "set_zone_cooling", "args": args}],
               **({"fallback": fallback} if fallback else {})}
        ans = self.ctx.act("set_zone_cooling", args, start=t0, end=t1, facts={**f, "sale": None}, who=P.CODE)
        rec["verdicts"].append({"call": {"tool": "set_zone_cooling", "args": args, "start": t0, "end": t1}, "answer": ans})
        v = float(ans["applied"].get("offset_c", 0.0))
        if ans["status"] in ("done", "adjusted") and v < 0:
            self.sim.hold_cold = True
            self.pre = {"c": v, "from": t0, "until": t1}
        if arm.all_agents() and (fallback or flags):        # the rule's pre-cool, in place of the agent's: its receipt
            self.receipt("pre-cool", ts, asked={"sale": [sale["start"], sale["end"]], "may_go_c": args["offset_c"]}, rule={"offset_c": args["offset_c"], "from": t0, "until": t1},
                         chose={"offset_c": v}, reason=NO_REASON if flags and NO_REASON in flags else fallback or "", facts=[], flags=list(flags or []),
                         by=f"rule: {NO_REASON}" if flags and NO_REASON in flags else f"rule: {fallback}", less=False)
        self.record(rec)
        self.guard.clear()
