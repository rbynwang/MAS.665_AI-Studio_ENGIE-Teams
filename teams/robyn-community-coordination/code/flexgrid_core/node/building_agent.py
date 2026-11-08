"""A building's agent: NodeAgent with a building's tools, words, guard and levers.

The loop is NodeAgent's, with what a building adds:
  many targets per tool, so what is used is a target, not a tool
  times from the card: code fills start and end per tool, and a guard's try_instead keeps its own
  the schedule watch: sensors far above the plan wake the agent, and the rise the zones may take falls at once
  the plant's release: zones at their limit end a cap or a stop, and the agent is told
  on any alarm every agent limit releases at once

What a call changes on the simulated building is in apply_call, the controller's side. sim.levers holds it all.
"""
from __future__ import annotations

import json
import math
from typing import Any, Callable

from . import buildings, context, owner
from . import tools as T
from .agent import CUSHION, LEAD_MIN, Brain, NodeAgent
from .building_guard import CLIMATE, RAMP_MIN, BuildingGuard, mins
from .building_see import can_act, comfort_c, free_shares, key, render, see_building, shortlist, steps_of

MAX_ASKS = 12          # model answers in one step: a campus sale takes about eight device calls
STANDBY_LEAD_MIN = 15
PRE_COOL = {"university": ("10:00", "13:00"), "shopping_center": ("08:00", "11:00"), "office": ("07:00", "13:00")}   # the playbook, MULTI_AGENT_DESIGN section 8
PRE_LEAD_MIN = 10
PRE_RAMP_MIN = 120     # how long the pull-down takes: well inside the card's limit, 0.25 C a step


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
    sim.rt_override("comfort", comfort(sim, 0.0))
    sim.house.cut = 0.0
    sim.hold_cold = False


def close_levers(sim: Any) -> None:
    """Every write ends by itself, and the building goes back to its own day."""
    sim.levers = None
    sim.overrides.pop("comfort", None)
    sim.house.cut = 0.0


def sync_comfort(sim: Any, zm: Any, free: dict[str, float]) -> float:
    """The rise the building may reach now: from its zones' offsets, or a stopped plant's limits, and who is in the rooms."""
    lev = sim.levers
    zones = dict(lev["zones"])
    if any(p.get("stop") for p in lev["plants"].values()):
        zones = {s: 3.0 for s in zm.spaces}
    c = comfort_c(zm, zones, free)
    sim.rt_override("comfort", comfort(sim, c))
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
        super().__init__(home, 0, brain, record, log)

    # ------------------------------------------------------------------ what differs by node
    def tools(self) -> list[str]:
        return T.building_shelf(self.zm.key)

    def make_guard(self) -> BuildingGuard:
        return BuildingGuard({"zones": self.zm.key})

    def view(self, ts: str, sale: dict[str, Any], owed: float | None = None) -> dict[str, Any]:
        f = see_building(self.sim, ts, sale, self.shelf, self.zm, self.owed_now(sale) if owed is None else owed)
        f["steps"] = steps_of(self.sim, max(ts, sale["start"]), sale["end"])
        return f

    def words(self, f: dict[str, Any], owed: float, done: list[str], why: str | None, block: str | None, used: set[str]) -> str:
        return render(f, owed, done, why, block, used)

    def menu(self, f: dict[str, Any], used: set[str], reads: bool) -> list[dict[str, Any]]:
        return shortlist(f, used, reads)

    def open(self) -> None:
        open_levers(self.sim)
        self.guard.clear()                          # an alarm or a lost model ends a sale without the usual close
        self.noticed.clear()
        self.retry.clear()
        self.seen_release = None

    def close(self) -> None:
        close_levers(self.sim)

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
        if "limit_ev_charging" in blocked:
            lev["dev"].pop("ev", None)
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
            self.sim.rt_override("comfort", comfort(self.sim, 0.0))

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
        kept = self.guard.kept()
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
                    reads = False                   # the step already showed every reading it has
                    continue
                if c["name"] in context.CONTEXT:
                    done.append(self.ctx.call(c["name"], c["args"])["view"])
                    stop = stop or c["name"] == "report_shortfall"
                    continue
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
        a = self.brain.ask(T.SYSTEM, tools, text)
        rec = {"node": self.name, "tier": 0, "ts": ts, "why": "pre-cool", "text": text, "shortlist": [x["function"]["name"] for x in tools], "answer": a["calls"], "said": a["text"], "ms": a["ms"],
               "model": self.brain.model, "prompt": T.PROMPT_VERSION, "owed_kw": 0.0, "verdicts": []}
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
        self.record(rec)
        self.guard.clear()

    def _pre_cool(self, ts: str) -> None:
        """The controller's side of a pre-cool: the setpoint comes down slowly, or the pull-down is a new morning peak. At its end the write ends
        by itself, and the stored cold stays in the mass."""
        p = self.pre
        if not p or ts < p["from"]:
            return
        if self.sim.levers is not None:             # the sale's levers are open: its offsets decide now
            self.pre = None
            return
        from ..beckn import add_minutes
        if add_minutes(ts, 15) >= p["until"]:       # the next step played is the one after ts
            self.pre = None
            self.sim.overrides.pop("comfort", None)
            return
        c = p["c"] * min(1.0, (self.sim.replay.minutes_between(p["from"], ts) + 15) / PRE_RAMP_MIN)
        self.sim.rt_override("comfort", {"max_rise_c": c, "max_rise_away_c": c})

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
        alarm = self.sim.overrides.get("alarm") or self.ctx.stopped
        if alarm:
            self.ctx.alarm(alarm)                   # nothing is sent and every agent limit releases
            self.calls = []
            self.guard.clear()
            self.pre = None                         # the morning's pre-cool is an agent limit too
            self.sim.hold_cold = False
            self.sim.overrides.pop("comfort", None)
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
