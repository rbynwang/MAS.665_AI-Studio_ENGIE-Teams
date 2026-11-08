"""A node's agent: code owns the steps and the deadlines, and the model chooses inside a step (MULTI_AGENT_DESIGN, section 6).

The model is asked only when something needs judgment. Opening a sale has no real choice in it, so code walks the ready list (walk). When something changes in a sale and there is
a real trade-off, one answer decides the whole plan (plan, set_plan): what opens, what is let go, what the desk hears. Code sizes every number, the guard holds every limit, and
kW left uncovered or a call let go needs a fact from the prompt. Every other moment belongs to the 5-minute controller. Each call goes agent, controller, guard, device.

  Brain       one model answer with tools, through Ollama, the settings the tool tests used
  NodeAgent   decide(): walk() on step 7, plan() on step 8; step(): when to wake up
  open_levers / apply_call / close_levers   the controller's side: what a call changes on the simulated device

If the model is down the node falls back to code: the controller may use every device. FG_WALK=model puts back the old step, one model answer per device, for comparison.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from collections import Counter
from typing import Any, Callable

import httpx

from ..rt_control import inside, person_limit
from . import context, owner, report, sellable
from . import record as R
from . import policy as P
from . import tools as T
from .guard import Guard
from .see import MACHINE_TARGET, can_act, render, see_home, shortlist
from . import loans
from .grounds import quoted, stray_numbers
from . import arm

MODEL = "qwen3.8-27b-q3:latest"
URL = "http://127.0.0.1:11434"
OPTIONS = {"temperature": 0.3, "top_p": 0.8, "top_k": 20, "num_predict": 300, "num_ctx": 32768}
LEAD_MIN = int(os.environ.get("FG_LEAD_MIN", "30"))     # the agent plans the sale this long before it opens. A run on a fast clock needs more: 23 nodes take about 7 wall minutes
MAX_CALLS = 6          # device calls in one step
MAX_PLAN_ASKS = 3      # model answers in one exception: a test or a reading, then the whole plan
MAX_OPTIONS = 8        # lines in a plan's list of what can open, and of what runs. SPEC section 5's limit
COVERED_KW = 0.05      # owed at or below this is covered: a plan stops opening there, and nothing is uncovered
MAX_WALK = 16          # device calls code makes in one walk down the list
WALK_BY_CODE = os.environ.get("FG_WALK", "code") != "model"     # FG_WALK=model puts the model back on every device of the sale, one answer each: the comparison
CUSHION = 1.1          # a call is sized this far over what is owed: expected kW run high, and homes gave 92% of what they sold where code gives 108%
SHORT_SHARE = 0.15     # the plan may fall this share short before the agent is woken
SHORT_KW = 0.1
QUIET_MIN = 30         # at least this long between wake-ups for a shortfall
MACHINE_KEY = {v: k for k, v in MACHINE_TARGET.items()}
CODE_STEPS = ("1 read schedule", "2 forecast", "3 check devices", "4 estimate sellable")
OPENS = "The sale opens."   # what changed, when every node is an agent and the opening is a plan of its own (node/arm.py)
NO_REASON = "no reason given"           # a receipt's word for an answer that gave less than the rule's and said nothing of why, even when asked again
NO_FACT = "reason cites no fact"        # a receipt's word for a reason that rests on nothing in the agent's input: it stands, flagged
REASON_AGAIN = ("Your plan gives less than code's plan and says no reason. Call set_plan again, with your reason for the owner in reason, in plain words, "
                "and the fact it rests on in because.")


class DecideBackend:
    """Steps 5 and 6 go through brain.HomeBrain as they always did: this gives it the same model, offline."""

    def __init__(self, url: str = URL, model: str = MODEL):
        from ..llm import LLM
        self.llm = LLM(url, model, label="node")

    def decide(self, role: str, system: str, facts: dict[str, Any], schema: dict[str, Any], *, budget_s: float = 40.0, session: str | None = None):
        return self.llm.decide(system, facts, schema, budget_s=budget_s)


class Brain:
    def __init__(self, url: str = URL, model: str = MODEL):
        self.url, self.model = url, model
        self.http = httpx.Client(timeout=120)
        self.options = arm.model_options(OPTIONS)        # the run's temperature and seed, if its config sets them (node/arm.py)
        self.calls = 0
        self.ms = 0.0
        self.tokens_in = 0
        self.tokens_out = 0
        self.errors = 0

    def ask(self, system: str, tools: list[dict[str, Any]], text: str) -> dict[str, Any]:
        t0 = time.time()
        # truncate and shift as llm.py sends them: without them a request can hang after the model's runner restarts
        body = {"model": self.model, "stream": False, "think": False, "keep_alive": "30m", "truncate": False, "shift": False, "tools": tools, "options": self.options,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}]}
        try:
            r = self.http.post(self.url + "/api/chat", json=body)
            r.raise_for_status()                    # a refusal from the queue is a lost model, not an empty answer
            j = r.json()
        except (httpx.HTTPError, ValueError) as e:
            self.errors += 1
            return {"calls": [], "text": "", "error": str(e)[:200], "ms": 0}
        m = j.get("message") or {}
        calls = [{"name": c["function"]["name"], "args": c["function"].get("arguments") or {}} for c in (m.get("tool_calls") or [])]
        ms = (time.time() - t0) * 1000
        self.calls += 1
        self.ms += ms
        self.tokens_in += int(j.get("prompt_eval_count") or 0)
        self.tokens_out += int(j.get("eval_count") or 0)
        return {"calls": calls, "text": (m.get("content") or "")[:300], "ms": round(ms), "error": None}


# --------------------------------------------------------------------------------------------------- the controller's side
def write_comfort(sim: Any, value: dict[str, float] | None = None) -> None:
    """The agent's door to the slot the plan reads. It never passes what a person said, which is in comfort_limit and isn't the agent's to write. None hands the slot back to the person."""
    said = sim.overrides.get("comfort_limit")
    if value is not None:
        sim.overrides["comfort"] = inside(value, said)
    elif said:
        sim.overrides["comfort"] = dict(said)
    else:
        sim.overrides.pop("comfort", None)


def open_levers(sim: Any) -> None:
    """From now on the controller uses only what the agent's calls open."""
    # the car's own lever is its kW, so it stays in "devices" and gives nothing until a call sets that
    sim.levers = {"battery_kw": 0.0, "ev": False, "ev_kw": None, "v2h_kw": 0.0, "v2h_kwh": None, "devices": {"v2h"}}
    write_comfort(sim, held_comfort(sim))


def held_comfort(sim: Any) -> dict[str, float]:
    """No easing, and no pre-cooling below where the house is: extra cooling before a sale nobody opened is load nobody pays for."""
    return {"max_rise_c": 0.0, "max_rise_away_c": 0.0, "precool_c": max(0.0, -sim.x)}


def close_levers(sim: Any) -> None:
    """Every write ends by itself, and the device goes back to its own day."""
    sim.levers = None
    write_comfort(sim)


def apply_call(sim: Any, call: dict[str, Any], ans: dict[str, Any]) -> None:
    tool, v = call["tool"], ans["applied"]
    lev = sim.levers
    if tool == "set_battery":
        lev["battery_kw"] = float(v.get("kw", 0.0)) if v.get("action") == "discharge" else 0.0
    elif tool == "limit_ev_charging":
        lev["ev"] = True
        amps = float(v.get("amps", 0.0))
        lev["ev_kw"] = amps * 240 / 1000 if amps > 0 else None
    elif tool == "discharge_car":
        lev["v2h_kw"] = float(v.get("kw", 0.0))
        lev["v2h_kwh"] = float(v["max_kwh"]) if v.get("max_kwh") else None
        lev["ev"] = True        # one charger can't draw and give at once, so a car that gives holds its charger, as the controller does on rules. On 29 Jun hh-090's car
        #                         charged every other quarter hour in the sale, the charger free, and gave 21 of the 40 kWh it owed
    elif tool in ("set_thermostat", "set_room_ac"):
        c = float(v.get("less_comfort_c", 0.0))
        # each keeps its own limit: the call is sized for who is home now, and a home that fills up in the sale goes back to the tighter one
        write_comfort(sim, {**held_comfort(sim), "max_rise_c": min(c, person_limit(sim, "max_rise_c")), "max_rise_away_c": min(c, person_limit(sim, "max_rise_away_c"))})
    elif tool == "set_water_heater" and v.get("action") == "hold":
        lev["devices"].add("hpwh")
    elif tool == "set_pool_pump":
        lev["devices"].add("pool")
    elif tool == "delay_appliance":
        lev["devices"].add(MACHINE_KEY[ans["target"]])


DOING = {"set_battery": "is discharging", "limit_ev_charging": "is paused", "discharge_car": "is giving power", "set_thermostat": "is eased", "set_room_ac": "are eased", "set_water_heater": "is on hold",
         "delay_appliance": "is held back", "set_pool_pump": "runs slower", "set_zone_cooling": "run warmer", "set_zone_standby": "are on standby", "limit_air_handler": "are slowed",
         "set_plant_demand_limit": "is capped", "stop_plant_early": "is stopped", "dim_lights": "are dimmed", "switch_plugs": "are off"}      # what a running call is doing, so letting it go reads the right way round


def fact(says: str, key: str, about: tuple[str, ...] = (), spaces: list[str] | None = None) -> dict[str, Any]:
    """What a plan may cite. about: the parts of the node it bears on. spaces: for a fact about certain rooms, those rooms, and nothing else. key: which fact this is, however its figures move."""
    return {"says": says, "key": key, "about": list(about), **({"spaces": list(spaces)} if spaces else {})}


def bears_on(x: dict[str, Any], tool: str, spaces: Any = None, closed: bool = False) -> bool:
    """Does this fact bear on this call? A reading of one floor said nothing about a 1,000 kW battery, and ended it. About certain rooms, a fact can end a call that presses on them. To keep
    something closed (closed=True) all of it must lie in those rooms: a classroom reading kept every zone of the campus shut in an emergency, and 800 kW with them."""
    rooms = set(x.get("spaces") or ())
    if rooms:
        have = set(spaces or ())
        return bool(have) and (have <= rooms if closed else bool(have & rooms))
    return owner.PART_OF.get(tool) in x["about"]


def letters(x: Any) -> list[str]:
    """The ids an answer names, once each, in its order. A model may send one as plain text."""
    return list(dict.fromkeys(str(i) for i in x)) if isinstance(x, list) else re.findall(r"[A-Z]\d?", x) if isinstance(x, str) else []


# --------------------------------------------------------------------------------------------------- the agent
class NodeAgent:
    """A home's agent. A building's agent subclasses this and changes the six small methods under `what differs by node`."""
    word = "home"

    def __init__(self, home: Any, tier: int, brain: Brain, record: Callable[[dict[str, Any]], None] | None = None, log: Any = None):
        self.home, self.sim, self.tier, self.brain = home, home.adapter, tier, brain
        self.name = home.name
        self.shelf = self.tools()
        self.guard = self.make_guard()
        self.ctx = context.ContextServer(self, log=log)
        home.agent = self
        self.record = record or (lambda r: None)
        self.done_for: str | None = None        # the sale already planned
        self.calls: list[dict[str, Any]] = []    # the calls running now
        self.last_wake: str | None = None
        self.noted: str | None = self.ctx.noted         # the door's note already read: after a restart it isn't read again, and its wish isn't held twice
        self.tripped = False
        self.plugged = False
        self.waiting: Any = None
        self.short_reason: str | None = None        # what this agent told the desk about a shortfall: rt_report repeats it until the sale ends
        self.lent_on: str | None = None             # made for one decision at a code-only node: node/loans.py
        self.filed = self.ctx.notes_read            # the notes on file already read
        self.market_seen: dict[Any, set[str]] = {}  # the market steps each call for offers has run
        self.asked: Any = None                      # the picture the model last decided on
        self.let_gone: set[str] = set()             # calls a plan let go in this sale
        self.spent: set[str] = set()                # facts that have carried a decision in this sale
        if WALK_BY_CODE or arm.all_agents():
            self.decide = self.by_step              # decide() stays below as it was, for FG_WALK=model. Every node an agent: by_step's plans, the opening's too
        self.fallbacks: Counter = Counter()         # points code decided because the model failed, by why. Only with every node an agent (node/arm.py)
        self.flags: Counter = Counter()             # receipts flagged: no reason given, or a reason that cites no fact
        self.receipts: list[dict[str, Any]] = []    # one per decision, with every node an agent: also in the node's decision log
        self.told: list[dict[str, Any]] = []        # what the owner or manager was told when the node gave less than the rule's amount
        self.wish_notes: dict[str, str] = {}        # each wish's note, in the person's words: an offer under the wish cites it
        loans.open_desk(self)
        self.call_kw = 0.0                          # kW dispatched on top of the sale since the last step
        self.log_calls: list[dict[str, Any]] = []   # every call this sale, for step 9
        self.reported: str | None = None
        self.reports: list[dict[str, Any]] = []
        self.steps: list[tuple[str, str]] = []      # (step, when): the workflow's record
        self.seen_limits: dict[str, float] | None = None     # the limits in force at the last look, while a sale runs

    @property
    def car_first(self) -> bool:
        """A wish put the car first, and no person has said anything of it since. It lives on the node, so a restart keeps it, and a person's own word ends it."""
        return bool(getattr(self.sim, "car_first", False))

    # ------------------------------------------------------------------ what differs by node
    def tools(self) -> list[str]:
        return T.shelf(self.name, self.tier)

    def make_guard(self) -> Guard:
        return Guard(T.home_setup().get(T.home_id(self.name)))

    def view(self, ts: str, sale: dict[str, Any]) -> dict[str, Any]:
        f = see_home(self.sim, ts, sale, self.shelf)
        if self.home.cfg.get("forecast") == "trained_2026":     # see_home reads the true future for cooling and machines, so the forecast's numbers go over it
            f = sellable.overlay(f, self.sim, self.home._models())
        return f

    def words(self, f: dict[str, Any], owed: float, done: list[str], why: str | None, block: str | None, used: set[str]) -> str:
        return render(f, owed, done, why, block, used)

    def menu(self, f: dict[str, Any], used: set[str], reads: bool) -> list[dict[str, Any]]:
        return shortlist(f, used, reads)

    def amount(self, tool: str, args: Any, f: dict[str, Any], owed: float) -> Any:
        """The model picks the device, and the kW are code's arithmetic: on real runs it copied the card's example and sent 1.4 kW with 4.75 owed."""
        d = next((x for x in f["devices"].values() if x["tool"] == tool), None)
        if not isinstance(args, dict) or d is None:
            return args
        if tool in ("set_thermostat", "set_room_ac"):           # the degrees that give what is owed, as the guard counts them, up to the home's limit
            return {**args, "less_comfort_c": round(min(d["limit_c"], owed * CUSHION * d["cop"] * d["hours"] / max(1e-6, d["cap"])), 2)}
        if tool == "discharge_car" or (tool == "set_battery" and args.get("action") == "discharge"):
            out = {**args, "kw": round(min(owed * CUSHION, d["can_give_kw"]), 2)}
            return {**out, "max_kwh": d["max_kwh"]} if tool == "discharge_car" else out
        return args

    def open(self) -> None:
        open_levers(self.sim)

    def close(self) -> None:
        close_levers(self.sim)

    def apply(self, call: dict[str, Any], ans: dict[str, Any]) -> None:
        apply_call(self.sim, call, ans)

    def limits_now(self) -> dict[str, float]:
        return {k: person_limit(self.sim, k) for k in ("max_rise_c", "max_rise_away_c")}

    def shrink(self, f: dict[str, Any]) -> None:
        """A limit fell: a running call can't count for more than its device can now give."""
        for c in self.calls:
            if c["tool"] in ("set_thermostat", "set_room_ac"):
                self.cap_call(c, next((float(d["can_give_kw"]) for d in f["devices"].values() if d["tool"] == c["tool"]), c["expected_kw"]))

    def cap_call(self, c: dict[str, Any], kw: float) -> None:
        if kw < c["expected_kw"]:
            c.update(expected_kw=round(kw, 2), says=self.gives(c["target"], kw))

    def gives(self, target: str, kw: float) -> str:
        return f"{target} gives {kw:.2f} kW."

    # ------------------------------------------------------------------ the same for every node
    def sale(self, ts: str) -> dict[str, Any] | None:
        """Today's sale from the contract the node holds: its window and the most it owes."""
        day = ts[:10]
        owed = {t: float(a["deliver_kw"]) + float(self.sim.rt_extra.get(t, 0.0)) for t, a in self.sim.plan.items()
                if t[:10] == day and "deliver_kw" in a and float(a["deliver_kw"]) > 0}
        if not owed:
            return None
        from ..beckn import add_minutes
        groups: list[list[str]] = []
        for t in sorted(owed):
            if groups and t == add_minutes(groups[-1][-1], 15):
                groups[-1].append(t)
            else:
                groups.append([t])
        # the first that isn't over. One that has ended stays until step() has closed it, so its levers can't stay shut through the gap
        g = next((g for g in groups if ts < add_minutes(g[-1], 15) or (self.done_for == g[0] and self.reported != g[0])), groups[-1])
        by = {t: owed[t] for t in g}
        return {"start": g[0], "end": add_minutes(g[-1], 15), "owed_kw": max(by.values()), "by_ts": by}

    def market(self, step: str, ts: str, need: dict[str, Any] | None = None, fc: dict[str, Any] | None = None, offer: dict[str, Any] | None = None) -> None:
        """Steps 1 to 6 run on the market's side, as a call for offers comes in: code reads the schedule, forecasts, checks the devices and works out what is sellable, and
        brain.HomeBrain asks the model. Each enters the record when it really happens, with what went in and what came out. ctx.step stays as it is: this comes on the market's
        thread, and a sale step may be waiting on the model at this moment."""
        seen = self.market_seen.setdefault((need or {}).get("event_id"), set()) if need else set()
        for s in (*CODE_STEPS, step) if step == "5 offer" else (step,):
            if s in seen:
                continue                            # it ran when it really happened, before the offer
            seen.add(s)
            self.steps.append((s, ts))
            self.ctx.steps.enter(s, ts)
            io = self.step_record(s, need, fc, offer)
            if io:
                self.ctx.log.add("step_io", step=s, ts=ts, **io)
                self.record({"node": self.name, "tier": self.tier, "ts": ts, "why": s, "by": "code", "model": "code", "text": "", "shortlist": [], "answer": [], "said": "", "ms": 0, "owed_kw": 0.0,
                             "prompt": T.PROMPT_VERSION, "verdicts": [], **io})
            self.ctx.steps.leave(s, ts, "done")

    def step_record(self, step: str, need: dict[str, Any] | None, fc: dict[str, Any] | None, offer: dict[str, Any] | None) -> dict[str, Any] | None:
        """What went into a code step and what came out of it."""
        if need is None:
            return None
        window = {"start": need["window_start"], "end": need["window_end"]}
        if step == "1 read schedule":
            from .grounds import schedule_in
            rows = schedule_in(self.home, need)
            return {"inputs": {"day": need["window_start"][:10], **window}, "outputs": {"blocks_in_window": len(rows), "blocks": rows}}
        if step == "2 forecast" and fc is not None:
            st = fc.get("steps") or []
            mean = lambda k: round(sum(float(x.get(k) or 0.0) for x in st) / max(1, len(st)), 3)      # noqa: E731
            return {"inputs": {"method": str(fc.get("method", ""))[:60], **window}, "outputs": {"load_kw": mean("load_p50"), "solar_kw": mean("pv_p50"), "car_kw": mean("ev_kw"), "steps": len(st)}}
        if step == "3 check devices" and offer is not None:
            s0 = (offer.get("slots") or [{}])[0]
            parts = {k: s0.get(k) for k in ("battery_kw", "ev_pause_kw", "ac_kw", "app_kw", "light_kw") if s0.get(k)}
            return {"inputs": {"shelf": list(self.shelf), "sensors": offer.get("health")}, "outputs": {"by_part_kw": parts, "by_device_kw": (offer.get("appliances") or {}).get("by_device") or {}}}
        if step == "4 estimate sellable" and offer is not None:
            from ..market import firm_part
            kw = [s.get("kw_offered", 0.0) for s in offer.get("slots") or []]
            return {"inputs": {"battery": offer.get("battery"), "forecast": str(offer.get("forecast_method", ""))[:60]},        # the firm part as the market will read it: a building's is set after this step
                    "outputs": {"safe_kw_per_half_hour": kw, "firm_kw": [round(firm_part(offer, s), 3) for s in offer.get("slots") or []], "energy_kwh": offer.get("energy_kwh")}}
        return None

    def delivered(self, sale: dict[str, Any]) -> str:
        """What the meter says of the sale, for the report: the kWh missed step by step, so a good first hour can't hide a short last one."""
        rows = [t for t in self.sim.recent.values() if sale["start"] <= t["ts"] < sale["end"]]
        gave = [t["grid_kw"] - t["grid_cf_kw"] for t in rows]
        short = sum(max(0.0, t["owed_kw"] - g) for t, g in zip(rows, gave)) * .25
        n = 0 if self.word == "building" else 1
        return f"The meter shows {sum(gave) / max(1, len(gave)):,.{n}f} kW delivered on average, and {short:,.{n}f} kWh missed in the steps that fell short."

    def could_give(self, start: str, end: str) -> dict[str, Any]:
        """A call asks how much more this node could give. The levers show only what calls have opened, so for this sum they stand open, less what a wish
        or a person keeps out. If the call comes, the agent wakes and opens it, and if the model has gone the controller does."""
        sim = self.sim
        lev, comfort = sim.levers, sim.overrides.get("comfort")
        if lev is None:
            return sim.rt_capacity(start, end)
        keep = self.ctx._kept()
        try:
            write_comfort(sim)                      # the agent's hold goes out of this sum, and a person's word stays in it
            self.all_open(keep)
            return sim.rt_capacity(start, end)
        finally:
            sim.levers = lev
            write_comfort(sim, comfort)

    def all_open(self, keep: set[str]) -> None:
        self.sim.levers = {"battery_kw": 1e9, "ev": True, "ev_kw": None, "v2h_kw": 1e9, "v2h_kwh": None, "devices": set(getattr(self.sim, "app", None) or ())}
        self.reopen(keep)

    def called(self, steps: dict[str, float]) -> None:
        """The coordinator dispatched more. The next step wakes the agent."""
        kw = max(steps.values(), default=0.0)
        if kw > SHORT_KW and self.sim.levers is not None:
            self.call_kw = max(self.call_kw, kw)

    def replan(self) -> None:
        """The sale's amount changed, so this step plans it afresh: each call below takes over from what runs, and what the model leaves stays open as it was."""
        self.calls = []
        self.guard.clear()

    def owed_now(self, sale: dict[str, Any]) -> float:
        return max(0.0, sale["owed_kw"] - sum(c["expected_kw"] for c in self.calls))

    def limit_fell(self, ts: str, sale: dict[str, Any]) -> str | None:
        """A person lowered a limit while the sale runs. The controller holds it from its next part. Here what's owed comes out right, and the agent hears why. Without this it read 0 kW owed and did nothing."""
        now, seen = self.limits_now(), self.seen_limits
        self.seen_limits = now
        if seen is None or all(v >= seen.get(k, v) - 1e-9 for k, v in now.items()):
            return None
        self.shrink(self.view(ts, sale))
        return "A limit was lowered."

    def decide(self, ts: str, sale: dict[str, Any], why: str | None = None) -> None:
        if self.sim.levers is None or self.ctx.handed_back or self.sim.overrides.get("stop"):
            return                                  # the model went earlier in this sale, or a person took the node out of it
        kept = self.ctx._kept()
        used = {c["tool"] for c in self.calls} | kept
        done = [f'{c["says"]}' for c in self.calls] + [owner.SAYS[w["wish"]] for w in self.guard.wishes]
        block, reply_retry, reads = None, False, True
        for _ in range(MAX_CALLS):
            f = self.view(ts, sale)
            owed = self.owed_now(sale)
            if owed <= 0.05 and not why:
                break                               # nothing owed: code doesn't call the model at all
            if not any(can_act(f, d) and d["tool"] not in used for d in f["devices"].values()):
                self.nothing_left(ts, why, owed)
                break                               # nothing left that could give anything
            text = self.words(f, owed, done, why, block, used)
            tools = self.ctx.menu(f, used, reads)
            a = self.brain.ask(T.SYSTEM, tools, text)
            rec = {"node": self.name, "tier": self.tier, "ts": ts, "why": why, "text": text, "shortlist": [x["function"]["name"] for x in tools],
                   "answer": a["calls"], "said": a["text"], "ms": a["ms"], "model": self.brain.model, "prompt": T.PROMPT_VERSION, "owed_kw": round(owed, 2), "verdicts": []}
            block = None
            stop = False
            if a.get("error"):
                rec["fallback"] = "the model did not answer, so the controller may use every device"
                self.ctx.hand_back()
                self.record(rec)
                return
            if not a["calls"]:
                self.record(rec)
                if reply_retry:
                    return
                reply_retry = True
                block = "Call one tool."
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
                t0 = max(ts, sale["start"])
                f = self.view(ts, sale)             # the controller ran on while the model answered: the guard judges the node as it is now
                c = {**c, "args": self.amount(c["name"], c["args"], f, self.owed_now(sale))}
                call = {"tool": c["name"], "args": c["args"], "start": t0, "end": sale["end"]}
                ans = self.ctx.act(c["name"], c["args"], start=t0, end=sale["end"], facts=f)
                rec["verdicts"].append({"call": call, "answer": ans})
                self.log_calls.append({"target": ans["target"], "status": ans["status"], "expected_kw": ans["expected_kw"], "why": ans["why"]})
                if ans["status"] in ("done", "adjusted"):
                    self.calls.append({"tool": c["name"], "target": ans["target"], "expected_kw": ans["expected_kw"],
                                       "says": f'{ans["target"]} gives {ans["expected_kw"]:.2f} kW.'})
                    used.add(c["name"])
                    done.append(self.calls[-1]["says"])
                else:
                    ti = ans.get("try_instead") or {}
                    args = ", ".join(f"{k}={json.dumps(v)}" for k, v in ti.items() if k != "tool")
                    block = f'The guard answered: status blocked. Why: {ans["why"]}' + (f' try_instead: {ti.get("tool")}({args}).' if ti else "")
            self.record(rec)
            why = None
            if stop:
                return

    def nothing_left(self, ts: str, why: str | None, owed: float) -> None:
        """Woken for a reason with no device left to call. The model isn't asked, so this is the record of it: on 9 Jul the office took a dispatch of 237.5 kW and nothing showed why no call followed."""
        if why:
            self.record({"node": self.name, "tier": self.tier, "ts": ts, "why": why, "text": "", "shortlist": [], "answer": [], "said": "No device is left to call: the plan runs on what is open.", "ms": 0,
                         "model": self.brain.model, "prompt": T.PROMPT_VERSION, "owed_kw": round(owed, 2), "verdicts": []})

    def read_note(self, ts: str) -> str | None:
        """New notes: the model says what the owner needs, and code keeps it from here on, until the wish ends (lapse). A note on file goes the same way, once, the first time this node sees it."""
        ov = self.sim.overrides
        said = None
        for n in list(ov.get("note_before") or ()):     # notes a newer one pushed aside before anyone read them, oldest first
            got, read = self.take_note(n.get("note"), n.get("at") or ts, ts)
            if not read:
                self.noted = None                   # the model is away: these and the newest wait for the next step
                return said
            q = ov.get("note_before") or []
            if n in q:
                q.remove(n)
            said = said or got
        if ov.get("note") and owner.note_key(ov) != self.noted:
            self.noted, read = owner.note_key(ov), False      # first: the model's wait lets go of the node, and a market message may look for notes meanwhile
            try:
                said, read = self.take_note(ov["note"], ov.get("note_at") or ts, ts, self.noted)
                if read:
                    self.sim.note_seen = self.noted
            finally:
                if not read:
                    self.noted = None               # the model never read it, so it's read again at the next step, not lost
        for n in self.on_file():
            if quoted(n, owner.NOTE_MAX) not in self.filed:
                self.filed.add(quoted(n, owner.NOTE_MAX))
                got, read = self.take_note(n, ts, ts)
                said = said or got
                if not read:
                    self.filed.discard(quoted(n, owner.NOTE_MAX))
        return said

    def on_file(self) -> list[str]:
        """The owner's or manager's own words on file with the node's mind, the last two as an offer's facts read them."""
        wishes = getattr(getattr(self.home, "brain", None), "wishes", None)
        return list(wishes.get().get("notes") or [])[-2:] if wishes is not None else []

    def take_note(self, note: str, at: str, ts: str, key: str | None = None) -> tuple[str | None, bool]:
        """One note, through its own short step. (What changed for a plan to hear, did the model read it.) A note that sets no wish bears on nothing, so it changes nothing.
        key: the door's note, by owner.note_key, so a restart knows it was read."""
        live = [w for w in self.guard.wishes if not w.get("until") or w["until"] > ts]
        r = owner.read_note(self.brain, note, at, "manager" if self.word == "building" else "owner", live)
        self.record({"node": self.name, "tier": self.tier, "ts": ts, "why": "owner's note", "text": quoted(note, owner.NOTE_MAX), "shortlist": ["set_owner_wish", "end_owner_wish", "no_wish"] if live else ["set_owner_wish", "no_wish"],
                     "answer": r["wishes"], "said": r["said"], "ms": r["ms"], "model": self.brain.model, "prompt": T.PROMPT_VERSION, "owed_kw": 0.0, "verdicts": [],
                     **({"note_key": key} if key and not r.get("error") else {}), **({"ended": r["ended"]} if r["ended"] else {})})
        new = [w for w in r["wishes"] if not w["until"] or w["until"] > ts]
        for w in new:
            self.wish_notes[w["wish"]] = quoted(note, owner.NOTE_MAX)     # the words behind each wish, which a later offer cites
        if arm.all_agents():                        # every node an agent: the note's receipt. Rules can't read text, so the rule's answer is that nothing changes
            who = "manager" if self.word == "building" else "owner"
            cost = None if r.get("error") else self.note_cost(ts, new)      # a sale already made: what the note takes out of it, against the amount without it
            told = owner.got_it(new, cost["gives"], cost["would"], owner.when_of(cost["start"], ts), self.kw_text) if cost else None
            if told:
                self.tell(told, "note", ts)
            self.receipt("note", ts, asked={"note": quoted(note, owner.NOTE_MAX), "from": who},
                         rule={"needs": [], "what": "rules can't read a note", **({"kw_per_half_hour": cost["without"]} if cost else {})},
                         chose={"needs": [w["wish"] for w in r["wishes"]], "ended": r["ended"], **({"kw_per_half_hour": cost["with"]} if cost else {})},
                         reason=owner.says(new, who) if cost else str(r.get("said") or ""), facts=[{"id": "note", "says": f'The {who} wrote: "{quoted(note, owner.NOTE_MAX)}"'}] if cost else [],
                         flags=[owner.FOR_NOTE] if cost else [], by="agent" if not r.get("error") else "deferred: the model did not answer, so the note is read again at the next step",
                         less=bool(cost), told=told)
        gone = [w for w in self.guard.wishes if "all" in r["ended"] or w["wish"] in r["ended"]]
        if gone:
            self.end(gone, ts)                      # first: "use the battery after all, but keep the car" ends one and sets the other
        wishes = [w for w in r["wishes"] if not w["until"] or w["until"] > ts]       # a date already gone is a fact about the past
        if not wishes:
            return (owner.TAKEN_BACK if gone else None), not r.get("error")
        self.guard.wishes += wishes
        self.lapse(ts)                              # one with no end of its own takes its sale's, and read again after a restart that sale may be over
        wishes = [w for w in wishes if w in self.guard.wishes]
        if not wishes:
            return None, True
        self.short_reason = context.SHORT_WORDS["owner_need"]
        blocked = owner.blocked_tools(wishes)
        if any(w["wish"] == "keep_car_ready" for w in wishes) and not self.sim.overrides.get("ev_now"):
            self.sim.rt_override("ev_now", True)                  # the owner's car first
            self.sim.car_first = True
        self.calls = [c for c in self.calls if c["tool"] not in blocked]
        if self.sim.levers is not None:
            self.reopen(blocked)
        elif self.ctx.handed_back:                  # a second note on a building on loan
            self.ctx.hand_back()
        return owner.SAYS[wishes[0]["wish"]], True

    def kw_text(self, x: float) -> str:
        return f"{x:,.0f}" if self.word == "building" else f"{x:.1f}"

    def note_cost(self, ts: str, wishes: list[dict[str, Any]]) -> dict[str, Any] | None:
        """What a note takes out of a sale this node already committed to, half hour by half hour from now: the committed kW, which is what it gives without the note and what
        a rule would give, since a rule can't read one; and what is left with the note, by the same keep_out that sizes an offer under a wish. None: no such sale, or nothing taken."""
        from ..beckn import add_minutes
        from .grounds import keep_out
        rows: list[tuple[str, float, float]] = []
        for c in (getattr(self.home, "contracts", None) or {}).values() if wishes else ():
            offer = (getattr(self.home, "offers", None) or {}).get(c.get("event_id"))
            if c.get("status") != "ACCEPTED" or not offer or str(c.get("window_end") or "") <= ts:
                continue
            kept = {s.get("id"): s for s in keep_out(offer, wishes, c).get("slots") or []}
            had = {s.get("id"): s for s in offer.get("slots") or []}
            for s in c.get("commitments") or []:
                if add_minutes(s["start"], 30) <= ts or s.get("id") not in had or s.get("id") not in kept:
                    continue                        # a half hour already over
                y = float(s["kw"])
                out = max(0.0, float(had[s["id"]].get("kw_offered") or 0.0) - float(kept[s["id"]].get("kw_offered") or 0.0))
                rows.append((s["start"], y, max(0.0, y - out)))
        if not rows or sum(y - x for _, y, x in rows) <= 0.01:
            return None
        n = len(rows)
        return {"start": min(r[0] for r in rows), "without": [round(y, 2) for _, y, _ in rows], "with": [round(x, 2) for _, _, x in rows],
                "would": sum(y for _, y, _ in rows) / n, "gives": sum(x for _, _, x in rows) / n}

    def lapse(self, ts: str) -> list[dict[str, Any]]:
        """Wishes end here and nowhere else. One with its own end keeps it, whatever sales pass. One with none holds through the next sale, and failing a sale by the day after it was read
        it ends then. "Going away tonight" held on into the next morning and every later commitment, and on a loan "until Friday" ended with the first sale."""
        if not self.guard.wishes:
            return []
        kept, ended = [], []
        for w in self.guard.wishes:
            since = w["from"] = w.get("from") or ts                # a wish from the old record has no date
            if not w.get("until"):
                w["until"] = self.sale_end_after(since) or (ts if ts[:10] > owner.next_day(since) else "")
            (ended if w.get("until") and ts >= w["until"] else kept).append(w)
        if ended:
            self.end(ended, ts)
        return ended

    def end(self, ended: list[dict[str, Any]], ts: str) -> None:
        """These wishes end now, by their date or because a note took them back. What they kept out is the sale's again."""
        self.guard.wishes = kept = [w for w in self.guard.wishes if w not in ended]
        for w in ended:
            self.ctx.log.add("wish_ended", ts=ts, wish=w["wish"], since=w.get("from") or "")
        if self.car_first and not any(w["wish"] == "keep_car_ready" for w in kept):
            self.sim.rt_override("ev_now", False)             # and car_first with it
        if self.ctx.handed_back:
            self.ctx.hand_back()                # what the wish kept out is the controller's again

    def sale_end_after(self, since: str) -> str | None:
        """The end of the first sale still running or to come at this moment, from the contract. Read again after a restart, an old note finds the sale it was for, which has gone."""
        from ..beckn import add_minutes
        owed = sorted(t for t, a in self.sim.plan.items() if t >= since and float(a.get("deliver_kw") or 0.0) > 0)
        if not owed:
            return None
        end = owed[0]
        for t in owed[1:]:
            if t != add_minutes(end, 15):
                break
            end = t
        return add_minutes(end, 15)

    def over(self, ts: str) -> bool:
        """A loan lasts as long as its wishes."""
        self.lapse(ts)
        return not self.guard.wishes

    def finish(self) -> None:
        """The node is code's again: every write ends by itself."""
        self.close()
        self.home.agent = None

    def reopen(self, blocked: set[str]) -> None:
        """Take back what the owner's wish stops: the device leaves the sale at once."""
        lev = self.sim.levers
        if "set_battery" in blocked:
            lev["battery_kw"] = 0.0
        if "limit_ev_charging" in blocked:
            lev["ev"], lev["ev_kw"] = False, None
        if "discharge_car" in blocked:
            lev["v2h_kw"] = 0.0
            if not any(c["tool"] == "limit_ev_charging" for c in self.calls):
                lev["ev"], lev["ev_kw"] = False, None      # the charger it held goes back, unless a pause of its own still runs
        if blocked & {"set_thermostat", "set_room_ac"}:
            write_comfort(self.sim, held_comfort(self.sim))
        held = lev.get("devices", set())            # a building's levers have none of these
        if "set_water_heater" in blocked:
            held.discard("hpwh")
        if "set_pool_pump" in blocked:
            held.discard("pool")
        if "delay_appliance" in blocked:
            held -= set(MACHINE_KEY.values())

    def step(self, ts: str) -> None:
        """Called every 15 minutes. Most steps do nothing: the controller is running the node. A wish ends after the step, so the sale's report still has it."""
        try:
            self.turn(ts)
        finally:
            self.lapse(ts)

    def turn(self, ts: str) -> None:
        from ..beckn import add_minutes
        loans.serve(self, ts)
        alarm = self.sim.overrides.get("alarm") or self.sim.overrides.get("alarm_hold") or self.ctx.stopped
        if alarm:
            self.ctx.alarm(alarm, ts)
            self.calls = []
            return
        note_why = self.read_note(ts)
        sale = self.sale(ts)
        if not sale:
            return
        if ts >= sale["end"]:
            if self.sim.levers is not None:
                self.close()
                self.guard.clear()
                self.calls = []
                self.short_reason = None
            if self.done_for == sale["start"] and self.reported != sale["start"]:
                self.reported = sale["start"]
                self.steps.append(("9 report", ts))
                self.ctx.enter("9 report", ts, sale)
                self.reports.append(report.send(self, report.write(self, sale, ts), sale))
                self.ctx.leave()
            return
        lead = add_minutes(sale["start"], -LEAD_MIN)
        if self.done_for != sale["start"] and ts >= lead:
            self.done_for = sale["start"]
            self.last_wake = ts
            self.steps.append(("7 watch", ts))
            self.ctx.enter("7 watch", ts, sale)
            self.log_calls = []
            self.open()
            self.seen_limits = self.limits_now()
            self.decide(ts, sale)
            self.ctx.leave()          # a step ends with its work: no tools stay open until the next one
            return
        if self.done_for != sale["start"]:
            return
        why = self.limit_fell(ts, sale) or note_why
        plugged = bool(self.sim.has_ev and self.sim.ev_deadline and self.sim.ev_left > 0.01)
        if plugged and not self.plugged and "limit_ev_charging" not in {c["tool"] for c in self.calls} and why is None:
            why = "the car plugged in."
        self.plugged = plugged
        waiting = "delay_appliance" in self.shelf and ts >= sale["start"] and next((d["name"] for d in self.view(ts, sale)["devices"].values() if d["tool"] == "delay_appliance"), None)
        if waiting and waiting != self.waiting and "delay_appliance" not in {c["tool"] for c in self.calls} and why is None:
            why = f"The {waiting.lower()} is waiting to start."
        self.waiting = waiting
        trip = bool(self.sim.E > 0 and self.sim.P <= 0)
        if trip and not self.tripped:
            self.tripped = True
            self.calls = [c for c in self.calls if c["tool"] != "set_battery"]
            why = "The battery shut off."
        if self.call_kw and why is None:
            why = f"The grid called for {self.call_kw:.1f} kW more."
            self.call_kw = 0.0
            self.replan()
        if why is None and ts >= sale["start"]:
            last = self.sim.ctl.last or {}
            short = max((v for t, v in (last.get("short_by_ts") or {}).items() if t >= ts), default=0.0)
            quiet = self.last_wake and ts < add_minutes(self.last_wake, QUIET_MIN)
            if short > max(SHORT_KW, SHORT_SHARE * sale["owed_kw"]) and not quiet:
                why = f"The plan falls {short:.1f} kW short."
                sale = {**sale, "owed_kw": short + sum(c["expected_kw"] for c in self.calls)}
        if why:
            self.last_wake = ts
            self.steps.append(("8 exception", ts))
            self.ctx.enter("8 exception", ts, sale)
            self.decide(ts, sale, why)
            self.ctx.leave()

    # ------------------------------------------------------------------ one device call, whoever chose it
    def key(self, d: dict[str, Any]) -> str:
        return f'{d["tool"]}:{d["target"]}'

    def can(self, f: dict[str, Any], d: dict[str, Any]) -> bool:
        return can_act(f, d) and (d["tool"] != "discharge_car" or d.get("status") == "idle")     # 10 of the 13 guard blocks were a car that couldn't give

    def kept_now(self) -> set[str]:
        return self.ctx._kept()

    def used_now(self) -> set[str]:
        return {f'{c["tool"]}:{c["target"]}' for c in self.calls} | self.let_gone      # what a plan let go stays closed for this sale: the next plan can't flip it back

    def may_decide(self) -> bool:
        return self.sim.levers is not None and not self.ctx.handed_back and not self.sim.overrides.get("stop")     # else the model went, or a person took the node out of this sale

    def live(self, f: dict[str, Any], used: set[str], kept: set[str]) -> list[dict[str, Any]]:
        """What can still be opened, largest first."""
        return sorted((d for d in f["devices"].values() if self.key(d) not in used and d["tool"] not in kept and self.can(f, d)), key=lambda d: -d.get("can_give_kw", 0.0))

    def still_open(self, f: dict[str, Any]) -> float:
        return sum(float(d.get("can_give_kw", 0.0)) for d in self.live(f, self.used_now(), self.kept_now()))

    def ready(self, d: dict[str, Any], f: dict[str, Any], owed: float) -> dict[str, Any]:
        """The call with the numbers the list gives. amount() sizes it after this."""
        tool, t = d["tool"], d["target"]
        if tool == "set_battery":
            return {"target": t, "action": "discharge", "kw": d["can_give_kw"]}
        if tool == "limit_ev_charging":
            return {"target": t, "amps": 0}
        if tool == "discharge_car":
            return {"target": t, "kw": d["can_give_kw"], "max_kwh": d["max_kwh"]}
        if tool in ("set_thermostat", "set_room_ac"):
            return {"target": t, "less_comfort_c": d["limit_c"]}
        if tool == "set_water_heater":
            return {"target": t, "action": "hold"}
        if tool == "delay_appliance":
            return {"target": t, "finish_by": d["done_by"]}
        if tool == "set_pool_pump":
            return {"target": t, "speed_percent": d["floor"]}
        return {"target": t}

    def times(self, tool: str, a: dict[str, Any], d: dict[str, Any] | None, sale: dict[str, Any], ts: str) -> tuple[str, str]:
        return max(ts, sale["start"]), sale["end"]

    def took(self, tool: str, ans: dict[str, Any], d: dict[str, Any] | None) -> None:
        self.calls.append({"tool": tool, "target": ans["target"], "expected_kw": ans["expected_kw"], "says": f'{ans["target"]} gives {ans["expected_kw"]:.2f} kW.'})

    def refused(self, ans: dict[str, Any], ts: str) -> None:
        """A building keeps the times of a guard's try_instead here."""

    def one_call(self, tool: str, args: dict[str, Any], ts: str, sale: dict[str, Any], rec: dict[str, Any], who: P.Who = P.CODE) -> dict[str, Any]:
        """One device call: code sizes it, and it goes through the step's menu, policy, guard, seal and driver, the way a model's call goes. who: whose choice it was, for the record."""
        f = self.view(ts, sale)                     # the controller ran on: the guard judges the node as it is now
        d = next((x for x in f["devices"].values() if x["tool"] == tool and x["target"] == args.get("target")), None)
        args = self.amount(tool, args, f, self.owed_now(sale))
        t0, t1 = self.times(tool, args, d, sale, ts)
        if hasattr(self.ctx, "show"):               # the limits fix holds a call to the turn's list: code's own turn lists this one call
            self.ctx.show([T.spec(self.ctx.side, tool, [str(args.get("target"))])])
        ans = self.ctx.act(tool, args, start=t0, end=t1, facts=f, who=who)
        rec["verdicts"].append({"call": {"tool": tool, "args": args, "start": t0, "end": t1}, "answer": ans})
        self.log_calls.append({"target": ans["target"], "status": ans["status"], "expected_kw": ans["expected_kw"], "why": ans["why"]})
        if ans["status"] in context.SENT:
            self.took(tool, ans, d)
        else:
            self.refused(ans, ts)
        return ans

    def blank(self, ts: str, why: str | None, owed: float, by: str = "code", text: str = "") -> dict[str, Any]:
        return {"node": self.name, "tier": self.tier, "ts": ts, "why": why, "text": text, "shortlist": [], "answer": [], "said": "", "ms": 0, "model": "code" if by == "code" else self.brain.model,
                "prompt": T.PROMPT_VERSION, "owed_kw": round(owed, 2), "verdicts": [], "by": by}

    # ------------------------------------------------------------------ step 7: the list, by code
    def next_device(self, f: dict[str, Any], live: list[dict[str, Any]], used: set[str]) -> dict[str, Any] | None:
        """The top of the list. On the real model a three-line rule matched 91% of home picks, and the rest changed nothing a meter could see."""
        return live[0] if live else None

    def code_call(self, d: dict[str, Any], f: dict[str, Any], owed: float, ts: str, sale: dict[str, Any], why: str | None, note: str) -> dict[str, Any]:
        """One call code chose, with its own line in the decision record. A guard's try_instead gets one go."""
        args = self.ready(d, f, owed)
        rec = self.blank(ts, why, owed, text=note)
        rec["answer"] = [{"name": d["tool"], "args": args}]
        ans = self.one_call(d["tool"], args, ts, sale, rec)
        ti = ans.get("try_instead") or {}
        if ans["status"] not in context.SENT and ti.get("tool") and ti["tool"] in self.shelf:
            more = {k: v for k, v in ti.items() if k not in ("tool", "start", "end")}
            rec["answer"].append({"name": ti["tool"], "args": more})
            ans = self.one_call(ti["tool"], more, ts, sale, rec)
        self.record(rec)
        return ans

    def walk(self, ts: str, sale: dict[str, Any], why: str | None = None) -> None:
        """Opening a sale has no real choice in it, so code walks the ready list until what is owed is covered: 69% of model answers changed nothing, and 8 answers a building
        took 2 minutes of a queue that exceptions need. Every call still goes through the guard and has its line in the record. The model is asked when something changes."""
        if not self.may_decide():
            return
        used, kept = self.used_now(), self.kept_now()
        for _ in range(MAX_WALK):
            f = self.view(ts, sale)
            owed = self.owed_now(sale)
            if owed <= 0.05:
                return
            d = self.next_device(f, self.live(f, used, kept), used)
            if d is None:
                return
            self.code_call(d, f, owed, ts, sale, why, f"Code took the next device on the list, with {owed:.1f} kW owed.")
            used.add(self.key(d))

    def by_step(self, ts: str, sale: dict[str, Any], why: str | None = None) -> None:
        """Step 7 is code's and step 8 the model's, as MULTI_AGENT_DESIGN has it. decide() is the step as it was, one model answer per device: FG_WALK=model."""
        if why:
            return self.plan(ts, sale, why)
        self.let_gone, self.spent = set(), set()    # a new sale
        if arm.all_agents():                        # every node an agent: the opening is the agent's plan, with code's walk shown as the rule's plan
            return self.plan(ts, sale, OPENS)
        return self.walk(ts, sale)

    # ------------------------------------------------------------------ step 8: the whole plan, in one answer
    def price_now(self, sale: dict[str, Any]) -> float:
        """$ per kWh in the contract behind this sale."""
        for c in self.home.contracts.values():
            if c.get("status") == "ACCEPTED" and c.get("window_start") == sale["start"]:
                return float(c.get("price_usd_per_kwh") or 0.0)
        return 0.0

    def cost_of(self, d: dict[str, Any], f: dict[str, Any]) -> str:
        """What opening this costs, in words the model can weigh."""
        tool = d["tool"]
        if tool == "set_battery":
            return "the battery ends lower and refills after the sale"
        if tool == "limit_ev_charging":
            return f'the car waits, and must be ready by {d.get("leaves")}'
        if tool == "discharge_car":
            return f'the car gives up to {d.get("max_kwh", 0.0):.1f} kWh and refills before {d.get("leaves")}'
        if tool in ("set_thermostat", "set_room_ac"):
            return f'the house may warm by up to {d.get("limit_c", 0.0):.2f} C'
        if tool == "set_water_heater":
            return "the tank cools a little"
        if tool == "delay_appliance":
            return f'it starts after the sale and ends by {d.get("done_by")}'
        if tool == "set_pool_pump":
            return "the filter runs slower"
        return "none known"

    def weighs(self, d: dict[str, Any]) -> bool:
        """People feel this one: choosing it over nothing is a real trade-off."""
        return d["tool"] in ("set_thermostat", "set_room_ac", "limit_ev_charging", "discharge_car", "delay_appliance")

    def option_line(self, d: dict[str, Any], f: dict[str, Any]) -> str:
        return f'{d["target"]} gives {d.get("can_give_kw", 0.0):.2f} kW.'

    def touched(self, c: dict[str, Any], f: dict[str, Any]) -> bool:
        """This running call now presses on something that changed, so letting it go is a live question. A building's zones say so here."""
        return False

    def facts_at_hand(self, f: dict[str, Any], why: str) -> list[dict[str, Any]]:
        """What a plan may cite to leave kW uncovered or let a call go, each with what it bears on: a wish its own part, a limit a person lowered its own device, a limit close by its own device.
        Each is read here, not assumed, and each is true now. A note that set no wish isn't here: the model read it, and it asks for nothing of the devices. A tripped battery isn't here either:
        it explains a shortfall, and it is no reason to warm the house less."""
        sim = self.sim
        out = [fact(owner.SAYS[w["wish"]], f'wish:{w["wish"]}', (owner.PART[w["wish"]],)) for w in self.guard.wishes]
        out += [fact(str(v), f"lowered:{t}", (owner.PART_OF.get(t, ""),)) for t, v in (f.get("lowered") or {}).items()]       # a limit a person lowered during the sale
        for d in f["devices"].values():
            if d["tool"] in ("set_thermostat", "set_room_ac") and d.get("limit_c") and float(sim.x) >= 0.8 * float(d["limit_c"]):
                out.append(fact(f'The house already runs {float(sim.x):.2f} C above normal, and its limit is {d["limit_c"]:.2f} C.', "warm", ("cooling",)))
            if d["tool"] == "limit_ev_charging" and d.get("leaves") and float(d.get("wait_h", 99.0)) < 2.0:
                out.append(fact(f'The car leaves at {d["leaves"]} with {max(0.0, float(d["wait_h"])):.1f} hours to spare.', "car", ("car",)))
        return out

    def situation(self, ts: str, sale: dict[str, Any], why: str, used: set[str], kept: set[str]) -> dict[str, Any]:
        """Everything a plan needs, with code's own plan beside it: what is owed, what runs, what can open, what each is worth, what may be cited."""
        f = self.view(ts, sale)
        owed = self.owed_now(sale)
        hours = max(0.25, self.sim.replay.minutes_between(max(ts, sale["start"]), sale["end"]) / 60.0)
        price = self.price_now(sale)
        opts: list[dict[str, Any]] = []
        refused: list[str] = []
        for d in self.live(f, used, kept):
            why_not = self.would_refuse(d, f, owed, ts, sale)
            if why_not:
                refused.append(f'{d["target"]}: {why_not}')
            elif len(opts) < MAX_OPTIONS:
                opts.append({"id": chr(65 + len(opts)), "d": d, "kw": float(d.get("can_give_kw", 0.0)) + float(d.get("behind_kw", 0.0)), "line": self.option_line(d, f), "cost": self.cost_of(d, f), "costs": self.weighs(d)})
        for o in opts:                              # what a device waits for, on its line: the guard blocks it alone, and code sets that first when a plan opens it (first_needs)
            pre = self.prerequisite(o["d"], f)      # on 29 Jun the office's air handlers read as if they could simply open, a test said blocked, and 76 kW stayed uncovered
            if pre is None:
                continue
            by = next((p for p in opts if self.key(p["d"]) == self.key(pre)), None)
            if by is not None:
                by.setdefault("unlocks", []).append(o["id"])
            first = by["id"] if by else pre["target"]
            o["line"] = f'{o["line"]} It needs {first} first, which eases its zones: if the plan opens {o["id"]}, code sets {first} first.'
        run = sorted((c for c in self.calls if c["expected_kw"] > 0 or self.touched(c, f)), key=lambda c: (not self.touched(c, f), -c["expected_kw"]))[:MAX_OPTIONS]
        on = {self.key(d): d for d in f["devices"].values()}
        run = [{"id": f"R{i + 1}", "c": c, "kw": float(c["expected_kw"]), "touched": self.touched(c, f), "cost": self.cost_of(on[self.key(c)], f) if self.key(c) in on else "",
                "costs": self.weighs(c)} for i, c in enumerate(run)]
        tol = max(0.05, 0.02 * float(sale["owed_kw"]))
        left, plan = owed, []
        for o in opts:
            if left <= tol:
                break
            plan.append(o["id"])
            left -= o["kw"]
        touches = [(o["d"]["tool"], o["d"].get("spaces"), True) for o in opts] + [(r["c"]["tool"], r["c"].get("spaces")) for r in run]
        facts = [{"id": f"F{i + 1}", **x} for i, x in enumerate(y for y in self.facts_at_hand(f, why) if y["key"] not in self.spent and any(bears_on(y, *t) for t in touches))]
        return {"f": f, "owed": owed, "covered": sum(c["expected_kw"] for c in self.calls), "hours": hours, "price": price, "options": opts, "running": run, "tol": tol, "default": plan,
                "left": max(0.0, left), "facts": facts, "why": why, "refused": refused[:4]}

    def would_refuse(self, d: dict[str, Any], f: dict[str, Any], owed: float, ts: str, sale: dict[str, Any]) -> str:
        """The guard's word on this device, asked of a copy. On the real model a blocked call came straight back: now it isn't on the list, and the plan is told why."""
        args = self.amount(d["tool"], self.ready(d, f, owed), f, max(owed, float(d.get("can_give_kw", 0.0))))
        t0, t1 = self.times(d["tool"], args, d, sale, ts)
        a = self.ctx.dry(d["tool"], args, t0, t1, f)
        return "" if a["status"] in context.SENT else str(a["why"])

    def worth_asking(self, st: dict[str, Any]) -> bool:
        """The model is asked only where its answer can change something. A running call that presses on what changed is a question. kW owed is one if a fact could carry leaving some
        uncovered, or if more than one set of devices covers it and people feel the difference. With no fact and nothing spare, code's plan is the only one the rules allow.
        The same picture again 30 minutes on isn't a new question."""
        chosen = [o for o in st["options"] if o["id"] in st["default"]]
        spare = sum(o["kw"] for o in st["options"]) - st["owed"] > st["tol"]
        owed = st["owed"] > st["tol"]
        real = (any(r["touched"] for r in st["running"]) or (owed and bool(st["options"]) and (bool(st["facts"]) or (spare and any(o["costs"] for o in chosen))))
                or (owed and bool(st["facts"]) and any(r["costs"] for r in st["running"])))
        seen = (tuple(sorted(self.key(o["d"]) for o in st["options"])), tuple(sorted(self.key(r["c"]) for r in st["running"])), tuple(x["key"] for x in st["facts"]), round(math.log1p(st["owed"]) * 2))
        if not real or seen == self.asked:
            return False
        self.asked = seen
        return True

    def brief(self, st: dict[str, Any], notes: list[str], block: str | None) -> str:
        """The whole situation, in one prompt. The model picks by letter, so a number it can't get wrong is a number it never sends."""
        f, sale = st["f"], st["f"]["sale"]
        where = f' {f["place"]}' if f.get("place") else ""
        fmt = (lambda x: f"{x:,.0f}") if self.word == "building" else (lambda x: f"{x:.2f}")
        out = [f'{f["day"]}, {f["now"]}{where}. Sale {sale["start"][11:16]} to {sale["end"][11:16]}, {st["hours"]:.1f} hours left, order price ${st["price"]:.2f} per kWh delivered, of which a node gets the share of its class. '
               "Only what the meter shows is paid, and there is no penalty on top. Code holds every limit whatever you choose.",
               f'What changed: {st["why"]}',
               f'Owed {fmt(float(sale["owed_kw"]))} kW. Calls running cover {fmt(st["covered"])} kW. Not covered: {fmt(st["owed"])} kW.']
        if st["facts"]:
            out.append("Facts you may cite: " + " ".join(f'{x["id"]}: {x["says"]}' for x in st["facts"]))
        if st["running"]:
            out.append(f'Running now, each already counted in the {fmt(st["covered"])} kW. Ending one takes its kW out of that, so more stays uncovered and its money is lost. It never covers anything. '
                       + " ".join(f'{r["id"]}: {r["c"]["target"]} {DOING.get(r["c"]["tool"], "is held")}, which gives {fmt(r["kw"])} kW{". Cost: " + r["cost"] if r["cost"] else ""}. If ended, it goes back to normal, {fmt(st["owed"] + r["kw"])} kW stay uncovered and up to ${r["kw"] * st["hours"] * st["price"]:,.2f} is lost'
                                  f'{" (it touches the spaces that changed)" if r["touched"] else ""}.' for r in st["running"]))
        out.append("Can open: " + (" ".join(f'{o["id"]}: {o["line"]} Cost: {o["cost"]}. {self.worth(o, st)}' for o in st["options"]) or "nothing."))
        if st["refused"]:
            out.append("Not on the list, because the guard would refuse: " + " ".join(st["refused"]))
        out.append(f'If you change nothing, code opens {", ".join(st["default"]) or "nothing"} and {fmt(st["left"])} kW stay uncovered{", and the desk hears of it" if st["left"] > st["tol"] else ""}.')
        out += notes
        if block:
            out.append(block)
        return " ".join(out)

    def worth(self, o: dict[str, Any], st: dict[str, Any]) -> str:
        """What opening this earns: only kW not yet covered. "Worth $503.35 tonight" went with 0 kW owed, and the real model took the bait. "Nothing is uncovered" only when
        nothing is: under the plan's 2% margin the mall's car ports and plugs read so with 391 kW uncovered, and the office's zone standby with 76."""
        if st["owed"] <= COVERED_KW:
            return "Nothing is uncovered, so opening it earns nothing."
        if o["kw"] <= 0.0:
            need = o.get("unlocks") or []
            return "It gives no kW by itself" + (f', but {" and ".join(need)} need{"s" if len(need) == 1 else ""} it first.' if need else ".")
        kw = min(o["kw"], st["owed"])
        return f'Worth up to ${kw * st["hours"] * st["price"]:,.2f} tonight.'

    def plan(self, ts: str, sale: dict[str, Any], why: str) -> None:
        """Something changed in a sale. One answer gives the whole plan: what opens, what is let go, what is told to the desk. Code checks it and carries it out.
        With no real choice, or no model, code's own plan stands, and the record says which it was."""
        if not self.may_decide():
            return
        used, kept = self.used_now(), self.kept_now()
        notes: list[str] = []
        block: str | None = None
        retried = reasked = False
        for n in range(MAX_PLAN_ASKS):
            st = self.situation(ts, sale, why, used, kept)
            if n == 0 and not arm.all_agents() and not self.worth_asking(st):      # every node an agent: always asked
                return self.carry_out(st, None, ts, sale, "nothing here to weigh")
            text = self.brief(st, notes, block)
            tools = self.ctx.plan_tools(st, self.plan_reads(st))
            a = self.brain.ask(T.PLAN_SYSTEM, tools, text)
            rec = {**self.blank(ts, why, st["owed"], "model", text), "shortlist": [x["function"]["name"] for x in tools], "answer": a["calls"], "said": a["text"], "ms": a["ms"],
                   "default": {"open": st["default"], "left_kw": round(st["left"], 2)}}
            block = None
            if a.get("error"):
                if arm.all_agents():                # every node an agent: code's plan stands for this point alone, and the next point asks again
                    return self.fall_back(st, ts, sale, f'the model did not answer: {str(a["error"])[:120]}', rec)
                rec["fallback"] = "the model did not answer, so the controller may use every device"
                self.ctx.hand_back()
                self.record(rec)
                return
            shown = {x["function"]["name"] for x in tools}
            c = next((x for x in a["calls"] if x["name"] in shown), None)      # release_call isn't shown: the model let a call go by name, with no fact behind it, and 13,626 kWh delivered became 8,231
            if c is None:
                self.record(rec)
                if retried:
                    return self.fall_back(st, ts, sale, "no plan came back") if arm.all_agents() else self.carry_out(st, None, ts, sale, "no plan came back")
                retried, block = True, "Call set_plan once, with the whole plan."
                continue
            if c["name"] in ("set_plan", "no_action"):
                said = c["args"] if isinstance(c.get("args"), dict) else {}      # a model can send text here: on 2d8adc8 that crashed the plan
                args = c["args"] if c["name"] == "set_plan" else {"open": [], "end_call": [], "report": "none", "because": "none", "reason": str(said.get("reason") or "")[:120]}
                if arm.all_agents() and not reasked and n + 1 < MAX_PLAN_ASKS and self.less_without_reason(st, args):
                    reasked, block = True, REASON_AGAIN         # every node an agent: less than code's plan needs a reason, so it is asked once more
                    self.record({**rec, "asked_again": NO_REASON})
                    continue
                return self.carry_out(st, args, ts, sale, None, rec)
            got = self.ctx.call(c["name"], self.plan_call(c, st))
            self.record(rec)
            if c["name"] == "report_shortfall" and got.get("status") == "done":
                return                              # nothing could open, so this was the plan
            notes.append(("" if got.get("simulated") else "Reading, nothing changed: ") + str(got.get("view") or got.get("why") or ""))
        st = self.situation(ts, sale, why, used, kept)
        if arm.all_agents():
            return self.fall_back(st, ts, sale, "no plan within the step's answers")
        self.carry_out(st, None, ts, sale, "no plan within the step's answers")

    def fall_back(self, st: dict[str, Any], ts: str, sale: dict[str, Any], why: str, rec: dict[str, Any] | None = None) -> None:
        """Every node an agent: the model failed at this point, so code's own plan stands for it, on the record as a fallback and counted. Nothing is handed back:
        the next point asks the model again. rec: the model's record of this try, which becomes the fallback's."""
        self.fallbacks[why.split(":")[0]] += 1
        base = rec or {**self.blank(ts, st["why"], st["owed"], "code", f"Code's plan: {why}."), "default": {"open": st["default"], "left_kw": round(st["left"], 2)}}
        self.carry_out(st, None, ts, sale, why, {**base, "by": "code", "model": "code", "fallback": why})

    def plan_reads(self, st: dict[str, Any]) -> list[str]:
        """The reading tools that matter to this exception. Each answers once."""
        why = st["why"]
        out = ["simulate_plan"] if st["options"] else []
        if any(o["costs"] for o in st["options"]) or self.word == "building":
            out.append("get_free_windows")
        if "short" in why:
            out.append("forecast_site")
        if "grid called" in why:
            out.append("estimate_sellable")
        if st["refused"]:
            out.append("lookup_device_card")
        return out

    def plan_call(self, c: dict[str, Any], st: dict[str, Any]) -> dict[str, Any]:
        """The arguments a reading tool gets: a test of letters becomes the calls they stand for."""
        a = dict(c.get("args") or {}) if isinstance(c.get("args"), dict) else {}
        if c["name"] == "simulate_plan" and "open" in a:
            ids = a.pop("open")
            ids = ids if isinstance(ids, list) else [ids]
            a = {"calls": [{"tool": o["d"]["tool"], **self.ready(o["d"], st["f"], st["owed"])} for o in st["options"] if o["id"] in ids][:context.MAX_SIMULATE] or [{"tool": "none"}]}
        return a

    def carry_out(self, st: dict[str, Any], args: dict[str, Any] | None, ts: str, sale: dict[str, Any], no_ask: str | None, rec: dict[str, Any] | None = None) -> None:
        """What was decided, in the order that holds: let go first, then open, then tell the desk. kW left uncovered needs a fact from the prompt, or code opens the rest of its own plan.
        args None: code's own plan."""
        why = st["why"]
        rec = rec or {**self.blank(ts, why, st["owed"], "code", f"Code's plan: {no_ask}."), "default": {"open": st["default"], "left_kw": round(st["left"], 2)}}
        a = args if isinstance(args, dict) else {}
        by_id = {o["id"]: o for o in st["options"]}
        run_id = {r["id"]: r for r in st["running"]}
        pick = [i for i in letters(a.get("open")) if i in by_id] if args is not None else list(st["default"])
        chose = set(pick) if args is not None else set()        # the model's letters: what code adds or opens by itself is code's choice, whatever the plan
        drop = [i for i in letters(a.get("end_call")) if i in run_id][:1] if args is not None else []     # one per plan: on the real model a confused answer let all seven of the mall's calls go
        cited = next((x for x in st["facts"] if x["id"] == a.get("because")), None)
        grounded = cited is not None
        left = st["owed"] + sum(run_id[i]["kw"] for i in drop) - sum(by_id[i]["kw"] for i in pick)
        check: dict[str, Any] = {"cites": a.get("because"), "grounded": grounded, "facts": [x["says"] for x in st["facts"]], "numbers_not_in_prompt": stray_numbers(str(a.get("reason") or ""), {"prompt": rec["text"]}, {})}
        if a.get("because") not in (None, "none") and not grounded:
            check["cites_a_fact_not_there"] = str(a.get("because"))[:20]
        flags: list[str] = []
        said = str(a.get("reason") or "").strip()
        every = arm.all_agents()
        if args is not None and every and (drop or left > st["left"] + st["tol"]):        # every node an agent: less than code's plan, so it needs a reason
            over = [i for i in drop if not (grounded and bears_on(cited, run_id[i]["c"]["tool"], run_id[i]["c"].get("spaces")))]
            add = [o["id"] for o in st["options"] if o["id"] not in pick and not (grounded and bears_on(cited, o["d"]["tool"], o["d"].get("spaces"), True))]
            if not said:                            # asked once more already: still no reason, so code's own plan goes out
                flags.append(NO_REASON)
                check["no_reason"] = {"asked": {"open": pick, "release": drop}, "why": "less than code's plan with no reason, so code's plan stands"}
                pick, drop, chose = list(st["default"]), [], set()
            elif over or add:                       # it stands, flagged: the reason rests on no fact that bears on what it leaves out
                flags.append(NO_FACT)
                check["no_fact"] = {"kept_running_would_be": over, "opened_would_be": add}
            left = st["owed"] + sum(run_id[i]["kw"] for i in drop) - sum(by_id[i]["kw"] for i in pick)
        elif args is not None and not every and left > st["tol"]:
            over = [i for i in drop if not (grounded and bears_on(cited, run_id[i]["c"]["tool"], run_id[i]["c"].get("spaces")))]      # letting go with no fact behind it doesn't stand
            add = [o["id"] for o in st["options"] if o["id"] not in pick and not (grounded and bears_on(cited, o["d"]["tool"], o["d"].get("spaces"), True))]
            if over or add:
                check["overruled"] = {"kept_running": over, "opened": add, "why": "kW left uncovered with no fact from the prompt that bears on it"}
            drop = [i for i in drop if i not in over]
            pick = pick + add
            left = st["owed"] + sum(run_id[i]["kw"] for i in drop) - sum(by_id[i]["kw"] for i in pick)
        if grounded and (drop or left > st["tol"]):
            self.spent.add(cited["key"])            # a fact carries one decision, whatever its figures read by then: on the real model the same note let go one call, then three more half an hour later
        rec["plan"] = {"open": pick, "release": drop, "report": a.get("report") or "none", "reason": str(a.get("reason") or "")[:160], "check": check}
        for i in drop:
            r = run_id[i]
            out = self.ctx.call("release_call", {"target": r["c"]["target"], "reason": str(a.get("reason") or why)[:120]})
            rec["verdicts"].append({"call": {"tool": "release_call", "args": {"target": r["c"]["target"]}}, "answer": out})
            if out.get("status") == "done":
                self.let_gone.add(self.key(r["c"]))
                self.log_calls.append({"target": r["c"]["target"], "status": "released", "expected_kw": 0.0, "why": str(a.get("reason") or why)[:120]})
        done: set[str] = self.used_now()
        opened: list[str] = []
        for i in pick:
            if self.owed_now(sale) <= 0.05:
                break                               # covered: the rest would only be sized down to nothing
            opened.append(i)
            f = self.view(ts, sale)
            d = next((x for x in f["devices"].values() if self.key(x) == self.key(by_id[i]["d"])), by_id[i]["d"])
            if self.key(d) in done:
                continue                            # set already, as what an earlier letter waited for: a second write would only spend the device's writes
            first = self.first_needs(d, f, done)
            if first is not None:
                self.one_call(first["tool"], {k: v for k, v in first.items() if k != "tool"}, ts, sale, rec)
            self.one_call(d["tool"], self.ready(d, f, self.owed_now(sale)), ts, sale, rec, P.AGENT if i in chose else P.CODE)
            done.add(self.key(d))
        rec["plan"]["opened"] = opened
        rec["changed"] = args is not None and (sorted(opened) != sorted(st["default"]) or bool(drop))      # what really went, against code's own plan: a letter that sized down to nothing changed nothing
        self.walk_unlocked(ts, sale, done, why)
        word = a.get("report")
        if isinstance(word, str) and word in context.SHORT_WORDS and self.owed_now(sale) > st["tol"]:
            rec["reported"] = self.ctx.call("report_shortfall", {"reason": word})
        self.ctx.log.add("tool", step=self.ctx.step, ts=ts, tool="set_plan", args=rec["plan"], who="agent" if args is not None else "code", result={"status": "done", "changed": rec["changed"]})
        if every:
            rec["receipt"] = self.plan_receipt(st, ts, sale, rec, args, pick, drop, run_id, cited, said, flags, no_ask)
        self.record(rec)

    def less_without_reason(self, st: dict[str, Any], args: Any) -> bool:
        """This plan gives less than code's (a call let go, or more kW left uncovered) and says nothing of why."""
        a = args if isinstance(args, dict) else {}
        by_id, run_id = {o["id"]: o for o in st["options"]}, {r["id"]: r for r in st["running"]}
        pick = [i for i in letters(a.get("open")) if i in by_id]
        drop = [i for i in letters(a.get("end_call")) if i in run_id][:1]
        left = st["owed"] + sum(run_id[i]["kw"] for i in drop) - sum(by_id[i]["kw"] for i in pick)
        return bool(drop or left > st["left"] + st["tol"]) and not str(a.get("reason") or "").strip()

    def plan_receipt(self, st, ts, sale, rec, args, pick, drop, run_id, cited, said, flags, no_ask) -> dict[str, Any]:
        """The receipt of a plan: what was asked, code's plan, what the agent chose, its reason and its fact. The owner or manager hears why when it gave less."""
        fmt = (lambda x: f"{x:,.0f}") if self.word == "building" else (lambda x: f"{x:.1f}")
        left = max(0.0, self.owed_now(sale))
        let_go = [run_id[i]["c"]["target"] for i in drop]
        less = bool(let_go) or left > st["left"] + st["tol"]
        told = None
        if args is not None and less and said and NO_REASON not in flags:
            parts = ([f'I let {", ".join(let_go)} go.'] if let_go else []) + ([f'I left {fmt(left)} kW of the {fmt(float(sale["owed_kw"]))} kW owed uncovered.'] if left > st["tol"] else [])
            told = " ".join(parts + [said])
            self.tell(told, "opening" if st["why"] == OPENS else "exception", ts)
        by = "agent" if args is not None and NO_REASON not in flags else f"rule: {NO_REASON}" if NO_REASON in flags else f'rule: {rec.get("fallback") or no_ask or "code"}'
        letters = {o["id"]: f'{o["d"]["tool"]} {o["d"].get("target") or ""}'.strip() for o in st["options"]}      # so the receipt reads by itself: a letter is this moment's
        letters.update({r["id"]: f'{r["c"]["tool"]} {r["c"].get("target") or ""}'.strip() for r in st["running"]})
        return self.receipt("opening" if st["why"] == OPENS else "exception", ts, asked={"what_changed": st["why"], "owed_kw": round(st["owed"], 2), "question": R.sha(rec.get("text") or ""),
                                                                                          "letters": letters},
                            rule={"open": st["default"], "left_kw": round(st["left"], 2)}, chose={"open": pick, "release": let_go, "left_kw": round(left, 2), "report": (args or {}).get("report") if isinstance(args, dict) else None},
                            reason=NO_REASON if NO_REASON in flags else said if args is not None else (rec.get("fallback") or no_ask or ""), facts=[{"id": cited["id"], "says": cited["says"]}] if cited else [],
                            flags=flags, by=by, less=less, told=told)

    def receipt(self, point: str, ts: str, *, asked: Any, rule: Any, chose: Any, reason: str, facts: list[dict[str, Any]], flags: list[str], by: str, less: bool,
                told: str | None = None) -> dict[str, Any]:
        """Every node an agent: one decision's receipt, in the node's own decision log (node/record.py): what was asked, the rule's plan, what the agent chose,
        its reason and the facts that reason rests on. A flag says a reason was missing or cited no fact, and each flag is counted."""
        for x in flags:
            self.flags[x] += 1
        r = {"node": self.name, "point": point, "ts": ts, "asked": asked, "rule": rule, "chose": chose, "less_than_rule": bool(less), "by": by,
             "reason": reason or (NO_REASON if less else ""), "facts": facts, "flags": list(flags), **({"told": told} if told else {})}
        self.receipts.append(r)
        self.ctx.log.add("receipt", **{k: v for k, v in r.items() if k != "node"})
        return r

    def tell(self, text: str, point: str, ts: str) -> None:
        """Every node an agent: the owner, or a building's manager, hears in plain words why the node gives less than the rule's amount. It goes on the person's
        channel, here the node's own list and log, and into the day's report."""
        m = {"ts": ts, "point": point, "to": "manager" if self.word == "building" else "owner", "text": str(text)[:400]}
        self.told.append(m)
        self.ctx.log.add("told", **m)

    def take_plan(self, args: dict[str, Any]) -> dict[str, Any]:
        """A plan that comes through the tool server: the same check and the same order."""
        ts, sale = self.ctx.ts, self.ctx.sale
        st = self.situation(ts, sale, "A plan came through the tool server.", self.used_now(), self.kept_now())
        self.carry_out(st, args, ts, sale, None)
        return {"status": "done", "view": f"Plan set. {self.owed_now(sale):.2f} kW are not covered."}

    def let_go(self, target: str) -> None:
        """One running call ends: its device leaves the sale at once, and its point is free again."""
        tools = {c["tool"] for c in self.calls if c["target"] == target}
        self.calls = [c for c in self.calls if c["target"] != target]
        self.guard.writers.pop(target, None)
        if self.sim.levers is not None and tools:
            self.reopen(tools)

    def first_needs(self, d: dict[str, Any], f: dict[str, Any], done: set[str]) -> dict[str, Any] | None:
        """The call this one needs in place before it. A building's fans wait for their zones."""
        return None

    def prerequisite(self, d: dict[str, Any] | None, f: dict[str, Any]) -> dict[str, Any] | None:
        """The device this one waits for, which code sets first when a plan opens it (first_needs). A home's devices wait for nothing."""
        return None

    def after_test(self, f: dict[str, Any], d: dict[str, Any] | None, args: dict[str, Any]) -> dict[str, Any]:
        """The node as a test sees it once this call has gone through: what it unlocks for the calls after it. A home's calls unlock nothing."""
        return f

    def walk_unlocked(self, ts: str, sale: dict[str, Any], done: set[str], why: str) -> None:
        """What a plan's calls opened up. A building's plant waits for its zones to coast."""
