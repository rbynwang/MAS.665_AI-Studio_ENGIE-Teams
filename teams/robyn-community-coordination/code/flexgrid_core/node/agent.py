"""A node's agent: code owns the steps and the deadlines, and the model chooses inside a step (MULTI_AGENT_DESIGN, section 6).

The agent steps in only when something needs judgment: before a sale, when the plan falls short, or when something changes.
Every other moment belongs to the 5-minute controller. Each call goes agent, controller, guard, device.

  Brain       one model answer with tools, through Ollama, the settings the tool tests used
  NodeAgent   decide(): the loop of one step; step(): when to wake up
  open_levers / apply_call / close_levers   the controller's side: what a call changes on the simulated device

If the model is down or answers nothing, the node falls back to code: the controller may use every device.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any, Callable

import httpx

from . import context, owner, report, sellable
from . import tools as T
from .guard import Guard
from .see import MACHINE_TARGET, can_act, render, see_home, shortlist

MODEL = "qwen3.8-27b-q3:latest"
URL = "http://127.0.0.1:11434"
OPTIONS = {"temperature": 0.3, "top_p": 0.8, "top_k": 20, "num_predict": 300, "num_ctx": 32768}
LEAD_MIN = int(os.environ.get("FG_LEAD_MIN", "30"))     # the agent plans the sale this long before it opens. A run on a fast clock needs more: 23 nodes take about 7 wall minutes
MAX_CALLS = 6          # device calls in one step
CUSHION = 1.1          # a call is sized this far over what is owed: expected kW run high, and homes gave 92% of what they sold where code gives 108%
SHORT_SHARE = 0.15     # the plan may fall this share short before the agent is woken
SHORT_KW = 0.1
QUIET_MIN = 30         # at least this long between wake-ups for a shortfall
MACHINE_KEY = {v: k for k, v in MACHINE_TARGET.items()}
CODE_STEPS = ("1 read schedule", "2 forecast", "3 check devices", "4 estimate sellable")


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
        self.calls = 0
        self.ms = 0.0
        self.tokens_in = 0
        self.tokens_out = 0
        self.errors = 0

    def ask(self, system: str, tools: list[dict[str, Any]], text: str) -> dict[str, Any]:
        t0 = time.time()
        # truncate and shift as llm.py sends them: without them a request can hang after the model's runner restarts
        body = {"model": self.model, "stream": False, "think": False, "keep_alive": "30m", "truncate": False, "shift": False, "tools": tools, "options": OPTIONS,
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
def open_levers(sim: Any) -> None:
    """From now on the controller uses only what the agent's calls open."""
    # the car's own lever is its kW, so it stays in "devices" and gives nothing until a call sets that
    sim.levers = {"battery_kw": 0.0, "ev": False, "ev_kw": None, "v2h_kw": 0.0, "v2h_kwh": None, "devices": {"v2h"}}
    sim.rt_override("comfort", held_comfort(sim))


def held_comfort(sim: Any) -> dict[str, float]:
    """No easing, and no pre-cooling below where the house is: extra cooling before a sale nobody opened is load nobody pays for."""
    return {"max_rise_c": 0.0, "max_rise_away_c": 0.0, "precool_c": max(0.0, -sim.x)}


def close_levers(sim: Any) -> None:
    """Every write ends by itself, and the device goes back to its own day."""
    sim.levers = None
    sim.overrides.pop("comfort", None)


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
    elif tool in ("set_thermostat", "set_room_ac"):
        c, lim = float(v.get("less_comfort_c", 0.0)), sim.ctl.limits
        # each keeps its own limit: the call is sized for who is home now, and a home that fills up in the sale goes back to the tighter one
        sim.rt_override("comfort", {**held_comfort(sim), "max_rise_c": min(c, lim.max_rise_c), "max_rise_away_c": min(c, lim.max_rise_away_c)})
    elif tool == "set_water_heater" and v.get("action") == "hold":
        lev["devices"].add("hpwh")
    elif tool == "set_pool_pump":
        lev["devices"].add("pool")
    elif tool == "delay_appliance":
        lev["devices"].add(MACHINE_KEY[ans["target"]])


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
        self.noted: str | None = None
        self.tripped = False
        self.plugged = False
        self.waiting: Any = None
        self.call_kw = 0.0                          # kW dispatched on top of the sale since the last step
        self.log_calls: list[dict[str, Any]] = []   # every call this sale, for step 9
        self.reported: str | None = None
        self.reports: list[dict[str, Any]] = []
        self.steps: list[tuple[str, str]] = []      # (step, when): the workflow's record

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

    def market(self, step: str, ts: str) -> None:
        """Steps 1 to 6 run on the market's side: code reads the schedule, forecast and devices to make an offer, and brain.HomeBrain asks the model. Here they enter the
        workflow's record. ctx.step stays as it is: this comes on the market's thread, and a sale step may be waiting on the model at this moment."""
        for s in (*CODE_STEPS, step) if step == "5 offer" else (step,):
            self.steps.append((s, ts))
            self.ctx.steps.enter(s, ts)
            self.ctx.steps.leave(s, ts, "done")

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
            sim.overrides.pop("comfort", None)
            self.all_open(keep)
            return sim.rt_capacity(start, end)
        finally:
            sim.levers = lev
            sim.rt_override("comfort", comfort)

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

    def decide(self, ts: str, sale: dict[str, Any], why: str | None = None) -> None:
        if self.sim.levers is None or self.ctx.handed_back:
            return                                  # the model went earlier in this sale, so the controller is running it
        kept = owner.blocked_tools(self.guard.wishes)
        used = {c["tool"] for c in self.calls} | kept
        done = [f'{c["says"]}' for c in self.calls] + [owner.SAYS[w["wish"]] for w in self.guard.wishes]
        block, reply_retry, reads = None, False, True
        for _ in range(MAX_CALLS):
            f = self.view(ts, sale)
            owed = self.owed_now(sale)
            if owed <= 0.05 and not why:
                break                               # nothing owed: code doesn't call the model at all
            if not any(can_act(f, d) and d["tool"] not in used for d in f["devices"].values()):
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
                    reads = False                   # the step already showed every reading it has
                    continue
                if c["name"] in context.CONTEXT:
                    done.append(self.ctx.call(c["name"], c["args"])["view"])
                    stop = stop or c["name"] == "report_shortfall"
                    continue
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

    def read_note(self, ts: str) -> str | None:
        """A new note: the model says what the owner needs, and code keeps it from here on."""
        note = self.sim.overrides.get("note")
        if not note or note == self.noted:
            return None
        self.noted = note
        r = owner.read_note(self.brain, note)
        self.record({"node": self.name, "tier": self.tier, "ts": ts, "why": "owner's note", "text": note, "shortlist": ["set_owner_wish", "no_wish"],
                     "answer": r["wishes"], "said": r["said"], "ms": r["ms"], "model": self.brain.model, "prompt": T.PROMPT_VERSION, "owed_kw": 0.0, "verdicts": []})
        if not r["wishes"]:
            return None
        self.guard.wishes += r["wishes"]
        blocked = owner.blocked_tools(r["wishes"])
        if any(w["wish"] == "keep_car_ready" for w in r["wishes"]):
            self.sim.rt_override("ev_now", True)                  # the owner's car first
        self.calls = [c for c in self.calls if c["tool"] not in blocked]
        if self.sim.levers is not None:
            self.reopen(blocked)
        return owner.SAYS[r["wishes"][0]["wish"]]

    def reopen(self, blocked: set[str]) -> None:
        """Take back what the owner's wish stops: the device leaves the sale at once."""
        lev = self.sim.levers
        if "set_battery" in blocked:
            lev["battery_kw"] = 0.0
        if "limit_ev_charging" in blocked:
            lev["ev"], lev["ev_kw"] = False, None
        if "discharge_car" in blocked:
            lev["v2h_kw"] = 0.0
        if blocked & {"set_thermostat", "set_room_ac"}:
            self.sim.rt_override("comfort", held_comfort(self.sim))
        held = lev.get("devices", set())            # a building's levers have none of these
        if "set_water_heater" in blocked:
            held.discard("hpwh")
        if "set_pool_pump" in blocked:
            held.discard("pool")
        if "delay_appliance" in blocked:
            held -= set(MACHINE_KEY.values())

    def step(self, ts: str) -> None:
        """Called every 15 minutes. Most steps do nothing: the controller is running the node."""
        from ..beckn import add_minutes
        alarm = self.sim.overrides.get("alarm") or self.ctx.stopped
        if alarm:
            self.ctx.alarm(alarm)
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
            if self.done_for == sale["start"] and self.reported != sale["start"]:
                self.reported = sale["start"]
                self.steps.append(("9 report", ts))
                self.ctx.enter("9 report", ts, sale)
                self.reports.append(report.write(self, sale, ts))
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
            self.decide(ts, sale)
            self.ctx.leave()          # a step ends with its work: no tools stay open until the next one
            return
        if self.done_for != sale["start"]:
            return
        why = note_why
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
