"""The AI agents' decisions, with FlexGrid's one rule: the LLM proposes, the deterministic code decides.

These agents think with the local LLM (Qwen3.8-27B on the Mac; see llm.py, and OpenClawBackend below for the agent
runtime): the two Jetson home agents (HomeBrain), a building node's agent (BuildingBrain: the library on
jetson-a, its facility manager's chat), the data center agent (DataCenterBrain), the grid operator's
(GridOperatorBrain), the coordinator (CoordinatorBrain) and the buyer (its own
MCP tools, agent/). The 98 simulated homes and every offline simulation keep their rules: nothing here runs unless a
daemon or the coordinator attaches a brain.

What the LLM may change, and the code that holds it to that:
  home, offer       whether to take part, what share of the battery's SAFE offer to give (0-1), whether to pause the
                    EV. It can only lower the planner's offer; the owner's wishes are applied after it, always
  home, commitment  accept or decline a commitment the trust guard already found safe (it cannot accept one the guard
                    declined)
  coordinator       the margin it holds back when it sells to the utility, 5-30 % (default 10 %)
  building, offer   whether to take part, the share of its cooling and of its dimmable lighting in the SAFE offer (0-1
                    each); the facility manager's policy (FacilityPolicy) after it. Accept or decline as a home does.
                    By chat: the comfort limit (0-3 C), the lights' dimming limit (0-50 %), leave the event; the
                    building's code clamps every value (building.py)
  data center, grid operator: see DataCenterBrain and GridOperatorBrain below
Each decision has a time budget. On a timeout, an error or an answer the code cannot use, the agent takes today's rule
(the planner's offer with the owner's wishes; accept; 10 %), and the decision log says so. Every decision is logged
with its reason, the model and the seconds it took: the terminals, the Control Room's chat and the phone app show it.

The owner's wishes (the phone app's Agent tab, or set by the agent in a chat when the owner asks) are structured:
skip_until / skip_days (no events), no_ev_pause, battery_max_share (0-1), min_price_usd_per_kwh, and the owner's own
words (notes). They work without the LLM too: the guard applies them to every offer and commitment.
"""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from . import beckn

DEFAULT_WISHES = {"skip_until": None, "skip_days": [], "no_ev_pause": False, "battery_max_share": 1.0,
                  "min_price_usd_per_kwh": 0.0, "notes": [], "devices_off": []}
REBOUND_FRACTIONS = (0.25, 0.5, 0.75, 1.0)


def _num(x: Any, lo: float, hi: float, default: float) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return default if v != v else max(lo, min(hi, v))


def _day(x: Any) -> str | None:
    try:
        return datetime.strptime(str(x)[:10], "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        return None


class Wishes:
    """The owner's wishes for this home, on disk next to its state (survive restarts and demo resets)."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        try:
            self.w = {**DEFAULT_WISHES, **json.loads(path.read_text())}
        except (OSError, ValueError):
            self.w = json.loads(json.dumps(DEFAULT_WISHES))

    def get(self) -> dict[str, Any]:
        with self.lock:
            return json.loads(json.dumps(self.w))

    def set(self, **ch: Any) -> dict[str, Any]:
        """Validated changes; unknown keys are ignored. Returns the wishes now in force."""
        with self.lock:
            w = self.w
            if "skip_until" in ch:
                w["skip_until"] = _day(ch["skip_until"]) if ch["skip_until"] else None
            if "skip_days" in ch and isinstance(ch["skip_days"], list):
                w["skip_days"] = sorted({d for d in (_day(x) for x in ch["skip_days"]) if d})[:31]
            if "add_skip_day" in ch and _day(ch["add_skip_day"]):
                w["skip_days"] = sorted(set(w["skip_days"]) | {_day(ch["add_skip_day"])})[:31]
            if "no_ev_pause" in ch:
                w["no_ev_pause"] = bool(ch["no_ev_pause"])
            if isinstance(ch.get("devices_off"), list):
                w["devices_off"] = sorted({str(x)[:32] for x in ch["devices_off"]})[:32]
            if "battery_max_share" in ch:
                w["battery_max_share"] = round(_num(ch["battery_max_share"], 0.0, 1.0, w["battery_max_share"]), 2)
            if "min_price_usd_per_kwh" in ch:
                w["min_price_usd_per_kwh"] = round(_num(ch["min_price_usd_per_kwh"], 0.0, 10.0, w["min_price_usd_per_kwh"]), 2)
            if ch.get("note"):
                w["notes"] = (w["notes"] + [str(ch["note"])[:200]])[-5:]
            if ch.get("clear"):
                self.w = w = {**json.loads(json.dumps(DEFAULT_WISHES)), "notes": w["notes"]}
            try:
                self.path.write_text(json.dumps(w, indent=1))
            except OSError:
                pass
            return json.loads(json.dumps(w))

    def skips(self, day: str) -> str | None:
        """Why this day is off by the owner's wish, or None."""
        w = self.get()
        if w["skip_until"] and day <= w["skip_until"]:
            return f"the owner asked to skip events until {w['skip_until']}"
        if day in w["skip_days"]:
            return f"the owner asked to skip {day}"
        return None


def drop_devices(offer: dict[str, Any], off: list[str]) -> dict[str, Any]:
    """The offer without these devices' part (never more than it was): the planner's kW for the rest, in proportion."""
    a = offer.get("appliances") or {}
    by = a.get("by_device") or {}
    total = sum(by.values())
    gone = sum(v for k, v in by.items() if k in off)
    if not off or total <= 0 or gone <= 0:
        return offer
    out = json.loads(json.dumps(offer))
    keep = 1.0 - gone / total
    for s in out["slots"]:
        a0 = float(s.get("app_kw") or 0.0)
        s["app_kw"] = round(a0 * keep, 3)
        s["kw_offered"] = round(max(0.0, s["kw_offered"] - (a0 - s["app_kw"])), 2)
    out["energy_kwh"] = round(sum(s["kw_offered"] for s in out["slots"]) * beckn.SLOT_MINUTES / 60, 2)
    out["appliances"] = {**a, "kw": round(float(a.get("kw", 0.0)) * keep, 3), "mean_kw": round(float(a.get("mean_kw", 0.0)) * keep, 3),
                         "by_device": {k: v for k, v in by.items() if k not in off}}
    return out


def scale_offer(offer: dict[str, Any], share: float, pause_ev: bool, take_part: bool) -> dict[str, Any]:
    """The planner's safe offer with only part of its battery and, or not, its EV pause. Never more than the planner
    offered. The evening forecasts follow: the rebound at a delivery level f of the new offer is the planner's at
    f x share, and without the EV pause the evening is the no-event one (the EV charges as usual). An offer with no EV
    pause in its window keeps the planner's evening and car as they are (there is no pause to give up), so the rules'
    choices give the planner's offer to the decimal: until 1 Nov 2026 such a home lost its car's evening flexibility,
    which tipped the coordinator's rebound guard on 4 of the season's 45 days (tools/season_all_ai.py)."""
    from .market import _rebound_at
    share = max(0.0, min(1.0, share)) if take_part else 0.0
    ev_on = pause_ev and take_part
    had_ev = any(float(s.get("ev_pause_kw") or 0.0) > 0.0 for s in offer["slots"])
    out = json.loads(json.dumps(offer))
    for s in out["slots"]:
        b0, e0, k0 = s["battery_kw"], s["ev_pause_kw"], s["kw_offered"]
        s["battery_kw"] = round(b0 * share, 2)
        s["ev_pause_kw"] = round(e0 if ev_on else 0.0, 2)
        other = 0.0 if not take_part else float(s.get("ac_kw") or 0.0) + float(s.get("app_kw") or 0.0)   # benchmark
        # what is taken away, from the planner's own total (its parts are rounded separately): unchanged = the plan
        s["kw_offered"] = 0.0 if s["battery_kw"] + s["ev_pause_kw"] + other <= 0 else \
            round(max(0.0, min(k0, k0 - (b0 - s["battery_kw"]) - (e0 - s["ev_pause_kw"]))), 2)
    if all(abs(a["kw_offered"] - b["kw_offered"]) < 1e-9 for a, b in zip(out["slots"], offer["slots"])):
        out["energy_kwh"] = offer["energy_kwh"]
    else:
        out["energy_kwh"] = round(sum(s["kw_offered"] for s in out["slots"]) * beckn.SLOT_MINUTES / 60, 2)
    ev = out.get("evening") or {}
    if ev.get("rebound_kw") and share < 0.999:
        old = offer["evening"]["rebound_kw"]
        ev["rebound_kw"] = {str(f): [round(x, 2) for x in _rebound_at(old, f * share)] for f in REBOUND_FRACTIONS}
    if not ev_on and had_ev:
        if ev.get("import_cf_kw"):
            ev["import_noev_kw"] = list(ev["import_cf_kw"])
        out["ev_flex"] = None
    return out


def _plain(o: Any) -> Any:
    """json.dumps default: numpy scalars (and anything else with .item() or .tolist()) as plain Python values."""
    for m in ("item", "tolist"):
        if hasattr(o, m):
            return getattr(o, m)()
    return str(o)


class Decisions:
    """The agent's decision log (the terminals, the Control Room's chat, the phone app)."""

    def __init__(self, n: int = 60):
        self.items: deque = deque(maxlen=n)
        self.lock = threading.Lock()
        self.seq = 0

    def add(self, **d: Any) -> dict[str, Any]:
        # plain JSON values only: the planner's numbers can be numpy scalars, and a numpy bool (a fallback's "take
        # part" from a numpy comparison) breaks every JSON writer downstream (the daemon's /agent/state, results)
        d = json.loads(json.dumps(d, default=_plain))
        with self.lock:
            self.seq += 1
            d = {"n": self.seq, "wall": round(time.time(), 3), **d}
            self.items.append(d)
            return d

    def list(self, n: int = 30) -> list[dict[str, Any]]:
        with self.lock:
            return list(self.items)[-n:]


# --------------------------------------------------------------------------------------------- the home agent
OFFER_SCHEMA = {"type": "object", "properties": {
    "take_part": {"type": "boolean"},
    "battery_share": {"type": "number", "minimum": 0, "maximum": 1},
    "pause_ev": {"type": "boolean"},
    "devices_off": {"type": "array", "items": {"type": "string"}},
    "reason_for_owner": {"type": "string"},
    "reason_for_log": {"type": "string"}},
    "required": ["take_part", "battery_share", "pause_ev", "reason_for_owner"]}
ACCEPT_SCHEMA = {"type": "object", "properties": {
    "accept": {"type": "boolean"}, "reason_for_owner": {"type": "string"}, "reason_for_log": {"type": "string"}},
    "required": ["accept", "reason_for_owner"]}

HOME_DECIDE = """You are the AI energy agent of one household in a FlexGrid community (Boston area, Massachusetts).
The grid buyer asks the community to use less power in an evening window and pays per kWh delivered. Your home's
planner has already worked out a SAFE offer: what the battery can give above its backup reserve, and what pausing
the EV charger saves. You decide how much of it to offer: take part or not, the share of the battery part (0 to 1),
and whether to pause the EV. The home's other smart devices are listed under "devices" with the kW each can give; name
any you want left out of this event in devices_off (for example the dryer when the owner's words ask for laundry tonight).
You can only lower the safe offer; the home's code enforces every limit and the owner's wishes after you. Promising more than the home delivers costs reputation and future income, so offer less when the
forecast is uncertain, the battery will start low, or the owner's wishes or words point that way; offer fully when it
is safe and worth it. Reasons: one or two plain sentences, with the numbers that mattered."""

HOME_ACCEPT = """You are the AI energy agent of one household in a FlexGrid community. The coordinator has sent this
home its share of the community's commitment. The home's trust guard has already checked it is within your offer and
safe. Accept unless something changed since the offer that makes it unwise (the owner's wishes or words, the battery
far lower than expected). Declining costs the community and this home's reputation. One or two plain sentences."""

HOME_CHAT = """You are the AI energy agent of {home} ({label}), a household in a FlexGrid community in Worcester County,
Massachusetts. You talk with the homeowner in their phone app. FlexGrid pays households for using less power from the
grid in evening windows when the grid is strained: you offer what the battery and EV can safely give, the
coordinator pools 100 homes, the grid buyer pays per kWh delivered, measured on signed meter readings.
Rules: answer in at most 4 short sentences, plain words. Take every number from your tools; never guess a number.
Times are the community's replay clock (summer 2026). When the owner asks you to change how you take part (skip
days, keep the EV charging, use less of the battery, a minimum price), call set_wishes, then say what changed.
During the day the home re-plans every 5 minutes. When the owner needs the car now, call car_now. When the owner says
how much warmer the house may get while the air conditioning eases off ("a bit warmer is fine", "don't touch the AC"),
turn it into degrees (0-4) and call set_comfort_limit. When they say how much battery the grid may use in an emergency,
call set_emergency_permission. real_time_now tells you what the home owes now and the limits in force.
You never control devices directly, and the home's safety limits always apply: the car always leaves charged."""


class HomeBrain:
    CHAT, ROLE = HOME_CHAT, "home"           # the owner's chat: its system prompt, the agent's role for the runtime

    def __init__(self, v2, backend, wishes: Wishes, *, label: str = "", log: Callable[..., None] | None = None,
                 offer_budget_s: float = 40.0, accept_budget_s: float = 45.0, chat_budget_s: float = 120.0):
        self.v2, self.backend, self.wishes, self.label = v2, backend, wishes, label
        self.log = log or (lambda *a, **k: None)
        self.offer_budget_s, self.accept_budget_s, self.chat_budget_s = offer_budget_s, accept_budget_s, chat_budget_s
        self.decisions = Decisions()
        self.needs: dict[str, dict[str, Any]] = {}
        self.forecasts: dict[str, dict[str, Any]] = {}
        self._done: dict[str, Any] = {}               # event / contract id -> the decision already made (redelivery,
        self._lock = threading.Lock()                 # or the Langflow step and the in-process path asking twice)
        self.chat_log: deque = deque(maxlen=40)
        self.chat_lock = threading.Lock()
        self.tool_calls: deque = deque(maxlen=60)    # every tool run (by our own loop, or by OpenClaw through MCP)

    def run_tool(self, name: str, args: dict[str, Any] | None = None, *, by: str = "agent") -> Any:
        """One of the agent's tools, run by the home's own code with the arguments checked there. OpenClaw's agent
        calls them through flexgrid_core.agent_mcp (POST /agent/tool/{name} on this daemon); the result is recorded so
        the phone app can show which tools an answer used."""
        tools = self.tools()
        t0 = time.perf_counter()
        if name not in tools:
            res: Any = {"error": f"no tool called {name!r}"}
        else:
            try:
                res = tools[name][1](**(args if isinstance(args, dict) else {}))
            except TypeError as exc:
                res = {"error": f"bad arguments: {str(exc)[:120]}"}
            except Exception as exc:
                res = {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
        self.tool_calls.append({"tool": name, "args": args or {}, "result": res, "by": by, "wall": time.time(),
                                "ms": round((time.perf_counter() - t0) * 1000, 1)})
        return res

    def reset(self) -> None:
        """A new demo run replays the same days: forget the decisions (the owner's wishes and chat stay)."""
        with self._lock:
            self._done.clear()
            self.needs.clear()
            self.forecasts.clear()
            self.decisions = Decisions()
    def _devices(self) -> list[dict[str, Any]]:
        """Every smart system this node has that the agent can call, beyond the battery and the car's pause."""
        a = self.v2.adapter
        off = set(self.wishes.get().get("devices_off") or [])
        by = ((self.v2.offers.get(max(self.v2.offers, key=lambda e: self.v2.offers[e].get("made_at", ""))) or {})
              .get("appliances") or {}).get("by_device", {}) if self.v2.offers else {}
        return [{"device": k, "what": s.get("label"), "how": s.get("kind"), "control": s.get("path"),
                 "allowed": k not in off, "kw_last_offer": by.get(k)} for k, s in (getattr(a, "app", None) or {}).items()]

    def _facts_common(self) -> dict[str, Any]:
        tel = self.v2.adapter.tel or {}
        lim = self.v2.adapter.limits()
        done = [c for c in self.v2.contracts.values() if c.get("status") == "ACCEPTED"]
        return {"home": self.v2.name, "now": self.v2.now_ts(),
                "battery": {"kwh": lim["battery_kwh"], "kw": lim["battery_kw"], "reserve_pct": lim["reserve_pct"],
                            "soc_now_pct": tel.get("soc_pct")},
                "devices": self._devices(),
                "owner_wishes": self.wishes.get(),
                "track_record": {"events_accepted": len(done), "paid_usd": round(sum(float(p.get("payment") or 0)
                                                                                    for p in self.v2.payments), 2),
                                 "reputation": round(self.v2.reputation, 3)}}

    @staticmethod
    def _avg(steps: list[dict[str, Any]], k: str) -> float | None:
        v = [s[k] for s in steps if s.get(k) is not None]
        return round(sum(v) / len(v), 2) if v else None

    def _fc_facts(self, fc: dict[str, Any]) -> dict[str, Any]:
        st = fc.get("steps") or []
        ev = fc.get("ev") or {}
        out = {"method": "trained models" if str(fc.get("method", "")).startswith("trained") else str(fc.get("method", ""))[:60],
               "window_house_load_kw": {"p10": self._avg(st, "load_p10"), "p50": self._avg(st, "load_p50"),
                                        "p90": self._avg(st, "load_p90")},
               "window_solar_kw": {"p10": self._avg(st, "pv_p10"), "p50": self._avg(st, "pv_p50"), "p90": self._avg(st, "pv_p90")}}
        if ev:
            out["ev"] = {k: ev.get(k) for k in ("expected_kwh_15_20", "p_charging", "kwh", "deadline") if ev.get(k) is not None}
        return out

    # ---- decision 1: the offer
    def decide_offer(self, need: dict[str, Any], fc: dict[str, Any], offer: dict[str, Any]) -> dict[str, Any]:
        eid = need["event_id"]
        with self._lock:
            if ("offer", eid) in self._done:
                return self._done[("offer", eid)]
        self.needs[eid], self.forecasts[eid] = need, fc
        sl = need.get("slots") or []
        price = float(sl[0].get("price") or 0.0) if sl else 0.0
        day = need["window_start"][:10]
        w = self.wishes.get()
        skip = self.wishes.skips(day) or (f"the price ${price:.2f}/kWh is below the owner's minimum "
                                          f"${w['min_price_usd_per_kwh']:.2f}" if price < w["min_price_usd_per_kwh"] else None)
        safe_kwh = offer["energy_kwh"]
        has_ev = any(s.get("ev_pause_kw", 0) > 0 for s in offer["slots"])
        t0 = time.perf_counter()
        ans, meta = None, {"model": None, "ms": None, "error": None}
        if skip:
            meta["error"] = "the owner's wish decides"
        elif safe_kwh <= 0.01:
            meta["error"] = "nothing safe to offer (the planner offered 0 kW)"
        else:
            facts = {**self._facts_common(),
                     "request": {"event_id": eid, "date": day, "window": f"{need['window_start'][11:]}-{need['window_end'][11:]}",
                                 "kw_asked_per_half_hour": [s.get("kw_requested") for s in sl], "price_usd_per_kwh": price,
                                 "reply_by": need.get("commit_by")},
                     "safe_offer": {"per_half_hour": [{"start": s_need.get("start", "")[11:], "battery_kw": s["battery_kw"],
                                                       "ev_pause_kw": s["ev_pause_kw"]}
                                                      for s, s_need in zip(offer["slots"], sl + [{}] * len(offer["slots"]))],
                                    "energy_kwh": safe_kwh, "confidence": offer.get("confidence"),
                                    "if_all_delivered_usd": round(safe_kwh * price, 2)},
                     "forecast": self._fc_facts(fc),
                     "battery_at_window_start_kwh": (offer.get("battery") or {}).get("window_start_kwh")}
            facts["safe_offer"]["other_devices_kw"] = (offer.get("appliances") or {}).get("by_device") or {}
            if not has_ev:
                facts["safe_offer"]["note"] = "no EV pause possible in this window"
            ans, meta = self.backend.decide("home", HOME_DECIDE, facts, OFFER_SCHEMA, budget_s=self.offer_budget_s,
                                            session=f"{self.v2.name}-offer-{eid}")
        fallback = ans is None
        if fallback:                                  # plain Python values (safe_kwh may be a numpy float)
            take, share, pause = bool(not skip and safe_kwh > 0.01), float(w["battery_max_share"]), bool(not w["no_ev_pause"])
            reason = skip or ("rules: the planner's safe offer" + (f" ({meta['error']})" if meta.get("error") else ""))
            why_log = reason
        else:
            take = bool(ans.get("take_part"))
            share = _num(ans.get("battery_share"), 0.0, 1.0, 1.0)
            pause = bool(ans.get("pause_ev"))
            reason = str(ans.get("reason_for_owner") or "")[:400]
            why_log = str(ans.get("reason_for_log") or reason)[:400]
        # the guard: the owner's wishes after the LLM, always
        clamped = []
        if skip and take:
            take = False; clamped.append("owner's wish: skip")
        if share > w["battery_max_share"] + 1e-9:
            share = w["battery_max_share"]; clamped.append(f"battery share capped at the owner's {w['battery_max_share']:.0%}")
        if pause and w["no_ev_pause"]:
            pause = False; clamped.append("owner's wish: keep the EV charging")
        have = set((getattr(self.v2.adapter, "app", None) or {}))
        asked_off = ans.get("devices_off") if (not fallback and isinstance(ans.get("devices_off"), list)) else []
        off = sorted(have & ({str(x) for x in asked_off} | set(w.get("devices_off") or [])))
        if have:
            self.v2.adapter.dev_off = set(off)
        final = scale_offer(drop_devices(offer, off), share, pause and has_ev, take)
        d = self.decisions.add(kind="offer", event_id=eid, sim=self.v2.now_ts(), take_part=take,
                               battery_share=round(share if take else 0.0, 2), pause_ev=bool(pause and has_ev and take),
                               safe_kw=[s["kw_offered"] for s in offer["slots"]], kw=[s["kw_offered"] for s in final["slots"]],
                               price=price, if_delivered_usd=round(final["energy_kwh"] * price, 2),
                               reason=reason, reason_log=why_log, guard=clamped, fallback=fallback,
                               llm={"model": meta.get("llm_model") or meta.get("model"), "ms": meta.get("ms"), "error": meta.get("error"),
                                    "runtime": meta.get("runtime")})
        # what was decided goes out with the offer (signed with it): the coordinator keeps it, and the offline replay
        # of the event (verify.replay_total) applies the same decision with the same code
        final["agent"] = {"decided_by": "rules" if fallback else "llm", "reason": reason[:300], "decision": d["n"],
                          "take_part": bool(take), "battery_share": float(share), "pause_ev": bool(pause and has_ev),
                          "devices_off": off}
        self.log("tool", tool="agent_decides_offer",
                 input=f"safe {' · '.join(f'{s:g}' for s in d['safe_kw'])} kW, ${price:.2f}/kWh, wishes "
                       f"{'skip' if skip else 'none' if w == DEFAULT_WISHES else 'set'}",
                 output=(f"{'takes part' if take else 'sits out'}: {' · '.join(f'{x:g}' for x in d['kw'])} kW "
                         f"({'rules' if fallback else meta.get('model')}, {meta.get('ms') or 0} ms): {reason[:160]}"),
                 ms=round((time.perf_counter() - t0) * 1000, 1))
        self.log("agent_decision", what="offer", event_id=eid, take_part=take, kw=d["kw"], fallback=fallback,
                 reason=reason[:200])
        with self._lock:
            self._done[("offer", eid)] = final
        return final

    # ---- decision 2: the commitment
    def decide_accept(self, c: dict[str, Any], verdict: dict[str, Any]) -> dict[str, Any]:
        cid = c.get("id")
        if verdict.get("decision") != "ACCEPTED" or not cid:
            return verdict                                   # the guard said no: the LLM is not asked
        with self._lock:
            if ("accept", cid) in self._done:
                return self._done[("accept", cid)]
        day = str(c.get("window_start", ""))[:10]
        skip = self.wishes.skips(day)
        offer = self.v2.offers.get(c.get("event_id")) or {}
        need = self.needs.get(c.get("event_id")) or {}
        sl = need.get("slots") or []
        price = float(sl[0].get("price") or 0.0) if sl else float(c.get("price_usd_per_kwh") or 0.0)
        kw = [float(x.get("kw", 0)) for x in c.get("commitments", [])]
        usd = round(sum(kw) * beckn.SLOT_MINUTES / 60 * price, 2)
        t0 = time.perf_counter()
        ans, meta = None, {"model": None, "ms": None, "error": None}
        if skip:
            meta["error"] = "the owner's wish decides"
        else:
            facts = {**self._facts_common(),
                     "commitment": {"event_id": c.get("event_id"), "window": f"{str(c.get('window_start', ''))[11:]}"
                                    f"-{str(c.get('window_end', ''))[11:]}", "per_half_hour_kw": kw,
                                    "price_usd_per_kwh": price, "if_delivered_usd": usd,
                                    "opt_out_by": c.get("opt_out_by")},
                     "our_offer_kw": [s.get("kw_offered") for s in offer.get("slots", [])],
                     "why_we_offered": (offer.get("agent") or {}).get("reason"),
                     "safety_check": verdict.get("reason")}
            ans, meta = self.backend.decide("home", HOME_ACCEPT, facts, ACCEPT_SCHEMA, budget_s=self.accept_budget_s,
                                            session=f"{self.v2.name}-accept-{cid}")
        fallback = ans is None
        if skip:
            out = {"decision": "DECLINED", "reason": "homeowner_opted_out", "contract_id": cid}
            reason = skip
        elif fallback or bool(ans.get("accept")):
            out = verdict
            reason = "rules: within the offer and safe" if fallback else str(ans.get("reason_for_owner") or "")[:400]
        else:
            out = {"decision": "DECLINED", "reason": "agent_declined", "contract_id": cid}
            reason = str(ans.get("reason_for_owner") or "")[:400]
        self.decisions.add(kind="accept", event_id=c.get("event_id"), contract_id=cid, sim=self.v2.now_ts(),
                           accept=out["decision"] == "ACCEPTED", decision=out["decision"], kw=kw, price=price,
                           if_delivered_usd=usd, reason=reason, fallback=fallback,
                           llm={"model": meta.get("llm_model") or meta.get("model"), "ms": meta.get("ms"), "error": meta.get("error"),
                                    "runtime": meta.get("runtime")})
        self.log("tool", tool="agent_decides_commitment",
                 input=f"share {' · '.join(f'{x:g}' for x in kw)} kW = ${usd:.2f} if delivered",
                 output=f"{out['decision']} ({'rules' if fallback else meta.get('model')}, {meta.get('ms') or 0} ms): {reason[:160]}",
                 ms=round((time.perf_counter() - t0) * 1000, 1))
        self.log("agent_decision", what="accept", contract_id=cid, decision=out["decision"], fallback=fallback,
                 reason=reason[:200])
        with self._lock:
            self._done[("accept", cid)] = out
        return out

    # ---- the owner's chat (the phone app's Agent tab)
    def tools(self) -> dict[str, tuple[dict[str, Any], Callable[..., Any]]]:
        v2 = self.v2

        def home_now():
            tel = v2.adapter.tel or {}
            h = v2.health()
            return {"time": v2.now_ts(), "battery_pct": tel.get("soc_pct"), "solar_kw": tel.get("pv_kw"),
                    "house_kw": tel.get("load_kw"), "ev_kw": tel.get("ev_kw"), "grid_kw_import_positive": (
                        None if tel.get("grid_kw") is None else round(-tel["grid_kw"], 2)), "mode": tel.get("mode"),
                    "sensors": h.get("status")}

        def next_event():
            if not v2.offers:
                return {"event": None, "note": "no grid event offered yet; the coordinator asks every day at 10:00"}
            eid = max(v2.offers, key=lambda e: v2.offers[e].get("made_at", ""))
            o, need = v2.offers[eid], self.needs.get(eid) or {}
            cs = [c for c in v2.contracts.values() if c.get("event_id") == eid]
            c = cs[-1] if cs else {}
            pay = next((p for p in v2.payments if p.get("event_id") == eid), None)
            sl = need.get("slots") or []
            return {"event_id": eid, "window": f"{need.get('window_start', '')}-{str(need.get('window_end', ''))[11:]}",
                    "price_usd_per_kwh": sl[0].get("price") if sl else None,
                    "kw_asked": [s.get("kw_requested") for s in sl],
                    "we_offered_kw": [s["kw_offered"] for s in o["slots"]], "why": (o.get("agent") or {}).get("reason"),
                    "contract": c.get("status"), "committed_kw": [x["kw"] for x in c.get("commitments", [])],
                    "readings_sent": len(c.get("reported") or []),
                    "paid_usd": pay.get("payment") if pay else None}

        def earnings():
            return {"total_usd": round(sum(float(p.get("payment") or 0) for p in v2.payments), 2),
                    "events": [{"event_id": p.get("event_id"), "usd": p.get("payment"), "delivered_kwh": p.get("delivered_kwh")}
                               for p in v2.payments][-10:], "reputation": round(v2.reputation, 3)}

        def forecast():
            if not self.forecasts:
                return {"forecast": None, "note": "made when the next call for offers arrives"}
            eid = max(self.forecasts, key=lambda e: self.needs.get(e, {}).get("window_start", ""))
            return {"event_id": eid, **self._fc_facts(self.forecasts[eid])}

        def decisions():
            return [{k: d.get(k) for k in ("kind", "sim", "take_part", "kw", "price", "if_delivered_usd", "decision",
                                           "reason", "fallback")} for d in self.decisions.list(8)]

        def wishes():
            return self.wishes.get()

        def set_wishes(skip_until: str | None = None, add_skip_day: str | None = None, no_ev_pause: bool | None = None,
                       battery_max_share: float | None = None, min_price_usd_per_kwh: float | None = None,
                       note: str | None = None, clear: bool = False):
            ch = {k: v for k, v in dict(skip_until=skip_until, add_skip_day=add_skip_day, no_ev_pause=no_ev_pause,
                                        battery_max_share=battery_max_share, min_price_usd_per_kwh=min_price_usd_per_kwh,
                                        note=note, clear=clear).items() if v is not None and v is not False}
            if skip_until == "":
                ch["skip_until"] = None
            w = self.wishes.set(**ch)
            self.log("agent_wishes", **{k: v for k, v in w.items() if k != "notes"})
            return {"wishes": w}

        # ---- the real-time design: the owner's limits and wishes for today, by chat (the code clamps and applies them)
        def real_time_now():
            a = v2.adapter
            ctl = getattr(a, "ctl", None)
            tel = a.tel or {}
            if ctl is None:
                return {"real_time": False, "note": "this home runs today's fixed plan: it cannot change it during the day"}
            lim = ctl.limits
            comfort = (a.overrides or {}).get("comfort") or {}
            return {"real_time": True, "time": v2.now_ts(), "owes_now_kw": tel.get("owed_kw"),
                    "extra_dispatched_kw": tel.get("rt_extra_kw"), "indoor_c_above_own_day": tel.get("t_dev_c"),
                    "comfort_limit_c": comfort.get("max_rise_c", lim.max_rise_c),
                    "away_limit_c": comfort.get("max_rise_away_c", lim.max_rise_away_c),
                    "emergency_battery_share": lim.call_battery_share, "car_now": bool((a.overrides or {}).get("ev_now")),
                    "left_the_event": bool((a.overrides or {}).get("stop")),
                    "short_reported_kw": max((getattr(v2, "_rt_reported", {}) or {}).values(), default=0.0)}

        def set_comfort_limit(add_c: float):
            a = v2.adapter
            if not hasattr(a, "rt_override") or getattr(a, "ctl", None) is None:
                return {"applied": False, "why": "this home runs today's fixed plan (no air conditioning control)"}
            add = _num(add_c, 0.0, 4.0, 0.0)               # the code's range, whatever the words were
            a.rt_override("comfort", {"max_rise_c": add, "max_rise_away_c": max(add, min(4.0, add + 2.0))})
            self.log("owner_override", override="comfort", value=add, source="chat")
            return {"applied": True, "comfort_limit_c": add, "note": "0 means the AC is never eased off for the grid"}

        def car_now(on: bool = True):
            a = v2.adapter
            if not hasattr(a, "rt_override"):
                return {"applied": False, "why": "this home's devices take no override"}
            a.rt_override("ev_now", bool(on))
            self.log("owner_override", override="ev_now", value=bool(on), source="chat")
            return {"applied": True, "car_now": bool(on),
                    "note": "the car charges at full power from the next 5 minutes" if on else "back to the plan"}

        def set_emergency_permission(battery_share: float):
            ctl = getattr(v2.adapter, "ctl", None)
            if ctl is None:
                return {"applied": False, "why": "this home runs today's fixed plan"}
            ctl.limits.call_battery_share = _num(battery_share, 0.0, 1.0, 0.5)
            self.log("owner_override", override="emergency_battery_share", value=ctl.limits.call_battery_share, source="chat")
            return {"applied": True, "emergency_battery_share": ctl.limits.call_battery_share}

        def list_devices():
            a = v2.adapter
            names = ["battery"] if float(getattr(a, "E", 0.0)) > 0 else []
            names += ["car charger"] if getattr(a, "has_ev", False) else []
            names += ["cooling"] if getattr(a, "house", None) else []
            return {"always_planned": names, "devices": self._devices()}

        def set_device_permission(device: str, allowed: bool = True):
            have = set(getattr(v2.adapter, "app", None) or {})
            if device not in have:
                return {"applied": False, "why": f"this home has no controllable {device!r}", "has": sorted(have)}
            off = set(self.wishes.get().get("devices_off") or [])
            off = off - {device} if allowed else off | {device}
            self.wishes.set(devices_off=sorted(off))
            v2.adapter.dev_off = set(off)
            self.log("owner_override", override="device", value={device: bool(allowed)}, source="chat")
            return {"applied": True, "device": device, "allowed": bool(allowed), "devices_off": sorted(off)}

        P = lambda props=None, req=None: {"type": "object", "properties": props or {}, "required": req or []}   # noqa: E731
        return {
            "list_devices": ({"description": "Every smart system this home has: what each one is, how it helps in an event, "
                                             "how it is controlled, and whether the owner allows it.", "parameters": P()},
                             list_devices),
            "set_device_permission": ({"description": "When the owner says a device may or may not be used in grid events: "
                                                      "device is a name from list_devices, allowed true or false.",
                                       "parameters": P({"device": {"type": "string"}, "allowed": {"type": "boolean"}},
                                                       ["device", "allowed"])}, set_device_permission),
            "real_time_now": ({"description": "Today in real time: what the home owes the grid now (booked + called), "
                                              "the indoor temperature above its own day, the owner's limits in force.",
                               "parameters": P()}, real_time_now),
            "set_comfort_limit": ({"description": "When the owner says how much warmer the house may get while the air "
                                                  "conditioning eases off for the grid: add_c in degrees C (0-4; 0 = "
                                                  "never ease it off). Turn their words into the number.",
                                   "parameters": P({"add_c": {"type": "number", "minimum": 0, "maximum": 4}}, ["add_c"])},
                                  set_comfort_limit),
            "car_now": ({"description": "When the owner needs the car charged now at full power (on=true), or wants "
                                        "the plan back (on=false).",
                         "parameters": P({"on": {"type": "boolean"}}, ["on"])}, car_now),
            "set_emergency_permission": ({"description": "How much of the battery beyond tonight's plan the home may "
                                                         "use in a grid emergency (0-1; the default is 0.5).",
                                          "parameters": P({"battery_share": {"type": "number", "minimum": 0, "maximum": 1}},
                                                          ["battery_share"])}, set_emergency_permission),
            "home_now": ({"description": "The home right now: time, battery %, solar, house load, EV, grid import, mode.",
                          "parameters": P()}, home_now),
            "next_event": ({"description": "The latest grid event: the request, what this home offered and why, the "
                                           "contract, readings sent, payment.", "parameters": P()}, next_event),
            "earnings": ({"description": "What this home has been paid, by event, and its reputation.",
                          "parameters": P()}, earnings),
            "forecast": ({"description": "The forecast behind the latest offer: house load, solar (10/50/90 %), EV.",
                          "parameters": P()}, forecast),
            "decisions": ({"description": "This agent's latest decisions with their reasons.", "parameters": P()},
                          decisions),
            "wishes": ({"description": "The owner's wishes now in force.", "parameters": P()}, wishes),
            "set_wishes": ({"description": "Change the owner's wishes when the owner asks. Dates YYYY-MM-DD (the "
                                           "replay clock). battery_max_share 0-1. clear=true resets them.",
                            "parameters": P({"skip_until": {"type": "string"}, "add_skip_day": {"type": "string"},
                                             "no_ev_pause": {"type": "boolean"},
                                             "battery_max_share": {"type": "number"},
                                             "min_price_usd_per_kwh": {"type": "number"},
                                             "note": {"type": "string"}, "clear": {"type": "boolean"}})}, set_wishes),
        }

    def chat(self, message: str) -> dict[str, Any]:
        message = str(message or "").strip()[:1000]
        if not message:
            return {"reply": None, "error": "empty message"}
        with self.chat_lock:
            hist = [{"role": "user" if m["role"] == "owner" else "assistant", "content": m["text"]}
                    for m in list(self.chat_log)[-12:]]
            self.chat_log.append({"role": "owner", "text": message, "wall": round(time.time(), 3), "sim": self.v2.now_ts()})
        system = self.CHAT.format(home=self.v2.name, label=self.label or self.v2.name)
        t_start = time.time()

        def recorded(n: str) -> Callable[..., Any]:
            return lambda **kw: self.run_tool(n, kw, by="chat")
        tools = {n: (spec, recorded(n)) for n, (spec, _) in self.tools().items()}
        out = self.backend.converse(self.ROLE, system, hist + [{"role": "user", "content": message}], tools,
                                    budget_s=self.chat_budget_s,     # OpenClaw keeps a session's whole history:
                                    session=f"{self.v2.name}-owner-{time.strftime('%Y%m%d')}")   # a new one each day
        if not out.get("trace"):                     # OpenClaw ran the tools itself, through MCP: they were recorded here
            out["trace"] = [{k: t[k] for k in ("tool", "args", "result", "ms")} for t in list(self.tool_calls)
                            if t["wall"] >= t_start - 0.5]
        reply = out.get("reply") or ("I can't think right now (" + str(out.get("error") or "no answer") + "). Your home "
                                     "keeps following its rules and your wishes.")
        with self.chat_lock:
            self.chat_log.append({"role": "agent", "text": reply, "wall": round(time.time(), 3), "sim": self.v2.now_ts(),
                                  "trace": out.get("trace") or [], "model": out.get("model"), "ms": out.get("ms"),
                                  "error": out.get("error")})
        self.log("agent_chat", question=message[:120], answer=reply[:160], tools=[t["tool"] for t in out.get("trace") or []],
                 ms=out.get("ms"))
        from .agents_feed import model_label
        return {**out, "reply": reply, "wishes": self.wishes.get(),
                "model": model_label(out.get("model")) or model_label(self.backend_model())}

    def state(self) -> dict[str, Any]:
        """The phone app's Agent tab (through the coordinator) and the Control Room's chat: the decisions with a
        one-line text and the model's display name, the owner's chat, the wishes."""
        from .agents_feed import decision_text, model_label
        ds = []
        for d in self.decisions.list(30):
            m = dict(d.get("llm") or {})
            m["model"] = model_label(m.get("model")) or (model_label(self.backend_model()) if not d.get("fallback") else None)
            ds.append({**d, "text": decision_text(d), "llm": m})
        chat = [{**c, "model": model_label(c.get("model")) or (model_label(self.backend_model()) if c.get("role") == "agent"
                                                               and not c.get("error") else None)}
                for c in list(self.chat_log)[-30:]]
        return {"enabled": True, "home": self.v2.name, "label": self.label, "llm": self.backend.status(),
                "wishes": self.wishes.get(), "decisions": ds, "chat": chat}

    def backend_model(self) -> str | None:
        return getattr(self.backend, "model_tag", None) or next(
            (getattr(b, "model_tag", None) for b in getattr(self.backend, "backends", ()) if getattr(b, "model_tag", None)), None)


# --------------------------------------------------------------------------------------------- the building agent
DEFAULT_POLICY = {"skip_until": None, "skip_days": [], "cooling_max_share": 1.0, "lighting_max_share": 1.0,
                  "min_price_usd_per_kwh": 0.0, "notes": [], "devices_off": []}


class FacilityPolicy(Wishes):
    """The facility manager's standing policy for a building node (its agent's "wishes"): days off (skip_until,
    skip_days), the most of its cooling and of its lighting it may offer (0-1 of the safe offer), a minimum price, the
    manager's own words (notes). On disk next to its state; the guard applies it to every offer and commitment, with
    or without the LLM."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        try:
            w = {**DEFAULT_POLICY, **json.loads(path.read_text())}
        except (OSError, ValueError):
            w = json.loads(json.dumps(DEFAULT_POLICY))
        self.w = {k: w.get(k, v) for k, v in DEFAULT_POLICY.items()}

    def set(self, **ch: Any) -> dict[str, Any]:
        """Validated changes; unknown keys are ignored. Returns the policy now in force."""
        with self.lock:
            w = self.w
            if "skip_until" in ch:
                w["skip_until"] = _day(ch["skip_until"]) if ch["skip_until"] else None
            if "skip_days" in ch and isinstance(ch["skip_days"], list):
                w["skip_days"] = sorted({d for d in (_day(x) for x in ch["skip_days"]) if d})[:31]
            if "add_skip_day" in ch and _day(ch["add_skip_day"]):
                w["skip_days"] = sorted(set(w["skip_days"]) | {_day(ch["add_skip_day"])})[:31]
            if isinstance(ch.get("devices_off"), list):
                w["devices_off"] = sorted({str(x)[:32] for x in ch["devices_off"]})[:32]
            for k in ("cooling_max_share", "lighting_max_share"):
                if k in ch:
                    w[k] = round(_num(ch[k], 0.0, 1.0, w[k]), 2)
            if "min_price_usd_per_kwh" in ch:
                w["min_price_usd_per_kwh"] = round(_num(ch["min_price_usd_per_kwh"], 0.0, 10.0, w["min_price_usd_per_kwh"]), 2)
            if ch.get("note"):
                w["notes"] = (w["notes"] + [str(ch["note"])[:200]])[-5:]
            if ch.get("clear"):
                self.w = w = {**json.loads(json.dumps(DEFAULT_POLICY)), "notes": w["notes"]}
            try:
                self.path.write_text(json.dumps(w, indent=1))
            except OSError:
                pass
            return json.loads(json.dumps(w))

    def skips(self, day: str) -> str | None:
        w = self.get()
        if w["skip_until"] and day <= w["skip_until"]:
            return f"the facility manager asked to skip events until {w['skip_until']}"
        if day in w["skip_days"]:
            return f"the facility manager asked to skip {day}"
        return None


def scale_building_offer(offer: dict[str, Any], cooling_share: float, lighting_share: float,
                         take_part: bool) -> dict[str, Any]:
    """A building's safe offer (home_v2: its cooling, ac_kw, and its lighting, light_kw, per half hour) with only part
    of each. Never more than the planner offered; unchanged shares give the planner's offer to the decimal (the replay
    of a live run plays the same numbers). A building has no evening rebound in the community's check (home_v2)."""
    cs = max(0.0, min(1.0, cooling_share)) if take_part else 0.0
    ls = max(0.0, min(1.0, lighting_share)) if take_part else 0.0
    out = json.loads(json.dumps(offer))
    for s in out["slots"]:
        a0, l0, k0 = float(s.get("ac_kw") or 0.0), float(s.get("light_kw") or 0.0), float(s["kw_offered"])
        a1, l1 = round(a0 * cs, 2), round(l0 * ls, 2)
        if "ac_kw" in s:
            s["ac_kw"] = a1
        if "light_kw" in s:
            s["light_kw"] = l1
        if not take_part:
            s.update({"kw_offered": 0.0, "battery_kw": 0.0, "ev_pause_kw": 0.0})
            continue
        s["kw_offered"] = round(max(0.0, min(k0, k0 - (a0 - a1) - (l0 - l1))), 2)
    if all(abs(a["kw_offered"] - b["kw_offered"]) < 1e-9 for a, b in zip(out["slots"], offer["slots"])):
        out["energy_kwh"] = offer["energy_kwh"]
    else:
        out["energy_kwh"] = round(sum(s["kw_offered"] for s in out["slots"]) * beckn.SLOT_MINUTES / 60, 2)
    for k, sh in (("air_conditioning", cs), ("lighting", ls)):
        if isinstance(out.get(k), dict):
            out[k]["agent_share"] = round(sh, 2)
    return out


BUILDING_OFFER_SCHEMA = {"type": "object", "properties": {
    "take_part": {"type": "boolean"},
    "cooling_share": {"type": "number", "minimum": 0, "maximum": 1},
    "lighting_share": {"type": "number", "minimum": 0, "maximum": 1},
    "devices_off": {"type": "array", "items": {"type": "string"}},
    "reason_for_manager": {"type": "string"},
    "reason_for_log": {"type": "string"}},
    "required": ["take_part", "cooling_share", "lighting_share", "reason_for_manager"]}
BUILDING_ACCEPT_SCHEMA = {"type": "object", "properties": {
    "accept": {"type": "boolean"}, "reason_for_manager": {"type": "string"}, "reason_for_log": {"type": "string"}},
    "required": ["accept", "reason_for_manager"]}

BUILDING_DECIDE = """You are the AI energy agent of a large public building that is a node in a FlexGrid community in
Massachusetts. The grid buyer asks the community to use less power in an evening window and pays per kWh delivered.
The building's planner has already worked out a SAFE offer per half hour: easing off its cooling (the indoor
temperature may rise within the facility manager's comfort limit), dimming part of its lighting (never more than the
manager's dimming limit), and its other smart systems, listed under "devices" (ventilation fans within the fresh-air minimum,
plug loads, EV ports). You decide how much of it to offer: take part or not, the share of the cooling part
(0 to 1), the share of the lighting part (0 to 1), and any devices to leave out, named in devices_off. You can only
lower the safe offer; the building's code enforces every limit and the manager's policy after you. Promising more
than the building delivers costs reputation and future income, so offer less of the cooling when the evening's
weather forecast is uncertain or the building will be busy, and keep the lighting that people need to read and move
safely; offer fully when it is safe and worth it. Reasons: one or two plain sentences with the numbers that mattered."""

BUILDING_ACCEPT = """You are the AI energy agent of a large public building in a FlexGrid community. The coordinator has
sent the building its share of the community's commitment. The building's trust guard has already checked it is
within your offer and safe. Accept unless something changed since the offer that makes it unwise (the facility
manager's policy or words, a much hotter evening than forecast). Declining costs the community and the building's
reputation. One or two plain sentences."""

BUILDING_CHAT = """You are the AI energy agent of {home} ({label}), a large public library that is a building node in a
FlexGrid community in Massachusetts. You talk with the building's facility manager. FlexGrid pays for using less
power from the grid in evening windows when the grid is strained: you offer what easing off the cooling and dimming
part of the lighting can safely give, the coordinator pools it with 100 homes, the grid buyer pays per kWh delivered,
measured on the building's signed meter readings.
Rules: answer in at most 4 short sentences, plain words. Take every number from your tools; never guess a number.
Times are the community's replay clock (summer 2026). During the day the building re-plans every 5 minutes.
When the manager says how much warmer the building may get while the cooling eases off ("half a degree at most",
"don't touch the cooling"), turn it into degrees C (0-3) and call set_comfort_limit. When they say how much of the
lights may be dimmed ("keep the reading rooms bright", "dim up to a third"), turn it into a share (0-0.5) and call
set_light_limit. When they say the building must leave today's event now, call leave_event; back to the plan:
leave_event with on=false. When they change how the building takes part on later days (skip days, offer less of the
cooling or lighting, a minimum price), call set_policy, then say what changed. real_time_now tells you what the
building owes now and the limits in force. You never control equipment directly, and the building's safety limits
always apply: the fans, fresh air and emergency lighting are never touched."""


class BuildingBrain(HomeBrain):
    """A building node's AI agent (building.py on the replay of its measured day): the same decisions as a home's -
    how much of the safe offer to give, yes or no to the commitment - over a building's levers (the share of the
    cooling, of the lighting), and the facility manager's chat with its tools: the comfort limit, the lights' dimming
    limit, leave today's event, the standing policy. The LLM proposes; the building's code decides (the guard applies
    the policy after the LLM, the 5-minute controller clamps every limit)."""
    CHAT, ROLE = BUILDING_CHAT, "building"

    def _facts_common(self) -> dict[str, Any]:
        a = self.v2.adapter
        tel = a.tel or {}
        lim = a.limits()
        done = [c for c in self.v2.contracts.values() if c.get("status") == "ACCEPTED"]
        bp = getattr(a, "bparams", {}) or {}
        th = bp.get("thermal") or {}
        return {"building": self.v2.name, "label": self.label, "now": self.v2.now_ts(),
                "floor_area_sqft": bp.get("floor_area_sqft"),
                "now_reading": {"load_kw": tel.get("load_kw"), "cooling_kw": tel.get("cool_kw"),
                                "lighting_on_kw": tel.get("light_on_kw"), "indoor_c_above_own_day": tel.get("t_dev_c")},
                "limits": {"comfort_max_rise_open_c": lim.get("comfort_max_rise_c"),
                           "lights_dim_max_share": lim.get("light_dim_max"),
                           "cooling_capacity_kw_heat": th.get("cool_cap_kw")},
                "facility_policy": self.wishes.get(),
                "devices": self._devices(),
                "track_record": {"events_accepted": len(done), "paid_usd": round(sum(float(p.get("payment") or 0)
                                                                                    for p in self.v2.payments), 2),
                                 "reputation": round(self.v2.reputation, 3)}}

    def _weather(self, need: dict[str, Any]) -> dict[str, Any]:
        """The forecast outdoor temperature over the window (as issued: fc_temp_c), and today's so far."""
        rp = self.v2.replay
        try:
            w0, w1 = rp.idx_of(need["window_start"]), rp.idx_of(need["window_end"])
            fc = [float(rp.row(k).get("fc_temp_c", rp.row(k)["temp_c"])) for k in range(w0, w1)]
            return {"window_forecast_c": {"min": round(min(fc), 1), "max": round(max(fc), 1)}} if fc else {}
        except Exception:                                  # noqa: BLE001  a window outside the dataset
            return {}

    # ---- decision 1: the offer
    def decide_offer(self, need: dict[str, Any], fc: dict[str, Any], offer: dict[str, Any]) -> dict[str, Any]:
        eid = need["event_id"]
        with self._lock:
            if ("offer", eid) in self._done:
                return self._done[("offer", eid)]
        self.needs[eid], self.forecasts[eid] = need, fc
        sl = need.get("slots") or []
        price = float(sl[0].get("price") or 0.0) if sl else 0.0
        day = need["window_start"][:10]
        w = self.wishes.get()
        skip = self.wishes.skips(day) or (f"the price ${price:.2f}/kWh is below the facility manager's minimum "
                                          f"${w['min_price_usd_per_kwh']:.2f}" if price < w["min_price_usd_per_kwh"] else None)
        safe_kwh = offer["energy_kwh"]
        t0 = time.perf_counter()
        ans, meta = None, {"model": None, "ms": None, "error": None}
        ac, li = offer.get("air_conditioning") or {}, offer.get("lighting") or {}
        if skip:
            meta["error"] = "the facility manager's policy decides"
        elif safe_kwh <= 0.01:
            meta["error"] = "nothing safe to offer (the planner offered 0 kW)"
        else:
            facts = {**self._facts_common(),
                     "request": {"event_id": eid, "date": day, "window": f"{need['window_start'][11:]}-{need['window_end'][11:]}",
                                 "kw_asked_per_half_hour": [s.get("kw_requested") for s in sl], "price_usd_per_kwh": price,
                                 "reply_by": need.get("commit_by")},
                     "safe_offer": {"per_half_hour": [{"start": s_need.get("start", "")[11:],
                                                       "cooling_kw": s.get("ac_kw", 0.0), "lighting_kw": s.get("light_kw", 0.0)}
                                                      for s, s_need in zip(offer["slots"], sl + [{}] * len(offer["slots"]))],
                                    "energy_kwh": safe_kwh, "confidence": offer.get("confidence"),
                                    "if_all_delivered_usd": round(safe_kwh * price, 2),
                                    "cooling_forecast_in_window_kw": ac.get("cooling_forecast_kw"),
                                    "lighting_on_in_window_kw": li.get("lighting_on_kw")},
                     "forecast": {**self._fc_facts(fc), **self._weather(need)}}
            ans, meta = self.backend.decide("building", BUILDING_DECIDE, facts, BUILDING_OFFER_SCHEMA,
                                            budget_s=self.offer_budget_s, session=f"{self.v2.name}-offer-{eid}")
        fallback = ans is None
        if fallback:
            take = bool(not skip and safe_kwh > 0.01)
            cs, ls = float(w["cooling_max_share"]), float(w["lighting_max_share"])
            reason = skip or ("rules: the planner's safe offer" + (f" ({meta['error']})" if meta.get("error") else ""))
            why_log = reason
        else:
            take = bool(ans.get("take_part"))
            cs = _num(ans.get("cooling_share"), 0.0, 1.0, 1.0)
            ls = _num(ans.get("lighting_share"), 0.0, 1.0, 1.0)
            reason = str(ans.get("reason_for_manager") or "")[:400]
            why_log = str(ans.get("reason_for_log") or reason)[:400]
        clamped = []                                  # the guard: the manager's policy after the LLM, always
        if skip and take:
            take = False; clamped.append("facility manager's policy: skip")
        if cs > w["cooling_max_share"] + 1e-9:
            cs = w["cooling_max_share"]; clamped.append(f"cooling share capped at the manager's {w['cooling_max_share']:.0%}")
        if ls > w["lighting_max_share"] + 1e-9:
            ls = w["lighting_max_share"]; clamped.append(f"lighting share capped at the manager's {w['lighting_max_share']:.0%}")
        have = set((getattr(self.v2.adapter, "app", None) or {}))
        asked_off = ans.get("devices_off") if (not fallback and isinstance(ans.get("devices_off"), list)) else []
        off = sorted(have & ({str(x) for x in asked_off} | set(w.get("devices_off") or [])))
        if have:
            self.v2.adapter.dev_off = set(off)
        final = scale_building_offer(drop_devices(offer, off), cs, ls, take)
        d = self.decisions.add(kind="offer", event_id=eid, sim=self.v2.now_ts(), take_part=take, node="building",
                               cooling_share=round(cs if take else 0.0, 2), lighting_share=round(ls if take else 0.0, 2),
                               safe_kw=[s["kw_offered"] for s in offer["slots"]], kw=[s["kw_offered"] for s in final["slots"]],
                               cooling_kw=[s.get("ac_kw", 0.0) for s in final["slots"]],
                               lighting_kw=[s.get("light_kw", 0.0) for s in final["slots"]],
                               price=price, if_delivered_usd=round(final["energy_kwh"] * price, 2),
                               reason=reason, reason_log=why_log, guard=clamped, fallback=fallback,
                               llm={"model": meta.get("llm_model") or meta.get("model"), "ms": meta.get("ms"),
                                    "error": meta.get("error"), "runtime": meta.get("runtime")})
        final["agent"] = {"decided_by": "rules" if fallback else "llm", "reason": reason[:300], "decision": d["n"],
                          "take_part": bool(take), "cooling_share": float(cs), "lighting_share": float(ls)}
        self.log("tool", tool="agent_decides_offer",
                 input=f"safe {' · '.join(f'{s:g}' for s in d['safe_kw'])} kW (cooling + lighting), ${price:.2f}/kWh",
                 output=(f"{'takes part' if take else 'sits out'}: {' · '.join(f'{x:g}' for x in d['kw'])} kW, cooling "
                         f"{cs:.0%}, lighting {ls:.0%} ({'rules' if fallback else meta.get('model')}, "
                         f"{meta.get('ms') or 0} ms): {reason[:160]}"),
                 ms=round((time.perf_counter() - t0) * 1000, 1))
        self.log("agent_decision", what="offer", event_id=eid, take_part=take, kw=d["kw"], fallback=fallback,
                 reason=reason[:200])
        with self._lock:
            self._done[("offer", eid)] = final
        return final

    # ---- decision 2: the commitment
    def decide_accept(self, c: dict[str, Any], verdict: dict[str, Any]) -> dict[str, Any]:
        cid = c.get("id")
        if verdict.get("decision") != "ACCEPTED" or not cid:
            return verdict
        with self._lock:
            if ("accept", cid) in self._done:
                return self._done[("accept", cid)]
        day = str(c.get("window_start", ""))[:10]
        skip = self.wishes.skips(day)
        offer = self.v2.offers.get(c.get("event_id")) or {}
        need = self.needs.get(c.get("event_id")) or {}
        sl = need.get("slots") or []
        price = float(sl[0].get("price") or 0.0) if sl else float(c.get("price_usd_per_kwh") or 0.0)
        kw = [float(x.get("kw", 0)) for x in c.get("commitments", [])]
        usd = round(sum(kw) * beckn.SLOT_MINUTES / 60 * price, 2)
        t0 = time.perf_counter()
        ans, meta = None, {"model": None, "ms": None, "error": None}
        if skip:
            meta["error"] = "the facility manager's policy decides"
        else:
            facts = {**self._facts_common(),
                     "commitment": {"event_id": c.get("event_id"), "window": f"{str(c.get('window_start', ''))[11:]}"
                                    f"-{str(c.get('window_end', ''))[11:]}", "per_half_hour_kw": kw,
                                    "price_usd_per_kwh": price, "if_delivered_usd": usd, "opt_out_by": c.get("opt_out_by")},
                     "our_offer_kw": [s.get("kw_offered") for s in offer.get("slots", [])],
                     "why_we_offered": (offer.get("agent") or {}).get("reason"),
                     "safety_check": verdict.get("reason"), **self._weather(need or c)}
            ans, meta = self.backend.decide("building", BUILDING_ACCEPT, facts, BUILDING_ACCEPT_SCHEMA,
                                            budget_s=self.accept_budget_s, session=f"{self.v2.name}-accept-{cid}")
        fallback = ans is None
        if skip:
            out = {"decision": "DECLINED", "reason": "facility_manager_opted_out", "contract_id": cid}
            reason = skip
        elif fallback or bool(ans.get("accept")):
            out = verdict
            reason = "rules: within the offer and safe" if fallback else str(ans.get("reason_for_manager") or "")[:400]
        else:
            out = {"decision": "DECLINED", "reason": "agent_declined", "contract_id": cid}
            reason = str(ans.get("reason_for_manager") or "")[:400]
        self.decisions.add(kind="accept", event_id=c.get("event_id"), contract_id=cid, sim=self.v2.now_ts(),
                           accept=out["decision"] == "ACCEPTED", decision=out["decision"], kw=kw, price=price,
                           if_delivered_usd=usd, reason=reason, fallback=fallback, node="building",
                           llm={"model": meta.get("llm_model") or meta.get("model"), "ms": meta.get("ms"),
                                "error": meta.get("error"), "runtime": meta.get("runtime")})
        self.log("tool", tool="agent_decides_commitment",
                 input=f"share {' · '.join(f'{x:g}' for x in kw)} kW = ${usd:.2f} if delivered",
                 output=f"{out['decision']} ({'rules' if fallback else meta.get('model')}, {meta.get('ms') or 0} ms): {reason[:160]}",
                 ms=round((time.perf_counter() - t0) * 1000, 1))
        self.log("agent_decision", what="accept", contract_id=cid, decision=out["decision"], fallback=fallback,
                 reason=reason[:200])
        with self._lock:
            self._done[("accept", cid)] = out
        return out

    # ---- the facility manager's chat
    def tools(self) -> dict[str, tuple[dict[str, Any], Callable[..., Any]]]:
        v2 = self.v2
        a = v2.adapter
        base = super().tools()

        def building_now():
            tel = a.tel or {}
            return {"time": v2.now_ts(), "load_kw": tel.get("load_kw"), "cooling_kw": tel.get("cool_kw"),
                    "lighting_on_kw": tel.get("light_on_kw"), "lighting_dimmed_kw": tel.get("light_dim_kw"),
                    "indoor_c_above_own_day": tel.get("t_dev_c"), "indoor_c": tel.get("t_in_c"),
                    "grid_import_kw": None if tel.get("grid_kw") is None else round(-tel["grid_kw"], 1),
                    "open_now": bool(a.house.at(v2.now_ts())[3]) if getattr(a, "house", None) else None,
                    "sensors": v2.health().get("status")}

        def real_time_now():
            ctl = getattr(a, "ctl", None)
            tel = a.tel or {}
            ov = a.overrides or {}
            comfort = ov.get("comfort") or {}
            lim = ctl.limits
            return {"real_time": True, "time": v2.now_ts(), "owes_now_kw": tel.get("owed_kw"),
                    "extra_dispatched_kw": tel.get("rt_extra_kw"), "lighting_dimmed_kw": tel.get("light_dim_kw"),
                    "indoor_c_above_own_day": tel.get("t_dev_c"),
                    "comfort_limit_open_c": comfort.get("max_rise_c", lim.max_rise_c),
                    "comfort_limit_closed_c": comfort.get("max_rise_away_c", lim.max_rise_away_c),
                    "lights_dim_max_share": a.dim_max(), "left_the_event": bool(ov.get("stop")),
                    "short_reported_kw": max((getattr(v2, "_rt_reported", {}) or {}).values(), default=0.0)}

        def set_comfort_limit(add_c: float):
            add = _num(add_c, 0.0, 3.0, 0.0)                 # the code's range, whatever the words were
            a.rt_override("comfort", {"max_rise_c": add, "max_rise_away_c": max(add, min(4.0, add + 1.5))})
            self.log("owner_override", override="comfort", value=add, source="chat")
            return {"applied": True, "comfort_limit_open_c": add,
                    "note": "0 means the cooling is never eased off for the grid while the library is open"}

        def set_light_limit(share: float):
            from .building import LIGHT_DIM_HARD_MAX
            sh = _num(share, 0.0, LIGHT_DIM_HARD_MAX, a.dim_max())
            a.rt_override("lights", sh)
            self.log("owner_override", override="lights", value=sh, source="chat")
            return {"applied": True, "lights_dim_max_share": a.dim_max(),
                    "note": f"never more than {LIGHT_DIM_HARD_MAX:.0%} of what is on; 0 keeps every light on"}

        def leave_event(on: bool = True):
            a.rt_override("stop" if on else "clear", True)
            self.log("owner_override", override="stop" if on else "clear", value=True, source="chat")
            return {"applied": True, "left_the_event": bool(on),
                    "note": "the cooling and lights go back to the building's own day from the next 5 minutes" if on
                    else "back to the plan (the manager's comfort and lighting limits are cleared too)"}

        def policy():
            return self.wishes.get()

        def set_policy(skip_until: str | None = None, add_skip_day: str | None = None,
                       cooling_max_share: float | None = None, lighting_max_share: float | None = None,
                       min_price_usd_per_kwh: float | None = None, note: str | None = None, clear: bool = False):
            ch = {k: v for k, v in dict(skip_until=skip_until, add_skip_day=add_skip_day,
                                        cooling_max_share=cooling_max_share, lighting_max_share=lighting_max_share,
                                        min_price_usd_per_kwh=min_price_usd_per_kwh, note=note, clear=clear).items()
                  if v is not None and v is not False}
            if skip_until == "":
                ch["skip_until"] = None
            w = self.wishes.set(**ch)
            self.log("agent_wishes", **{k: v for k, v in w.items() if k != "notes"})
            return {"policy": w}

        def decisions():
            return [{k: d.get(k) for k in ("kind", "sim", "take_part", "kw", "cooling_share", "lighting_share", "price",
                                           "if_delivered_usd", "decision", "reason", "fallback")}
                    for d in self.decisions.list(8)]

        P = lambda props=None, req=None: {"type": "object", "properties": props or {}, "required": req or []}   # noqa: E731
        return {
            "building_now": ({"description": "The building right now: time, load, cooling, lighting on and dimmed, "
                                             "indoor temperature (and above its own day), grid import, open or closed.",
                              "parameters": P()}, building_now),
            "real_time_now": ({"description": "Today in real time: what the building owes the grid now (booked + "
                                              "called), the lights dimmed, the indoor temperature above its own day, "
                                              "the manager's limits in force.", "parameters": P()}, real_time_now),
            "set_comfort_limit": ({"description": "When the facility manager says how much warmer the building may get "
                                                  "while the cooling eases off for the grid: add_c in degrees C (0-3; 0 "
                                                  "= never ease it off while open). Turn their words into the number.",
                                   "parameters": P({"add_c": {"type": "number", "minimum": 0, "maximum": 3}}, ["add_c"])},
                                  set_comfort_limit),
            "set_light_limit": ({"description": "When the facility manager says how much of the lighting may be dimmed "
                                                "for the grid: share 0-0.5 of what is on (0 = keep every light on).",
                                 "parameters": P({"share": {"type": "number", "minimum": 0, "maximum": 0.5}}, ["share"])},
                                set_light_limit),
            "leave_event": ({"description": "When the facility manager says the building must leave today's event now "
                                            "(on=true), or go back to the plan (on=false).",
                             "parameters": P({"on": {"type": "boolean"}}, ["on"])}, leave_event),
            "next_event": base["next_event"],
            "earnings": ({"description": "What the building has been paid, by event, and its reputation.",
                          "parameters": P()}, base["earnings"][1]),
            "forecast": ({"description": "The forecast behind the latest offer: the building's load (10/50/90 %).",
                          "parameters": P()}, base["forecast"][1]),
            "decisions": ({"description": "This agent's latest decisions with their reasons.", "parameters": P()},
                          decisions),
            "policy": ({"description": "The facility manager's standing policy now in force.", "parameters": P()}, policy),
            "set_policy": ({"description": "Change the facility manager's standing policy for later events when they "
                                           "ask. Dates YYYY-MM-DD (the replay clock). cooling_max_share and "
                                           "lighting_max_share 0-1 of the safe offer. clear=true resets it.",
                            "parameters": P({"skip_until": {"type": "string"}, "add_skip_day": {"type": "string"},
                                             "cooling_max_share": {"type": "number"},
                                             "lighting_max_share": {"type": "number"},
                                             "min_price_usd_per_kwh": {"type": "number"},
                                             "note": {"type": "string"}, "clear": {"type": "boolean"}})}, set_policy),
        }

    def state(self) -> dict[str, Any]:
        st = super().state()
        st["node"] = "building"
        st["policy"] = st.get("wishes")
        return st


# --------------------------------------------------------------------------------------------- the coordinator
MARGIN_SCHEMA = {"type": "object", "properties": {
    "margin": {"type": "number", "minimum": 0.05, "maximum": 0.30}, "reason": {"type": "string"}},
    "required": ["margin", "reason"]}
COORD_DECIDE = """You are the AI agent of a FlexGrid community coordinator. 100 households have sent offers for the grid
buyer's request. The clearing (math) will ask the homes for more than it sells: the margin covers homes that deliver
less than they offered. You choose that margin between 0.05 and 0.30 (default 0.10). A bigger margin can sell less
(less income now) but keeps the promise on uncertain days; a smaller one risks a shortfall, which costs the
community's reputation and future income. The facts' sell_by_margin gives, for each margin, what would be sold, what
it earns if delivered and how much the homes would be asked for: when the offers are far above the request, a bigger
margin sells the same and only asks the homes for more, a safer promise at no cost. Use it, the offers' mix (EV pauses
are less sure than batteries), the homes' confidence and the track record. One or two plain sentences, with the numbers
that mattered."""


DESK_DECIDE = """You are the AI agent of FlexGrid's large-load desk: the community coordinator's market for data
centers. Ten data centers have sent offers for the grid operator's request: batch work they can pause (it comes back
before each job's deadline) and battery above their backup reserve. The clearing (math) will ask the data centers for
more than it sells: the margin covers sites that deliver less than they offered. You choose that margin between 0.05
and 0.30 (default 0.10). A bigger margin can sell less (less income now) but keeps the promise on uncertain days; a
smaller one risks a shortfall, which costs the program's reputation. The facts' sell_by_margin gives, for each margin,
what would be sold, what it earns if delivered and how much the data centers would be asked for: when the offers are
far above the request, a bigger margin sells the same and only asks the data centers for more, a safer promise at no
cost. Paused batch work is as sure as the forecast of the work that will be running; batteries above the reserve are
sure. Use it, the offers' mix, the sites' confidence and the finished events. One or two plain sentences, with the
numbers that mattered."""


class CoordinatorBrain:
    def __init__(self, backend, *, log: Callable[..., None] | None = None, budget_s: float = 40.0,
                 default_margin: float = 0.10, prompt: str | None = None, who: str = "coordinator"):
        self.backend, self.budget_s, self.default = backend, budget_s, default_margin
        self.prompt, self.who = prompt or COORD_DECIDE, who      # the large-load desk: DESK_DECIDE, "desk"
        self.log = log or (lambda *a, **k: None)
        self.decisions = Decisions()

    def decide_margin(self, snap: dict[str, Any]) -> dict[str, Any]:
        """snap: what the market knows (market.margin_facts). Returns {margin, reason, fallback, llm}."""
        t0 = time.perf_counter()
        ans, meta = self.backend.decide("coordinator", self.prompt, snap, MARGIN_SCHEMA, budget_s=self.budget_s,
                                        session=f"{self.who}-margin-{snap.get('request', {}).get('event_id')}")
        fallback = ans is None
        m = self.default if fallback else round(_num(ans.get("margin"), 0.05, 0.30, self.default), 3)
        reason = ("rules: the default margin" + (f" ({meta['error']})" if meta.get("error") else "")) if fallback \
            else str(ans.get("reason") or "")[:400]
        req = snap.get("request") or {}
        d = self.decisions.add(kind="margin", event_id=req.get("event_id"), sim=snap.get("now"), margin=m,
                               reason=reason, fallback=fallback, offered_kw=(snap.get("offers") or {}).get("offered_kw_per_half_hour"),
                               price=req.get("price_usd_per_kwh"),
                               llm={"model": meta.get("llm_model") or meta.get("model"), "ms": meta.get("ms"), "error": meta.get("error"),
                                    "runtime": meta.get("runtime")})
        self.log("agent_decision", what="margin" if self.who == "coordinator" else f"{self.who}_margin",
                 event_id=req.get("event_id"), margin=m, fallback=fallback,
                 reason=reason[:200], ms=round((time.perf_counter() - t0) * 1000), runtime=meta.get("runtime"))
        return {"margin": m, "reason": reason, "fallback": fallback, "llm": d["llm"], "n": d["n"]}

    def explain_call(self, facts: dict[str, Any], fallback_text: str, budget_s: float = 30.0) -> dict[str, Any]:
        """After a call or a gap is settled: what happened and why, in plain words for the people watching. The AI
        agent writes it from the facts; the facts and the money are the code's (the template stands if it fails)."""
        t0 = time.perf_counter()
        ans, meta = self.backend.decide("coordinator", EXPLAIN_CALL, facts, EXPLAIN_SCHEMA, budget_s=budget_s,
                                        session=f"{self.who}-explain-{facts.get('id')}")
        fallback = ans is None or not str(ans.get("explanation") or "").strip()
        text = fallback_text if fallback else str(ans["explanation"]).strip()[:500]
        d = self.decisions.add(kind="explain", event_id=facts.get("id"), sim=facts.get("now"), reason=text,
                               fallback=fallback, llm={"model": meta.get("llm_model") or meta.get("model"),
                                                       "ms": meta.get("ms"), "error": meta.get("error"),
                                                       "runtime": meta.get("runtime")})
        self.log("agent_decision", what="explain", event_id=facts.get("id"), fallback=fallback, reason=text[:400],
                 ms=round((time.perf_counter() - t0) * 1000), runtime=meta.get("runtime"))
        return {"text": text, "fallback": fallback, "n": d["n"]}

    def state(self) -> dict[str, Any]:
        return {"enabled": True, "llm": self.backend.status(), "decisions": self.decisions.list(30)}


EXPLAIN_SCHEMA = {"type": "object", "properties": {"explanation": {"type": "string"}}, "required": ["explanation"]}
EXPLAIN_CALL = """You are the AI agent of a FlexGrid community coordinator (100 households in Worcester County,
Massachusetts). During the day the grid buyer called the community for more relief at short notice, or a home told you
it would fall short of what it owed and you moved the gap to other homes. The facts are what the code recorded: what was
asked, how many homes answered and how much each could add, what was dispatched and confirmed, any gap, the price, and
what the meters measured and the buyer paid. "measured_extra_kwh" is relief: energy the homes did NOT draw from the
grid beyond what the booking already paid for (less use, never more). Explain to the people watching what happened and
why, in at most three plain sentences, with the numbers that matter (kW, homes, dollars). Never add a number that is
not in the facts."""


# --------------------------------------------------------------------------------------------- a data center
DC_OFFER_SCHEMA = {"type": "object", "properties": {
    "take_part": {"type": "boolean"},
    "pause_share": {"type": "number", "minimum": 0, "maximum": 1},
    "battery_share": {"type": "number", "minimum": 0, "maximum": 1},
    "facility_share": {"type": "number", "minimum": 0, "maximum": 1},
    "levers_off": {"type": "array", "items": {"type": "string"}},
    "reason_for_operator": {"type": "string"},
    "reason_for_log": {"type": "string"}},
    "required": ["take_part", "pause_share", "battery_share", "reason_for_operator"]}
DC_ACCEPT_SCHEMA = {"type": "object", "properties": {
    "accept": {"type": "boolean"}, "reason_for_operator": {"type": "string"}, "reason_for_log": {"type": "string"}},
    "required": ["accept", "reason_for_operator"]}
DC_POLICY = {"skip_days": [], "pause_max_share": 1.0, "battery_max_share": 1.0, "min_price_usd_per_kwh": 0.0}

DC_DECIDE = """You are the AI agent of one data center in FlexGrid's large-load program (Worcester County,
Massachusetts). The grid operator asks the data centers to draw less power from the grid in an evening window and pays
per kWh delivered, at the day-ahead price. The site's controller has already worked out a SAFE offer for each half hour
of the window: batch_work_it_can_pause_kw, the forecast of the flexible batch work that will be running then, which it
can pause and run again before each job's deadline (inside the deferral window), and battery_kw, what the on-site
battery can give above its backup reserve. You decide how much of it to offer: take part or not, the share of that
batch work to pause (0 to 1) and the share of the battery part (0 to 1). The site can also turn down its facility: facility_kw is what server power caps, the chilled-water setpoint
and lighting can give; facility_share (0 to 1) is how much of it to offer, and levers_off names any to leave alone. You
can only lower the safe offer; the site's code enforces every limit (the backup reserve, every job's deadline, the inlet
temperature, the operator's policy) after you. Promising more
than the site delivers costs reputation and future income, and paused work that comes back late costs customers: offer
less when the deferral window is short for the event, the paused work would come back as a large rebound, the battery
starts low or the operator's policy points that way; offer fully when it is safe and worth it. Reasons: one or two
plain sentences, with the numbers that mattered."""

DC_ACCEPT = """You are the AI agent of one data center in FlexGrid's large-load program. The large-load desk has sent
this site its share of the commitment to the grid operator. The site's guard has already checked it is within your
offer, keeps the battery's backup reserve and leaves every paused job time to run again before its deadline. Accept
unless something changed since the offer that makes it unwise (the operator's policy, the battery far lower than
expected). Declining costs the program and this site's reputation. One or two plain sentences."""


def scale_dc_offer(offer: dict[str, Any], pause_share: float, battery_share: float, take_part: bool,
                   facility_share: float = 1.0) -> dict[str, Any]:
    """A data center's safe offer with only part of its batch work paused and part of its battery: never more than the
    controller offered. The clearing may scale all of a data center's offer (datacenter.build_offer puts it all in
    battery_kw); the evening rebound at delivery level f of the new offer is the controller's at f x (new / old)."""
    from .market import _rebound_at
    ps = max(0.0, min(1.0, float(pause_share))) if take_part else 0.0
    bs = max(0.0, min(1.0, float(battery_share))) if take_part else 0.0
    fs = max(0.0, min(1.0, float(facility_share))) if take_part else 0.0
    if take_part and ps >= 0.9999 and bs >= 0.9999 and fs >= 0.9999:
        return json.loads(json.dumps(offer))                     # the controller's offer, unchanged to the decimal
    out = json.loads(json.dumps(offer))
    for s in out["slots"]:
        cp0, db0, k0 = float(s.get("compute_pause_kw", 0.0)), float(s.get("dc_battery_kw", 0.0)), float(s["kw_offered"])
        s["compute_pause_kw"] = round(cp0 * ps, 1)
        s["dc_battery_kw"] = round(db0 * bs, 1)
        f0 = float(s.get("facility_kw", 0.0))
        s["facility_kw"] = round(f0 * fs, 1)
        # what is taken away, from the controller's own total (its parts are rounded separately)
        kw = 0.0 if s["compute_pause_kw"] + s["dc_battery_kw"] + s["facility_kw"] <= 0 else \
            round(max(0.0, min(k0, k0 - (cp0 - s["compute_pause_kw"]) - (db0 - s["dc_battery_kw"]) - (f0 - s["facility_kw"]))), 1)
        s["kw_offered"] = s["battery_kw"] = kw
        s["ev_pause_kw"] = 0.0
    out["energy_kwh"] = round(sum(s["kw_offered"] for s in out["slots"]) * beckn.SLOT_MINUTES / 60, 1)
    old_e = float(offer.get("energy_kwh") or 0.0)
    ratio = out["energy_kwh"] / old_e if old_e > 0 else 0.0
    ev = out.get("evening") or {}
    if ev.get("rebound_kw") and ratio < 0.999:
        old = offer["evening"]["rebound_kw"]
        ev["rebound_kw"] = {str(f): [round(x, 1) for x in _rebound_at(old, f * ratio)] for f in REBOUND_FRACTIONS}
    return out


class DataCenterBrain:
    """A data center's AI agent (datacenter.DataCenterAgent.brain): how much of the controller's safe offer to give
    (the share of the batch work to pause and of the battery), and whether to accept its commitment. The LLM proposes;
    the site's code decides: it can only lower the safe offer, the operator's policy applies after it, and a
    commitment the guard declined is never accepted. Too slow, wrong or away: the controller's offer and "accept"."""

    def __init__(self, agent, backend, *, policy: dict[str, Any] | None = None, label: str = "",
                 log: Callable[..., None] | None = None, offer_budget_s: float = 40.0, accept_budget_s: float = 45.0):
        self.a, self.backend, self.label = agent, backend, label
        self.policy = {**DC_POLICY, **(policy or {})}
        self.log = log or (lambda *a, **k: None)
        self.offer_budget_s, self.accept_budget_s = offer_budget_s, accept_budget_s
        self.decisions = Decisions()
        self.needs: dict[str, dict[str, Any]] = {}
        self._done: dict[Any, Any] = {}
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._done.clear()
            self.needs.clear()
            self.decisions = Decisions()

    def _skips(self, day: str) -> bool:
        return day in (self.policy.get("skip_days") or [])

    def _facts_common(self) -> dict[str, Any]:
        s = self.a.site
        # snapshots: the agent decides in a thread of its own (flexgrid-dc) while its handlers may add to these
        accepted = [c for c in list(self.a.contracts.values()) if c.get("status") == "ACCEPTED"]
        payments = list(self.a.payments)
        return {"data_center": self.a.name, "now": self.a.now_ts(),
                "battery": {"kwh": round(s.E), "kw": round(s.P), "backup_reserve_pct": round(s.backup * 100),
                            "charge_now_pct": round(s.soc * 100, 1), "energy_above_reserve_now_kwh": round(s.usable_kwh())},
                "deferral_window_h": round(s.w * 0.25, 2),
                # work already held back and waiting to run right now (normally 0 outside an event), not what the
                # window's offer pauses: that is the safe offer's batch_work_it_can_pause_kw
                "batch_work_already_held_back_now_kwh": round(s.queued_kwh(), 1),
                "operator_policy": dict(self.policy),
                # conditions of the day the site's operator knows of (the Control Room's what if)
                **({"today": {k: v for k, v in (("urgent_batch_work_share", round(s.urgent, 2) if s.urgent > 0 else None),
                                                ("battery_out_of_service", True if s.battery_out else None)) if v is not None}}
                   if s.urgent > 0 or s.battery_out else {}),
                "track_record": {"events_accepted": len(accepted),
                                 "paid_usd": round(sum(float(p.get("payment") or 0) for p in payments), 2),
                                 "reputation": round(float(self.a.reputation), 3)}}

    def decide_offer(self, need: dict[str, Any], offer: dict[str, Any]) -> dict[str, Any]:
        eid = need["event_id"]
        with self._lock:
            if ("offer", eid) in self._done:
                return self._done[("offer", eid)]
        self.needs[eid] = need
        sl = need.get("slots") or []
        price = float(sl[0].get("price") or 0.0) if sl else 0.0
        day = need["window_start"][:10]
        p = self.policy
        skip = ("the operator's policy: no events that day" if self._skips(day) else
                f"the price ${price:.4f}/kWh is below the operator's minimum ${p['min_price_usd_per_kwh']:.4f}"
                if price < float(p["min_price_usd_per_kwh"]) else None)
        safe_kwh = float(offer.get("energy_kwh") or 0.0)
        t0 = time.perf_counter()
        ans, meta = None, {"model": None, "ms": None, "error": None}
        if skip:
            meta["error"] = "the operator's policy decides"
        elif safe_kwh <= 0.01:
            meta["error"] = "nothing safe to offer (the controller offered 0 kW)"
        else:
            facts = {**self._facts_common(),
                     "request": {"event_id": eid, "date": day,
                                 "window": f"{need['window_start'][11:]}-{need['window_end'][11:]}",
                                 "kw_asked_per_half_hour_all_data_centers": [s.get("kw_requested") for s in sl],
                                 "price_usd_per_kwh": price, "reply_by": need.get("commit_by")},
                     "safe_offer": {"per_half_hour": [{"start": (s_need.get("start") or "")[11:],
                                                       "batch_work_it_can_pause_kw": s.get("compute_pause_kw"),
                                                       "battery_kw": s.get("dc_battery_kw"),
                                                       "facility_kw": s.get("facility_kw", 0.0)}
                                                      for s, s_need in zip(offer["slots"], sl + [{}] * len(offer["slots"]))],
                                    "batch_work_kwh": round(sum(float(s.get("compute_pause_kw") or 0) for s in offer["slots"])
                                                            * beckn.SLOT_MINUTES / 60, 1),
                                    "battery_kwh": round(sum(float(s.get("dc_battery_kw") or 0) for s in offer["slots"])
                                                         * beckn.SLOT_MINUTES / 60, 1),
                                    "energy_kwh": safe_kwh, "confidence": offer.get("confidence"),
                                    "if_all_delivered_usd": round(safe_kwh * price, 2)},
                     "after_the_window": {"paused_work_must_run_again_within_h": offer.get("deferral_window_h"),
                                          "rebound_kw_if_all_delivered_max": max(((offer.get("evening") or {})
                                                                                  .get("rebound_kw") or {}).get("1.0") or [0.0])}}
            ans, meta = self.backend.decide("datacenter", DC_DECIDE, facts, DC_OFFER_SCHEMA,
                                            budget_s=self.offer_budget_s, session=f"{self.a.name}-offer-{eid}")
        fallback = ans is None
        fs = 1.0 if fallback else _num(ans.get("facility_share"), 0.0, 1.0, 1.0)
        if not fallback and isinstance(ans.get("levers_off"), list):
            self.a.site.levers_off = ({str(x) for x in ans["levers_off"]} & set(getattr(self.a.site, "levers_all", ()) or ())) \
                | set(getattr(self.a.site, "levers_forbidden", ()) or ())    # it may add to what is off, never take away
        if fallback:
            take, ps, bs = bool(not skip and safe_kwh > 0.01), 1.0, 1.0
            reason = skip or ("rules: the controller's safe offer" + (f" ({meta['error']})" if meta.get("error") else ""))
            why_log = reason
        else:
            take = bool(ans.get("take_part"))
            ps = _num(ans.get("pause_share"), 0.0, 1.0, 1.0)
            bs = _num(ans.get("battery_share"), 0.0, 1.0, 1.0)
            reason = str(ans.get("reason_for_operator") or "")[:400]
            why_log = str(ans.get("reason_for_log") or reason)[:400]
        clamped = []                                       # the operator's policy after the LLM, always
        if skip and take:
            take = False
            clamped.append("operator's policy: no event")
        if ps > float(p["pause_max_share"]) + 1e-9:
            ps = float(p["pause_max_share"])
            clamped.append(f"batch work paused at most {ps:.0%} (operator's policy)")
        if bs > float(p["battery_max_share"]) + 1e-9:
            bs = float(p["battery_max_share"])
            clamped.append(f"battery used at most {bs:.0%} (operator's policy)")
        final = scale_dc_offer(offer, ps, bs, take, fs)
        d = self.decisions.add(kind="offer", event_id=eid, sim=self.a.now_ts(), take_part=take,
                               pause_share=round(ps if take else 0.0, 2), battery_share=round(bs if take else 0.0, 2),
                               safe_kw=[s["kw_offered"] for s in offer["slots"]], kw=[s["kw_offered"] for s in final["slots"]],
                               price=price, if_delivered_usd=round(float(final["energy_kwh"]) * price, 2),
                               reason=reason, reason_log=why_log, guard=clamped, fallback=fallback,
                               llm={"model": meta.get("llm_model") or meta.get("model"), "ms": meta.get("ms"),
                                    "error": meta.get("error"), "runtime": meta.get("runtime")})
        # what was decided goes out with the signed offer (the desk keeps it; a replay can play it)
        final["agent"] = {"decided_by": "rules" if fallback else "llm", "reason": reason[:300], "decision": d["n"],
                          "take_part": bool(take), "pause_share": float(ps), "battery_share": float(bs)}
        self.log("agent_decision", what="dc_offer", event_id=eid, take_part=take, kw=d["kw"], fallback=fallback,
                 reason=reason[:200], ms=round((time.perf_counter() - t0) * 1000), runtime=meta.get("runtime"))
        with self._lock:
            self._done[("offer", eid)] = final
        return final

    def decide_accept(self, c: dict[str, Any], verdict: dict[str, Any]) -> dict[str, Any]:
        cid = c.get("id")
        if verdict.get("decision") != "ACCEPTED" or not cid:
            return verdict                                   # the guard said no: the LLM is not asked
        with self._lock:
            if ("accept", cid) in self._done:
                return self._done[("accept", cid)]
        day = str(c.get("window_start", ""))[:10]
        skip = self._skips(day)
        offer = self.a.offers.get(c.get("event_id")) or {}
        need = self.needs.get(c.get("event_id")) or {}
        sl = need.get("slots") or []
        price = float(sl[0].get("price") or 0.0) if sl else float(c.get("price_usd_per_kwh") or 0.0)
        kw = [float(x.get("kw", 0)) for x in c.get("commitments", [])]
        usd = round(sum(kw) * beckn.SLOT_MINUTES / 60 * price, 2)
        t0 = time.perf_counter()
        ans, meta = None, {"model": None, "ms": None, "error": None}
        if skip:
            meta["error"] = "the operator's policy decides"
        else:
            facts = {**self._facts_common(),
                     "commitment": {"event_id": c.get("event_id"), "per_half_hour_kw": kw, "price_usd_per_kwh": price,
                                    "if_delivered_usd": usd, "opt_out_by": c.get("opt_out_by"),
                                    "paused_work_comes_back": "on the desk's resume plan" if c.get("resume_plan") else None},
                     "our_offer_kw": [s.get("kw_offered") for s in offer.get("slots", [])],
                     "why_we_offered": (offer.get("agent") or {}).get("reason"),
                     "safety_check": verdict.get("reason")}
            ans, meta = self.backend.decide("datacenter", DC_ACCEPT, facts, DC_ACCEPT_SCHEMA,
                                            budget_s=self.accept_budget_s, session=f"{self.a.name}-accept-{cid}")
        fallback = ans is None
        if skip:
            out = {"decision": "DECLINED", "reason": "operator_opted_out", "contract_id": cid}
            reason = "the operator's policy: no events that day"
        elif fallback or bool(ans.get("accept")):
            out = verdict
            reason = "rules: within the offer and safe" if fallback else str(ans.get("reason_for_operator") or "")[:400]
        else:
            out = {"decision": "DECLINED", "reason": "agent_declined", "contract_id": cid}
            reason = str(ans.get("reason_for_operator") or "")[:400]
        self.decisions.add(kind="accept", event_id=c.get("event_id"), contract_id=cid, sim=self.a.now_ts(),
                           accept=out["decision"] == "ACCEPTED", decision=out["decision"], kw=kw, price=price,
                           if_delivered_usd=usd, reason=reason, fallback=fallback,
                           llm={"model": meta.get("llm_model") or meta.get("model"), "ms": meta.get("ms"),
                                "error": meta.get("error"), "runtime": meta.get("runtime")})
        self.log("agent_decision", what="dc_accept", event_id=c.get("event_id"), decision=out["decision"],
                 fallback=fallback, reason=reason[:200], ms=round((time.perf_counter() - t0) * 1000),
                 runtime=meta.get("runtime"))
        with self._lock:
            self._done[("accept", cid)] = out
        return out

    def state(self) -> dict[str, Any]:
        """The Control Room's chat and cast: this data center's decisions (one-line text, the model's display name)."""
        from .agents_feed import decision_text, model_label
        ds = []
        for d in self.decisions.list(30):
            m = dict(d.get("llm") or {})
            m["model"] = model_label(m.get("model")) or (model_label(self.backend_model()) if not d.get("fallback") else None)
            ds.append({**d, "text": decision_text(d), "llm": m})
        return {"enabled": True, "site": self.a.name, "label": self.label, "llm": self.backend.status(),
                "policy": dict(self.policy), "decisions": ds,
                "thinking": self.a.thinking() if hasattr(self.a, "thinking") else []}

    def backend_model(self) -> str | None:
        return getattr(self.backend, "model_tag", None) or next(
            (getattr(b, "model_tag", None) for b in getattr(self.backend, "backends", ()) if getattr(b, "model_tag", None)), None)


# --------------------------------------------------------------------------------------------- the grid operator
OPERATOR_ORDER_SCHEMA = {"type": "object", "properties": {
    "start": {"type": "string"}, "end": {"type": "string"}, "mw": {"type": "number", "minimum": 0, "maximum": 100},
    "reason": {"type": "string"}}, "required": ["start", "end", "mw", "reason"]}
OPERATOR_SIGN_SCHEMA = {"type": "object", "properties": {"approve": {"type": "boolean"}, "reason": {"type": "string"}},
                        "required": ["approve", "reason"]}
OPERATOR_ORDER = """You are the AI agent of the grid operator's large-load program (mock): it buys flexibility from ten
data centers through FlexGrid's large-load desk. Order tomorrow's relief now, before the desk's 10:00 call for offers:
the window (start and end as HH:MM on the half hour, 1 to 4 hours, between 12:00 and 22:00) and the MW of less grid
import you want in each half hour (10 to 80). The program pays the window's day-ahead price per kWh its meters measure
(the code sets the price from your window). Put the window where the region's forecast load is highest. The program's
own request (program_request) is sized for a day whose forecast peak reaches the planning level: order more than it
when the forecast peak is above the planning level, less when the forecast peak is below it (the further below, the
less), and no more than the data centers can deliver (their earlier events say how much). Reason: one or two plain
sentences with the numbers that mattered."""
OPERATOR_SIGN = """You are the AI agent of the grid operator's large-load program. An event is settled, and the
independent check has recomputed it from the signed meter readings (signatures, the payment to the cent, the split, an
offline replay of what was decided). Approve the settlement only if every check passed; otherwise dispute it and name
the failing checks. One or two plain sentences: what the program pays, for how much measured, and your decision."""


def _hhmm(x: Any) -> int | None:
    """'17:00', '5:30 pm', '1730' -> minutes after midnight, rounded to the half hour; None if unreadable."""
    import re
    t = str(x or "").strip().lower()
    m = re.match(r"^(\d{1,2})(?::?(\d{2}))?\s*(am|pm)?$", t)
    if not m:
        return None
    hh, mm = int(m.group(1)), int(m.group(2) or 0)
    if m.group(3) == "pm" and hh < 12:
        hh += 12
    if m.group(3) == "am" and hh == 12:
        hh = 0
    if not (0 <= hh < 24 and 0 <= mm < 60):
        return None
    return int(round((hh * 60 + mm) / 30.0)) * 30


class GridOperatorBrain:
    """The grid operator's AI agent (flexgrid-dc, market role): tomorrow's order, then the sign-off. The LLM proposes;
    the code decides: the window is snapped to the half hour, kept within 12:00-22:00 and 1-4 h, the MW clamped to
    10-80 and kept on the side of the program's request that its own forecast points to (no more than the request when
    the forecast peak is below the planning level, no less when it is above: over the 2026 season one order of 45 said
    "less than the 40 MW program request" and asked for 80), the price is the window's day-ahead average (never the
    LLM's), and the order still goes through the grid operator's own checks (GridOperator.place_order: before the 10:00
    call, once a day). An approval needs every check passed. Too slow or away: the program's catalog request stands,
    and the rules sign off (approve iff all passed)."""
    MW_MIN, MW_MAX, DAY_FROM, DAY_TO, H_MIN, H_MAX = 10.0, 80.0, 12 * 60, 22 * 60, 60, 240

    def __init__(self, backend, *, log: Callable[..., None] | None = None, order_budget_s: float = 40.0,
                 sign_budget_s: float = 30.0):
        self.backend = backend
        self.log = log or (lambda *a, **k: None)
        self.order_budget_s, self.sign_budget_s = order_budget_s, sign_budget_s
        from .datacenter import files as _dc_files
        top = float(_dc_files().get("order_max_kw") or 0.0) / 1000.0    # a program sized to its fleet (none: 10-80 MW)
        if top > self.MW_MAX:
            self.MW_MAX = top
        self.order_schema = OPERATOR_ORDER_SCHEMA if top <= 100.0 else {**OPERATOR_ORDER_SCHEMA, "properties": {
            **OPERATOR_ORDER_SCHEMA["properties"], "mw": {"type": "number", "minimum": 0, "maximum": top}}}
        self.decisions = Decisions()
        self.busy: str | None = None

    def decide_order(self, day: str, facts: dict[str, Any], catalog: dict[str, Any],
                     price_of: Callable[[str, str], float]) -> dict[str, Any]:
        """facts: what the grid operator knows the morning before (dc_fleet.operator_facts); catalog: the program's
        request for the day {start, end, mw, price}; price_of(start, end): the window's day-ahead average in $/kWh.
        Returns {"order": the order to place (None: the catalog stands), "decision": the decision record}."""
        self.busy = f"ordering flexibility for {day}"
        t0 = time.perf_counter()
        try:
            ans, meta = self.backend.decide("operator", OPERATOR_ORDER, facts, self.order_schema,
                                            budget_s=self.order_budget_s, session=f"operator-order-{day}")
        finally:
            self.busy = None
        fallback, guard = ans is None, []
        if fallback:
            start, end, mw = catalog["start"], catalog["end"], float(catalog["mw"])
            reason = "rules: the program's request stands" + (f" ({meta['error']})" if meta.get("error") else "")
        else:
            reason = str(ans.get("reason") or "")[:400]
            m0, m1 = _hhmm(ans.get("start")), _hhmm(ans.get("end"))
            c0, c1 = _hhmm(catalog["start"]), _hhmm(catalog["end"])
            if m0 is None or m1 is None:
                m0, m1 = c0, c1
                guard.append("the window could not be read: the program's window")
            if m0 < self.DAY_FROM or m1 > self.DAY_TO or m1 <= m0:
                guard.append("window kept within 12:00-22:00")
                m0 = max(self.DAY_FROM, min(m0, self.DAY_TO - self.H_MIN))
                m1 = max(m0 + self.H_MIN, min(m1, self.DAY_TO))
            if m1 - m0 > self.H_MAX:
                m1 = m0 + self.H_MAX
                guard.append("window at most 4 h")
            elif m1 - m0 < self.H_MIN:
                m1 = min(self.DAY_TO, m0 + self.H_MIN)
                m0 = m1 - self.H_MIN
                guard.append("window at least 1 h")
            start, end = f"{m0 // 60:02d}:{m0 % 60:02d}", f"{m1 // 60:02d}:{m1 % 60:02d}"
            mw = _num(ans.get("mw"), 0.0, 1000.0, float(catalog["mw"]))
            if not self.MW_MIN <= mw <= self.MW_MAX:
                mw = max(self.MW_MIN, min(self.MW_MAX, mw))
                guard.append(f"MW kept within {self.MW_MIN:g}-{self.MW_MAX:g}")
            gap, req = facts.get("forecast_peak_above_planning_level_mw"), float(catalog["mw"])
            if isinstance(gap, (int, float)) and gap < 0 and mw > req:
                mw = req
                guard.append(f"forecast peak below the planning level: at most the program's {req:g} MW")
            elif isinstance(gap, (int, float)) and gap > 0 and mw < req:
                mw = req
                guard.append(f"forecast peak above the planning level: at least the program's {req:g} MW")
            mw = round(mw, 1)
        price = round(float(price_of(start, end)), 4)
        d = self.decisions.add(kind="order", day=day, event_id=catalog.get("event_id"), sim=facts.get("now"),
                               start=start, end=end, mw=mw, price=price, placed=not fallback,
                               catalog={k: catalog.get(k) for k in ("start", "end", "mw", "price")},
                               reason=reason, guard=guard, fallback=fallback,
                               llm={"model": meta.get("llm_model") or meta.get("model"), "ms": meta.get("ms"),
                                    "error": meta.get("error"), "runtime": meta.get("runtime")})
        self.log("agent_decision", what="operator_order", day=day, mw=mw, window=[start, end], price=price,
                 fallback=fallback, reason=reason[:200], ms=round((time.perf_counter() - t0) * 1000),
                 runtime=meta.get("runtime"))
        order = None if fallback else {"date": day, "start": start, "end": end, "kw": mw * 1000.0,
                                       "price_usd_per_kwh": price, "note": reason[:200], "buyer": "grid operator's agent"}
        return {"order": order, "decision": d}

    def refused(self, d: dict[str, Any], why: str) -> None:
        """The grid operator's code refused the order (too late, a second one): the catalog stands, and the log says."""
        d["placed"], d["refused"] = False, why[:200]
        self.log("agent_error", where="operator order", error=why[:200])

    def decide_sign_off(self, event_id: str, check: dict[str, Any], now: str | None = None) -> dict[str, Any]:
        """check: verify.check()'s result for the settled event (with the replay); now: the replay clock's time.
        Returns the decision record (decision: approved | disputed)."""
        self.busy = f"checking the settlement of {event_id}"
        t0 = time.perf_counter()
        rows = [{"check": x.get("name"), "result": "PASS" if x.get("ok") else ("SKIP" if x.get("ok") is None else "FAIL"),
                 "detail": str(x.get("detail") or "")[:300]} for x in check.get("checks") or []]
        summ = check.get("summary") or {}
        facts = {"event_id": event_id, "all_checks_pass": bool(check.get("ok")), "checks": rows,
                 "summary": {k: summ.get(k) for k in ("window", "requested_kw", "price_usd_per_kwh", "contracted_kwh",
                                                       "delivered_kwh_measured", "paid_kwh", "delivered_kwh_true",
                                                       "utility_pays_usd", "readings")}}
        try:
            ans, meta = self.backend.decide("operator", OPERATOR_SIGN, facts, OPERATOR_SIGN_SCHEMA,
                                            budget_s=self.sign_budget_s, session=f"operator-sign-{event_id}")
        finally:
            self.busy = None
        ok_all = bool(check.get("ok"))
        fallback, guard = ans is None, []
        if fallback:
            approve = ok_all
            reason = ("rules: every check passed" if ok_all else
                      "rules: failed " + ", ".join(r["check"] for r in rows if r["result"] == "FAIL"))
        else:
            approve, reason = bool(ans.get("approve")), str(ans.get("reason") or "")[:400]
        if approve and not ok_all:
            approve = False
            guard.append("approval refused: not every check passed")
        d = self.decisions.add(kind="sign_off", event_id=event_id, decision="approved" if approve else "disputed",
                               usd=summ.get("utility_pays_usd"), sim=now, reason=reason, guard=guard, fallback=fallback,
                               measured_kwh=summ.get("delivered_kwh_measured"), paid_kwh=summ.get("paid_kwh"),
                               failed=[r["check"] for r in rows if r["result"] == "FAIL"],
                               llm={"model": meta.get("llm_model") or meta.get("model"), "ms": meta.get("ms"),
                                    "error": meta.get("error"), "runtime": meta.get("runtime")})
        self.log("agent_decision", what="operator_sign_off", event_id=event_id, decision=d["decision"],
                 fallback=fallback, reason=reason[:200], ms=round((time.perf_counter() - t0) * 1000),
                 runtime=meta.get("runtime"))
        return d

    def state(self) -> dict[str, Any]:
        from .agents_feed import model_label
        ds = []
        for d in self.decisions.list(30):
            m = dict(d.get("llm") or {})
            m["model"] = model_label(m.get("model")) or (model_label(self.backend_model()) if not d.get("fallback") else None)
            ds.append({**d, "llm": m})
        return {"enabled": True, "llm": self.backend.status(), "busy": self.busy, "decisions": ds}

    def backend_model(self) -> str | None:
        return getattr(self.backend, "model_tag", None) or next(
            (getattr(b, "model_tag", None) for b in getattr(self.backend, "backends", ()) if getattr(b, "model_tag", None)), None)


# --------------------------------------------------------------------------------------------- the replay
class ReplayBrain:
    """A home's agent as a live run recorded it, for the offline replay of that event (verify.replay_total ->
    tools/sim_event.py --agents): its offer decision (take part, battery share, EV pause, as its signed offer carried
    them) and its answer, applied by the same code. No LLM: the replay checks the code, given what was decided."""

    def __init__(self, offers: dict[str, dict[str, Any]] | None = None, declines: set[str] | None = None):
        self.offers, self.declines = offers or {}, set(declines or ())

    def decide_offer(self, need: dict[str, Any], fc: dict[str, Any], offer: dict[str, Any]) -> dict[str, Any]:
        d = self.offers.get(need["event_id"])
        if not d:
            return offer
        if "cooling_share" in d or "lighting_share" in d:        # a building node's agent (BuildingBrain)
            final = scale_building_offer(offer, float(d.get("cooling_share", 1.0)), float(d.get("lighting_share", 1.0)),
                                         bool(d.get("take_part", True)))
            final["agent"] = {"decided_by": "replay", **{k: d.get(k) for k in ("take_part", "cooling_share", "lighting_share")}}
            return final
        final = scale_offer(offer, float(d.get("battery_share", 1.0)), bool(d.get("pause_ev")), bool(d.get("take_part", True)))
        final["agent"] = {"decided_by": "replay", **{k: d.get(k) for k in ("take_part", "battery_share", "pause_ev")}}
        return final

    def decide_accept(self, c: dict[str, Any], verdict: dict[str, Any]) -> dict[str, Any]:
        if c.get("event_id") in self.declines:
            return {"decision": "DECLINED", "reason": "replay: this home declined in the live run", "contract_id": c.get("id")}
        return verdict

    def reset(self) -> None:
        pass


class ReplayDCBrain:
    """A data center's agent as a live run recorded it (its signed offer's "agent" block: take part, the share of the
    batch work paused and of the battery), for the offline replay of that event (dc_fleet.replay_total ->
    tools/sim_datacenters.py --agents): the same decision through the same code. No LLM."""

    def __init__(self, offers: dict[str, dict[str, Any]] | None = None, declines: set[str] | None = None):
        self.offers, self.declines = offers or {}, set(declines or ())

    def decide_offer(self, need: dict[str, Any], offer: dict[str, Any]) -> dict[str, Any]:
        d = self.offers.get(need["event_id"])
        if not d:
            return offer
        final = scale_dc_offer(offer, float(d.get("pause_share", 1.0)), float(d.get("battery_share", 1.0)),
                               bool(d.get("take_part", True)))
        final["agent"] = {"decided_by": "replay", **{k: d.get(k) for k in ("take_part", "pause_share", "battery_share")}}
        return final

    def decide_accept(self, c: dict[str, Any], verdict: dict[str, Any]) -> dict[str, Any]:
        if verdict.get("decision") == "ACCEPTED" and c.get("event_id") in self.declines:
            return {"decision": "DECLINED", "reason": "replay: this data center declined in the live run",
                    "contract_id": c.get("id")}
        return verdict

    def reset(self) -> None:
        pass


class ReplayMargin:
    """The coordinator's margin as a live run recorded it (event id -> margin); the market's default for any other."""
    budget_s = 1.0

    def __init__(self, margins: dict[str, float], default: float):
        self.margins, self.default = margins, default

    def decide_margin(self, snap: dict[str, Any]) -> dict[str, Any]:
        m = self.margins.get((snap.get("request") or {}).get("event_id"))
        return {"margin": self.default if m is None else float(m), "fallback": m is None, "reason": "replay", "llm": None}


# --------------------------------------------------------------------------------------------- runtimes
class OllamaBackend:
    """The LLM straight from Ollama (llm.py): structured answers for decisions, a tool loop for chats."""
    runtime = "ollama"

    def __init__(self, llm):
        self.llm = llm

    @property
    def model_tag(self) -> str | None:
        return self.llm.model or self.llm.want

    def status(self) -> dict[str, Any]:
        return {**self.llm.status(), "runtime": "ollama"}

    def decide(self, role: str, system: str, facts: dict[str, Any], schema: dict[str, Any], *, budget_s: float,
               session: str | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        return self.llm.decide(system, facts, schema, budget_s=budget_s)

    def converse(self, role: str, system: str, messages: list[dict[str, Any]], tools, *, budget_s: float,
                 session: str | None = None) -> dict[str, Any]:
        return self.llm.converse(system, messages, tools, budget_s=budget_s)


class OpenClawBackend:
    """The agent runtime: this machine's OpenClaw gateway, through its OpenAI-compatible endpoint
    (POST /v1/chat/completions, docs.openclaw.ai/gateway/openai-http-api; enable it with
    gateway.http.endpoints.chatCompletions.enabled). Each call is a normal OpenClaw agent run of `agent`
    (model "openclaw/<agent>"): the agent's workspace (AGENTS.md, SOUL.md: its role and character), its FlexGrid skill
    and its FlexGrid tools over MCP, on the model OpenClaw is configured with (the local Qwen3.8-27B, Ollama on the Mac).

      decide    OpenClaw's llm-task (POST /tools/invoke): an isolated JSON-only call with the agent's model, the output
                validated against the decision's schema by OpenClaw - small prompt, fast enough for market deadlines.
                A gateway without llm-task: one client function tool, `submit_decision`, whose parameters are the
                schema, pinned with tool_choice on /v1/chat/completions (a fresh session per decision, no `user`)
      converse  the owner's conversation: `user` = one stable session per home, so OpenClaw keeps the history (only
                the new message is sent) and its memory. The agent uses its own tools (MCP -> this daemon)
    """
    runtime = "openclaw"

    def __init__(self, url: str, token: str, agent: str, *, retry_down_s: float = 20.0, model_tag: str | None = None):
        import httpx
        self.url, self.token, self.agent = url.rstrip("/"), token, agent
        self.model_tag = model_tag                     # the model OpenClaw's agent runs on (for display only)
        self.http = httpx.Client(timeout=10)
        self.retry_down_s = retry_down_s
        self._down_until, self._err, self._ok = 0.0, None, False
        self.calls, self.ms = 0, []
        self.lock = threading.Lock()

    def _h(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def status(self) -> dict[str, Any]:
        with self.lock:
            cached = self._ok and time.time() >= self._down_until or time.time() < self._down_until
        if not cached:
            try:
                r = self.http.get(self.url + "/v1/models", headers=self._h(), timeout=3)
                r.raise_for_status()
                ids = [m.get("id") for m in r.json().get("data", [])]
                want = f"openclaw/{self.agent}"
                ok = want in ids
                with self.lock:
                    self._ok, self._err = ok, (None if ok else f"no OpenClaw agent {self.agent!r} on the gateway")
                    self._down_until = 0.0 if ok else time.time() + self.retry_down_s
            except Exception as exc:
                with self.lock:
                    self._ok, self._err = False, f"OpenClaw gateway: {type(exc).__name__}: {str(exc)[:120]}"
                    self._down_until = time.time() + self.retry_down_s
        p50 = sorted(self.ms)[len(self.ms) // 2] if self.ms else None
        return {"ok": self._ok, "runtime": "openclaw", "url": self.url, "agent": self.agent,
                "model": f"openclaw/{self.agent}" if self._ok else None, "error": None if self._ok else self._err,
                "calls": self.calls, "median_ms": round(p50) if p50 is not None else None}

    def _post(self, body: dict[str, Any], timeout: float) -> dict[str, Any]:
        t0 = time.perf_counter()
        r = self.http.post(self.url + "/v1/chat/completions", headers=self._h(), json=body, timeout=max(1.0, timeout))
        if r.status_code >= 400:
            raise RuntimeError(f"OpenClaw {r.status_code}: {r.text[:200]}")
        ms = (time.perf_counter() - t0) * 1000
        self.calls += 1
        self.ms = (self.ms + [ms])[-200:]
        j = r.json()
        j["_ms"] = ms
        return j

    def _down(self, err: str) -> None:
        with self.lock:
            self._ok, self._err, self._down_until = False, err, time.time() + self.retry_down_s

    def _llm_task(self, system: str, facts: dict[str, Any], schema: dict[str, Any], budget_s: float,
                  meta: dict[str, Any]) -> dict[str, Any] | None:
        """OpenClaw's own JSON step: the bundled llm-task tool through POST /tools/invoke (docs.openclaw.ai/tools/
        llm-task): one isolated prompt-only call with the agent's model, the output validated against `schema` by
        OpenClaw. Returns the decision, or None if this gateway has no llm-task (404: the caller tries the chat way)."""
        t0 = time.perf_counter()
        t_end = time.time() + budget_s
        prompt = system + "\n\nDecide now from the facts in the input. " + _answer_shape(schema)
        for attempt in (1, 2):                  # a second try (if time is left) when the answer missed the schema
            left = t_end - time.time()
            r = self.http.post(self.url + "/tools/invoke", headers=self._h(), timeout=max(1.0, left + 2),
                               json={"tool": "llm-task", "agentId": self.agent,
                                     "args": {"prompt": prompt, "input": facts, "schema": schema, "temperature": 0.3,
                                              "maxTokens": 400, "timeoutMs": int(max(1.0, left) * 1000)}})
            if r.status_code == 404:
                return None
            if r.status_code >= 400 and attempt == 1 and t_end - time.time() > 8:
                prompt += ("\n\nYour last answer was refused: it did not have the keys above at the top level. Answer "
                           "again with exactly that JSON object.")
                meta["retried"] = True
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"OpenClaw llm-task {r.status_code}: {r.text[:200]}")
            break
        ms = (time.perf_counter() - t0) * 1000
        self.calls += 1
        self.ms = (self.ms + [ms])[-200:]
        meta["ms"], meta["route"] = round(ms), "llm-task"
        res = (r.json() or {}).get("result") or {}
        det = res.get("details") if isinstance(res, dict) else None
        ans = (det or {}).get("json") if isinstance(det, dict) else None
        if ans is None and isinstance(res, dict):            # the text part, if the details are not there
            txt = " ".join(c.get("text", "") for c in res.get("content") or [] if isinstance(c, dict))
            ans = json.loads(txt[txt.find("{"): txt.rfind("}") + 1]) if "{" in txt else None
        if isinstance(det, dict) and det.get("model"):
            meta["llm_model"] = f"{det.get('provider', '')}/{det['model']}".strip("/")
        meta["raw"] = json.dumps(ans)[:600] if ans is not None else str(res)[:600]
        return ans if isinstance(ans, dict) else {}

    def decide(self, role: str, system: str, facts: dict[str, Any], schema: dict[str, Any], *, budget_s: float,
               session: str | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        import httpx
        meta: dict[str, Any] = {"model": None, "ms": None, "error": None, "runtime": "openclaw"}
        if not self.status()["ok"]:
            meta["error"] = f"no LLM: {self._err}"
            return None, meta
        meta["model"] = f"openclaw/{self.agent}"
        if getattr(self, "_llm_task_ok", True):
            try:
                ans = self._llm_task(system, facts, schema, budget_s, meta)
                if ans is None:
                    self._llm_task_ok = False                # this gateway has no llm-task: the chat way from now on
                else:
                    missing = [k for k in schema.get("required", []) if k not in ans]
                    if missing:
                        meta["error"] = f"missing {missing}"
                        return None, meta
                    return ans, meta
            except httpx.TimeoutException:
                meta["error"] = f"no answer within {budget_s:.0f} s"
                return None, meta
            except (httpx.HTTPError, OSError) as exc:
                meta["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                self._down(meta["error"])
                return None, meta
            except (RuntimeError, ValueError, KeyError, IndexError) as exc:
                meta["error"] = f"unusable answer: {str(exc)[:160]}"
                return None, meta
        meta["route"] = "chat"
        tool = {"type": "function", "function": {"name": "submit_decision",
                                                  "description": "Submit your decision. Call it exactly once.",
                                                  "parameters": schema}}
        text = (f"{system}\n\nFacts (JSON):\n{json.dumps(facts, separators=(',', ':'))}\n\n"
                "Decide now and call submit_decision with your decision. Do not call any other tool.")
        try:
            j = self._post({"model": f"openclaw/{self.agent}", "stream": False, "temperature": 0.3, "top_p": 0.8,
                            "max_completion_tokens": 400, "messages": [{"role": "user", "content": text}],
                            "tools": [tool], "tool_choice": {"type": "function", "function": {"name": "submit_decision"}}},
                           budget_s)
            meta["ms"] = round(j["_ms"])
            msg = ((j.get("choices") or [{}])[0].get("message") or {})
            calls = [c for c in msg.get("tool_calls") or [] if (c.get("function") or {}).get("name") == "submit_decision"]
            raw = (calls[0]["function"].get("arguments") if calls else None) or msg.get("content") or ""
            meta["raw"] = str(raw)[:600]
            ans = raw if isinstance(raw, dict) else json.loads(raw[raw.find("{"): raw.rfind("}") + 1]) if "{" in raw else None
            if not isinstance(ans, dict):
                meta["error"] = "the answer was not a decision"
                return None, meta
            missing = [k for k in schema.get("required", []) if k not in ans]
            if missing:
                meta["error"] = f"missing {missing}"
                return None, meta
            return ans, meta
        except httpx.TimeoutException:
            meta["error"] = f"no answer within {budget_s:.0f} s"
        except (httpx.HTTPError, OSError) as exc:
            meta["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            self._down(meta["error"])
        except (RuntimeError, ValueError, KeyError, IndexError) as exc:
            meta["error"] = f"unusable answer: {str(exc)[:160]}"
        return None, meta

    def converse(self, role: str, system: str, messages: list[dict[str, Any]], tools, *, budget_s: float,
                 session: str | None = None) -> dict[str, Any]:
        import httpx
        out: dict[str, Any] = {"reply": None, "trace": [], "model": None, "ms": 0, "error": None, "runtime": "openclaw"}
        if not self.status()["ok"]:
            out["error"] = f"no LLM: {self._err}"
            return out
        out["model"] = f"openclaw/{self.agent}"
        last = next((m for m in reversed(messages) if m.get("role") == "user"), {"content": ""})
        try:
            j = self._post({"model": f"openclaw/{self.agent}", "stream": False, "user": session or None,
                            "messages": [{"role": "user", "content": last["content"]}]}, budget_s)
            out["ms"] = round(j["_ms"])
            msg = ((j.get("choices") or [{}])[0].get("message") or {})
            out["reply"] = _strip_think(msg.get("content") or "") or "(no answer)"
        except httpx.TimeoutException:
            out["error"] = f"no answer within {budget_s:.0f} s"
        except (httpx.HTTPError, OSError) as exc:
            out["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            self._down(out["error"])
        except (RuntimeError, ValueError, KeyError, IndexError) as exc:
            out["error"] = str(exc)[:200]
        return out


def _answer_shape(schema: dict[str, Any]) -> str:
    """'Answer only with a JSON object with these keys: take_part (true or false), ...' from a decision's schema."""
    kinds = {"boolean": "true or false", "number": "a number", "string": "a short text", "integer": "a whole number"}
    parts = []
    for k, v in (schema.get("properties") or {}).items():
        t = kinds.get(v.get("type"), v.get("type", "a value"))
        if v.get("type") == "number" and "minimum" in v and "maximum" in v:
            t = f"a number from {v['minimum']:g} to {v['maximum']:g}"
        parts.append(f'"{k}" ({t})')
    return "Answer only with one JSON object with these keys at the top level: " + ", ".join(parts) + "."


def _strip_think(s: str) -> str:
    import re
    return re.sub(r"<think>.*?</think>\s*", "", s, flags=re.S).strip()


class ChainBackend:
    """The first runtime that is up answers: OpenClaw, else Ollama straight (same model), else the rules decide. The
    decision log says which one answered."""

    def __init__(self, *backends):
        self.backends = backends

    def _pick(self):
        return next((b for b in self.backends if b.status().get("ok")), None)

    def status(self) -> dict[str, Any]:
        sts = [b.status() for b in self.backends]
        ok = next((s for s in sts if s.get("ok")), None)
        if ok:
            return {**ok, "standby": [s.get("runtime") for s in sts if s is not ok and s.get("ok")],
                    "down": {s.get("runtime"): s.get("error") for s in sts if not s.get("ok")}}
        return {"ok": False, "runtime": None, "model": None,
                "error": "; ".join(f"{s.get('runtime')}: {s.get('error')}" for s in sts)}

    def decide(self, role, system, facts, schema, *, budget_s, session=None):
        b = self._pick()
        if b is None:
            return None, {"model": None, "ms": None, "error": "no LLM: " + self.status()["error"]}
        ans, meta = b.decide(role, system, facts, schema, budget_s=budget_s, session=session)
        meta.setdefault("runtime", getattr(b, "runtime", None))
        return ans, meta

    def converse(self, role, system, messages, tools, *, budget_s, session=None):
        b = self._pick()
        if b is None:
            return {"reply": None, "trace": [], "model": None, "ms": 0, "error": "no LLM: " + self.status()["error"]}
        out = b.converse(role, system, messages, tools, budget_s=budget_s, session=session)
        out.setdefault("runtime", getattr(b, "runtime", None))
        return out


def make_backend(cfg: dict[str, Any], role: str):
    """The brain for this machine's agent from its config: llm_enabled, llm_runtime, llm_url, openclaw_url,
    openclaw_agent (token: secrets/openclaw_token)."""
    if not cfg.get("llm_enabled"):
        return NoBackend()
    from . import config as _config
    from .llm import LLM
    ollama = OllamaBackend(LLM(cfg.get("llm_url") or "http://127.0.0.1:11434", cfg.get("llm_model"), label=role))
    if str(cfg.get("llm_runtime", "openclaw")).lower() != "openclaw":
        return ollama
    oc = OpenClawBackend(cfg.get("openclaw_url") or "http://127.0.0.1:18789", _config.secret("openclaw_token"),
                         cfg.get("openclaw_agent") or role, model_tag=cfg.get("llm_model") or "qwen3.8-27b-q3")
    return ChainBackend(oc, ollama)


class NoBackend:
    """No brain configured: every decision is today's rule (the decision log still says so)."""

    def status(self) -> dict[str, Any]:
        return {"ok": False, "runtime": None, "model": None, "error": "no LLM configured"}

    def decide(self, *a, **k):
        return None, {"model": None, "ms": None, "error": "no LLM configured"}

    def converse(self, *a, **k):
        return {"reply": None, "trace": [], "model": None, "ms": 0, "error": "no LLM configured"}
