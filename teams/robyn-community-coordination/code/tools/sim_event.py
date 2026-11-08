"""Offline end-to-end run of grid events with the real v2 code, before (or instead of) running them live.

The utility ("Utility or ENGIE" mock), the coordinator's market and all 100 homes run in one process with the same
classes the live agents use (Utility, Market, HomeV2, SimulatedHome). Two things are swapped out:
  * NANDA Town -> an in-memory queue (every message is still a signed Beckn body, checked on arrival)
  * the wall clock -> a virtual clock that moves one 15-minute dataset step per loop
so a whole event (notice the day before -> offers -> contract -> pre-charge -> delivery -> readings -> baselines ->
settlement -> payments) takes seconds and is repeatable.

What it measures, per event:
  committed vs delivered   measured delivery = sum over meters of clamp0(BASELINE - USAGE) (what the utility pays on)
                           true delivery     = counterfactual twin minus actual meter (what really changed)
  baseline accuracy        the utility's Ameren baseline vs the true no-event usage, summed over meters
  money                    what the utility pays the community, and how it is split between homes
  feeder                   the 100 homes' net import with and without the event, every 15 minutes of the day:
                           peak in the window, pre-charge before it, rebound after it (EV restarts)

  python tools/sim_event.py --event 2023-08-24
  python tools/sim_event.py --event all                 # the 10 events, each from a fresh start
  python tools/sim_event.py --event 2023-08-24 --opt-out 5 --out results/
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
os.environ["FLEXGRID_CONFIG_DIR"] = tempfile.mkdtemp(prefix="flexgrid-sim-")
if os.environ.get("FG_MARKET") == "2026":                     # a 2026 run that names no dataset reads the new data. Old folders stay, by name
    os.environ.setdefault("FLEXGRID_DATASET", "ma_year_t1b")
sys.path.insert(0, str(REPO))

from flexgrid_core import beckn, config, signing  # noqa: E402
from flexgrid_core import replay as replay_mod  # noqa: E402
from flexgrid_core import building as building_mod  # noqa: E402
from flexgrid_core.adapters.simulated import SimulatedHome  # noqa: E402
from flexgrid_core.home_v2 import HomeV2  # noqa: E402
from flexgrid_core.market import Market  # noqa: E402
from flexgrid_core.replay import Replay  # noqa: E402
from flexgrid_core.report_rule import PoolNode, QuietNode  # noqa: E402
from flexgrid_core.stand_in import StandInBuyer  # noqa: E402
from flexgrid_core.utility import Utility  # noqa: E402


class VirtualTime:
    """Stands in for the `time` module inside replay.py: the replay clocks read this instead of the wall clock."""
    def __init__(self, t0: float):
        self.t = t0

    def time(self) -> float:
        return self.t


LAST: dict = {}        # the last run's agents, for debugging in a REPL
STEP_S = 60.0          # one dataset step (15 min) per 60 virtual seconds, as on the live agents
T0 = 1_700_000_000.0   # virtual unix time at which the run starts


def kept_days() -> set[str]:
    """The days the benchmark keeps unseen, from its own list. On a machine without the kit there is none."""
    f = Path(__file__).resolve().parents[1] / "benchmark/days.json"
    if not f.exists():
        return set()
    sets = json.loads(f.read_text())["sets"]
    return set(sets["tuning"]) | set(sets["fresh"])


def refuse_kept(day: str) -> None:
    """A run plays the day before and on into the next morning, so all three must be clear. On 5 and 6 Nov runs passed through two kept days this way."""
    from datetime import date, timedelta
    hit = {str(date.fromisoformat(day) + timedelta(days=k)) for k in (-1, 0, 1)} & kept_days()
    if hit:
        raise SystemExit(f"refused: a run of {day} plays {', '.join(sorted(hit))}, kept for the benchmark. Only the benchmark kit runs those")


def run_event(event_day: str, opt_out: int = 0, quiet: bool = True, tolerance: float = 0.0,
              forecast: str = "temp_regression", only: set[str] | None = None, on_step=None,
              on_message=None, baseline: str = "ptr", order: dict | None = None, agents: dict | None = None,
              on_homes=None, control_mode: str = "fixed", owner_limits: dict | None = None,
              surprises: list[dict] | None = None, buildings: list[str] | None = None, orders: list[dict] | None = None,
              report_rule: str = "every_slot", market_agent=None, kept_ok: bool = False) -> dict:
    """order: a buyer agent's order for this day ({start, end, kw, price_usd_per_kwh, buyer, note}); it is placed with
    the utility before the run starts (POST /needs on the live utility) and replaces the day's catalog need.
    agents: what the AI agents decided in a live run of this event (flexgrid_core/verify.py replay_agents: each
    agent-home's offer decision, the coordinator's margin, the homes that declined, the conditions the homes declared
    in their offers: what-if weather, a battery out of service), played the same way.
    on_homes(homes): called once the homes are built, before the first step (tests: a what-if at the start).
    surprises: what happens during the day that nobody planned for (the test bench; the audience's controls live),
    each {"kind", "at": "YYYY-MM-DD HH:MM", ...}:
      emergency    {kw, minutes, why}        the grid operator tells the buyer; the buyer calls the community
      price_spike  {factor, hours}           the real-time price times factor (the buyer's price rule may call)
      weather      {hours, pv, dT}           solar times pv and dT degrees C warmer, for every home (a storm, heat)
      car_now      {homes, leaves_in_min}    "I need the car now": full-power charging from that minute, and it leaves after those minutes
      depart_at    {homes, minutes_early}    the car leaves earlier than its owner said
      comfort      {homes, add_c}            the owner allows add_c more degrees (by chat)
      stop         {homes}                   the owner leaves the event
      battery_trip {homes}                   the battery's inverter trips (0 kW)
      offline      {homes, minutes}          the home's agent cannot talk (its device keeps its plan)
      lights       {homes, share}            a building's facility manager: the share of lights that may be dimmed
      note         {homes, text}             the owner writes a note to the home's agent
      heat         {hours, dT}               hotter than forecast, for every node: each load rises by its own kW per degree
      cloud        {hours, pv}               solar times pv, for every node
      building_closed {homes, share}         a building closes for the day: its load falls to `share` of normal and it leaves the sale
      zone_event   {homes, share, hours}     an event in a sold zone: that share of the building can't be dimmed or eased
      alarm        {homes, why}              a fire, smoke or test signal: on the new path the node leaves the sale at once
      emergency also takes {buyer}: "utility" or "grid-operator", on the orders path
    buildings: building nodes that take part too (data_ma/buildings.json, e.g. ["bld-library"]): they offer, are
    contracted, deliver and are paid like a home; the feeder's homes-only columns keep the community's peaks
    comparable, and the scorecard scores each building on its own.
    orders: fixed orders, each {to: "utility" | "grid-operator", start, end, kw, price_usd_per_kwh, buyer, note, deposit_usd}. This is the
    new path: two stand-in buyers (stand_in.py), and orders on one window fill from one pool (market.py). None: the old one-buyer path.
    report_rule: "every_slot", as before, or "exception": nodes speak only outside expected limits (report_rule.py).
    market_agent: the coordinator's agent (coord_agent.py). None: the code's default margin."""
    if not kept_ok:                                   # only the kit's door says this, after its own checks
        refuse_kept(event_day)
    vt = VirtualTime(T0)
    replay_mod.time = vt                                       # only replay.py reads the clock
    start = beckn.add_minutes(f"{event_day} 00:00", -1440)     # one day of warm-up; notice comes at 10:00
    stop = beckn.add_minutes(f"{event_day} 23:45", 8 * 60)     # through the 8 h evening the guard checks

    fleet = json.loads(config.data_path("fleet_v2.json").read_text())["homes"]
    if only:                                                   # replay a live run in which only these homes got the call
        fleet = [h for h in fleet if h["name"] in only]
    names = [h["name"] for h in fleet]
    opted = set(names[2::max(1, len(names) // max(1, opt_out))][:opt_out]) if opt_out else set()
    bnodes = building_mod.enabled({"buildings": buildings or []})
    names += [b["name"] for b in bnodes]
    quiet_path = report_rule == "exception"
    cfg = {**config.load(), "replay_start": start, "replay_step_seconds": STEP_S, "replay_loop_days": 0,
           "rebound_tolerance": tolerance, "sync_broadcast": True, "baseline_method": baseline,
           "buyers": ["utility", "grid-operator"] if orders is not None else None, "report_rule": report_rule,
           "call_settle_after_min": 120 if quiet_path else 60}   # the quiet path's readings come after the window
    config.save({**cfg, "agent_name": "coordinator", "role": "coordinator"})

    bus: deque = deque()
    log: list[dict] = []
    traffic = {"messages": 0, "by_kind": {}}
    spoke: set[tuple[str, str]] = set()                         # (node, step) pairs in which a node sent anything

    def clock(profile: str) -> Replay:
        r = Replay(profile, start, STEP_S, 0)
        r.configure(start, STEP_S, 0, at=start, at_wall=T0)
        return r

    def logger(who: str):
        def f(kind: str, **d):
            log.append({"sim": coord_clock.ts(coord_clock.current()), "who": who, "kind": kind, **d})
        return f

    # coordinator (market)
    ckey = signing.load_or_create(config.CONFIG_DIR / "secrets" / "signing.key")
    coord_clock = clock(cfg["clock_profile"])
    market = Market("coordinator", ckey, lambda to, k, b, mid: bus.append(("coordinator", to, k, b)), coord_clock,
                    names, {}, lambda: None, logger("coordinator"), cfg)

    # the buyers: the utility alone, or the two stand-ins with their fixed orders
    util = Utility() if orders is None else StandInBuyer("utility")
    buyers = {"utility": util} if orders is None else {"utility": util, "grid-operator": StandInBuyer("grid-operator")}

    class _Town:
        def __init__(self, name):
            self.name = name

        def send(self, to, kind, body, mid):
            bus.append((self.name, to, kind, body))

    def wire_buyer(nm, u):
        u.clock = clock(cfg["clock_profile"])
        u.fleet.update({b["name"]: building_mod.meter_entry(b) for b in bnodes})   # their meters' history
        u.town = _Town(nm)
        u.event = lambda kind, **d: log.append({"sim": u.now(), "who": nm, "kind": kind, **d})
    for nm, u in buyers.items():
        wire_buyer(nm, u)
    if order:
        util.place_order({**{k: v for k, v in order.items() if k not in ("date", "at")}, "date": event_day},
                         at=order.get("at"))                       # the buyer's own clock, for the record
    for o in orders or []:
        buyers[o.get("to", "utility")].place_order({**{k: v for k, v in o.items() if k not in ("date", "at", "to")}, "date": event_day},
                                                   at=o.get("at"))

    # homes: the two hardware homes with their datasheet batteries, the 98 replay homes from the dataset
    homes: dict[str, HomeV2] = {}
    Node = QuietNode if quiet_path else HomeV2 if orders is None else PoolNode
    cpub = signing.public_hex(ckey)
    keydir = config.CONFIG_DIR / "keys"
    keydir.mkdir(parents=True, exist_ok=True)
    for h in fleet:
        rp = clock(h["profile"])
        b = h["battery"]
        acfg = {"battery_kwh": b["kwh"], "battery_kw": b["kw"], "reserve_pct": b["reserve_pct"],
                "soc_max_pct": b["soc_max_pct"], "soc0_pct": b.get("soc0_pct", 55.0), "eta": b["eta"],
                "ev_max_kw": rp.sessions[0]["max_kw"] if rp.sessions else 0.0, "has_ev": h["has_ev"],
                "export_limit_kw": h["export_limit_kw"], "control": "auto", "forecast": forecast,
                "opt_out_days": [event_day] if h["name"] in opted else [],
                "name": h["name"], "control_mode": control_mode, "owner_limits": owner_limits or {}}
        sim = SimulatedHome(acfg, rp)
        key = signing.load_or_create(keydir / f"{h['name']}.key")
        homes[h["name"]] = Node(h["name"], acfg, sim, rp, key,
                                send=lambda to, k, body, mid, n=h["name"]: bus.append((n, to, k, body)),
                                trusted=lambda: [cpub], log=logger(h["name"]))

    for b in bnodes:              # the building nodes: the same protocol code, a building's levers (building.py)
        rp = clock(b["profile"])
        acfg = {"agent_name": b["name"], "name": b["name"], "building": b["name"], "control": "auto",
                "forecast": forecast, "control_mode": "rolling", "opt_out_days": []}
        sim = building_mod.SimulatedBuilding(acfg, rp)
        key = signing.load_or_create(keydir / f"{b['name']}.key")
        homes[b["name"]] = Node(b["name"], acfg, sim, rp, key,
                                send=lambda to, k, body, mid, n=b["name"]: bus.append((n, to, k, body)),
                                trusted=lambda: [cpub], log=logger(b["name"]))

    if agents:                    # the replay of a live run with AI agents: what they decided, through the same code
        from flexgrid_core.brain import ReplayBrain, ReplayMargin
        for n, cnd in (agents.get("conditions") or {}).items():          # the what-if weather each home declared
            if n in homes and isinstance(cnd.get("weather"), dict):     # live, from the time it began
                homes[n].replay.weather = dict(cnd["weather"])
            if n in homes and cnd.get("battery_out"):                   # its battery out of service when it offered
                homes[n].adapter.set_battery_fault(True)
        eid = agents.get("event_id") or f"flex-need-{event_day}"
        if agents.get("margin") is not None:
            market.brain = ReplayMargin({eid: float(agents["margin"])}, market.margin)
        offs, decl = agents.get("offers") or {}, set(agents.get("declines") or ())
        for n in set(offs) | decl:
            if n in homes:
                homes[n].brain = ReplayBrain({eid: offs[n]} if n in offs else {}, {eid} if n in decl else set())

    if market_agent is not None:
        market.brain = market_agent
    for h in homes.values():                                   # the zone maps outlive a run
        zone = _zones(h) if h.is_building else None
        if zone:
            zone.surprises.clear()

    if on_homes:
        on_homes(homes)
    if orders is not None:                                     # the coordinator enrolled each node's nameplate, not its word
        market.rated = {n: rated_kw(h.adapter) for n, h in homes.items()}

    todo = sorted(surprises or [], key=lambda x: x["at"])
    offline: dict[str, str] = {}                               # home -> when it is back
    held: list = []                                            # messages to or from a home while it is away
    for sp in todo:
        k, at = sp["kind"], sp["at"]
        if k == "emergency":
            buyers[sp.get("buyer", "utility")].watch.alert(float(sp["kw"]), float(sp.get("minutes", 60)), sp.get("why", "grid emergency"), at=at)
            sp["done"] = True
        elif k in ("heat", "cloud"):
            end = beckn.add_minutes(at, 60 * float(sp.get("hours", 3)))
            for h in homes.values():                           # forecasts made before `at` do not know it
                h.replay.weather = {**(h.replay.weather or {}), **_case_weather(h, k, at, end, sp)}
            sp["done"] = True
        elif k == "price_spike":
            util.clock.weather = {"from": None, "pv": 1.0, "load": 1.0, "price": float(sp.get("factor", 3.0)),
                                  "price_from": at, "price_to": beckn.add_minutes(at, 60 * float(sp.get("hours", 1)))}
            sp["done"] = True
        elif k == "weather":
            w = {"from": at, "to": beckn.add_minutes(at, 60 * float(sp.get("hours", 2))), "pv": float(sp.get("pv", 1.0)),
                 "load": 1.0 + 0.04 * float(sp.get("dT", 0.0)), "dT": float(sp.get("dT", 0.0))}
            for h in homes.values():                           # forecasts made before `at` do not know it
                h.replay.weather = dict(w)
            sp["done"] = True

    def surprise(now: str) -> None:
        for sp in todo:
            if sp.get("done") or sp["at"] > now:
                continue
            sp["done"] = True
            for nm in sp.get("homes") or []:
                if nm not in homes:
                    continue
                a = homes[nm].adapter
                k = sp["kind"]
                if k == "car_now":
                    a.rt_override("ev_now", True)
                    if sp.get("leaves_in_min") is not None:             # the owner takes it: gone from the plug, whatever it had
                        a.rt_override("depart_at", beckn.add_minutes(now, float(sp["leaves_in_min"])))
                elif k == "depart_at" and a.ev_deadline:
                    a.rt_override("depart_at", beckn.add_minutes(a.ev_deadline, -float(sp.get("minutes_early", 60))))
                elif k == "comfort":
                    add = float(sp.get("add_c", 1.0))
                    a.rt_override("comfort", {"max_rise_c": 2.0 + add, "max_rise_away_c": 4.0 + add})
                elif k == "stop":
                    a.rt_override("stop", True)
                elif k == "battery_trip":
                    a.set_battery_fault(True)
                elif k == "offline":
                    offline[nm] = beckn.add_minutes(sp["at"], float(sp.get("minutes", 30)))
                elif k == "lights":
                    a.rt_override("lights", float(sp.get("share", 0.0)))
                elif k == "note":                                       # the owner writes to the agent; rules can't read it
                    a.rt_override("note", sp.get("text", ""))
                elif k == "building_closed":
                    _close_building(homes[nm], sp, now)
                elif k == "zone_event":
                    _zone_event(homes[nm], sp, now)
                elif k == "alarm":
                    a.rt_override("alarm", sp.get("why", "fire alarm"))
                log.append({"sim": now, "who": nm, "kind": "surprise", "surprise": k})

    def deliver() -> None:
        now = coord_clock.ts(coord_clock.current())
        for nm in [x for x, back in offline.items() if now >= back]:
            offline.pop(nm)                                    # back: what waited goes out (and in) now
            bus.extend(m for m in held if nm in (m[0], m[1]))
            held[:] = [m for m in held if nm not in (m[0], m[1])]
        while bus:
            frm, to, kind, body = bus.popleft()
            if frm in offline or to in offline:
                held.append((frm, to, kind, body))
                continue
            if on_message:                                      # tools/agent_talk.py prints the conversation
                on_message(coord_clock.ts(coord_clock.current()), frm, to, kind, body)
            traffic["messages"] += 1
            traffic["by_kind"][kind] = traffic["by_kind"].get(kind, 0) + 1
            if frm in homes:
                spoke.add((frm, now))
            if to == "coordinator":
                market.handle(kind, frm, body)
            elif to in buyers:
                # a node's signed readings reach every buyer of the pool: each settles its own share on the same meters
                m = body.get("message") or {}
                copies = buyers.values() if frm in homes and ("performance" in m or "batch" in m) else [buyers[to]]
                for u in copies:
                    u.handle(kind, frm, body)
                extra = len(copies) - 1                         # a copy is a message too
                traffic["messages"] += extra
                traffic["by_kind"][kind] = traffic["by_kind"].get(kind, 0) + extra
            elif to in homes:
                homes[to].handle(kind, frm, body)

    # feeder series: every home's actual and no-event meter, every step of the event day
    feeder: dict[str, dict[str, float]] = {}
    t_start = time.time()
    while True:
        now = coord_clock.ts(coord_clock.current())
        surprise(now)
        for u in buyers.values():
            u.watch.tick()                                     # the buyer's calls first: homes plan this step with them
        deliver()
        for h in homes.values():
            h.tick()
        deliver()
        market.tick()
        deliver()
        for u in buyers.values():
            u.tick()
        deliver()
        if on_step:                                            # tools/timeline.py records every step for the monitor
            on_step(now, homes, market, util, traffic, log)
        if now >= f"{event_day} 00:00":
            row = feeder.setdefault(now, {"import_kw": 0.0, "import_cf_kw": 0.0, "pv_kw": 0.0, "ev_kw": 0.0,
                                          "batt_kw": 0.0, "ac_extra_kw": 0.0, "import_homes_kw": 0.0,
                                          "import_cf_homes_kw": 0.0})
            for h in homes.values():
                t = h.adapter.tel
                row["import_kw"] += -t["grid_kw"]
                row["import_cf_kw"] += -t["grid_cf_kw"]
                row["pv_kw"] += t["pv_kw"]
                row["ev_kw"] += t["ev_kw"]
                row["batt_kw"] += t.get("batt_kw", 0.0)                 # + discharging
                row["ac_extra_kw"] += t.get("cool_kw", 0.0) - t.get("cool_own_day_kw", 0.0) if "cool_kw" in t else 0.0
                if t.get("node") == "building":                         # a building on its own columns: its 1 MW
                    for k2, v in (("bld_import_kw", -t["grid_kw"]), ("bld_import_cf_kw", -t["grid_cf_kw"]),
                                  ("bld_light_dim_kw", t.get("light_dim_kw", 0.0)),
                                  ("bld_cool_delta_kw", t.get("cool_kw", 0.0) - t.get("cool_own_day_kw", 0.0))):
                        row[k2] = row.get(k2, 0.0) + v
                else:                                                   # would hide the homes' evening peak
                    row["import_homes_kw"] += -t["grid_kw"]
                    row["import_cf_homes_kw"] += -t["grid_cf_kw"]
        if now >= stop:
            break
        vt.t += STEP_S

    LAST.update(homes=homes, market=market, utility=util, feeder=feeder, log=log, buyers=buyers)
    sys.path.insert(0, str(REPO / "tools"))
    from scorecard import score
    seen = util if orders is None else SimpleNamespace(clock=util.clock, watch=SimpleNamespace(
        calls={k: v for u in buyers.values() for k, v in u.watch.calls.items()}))      # both buyers' calls on the card
    card = score(event_day, stop, homes, market, seen, feeder, traffic, log)
    ev = next(iter(market.events.values()), None)
    if not ev or "result" not in ev:
        return {"event_day": event_day, "error": "event did not settle", "phase": ev and ev["phase"],
                "log_tail": log[-15:]}
    n = ev["need"]
    res = ev["result"]
    # baseline accuracy: utility baseline vs true no-event usage, summed over meters, per slot
    base_vs_true = []
    for s in n["slots"]:
        sid = str(s["id"])
        b = sum(m[sid]["BASELINE"] for m in ev["baselines"].values() if sid in m)
        t = sum(r[s["id"]]["truth"]["usage_without_event_kw"] for r in ev["readings"].values() if s["id"] in r)
        u = sum(r[s["id"]]["usage_kw"] for r in ev["readings"].values() if s["id"] in r)
        base_vs_true.append({"slot": s["id"], "start": s["start"], "baseline_kw": round(b, 1),
                             "true_no_event_kw": round(t, 1), "actual_kw": round(u, 1)})
    rows = sorted(feeder.items())
    in_win = [(ts, r) for ts, r in rows if n["window_start"] <= ts < n["window_end"]]
    pre = [(ts, r) for ts, r in rows if beckn.add_minutes(n["window_start"], -120) <= ts < n["window_start"]]
    post = [(ts, r) for ts, r in rows if n["window_end"] <= ts < beckn.add_minutes(n["window_end"], 180)]

    def kwh(sel, k):
        return round(sum(r[k] for _, r in sel) * 0.25, 1)

    paid = sorted(ev["result"]["split"]["per_home"].values())
    cs = ev["commitments"].values()
    out = {
        "event_id": n["event_id"], "window": [n["window_start"], n["window_end"]],
        "price_usd_per_kwh": n["slots"][0]["price"], "requested_kw": n["slots"][0]["kw_requested"],
        "homes": len(homes), "opted_out": sorted(opted),
        "offers": len(ev["offers"]), "offered_kw_raw": ev["clear_summary"]["offered_kw_raw"],
        "capacity_offered_kw": ev.get("capacity_offered_final"),
        "contracts": {"sent": len(ev["commitments"]), "accepted": sum(1 for c in cs if c["status"] == "ACCEPTED"),
                      "declined": sum(1 for c in cs if c["status"] == "DECLINED")},
        "committed_kwh": round(sum(x["kw"] for c in cs if c["status"] == "ACCEPTED"
                                   for x in c["contract"]["commitments"]) * 0.5, 1),
        "promised_kwh": round(sum((ev.get("capacity_offered_final") or {}).values()) * 0.5, 1),
        "delivered_kwh_measured": res["delivered_kwh_measured"], "delivered_kwh_true": res["delivered_kwh_true"],
        "utility_pays_usd": res["seller_usd"], "net_zero_ok": res["net_zero_ok"],
        "per_slot": res["components"], "baseline_vs_truth": base_vs_true,
        "payments_usd": {"homes_paid": len(paid), "min": paid[0] if paid else 0, "median": paid[len(paid) // 2] if paid else 0,
                         "max": paid[-1] if paid else 0},
        "feeder": {"peak_window_kw": round(max(r["import_homes_kw"] for _, r in in_win), 1),   # the homes'
                   "peak_window_no_event_kw": round(max(r["import_cf_homes_kw"] for _, r in in_win), 1),   # feeder
                   "window_kwh": kwh(in_win, "import_homes_kw"), "window_kwh_no_event": kwh(in_win, "import_cf_homes_kw"),
                   "pre_2h_kwh": kwh(pre, "import_homes_kw"), "pre_2h_kwh_no_event": kwh(pre, "import_cf_homes_kw"),
                   "post_3h_kwh": kwh(post, "import_homes_kw"), "post_3h_kwh_no_event": kwh(post, "import_cf_homes_kw"),
                   "post_3h_peak_kw": round(max(r["import_homes_kw"] for _, r in post), 1),
                   "post_3h_peak_no_event_kw": round(max(r["import_cf_homes_kw"] for _, r in post), 1),
                   "day_peak_kw": round(max(r["import_homes_kw"] for _, r in rows), 1),          # event day 00:00 to
                   "day_peak_no_event_kw": round(max(r["import_cf_homes_kw"] for _, r in rows), 1)},  # 07:45 next day
        "clearing": ev["clear_summary"],
        "rebound_check": _rebound_check(ev, feeder),
        "series": [{"ts": ts, **{k: round(v, 2) for k, v in r.items()}} for ts, r in rows],
        "messages": traffic, "bad_signatures": sum(1 for x in log if x["kind"] == "v2_bad_signature"),
        "rejected_readings": ev["settlement"].get("rejected_readings"),
        "runtime_s": round(time.time() - t_start, 1),
        "control_mode": control_mode, "surprises": [{k: v for k, v in x.items() if k != "done"} for x in todo],
        "scorecard": card, "calls": util.watch.summary(), "desk": market.rt.summary(),
    }
    if orders is not None:
        half: dict[str, float] = {}                            # the twin's measure: for the scorer, never for a decision
        for ts, r in in_win:
            k = ts[:14] + ("00" if ts[14:16] < "30" else "30")
            half[k] = half.get(k, 0.0) + (r["import_cf_kw"] - r["import_kw"]) * 0.25
        out["pool"] = {"orders": ev["clear_summary"].get("orders"), "pooled_offer": ev["clear_summary"].get("pooled_offer"),
                       "by_buyer": res.get("by_buyer"), "outlook": ev.get("outlook"),
                       "true_kwh_by_buyer": {b: round(sum(half.get(s["start"], 0.0) * od["share"].get(s["id"], 0.0) for s in n["slots"]), 1)
                                             for b, od in ev["orders"].items()},
                       "calls": {nm: u.watch.summary() for nm, u in buyers.items()}}
    day_steps = {t for _, t in spoke if t[:10] == event_day}
    busy = sum(1 for _, t in spoke if t[:10] == event_day)
    out["report_rule"] = {"rule": report_rule, "messages": traffic["messages"], "nodes": len(homes),
                          "messages_per_true_kwh": round(traffic["messages"] / max(res["delivered_kwh_true"], 1e-6), 3),
                          "messages_per_node": round(traffic["messages"] / max(len(homes), 1), 2),
                          # of a node's 288 five-minute steps in the day, the share with nothing sent
                          "silent_share": round(1 - busy / max(len(homes) * 288, 1), 4), "steps_with_a_message": len(day_steps)}
    if not quiet:
        print(json.dumps({k: v for k, v in out.items() if k not in ("series", "per_slot", "baseline_vs_truth", "rebound_check")},
                         indent=1))
    return out


def rated_kw(a) -> float:
    """A node's nameplate: all it could ever shed or give at once. What the coordinator checks an offer against."""
    kw = float(a.P) + (float(a.ev_max) if a.has_ev else 0.0)
    if a.house is not None:
        kw += float(a.house.qmax) / 2.0                       # kW of heat removed, at a low COP
    for k in (a.app or {}):
        kw += float(a.dev_arr[k][a.dev_row].max())
    light = getattr(a.house, "light", None)
    if light:
        kw += max(light.values())
    return round(kw, 1)


def _case_weather(h, kind: str, at: str, end: str, sp: dict) -> dict:
    """heat: the node's own kW per degree, as a factor on its load over those hours. cloud: solar only."""
    if kind == "cloud":
        return {"from": at, "to": end, "pv": float(sp.get("pv", 0.2)), "load": 1.0}
    rp, dT = h.replay, float(sp.get("dT", 4.0))
    rows = [rp.rows[i] for i in range(rp.idx_of(at), rp.idx_of(end))]
    base = sum(r["base_kw"] for r in rows) / len(rows)
    per_c = sum(r.get("dkw_dT", 0.04 * r["base_kw"]) for r in rows) / len(rows)
    return {"from": at, "to": end, "pv": 1.0, "load": 1.0 + per_c * dT / max(base, 0.1), "dT": dT}


def _zones(h):
    """The building's zone map, for what its sensors read. None for a node without one."""
    from flexgrid_core.node import buildings as zones
    return zones.load(h.name) if h.name in zones.KEYS else None


def _close_building(h, sp: dict, now: str) -> None:
    """Closed for the day: its true load falls, the no-event twin's with it, so nothing it does from here counts as relief. It leaves the
    sale. Its report and its agent's answer are theirs to make."""
    a, rp = h.adapter, h.replay
    i = rp.idx_of(now)
    night = sum(rp.rows[j]["base_kw"] for j in range(rp.idx_of(f"{now[:10]} 02:00"), rp.idx_of(f"{now[:10]} 04:00"))) / 8
    share = float(sp["share"]) if sp.get("share") is not None else min(1.0, night / max(rp.rows[i]["base_kw"], 1.0))
    h.replay.weather = {**(h.replay.weather or {}), "from": now, "to": beckn.add_minutes(f"{now[:10]} 00:00", 1440), "pv": 1.0, "load": share}
    a.rt_override("closed", {"from": now, "share": share})
    a.rt_override("stop", True)
    zone = _zones(h)
    if zone:
        zone.surprises.append({"from": now, "to": beckn.add_minutes(f"{now[:10]} 00:00", 1440), "space": None, "share": 0.0})


def _zone_event(h, sp: dict, now: str) -> None:
    """A zone already sold must run as normal from now on: that share of the lights can't dim and that share of the comfort room is gone.
    The controller finds what's left, and the node reports the rest."""
    a, z = h.adapter, float(sp.get("share", 0.25))
    end = beckn.add_minutes(now, 60 * float(sp.get("hours", 3.5)))
    a.rt_override("zone_event", {"from": now, "to": end, "share": z, "space": sp.get("space")})
    zone = _zones(h)
    if zone:                                                   # the sensors read the room in use: the building's agent sees it itself
        s = sp.get("space") or max((k for k, v in zone.spaces.items() if not v.get("closed")), key=lambda k: zone.spaces[k].get("area", 0.0))
        zone.surprises.append({"from": now, "to": end, "space": s, "share": 1.0})
    if hasattr(a, "dim_max"):
        a.rt_override("lights", a.dim_max() * (1 - z))
    if hasattr(a, "sell"):                                     # its fans and plugs too: the sale's part of them shrinks with the free space
        a.sell = {**a.sell, **{k: float(a.sell.get(k, a.app[k].get("share", 1.0))) * (1 - z) for k in a.app if k != "ev"}}
    lim = a.ctl.limits
    a.rt_override("comfort", {"max_rise_c": lim.max_rise_c * (1 - z), "max_rise_away_c": lim.max_rise_away_c * (1 - z)})


def _rebound_check(ev: dict, feeder: dict) -> dict:
    """The coordinator's evening forecast (made the day before) against what the meters then showed."""
    g = ev.get("guard") or {}
    if "with_event_kw" not in g:
        return {}
    t0 = ev["need"]["window_end"]
    rows = []
    for k, (fe, fn) in enumerate(zip(g["with_event_kw"], g["no_event_kw"])):
        ts = beckn.add_minutes(t0, 15 * k)
        r = feeder.get(ts)
        if r:
            rows.append({"ts": ts, "forecast_kw": fe, "actual_kw": round(r.get("import_homes_kw", r["import_kw"]), 1),
                         "forecast_no_event_kw": fn,
                         "actual_no_event_kw": round(r.get("import_cf_homes_kw", r["import_cf_kw"]), 1)})
    mae = lambda a, b: round(sum(abs(r[a] - r[b]) for r in rows) / max(1, len(rows)), 1)  # noqa: E731
    return {"limit_kw": g.get("limit_kw"), "battery_factor": g.get("battery_factor"), "met_in_forecast": g.get("met"),
            "peak_forecast_kw": g.get("peak_forecast_kw"), "peak_without_guard_kw": g.get("peak_without_guard_kw"),
            "peak_actual_kw": max((r["actual_kw"] for r in rows), default=None),
            "peak_actual_no_event_kw": max((r["actual_no_event_kw"] for r in rows), default=None),
            "mae_with_event_kw": mae("forecast_kw", "actual_kw"),
            "mae_no_event_kw": mae("forecast_no_event_kw", "actual_no_event_kw"),
            "ev_restarts": sorted(g.get("ev_restarts", {}).values()), "steps": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default=config.dataset_meta()["default_event"], help="event day (YYYY-MM-DD) or 'all'")
    ap.add_argument("--opt-out", type=int, default=0, help="homes that opt out of the event (tests the top-up)")
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="how far the evening peak may rise above the no-event forecast (0 = not at all; 9 = guard off)")
    ap.add_argument("--forecast", default=config.load().get("forecast") or "temp_regression",
                    choices=["temp_regression", "naive", "trained", "trained_2026"],
                    help="the homes' forecast (default: the dataset's, trained models for Massachusetts 2025-26)")
    ap.add_argument("--only", default="", help="comma-separated home names: only these homes get the call "
                    "(to replay a live run in which some homes were not reached)")
    ap.add_argument("--baseline", default="ptr", choices=["ptr", "weather_regression"],
                    help="the utility's settlement baseline: Ameren Rider PTR (3 of 14) or weather-normalized regression")
    ap.add_argument("--buildings", default="", help="comma-separated building nodes that take part too "
                    "(data_ma/buildings.json, e.g. bld-library); they run the 5-minute plan")
    ap.add_argument("--control-mode", default="fixed", choices=["fixed", "rolling"], help="the homes' plan")
    ap.add_argument("--tag", default="", help="suffix for the output file name")
    ap.add_argument("--out", default=str(REPO / "results"))
    ap.add_argument("--order", default="", help="a buyer agent's order JSON for the day (start, end, kw, "
                    "price_usd_per_kwh, buyer): it replaces the catalog need, as POST /needs does live")
    ap.add_argument("--agents", default="", help="what the AI agents decided in a live run of this event (JSON: "
                    "event_id, margin, offers {home: {take_part, battery_share, pause_ev}}, declines [home]): "
                    "played the same way (flexgrid_core/verify.py replay_agents)")
    a = ap.parse_args()
    order = json.loads(a.order) if a.order else None
    agents = json.loads(a.agents) if a.agents else None
    only = {x.strip() for x in a.only.split(",") if x.strip()} or None
    cat = json.loads(config.events_file().read_text())["message"]["catalogs"][0]
    offers = {o["resourceIds"][0]: o for o in cat["offers"]}
    days = sorted(beckn.need_from_catalog(r, offers[r["id"]])["window_start"][:10] for r in cat["resources"])
    todo = days if a.event == "all" else [a.event]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    table = []
    for d in todo:
        r = run_event(d, a.opt_out, quiet=len(todo) > 1, tolerance=a.tolerance, forecast=a.forecast, only=only,
                      baseline=a.baseline, order=order, agents=agents, control_mode=a.control_mode,
                      buildings=[x.strip() for x in a.buildings.split(",") if x.strip()] or None)
        tag = ("" if a.tolerance == 0 else f"_tol{a.tolerance:g}") + {"temp_regression": "", "naive": "_naive", "trained": "_trained",
                                                                    "trained_2026": "_trained_2026"}[a.forecast]
        tag += "" if a.baseline == "ptr" else "_regbase"
        tag += f"_{a.tag}" if a.tag else ""
        (Path(a.out) / f"sim_{d}{tag}.json").write_text(json.dumps(r, indent=1))
        if "error" in r:
            print(d, r["error"], r.get("phase"))
            continue
        f, rc = r["feeder"], r["rebound_check"]
        table.append({"event": d, "window": r["window"][0][11:] + "-" + r["window"][1][11:],
                      "accepted": r["contracts"]["accepted"], "promised_kwh": r["promised_kwh"],
                      "measured_kwh": r["delivered_kwh_measured"], "true_kwh": r["delivered_kwh_true"],
                      "paid_usd": r["utility_pays_usd"],
                      "window_peak_kw": f["peak_window_kw"], "window_peak_no_event_kw": f["peak_window_no_event_kw"],
                      "battery_factor": rc.get("battery_factor"), "evening_limit_kw": rc.get("limit_kw"),
                      "evening_peak_kw": rc.get("peak_actual_kw"), "evening_peak_no_event_kw": rc.get("peak_actual_no_event_kw"),
                      "forecast_mae_kw": rc.get("mae_no_event_kw")})
    if len(table) > 1:
        cols = list(table[0])
        print("\t".join(cols))
        for t in table:
            print("\t".join(str(t[c]) for c in cols))
        (Path(a.out) / f"sim_all_events{tag}.json").write_text(json.dumps(table, indent=1))


if __name__ == "__main__":
    main()
