#!/usr/bin/env python3
"""One market day on the new path: homes and the three buildings on one desk, two stand-in buyers with fixed orders, offline.
For building and debugging, never scored. Debug days only.

  FLEXGRID_DATASET=ma_year_t1 FG_DEVICES=hpwh,pool,dryer,dish,v2h FG_MARKET=2026 FG_BATTERY_MODE=backup \\
      python tools/market_day.py 2026-07-02 --homes 20 --case heat --out /tmp/day.json

--case       ordinary, heat, cloud, emergency, car_now, battery_trip, offline, closed, zone_event, note, sized as HW7_PLAN.md section 4 says
--rule       exception (nodes speak outside limits) or every_slot (as before)
--agent      the coordinator's margin through Qwen
--thinking   how many homes use Qwen, every 10th first
--thinking-buildings   which buildings use Qwen: a comma list, or all
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

R1 = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R1))
sys.path.insert(0, str(R1 / "tools"))
import sim_event  # noqa: E402
from flexgrid_core import config  # noqa: E402

BUILDINGS = ["bld-university", "bld-mall", "bld-office"]


def member_homes(n: int) -> list[str]:
    """Every 10th home first, the benchmark's thinking homes, as tools/node_day.py picks them."""
    fleet = json.loads(config.data_path("fleet_v2.json").read_text())["homes"]
    return ([h["name"] for h in fleet[9::10]] + [h["name"] for i, h in enumerate(fleet) if (i + 1) % 10])[:n]


def set_owners(homes: dict) -> None:
    own = json.load(open(config.data_path("owners.json")))["homes"]
    for nm, hv in homes.items():
        o = own.get(nm)
        if o is None:
            continue
        x = hv.adapter
        lim = x.ctl.limits
        lim.max_rise_c, lim.max_rise_away_c, lim.precool_c = o["max_rise_c"], o["max_rise_away_c"], o["precool_c"]
        x.P_ch = float(o["charge_kw"]) if o.get("charge_kw") else x.P_ch
        if "v2h" in (x.app or {}):
            x.app = {**x.app, "v2h": {**x.app["v2h"], "event_kwh": float(o["v2h_kwh"])}}


def fixed_orders(utility_kw: float, operator_kw: float, deposit_usd: float, start: str = "18:00", end: str = "21:00") -> list[dict]:
    """The stand-ins' orders, at the price book's price. The deposit is split in proportion to the kW."""
    from flexgrid_core import settlement
    price = float(settlement.price_book()["event_price_usd_kwh"])
    tot = max(utility_kw + operator_kw, 1e-6)
    return [{"to": "utility", "start": start, "end": end, "kw": utility_kw, "price_usd_per_kwh": price, "buyer": "utility stand-in",
             "note": "fixed order", "deposit_usd": round(deposit_usd * utility_kw / tot, 2)},
            {"to": "grid-operator", "start": start, "end": end, "kw": operator_kw, "price_usd_per_kwh": price, "buyer": "grid operator stand-in",
             "note": "fixed order", "deposit_usd": round(deposit_usd * operator_kw / tot, 2)}]


def case(name: str, day: str, homes: list[str], buildings: list[str], ask_kw: float, start: str = "18:00", end: str = "21:00") -> list[dict]:
    """One surprise, sized as HW7_PLAN.md section 4 words it. A framework hook, not the benchmark's injector."""
    quarter = homes[::4]
    at = lambda m: sim_event.beckn.add_minutes(f"{day} {start}", m)  # noqa: E731
    mid = at(60)
    if name == "ordinary":
        return []
    if name == "heat":
        return [{"kind": "heat", "at": f"{day} {start}", "hours": 3, "dT": 4.0}]
    if name == "cloud":
        return [{"kind": "cloud", "at": f"{day} 06:00", "hours": 15, "pv": 0.2}]
    if name == "emergency":
        return [{"kind": "emergency", "at": at(45), "kw": 0.5 * ask_kw, "minutes": 60, "why": "grid emergency", "buyer": "grid-operator"}]
    if name == "car_now":
        return [{"kind": "car_now", "at": mid, "homes": quarter, "leaves_in_min": 15}]
    if name == "battery_trip":
        return [{"kind": "battery_trip", "at": mid, "homes": quarter}]
    if name == "offline":
        return [{"kind": "offline", "at": mid, "homes": quarter + buildings[-1:], "minutes": 30}]
    if name == "closed":
        return [{"kind": "building_closed", "at": f"{day} 07:00", "homes": buildings[-1:]}]
    if name == "zone_event":
        return [{"kind": "zone_event", "at": at(-30), "homes": buildings[:1], "share": 0.25}]
    if name == "note":
        return [{"kind": "note", "at": at(-60), "homes": homes[:3], "text": "Going away tonight, need a full battery"}]
    raise SystemExit(f"no case {name}")


def thinkers(tier: int, homes_n: int, buildings: list[str]) -> tuple:
    """(attach, step, brain, decisions): Qwen for these nodes, as tools/node_day.py and tools/node_building_day.py run them."""
    from flexgrid_core.brain import HomeBrain, Wishes
    from flexgrid_core.node.agent import Brain, DecideBackend, NodeAgent
    from flexgrid_core.node.building_agent import BuildingAgent
    brain, backend, decisions, agents = Brain(), DecideBackend(), [], {}

    def attach(homes: dict, names: list[str]) -> None:
        for nm in names[:homes_n]:
            agents[nm] = NodeAgent(homes[nm], tier, brain, decisions.append)
            homes[nm].brain = HomeBrain(homes[nm], backend, Wishes(Path(config.CONFIG_DIR) / f"{nm}_wishes.json"), label=nm)
        for nm in buildings:
            agents[nm] = BuildingAgent(homes[nm], brain, decisions.append)

    def step(now, homes, market, util, traffic, log) -> None:
        for ag in agents.values():
            ag.step(now)
    return attach, step, brain, decisions


def run(day: str, homes_n: int = 20, buildings: list[str] | None = None, case_name: str = "ordinary", rule: str = "exception",
        utility_kw: float = 2500.0, operator_kw: float = 1500.0, deposit_usd: float = 2000.0, agent=None, on_step=None, on_homes=None,
        start: str = "18:00", end: str = "21:00", extra_surprises: list[dict] | None = None, surprises: list[dict] | None = None,
        orders: list[dict] | None = None, attach=None, forecast: str = "temp_regression") -> dict:
    names = member_homes(homes_n)
    b = BUILDINGS if buildings is None else buildings
    sp = surprises if surprises is not None else case(case_name, day, names, b, utility_kw + operator_kw, start, end) + (extra_surprises or [])

    def prepared(homes: dict) -> None:
        set_owners(homes)
        if attach:
            attach(homes, names)
        if on_homes:
            on_homes(homes)
    return sim_event.run_event(day, quiet=True, forecast=forecast, only=set(names), control_mode="rolling", on_homes=prepared,
                               buildings=b, orders=orders if orders is not None else fixed_orders(utility_kw, operator_kw, deposit_usd, start, end),
                               report_rule=rule, surprises=sp, market_agent=agent, on_step=on_step)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("day")
    p.add_argument("--homes", type=int, default=20)
    p.add_argument("--case", default="ordinary")
    p.add_argument("--rule", default="exception", choices=["exception", "every_slot"])
    p.add_argument("--utility-kw", type=float, default=2500.0)
    p.add_argument("--operator-kw", type=float, default=1500.0)
    p.add_argument("--deposit", type=float, default=2000.0)
    p.add_argument("--agent", action="store_true", help="the coordinator's margin through Qwen")
    p.add_argument("--tier", type=int, default=1, help="the dataset's tier, for the homes that think")
    p.add_argument("--thinking", type=int, default=0)
    p.add_argument("--thinking-buildings", default="")
    p.add_argument("--forecast", default="temp_regression", help="the nodes' forecast: temp_regression, or trained_2026 on a *b dataset")
    p.add_argument("--out", default="")
    a = p.parse_args()
    agent = None
    if a.agent:
        from flexgrid_core.coord_agent import make_agent
        agent = make_agent()
    who = BUILDINGS if a.thinking_buildings == "all" else [x for x in a.thinking_buildings.split(",") if x]
    attach = step = brain = None
    if a.thinking or who:
        attach, step, brain, _ = thinkers(a.tier, a.thinking, who)
    t0 = time.time()
    r = run(a.day, a.homes, None, a.case, a.rule, a.utility_kw, a.operator_kw, a.deposit, agent, on_step=step, attach=attach, forecast=a.forecast)
    keep = {k: r.get(k) for k in ("promised_kwh", "committed_kwh", "delivered_kwh_true", "delivered_kwh_measured", "utility_pays_usd", "error", "pool",
                                  "report_rule", "contracts", "desk")}
    print("wall", round(time.time() - t0), "s", "qwen calls", brain.calls if brain else 0)
    print(json.dumps(keep, indent=1, default=str)[:6000])
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
