"""flexgrid-mcp - FlexGrid's front door for agents: the MCP tools a buyer agent (a utility's or ENGIE's demand-response
desk, in any harness: Hermes, OpenClaw, Claude Code, ...) uses to buy a neighborhood's flexibility.

The buyer agent never talks to homes. It places an order with its utility (POST /needs, published as a DEG
DemandFlexNeed + DemandFlexBuyOffer); at 10:00 the day before, the community's coordinator discovers it, 100 home
agents offer and commit over NANDA Town in signed Beckn messages, their signed meter readings settle it, and the agent
gets the result back, checks it (flexgrid_core/verify.py) and signs off.

  list_communities()                                       who sells flexibility, its size, the clock, the deadline
  get_grid_outlook()                                       today's real MISO grid: fuel mix, prices, the peak window
  forecast_community_demand(date)                          the trained model's hourly 10/50/90 % demand forecast
  request_flexibility(date, start, end, kw, price_usd_per_kwh, note)    place the order
  get_contract(event_id, wait_seconds)                     where the deal stands, step by step
  get_settlement(event_id, wait_seconds)                   kW contracted / delivered per slot, payment, split
  verify_settlement(event_id)                              the independent check, PASS/FAIL per check
  sign_off(event_id, approve, reason)                      the buyer's decision, signed and kept by the utility
                                                           (approve only after a passing verify_settlement)
Backends
  live     the running agents: coordinator FLEXGRID_COORDINATOR_URL (http://127.0.0.1:7871), utility
           FLEXGRID_UTILITY_URL (http://127.0.0.1:7873); the event plays out on the shared replay clock
  offline  the same classes in one process (tools/sim_event.py: in-memory NANDA Town, virtual clock); the whole event
           runs when the order is placed (a few seconds). Every result names the backend that produced it.

  flexgrid-mcp                                             # live, stdio
  flexgrid-mcp --offline --today "2023-08-23 08:00"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from . import beckn, community_forecast, config, verify

INSTRUCTIONS = (
    "FlexGrid sells a neighborhood's demand flexibility (home batteries and EV charging that can be shifted) to a "
    "utility. You act for the buyer. Orders are for a day's evening window in 30-minute slots and must be placed "
    "before 10:00 the day before (the community's call for offers). After settlement, run verify_settlement before "
    "you sign off; approve only if every check passes."
)

STEPS = {   # coordinator / utility log kinds -> what happened, in order
    "need_posted": "buyer order published as a DEG demand-flex need",
    "v2_discover": "coordinator asked the utility for the next day's needs",
    "on_discover": "utility answered with the order",
    "v2_call_for_offers": "call for offers sent to the homes",
    "v2_cleared": "offers cleared: homes and kW per slot chosen",
    "contract_active": "utility contract ACTIVE (select, init, confirm)",
    "v2_commitments_sent": "commitments sent to the chosen homes",
    "v2_final": "homes answered: accepted / declined",
    "baselines_sent": "utility sent each meter's baseline",
    "v2_event_notice": "event notice: homes pre-charge",
    "v2_telemetry_to_utility": "window over: meter readings complete",
    "settled": "utility settled on the signed readings",
    "v2_settled": "community split the payment among the homes",
    "buyer_review": "buyer signed off",
}


def _steps(entries: list[dict[str, Any]], event_id: str, clock_key: str) -> list[dict[str, Any]]:
    seen: dict[str, dict[str, Any]] = {}
    for e in entries:
        mine = e.get("event_id") == event_id or event_id in (e.get("needs") or []) \
            or (e.get("kind") == "v2_discover" and f"flex-need-{e.get('date')}" == event_id)
        if e.get("kind") in STEPS and mine and e["kind"] not in seen:
            d = {k: v for k, v in e.items() if k in ("at_wall", "homes", "accepted", "declined", "meters", "readings",
                                                       "seller_usd", "homes_paid", "capacity_offered_kw", "decision")}
            seen[e["kind"]] = {"at": e.get(clock_key), "step": STEPS[e["kind"]], **d}
    return sorted(seen.values(), key=lambda x: list(STEPS).index(next(k for k, v in STEPS.items() if v == x["step"])))


def _settlement_view(need: dict[str, Any], res: dict[str, Any] | None, utility_settlement: dict[str, Any] | None,
                     offered: dict[str, Any]) -> dict[str, Any]:
    if not res or not utility_settlement:
        return {"settled": False}
    per_home = sorted(res["split"]["per_home"].items(), key=lambda x: -x[1])
    return {"settled": True, "event_id": need["event_id"], "currency": need["currency"],
            "utility_pays": res["seller_usd"],
            "slots": [{"slot": s["start"][11:], "requested_kw": s["kw_requested"],
                       "contracted_kw": float(offered.get(str(s["id"]), offered.get(s["id"], 0.0))),
                       "delivered_kw_measured": c["delivered"], "paid_kw": c["eligible"], "usd": c["net"]}
                      for s, c in zip(need["slots"], utility_settlement["components"])],
            "delivered_kwh_measured": res["delivered_kwh_measured"],
            "delivered_kwh_true_simulated": res.get("delivered_kwh_true"),
            "homes_paid": res["paid_homes"], "silent_homes": res.get("silent_homes", []),
            "top_payouts": dict(per_home[:5]), "split_rule": res["split"]["rule"],
            "note": "measured = baseline minus metered usage (what the utility pays on); payment per slot is capped "
                    "at the contracted kW. 'true' is the simulator's counterfactual, which a real utility cannot see."}


class Live:
    name = "live"

    def __init__(self):
        import httpx
        self.c = os.environ.get("FLEXGRID_COORDINATOR_URL", "http://127.0.0.1:7871").rstrip("/")
        self.u = os.environ.get("FLEXGRID_UTILITY_URL", "http://127.0.0.1:7873").rstrip("/")
        self.http = httpx.Client(timeout=30)

    def _get(self, base: str, path: str, **params) -> Any:
        r = self.http.get(base + path, params=params)
        r.raise_for_status()
        return r.json()

    def communities(self) -> dict[str, Any]:
        ch, cs = self._get(self.c, "/health"), self._get(self.c, "/v2/summary")
        uh, needs = self._get(self.u, "/health"), self._get(self.u, "/needs")
        cfg = config.load()
        return {"communities": [{"name": "flexgrid-community", "aggregator": ch["agent"], "pubkey": ch["pubkey"],
                                 "transport": ch.get("transport")
                                 or f"NANDA Town {cfg.get('town_url')} run {cfg.get('town_run_id')}",
                                 "homes": cs["roster"], "past_events": {k: v["phase"] for k, v in cs["events"].items()}}],
                "buyer": uh["label"], "now": uh["now"], "needs": needs["needs"]}

    def grid(self) -> dict[str, Any]:
        return self._get(self.c, "/grid/now")

    def now(self) -> str:
        return self._get(self.u, "/health")["now"]

    def order(self, o: dict[str, Any]) -> dict[str, Any]:
        r = self.http.post(self.u + "/needs", json=o)
        if r.status_code >= 400:
            raise ValueError(r.json().get("detail", r.text))
        return r.json()

    def contract(self, event_id: str) -> dict[str, Any]:
        s = self._get(self.c, "/v2/summary")
        ev = s["events"].get(event_id)
        uc = [c for c in self._get(self.u, "/contracts").values() if c.get("event_id") == event_id]
        log = self._get(self.u, "/events", n=300) + self._get(self.c, "/events", n=300)
        for e in log:
            e["at_wall"] = time.strftime("%H:%M:%S", time.localtime(e["at"]))
        return {"now": s["now"], "event": ev, "utility_contract": uc[-1]["status"] if uc else None,
                "steps": _steps(sorted(log, key=lambda e: e["at"]), event_id, "sim")}   # dataset time + wall time

    def settlement(self, event_id: str) -> dict[str, Any]:
        try:
            ev = self._get(self.c, f"/v2/events/{event_id}")
            ue = self._get(self.u, f"/evidence/{event_id}")
        except Exception:
            return {"settled": False}
        return _settlement_view(ev["need"], ev.get("result"), ue.get("settlement"), ue.get("offered", {}))

    def evidence(self, event_id: str) -> dict[str, Any]:
        return verify.fetch_live(event_id, self.c, self.u)

    def replay(self, b: dict[str, Any]) -> float | None:
        return verify.replay_total(b)

    def sign(self, event_id: str, approve: bool, reason: str, checks: Any) -> dict[str, Any]:
        r = self.http.post(f"{self.u}/reviews/{event_id}", json={"approve": approve, "reason": reason,
                                                                  "checks": checks, "by": "buyer-agent (MCP)"})
        if r.status_code >= 400:
            raise ValueError(r.json().get("detail", r.text))
        return r.json()


class Offline:
    """The same agents' code in one process; the event runs to settlement when the order is placed."""
    name = "offline replay (tools/sim_event.py: the live agents' classes, in-memory NANDA Town, virtual clock)"

    def __init__(self, today: str):
        self.today = today
        self.runs: dict[str, dict[str, Any]] = {}
        cat = json.loads(config.events_file().read_text())["message"]["catalogs"][0]
        offers = {o["resourceIds"][0]: o for o in cat["offers"]}
        self.catalog = [beckn.need_from_catalog(r, offers[r["id"]]) for r in cat["resources"] if r["id"] in offers]
        self.homes = len(json.loads(config.data_path("fleet_v2.json").read_text())["homes"])

    def communities(self) -> dict[str, Any]:
        upcoming = [{"event_id": n["event_id"], "window": [n["window_start"], n["window_end"]],
                     "kw": n["slots"][0]["kw_requested"], "price_usd_per_kwh": n["slots"][0]["price"],
                     "notice": f"{beckn.add_minutes(n['window_start'], -1440)[:10]} 10:00", "source": "catalog"}
                    for n in self.catalog if n["window_start"] >= self.today][:3]
        return {"communities": [{"name": "flexgrid-community", "aggregator": "coordinator", "homes": self.homes,
                                 "transport": "in-memory NANDA Town (offline replay)",
                                 "past_events": {k: "done" for k in self.runs}}],
                "buyer": "Utility or ENGIE (mock, public data)", "now": self.today, "needs": upcoming}

    def grid(self) -> dict[str, Any]:
        from . import grid_signals
        return grid_signals.snapshot()

    def now(self) -> str:
        return self.today

    def order(self, o: dict[str, Any]) -> dict[str, Any]:
        d0 = str(o["date"]) + " 00:00"
        notice = f"{beckn.add_minutes(d0, -1440)[:10]} 10:00"
        if self.today >= notice:
            raise ValueError(f"too late for {o['date']}: the call for offers went out at {notice} (now {self.today})")
        sys.path.insert(0, str(config.REPO_DIR / "tools"))
        from verify_run import sim_bundle
        import sim_event
        b, r = sim_bundle(o["date"], {**{k: v for k, v in o.items() if k != "date"}, "at": self.today})
        self.runs[r["event_id"]] = {"bundle": b, "result": r, "log": list(sim_event.LAST["log"]),
                                    "utility": sim_event.LAST["utility"], "market": sim_event.LAST["market"]}
        self.today = beckn.add_minutes(r["window"][1], 75)         # the replay has played the event through
        return b["utility"]["order"]

    def _run(self, event_id: str) -> dict[str, Any]:
        if event_id not in self.runs:
            raise ValueError(f"no order for {event_id} in this session")
        return self.runs[event_id]

    def contract(self, event_id: str) -> dict[str, Any]:
        run = self._run(event_id)
        ev = run["market"].summary()[event_id]
        return {"now": self.today, "event": json.loads(json.dumps(ev, default=str)),
                "utility_contract": run["bundle"]["utility"]["status"],
                "steps": _steps(run["log"], event_id, "sim")}

    def settlement(self, event_id: str) -> dict[str, Any]:
        b = self._run(event_id)["bundle"]
        return _settlement_view(b["utility"]["need"], b["coordinator"]["result"], b["utility"]["settlement"],
                                b["utility"]["offered"])

    def evidence(self, event_id: str) -> dict[str, Any]:
        return self._run(event_id)["bundle"]

    def replay(self, b: dict[str, Any]) -> float | None:
        return None                                                  # this backend is the replay

    def sign(self, event_id: str, approve: bool, reason: str, checks: Any) -> dict[str, Any]:
        rec = self._run(event_id)["utility"].review(event_id, approve, reason, checks, "buyer-agent (MCP)")
        self.runs[event_id]["bundle"]["utility"]["buyer_review"] = rec
        return rec


class ToolTable:
    """Stands in for the MCP server when the same tools are served another way (tools/agent_desk.py: plain HTTP for
    an agent harness without MCP): tool() records each function under its name."""

    def __init__(self):
        self.tools: dict[str, Any] = {}

    def tool(self):
        def deco(fn):
            self.tools[fn.__name__] = fn
            return fn
        return deco


def build(backend, app: Any = None) -> Any:
    if app is None:
        try:
            from mcp.server.mcpserver import MCPServer                 # mcp 2.x
        except ImportError:
            from mcp.server.fastmcp import FastMCP as MCPServer        # mcp 1.x
        app = MCPServer("flexgrid", instructions=INSTRUCTIONS)
    verified: dict[str, dict[str, Any]] = {}

    def wait(fn, done, wait_seconds: int):
        t_end = time.time() + max(0, min(int(wait_seconds), 300))
        out = fn()
        while not done(out) and time.time() < t_end:
            time.sleep(3)
            out = fn()
        return out

    @app.tool()
    def list_communities() -> dict:
        """The flexibility sellers this buyer can contract with (an aggregator of home agents), how many homes each
        has, the buyer's current clock time ('now', dataset time) and the needs already published with their notice
        time. Orders for a day must be placed before 10:00 the day before."""
        return {"backend": backend.name, **backend.communities(), "dataset": config.dataset_meta().get("label", "Illinois 2023"),
                "program": config.dataset_meta().get("program_label", "Ameren Illinois Peak Time Rewards: $0.75 per kWh reduced")}

    @app.tool()
    def get_grid_outlook() -> dict:
        """Today's real grid in MISO (MISO's public data, live; for the Illinois community only, it answers
        'not available' elsewhere): generation by fuel and the
        share from wind and solar, the load now and forecast, the price at the Illinois hub now, and today's day-ahead
        price by hour with the most expensive 2-hour window and the cheapest hours. The community trades on the 2023
        dataset's clock, so use this to choose WHICH HOURS to order (the window where the grid is tight today), not
        the date. With the data centers running (live backend), also what the ten data center agents plan for their
        batch work on today's prices (datacenters_today: MWh moved, cost with and without, the peak window's draw).
        Parts that MISO did not answer are listed under errors."""
        if config.dataset_meta().get("name") != "il2023":
            return {"available": False, "error": "the live grid feed is MISO's (Illinois); this community is on ISO New "
                    "England, which has no live feed here: use forecast_community_demand to choose the hours"}
        try:
            g = backend.grid()
        except Exception as exc:                                     # noqa: BLE001
            return {"available": False, "error": str(exc)[:200]}
        da, fm, ld, pn = g.get("day_ahead") or {}, g.get("fuel_mix") or {}, g.get("load") or {}, g.get("price_now") or {}
        return {"available": bool(da or fm), "source": g.get("source"), "day": g.get("day"),
                "plan": g.get("plan"), "peak_window": da.get("peak_window"), "cheapest_hours": da.get("cheapest_hours"),
                "day_ahead_usd_mwh_by_hour": dict(zip(da.get("starts", []), da.get("lmp_usd_mwh", []))),
                "price_now_illinois_hub_usd_mwh": pn.get("illinois_hub_usd_mwh"), "price_now_interval_est": pn.get("interval_est"),
                "wind_solar_pct_now": fm.get("wind_solar_pct"), "carbon_free_pct_now": fm.get("carbon_free_pct"),
                "load_now_mw": ld.get("now_mw"), "load_peak_forecast": ld.get("peak_forecast"), "errors": g.get("errors"),
                "datacenters_today": ({k: v for k, v in g["datacenters_today"].items() if k not in ("series_mw", "sites")}
                                      if isinstance(g.get("datacenters_today"), dict) else None)}

    @app.tool()
    def forecast_community_demand(date: str) -> dict:
        """The community's demand for every hour of `date` (YYYY-MM-DD) as FlexGrid's trained forecasting model sees
        it at 10:00 the day before (the call for offers): 10 %, 50 % and 90 % values in kW for all the homes together
        (their base load; EVs are the homes' own flexibility), and the 3-hour evening window with the highest forecast
        demand. Use it to choose the window and the kW you order; the 10-90 % range is honest about uncertainty.
        Built only from the community's metered history and the weather forecast as issued, nothing from the day
        itself (Massachusetts 2025-26 data)."""
        issue = f"{beckn.add_minutes(f'{date} 00:00', -1440)[:10]} 10:00"   # the call for offers ...
        try:
            now = str(backend.now())[:16]
        except Exception:                                            # noqa: BLE001
            now = issue
        issue = min(issue, now)                                       # ... or now, if that is earlier: no peeking
        try:
            return {"backend": backend.name, **community_forecast.forecast(date, issued_at=issue)}
        except (ValueError, OSError) as exc:
            return {"backend": backend.name, "available": False, "error": str(exc)[:300]}

    @app.tool()
    def request_flexibility(date: str, start: str, end: str, kw: float, price_usd_per_kwh: float,
                            note: str = "") -> dict:
        """Order load relief from the community for one window. date: YYYY-MM-DD; start/end: local HH:MM on the
        half hour (30 min to 4 h); kw: relief wanted in every 30-minute slot (max 500); price_usd_per_kwh: what you pay
        per kWh delivered (max 5; list_communities shows what the area's program pays). Paid only on measured
        delivery, capped at the kW the
        community contracts. Returns the published order (event_id, signed by the utility)."""
        cap = float(os.environ.get("FLEXGRID_BUYER_MAX_PRICE") or 0)      # this buyer's own limit (the code decides)
        if cap and float(price_usd_per_kwh) > cap + 1e-9:
            return {"backend": backend.name, "accepted": False,
                    "error": f"price ${float(price_usd_per_kwh):g}/kWh is above this buyer's limit ${cap:g}/kWh: "
                             f"order again at or below it"}
        try:
            rec = backend.order({"date": date, "start": start, "end": end, "kw": kw,
                                 "price_usd_per_kwh": price_usd_per_kwh, "note": note, "buyer": "buyer-agent (MCP)"})
        except ValueError as exc:
            return {"backend": backend.name, "accepted": False, "error": str(exc)}
        return {"backend": backend.name, "accepted": True, "event_id": rec["event_id"], "order": rec["order"],
                "call_for_offers_at": rec["notice"], "homes_commit_by": rec["commit_by"],
                "signed_by_utility": rec["signer"][:16] + "...",
                "next": "get_contract(event_id, wait_seconds) to follow the deal; get_settlement after the window"}

    @app.tool()
    def get_contract(event_id: str, wait_seconds: int = 0) -> dict:
        """Where the deal stands: the community's phase (offers -> utility -> homes -> done), offers received,
        homes accepted/declined, kW contracted per slot, the utility contract status, and each step with its time.
        wait_seconds (max 300) waits for the next phase change before answering."""
        first = backend.contract(event_id)
        phase0 = (first.get("event") or {}).get("phase")
        out = wait(lambda: backend.contract(event_id), lambda x: (x.get("event") or {}).get("phase") != phase0
                   or phase0 == "done", wait_seconds) if wait_seconds else first
        if out.get("event"):                                        # the money is get_settlement's answer
            out["event"] = {k: v for k, v in out["event"].items() if k != "result"}
        return {"backend": backend.name, **out}

    @app.tool()
    def get_settlement(event_id: str, wait_seconds: int = 0) -> dict:
        """After the window: per slot the kW requested, contracted and delivered (measured against each meter's
        baseline), what the utility pays, and how the payment is split among the homes. settled=false until the
        utility has settled. wait_seconds (max 300) waits for the settlement."""
        out = wait(lambda: backend.settlement(event_id), lambda x: x.get("settled"), wait_seconds)
        return {"backend": backend.name, **out}

    @app.tool()
    def verify_settlement(event_id: str) -> dict:
        """Check the settlement independently from what the coordinator and the utility each hold: the contracted
        need is your signed order; every meter reading verifies against the home's pinned key and was counted as
        signed; the payment recomputed from the signed readings matches to the cent; the split adds up; and (live)
        an offline replay of the same order with the same homes pays the same. Returns PASS/FAIL per check."""
        b = backend.evidence(event_id)
        try:
            rt = backend.replay(b)
        except Exception as exc:
            rt = None
            b.setdefault("notes", []).append(f"replay skipped: {exc}")
        r = verify.check(b, rt)
        path = config.REPO_DIR / "results" / f"evidence_{event_id}_{b['source']}.json"
        try:
            path.write_text(json.dumps(b, indent=1))
        except OSError:
            path = None
        verified[event_id] = r
        return {"backend": backend.name, "all_pass": r["ok"],
                "checks": [{"check": x["name"], "result": "PASS" if x["ok"] else ("SKIP" if x["ok"] is None else "FAIL"),
                            "detail": x["detail"]} for x in r["checks"]],
                "summary": r["summary"], "evidence_file": str(path) if path else None}

    @app.tool()
    def sign_off(event_id: str, approve: bool, reason: str) -> dict:
        """Record your decision on a settled event with the utility (signed, kept on the contract): approve, or
        dispute with a reason. Approval is refused unless verify_settlement passed for this event in this session."""
        v = verified.get(event_id)
        if approve and not (v and v["ok"]):
            return {"recorded": False, "error": "run verify_settlement first; approve only when every check passes"}
        checks = {x["name"]: x["ok"] for x in v["checks"]} if v else None
        try:
            rec = backend.sign(event_id, approve, reason, checks)
        except ValueError as exc:
            return {"recorded": False, "error": str(exc)}
        return {"backend": backend.name, "recorded": True, "decision": rec["decision"], "at": rec["at"],
                "settlement_usd": rec["settlement_usd"], "signed_by_utility": rec["signer"][:16] + "..."}

    return app


def main() -> None:
    ap = argparse.ArgumentParser(prog="flexgrid-mcp")
    ap.add_argument("--offline", action="store_true", help="run events in-process instead of on the live agents")
    ap.add_argument("--today", default=None, help="offline: the buyer's clock (dataset time; default 08:00 the day before the dataset's default event)")
    a = ap.parse_args()
    import logging
    logging.getLogger("httpx").setLevel(logging.WARNING)        # not every request in the harness's server log
    today = a.today or beckn.add_minutes(f"{config.dataset_meta()['default_event']} 08:00", -1440)
    build(Offline(today) if a.offline else Live()).run()


if __name__ == "__main__":
    main()
