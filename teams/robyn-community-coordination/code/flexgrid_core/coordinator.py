"""flexgrid-coordinator - the community coordinator agent (runs on the Mac).

Same pattern as the homes: a small daemon owns everything that touches the outside world
(NANDA Town session, the coordinator's signing key, the pinned home keys, the receipt ledger),
and Langflow runs and shows the decision steps.

v2 (Beckn demand flexibility, market.py) - one contract per grid event with the utility:
  C3 Clear                when the homes' offers are in: Collect Offers -> Clear with Rebound Guard
                          -> Contract with the Utility (Beckn select / init / confirm)
  C4 Settle               when the utility has settled: Utility Settlement -> Split and Pay
  The rest of the exchange (calls for offers, commitments, event notices, readings to the utility) runs in the
  daemon on the shared replay clock; every message is a signed Beckn body sent over NANDA Town.

v1 (off unless agent.json has "v1_dispatch": true; kept until v2 passes the same tests):
  C1 Community dispatch   one run per fresh flex.report:
                          Verify Report -> Community View -> Plan Dispatch (rules v0) -> Grid Safety Check
                          -> Sign and Send Dispatch
  C2 Receipt check        one run per flex.dispatch_ack:  Verify Receipt -> Record in Ledger

Rules v0 (the MPC planner replaces Plan Dispatch later; the safety check stays):
  * 17:00-21:00  evening peak  -> ask the home to export / cut import, up to 3 kW   (delta_kw > 0)
  * 09:00-15:00  solar surplus -> ask a home with room to absorb, up to 2.5 kW      (delta_kw < 0)
Grid safety check: fresh verified report, healthy sensors, inside the envelope, one dispatch per home
per hour, one new dispatch per 15-minute slot across the community (no herding), feeder limit.
Send re-runs the safety check itself, so a flow can block more but never allow more.

  flexgrid-coordinator run                        # daemon on 127.0.0.1:7871 + Town inbox loop
  flexgrid-coordinator pubkey
  flexgrid-coordinator send --to home-tesla --delta 2.5 --minutes 30   # manual test dispatch
  flexgrid-coordinator send --to home-tesla --delta 20                 # over the home's hard limit: must be rejected
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import threading
import time
import traceback
import uuid
from collections import deque
from datetime import datetime, timedelta
from typing import Any

import httpx

try:            # at module level: the routes' annotations are strings (postponed), resolved in this module's globals,
    from fastapi import Request     # so a Request imported inside build_app would read as a query parameter
except ImportError:                 # pragma: no cover (the coordinator itself always has FastAPI)
    Request = Any

from . import beckn, config, doors, keep, signing
from .market import Market
from .replay import Replay
from . import meter_check
from .town import InboxPoller, LagMeter, TownSession

log = logging.getLogger("flexgrid-coordinator")
TS = "%Y-%m-%d %H:%M"


def _sim(ts: str) -> datetime:
    return datetime.strptime(ts[:16], TS)


class Coordinator:
    def __init__(self):
        self.cfg = config.load()
        self.name = self.cfg.get("agent_name") or "coordinator"
        self.key = signing.load_or_create(config.CONFIG_DIR / "secrets" / "signing.key")
        self.pub = signing.public_hex(self.key)
        tok = config.secret("town_token")
        self.town = TownSession(self.cfg["town_url"], self.cfg["town_run_id"], self.name, tok) \
            if self.cfg.get("town_run_id") and tok else None
        self.damaged: list[tuple[str, str]] = []          # files read here that couldn't be used: keep.for_coordinator makes each one an alarm
        self.pins_path = config.state_dir() / "home_pins.json"
        self.pins: dict[str, str] = self.read_pins()
        self.keeper = None                                # keep.for_coordinator: live only
        self.latest: dict[str, dict[str, Any]] = {}       # home -> {"env", "at"}
        self.last_sent: dict[str, str] = {}               # home -> "YYYY-MM-DD HH" of its last dispatch
        self.slot_starts: dict[str, str] = {}             # sim slot -> home that got a new dispatch in it
        self.dispatches: dict[str, dict[str, Any]] = {}   # dispatch_id -> signed dispatch (+ slot_start)
        self.events: deque = deque(maxlen=300)
        self.stale_skipped = 0
        self.expired_skipped = 0                          # Beckn messages past their ttl (dropped, not acted on)
        self.inbox_ready = False                          # True once the Town inbox has been empty: caught up
        self.lag = LagMeter(window=float(self.cfg.get("town_lag_window_s", 20.0)))   # how far behind the inbox is
        self.started_wall = time.time()
        self.lock = threading.RLock()
        self.db = self.open_ledger(config.state_dir() / "ledger.db")
        # v2 market: one Beckn contract per grid event (see market.py)
        c = self.cfg
        roster = c.get("v2_roster") or [h["name"] for h in
                                        json.loads(config.data_path("fleet_v2.json").read_text())["homes"]]
        from . import building                            # the building nodes this config enables (cfg "buildings"):
        roster = roster + [b["name"] for b in building.enabled(c) if b["name"] not in roster]   # asked like a home
        self.clock = Replay(c.get("clock_profile", "tesla_calhoun"), c["replay_start"], c["replay_step_seconds"],
                            c["replay_loop_days"])
        self.market = Market(self.name, self.key, self._town_send, self.clock, roster, self.pins, self.save_pins, self.event, c)
        self.market.on_clear, self.market.on_settle = self.c3, self.c4
        self.market.town_lag = self.town_lag              # the real-time desk waits longer for answers when Town is behind
        if self.market.pooled:                            # what each member may claim comes from its enrolment, not its word: an offer's limit, and a live run's too
            self.market.rated = meter_check.rated(roster)
            self.event("v2_nameplates", members=len(roster), on_file=len(self.market.rated))
        from .monitor import LiveRecorder               # the Control Room's record of the current event
        self.monitor = LiveRecorder(self.market, c, self.event, lambda: list(self.events))
        # the Mac's AI agents (llm_enabled): the coordinator's own (it picks each event's margin; brain.py) and the grid
        # buyer's (it orders tomorrow's flexibility and signs off the settlement; buyer_agent.py). Both think with the
        # local LLM through the Mac's OpenClaw gateway (agents "coordinator" and "buyer"), Ollama straight as stand-in
        self.brain = self.buyer = None
        if c.get("llm_enabled"):
            from .brain import make_backend
            from .coord_agent import for_market
            self.brain = for_market(make_backend({**c, "openclaw_agent": "coordinator"}, "coordinator"), self.market.pooled,
                                    log=self.event, budget_s=float(c.get("llm_margin_budget_s", 40)), default_margin=float(c.get("clear_margin", 0.10)))
            self.market.brain = self.monitor.coord_brain = self.brain
            if c.get("buyer_agent", True):
                from .buyer_agent import BuyerAgent
                self.buyer = BuyerAgent(c, self.event, self.monitor.src["utility"])
                self.monitor.buyer = self.buyer
        self.buyer_signed: set[str] = set()               # events whose sign-off the buyer agent has been asked for

    def read_pins(self) -> dict[str, str]:
        """A file that can't be read costs the pins, not the start. It's set aside and it's an alarm, because until each node speaks again it's pinned to whoever speaks first."""
        try:
            pins = json.loads(self.pins_path.read_text()) if self.pins_path.exists() else {}
            if not isinstance(pins, dict) or not all(isinstance(v, str) for v in pins.values()):
                raise ValueError("it isn't a table of keys")
            return pins
        except (OSError, ValueError, RecursionError) as exc:
            self.damaged.append(("home_pins", f"{type(exc).__name__}: {str(exc)[:120]}"))
            keep.set_aside(self.pins_path)
            return {}

    def open_ledger(self, path: Any) -> sqlite3.Connection:
        """The record of the old path's receipts. A file that isn't a database costs that record, not the start. If it can't even be moved, the ledger lasts as long as the process."""
        def opened(where: str) -> sqlite3.Connection:
            db = sqlite3.connect(where, check_same_thread=False)
            try:
                db.execute("CREATE TABLE IF NOT EXISTS receipts (n INTEGER PRIMARY KEY AUTOINCREMENT, home TEXT, "
                           "dispatch_id TEXT, status TEXT, reason TEXT, requested REAL, applied REAL, hash TEXT, "
                           "prev_hash TEXT, chain TEXT, body TEXT, at REAL)")
                db.commit()
                said = db.execute("PRAGMA quick_check").fetchone()[0]
                if said != "ok":
                    raise sqlite3.DatabaseError(str(said)[:120])
                return db
            except sqlite3.Error:
                db.close()
                raise
        try:
            return opened(str(path))
        except sqlite3.Error as exc:
            self.damaged.append(("ledger", f"{type(exc).__name__}: {str(exc)[:120]}"))
        keep.set_aside(path)
        try:
            return opened(str(path))
        except sqlite3.Error:
            return opened(":memory:")

    def save_pins(self) -> None:
        """All or nothing: a file cut short by a crash costs every pin at the next start."""
        keep.write_atomic(self.pins_path, json.dumps(self.pins, indent=1).encode())

    def _v2_flow(self, flow: str, event_id: str, phase: str, fallback) -> None:
        """Run a v2 Langflow flow for one event in the background; if it is not set up or does not get the event
        past `phase`, the daemon does the same steps itself."""
        def run():
            try:
                out = self.run_flow(flow, json.dumps({"event_id": event_id}),
                                    f"{flow} {time.strftime('%Y-%m-%d %H')}h · {event_id}")
                if out is not None:
                    self.event(f"{flow.lower()}_langflow", event_id=event_id, output=out[:200])
            except Exception as exc:
                self.event("v2_flow_error", flow=flow, event_id=event_id, error=str(exc)[:200])
            with self.market.lock:
                ev = self.market.events.get(event_id)
                if ev is None or ev["phase"] != phase:
                    return
                try:
                    fallback(event_id)
                    self.event(f"{flow.lower()}_inprocess", event_id=event_id)
                except Exception as exc:                  # noqa: BLE001  this thread dying says nothing, and no tick picks these phases up: the alarm reads this (keep.market_risks)
                    ev["flow_failed"] = f"{type(exc).__name__}: {exc}"[:160]
                    self.event("v2_flow_failed", flow=flow, event_id=event_id, error=ev["flow_failed"])
                    log.error("%s for %s: %s", flow, event_id, traceback.format_exc())
        threading.Thread(target=run, daemon=True).start()

    def c3(self, event_id: str) -> None:
        """Offers are in: flow C3 Clear (Collect Offers -> Clear with Rebound Guard -> Contract with the Utility)."""
        ev = self.market.events[event_id]
        ev["phase"], ev["clearing_wall"] = "clearing", time.time()
        self._v2_flow("C3", event_id, "clearing", self.market.clear_and_commit)

    def c4(self, event_id: str) -> None:
        """The utility has settled: flow C4 Settle (Utility Settlement -> Split and Pay). Once per event: a settlement
        that arrives again (Town delivers at least once) never sends the event back to 'settling'."""
        ev = self.market.events[event_id]
        if ev.get("result") or ev["phase"] in ("settling", "done"):
            self.event("v2_settle_repeat_ignored", event_id=event_id, phase=ev["phase"])
            return
        ev["phase"] = "settling"
        self._v2_flow("C4", event_id, "settling", self.market.settle)

    def _town_send(self, to: str, kind: str, body: dict[str, Any], message_id: str) -> dict[str, Any]:
        if not self.town:
            raise RuntimeError("no NANDA Town session")
        out = self.town.send(to, kind, body, message_id)
        if isinstance(out, dict) and out.get("duplicate"):    # Town refused it: see town.send
            self.event("town_send_dropped", to=to, message_kind=kind, message_id=message_id)
        return out

    def market_loop(self) -> None:
        while True:
            try:
                self.market.tick()
                self._buyer_tick()
                self.market.failing.pop("the tick", None)
            except Exception as exc:
                self.event("market_error", error=str(exc)[:200])
                log.error("market: %s", traceback.format_exc())
                self.market.failed("the tick", exc)
            time.sleep(1.0)

    def _buyer_tick(self) -> None:
        """An event the utility has settled and the community has paid: the buyer's AI agent checks the settlement
        and signs off (once per event, in the background)."""
        if self.buyer is None:
            return
        with self.market.lock:
            done = [eid for eid, ev in self.market.events.items() if ev.get("phase") == "done" and eid not in self.buyer_signed]
            self.buyer_signed.update(done)
        for eid in done:
            threading.Thread(target=self.buyer.sign_off, args=(eid, float(self.cfg.get("llm_signoff_budget_s", 120))),
                             daemon=True).start()

    # ------------------------------------------------------------------ helpers
    def event(self, kind: str, **detail) -> None:
        try:
            sim = self.clock.ts(self.clock.current())            # dataset time, next to the wall time
        except Exception:
            sim = None
        self.events.append({"at": time.time(), "sim": sim, "kind": kind, **detail})
        log.info("%s %s", kind, json.dumps(detail, default=str)[:300])

    def sim_now(self) -> datetime | None:
        slots = [_sim(v["env"]["slot_start"]) for v in self.latest.values()]
        return max(slots) if slots else None

    def active_dispatches(self) -> list[dict[str, Any]]:
        now = self.sim_now()
        if now is None:
            return []
        out = []
        for d in self.dispatches.values():
            start = _sim(d["slot_start"])
            if start <= now < start + timedelta(minutes=float(d["minutes"])):
                out.append({"dispatch_id": d["dispatch_id"], "to": d["to"], "delta_kw": d["delta_kw"],
                            "reason": d["reason"], "until": (start + timedelta(minutes=float(d["minutes"]))).strftime(TS)})
        return out

    # ------------------------------------------------------------------ C1 steps
    def verify_report(self, sender: str, report: dict[str, Any]) -> dict[str, Any]:
        """Signature against the home's pinned key (pinned on first use), sender = agent, freshness."""
        signer = report.get("signer", "")
        with self.lock:
            pinned = self.pins.get(sender)
            first = pinned is None
            if first and signer:
                self.pins[sender] = pinned = signer
                self.save_pins()
        if not signing.verify(report, pinned or ""):
            return {"ok": False, "reason": "bad_signature", "home": sender}
        if report.get("agent") not in (None, sender):
            return {"ok": False, "reason": "sender_mismatch", "home": sender}
        age = time.time() - float(report.get("published_wall", 0))
        if age > float(self.cfg["report_max_age_s"]):
            return {"ok": False, "reason": "stale", "home": sender, "age_s": round(age, 1)}
        with self.lock:
            self.latest[sender] = {"env": report, "at": time.time()}
        return {"ok": True, "reason": "pinned_on_first_use" if first else "signature_ok", "home": sender,
                "seq": report.get("seq"), "slot_start": report.get("slot_start"), "age_s": round(age, 1)}

    def community(self) -> dict[str, Any]:
        now = time.time()
        homes = []
        with self.lock:
            for h, v in sorted(self.latest.items()):
                e = v["env"]
                homes.append({"home": h, "site": e.get("site"), "slot_start": e["slot_start"], "up_kw": e["up_kw"],
                              "down_kw": e["down_kw"], "health": e.get("health"), "confidence": e.get("confidence"),
                              "grid_kw_now": e.get("grid_kw_now"), "age_s": round(now - v["at"], 1)})
            active = self.active_dispatches()
        return {"homes": homes, "n_homes": len(homes),
                "total_up_kw": round(sum(x["up_kw"] for x in homes), 2),
                "total_down_kw": round(sum(x["down_kw"] for x in homes), 2),
                "grid_kw_now": round(sum((x["grid_kw_now"] or 0) for x in homes), 2),
                "active_dispatches": active, "feeder_limit_kw": self.cfg["feeder_limit_kw"]}

    def plan(self, sender: str, report: dict[str, Any], verified: dict[str, Any] | None) -> dict[str, Any]:
        """Rules v0. Returns a proposal; the safety check decides whether it may go out."""
        base = {"to": sender, "envelope_seq": report.get("seq"), "slot_start": report.get("slot_start"),
                "minutes": 30, "planner": "rules_v0"}
        if not verified or not verified.get("ok"):
            return {**base, "action": "none", "reason": "report_not_verified"}
        if not self.cfg.get("v1_dispatch", False):
            return {**base, "action": "none", "reason": "v1_rules_off_v2_market_runs_events"}
        hour = int(report["slot_start"][11:13])
        if 17 <= hour < 21:
            if report["up_kw"] <= 0.3:
                return {**base, "action": "none", "reason": "evening_peak_but_no_up_headroom"}
            return {**base, "action": "dispatch", "delta_kw": round(min(report["up_kw"], 3.0), 2),
                    "reason": "evening_peak_support"}
        if 9 <= hour < 15:
            surplus = any(b["grid_p50"] > 0.5 for b in report.get("baseline_next", [])[:4])
            if surplus:
                return {**base, "action": "none", "reason": "home_has_own_surplus_coming"}
            if report["down_kw"] <= 0.3:
                return {**base, "action": "none", "reason": "midday_but_no_down_headroom"}
            return {**base, "action": "dispatch", "delta_kw": round(-min(report["down_kw"], 2.5), 2),
                    "reason": "absorb_regional_solar"}
        return {**base, "action": "none", "reason": "outside_peak_and_solar_windows"}

    def check(self, p: dict[str, Any]) -> dict[str, Any]:
        """Grid safety check. approved / clamped / blocked / no_action."""
        if p.get("action") != "dispatch":
            return {"status": "no_action", "reason": p.get("reason", "nothing_planned")}
        to, delta = p["to"], float(p["delta_kw"])
        with self.lock:
            cur = self.latest.get(to)
            if not cur:
                return {"status": "blocked", "reason": "no_verified_report"}
            env = cur["env"]
            if time.time() - cur["at"] > float(self.cfg["report_max_age_s"]):
                return {"status": "blocked", "reason": "report_too_old"}
            if env.get("seq") != p.get("envelope_seq"):
                return {"status": "blocked", "reason": "newer_envelope_arrived"}
            if env.get("health") not in ("ok", None):
                return {"status": "blocked", "reason": f"home_sensor_{env.get('health')}"}
            if abs(delta) < 0.2:
                return {"status": "blocked", "reason": "too_small"}
            if self.last_sent.get(to) == p["slot_start"][:13]:
                return {"status": "blocked", "reason": "one_dispatch_per_home_per_hour"}
            other = self.slot_starts.get(p["slot_start"])
            if other and other != to:
                return {"status": "blocked", "reason": f"stagger_{other}_started_this_slot"}
            same_sign = sum(abs(a["delta_kw"]) for a in self.active_dispatches()
                            if (a["delta_kw"] > 0) == (delta > 0) and a["to"] != to)
            room = float(self.cfg["feeder_limit_kw"]) - same_sign
            if room < 0.2:
                return {"status": "blocked", "reason": "feeder_limit_reached", "feeder_room_kw": round(room, 2)}
            limit = env["up_kw"] if delta > 0 else env["down_kw"]
            new = max(-limit, min(limit, delta))
            reasons = []
            if abs(new - delta) > 1e-6:
                reasons.append("clamped_to_envelope")
            if abs(new) > room:
                new = room if new > 0 else -room
                reasons.append("clamped_to_feeder_limit")
        return {"status": "clamped" if reasons else "approved", "reason": ",".join(reasons) or "inside_envelope",
                "delta_kw": round(new, 2), "feeder_room_kw": round(room, 2)}

    def send(self, p: dict[str, Any], flow_check: dict[str, Any] | None = None) -> dict[str, Any]:
        c = self.check(p)                                   # never trust the caller's check
        if flow_check and flow_check.get("status") in ("blocked", "no_action"):
            c = flow_check                                  # a flow may block more, never allow more
        if c["status"] not in ("approved", "clamped"):
            return {"sent": False, "check": c, "summary": f"No dispatch to {p.get('to')}: {c['reason']}."}
        d = self.dispatch(p["to"], c["delta_kw"], p["minutes"], p["reason"], p.get("envelope_seq"), p["slot_start"])
        with self.lock:
            self.last_sent[p["to"]] = p["slot_start"][:13]
            self.slot_starts[p["slot_start"]] = p["to"]
        return {"sent": True, "check": c, "dispatch": d,
                "summary": (f"Dispatch {d['dispatch_id']} -> {p['to']}: {c['delta_kw']:+.2f} kW for {p['minutes']} min "
                            f"({p['reason']}, {c['status']}). Signed and sent over NANDA Town.")}

    def dispatch(self, to: str, delta_kw: float, minutes: float = 30, reason: str = "manual_test",
                 envelope_seq: int | None = None, slot_start: str | None = None) -> dict[str, Any]:
        if not self.town:
            raise RuntimeError("no NANDA Town session (town_run_id / town_token missing)")
        now = time.time()
        d = {"type": "flex.dispatch_request", "version": "flexgrid.dispatch.v1",
             "dispatch_id": f"d-{uuid.uuid4().hex[:12]}", "to": to, "delta_kw": round(float(delta_kw), 3),
             "minutes": minutes, "reason": reason, "envelope_seq": envelope_seq,
             "issued_wall": now, "expires_wall": now + float(self.cfg["dispatch_ttl_s"])}
        d = signing.sign(d, self.key)
        self.town.send(to, "flex.dispatch_request", d, f"{self.name}:{d['dispatch_id']}")
        slot = slot_start or (self.latest.get(to, {}).get("env", {}).get("slot_start")) or "2000-01-01 00:00"
        with self.lock:
            self.dispatches[d["dispatch_id"]] = {**d, "slot_start": slot}
            if len(self.dispatches) > 500:
                self.dispatches.pop(next(iter(self.dispatches)))
        self.event("dispatch_sent", dispatch_id=d["dispatch_id"], to=to, delta=d["delta_kw"], reason=reason)
        return d

    # ------------------------------------------------------------------ C2 steps
    def verify_receipt(self, sender: str, receipt: dict[str, Any]) -> dict[str, Any]:
        body = {k: v for k, v in receipt.items() if k != "receipt_hash"}
        pinned = self.pins.get(sender)
        if not signing.verify(body, pinned or ""):
            return {"ok": False, "reason": "bad_signature" if pinned else "unknown_home", "home": sender}
        h = signing.digest(body)
        if receipt.get("receipt_hash") and receipt["receipt_hash"] != h:
            return {"ok": False, "reason": "hash_mismatch", "home": sender}
        ours = receipt.get("dispatch_id") in self.dispatches
        row = self.db.execute("SELECT hash FROM receipts WHERE home=? ORDER BY n DESC LIMIT 1", (sender,)).fetchone()
        chain = "start" if row is None else ("linked" if body.get("prev_hash") == row[0] else "gap")
        return {"ok": True, "reason": "signature_ok", "home": sender, "hash": h, "chain": chain,
                "matches_our_dispatch": ours, "status": body.get("status")}

    def record_receipt(self, sender: str, receipt: dict[str, Any], v: dict[str, Any]) -> dict[str, Any]:
        if not v.get("ok"):
            self.event("receipt_rejected", home=sender, reason=v.get("reason"))
            return {"recorded": False, "summary": f"Receipt from {sender} NOT recorded: {v.get('reason')}."}
        body = {k: v2 for k, v2 in receipt.items() if k != "receipt_hash"}
        with self.lock:
            self.db.execute("INSERT INTO receipts (home, dispatch_id, status, reason, requested, applied, hash, prev_hash, "
                            "chain, body, at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                            (sender, body.get("dispatch_id"), body.get("status"), body.get("reason"),
                             body.get("requested_kw"), body.get("applied_kw"), v["hash"], body.get("prev_hash"),
                             v["chain"], json.dumps(body), time.time()))
            self.db.commit()
            n, delivered = self.db.execute("SELECT COUNT(*), COALESCE(SUM(ABS(applied)),0) FROM receipts").fetchone()
        self.event("receipt_recorded", home=sender, dispatch_id=body.get("dispatch_id"), status=body.get("status"),
                   applied=body.get("applied_kw"), chain=v["chain"])
        return {"recorded": True, "ledger_rows": n, "summary":
                (f"{sender} {body.get('status')} {body.get('dispatch_id')} ({body.get('reason')}): requested "
                 f"{body.get('requested_kw')} kW, applied {body.get('applied_kw')} kW. Signature ok, chain {v['chain']}. "
                 f"Ledger: {n} receipts so far, {delivered:.2f} kW of dispatches applied.")}

    # ------------------------------------------------------------------ Langflow bridge
    def run_flow(self, name: str, text: str, session: str) -> str | None:
        flow_id = (self.cfg.get("flows") or {}).get(name)
        key = config.secret("langflow_api_key")
        if not flow_id or not key:
            return None
        r = httpx.post(f"{self.cfg['langflow_url'].rstrip('/')}/api/v1/run/{flow_id}",
                       params={"stream": "false"}, headers={"x-api-key": key}, timeout=120,
                       json={"input_value": text, "input_type": "chat", "output_type": "chat", "session_id": session})
        r.raise_for_status()
        try:
            return r.json()["outputs"][0]["outputs"][0]["results"]["message"]["text"]
        except Exception:
            return json.dumps(r.json())[:500]

    def c1(self, sender: str, report: dict[str, Any]) -> str:
        session = f"C1 {time.strftime('%Y-%m-%d %H')}h · replay {str(report.get('slot_start', ''))[:10]}"
        try:
            out = self.run_flow("C1", json.dumps({"from": sender, "report": report}), session)
            if out is not None:
                self.event("c1_langflow", home=sender, output=out[:200])
                return out
        except Exception as exc:
            self.event("c1_error", error=str(exc)[:200])
        v = self.verify_report(sender, report)              # same steps in-process: the UI is never load-bearing
        res = self.send(self.plan(sender, report, v))
        self.event("c1_inprocess", home=sender, output=res["summary"])
        return res["summary"]

    def c2(self, sender: str, receipt: dict[str, Any]) -> str:
        session = f"C2 {time.strftime('%Y-%m-%d %H')}h · replay {str(receipt.get('sim_ts', ''))[:10]}"
        try:
            out = self.run_flow("C2", json.dumps({"from": sender, "receipt": receipt}), session)
            if out is not None:
                self.event("c2_langflow", home=sender, output=out[:200])
                return out
        except Exception as exc:
            self.event("c2_error", error=str(exc)[:200])
        res = self.record_receipt(sender, receipt, self.verify_receipt(sender, receipt))
        self.event("c2_inprocess", home=sender, output=res["summary"])
        return res["summary"]

    # ------------------------------------------------------------------ Town inbox
    def town_lag(self) -> float:
        """Seconds NANDA Town is behind: the median wait of the messages the coordinator claimed lately, or of the
        utility's (the monitor reads it with the utility's traffic), whichever is longer. 0 when the inboxes keep up."""
        util = float(getattr(self.monitor, "util_lag", 0.0) or 0.0) \
            if time.time() - float(getattr(self.monitor, "util_lag_at", 0.0) or 0.0) < 30 else 0.0
        return max(self.lag.value(), util)

    def handle(self, msg: dict[str, Any]) -> None:
        self.lag.seen(msg)
        body, kind, sender = msg.get("body") or {}, msg.get("kind"), msg.get("from")
        status, note = "processed", {}
        if str(kind).startswith("beckn.") and beckn.expired(body, default_ttl=float(self.cfg.get("max_message_age_s", 900))):
            self.expired_skipped += 1                     # waited in the inbox past its validity (e.g. while the
            status, note = "rejected", {"skipped": "expired", "late_s": round(beckn.expired(body) or 0)}   # coordinator was off): drop
        elif str(kind).startswith("beckn."):
            try:
                status, note = beckn.ack_of(self.market.handle(kind, sender, body), 300, "result")
            except Exception as exc:
                status, note = "rejected", {"error": str(exc)[:200]}
                log.error("v2: %s", traceback.format_exc())
        elif kind == "flex.report":
            age = time.time() - float(body.get("published_wall", 0))
            if age > float(self.cfg["report_max_age_s"]):      # backlog from while the coordinator was off
                self.stale_skipped += 1
                status, note = "rejected", {"skipped": "stale", "age_s": round(age)}
            else:
                note = {"result": self.c1(sender, body)[:300]}
        elif kind == "flex.dispatch_ack":
            note = {"result": self.c2(sender, body)[:300]}
        else:
            status, note = "rejected", {"reason": "unknown_kind"}
        if self.keeper is not None and sender in self.market.buyers:
            self.keeper.save()                            # on disk before Town hears it: nobody sends a contract step, a settlement or a call twice
        self.town.ack(msg["message_id"], msg["fence"], status, note)

    def inbox_loop(self) -> None:
        """The Town inbox, watched with short checks (town.InboxPoller: no held long-poll, which costs NANDA Town 0.2
        about 20 database scans a second). A check every `town_poll_seconds` (1 s) while messages flow, every
        `town_poll_idle_seconds` (3 s) after a quiet minute; a pass that finds a message brings in `inbox_workers`
        claimers (`catchup_workers` until the inbox has been empty once, to clear a backlog after downtime)."""
        while not self.town:
            time.sleep(5)
        cfg = self.cfg

        def ready() -> None:                              # the inbox was empty once: everything old is cleared
            if not self.inbox_ready:
                self.inbox_ready = True
                self.event("inbox_ready", stale_reports_skipped=self.stale_skipped,
                           expired_skipped=self.expired_skipped, seconds=round(time.time() - self.started_wall))
                self.poller.claimers = max(1, int(cfg.get("inbox_workers", 4)))

        def err(where: str, exc: Exception) -> None:
            self.event("town_error", error=f"{where}: {str(exc)[:200]}")
            log.debug(traceback.format_exc())

        self.poller = InboxPoller(self.town, self.handle, every=float(cfg.get("town_poll_seconds", 1.0)),
                                  idle_every=float(cfg.get("town_poll_idle_seconds", 3.0)),
                                  claimers=max(1, int(cfg.get("catchup_workers", 16))), on_error=err, on_empty=ready)
        self.poller.run()


# ---- the live run's brake (POST /demo/pace): when the clock must stay at 1x because agents are talking. These read
# only what the coordinator and its monitor already know; tests/test_pace.py checks them directly.
def day_label(day: str) -> str:
    """'2026-07-03' -> '3 Jul'."""
    d = datetime.strptime(day[:10], "%Y-%m-%d")
    return f"{d.day} {d:%b}"


TOWN_LAG_BRAKE_S = 8.0       # messages reach the Mac's inboxes this late: 1x
TOWN_LAG_HOLD_S = 20.0       # this late: slower than 1x (2x, 4x, 8x the seconds a step), so the data's time does not run
                             # ahead of its messages


def town_hold(lag_s: float, slow_s: float, brake_s: float = TOWN_LAG_BRAKE_S,
              hold_s: float = TOWN_LAG_HOLD_S) -> tuple[float, str] | None:
    """How the live clock waits for a NANDA Town that is behind: (seconds a step, why) or None. Town hands each inbox
    its messages oldest first, so an urgent one (a grid call, the homes' answers) waits behind every meter reading
    sent before it; at 1x (12 s a step) a 3-minute backlog is 4 dataset hours (seen live 1 Nov 2026: a call reached
    the coordinator after its hour was over). The steps are 1x, then 2x, 4x and 8x slower as the backlog grows, so the
    pace changes only when the backlog crosses a level."""
    if lag_s < brake_s:
        return None
    if lag_s < hold_s:
        return slow_s, f"NANDA Town is {lag_s:.0f} s behind: 1x until its messages catch up"
    k = 2.0 if lag_s < 2 * hold_s else 4.0 if lag_s < 4 * hold_s else 8.0
    return slow_s * k, f"NANDA Town is {lag_s:.0f} s behind: the clock waits for its messages ({k:g}x slower)"


def readings_hold(due: int, slow_s: float) -> tuple[float, str] | None:
    """How the live clock waits for readings the sale is still owed: (seconds a step, why) or None. town_hold follows arrivals, so it lets go when they stop and when
    a backlog drains in bursts (14 Jul: from 200 s a step back to 25 s with 340 s of backlog, into the readings deadline with 300 of 1,200 in). This follows who owes
    what, 6 readings a node. It slows the clock and never stops it, so a node that's gone holds it only until the market's own deadline."""
    if due <= 0:
        return None
    k = 2.0 if due < 12 else 4.0 if due < 60 else 8.0
    return slow_s * k, f"{due} readings still due: the clock waits for them ({k:g}x slower)"


def opens_brake(answered: str | None, now_ts: str, soon: str) -> bool:
    """Every day at 10:00 a market asks its buyer for tomorrow's request (the coordinator the utility, the data centers'
    desk the grid operator). Hold 1x from just before 10:00 until the answer is in (it says whether tomorrow is an
    event day, and that day's market starts at once), or until 16:00 without one. `answered`: the last day whose
    10:00 question the buyer has answered; `soon`: the step a slow-down can still reach in time."""
    d = soon[:10]
    return soon[11:16] >= "10:00" and now_ts < f"{d} 16:00" and (answered or "") < d


def dc_brake(d: dict[str, Any] | None, day: str | None, now_ts: str, soon: str) -> str | None:
    """The data centers' own market (flexgrid-dc: the desk, the grid operator and the ten sites over NANDA Town, none
    of it in the coordinator's wire logs), from the record the monitor last read (GET /dc/state, every poll). Hold 1x
    while the desk asks the grid operator for tomorrow's request, while any day's market waits for offers, the contract
    or the data centers' answers (on the event day at 10:00 the next day's opens), and while the run's own event
    waits for its meter readings and settlement."""
    if not isinstance(d, dict):
        return None
    n = len(d.get("sites") or []) or 10
    if d.get("operator_busy"):                     # the grid operator's AI agent orders, or checks a settlement
        return f"the grid operator's AI agent is {d['operator_busy']}"
    if "discover_answered" in d and opens_brake(d.get("discover_answered"), now_ts, soon):
        return "the data centers' market opens: the desk asks the grid operator for its request"
    e = d.get("event") if isinstance(d.get("event"), dict) else None
    if e and day and e.get("day") == day:
        ph, w = e.get("phase"), e.get("window") or [None, None]
        if ph == "offers":
            return f"the data centers' market: {e.get('offers', 0)} of {n} offers in"
        if ph in ("clearing", "cleared", "utility"):
            return "the data centers' desk and the grid operator sign the contract"
        if ph == "homes" and (e.get("accepted", 0) + e.get("declined", 0)) < e.get("offers", 0):
            return "the data centers are accepting their commitments"
        if w[1] and (w[1] <= soon or now_ts >= w[1]) and ph in ("homes", "settling") and e.get("seller_usd") is None:
            return "the data centers' meter readings and settlement"
    for eid, s in sorted((d.get("events") or {}).items()):
        w, cb = s.get("window") or [None, None], s.get("commit_by")
        if (e and day and e.get("day") == day and eid == e.get("id")) or not w[0] or not cb \
                or now_ts >= beckn.add_minutes(cb, 15):
            continue                       # the run's own event (above), or past the deadline its market ends by
        ph, lab = s.get("phase"), f"the data centers' market for {day_label(w[0])}"
        if ph == "offers":
            return f"{lab}: {s.get('offers', 0)} of {n} offers in"
        if ph in ("clearing", "cleared", "utility"):
            return f"{lab}: the desk and the grid operator sign the contract"
        if ph == "homes" and (s.get("accepted", 0) + s.get("declined", 0)) < s.get("committed", 0):
            return f"{lab}: the data centers are accepting their commitments"
    return None


def code_stamp() -> str:
    """A short hash of flexgrid_core/*.py as this process found them at its start (GET /health "code"): which code a
    running coordinator has, to compare with the FlexGrid folder after an update."""
    import hashlib
    h = hashlib.sha1()
    for p in sorted((config.REPO_DIR / "flexgrid_core").glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def _admin_restart(c: Coordinator, request: Any) -> dict[str, Any]:
    """POST /admin/restart (see there): checks, then mac/restart_mac.sh in its own session, a second from now."""
    return admin_restart(c.cfg, request, c.event, "POST /admin/restart")


def admin_restart(cfg: dict[str, Any], request: Any, event: Any, how: str) -> dict[str, Any]:
    """The Mac's restart, asked over loopback: the coordinator's endpoint, and flexgrid-dc's (which stays up when the
    coordinator could not start, e.g. NANDA Town down at its start, 1 Nov 2026). Only on the Mac (macOS)."""
    import os
    import subprocess
    import sys
    from fastapi import HTTPException
    if sys.platform != "darwin":
        raise HTTPException(404, "the restart is the Mac's (mac/restart_mac.sh)")
    host = (request.client.host if request.client else "") or ""
    if host not in ("127.0.0.1", "::1"):
        raise HTTPException(403, "only from this Mac")
    if request.headers.get("x-flexgrid-admin") != "restart":
        raise HTTPException(403, "the header X-FlexGrid-Admin: restart is required")
    if not cfg.get("admin_restart", True):
        raise HTTPException(403, "turned off (\"admin_restart\": false in agent.json)")
    script = config.REPO_DIR / "mac" / "restart_mac.sh"
    if not script.exists():
        raise HTTPException(404, f"{script} not found")
    log_path = config.REPO_DIR / "logs" / "restart.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"\n== {time.strftime('%Y-%m-%d %H:%M:%S')} restart asked from {host} ({how})\n")
    # detached twice: the script must not be this process's descendant. stop_mac.sh ends this coordinator, and
    # macOS's pkill and pgrep leave out their own ancestors (live, 1 Nov 2026: the old coordinator kept its port
    # and the new one could not start). The outer shell exits at once; the script is then launchd's child.
    p = subprocess.Popen(["/bin/bash", "-c", '( sleep 1; exec /bin/bash "$0" ) >> "$1" 2>&1 < /dev/null &',
                          str(script), str(log_path)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, cwd=os.path.expanduser("~"), start_new_session=True, close_fds=True)
    p.wait(timeout=10)
    event("admin_restart", by=host, script="mac/restart_mac.sh", via=how)
    return {"restarting": True, "script": "mac/restart_mac.sh", "log": "logs/restart.log", "starts_in_s": 1}


def build_app(c: Coordinator):
    from fastapi import Body, FastAPI, HTTPException

    app = FastAPI(title="flexgrid-coordinator", version="0.2.0", dependencies=[doors.door()])
    started_at, code_at_start = time.time(), code_stamp()

    @app.get("/health")
    def health():
        p = getattr(c, "poller", None)
        return {"ok": True, "ready": c.inbox_ready, "agent": c.name, "role": "coordinator", "pubkey": c.pub,
                "started_at": round(started_at, 1), "code": code_at_start,
                "town_calls": p.stats() if p else None,
                "town_lag_s": round(c.town_lag(), 1), "inbox_lag_s": round(c.lag.value(), 1),
                "town_ms": c.town.timing() if c.town is not None and hasattr(c.town, "timing") else None,
                "town": bool(c.town), "doors": doors.state(), "transport": getattr(c.town, "label", None)
                or (f"NANDA Town {c.cfg.get('town_url')} run {c.cfg.get('town_run_id')}" if c.town else None),
                "expired_skipped": c.expired_skipped,
                "pinned_homes": sorted(c.pins), "homes_reporting": sorted(c.latest), "stale_reports_skipped":
                c.stale_skipped, "flows": c.cfg.get("flows"), **keep.health(c.keeper)}

    @app.post("/admin/restart")
    def admin_restart(request: Request):
        """Restart everything on this Mac with the code in ~/flexgrid: mac/restart_mac.sh, exactly what the presenter
        would paste in Terminal (coordinator, utility, data centers, Langflow, tunnels, the fleet on jetson-b, this
        Mac's OpenClaw gateway). Only from this Mac (loopback) and only with the header X-FlexGrid-Admin: restart, which
        a page from another site cannot send without a CORS preflight this server never grants; "admin_restart": false
        in agent.json turns it off. Answers at once; the restart starts a second later (logs/restart.log)."""
        return _admin_restart(c, request)

    @app.post("/report/verify")
    def report_verify(body: dict = Body(...)):
        return c.verify_report(body["from"], body["report"])

    @app.get("/community")
    def community():
        return c.community()

    @app.post("/plan")
    def plan(body: dict = Body(...)):
        return c.plan(body["from"], body["report"], body.get("verify"))

    @app.post("/dispatch/check")
    def dispatch_check(body: dict = Body(...)):
        return c.check(body["proposal"])

    @app.post("/dispatch/send")
    def dispatch_send(body: dict = Body(...)):
        return c.send(body["proposal"], body.get("check"))

    @app.post("/dispatch/manual")
    def dispatch_manual(body: dict = Body(...)):
        try:
            return c.dispatch(body["to"], float(body["delta_kw"]), float(body.get("minutes", 30)),
                              body.get("reason", "manual_test"))
        except RuntimeError as exc:
            raise HTTPException(503, str(exc))

    @app.post("/receipt/verify")
    def receipt_verify(body: dict = Body(...)):
        return c.verify_receipt(body["from"], body["receipt"])

    @app.post("/receipt/record")
    def receipt_record(body: dict = Body(...)):
        return c.record_receipt(body["from"], body["receipt"], body["verify"])

    @app.get("/dispatches")
    def dispatches(n: int = 20):
        return list(c.dispatches.values())[-n:]

    @app.get("/ledger")
    def ledger(n: int = 20):
        rows = c.db.execute("SELECT home, dispatch_id, status, reason, requested, applied, chain, at FROM receipts "
                            "ORDER BY n DESC LIMIT ?", (n,)).fetchall()
        keys = ["home", "dispatch_id", "status", "reason", "requested_kw", "applied_kw", "chain", "at"]
        return [dict(zip(keys, r)) for r in rows]

    @app.get("/events")
    def events(n: int = 50):
        return list(c.events)[-n:]

    grid_cache = {}

    @app.get("/grid/now")
    def grid_now(refresh: bool = False):
        """Today's real MISO grid (flexgrid_core/grid_signals.py): fuel mix, load, the Illinois hub price now and
        today's day-ahead prices with the peak window and the cheapest hours. Cached for 2 minutes. With flexgrid-dc
        running, also each data center agent's plan for today on those prices (`datacenters_today`, GET :7874/dc/today)."""
        from . import grid_signals
        cache = grid_cache.setdefault("c", grid_signals.Cache(ttl=float(c.cfg.get("grid_now_cache_s", 120))))
        if refresh:
            cache.value = None
        out = dict(cache.get())
        if c.cfg.get("monitor_datacenters", True) and out.get("day_ahead"):
            try:                                           # the data center agents plan on this same snapshot
                r = httpx.post(c.monitor.src["datacenters"] + "/dc/today", json=out, timeout=8, headers=doors.headers())
                if r.status_code == 200:
                    out["datacenters_today"] = r.json()
            except Exception:                              # flexgrid-dc not running: the grid answer stands alone
                pass
        return out

    @app.get("/v2/summary")
    def v2_summary():
        return {"now": c.market.now(), "roster": len(c.market.roster), "events": c.market.summary(),
                "reputation": c.market.reputation}

    @app.get("/rt")
    def rt_state():
        """The real-time desk (rt_market.py): calls from the buyer and gaps from homes, what was asked, answered,
        dispatched, confirmed or warned, and what was paid; with the buyer's side of each call (rt_buyer.py)."""
        with c.market.lock:
            desk = c.market.rt.summary()
        buyer: dict[str, Any] = {}
        try:
            r = httpx.get(c.monitor.src["utility"] + "/calls", timeout=4)
            if r.status_code == 200:
                buyer = r.json()
        except Exception as exc:
            buyer = {"error": f"not reached: {str(exc)[:80]}"}
        # how the homes run (each home's telemetry says it: "rolling" for the 5-minute plan), as the monitor last saw
        modes = {}
        for n, by_ts in list(getattr(c.monitor, "tel_seen", {}).items()):
            if by_ts:
                modes[n] = (by_ts[max(by_ts)] or {}).get("control") or "fixed"
        rolling = sum(1 for v in modes.values() if v == "rolling")
        mode = "rolling" if rolling and rolling >= len(modes) / 2 else (c.cfg.get("control_mode") or "fixed")
        return {"now": c.market.now(), "desk": desk, "buyer": buyer, "control_mode": mode,
                "homes_rolling": rolling, "homes_seen": len(modes)}

    @app.get("/v2/events/{event_id}")
    def v2_event(event_id: str, signed: bool = False):
        """One event as the market holds it. signed=true: with every home's signed meter reading as it arrived and the
        keys the coordinator pinned on first contact (the evidence flexgrid_core/verify.py checks)."""
        ev = c.market.events.get(event_id)
        if not ev:
            raise HTTPException(404, "unknown event")
        if signed:
            with c.market.lock:
                return json.loads(json.dumps({**ev, "pins": dict(c.market.pins)}, default=str))
        return {k: v for k, v in ev.items() if k != "readings"} | {
            "readings": {h: {s: {k: v for k, v in r.items() if k != "signed_body"} for s, r in rs.items()}
                         for h, rs in ev["readings"].items()}}

    @app.post("/v2/clear")
    def v2_clear(body: dict = Body(...)):
        with c.market.lock:
            return c.market.clear(body["event_id"])

    @app.post("/v2/commit")
    def v2_commit(body: dict = Body(...)):
        with c.market.lock:
            return c.market.commit_to_utility(body["event_id"])

    @app.post("/v2/settle")
    def v2_settle(body: dict = Body(...)):
        """Split and pay (flow C4). Idempotent: once settled, it returns the stored result and pays nobody again."""
        with c.market.lock:
            ev = c.market.events.get(body["event_id"])
            if not ev:
                raise HTTPException(404, "unknown event")
            if not ev.get("settlement"):
                raise HTTPException(409, "the utility has not settled this event yet")
            return c.market.settle(body["event_id"])

    # ---- Control Room (viz/live): the pages and the live record they read
    from pathlib import Path
    from fastapi.responses import FileResponse, JSONResponse
    repo = Path(__file__).resolve().parents[1]

    def page(name: str):
        """The page as built, with the door's few lines in its head while this machine has a token (doors.page_door): its POST buttons ask a person for it once."""
        from fastapi.responses import HTMLResponse
        f = repo / "viz" / "live" / "dist" / name
        if not f.exists():
            raise HTTPException(404, f"{name} not built: python3 viz/live/build.py")
        return HTMLResponse(doors.page_door(f.read_text(encoding="utf-8")), headers={"Cache-Control": "no-store"})

    @app.get("/monitor")
    def monitor_page():
        return page("monitor.html")

    @app.get("/phone")
    def phone_page():
        return page("phone.html")

    @app.get("/monitor/state")
    def monitor_state():
        return JSONResponse(c.monitor.state(), headers={"Cache-Control": "no-store"})

    # ---- the AI agents: the phone app's Agent tab (each Jetson home's agent, through the Mac's tunnels), the
    # coordinator's and the buyer's state, and the coordinator agent's tools (OpenClaw reaches them over MCP:
    # python -m flexgrid_core.agent_mcp --base http://127.0.0.1:7871/agent/coordinator)
    def _home_url(home: str) -> str | None:
        return c.monitor.src.get(home) if home in c.monitor.polled else None

    def _no_agent(home: str) -> dict[str, Any]:
        return {"enabled": False, "home": home,
                "why": "Only the Tesla and Enphase homes (one on each Jetson) and the building nodes have an AI agent; "
                       "the 98 simulated homes run on FlexGrid's rules."}

    @app.get("/agent/coordinator/state")
    def agent_coordinator_state():
        if c.brain is None:
            return {"enabled": False, "why": "the coordinator's AI agent is off (llm_enabled in agent.json)"}
        return {**c.brain.state(), "buyer": c.buyer.status() if c.buyer else None}

    def _coordinator_tools() -> dict[str, tuple[dict[str, Any], Any]]:
        def market_now():
            with c.market.lock:
                ev = c.monitor._current_event(c.market.now())
                if not ev:
                    return {"now": c.market.now(), "event": None}
                return {"now": c.market.now(), **{k: v for k, v in c.market.summary().get(ev["need"]["event_id"], {}).items()},
                        "margin": c.market.margin_of(ev), "margin_reason": (ev.get("margin_decision") or {}).get("reason")}

        def last_clearing():
            with c.market.lock:
                evs = [e for e in c.market.events.values() if e.get("clear_summary")]
                return evs[-1]["clear_summary"] if evs else {"clearing": None}

        def backtest():
            return c.market._backtest(c.market.now()[:10]) or {"backtest": None}

        def decisions():
            return c.brain.decisions.list(10) if c.brain else []
        P = {"type": "object", "properties": {}, "required": []}
        return {"market_now": ({"description": "The current event: request, phase, offers, commitments, kW sold, margin.",
                                "parameters": P}, market_now),
                "last_clearing": ({"description": "The last clearing: kW sold per half hour, homes asked, battery share, "
                                                  "the rebound guard, the margin and why.", "parameters": P}, last_clearing),
                "backtest": ({"description": "The same homes and clearing replayed on this season's earlier event days: "
                                             "promised vs delivered.", "parameters": P}, backtest),
                "decisions": ({"description": "This agent's latest margin decisions with reasons.", "parameters": P},
                              decisions)}

    @app.get("/agent/coordinator/tools")
    def agent_coordinator_tools():
        return [{"name": n, **spec} for n, (spec, _) in _coordinator_tools().items()]

    @app.post("/agent/coordinator/tool/{name}")
    def agent_coordinator_tool(name: str, body: dict = Body(default={})):
        t = _coordinator_tools().get(name)
        if not t:
            raise HTTPException(404, f"no tool {name}")
        return {"result": t[1](**(body or {}))}

    @app.get("/agent/{home}/state")
    def agent_home_state(home: str):
        url = _home_url(home)
        if not url:
            return _no_agent(home)
        try:
            r = httpx.get(url + "/agent/state", timeout=8)
            return r.json() if r.status_code == 200 else {"enabled": False, "home": home, "why": f"its Jetson answered {r.status_code}"}
        except Exception as exc:                         # noqa: BLE001
            return {"enabled": False, "home": home, "why": f"its Jetson is not reachable ({type(exc).__name__})"}

    @app.post("/agent/{home}/chat")
    def agent_home_chat(home: str, body: dict = Body(...)):
        url = _home_url(home)
        if not url:
            return {**_no_agent(home), "reply": None, "error": "no AI agent in this home"}
        try:
            r = httpx.post(url + "/agent/chat", json={"message": str(body.get("message") or "")[:1000]},
                           timeout=float(c.cfg.get("llm_chat_budget_s", 120)) + 15, headers=doors.headers())
            if r.status_code != 200:
                return {"reply": None, "error": f"its Jetson answered {r.status_code}: {r.text[:160]}"}
            return r.json()
        except Exception as exc:                         # noqa: BLE001
            return {"reply": None, "error": f"its Jetson is not reachable ({type(exc).__name__}: {str(exc)[:120]})"}

    @app.post("/agent/{home}/wishes")
    def agent_home_wishes(home: str, body: dict = Body(...)):
        url = _home_url(home)
        if not url:
            raise HTTPException(404, "no AI agent in this home")
        r = httpx.post(url + "/agent/wishes", json=body or {}, timeout=8, headers=doors.headers())
        if r.status_code != 200:
            raise HTTPException(r.status_code, r.text[:200])
        return r.json()

    @app.get("/monitor/replay/{day}")
    def monitor_replay(day: str):
        f = repo / "results" / f"timeline_{day}.json"
        if not f.exists() or "/" in day:
            raise HTTPException(404, "no recorded timeline for that day (tools/timeline.py)")
        return FileResponse(f, media_type="application/json")

    @app.post("/demo/start")
    def demo_start(body: dict = Body(default={})):
        """Start a live event from the Control Room: what tools/demo_event.py does, through the Mac's tunnels.
        Every agent (coordinator, utility, the fleet, both Jetson daemons, the building nodes' daemons, and flexgrid-dc
        with the data centers)
        jumps to 09:45 the day before the event at the same wall-clock moment, then plays one 15-minute step every
        `step_seconds`."""
        if not c.inbox_ready and not body.get("force"):
            raise HTTPException(503, f"the coordinator is still clearing old messages from its NANDA Town inbox "
                                     f"({c.stale_skipped + c.expired_skipped} so far); the event starts once it "
                                     f"has caught up - try again in a minute")
        busy = [] if body.get("force") else _first_pass()
        if busy:                               # before anything is reset or moved: on 7 Nov two starts reset the buyers and the buildings, lost the fleet, and stopped there
            raise HTTPException(503, f"{', '.join(busy)} still on the first pass over their nodes since a restart, and a reset now would time out. Nothing was changed: "
                                     f"try again in a minute")
        day = body.get("event", config.dataset_meta().get("default_event", "2023-08-24"))
        step = float(body.get("step_seconds", 10))
        try:
            datetime.strptime(day, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(400, "event must be YYYY-MM-DD")
        if not 2 <= step <= 120:
            raise HTTPException(400, "step_seconds must be 2-120")
        at = (datetime.strptime(day, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d") + " 09:45"
        scen = body.get("scenario") or None
        if scen:                               # the conditions of the day the presenter picked (scenarios.py)
            from . import scenarios
            if scen not in scenarios.PRESETS:
                raise HTTPException(400, f"no scenario {scen!r}: one of {', '.join(scenarios.PRESETS)}")
        lead = max(1.0, min(60.0, float(body.get("lead_s") or 15)))     # every agent starts together, this much later
        buyer_orders = c.buyer is not None and not isinstance(body.get("order"), dict) and body.get("buyer_agent", True)
        if buyer_orders:                       # the buyer's AI agent orders before the 10:00 call for offers: the clock
            lead = max(lead, float(c.cfg.get("buyer_order_lead_s", 60)))    # waits at 09:45 while it thinks
        targets = _clock_targets()
        # the grid operator's AI agent (flexgrid-dc) orders the data centers' MW for the day at the same moment. One
        # model on the Mac answers the two in turn, so the clock waits for both
        op_orders = "datacenters" in targets and bool(c.cfg.get("llm_enabled")) and \
            bool(c.cfg.get("dc_operator_llm", True)) and body.get("operator_agent", True) and \
            not isinstance(body.get("dc_order"), dict)
        if op_orders:
            lead += float(c.cfg.get("operator_order_lead_s", 30))
        clock = {"start": at, "at": at, "at_wall": time.time() + lead, "step_seconds": step, "loop_days": 0, **({"hold": True} if c.cfg.get("clock_waits") else {})}
        _pace.update(gen=_pace["gen"] + 1, slow=step, day=day, why=None, brake=False, changes=deque(maxlen=40),
                     start_wall=clock["at_wall"], **_pace_choice(body.get("pace")))
        out = {}
        with httpx.Client(timeout=20, headers=doors.headers()) as http:
            for name, base in targets.items():
                try:
                    http.post(base + "/demo/reset", json={"clear_orders": True} if name == "utility" and buyer_orders
                              else {}).raise_for_status()
                    j = http.post(base + "/demo/clock", json=clock).json()
                    out[name] = j.get("now", "ok")
                    if j.get("campus"):                   # flexgrid-dc in the market role: the data center campus
                        out["dc-campus"] = j["campus"]    # got the same clock (or says it was not reached)
                except Exception as exc:
                    out[name] = f"not reached: {str(exc)[:80]}"
        c.market.reset()
        c.buyer_signed.clear()
        r = c.clock
        r.configure(clock["start"], step, 0, clock["at"], clock["at_wall"], hold=bool(clock.get("hold")))
        _cond.update(since=clock["at_wall"], waits=None, changed=0.0, went_on=deque(maxlen=200), steps=0, held_s=0.0, at_gate=None, unreached=set())   # gates stay: a runner sets them first
        c.monitor.reset()
        c.what_if = {}
        out["coordinator"] = r.ts(r.current())
        # the event's request and price, when the presenter set them: a buyer order to the utility for that day's
        # window, placed now (09:45 the day before, before the 10:00 call for offers), signed by the utility
        order = body.get("order") if isinstance(body.get("order"), dict) else None
        if order:
            try:
                with httpx.Client(timeout=20, headers=doors.headers()) as http:
                    nd = http.get(targets["utility"] + "/needs").json()
                    need = next((x for x in nd.get("needs") or [] if x["window"][0][:10] == day), None)
                    w0, w1 = (need["window"] if need else [f"{day} 17:00", f"{day} 20:00"])
                    o = {"date": day, "start": str(order.get("start") or w0[11:16]), "end": str(order.get("end") or w1[11:16]),
                         "kw": float(order.get("kw") or (need or {}).get("kw") or 100),
                         "price_usd_per_kwh": float(order.get("price_usd_per_kwh") or (need or {}).get("price_usd_per_kwh") or 1.0),
                         "buyer": "Control Room (what if)", "note": "set by the presenter before the run"}
                    rr = http.post(targets["utility"] + "/needs", json=o)
                    out["order"] = rr.json() if rr.status_code == 200 else f"refused: {rr.text[:160]}"
                    if rr.status_code == 200:
                        out["order"] = {k: o[k] for k in ("date", "start", "end", "kw", "price_usd_per_kwh")}
            except Exception as exc:
                out["order"] = f"not placed: {str(exc)[:120]}"
        dc_order = body.get("dc_order") if isinstance(body.get("dc_order"), dict) else None
        if dc_order and "datacenters" in targets:     # the presenter's request for the data centers (MW)
            try:
                with httpx.Client(timeout=10, headers=doors.headers()) as http:
                    rr = http.post(targets["datacenters"] + "/dc/needs", json={"day": day, "mw": float(dc_order.get("mw") or 40)})
                out["dc_order"] = rr.json() if rr.status_code == 200 else f"refused: {rr.text[:160]}"
            except Exception as exc:           # noqa: BLE001
                out["dc_order"] = f"not placed: {str(exc)[:120]}"
        if scen and scen != "normal":          # the day's conditions reach the agents before any of them orders
            try:
                wi = _what_if({"preset": scen})
                out["scenario"] = {"preset": scen, "agents": {k: ("ok" if "error" not in v else v["error"])
                                                              for k, v in (wi.get("agents") or {}).items()}}
            except Exception as exc:           # noqa: BLE001  the run goes on as a normal day
                out["scenario"] = f"not set: {str(exc)[:120]}"
        if op_orders:                          # first: one short decision, then the model is the buyer's
            try:
                with httpx.Client(timeout=10, headers=doors.headers()) as http:
                    rr = http.post(targets["datacenters"] + "/dc/operator/order",
                                   json={"day": day, "budget_s": max(10.0, lead + step - 4)})
                out["operator_agent"] = rr.json() if rr.status_code == 200 else f"refused: {rr.text[:160]}"
            except Exception as exc:           # noqa: BLE001  the program's catalog request stands
                out["operator_agent"] = f"not reached: {str(exc)[:120]}"
        if buyer_orders:                       # in the background: the order lands before 10:00 or the catalog's stands
            try:
                with httpx.Client(timeout=10, headers=doors.headers()) as http:
                    nd = http.get(targets["utility"] + "/needs").json()
                cat = next((x for x in nd.get("needs") or [] if x["window"][0][:10] == day), None)
            except Exception:                  # noqa: BLE001
                cat = None
            from .buyer_agent import limit_price
            lim = limit_price(c.cfg, (cat or {}).get("price_usd_per_kwh"))
            budget = max(10.0, lead + step - 4)
            note = (getattr(c, "what_if", None) or {}).get("note")
            threading.Thread(target=c.buyer.order, args=(day, lim, budget), kwargs={"note": note}, daemon=True).start()
            out["buyer_agent"] = {"ordering": day, "limit_usd_per_kwh": lim, "budget_s": round(budget)}
        c.event("demo_start", event=day, step_seconds=step, starts_at=at, targets=out)
        try:
            _talk_start()
        except Exception as exc:
            out["talk"] = f"transcript not started: {str(exc)[:120]}"
        try:
            _term_mark()
        except Exception as exc:
            out["terminals"] = f"not marked: {str(exc)[:120]}"
        _pace_start()
        return {"event": day, "starts_at": at, "step_seconds": step, "wall_start": clock["at_wall"],
                "minutes_total": round((96 + 57) * step / 60, 1), "agents": out, "pace": _pace_view()}

    def _first_pass() -> list[str]:
        """The node services still on their first pass since they started: /health says ready: false, or is too busy to say in 6 s, which a reset wouldn't get past either. One that
        doesn't answer at all is "not reached" below, as before. The reads go at once, so this takes 6 s at most whatever the links do: live_bench's own call to here waits 30."""
        from concurrent.futures import ThreadPoolExecutor
        targets = _clock_targets()
        names = [n for n in ("fleet", *c.monitor.bnames) if n in targets]

        def busy(name: str) -> bool:
            try:
                return httpx.get(targets[name] + "/health", timeout=6).json().get("ready") is False
            except httpx.TimeoutException:
                return True
            except (httpx.HTTPError, ValueError, AttributeError):
                return False
        with ThreadPoolExecutor(max_workers=max(1, len(names))) as pool:
            return [n for n, b in zip(names, pool.map(busy, names)) if b]

    def _clock_targets() -> dict[str, str]:
        """Every agent that keeps the replay clock, besides the coordinator: the utility, the fleet, both Jetson homes,
        the building nodes (cfg "buildings") and flexgrid-dc with the data centers (when the dataset has them: Illinois
        2023, Massachusetts 2026)."""
        from . import datacenter as dc
        t = {"utility": c.monitor.src["utility"], "fleet": c.monitor.src["fleet"],
             "home-tesla": c.monitor.src["home-tesla"], "home-enphase": c.monitor.src["home-enphase"],
             "datacenters": c.monitor.src["datacenters"]}
        if not dc.available() or not c.cfg.get("monitor_datacenters", True):   # this dataset has no data center node
            t.pop("datacenters")
        for b in c.monitor.bnames:                        # the building nodes (their own daemons, building.py)
            t[b] = c.monitor.src[b]
        t.update(c.cfg.get("clock_extra_targets") or {})   # a clock the coordinator doesn't know by itself: the second buyer, a node in a process of its own
        return t

    # ---- What if (the Control Room): change the homes' weather, take a Jetson offline, trip batteries - live
    HOMES_AT = {"home-tesla": "jetson-a", "home-enphase": "jetson-b", "fleet": "jetson-b",
                **{b["name"]: b.get("host") for b in c.monitor.bnodes}}

    @app.get("/demo/scenario")
    def what_if_get():
        from . import scenarios
        return {"what_if": getattr(c, "what_if", {}), "now": c.clock.ts(c.clock.current()), "presets": scenarios.catalog()}

    @app.post("/demo/scenario")
    def what_if(body: dict = Body(default={})):
        """Forward one what-if to the agents that it concerns and log it (it shows in the Control Room's feed and the
        agents' conversation). Body: {"preset": "heat"|"cool"|"dc_busy"|"battery_trouble"|"normal"} (scenarios.py:
        everything back to normal, then the preset's changes); or any of
              {"weather": {"pv": 0..1.5, "load": 0.5..2}} (all 100 homes, from now on);
              {"offline": {"home": "home-tesla"|"home-enphase", "seconds": 0..1800}} (0 = back now);
              {"battery_fault": {"home": "home-tesla"|"home-enphase"|"fleet", "n": 10, "on": true}};
              {"region": {"load_x": 0.8..1.25, "note": "..."}} (the grid operator's forecast of the region and what the
              weather service says: its AI agent reads them before it orders);
              {"datacenters": {"urgent_share": 0..0.9, "battery_out": ["dc-03"]}} (the data centers' conditions);
              {"note": "..."} (what the utility's buyer agent hears about tomorrow before it orders);
              {"owner": {"home": "bld-library", "kind": "comfort"|"lights"|"stop"|"clear", "add_c": 0..3,
              "share": 0..0.5}} (a building node's facility manager; a home's owner: kind ev_now|comfort|stop|clear);
              {"reset": true}. The agents react with their own code; nothing here decides for them."""
        return _what_if(body or {})

    def _what_if(body: dict[str, Any]) -> dict[str, Any]:
        from . import scenarios
        if body.get("preset"):
            try:
                body = scenarios.expand(body["preset"])
            except ValueError as exc:
                raise HTTPException(400, str(exc))
        src = c.monitor.src
        sent: dict[str, Any] = {}
        calls: list[tuple[str, dict]] = []
        nodes = ("fleet", "home-tesla", "home-enphase", *c.monitor.bnames)      # every node that lives in the weather
        if body.get("reset"):
            calls += [(k, {"reset": True}) for k in nodes]
        if isinstance(body.get("weather"), dict):
            w = {k: float(v) for k, v in body["weather"].items() if k in ("pv", "load") and v is not None}
            calls += [(k, {"weather": w}) for k in nodes]
        if isinstance(body.get("offline"), dict):
            o = body["offline"]
            o["seconds"] = max(0.0, min(2700.0, float(o.get("seconds") or 0)))
            if o.get("home") not in ("home-tesla", "home-enphase"):
                raise HTTPException(400, "offline: home must be home-tesla or home-enphase (the Jetson homes)")
            calls.append((o["home"], {"offline_s": float(o.get("seconds") or 0)}))
        if isinstance(body.get("battery_fault"), dict):
            b = body["battery_fault"]
            if b.get("home") == "fleet":
                calls.append(("fleet", {"battery_fault": {"n": int(b.get("n") or 10), "on": bool(b.get("on", True))}}))
            elif b.get("home") in ("home-tesla", "home-enphase"):
                calls.append((b["home"], {"battery_fault": bool(b.get("on", True))}))
            else:
                raise HTTPException(400, "battery_fault: home must be home-tesla, home-enphase or fleet")
        # the real-time conditions (the audience's "right now" panel): each reaches the agents it concerns, which react
        # with their own code at their next 5-minute step
        util_posts: list[tuple[str, dict]] = []
        wn = body.get("weather_now")
        if isinstance(wn, dict):
            kind = wn.get("kind")
            if kind == "storm":
                w = {"pv": 0.2, "dT": -6.0, "load": 0.8, "hours": float(wn.get("hours") or 2)}
            elif kind == "heat":
                w = {"pv": 1.0, "dT": 3.0, "load": 1.12, "hours": float(wn.get("hours") or 3)}
            elif kind == "clear":
                w = {"pv": 1.0, "dT": 0.0, "load": 1.0}
            else:
                raise HTTPException(400, "weather_now: kind storm|heat|clear")
            calls += [(k, {"weather": w}) for k in nodes]
        if isinstance(body.get("price_spike"), dict):
            ps = body["price_spike"]
            util_posts.append(("/demo/price", {"factor": max(1.0, min(4.0, float(ps.get("factor") or 3))),
                                               "hours": max(0.25, min(2.0, float(ps.get("hours") or 1)))}))
        if isinstance(body.get("emergency"), dict):
            em = body["emergency"]
            util_posts.append(("/alert", {"pct": max(10.0, min(100.0, float(em.get("pct") or 50))),
                                          "minutes": max(15.0, min(180.0, float(em.get("minutes") or 60))),
                                          "why": str(em.get("why") or "grid emergency (Control Room)")[:120]}))
        if isinstance(body.get("owner"), dict) and body["owner"].get("home") in c.monitor.bnames:
            # a building's facility manager: how much warmer (0-3 C while open), how much of the lights may be dimmed
            # (0-50 %), leave the event, back to the plan. The building's code clamps every value
            ow = body["owner"]
            kind = ow.get("kind")
            if kind == "comfort":
                add = max(0.0, min(3.0, float(ow.get("add_c") if ow.get("add_c") is not None else 1.5)))
                val = {"max_rise_c": add, "max_rise_away_c": max(add, min(4.0, add + 1.5))}
            elif kind == "lights":
                val = max(0.0, min(0.5, float(ow.get("share") if ow.get("share") is not None else 0.3)))
            elif kind in ("stop", "clear"):
                val = True
            else:
                raise HTTPException(400, "owner (a building): kind comfort|lights|stop|clear")
            calls.append((ow["home"], {"override": {"kind": kind, "value": val,
                                                    "source": str(ow.get("source") or "control room")[:40]}}))
        elif isinstance(body.get("owner"), dict):        # an owner's wish: the car now, a comfort limit, leave the event
            ow = body["owner"]
            home, kind = ow.get("home"), ow.get("kind")
            if kind == "comfort":
                add = max(0.0, min(4.0, float(ow.get("add_c") or 0)))
                val: Any = {"max_rise_c": 2.0 + add, "max_rise_away_c": 4.0 + add}
            elif kind in ("ev_now", "stop", "clear"):
                val = True
            else:
                raise HTTPException(400, "owner: kind ev_now|comfort|stop|clear")
            ov = {"kind": kind, "value": val, "source": str(ow.get("source") or "control room")[:40]}
            if home in ("home-tesla", "home-enphase"):
                calls.append((home, {"override": ov}))
            elif isinstance(home, str) and home.startswith("home-"):
                calls.append(("fleet", {"override": {**ov, "home": home}}))
            else:
                raise HTTPException(400, "owner: home must be one of the 100 homes")
        # the data centers' side (flexgrid-dc): the grid operator's forecast of the region and the sites' conditions
        dcb: dict[str, Any] = {}
        if body.get("reset"):
            dcb["reset"] = True
        if isinstance(body.get("region"), dict):
            rg = body["region"]
            dcb["operator"] = {"region_load_x": float(rg.get("load_x") or 1.0), "note": str(rg.get("note") or "")[:300]}
        if isinstance(body.get("datacenters"), dict):
            d = body["datacenters"]
            if any(k in d for k in ("urgent_share", "battery_out")):
                dcb["sites"] = {k: d[k] for k in ("urgent_share", "battery_out") if k in d}
            if isinstance(d.get("urgent_job"), dict):     # the operator's urgent job: a site, MW, hours
                dcb["urgent_job"] = {"site": str(d["urgent_job"].get("site") or "dc-03"),
                                     "mw": max(1.0, min(10.0, float(d["urgent_job"].get("mw") or 5))),
                                     "hours": max(1.0, min(4.0, float(d["urgent_job"].get("hours") or 2)))}
        note = str(body.get("note") or "").strip()[:400] if "note" in body else None
        if not calls and not dcb and note is None and not util_posts:
            raise HTTPException(400, "nothing to change: give a preset, weather, weather_now, price_spike, emergency, "
                                     "owner, offline, battery_fault, region, datacenters, note or reset")
        dc_url = _clock_targets().get("datacenters")
        if (dcb.get("operator") or dcb.get("sites") or dcb.get("urgent_job")) and not dc_url:
            sent["datacenters"] = {"error": "no data centers in this dataset"}
        with httpx.Client(timeout=15, headers=doors.headers()) as http:
            for name, payload in calls:
                try:
                    r = http.post(src[name] + "/demo/scenario", json=payload)
                    r.raise_for_status()
                    sent.setdefault(name, {}).update(r.json())
                except Exception as exc:
                    sent.setdefault(name, {})["error"] = f"not reached: {str(exc)[:100]}"
            for path, payload in util_posts:
                try:
                    r = http.post(c.monitor.src["utility"] + path, json=payload)
                    r.raise_for_status()
                    sent.setdefault("utility", {})[path.strip("/")] = r.json()
                except Exception as exc:
                    sent.setdefault("utility", {})["error"] = f"not reached: {str(exc)[:100]}"
            if dcb and dc_url:
                try:
                    r = http.post(dc_url + "/dc/conditions", json=dcb)
                    r.raise_for_status()
                    j = r.json()
                    sent["datacenters"] = j if not str(j.get("campus", "ok")).startswith("not reached") else \
                        {**j, "error": f"the campus: {j['campus']}"}
                except Exception as exc:
                    sent["datacenters"] = {"error": f"not reached: {str(exc)[:100]}"}
        wi = getattr(c, "what_if", None)
        if wi is None or body.get("reset"):
            wi = c.what_if = {}
        if body.get("preset"):
            wi["preset"] = body["preset"]
        if "weather" in body:
            wi["weather"] = body["weather"]
        if "offline" in body:
            wi.setdefault("offline", {})[body["offline"]["home"]] = {"seconds": body["offline"].get("seconds"), "at": time.time()}
        if "battery_fault" in body:
            wi.setdefault("battery_fault", {})[body["battery_fault"]["home"]] = bool(body["battery_fault"].get("on", True))
        if "region" in body:
            wi["region"] = body["region"]
        if "datacenters" in body:
            wi["datacenters"] = {**(wi.get("datacenters") or {}), **body["datacenters"]}
        if note is not None:
            wi["note"] = note or None
        for k in ("weather_now", "price_spike", "emergency"):
            if k in body:
                wi[k] = {**body[k], "at": c.clock.ts(c.clock.current())}
        if "owner" in body:
            wi.setdefault("owners", {})[body["owner"].get("home")] = {**body["owner"], "at": c.clock.ts(c.clock.current())}
        c.event("what_if", change={k: v for k, v in body.items() if k in ("preset", "weather", "offline", "battery_fault",
                                                                         "region", "datacenters", "note", "reset",
                                                                         "weather_now", "price_spike", "emergency", "owner")},
                agents={k: ("ok" if "error" not in v else v["error"]) for k, v in sent.items()})
        return {"now": c.clock.ts(c.clock.current()), "agents": sent, "what_if": wi, "hosts": HOMES_AT}

    # ---- every signed message the Mac's agents sent or received over NANDA Town (the wire logs), for the Control Room
    _logs_dir = __import__("pathlib").Path(c.cfg.get("wire_logs_dir") or __import__("pathlib").Path(__file__).resolve().parents[1] / "logs")
    _wire_files = [_logs_dir / f for f in ("town_wire.jsonl", "town_wire_utility.jsonl")]
    _links: dict[str, Any] = {"at": 0.0, "data": {}}

    def _tail(path, max_bytes: int = 600_000) -> list[dict]:
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - max_bytes))
                lines = f.read().decode("utf-8", "replace").splitlines()
            if size > max_bytes:
                lines = lines[1:]                         # the first line may be cut
        except FileNotFoundError:
            return []
        out = []
        for ln in lines:
            try:
                out.append(json.loads(ln))
            except ValueError:
                pass
        return out

    # ---- Speed of a live run (the Control Room): 1x-8x, or Auto, changed on every agent at the same step.
    # 1x is the run's step_seconds (12 s per 15 minutes on NANDA Town). Auto plays the quiet hours at 12x and the
    # delivery window at 2x. A brake holds 1x whenever messages need time: Town delivers in 2-20 s of real time and
    # the market, the pre-charge notice, the readings and settlement have deadlines in dataset time, so a faster
    # clock would turn a 20 s delivery into hours of dataset time. The homes' physics needs no messages: a what-if
    # (weather, a battery trip) plays at any speed.
    import re
    PACE_MIN_S = float(c.cfg.get("pace_min_seconds", 1.0))
    _pace: dict[str, Any] = {"x": 1.0, "auto": False, "slow": None, "day": None, "why": None, "gen": 0,
                             "pending": False, "loop": False, "changes": deque(maxlen=40)}
    c.pace = _pace                                        # keep.py saves the run's part of it
    _T_RE = re.compile(r'^\{"t":([0-9.]+)')

    def _pace_choice(p: Any) -> dict[str, Any]:
        """{"x": 1|2|4|8, "auto": bool} from a request: "auto", a number, or a dict."""
        if p == "auto":
            return {"x": 8.0, "auto": True}
        if isinstance(p, (int, float)):
            return {"x": max(1.0, min(8.0, float(p))), "auto": False}
        if isinstance(p, dict):
            return {"x": max(1.0, min(8.0, float(p.get("x") or 1.0))), "auto": bool(p.get("auto"))}
        return {"x": 1.0, "auto": False}

    def _wire_recent(ids: list[str] | None, quiet: float) -> bool:
        """A message about one of these events (any message while there is none) on either wire log in the last
        `quiet` seconds. Meter readings do not count: they wait for the window's end (and the brake holds from there)."""
        now = time.time()
        for p in _wire_files:
            try:
                if now - p.stat().st_mtime > quiet:
                    continue
                with open(p, "rb") as f:
                    f.seek(0, 2)
                    size = f.tell()
                    start = max(0, size - 48_000)
                    f.seek(start)
                    lines = f.read().decode("utf-8", "replace").splitlines()
                if start > 0:
                    lines = lines[1:]
            except OSError:
                continue
            for ln in reversed(lines):
                m = _T_RE.match(ln)
                if not m:
                    continue
                if now - float(m.group(1)) > quiet:
                    break
                if '"usage_kw"' in ln:
                    continue
                if not ids or any(x in ln for x in ids):
                    return True
        return False

    def _next_contracted_day(day: str) -> str | None:
        """An event after the run's day that the run already contracted (its market ran during the run: the homes have
        their final commitments) and that is not settled yet: the clock keeps the run's pace through it."""
        with c.market.lock:
            days = sorted(e["need"]["window_start"][:10] for e in c.market.events.values()
                          if e["need"]["window_start"][:10] > day and e.get("final_sent") and e.get("phase") != "done")
        return days[0] if days else None

    def _run_event(day: str | None = None) -> dict[str, Any] | None:
        """The live run's event (what the market knows of it), or None before the utility's request arrives. day: an
        event the run follows on into (_next_contracted_day) instead of the run's own."""
        day = day or _pace.get("day")
        with c.market.lock:
            evs = list(c.market.events.values())
            if day:
                ev = next((e for e in evs if e["need"]["window_start"][:10] == day), None)
            else:
                now = c.clock.ts(c.clock.current())
                live = [e for e in evs if now <= beckn.add_minutes(e["need"]["window_end"], 6 * 60)]
                ev = min(live, key=lambda e: e["need"]["window_start"]) if live else None
            if not ev:
                return None
            return {"need": dict(ev["need"]), "phase": ev.get("phase"), "offers": len(ev.get("offers") or {}),
                    "margin_thinking": bool(ev.get("margin_asked")) and "margin" not in ev,
                    **{k: ev.get(k) for k in ("final_sent", "telemetry_sent", "notice_sent", "finalized")}}

    def _other_markets(run_eid: str | None, now_ts: str) -> list[dict[str, Any]]:
        """Other days' events still in their market (on the event day at 10:00 the next day's opens): their offers,
        the utility's contract and the homes' commitments need NANDA Town as much as the run's own. Past its deadline
        (commit_by) every one of those phases has ended, so an event stuck there cannot hold the clock."""
        out = []
        with c.market.lock:
            for e in c.market.events.values():
                n = e["need"]
                if n["event_id"] == run_eid or now_ts >= beckn.add_minutes(n["commit_by"], 15):
                    continue
                ph = e.get("phase")
                if ph in ("offers", "clearing", "cleared", "utility") or (ph == "homes" and not e.get("final_sent")):
                    out.append({"id": n["event_id"], "day": n["window_start"][:10], "phase": ph,
                                "offers": len(e.get("offers") or {})})
        return sorted(out, key=lambda x: x["day"])

    def _pace_brake(ev: dict[str, Any] | None) -> str | None:
        """Why the clock must run at 1x now (messages need time), or None."""
        r = c.clock
        i = r.current()
        last = len(r.rows) - 1
        now_ts, soon = r.ts(i), r.ts(min(i + 3, last))    # a slow-down takes effect up to 2 fast steps later
        eid = ev["need"]["event_id"] if ev else None
        others = _other_markets(eid, now_ts)
        ids = [x for x in [eid] + [o["id"] for o in others] if x]
        if _wire_recent(ids or None, float(c.cfg.get("pace_quiet_seconds", 8.0))):
            return "messages are on their way over NANDA Town"
        if c.buyer is not None and c.buyer.busy:          # the buyer's AI agent ordering, or checking the settlement
            return f"the buyer's AI agent is {c.buyer.busy}"
        if ev and ev.get("margin_thinking"):
            return "the market: the coordinator's AI agent picks its margin"
        rt_open = [r for r in c.market.rt.calls.values() if r.get("phase") == "asking"]
        if rt_open:                                       # a call or a gap: homes are answering, at 1x so it is seen
            r0 = rt_open[0]
            return ("a grid call: the homes are answering" if r0["kind"] != "gap" else
                    "a gap: the coordinator asks other homes to cover it")
        day = (ev["need"]["window_start"][:10] if ev else None) or _pace.get("day")   # the event followed on into, too
        dcb = dc_brake(getattr(c.monitor, "dc", None), day, now_ts, soon)
        if dcb:
            return dcb
        if opens_brake(c.market.last_discover_day, now_ts, soon):
            tomorrow = beckn.add_minutes(f"{soon[:10]} 00:00", 1440)[:10]
            return ("the market opens: the coordinator asks the utility for its request" if tomorrow == day else
                    f"the market for {day_label(tomorrow)} opens: the coordinator asks the utility for its request")
        for o in others:
            lab = f"the market for {day_label(o['day'])}"
            if o["phase"] == "offers":
                return f"{lab}: {o['offers']} offers in, waiting for the rest"
            if o["phase"] in ("clearing", "cleared", "utility"):
                return f"{lab}: the coordinator and the utility sign the contract"
            return f"{lab}: homes are accepting their commitments"
        if not ev:
            return None
        n = ev["need"]
        w0, w1 = n["window_start"], n["window_end"]
        ph = ev.get("phase")
        if ph == "offers":
            return f"the market: {ev.get('offers', 0)} offers in, waiting for the rest"
        if ph in ("clearing", "cleared", "utility"):
            return "the market: the coordinator and the utility sign the contract"
        if ph == "homes" and not ev.get("final_sent"):
            return "the market: homes are accepting their commitments"
        notice = beckn.add_minutes(w0, -120)
        if notice <= soon and now_ts < beckn.add_minutes(notice, 30):
            return "the pre-charge notice to the homes"
        if (w1 <= soon or now_ts >= w1) and ev.get("final_sent") and not ev.get("telemetry_sent") and ph == "homes":
            return "meter readings to the utility"
        if ph == "settling" or (ev.get("telemetry_sent") and ph == "homes"):
            return "settlement and payments"
        return None

    def _pace_want() -> tuple[float, str | None, str | None]:
        """(seconds per step wanted now, why the brake holds 1x or None, what Auto is doing)."""
        S = float(_pace.get("slow") or c.clock.step_seconds)
        hold = town_hold(c.town_lag(), S) if c.cfg.get("pace_town_hold", True) else None
        due = readings_hold(c.market.readings_due(), S) if c.cfg.get("pace_town_hold", True) else None
        hold = max(hold, due, key=lambda x: x[0]) if hold and due else hold or due
        if hold:                          # NANDA Town is behind, or readings are still due: the clock waits for the messages, whatever else
            return hold[0], hold[1], None
        day = _pace.get("day")
        if day and c.clock.ts(c.clock.current()) >= beckn.add_minutes(f"{day} 23:45", 15):
            # past the run's day: on at the run's pace through an event the run already contracted (the next day's,
            # whose market ran during the run) until it is settled, then 1x (not on into the following days at 8x)
            nxt = _next_contracted_day(day) if c.cfg.get("pace_follow_next", True) else None
            if not nxt:
                return S, None, "the run is over, so back to 1x"
            ev = _run_event(nxt)
        else:
            ev = _run_event()
        brake = _pace_brake(ev)
        if brake:
            return S, brake, None
        if _pace.get("auto"):             # the quiet hours at 12x (1 s a step), the delivery window at 2x (6 s a step)
            r = c.clock
            i = r.current()
            now_ts, soon = r.ts(i), r.ts(min(i + 3, len(r.rows) - 1))
            if ev and ev.get("final_sent") and ev["need"]["window_start"] <= soon and now_ts < ev["need"]["window_end"]:
                return max(PACE_MIN_S, S / float(c.cfg.get("pace_auto_window_x", 2))), None, "the delivery window, slower so you can watch it"
            return max(PACE_MIN_S, S / float(c.cfg.get("pace_auto_quiet_x", 12))), None, "nobody needs to talk"
        return max(PACE_MIN_S, S / float(_pace.get("x") or 1.0)), None, None

    def _retime(new: float, why: str, gen: int) -> None:
        """Change the live clock's pace on every agent at the same step boundary (Replay.retime_plan: when to tell
        them so that none steps back). `keep`: the agents go on with their state (a new run resets it)."""
        r = c.clock
        old = float(r.step_seconds)
        if abs(new - old) < 1e-6 or r.at_wall is None or r.loop_days:
            return
        plan = r.retime_plan(new)
        time.sleep(max(0.0, plan["send"] - time.time()))
        if gen != _pace["gen"] or abs(float(r.step_seconds) - old) > 1e-6:
            return                                         # a new run started, or the pace changed meanwhile
        if new < old and _pace_want()[0] > new + 1e-6:
            return                                         # something to wait for came up meanwhile: stay
        body = {k: plan[k] for k in ("start", "at", "at_wall", "step_seconds", "loop_days")}
        body["keep"] = True
        res: dict[str, Any] = {}

        def post(name: str, base: str) -> None:
            try:
                rr = httpx.post(base + "/demo/clock", json=body, timeout=3, headers=doors.headers())
                rr.raise_for_status()
                res[name] = rr.json().get("now", "ok")
            except Exception as exc:
                res[name] = f"not reached: {str(exc)[:80]}"
        ths = [threading.Thread(target=post, args=kv, daemon=True) for kv in _clock_targets().items()]
        for t in ths:
            t.start()
        r.configure(plan["start"], new, 0, plan["at"], plan["at_wall"])
        for t in ths:
            t.join(4)
        _pace["changes"].append({"wall": plan["at_wall"], "sim": plan["at"], "step_seconds": new, "why": why})
        c.event("pace", step_seconds=new, was=old, sim=plan["at"], why=why, agents=res)

    def _pace_tick() -> None:
        if c.cfg.get("clock_waits"):                      # the clock waits for the agents: the conductor sets each step, and no pace applies
            return _conductor_tick()
        if _pace["pending"] or not _pace.get("slow") or c.clock.at_wall is None \
                or time.time() < float(_pace.get("start_wall") or 0) + 0.5:      # the run has not started yet
            return
        want, brake, auto = _pace_want()
        _pace["why"] = brake or auto
        _pace["brake"] = bool(brake)
        if abs(want - float(c.clock.step_seconds)) < 1e-6:
            return
        gen = _pace["gen"]
        _pace["pending"] = True

        def run() -> None:
            try:
                _retime(want, brake or auto or f"{_pace.get('x', 1):g}x", gen)
            except Exception:
                log.error("pace: %s", traceback.format_exc())
            finally:
                _pace["pending"] = False
        threading.Thread(target=run, daemon=True).start()

    def _clock_again() -> None:
        """An agent that restarted came back on the line it saved. If the pace changed while it was away, nothing tells it again, and it plays on at the old speed. Any
        agent on another line gets this one, and goes on with its state (keep). An agent that doesn't answer is left for the next round: _retime reports those."""
        r = c.clock
        wall0, step, i0, loop, held = r._line
        if r.at_wall is None or loop or _pace["pending"] or _cond.get("posting"):
            return
        at = r.ts(i0)
        for name, base in _clock_targets().items():
            try:
                got = httpx.get(base + "/demo/clock", timeout=3).json()
                if abs(float(got["at_wall"] or 0.0) - wall0) > 0.5 or abs(float(got["step_seconds"]) - step) > 1e-6 or bool(got.get("held")) != (held is not None) \
                        or (held is not None and got.get("now") != at):
                    httpx.post(base + "/demo/clock", timeout=3, headers=doors.headers(), json={"start": at, "at": at, "at_wall": wall0, "step_seconds": step, "loop_days": 0, "keep": True,
                                                                                              **({"hold": True} if held is not None else {})}).raise_for_status()
                    c.event("clock_sent_again", agent=name, was=got.get("now"), now=r.ts(r.current()))
            except (httpx.HTTPError, KeyError, TypeError, ValueError):
                continue

    # ---- The clock waits for the agents (cfg clock_waits; agentic/DESIGN.md section 3). Every clock holds at a step until every node service has ticked it
    # and no agent there is still deciding, and until nothing the market waits on is still coming: offers, a contract, commitments, a call's answers, readings,
    # settlement. Then all of them move one step together. Waits that don't move for clock_wait_idle_s let the clock go on, and that is on the record.
    _cond: dict[str, Any] = {"since": 0.0, "waits": None, "changed": 0.0, "went_on": deque(maxlen=200), "steps": 0, "held_s": 0.0, "posting": False,
                             "gates": set(), "at_gate": None, "unreached": set()}
    c.cond = _cond

    def _agent_services() -> list[str]:
        """The node services' base URLs, where /agents says who is still deciding: cfg clock_wait_urls, or the fleet's and the buildings' own."""
        urls = list(c.cfg.get("clock_wait_urls") or [])
        if not urls:
            src = c.monitor.src
            if src.get("fleet"):
                urls.append(src["fleet"])
            for b in c.monitor.bnames:
                base = str(src.get(b) or "").split("/b/")[0]
                if base and base not in urls:
                    urls.append(base)
        return urls

    def _market_waits(now_ts: str) -> list[str]:
        """What the market waits on now that only an answer can bring: sim time doesn't pass while the clock holds."""
        out = []
        with c.market.lock:
            members = len(c.market.roster)
            for e in c.market.events.values():
                n, ph = e["need"], e.get("phase")
                eid = n["event_id"]
                if ph == "offers" and len(e.get("offers") or {}) < members:
                    out.append(f"{eid}: {members - len(e.get('offers') or {})} offers to come")
                elif ph in ("clearing", "cleared", "utility"):
                    out.append(f"{eid}: the buyers' contract")
                elif ph == "homes" and not e.get("final_sent"):
                    left = sum(1 for x in (e.get("commitments") or {}).values() if x.get("status") == "SENT")
                    if left:
                        out.append(f"{eid}: {left} commitments to answer")
                if e.get("telemetry_sent") and ph not in ("done", "failed") and now_ts < beckn.add_minutes(n["window_end"], 1440):
                    out.append(f"{eid}: the buyers' settlement")
            for r in c.market.rt.calls.values():
                if r.get("phase") == "asking":
                    out.append(f'call {r.get("id")}: answers to come')
        due = c.market.readings_due()
        if due:
            out.append(f"{due} readings to come")
        if opens_brake(c.market.last_discover_day, now_ts, now_ts):
            out.append("the buyer's request for tomorrow")
        return out

    def _node_waits(now_ts: str) -> list[str]:
        out = []
        for base in _agent_services():
            try:
                cl = (httpx.get(base + "/agents", timeout=3).json() or {}).get("clock") or {}
            except (httpx.HTTPError, ValueError):
                out.append(f"{base}: no answer")
                continue
            if cl.get("ticked") != now_ts:
                out.append(f'{base}: its nodes at {cl.get("ticked")}')
            if cl.get("busy"):
                out.append(f'{base}: {len(cl["busy"])} deciding ({", ".join(cl["busy"][:4])})')
        return out

    def _hold_all(j: int) -> None:
        """Every clock to step j, held there."""
        r = c.clock
        at = r.ts(j)
        body = {"start": at, "at": at, "at_wall": time.time(), "step_seconds": float(r.step_seconds), "loop_days": 0, "keep": True, "hold": True}
        res: dict[str, Any] = {}

        def post(name: str, base: str) -> None:
            try:
                httpx.post(base + "/demo/clock", json=body, timeout=5, headers=doors.headers()).raise_for_status()
            except Exception as exc:                      # noqa: BLE001  it gets the line again from _clock_again
                res[name] = f"not reached: {str(exc)[:80]}"
        _cond["posting"] = True
        try:
            ths = [threading.Thread(target=post, args=kv, daemon=True) for kv in _clock_targets().items()]
            for t in ths:
                t.start()
            r.configure(at, float(r.step_seconds), 0, at, body["at_wall"], hold=True)
            for t in ths:
                t.join(6)
        finally:
            _cond["posting"] = False
        if set(res) != _cond["unreached"]:               # once when it changes, not at every step
            c.event("clock_not_reached", sim=at, agents=res)
        _cond["unreached"] = set(res)

    def _conductor_tick() -> None:
        r = c.clock
        if r.at_wall is None or r.loop_days or not r.held or time.time() < _cond["since"]:
            return
        if time.time() < _cond["since"] + float(r.step_seconds):      # the run's pace: a step lasts at least this long
            return
        i = r.current()
        if i >= len(r.rows) - 1:
            return
        now_ts = r.ts(i)
        waits = _node_waits(now_ts) + _market_waits(now_ts)
        t = time.time()
        if waits:
            sig = tuple(waits)
            if sig != _cond["waits"]:
                _cond["waits"], _cond["changed"] = sig, t
            if t - _cond["changed"] < float(c.cfg.get("clock_wait_idle_s", 180)):
                return
            _cond["went_on"].append({"sim": now_ts, "wall": round(t, 1), "without": list(waits)[:8]})
            c.event("clock_went_on", sim=now_ts, waited_s=round(t - _cond["since"], 1), without=list(waits)[:8])
        nxt = r.ts(i + 1)
        if nxt in _cond["gates"]:                         # a test runner posts what happens at nxt before any node sees it, then opens the gate
            _cond["at_gate"] = nxt
            return
        _cond["held_s"] += max(0.0, t - _cond["since"] - float(r.step_seconds))
        _cond["steps"] += 1
        _hold_all(i + 1)
        _cond.update(since=time.time(), waits=None, changed=0.0)

    @app.get("/demo/conductor")
    def conductor_get():
        """The waiting clock: steps moved, seconds held beyond the pace, and every time it went on without something."""
        return {"on": bool(c.cfg.get("clock_waits")), "now": c.clock.ts(c.clock.current()), "held": c.clock.held, "steps": _cond["steps"],
                "held_s": round(_cond["held_s"], 1), "waiting_on": list(_cond["waits"] or ()), "went_on": list(_cond["went_on"]),
                "gates": sorted(_cond["gates"]), "at_gate": _cond["at_gate"]}

    @app.post("/demo/conductor")
    def conductor_post(body: dict = Body(default={})):
        """Gates, for a test runner: {"gates": [ts, ...]} are the steps the clock stops before, held one step earlier with everything else done, until
        {"open": ts}. So a what-if lands on its own step, as in the offline day, whatever the runner's own timing."""
        if isinstance(body.get("gates"), list):
            _cond["gates"] = {str(x) for x in body["gates"]}
            _cond["at_gate"] = None
        if body.get("open"):
            _cond["gates"].discard(str(body["open"]))
            if _cond["at_gate"] == str(body["open"]):
                _cond["at_gate"] = None
        return conductor_get()

    def _pace_start() -> None:
        if _pace["loop"]:
            return
        _pace["loop"] = True

        def again() -> None:
            while True:
                time.sleep(20.0)
                _clock_again()
        threading.Thread(target=again, daemon=True, name="clock-again").start()

        def loop() -> None:
            while True:
                time.sleep(0.4)
                try:
                    _pace_tick()
                except Exception:
                    log.debug("pace loop: %s", traceback.format_exc())
        threading.Thread(target=loop, daemon=True, name="pace").start()

    def _pace_view() -> dict[str, Any]:
        S = float(_pace.get("slow") or c.clock.step_seconds)
        st = float(c.clock.step_seconds)
        return {"x": _pace.get("x", 1.0), "auto": bool(_pace.get("auto")), "slow": S, "step_seconds": st,
                "speed": round(S / st, 2) if st else 1.0, "brake": bool(_pace.get("brake")), "why": _pace.get("why"),
                "changing": bool(_pace["pending"]), "run": bool(_pace.get("day")), "changes": list(_pace["changes"])[-6:]}

    @app.get("/demo/pace")
    def pace_get():
        return _pace_view()

    @app.post("/demo/pace")
    def pace_set(body: dict = Body(default={})):
        """The live run's speed: {"x": 1|2|4|8} or {"auto": true}. Every agent changes pace at the same step; the
        brake still holds 1x while messages need time."""
        if c.clock.at_wall is None:
            raise HTTPException(409, "no live run: start one first (Run the event live)")
        if not _pace.get("slow"):                            # a run started before this coordinator: adopt it
            _pace.update(slow=float(c.clock.step_seconds), gen=_pace["gen"] + 1)
        _pace.update(**_pace_choice(body if isinstance(body, dict) and ("x" in body or "auto" in body) else body.get("pace")))
        c.event("pace_request", x=_pace["x"], auto=_pace["auto"])
        _pace_start()
        _pace_tick()
        return _pace_view()

    # ---- the run as the Mac's terminal shows it: tools/live_talk.py's transcript (HW6), served to the Control Room
    _talk: dict[str, Any] = {"talk": None, "out": None, "run": 0, "loop": False, "lock": threading.Lock()}

    def _talk_start() -> None:
        """A fresh transcript from the current end of the logs (called when a live run starts, or on first use)."""
        import importlib.util
        spec = importlib.util.spec_from_file_location("fg_live_talk", str(repo / "tools" / "live_talk.py"))
        lt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(lt)
        out = lt.BufferOut(path=str(_logs_dir / "live_talk_latest.txt"))
        talk = lt.Talk(_logs_dir, out, {})
        lt.intro(out, talk)
        with _talk["lock"]:
            _talk.update(talk=talk, out=out, run=_talk["run"] + 1)
        if not _talk["loop"]:
            _talk["loop"] = True

            def loop() -> None:
                while True:
                    t = _talk.get("talk")
                    if t is not None:
                        try:
                            with _talk["lock"]:
                                t.tick()
                        except Exception as exc:              # a bad line must never stop the transcript
                            log.warning("live talk: %s", str(exc)[:200])
                    time.sleep(0.5)
            threading.Thread(target=loop, name="live-talk", daemon=True).start()

    @app.get("/monitor/talk")
    def monitor_talk(after: int = 0, n: int = 600):
        """The transcript's lines from number `after` on (ANSI colours as in the terminal); `run` changes when a new
        live run starts the transcript again."""
        if _talk["out"] is None:
            try:
                _talk_start()
            except Exception as exc:
                raise HTTPException(503, f"live talk not available: {str(exc)[:160]}")
        out = _talk["out"]
        with _talk["lock"]:
            base, lines = out.dropped, out.lines
            start = max(0, after - base)
            sel = [ln for _, ln in lines[start:start + max(1, min(2000, n))]]
        return JSONResponse({"run": _talk["run"], "from": base + start, "next": base + start + len(sel), "lines": sel},
                            headers={"Cache-Control": "no-store"})

    _term_run: dict[str, int] = {}

    def _term_mark() -> None:
        """Where each terminal stands when a live run starts (so the run can be read back from its first line)."""
        for k, f in (("coordinator", "coordinator.log"), ("utility", "utility.log")):
            try:
                _term_run[k] = (_logs_dir / f).stat().st_size
            except OSError:
                _term_run[k] = 0
        with httpx.Client(timeout=4) as http:
            for k in ("home-enphase", "fleet", "home-tesla", *c.monitor.bnames):
                try:
                    _term_run[k] = int(http.get(c.monitor.src[k] + "/logs/tail", params={"after": -1}).json().get("next", 0))
                except Exception:
                    _term_run.pop(k, None)

    @app.get("/monitor/terminals")
    def monitor_terminals(after: str = ""):
        """The terminals of the three machines, as they update: the Mac (logs/coordinator.log, logs/utility.log),
        jetson-b (the Enphase home's logs/daemon.log, the fleet's logs/fleet.log) and jetson-a (the Tesla home's
        logs/daemon.log). `after` = "name:offset,..." from the previous answer's `next`; -1 (or missing) = the last
        ~64 kB. Each source: {host, next, lines}."""
        from . import wire
        offs = dict(_term_run) if after == "run" else {}     # "run": from where the latest live run began
        for part in ("" if after == "run" else after).split(","):
            k, _, v = part.partition(":")
            if k and v.lstrip("-").isdigit():
                offs[k] = int(v)
        out: dict[str, Any] = {}
        for k, f in (("coordinator", "coordinator.log"), ("utility", "utility.log")):
            out[k] = {"host": "mac", **wire.tail(_logs_dir / f, offs.get(k, -1))}
        with httpx.Client(timeout=4) as http:
            for k, host in (("home-enphase", "jetson-b"), ("fleet", "jetson-b"), ("home-tesla", "jetson-a"),
                            *((b["name"], b.get("host")) for b in c.monitor.bnodes)):   # a building node's own log
                try:
                    r = http.get(c.monitor.src[k] + "/logs/tail", params={"after": offs.get(k, -1)})
                    r.raise_for_status()
                    out[k] = {"host": host, **r.json()}
                except Exception as exc:
                    out[k] = {"host": host, "next": offs.get(k, -1), "lines": [], "error": str(exc)[:120]}
        return JSONResponse({"now": time.time(), "sources": out}, headers={"Cache-Control": "no-store"})

    @app.get("/monitor/wire")
    def monitor_wire(after: float = 0.0, n: int = 120):
        """The newest `n` messages after wall time `after`, from the coordinator's and the utility's wire logs: wall
        time, direction, sender, receiver, Beckn action, signature check, dataset time, message id, how long a received
        message took from its sender (s), its size and its first characters. Plus how fast NANDA Town delivers to each
        Jetson (their own measurement), at most every 10 s."""
        n = max(1, min(400, n))
        recs = []
        for path, who in zip(_wire_files, ("coordinator", "utility")):
            for r in _tail(path):
                if r.get("t", 0) <= after:
                    continue
                m = r.pop("message", None)
                js = json.dumps(m, separators=(",", ":"), default=str) if m is not None else ""
                r["size"] = len(js)
                r["peek"] = js[:420]
                if r.get("dir") == "recv" and isinstance(r.get("sent"), (int, float)):
                    r["delay_s"] = round(max(0.0, r["t"] - r["sent"]), 2)
                r["log"] = who
                recs.append(r)
        recs.sort(key=lambda r: r.get("t", 0))
        recs = recs[-n:]
        if time.time() - _links["at"] > 10:
            _links["at"] = time.time()
            d = {}
            with httpx.Client(timeout=4) as http:
                for k in ("home-tesla", "home-enphase", "fleet", *c.monitor.bnames):
                    try:
                        hj = http.get(c.monitor.src[k] + "/health").json()
                        d[k] = {"host": HOMES_AT.get(k), "delivery_s": hj.get("town_delivery_s") or (hj.get("town_stats") or {}).get("delivery_s"),
                                "what_if": hj.get("what_if"), "ok": True}
                    except Exception as exc:
                        d[k] = {"host": HOMES_AT.get(k), "ok": False, "error": str(exc)[:80]}
            _links["data"] = d
        return {"now": time.time(), "sim": c.clock.ts(c.clock.current()), "step_seconds": c.clock.step_seconds,
                "messages": recs, "links": _links["data"],
                "town": c.cfg.get("town_url"), "run": c.cfg.get("town_run_id"), "what_if": getattr(c, "what_if", {}),
                "pace": _pace_view()}

    @app.post("/demo/reset")
    def demo_reset(body: dict = Body(default={})):
        c.market.reset()
        c.monitor.reset()
        return {"ok": True}

    @app.post("/demo/clock")
    def demo_clock(body: dict = Body(default={})):
        r = c.clock
        r.configure(body.get("start", r.start), float(body.get("step_seconds", r.step_seconds)),
                    int(body.get("loop_days", r.loop_days)), body.get("at"), body.get("at_wall"), hold=bool(body.get("hold")))
        return r.clock()

    if getattr(c, "pace_saved", None):                    # the clock came back with its run, so the brake on it must too
        _pace.update(c.pace_saved, gen=_pace["gen"] + 1)
        _pace_start()
    return app


def main() -> None:
    ap = argparse.ArgumentParser(prog="flexgrid-coordinator")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("run")
    sub.add_parser("pubkey")
    s = sub.add_parser("send")
    s.add_argument("--to", required=True)
    s.add_argument("--delta", type=float, required=True)
    s.add_argument("--minutes", type=float, default=30)
    s.add_argument("--reason", default="manual_test")
    a = ap.parse_args()
    cfg = config.load()
    port = int(cfg["coordinator_port"])
    if a.cmd == "pubkey":
        print(signing.public_hex(signing.load_or_create(config.CONFIG_DIR / "secrets" / "signing.key")))
        return
    if a.cmd == "send":
        try:                                    # use the running daemon's Town session (joining twice kicks it)
            r = httpx.post(f"http://127.0.0.1:{port}/dispatch/manual", timeout=30, headers=doors.headers(),
                           json={"to": a.to, "delta_kw": a.delta, "minutes": a.minutes, "reason": a.reason})
            r.raise_for_status()
            d = r.json()
        except httpx.ConnectError:
            d = Coordinator().dispatch(a.to, a.delta, a.minutes, a.reason)
        print(f"-> {a.to} dispatch {d['dispatch_id']} delta={a.delta:+.2f} kW for {a.minutes} min ({a.reason})")
        return
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)      # one line per Town poll is noise
    c = Coordinator()
    c.cfg.setdefault("readings_quiet_s", 30.0)            # live: readings still on their way through a busy Town aren't asked for twice. The market holds this dict
    _logs = __import__("pathlib").Path(__file__).resolve().parents[1] / "logs"
    _logs.mkdir(exist_ok=True)
    from . import wire
    wire.open_log(str(_logs / "town_wire.jsonl"))         # every Beckn message sent or received (tools/live_talk.py)
    print(f"coordinator {c.name} pubkey {c.pub}; Town run {c.cfg.get('town_run_id')}; API http://127.0.0.1:{port}",
          flush=True)
    keeper = keep.for_coordinator(c)                      # a sale that was running carries on, before the inbox and the tick start
    if c.town:                                   # joins in the background (again and again while Town is down), then
        c.town.join_until_up(then=lambda: threading.Thread(target=c.inbox_loop, daemon=True).start(),   # reads its
                             log=lambda m: (print(m, flush=True), c.event("town_error", error=m)))      # inbox
    else:
        threading.Thread(target=c.inbox_loop, daemon=True).start()    # short checks, `inbox_workers` claimers
    from .daemon import retention_loop
    threading.Thread(target=retention_loop, args=(c.cfg, c.event), daemon=True).start()
    threading.Thread(target=c.market_loop, daemon=True).start()
    logs = __import__("pathlib").Path(__file__).resolve().parents[1] / "logs"
    logs.mkdir(exist_ok=True)
    threading.Thread(target=c.monitor.loop, args=(str(logs / "monitor_state.json"),), daemon=True).start()
    print(f"Control Room: http://127.0.0.1:{port}/monitor   household app: http://127.0.0.1:{port}/phone", flush=True)
    try:
        uvicorn.run(build_app(c), host="127.0.0.1", port=port, log_level="warning")
    finally:
        keeper.close()


if __name__ == "__main__":
    main()
