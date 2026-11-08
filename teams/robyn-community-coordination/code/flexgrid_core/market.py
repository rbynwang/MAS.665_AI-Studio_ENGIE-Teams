"""The coordinator's v2 market: one Beckn contract per grid event, delivered by the homes that accept a share.

Timeline of one event (sim time = the shared replay clock):
  every day at 10:00   discover -> utility; the utility answers with tomorrow's flex needs (notice the day before)
  need received        discover (call for offers) -> every home in the roster
  offers in            clear (all offers in, or 3 h before the utility's deadline) -> select / init / confirm -> utility
  utility ACTIVE       confirm (commitment) -> each allocated home (re-sent to homes that have not answered, at
                       most `confirm_retries` times); declines are topped up from spare guard-approved offers;
                       the final meters and capacity go to the utility with update (re-sent until Town takes it)
  window - 2 h         update (event notice: pre-charge) -> committed homes; the utility has sent baselines
  each slot            homes send signed meter readings (on_status) to the coordinator and to the utility
  window end + 15 min  "readings are in" -> utility (update); utility answers with actuals and settlement (the
                       settlement message carries the actuals too)
  settlement           split pro rata to measured delivery -> on_status payment to each home; reputation update
                       (once per event: a repeated settlement message, or a re-run of C4, changes nothing)

Clearing rule (documented, deterministic; no ML; the MPC planner can replace it later), per 30-minute slot:
  capacity offered to the utility = min(kW requested, sum of confidence x kW offered)
     (confidence = 0.9 x sensor health x the home's reputation, computed by the home; see home_v2.build_offer)
  1. EV pauses first: every home whose EV would charge in the window pauses it (the energy is only deferred)
  2. batteries cover the rest, each home the same share of its battery offer:
       share = (capacity x (1 + margin) - EV pauses) / battery offers, at most 1     margin = 10 %
     (the community asks 10 % more of its homes than it promises the utility, to cover declines and misses;
      the utility pays at most the capacity offered anyway)
  3. rebound guard: the offers carry each home's evening forecast (net import with no event, and the extra import
     if its battery has delivered). The coordinator adds them up, places every paused EV's restart in the lowest
     part of the evening (valley filling, never before the car arrives, and each EV still finishes before its
     departure), and checks that the
     community's evening peak stays at or below the no-event forecast peak. If not, the battery share is lowered
     in 5 % steps until it does, and the capacity offered to the utility drops with it.

One desk, two buyers (cfg "buyers", the new path; tools/sim_event.py orders=): fixed orders on one window fill from one pool, so a
meter is never sold twice and a gap can move to any member.
  * one call for offers, one clearing, one contract per node. The node doesn't know who buys
  * each offer has a firm part and a tentative part. Firm is sold first. The tentative part is sold only as far as a buyer's deposit
    reserved it, and what stays unsold is what a gap can still call on
  * each buyer takes a share of each half hour, in proportion to what it was sold. It gets the same signed readings and settles its own
    share at its own price, on the same meters
  * the coordinator asks the community forecast itself when the sale opens (outlook()), and its agent sees it when it picks the margin
"""
from __future__ import annotations

import json
import math
import threading
from collections import Counter
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

from . import beckn, config, signing, wire
from .replay import Replay

SLOT_H = beckn.SLOT_MINUTES / 60
MAX_CONFIDENCE = 0.9 * 1.2          # a home's confidence: 0.9 x sensor health x reputation (reputation is capped at 1.2)
PARTS = ("battery_kw", "ev_pause_kw", "ac_kw", "light_kw", "app_kw")   # what an offer's kW is made of
DEPOSIT_SHARE = 0.10                # our choice, no real rule exists: a deposit is this share of the energy it holds, at the order's price
BUILDING_FIRM_SHARE = 0.534         # stage0/buildings_sell.json: the firm share of what a building sheds, where its offer gives none


def firm_part(o: dict[str, Any], x: dict[str, Any]) -> float:
    """kW of this offer slot that is firm. The node's own estimate comes first. Without one, stored energy is firm and the rest is
    tentative, except for a building's signed-off firm share of what it sheds."""
    kw = float(x.get("kw_offered") or 0.0)
    if x.get("firm_kw") is not None:
        return max(0.0, min(kw, float(x["firm_kw"])))
    stored = min(kw, float(x.get("battery_kw") or 0.0))
    return stored + (kw - stored) * (BUILDING_FIRM_SHARE if o.get("node") == "building" else 0.0)


def offer_problem(o: Any) -> str | None:
    """Why this offer can't go in the pool, or None. One node's bad numbers would otherwise reach everyone: a negative offer sells nothing for
    the whole community, a NaN spreads through the book, and a slot with no id stops the clearing."""
    def real(v: Any) -> bool:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)

    def good(v: Any) -> bool:
        return real(v) and v >= 0

    def when(v: Any) -> bool:
        try:
            return bool(beckn.add_minutes(v, 0))
        except (TypeError, ValueError):
            return False
    if not isinstance(o, dict) or not isinstance(o.get("slots"), list) or not o["slots"]:
        return "no slots"
    for x in o["slots"]:
        if not isinstance(x, dict) or not isinstance(x.get("id"), int) or not good(x.get("kw_offered")):
            return "a slot without a real id and kW"
        if any(x.get(k) is not None and not good(x[k]) for k in PARTS + ("firm_kw", "tentative_kw")):
            return "a slot with a part that isn't a real kW"
    if not good(o.get("confidence")) or not good(o.get("energy_kwh")):
        return "no real confidence or energy"
    e = o.get("evening")
    if e is not None:
        cf = e.get("import_cf_kw") if isinstance(e, dict) else None
        curves = e.get("rebound_kw") if isinstance(e, dict) else None
        rows = [cf, e.get("import_noev_kw")] + (list(curves.values()) if isinstance(curves, dict) else [curves]) if isinstance(e, dict) else [None]
        if not all(isinstance(r, list) and len(r) == len(cf) and all(real(q) for q in r) for r in rows):
            return "an evening the rebound guard can't read"
    f = o.get("ev_flex")
    if f is not None and not (isinstance(f, dict) and good(f.get("kwh")) and real(f.get("max_kw")) and f["max_kw"] > 0 and f["kwh"] <= 48 * f["max_kw"]
                              and when(f.get("latest_finish")) and (not f.get("arrival") or when(f["arrival"]))):
        return "a car the evening's restart can't place"    # at no power its charge never ends, and the clearing would wait for it forever
    return None


def _rebound_at(curves: dict[str, list[float]] | list[float], frac: float) -> list[float]:
    """A home's evening rebound for delivering `frac` of its battery offer, interpolated between the delivery
    levels it forecast (0 % = no rebound)."""
    if isinstance(curves, list):
        return [frac * x for x in curves]
    pts = [(0.0, [0.0] * len(next(iter(curves.values()))))] + sorted((float(k), v) for k, v in curves.items())
    for (f0, c0), (f1, c1) in zip(pts, pts[1:]):
        if frac <= f1 + 1e-9:
            w = 0.0 if f1 == f0 else (frac - f0) / (f1 - f0)
            return [a + w * (b - a) for a, b in zip(c0, c1)]
    return pts[-1][1]


class Market:
    def __init__(self, name: str, key, send: Callable, clock, roster: list[str], pins: dict[str, str],
                 save_pins: Callable, log: Callable[..., None], cfg: dict[str, Any]):
        self.name, self.key, self.send, self.clock = name, key, send, clock
        self.roster, self.pins, self.save_pins, self.log, self.cfg = roster, pins, save_pins, log, cfg
        self.utility = cfg.get("utility_name", "utility")
        self.buyers = list(cfg.get("buyers") or [self.utility])
        self.pooled = bool(cfg.get("buyers"))              # the new path: fixed orders on a window fill from one pool
        self.rated: dict[str, float] = {}                # each node's nameplate kW, from its enrolment: an offer can't claim more
        self.quiet = cfg.get("report_rule") == "exception"   # nodes speak only outside expected limits (report_rule.py)
        self.margin = float(cfg.get("clear_margin", 0.10))
        self.fee = float(cfg.get("coordinator_fee_share", 0.0))
        self.events: dict[str, dict[str, Any]] = {}
        self.reputation: dict[str, float] = {}
        self.last_discover_day: str | None = None        # the last day whose discover the utility has answered
        self.other_discover: dict[str, str] = {}         # the same for any other buyer
        self.discover_asked: dict[str, str] = {}         # day -> sim time the discover was last sent (retries)
        self.discover_asked_wall: dict[str, float] = {}  # day -> wall time of that send (retry floor)
        self.too_late: set[str] = set()                  # needs that arrived after their deadline (logged once)
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=int(cfg.get("send_workers", 8)))
        self.bg = ThreadPoolExecutor(max_workers=1)        # broadcasts run here, one after another
        self.on_clear: Callable | None = None           # hook: Langflow C3 Clear runs the clearing when configured
        self.on_settle: Callable | None = None          # hook: Langflow C4 Settle
        self.traffic: Counter = Counter()               # Beckn messages sent and received, by action (monitor)
        self.town_lag: Callable[[], float] = lambda: 0.0   # seconds NANDA Town is behind (the coordinator sets it)
        # the coordinator's AI agent (brain.CoordinatorBrain, llm_enabled): picks each event's margin (5-30 %) before
        # the clearing, in the background (the market lock is never held while it thinks). Without it: clear_margin
        self.brain = None
        self.brain_pool = ThreadPoolExecutor(max_workers=1)
        self._backtests: dict[str, dict[str, Any] | None] = {}
        from .rt_market import RtDesk                    # the real-time desk: calls, gaps, dispatch (rt_market.py)
        self.rt = RtDesk(self)

    def reset(self) -> None:
        """Forget all events (demo: replay the same day again). Reputation is kept unless asked."""
        with self.lock:
            self.events.clear()
            self.last_discover_day = None
            self.other_discover.clear()
            self.discover_asked.clear()
            self.discover_asked_wall.clear()
            self.too_late.clear()
            self.traffic.clear()

    # ------------------------------------------------------------------ helpers
    def now(self) -> str:
        return self.clock.ts(self.clock.current())

    def _msg(self, action: str, to: str, txn: str, message: dict[str, Any], role: str = "bap",
             ttl_s: float | None = None) -> dict[str, Any]:
        return beckn.make(action, sender=self.name, receiver=to, transaction_id=txn, message=message, key=self.key,
                          sender_role=role, ttl_s=ttl_s)

    def _send(self, to: str, action: str, body: dict[str, Any], mid: str) -> bool:
        try:                                  # unique per send: Town drops a repeated message id as a duplicate,
            self.send(to, beckn.kind(action), body, f"{mid}:{uuid.uuid4().hex[:6]}")   # and demo days get replayed
            self.traffic[action] += 1
            wire.note("send", self.name, to, action, body, sim=self.now())
            return True
        except Exception as exc:                                  # one failed send must not stop a broadcast
            self.log("v2_send_error", to=to, action=action, error=str(exc)[:160])
            return False

    def _broadcast(self, items: list[tuple[str, str, dict[str, Any], str]]) -> None:
        """Send to many homes in parallel, in the background: the inbox handler that triggered it must ack its
        Town message within the lease (30 s), or Town hands the message out again. Tests set sync_broadcast."""
        if self.cfg.get("sync_broadcast"):
            self._broadcast_now(items)
        else:
            self.bg.submit(self._broadcast_now, items)

    def _broadcast_now(self, items: list[tuple[str, str, dict[str, Any], str]]) -> None:
        """Sends that failed (a Town timeout) are tried twice more. With sync_broadcast (the offline simulation) the
        sends go out one after another, in roster order, so a run repeats exactly."""
        for attempt in range(3):
            if self.cfg.get("sync_broadcast"):
                ok = [self._send(*it) for it in items]
            else:
                ok = list(self.pool.map(lambda it: self._send(*it), items))
            items = [it for it, good in zip(items, ok) if not good]
            if not items:
                return
            self.log("v2_send_retry", attempt=attempt + 1, failed=len(items))
            time.sleep(2)

    def _verify(self, sender: str, body: dict[str, Any]) -> bool:
        signer = body.get("signer", "")
        with self.lock:
            if sender not in self.pins and signer:
                self.pins[sender] = signer                         # first contact pins the key (NANDA lookup: phase 3)
                self.save_pins()
        return beckn.check(body, self.pins.get(sender, ""))

    def _report(self, ev: dict[str, Any]) -> None:
        d = config.state_dir() / "events"
        d.mkdir(exist_ok=True)
        (d / f"{ev['need']['event_id']}.json").write_text(json.dumps(ev, indent=1, default=str))

    def _utility_retry_due(self, ev: dict[str, Any], now: str) -> bool:
        """Retry select/init/confirm with backoff: 30, 60, 120 ... replay minutes AND 60, 120, 240 ... wall seconds
        (a fast demo clock must not resend before the utility's answer can be back: each resend restarted the exchange
        and the coordinator never got past it)."""
        tries = ev.get("utility_tries", 1)
        wait = float(self.cfg.get("utility_retry_minutes", 30)) * 2 ** (tries - 1)
        floor = 0.0 if self.cfg.get("sync_broadcast") else float(self.cfg.get("utility_retry_wall_seconds", 60)) * 2 ** (tries - 1)
        return (now >= beckn.add_minutes(ev.get("utility_asked_at", now), wait)
                and time.time() >= ev.get("utility_asked_wall", 0.0) + floor)

    # ------------------------------------------------------------------ the clock drives the timeline
    def _ask_buyer(self, buyer: str, day: str, now: str) -> None:
        """Tomorrow's needs, asked until this buyer answers (on_discover)."""
        first = buyer == self.utility
        if (self.last_discover_day if first else self.other_discover.get(buyer)) == day:
            return
        key = day if first else f"{buyer}:{day}"
        asked = self.discover_asked.get(key)
        floor = 0.0 if self.cfg.get("sync_broadcast") else float(self.cfg.get("discover_retry_wall_seconds", 30))
        if asked is None or (now >= beckn.add_minutes(asked, float(self.cfg.get("discover_retry_minutes", 30)))
                             and time.time() >= self.discover_asked_wall.get(key, 0.0) + floor):
            tomorrow = beckn.add_minutes(f"{day} 00:00", 1440)[:10]
            body = self._msg("discover", buyer, f"discover-{day}", {"intent": {"date": tomorrow}})
            if self._send(buyer, "discover", body, f"{self.name}:discover:{day}"):
                self.discover_asked[key] = now                    # a failed send is tried again on the next tick
                self.discover_asked_wall[key] = time.time()
                self.log("v2_discover", date=tomorrow, repeat=asked is not None, **({} if first else {"buyer": buyer}))

    def tick(self) -> None:
        with self.lock:
            self._tick()

    def _tick(self) -> None:
        now = self.now()
        day, hm = now[:10], now[11:16]
        if hm >= "10:00":
            for b in self.buyers:
                self._ask_buyer(b, day, now)
        self.rt.tick()
        for ev in list(self.events.values()):
            n = ev["need"]
            ph = ev["phase"]
            lead = -float(self.cfg.get("clear_lead_minutes", 180))      # time left for the utility exchange and
            if ph == "offers" and (len(ev["offers"]) >= len(self.roster) or now >= beckn.add_minutes(n["commit_by"], lead)) \
                    and self._margin_ready(ev):
                (self.on_clear or self.clear_and_commit)(n["event_id"])
            if ph == "cleared" and now >= beckn.add_minutes(ev.get("cleared_at", now), 30):
                self.log("v2_commit_retry", event_id=n["event_id"])    # cleared (C3) but never contracted
                self.commit_to_utility(n["event_id"])
            if ph == "utility":
                tries = ev.get("utility_tries", 1)
                due = self._utility_retry_due(ev, now)
                if now >= n["commit_by"] or (due and tries >= int(self.cfg.get("utility_retries", 5))):
                    ev["phase"] = "failed"                         # the utility's deadline passed, or it never
                    self.log("v2_utility_gave_up", event_id=n["event_id"], tries=tries, deadline=n["commit_by"])
                elif due:                                          # select/init/confirm lost on the way: again
                    self.log("v2_utility_retry", event_id=n["event_id"], attempt=tries + 1)
                    self.commit_to_utility(n["event_id"])
            if ph == "homes" and not ev.get("finalized"):
                self.resend_commitments(ev, now)
            if ph in ("homes", "active") and not ev.get("final_sent") and (ev.get("finalized") or now >= n["commit_by"]
                                                                            or all(c["status"] != "SENT" for c in ev["commitments"].values())):
                self.finalize(ev)                                      # sent again on every tick until Town takes it
            if ev.get("finalized") and not ev.get("notice_sent") and now >= beckn.add_minutes(n["window_start"], -120):
                self.notify_homes(ev)
            if ev.get("final_sent") and not ev.get("telemetry_sent") and now >= beckn.add_minutes(n["window_end"], 15):
                expected = sum(len(c["contract"]["commitments"]) for c in ev["commitments"].values() if c["status"] == "ACCEPTED")
                got = sum(len(v) for v in ev["readings"].values())
                if got >= expected or now >= beckn.add_minutes(n["window_end"], float(self.cfg.get("readings_wait_minutes", 120))):
                    self.send_telemetry(ev)

    # ------------------------------------------------------------------ inbox
    def handle(self, kind: str, sender: str, body: dict[str, Any]) -> str:
        with self.lock:
            try:
                return self._handle(kind, sender, body)
            except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
                if not self.pooled:
                    raise
                self.log("v2_bad_message", sender=sender, message_kind=kind, error=f"{type(exc).__name__}: {exc}"[:160])
                return "rejected: that message can't be read"     # one sender's junk must not stop the desk for everyone

    def _handle(self, kind: str, sender: str, body: dict[str, Any]) -> str:
        if not self._verify(sender, body):
            self.log("v2_bad_signature", sender=sender, message_kind=kind)
            wire.note("recv", sender, self.name, beckn.action_of(kind), body, ok=False, sim=self.now())
            return "rejected: bad signature"
        act = beckn.action_of(kind)
        self.traffic[act] += 1
        wire.note("recv", sender, self.name, act, body, sim=self.now())
        m = body.get("message", {})
        if act == "on_status" and "batch" in m:             # one envelope of a node's signed readings: each counts as if sent alone
            return "; ".join(str(self._handle(kind, sender, b)) for b in m["batch"])[:300]
        rt = self.rt.handle(act, sender, body)
        if rt is not None:
            return rt
        if sender in self.buyers:
            return self.from_utility(act, body, sender)
        if act == "on_discover" and "offer" in m:
            o = m["offer"]
            ev = self.events.get(o["event_id"])
            if not ev or ev["phase"] != "offers" or o.get("home") != sender:
                return "offer ignored"
            if self.pooled:
                why = "not a member" if sender not in self.roster else "no nameplate on file" if self.rated and sender not in self.rated \
                    else offer_problem(o)
                if why:
                    self.log("v2_offer_refused", event_id=o["event_id"], home=sender, why=why)
                    return f"offer refused: {why}"
            ev["offers"][sender] = o
            return f"offer from {sender}: {o['energy_kwh']} kWh"
        if act == "on_confirm":
            c = m["contract"]
            ev = self.events.get(c["event_id"])
            if not ev or sender not in ev["commitments"]:
                return "answer ignored"
            cm = ev["commitments"][sender]
            if cm["status"] != "SENT":          # an answer is final: a repeat (Town redelivery, or the home answering
                return f"{sender} answer already recorded ({cm['status']})"   # a re-sent confirm) changes nothing
            if ev.get("finalized") and c["status"] == "ACCEPTED":
                # its yes came after the contract with the buyer was frozen without it: it is not in the deal, so it is
                # told (signed) to stand down instead of delivering for nothing (seen under a bad network, 29 Oct 2026)
                cm["status"], cm["reason"] = "LATE", "accepted after the contract with the buyer was final"
                cid = (cm.get("contract") or {}).get("id") or c.get("id")
                body = self._msg("cancel", sender, ev["ctxn"], {"contract": {"id": cid, "event_id": c["event_id"],
                                                                              "reason": "accepted_after_final"}})
                self._send(sender, "cancel", body, f"{self.name}:cancel:{cid}")
                self.log("v2_late_answer_cancelled", event_id=c["event_id"], home=sender, contract_id=cid)
                return f"{sender} accepted after the contract was final: cancelled"
            cm["status"] = c["status"]
            cm["reason"] = c.get("reason")
            if c["status"] == "DECLINED":
                self.top_up(ev, sender)
            return f"{sender} {c['status']} ({c.get('reason')})"
        if act == "on_status" and "performance" in m:
            r = m["performance"]
            ev = self.events.get(r["event_id"])
            if not ev:
                return "reading ignored"
            if ev.get("finalized") and sender not in (ev.get("final_meters") or []):
                return "reading ignored: not in the final contract"
            ev["readings"].setdefault(sender, {})[r["slot_id"]] = {**r, "signer": body.get("signer"), "sig": body.get("sig"),
                                                                     "signed_body": body}
            return f"reading {sender} slot {r['slot_id']}: {r['usage_kw']} kW"
        return f"ignored {kind}"

    # ------------------------------------------------------------------ utility side (C3 grid interface)
    def from_utility(self, act: str, body: dict[str, Any], sender: str | None = None) -> str:
        m = body["message"]
        sender = sender or self.utility
        if act == "on_discover":
            txn = body.get("context", {}).get("transactionId", "")
            if txn.startswith("discover-"):                       # the buyer answered that day's discover
                if sender == self.utility:
                    self.last_discover_day = max(self.last_discover_day or "", txn[len("discover-"):])
                else:
                    self.other_discover[sender] = max(self.other_discover.get(sender, ""), txn[len("discover-"):])
            new = []
            for cat in m.get("catalogs", []):
                offers = {o["resourceIds"][0]: o for o in cat.get("offers", [])}
                for res in cat.get("resources", []):
                    if res["id"] in self.events or res["id"] not in offers:
                        continue
                    need = beckn.need_from_catalog(res, offers[res["id"]])
                    last = beckn.add_minutes(need["commit_by"], -float(self.cfg.get("clear_lead_minutes", 180)))
                    if self.now() >= last:                        # no time left to collect offers (e.g. an answer
                        if need["event_id"] not in self.too_late:  # that waited in a backlog): do not start it
                            self.too_late.add(need["event_id"])
                            self.log("v2_need_too_late", event_id=need["event_id"], commit_by=need["commit_by"],
                                     now=self.now())
                        continue
                    od = self._new_order(need, sender, offers[res["id"]])
                    if self.pooled and self._join(need, od):
                        continue
                    self.events[need["event_id"]] = {"need": need, "phase": "offers", "offers": {}, "commitments": {},
                                                     "readings": {}, "baselines": {}, "utxn": od["utxn"],
                                                     "ctxn": f"community-{need['event_id']}", "log": [], "orders": {sender: od}}
                    new.append(need["event_id"])
                    self.call_homes(self.events[need["event_id"]])
            return f"needs received: {new or 'none new'}"
        ev = self._event_by_utxn(body["context"]["transactionId"])
        if not ev:
            return "unknown transaction"
        od = self._order_of(ev, body["context"]["transactionId"])
        first = od["utxn"] == ev["utxn"]
        c = m.get("contract", {})
        if act in ("on_select", "on_init", "on_confirm") and ev["phase"] != "utility":
            return f"duplicate {act} ignored"              # NANDA Town delivers at least once: be idempotent
        if act in ("on_select", "on_init", "on_confirm"):
            od["contract"] = c
            if first:
                ev["utility_contract"] = c
        if act == "on_select":
            self._to_order(ev, od, "init", {"contract": {**c, "seller": "flexgrid-community",
                                                         "participatingMeters": sorted(ev["alloc"]),
                                                         "capacityOffered": {str(k): v for k, v in
                                                                             self._tranche(ev, od, ev["capacity_offered"]).items()},
                                                         **self._terms(ev, od, ev["capacity_offered"])}})
            return "on_select -> init"
        if act == "on_init":
            self._to_order(ev, od, "confirm", {"contract": c})
            return "on_init -> confirm"
        if act == "on_confirm":
            od["active"] = c.get("status") == "ACTIVE"
            if all(o.get("active") for o in self._orders(ev)):    # every buyer is signed: now the nodes get their contracts
                ev["phase"] = "homes"
                self.commit_homes(ev, sorted(ev["alloc"]))
            return f"utility contract {c.get('status')}"
        if act == "on_status":
            perf = m.get("performance", {})
            if perf.get("kind") == "baselines":
                if first or not ev.get("baselines"):
                    ev["baselines"] = perf["meters"]
                return f"baselines for {len(perf['meters'])} meters"
            if perf.get("kind") == "actuals":
                if not ev.get("result") and first:
                    ev["actuals"] = perf["meters"]
                    if ev.get("settle_waiting"):
                        self.settle(ev["need"]["event_id"])       # a settlement without actuals was waiting for them
                return "actuals received"
            if "settlement" in m:
                if ev.get("result") or ev["phase"] in ("settling", "done"):
                    return "duplicate settlement ignored"        # NANDA Town delivers at least once: settle once
                s = m["settlement"]
                od["settlement"] = s
                if first:
                    if isinstance(s.get("actuals"), dict):        # the actuals travel inside the settlement
                        ev["actuals"] = s["actuals"]
                    ev["settlement"] = s
                if all(o.get("settlement") for o in self._orders(ev)):
                    (self.on_settle or self.settle)(ev["need"]["event_id"])
                return "settlement received"
        return f"ignored utility {act}"

    def _orders(self, ev: dict[str, Any]) -> list[dict[str, Any]]:
        """The orders this sale fills. An event from before orders existed has the utility's alone."""
        if not ev.get("orders"):
            ev["orders"] = {self.utility: {"buyer": self.utility, "need": ev["need"], "utxn": ev["utxn"], "deposit_usd": 0.0}}
        return list(ev["orders"].values())

    def _new_order(self, need: dict[str, Any], buyer: str, offer: dict[str, Any]) -> dict[str, Any]:
        """One buyer's fixed order. A deposit travels in its signed offer (utility.place_order)."""
        dep = (offer.get("offerAttributes") or {}).get("deposit") or {}
        return {"buyer": buyer, "need": need, "utxn": str(uuid.uuid4()), "deposit_usd": max(0.0, float(dep.get("usd") or 0.0))}

    def _join(self, need: dict[str, Any], od: dict[str, Any]) -> bool:
        """True if this order is dealt with here: it joins the open sale on its window, or it's refused for overlapping one on another
        window, since a node can owe only one thing at a time."""
        for ev in self.events.values():
            n = ev["need"]
            if need["window_start"] >= n["window_end"] or need["window_end"] <= n["window_start"]:
                continue
            same = (need["window_start"], need["window_end"]) == (n["window_start"], n["window_end"])
            if same and od["buyer"] in ev.get("orders", {}):
                return True                                       # the same buyer's order, heard again
            if same and ev["phase"] == "offers":
                ev["orders"][od["buyer"]] = od
                self.log("v2_order_joined", event_id=n["event_id"], buyer=od["buyer"], order=need["event_id"],
                         kw=need["slots"][0]["kw_requested"])
                return True
            self.log("v2_order_refused", event_id=need["event_id"], buyer=od["buyer"], overlaps=n["event_id"], phase=ev["phase"])
            return True
        return False

    def _event_by_utxn(self, txn: str) -> dict[str, Any] | None:
        return next((e for e in self.events.values() if e["utxn"] == txn or any(o["utxn"] == txn for o in self._orders(e))), None)

    def _order_of(self, ev: dict[str, Any], txn: str) -> dict[str, Any]:
        return next(o for o in self._orders(ev) if o["utxn"] == txn)

    def _to_order(self, ev: dict[str, Any], od: dict[str, Any], action: str, message: dict[str, Any],
                  ttl_s: float | None = None) -> bool:
        body = self._msg(action, od["buyer"], od["utxn"], message, ttl_s=ttl_s)
        ok = self._send(od["buyer"], action, body, f"{self.name}:{action}:{od['utxn']}:{uuid.uuid4().hex[:6]}")
        self.log(f"v2_utility_{action}", event_id=ev["need"]["event_id"], sent=ok, **({"buyer": od["buyer"]} if self.pooled else {}))
        return ok

    def _to_utility(self, ev: dict[str, Any], action: str, message: Any, ttl_s: float | None = None) -> bool:
        """To every buyer of this sale. message: the same for each, or a function of the order."""
        ok = True
        for od in self._orders(ev):
            ok = self._to_order(ev, od, action, message(od) if callable(message) else message, ttl_s) and ok
        return ok

    def _tranche(self, ev: dict[str, Any], od: dict[str, Any], cap: dict[int, float]) -> dict[int, float]:
        """This buyer's part of the capacity sold."""
        if not self.pooled:
            return cap
        share = od.get("share") or {}
        return {k: round(float(v) * share.get(k, 0.0), 2) for k, v in cap.items()}

    def _terms(self, ev: dict[str, Any], od: dict[str, Any], cap: dict[int, float]) -> dict[str, Any]:
        """What a buyer needs to settle its share of the pool. Nothing on the one-buyer path."""
        if not self.pooled:
            return {}
        return {"community_event": ev["need"]["event_id"], "poolCapacity": {str(k): v for k, v in cap.items()},
                "share": {str(k): round(v, 6) for k, v in (od.get("share") or {}).items()},
                "firm": {str(k): round(v, 2) for k, v in (od.get("firm") or {}).items()},
                "reserved": {str(k): round(v, 2) for k, v in (od.get("reserved") or {}).items()},
                "deposit_held_usd": round(float(od.get("deposit_held_usd") or 0.0), 4), "pooled_offer": ev.get("pooled_offer")}

    # ------------------------------------------------------------------ community side
    def call_homes(self, ev: dict[str, Any]) -> None:
        n = ev["need"]
        items = [(h, "discover", self._msg("discover", h, ev["ctxn"], {"intent": {"need": n}}),
                  f"{self.name}:call:{n['event_id']}:{h}") for h in self.roster]
        self._broadcast(items)
        self.log("v2_call_for_offers", event_id=n["event_id"], homes=len(items))
        if self.cfg.get("community_forecast", self.pooled):
            if self.cfg.get("sync_broadcast"):
                self.outlook(ev)
            else:                                         # seconds of work: not under the market lock on a live coordinator
                self.brain_pool.submit(self.outlook, ev)

    def outlook(self, ev: dict[str, Any]) -> dict[str, Any]:
        """The community forecast, called by the coordinator itself when the sale opens: the homes' demand at the sale hours with its
        range, and whether the window sits on the forecast peak. Its agent reads this when it picks the margin. No model or no data
        for the day never stops a sale."""
        from . import community_forecast
        n = ev["need"]
        try:
            f = community_forecast.forecast(n["window_start"][:10], issued_at=self.now(), members=list(self.roster))
        except (ValueError, OSError, KeyError, IndexError) as exc:
            ev["outlook"] = {"error": str(exc)[:160]}
            self.log("v2_outlook_failed", event_id=n["event_id"], error=ev["outlook"]["error"])
            return ev["outlook"]
        h0, h1 = int(n["window_start"][11:13]), int(n["window_end"][11:13]) + (1 if n["window_end"][14:16] != "00" else 0)
        rows = [x for x in f["hours"] if h0 <= x["hour"] < h1] or f["hours"]
        mean = lambda k: sum(x[k] for x in rows) / len(rows)  # noqa: E731
        p50 = mean("p50_kw")
        pk = f["peak_window"]
        ev["outlook"] = {"issued_at": f["issued_at"], "homes": f["homes"], "window_p10_kw": round(mean("p10_kw"), 1),
                         "window_p50_kw": round(p50, 1), "window_p90_kw": round(mean("p90_kw"), 1),
                         "range_pct_of_p50": round(100 * (mean("p90_kw") - mean("p10_kw")) / max(p50, 1e-6), 1),
                         "peak_window": [pk["start"], pk["end"]],
                         "window_on_peak": n["window_start"][11:16] < pk["end"] and n["window_end"][11:16] > pk["start"]}
        self.log("v2_outlook", event_id=n["event_id"], **ev["outlook"])
        return ev["outlook"]

    # ------------------------------------------------------------------ the book: firm first, then what deposits reserved
    @staticmethod
    def _slot_of(o: dict[str, Any], sid: int) -> dict[str, Any]:
        return next((x for x in o["slots"] if x["id"] == sid), {})

    @staticmethod
    def _order_kw(od: dict[str, Any], sid: int) -> float:
        return float(next((s["kw_requested"] for s in od["need"]["slots"] if s["id"] == sid), 0.0))

    @staticmethod
    def deposit_rate(need: dict[str, Any]) -> float:
        """USD a deposit pays to hold one kW through the window."""
        s = need["slots"]
        return DEPOSIT_SHARE * float(s[0]["price"]) * len(s) * SLOT_H

    def _book(self, ev: dict[str, Any]) -> dict[int, dict[str, Any]]:
        """Per half hour: the pool (confidence-weighted, as the clearing counts it) and what each order takes of it. Firm goes to the
        orders in proportion to their size. The rest of each order takes tentative power, as far as its deposit reserved it."""
        orders = self._orders(ev)
        rows = {}
        for s in ev["need"]["slots"]:
            sid = s["id"]
            firm = tent = 0.0
            for h, o in ev["offers"].items():
                x, c = self._slot_of(o, sid), self._confidence(o)
                kw = min(float(x.get("kw_offered") or 0.0), self.rated.get(h, float("inf")))   # no more than its nameplate
                f = min(firm_part(o, x), kw)
                firm += f * c
                tent += (kw - f) * c
            want = {od["buyer"]: self._order_kw(od, sid) for od in orders}
            total = sum(want.values())
            k = min(1.0, firm / total) if total > 0 else 0.0
            firm_to = {b: w * k for b, w in want.items()}
            hold = {od["buyer"]: min(want[od["buyer"]] - firm_to[od["buyer"]],
                                     od["deposit_usd"] / max(self.deposit_rate(od["need"]), 1e-9)) for od in orders}
            r = min(1.0, tent / sum(hold.values())) if sum(hold.values()) > 0 else 0.0
            rows[sid] = {"firm_pool_kw": firm, "tentative_pool_kw": tent, "firm": firm_to, "reserved": {b: v * r for b, v in hold.items()},
                         "tentative_sold": min(1.0, r * sum(hold.values()) / tent) if tent > 0 else 0.0}
        return rows

    def _sellable(self, ev: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """The offers as they may be sold. One buyer on the old path: all of each offer. In the pool: each node's firm part and the
        same share of its tentative part, the share the deposits reserved."""
        if not self.pooled:
            return ev["offers"]
        book = self._book(ev)
        out = {}
        for h, o in ev["offers"].items():
            slots = []
            for x in o["slots"]:
                kw, f = float(x.get("kw_offered") or 0.0), firm_part(o, x)
                sold = min(f + (kw - f) * book[x["id"]]["tentative_sold"], self.rated.get(h, kw))
                k = sold / kw if kw > 0 else 0.0
                parts = k * sum(float(x.get(p) or 0.0) for p in PARTS)
                if parts > 1.01 * sold + 0.05:                     # the clearing allocates on the parts: a claim there can't get past the total
                    k *= sold / parts
                slots.append({**x, "kw_offered": round(sold, 3), "firm_kw": round(f, 3),
                              **{p: round(float(x.get(p) or 0.0) * k, 3) for p in PARTS if p in x}})
            out[h] = {**o, "slots": slots}
        return out

    def _fill_orders(self, ev: dict[str, Any]) -> None:
        """After the clearing: each buyer's share of each half hour, and what its deposit holds."""
        book = self._book(ev)
        for od in self._orders(ev):
            b = od["buyer"]
            od["firm"] = {sid: r["firm"][b] for sid, r in book.items()}
            od["reserved"] = {sid: r["reserved"][b] for sid, r in book.items()}
            od["share"] = {sid: (r["firm"][b] + r["reserved"][b]) / max(sum(r["firm"].values()) + sum(r["reserved"].values()), 1e-9)
                           for sid, r in book.items()}
            held = max(od["reserved"].values(), default=0.0) * self.deposit_rate(od["need"])
            od["deposit_held_usd"] = min(od["deposit_usd"], held)
            od["deposit_back_usd"] = round(od["deposit_usd"] - od["deposit_held_usd"], 4)   # what the pool couldn't hold goes back now
        ev["pooled_offer"] = {"firm_kw": {sid: round(r["firm_pool_kw"], 1) for sid, r in book.items()},
                              "tentative_kw": {sid: round(r["tentative_pool_kw"], 1) for sid, r in book.items()}}

    # ------------------------------------------------------------------ C1 clearing
    @staticmethod
    def _place_evs(base: list[float], evs: list[tuple[str, dict[str, Any]]], t0: str) -> tuple[list[float], dict[str, str]]:
        """Valley filling: each paused EV (largest first) restarts at the 15-minute step that keeps the running
        community total lowest while it charges, not before the car arrives, and still finishes before its
        departure."""
        total, starts = list(base), {}
        n = len(total)
        for h, f in sorted(evs, key=lambda x: -x[1]["kwh"]):
            prof, left = [], f["kwh"]
            while left > 1e-6:
                kw = min(f["max_kw"], left / 0.25)
                prof.append(kw)
                left -= kw * 0.25
            first = max(0, math.ceil(Replay.minutes_between(t0, f["arrival"]) / 15)) if f.get("arrival") else 0
            last = min(n, int(Replay.minutes_between(t0, f["latest_finish"]) // 15)) - len(prof)
            if first >= n:                                 # arrives after the evening the guard looks at
                starts[h] = beckn.add_minutes(t0, 15 * first)
                continue
            best = None
            for k in range(first, max(first, last) + 1):
                peak = max(total[k + j] + prof[j] for j in range(len(prof)) if k + j < n) if prof else 0.0
                if best is None or peak < best[0] - 1e-9:
                    best = (peak, k)
            k = best[1] if best else 0
            for j, kw in enumerate(prof):
                if k + j < n:
                    total[k + j] += kw
            starts[h] = beckn.add_minutes(t0, 15 * k)
        return total, starts

    @staticmethod
    def _confidence(o: dict[str, Any]) -> float:
        """The home's own confidence (0.9 x sensor health x its reputation), kept within what that formula allows."""
        try:
            c = float(o.get("confidence", 0.0))
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(MAX_CONFIDENCE, c)) if math.isfinite(c) else 0.0     # NaN would pass min() and count as the most confident

    # ------------------------------------------------------------------ the coordinator's AI agent: the margin
    def margin_of(self, ev: dict[str, Any]) -> float:
        """This event's margin: the AI agent's choice (clamped to 5-30 % by CoordinatorBrain) or clear_margin."""
        return float(ev.get("margin", self.margin))

    def _margin_ready(self, ev: dict[str, Any]) -> bool:
        """True when the clearing may run. With an AI agent the first call asks it (in the background) and returns
        False; the clearing runs on a later tick once it has answered, or fallen back, or failed to answer in time
        (its budget + 15 s: the market never waits longer). Without one: always True, nothing changes."""
        if self.brain is None or "margin" in ev:
            return True
        if not ev.get("margin_asked"):
            ev["margin_asked"] = time.time()
            facts = self.margin_facts(ev)
            if self.pooled:
                ev["margin_inputs"] = facts               # the record: this decision can be read and replayed
            if self.cfg.get("sync_broadcast"):            # the offline simulation: decided now, on its own clock
                self._decide_margin(ev["need"]["event_id"], facts)
                return "margin" in ev
            self.brain_pool.submit(self._decide_margin, ev["need"]["event_id"], facts)
            return False
        if time.time() - ev["margin_asked"] > float(getattr(self.brain, "budget_s", 40.0)) + 15:
            ev["margin"] = self.margin
            ev["margin_decision"] = {"margin": self.margin, "fallback": True,
                                     "reason": "rules: the default margin (the agent did not answer in time)"}
            return True
        return False

    def _decide_margin(self, event_id: str, facts: dict[str, Any]) -> None:
        try:
            d = self.brain.decide_margin(facts)
        except Exception as exc:                        # noqa: BLE001  the brain falls back by itself; belt and braces
            d = {"margin": self.margin, "fallback": True, "reason": f"rules: the default margin ({type(exc).__name__})"}
        with self.lock:
            ev = self.events.get(event_id)
            if ev is not None and "margin" not in ev and ev.get("phase") == "offers":
                ev["margin"] = max(0.0, min(0.5, float(d["margin"])))
                ev["margin_decision"] = d

    def _backtest(self, day: str) -> dict[str, Any] | None:
        """The same homes and clearing replayed offline on the season's earlier event days (results/, written by the
        season simulation): promised vs delivered. A backtest of the coordinator's own homes, labelled as such."""
        if day in self._backtests:
            return self._backtests[day]
        out = None
        try:
            ds = config.DATASET
            d = config.REPO_DIR / "results" / (ds if ds != "il2023" else "")
            pat = "sim_*_trained.json" if ds != "il2023" else "sim_2023-??-??.json"
            rows = []
            for f in sorted(d.glob(pat)):
                dd = f.name[4:14]
                if dd >= day or "tol9" in f.name or "naive" in f.name:
                    continue
                r = json.loads(f.read_text())
                pk, tk = r.get("promised_kwh") or 0, r.get("delivered_kwh_true")
                if pk and tk is not None:
                    rows.append({"date": dd, "promised_kwh": round(pk, 1), "delivered_kwh": round(tk, 1),
                                 "delivered_pct": round(100 * tk / pk), "price_usd_per_kwh": r.get("price_usd_per_kwh")})
            if rows:
                pct = [x["delivered_pct"] for x in rows]
                out = {"what": "the same homes and clearing replayed on this season's earlier event days (simulation)",
                       "events": len(rows), "delivered_pct_mean": round(sum(pct) / len(pct)),
                       "delivered_pct_worst": min(pct), "days_below_100_pct": sum(1 for p in pct if p < 100),
                       "last_events": rows[-5:]}
        except (OSError, ValueError):
            out = None
        self._backtests[day] = out
        return out

    MARGINS_SHOWN = (0.05, 0.10, 0.15, 0.20, 0.30)

    def sell_by_margin(self, ev: dict[str, Any]) -> list[dict[str, Any]]:
        """For each margin the agent could pick: what the clearing (clear(), before the evening rebound guard) would
        sell and earn if delivered, and what it would ask of the sellers, per half hour (mean over the window). The
        agent sees the trade-off itself: with offers far above the request a bigger margin sells the same."""
        n, offers = ev["need"], self._sellable(ev)
        price = float((n.get("slots") or [{}])[0].get("price") or 0.0)
        get = lambda o, sid, k: next((x.get(k, 0.0) for x in o["slots"] if x["id"] == sid), 0.0)  # noqa: E731
        per = []
        for s in n.get("slots") or []:
            sid = s["id"]
            weighted = sum(get(o, sid, "kw_offered") * self._confidence(o) for o in offers.values())
            per.append((min(self._asked(ev, s), weighted), sum(get(o, sid, "ev_pause_kw") for o in offers.values()),
                        sum(get(o, sid, "battery_kw") + get(o, sid, "ac_kw") + get(o, sid, "light_kw") + get(o, sid, "app_kw")
                            for o in offers.values())))
        out = []
        for m in self.MARGINS_SHOWN:
            sold, asked = [], []
            for cap, evk, bat in per:
                share = 0.0 if bat <= 0 else max(0.0, min(1.0, (cap * (1 + m) - evk) / bat))
                a = evk + share * bat
                sold.append(min(cap, a / (1 + m)))
                asked.append(a)
            k = max(1, len(per))
            out.append({"margin": m, "sold_kw_per_half_hour": round(sum(sold) / k, 1),
                        "income_if_delivered_usd": round(sum(sold) * SLOT_H * price, 2),
                        "asked_of_sellers_kw_per_half_hour": round(sum(asked) / k, 1)})
        return out

    def _asked(self, ev: dict[str, Any], s: dict[str, Any]) -> float:
        """kW wanted in this half hour: every order on the sale."""
        return sum(self._order_kw(od, s["id"]) for od in self._orders(ev)) if self.pooled else s["kw_requested"]

    def margin_facts(self, ev: dict[str, Any]) -> dict[str, Any]:
        """What the coordinator knows when it picks the margin: the request, the offers (structured: no home's free
        text), what each margin would sell (sell_by_margin), this run's finished events, and the backtest."""
        n, offers = ev["need"], self._sellable(ev)
        get = lambda o, sid, k: next((x.get(k, 0.0) for x in o["slots"] if x["id"] == sid), 0.0)  # noqa: E731
        per = []
        for s in n["slots"]:
            sid = s["id"]
            per.append({"start": s["start"][11:], "kw_asked": self._asked(ev, s),
                        "battery_kw_offered": round(sum(get(o, sid, "battery_kw") for o in offers.values()), 1),
                        "ev_pause_kw_offered": round(sum(get(o, sid, "ev_pause_kw") for o in offers.values()), 1),
                        "confidence_weighted_kw": round(sum(get(o, sid, "kw_offered") * self._confidence(o)
                                                            for o in offers.values()), 1)})
        conf = sorted(self._confidence(o) for o in offers.values()) or [0.0]
        done = []
        for e in self.events.values():
            res = e.get("result") or {}
            if e is ev or not res:
                continue
            cap = e.get("capacity_offered_final") or e.get("capacity_offered") or {}
            prom = sum(cap.values()) * SLOT_H
            done.append({"event_id": e["need"]["event_id"], "promised_kwh": round(prom, 1),
                         "measured_kwh": res.get("delivered_kwh_measured"), "paid_usd": res.get("seller_usd")})
        extra: dict[str, Any] = {"community_forecast": ev["outlook"]} if ev.get("outlook") else {}
        if self.pooled:
            book = self._book(ev)
            sid0 = n["slots"][0]["id"]
            extra = {**extra, "nodes": {"homes": sum(1 for o in ev["offers"].values() if o.get("node") != "building"),
                                        "buildings": sum(1 for o in ev["offers"].values() if o.get("node") == "building")},
                     "pool_first_half_hour": {"firm_kw": round(book[sid0]["firm_pool_kw"], 1),
                                              "tentative_kw": round(book[sid0]["tentative_pool_kw"], 1),
                                              "firm_share_of_orders_pct": round(100 * min(1.0, book[sid0]["firm_pool_kw"] / max(self._asked(ev, n["slots"][0]), 1e-6)))},
                     "orders": [{"buyer": od["buyer"], "kw": self._order_kw(od, sid0), "price_usd_per_kwh": od["need"]["slots"][0]["price"],
                                 "deposit_usd": od["deposit_usd"]} for od in self._orders(ev)]}
        return {"now": self.now(), "default_margin": self.margin, **extra,
                "request": {"event_id": n["event_id"], "date": n["window_start"][:10],
                            "window": f"{n['window_start'][11:]}-{n['window_end'][11:]}",
                            "price_usd_per_kwh": n["slots"][0].get("price"), "reply_to_buyer_by": n.get("commit_by")},
                "offers": {"homes": len(offers), "homes_offering_nothing": sum(1 for o in offers.values()
                                                                               if (o.get("energy_kwh") or 0) <= 0.01),
                           "homes_with_ai_agent": sorted(h for h, o in offers.items() if o.get("agent")),
                           "confidence": {"min": round(conf[0], 2), "median": round(conf[len(conf) // 2], 2),
                                          "max": round(conf[-1], 2)},
                           "per_half_hour": per},
                "sell_by_margin": self.sell_by_margin(ev),
                "this_run_finished_events": done[-5:], "backtest": self._backtest(n["window_start"][:10]),
                "how_payment_works": "the buyer pays per kWh measured against each meter's baseline, at most what was "
                                     "sold in each half hour",
                "how_margin_works": "homes are asked for (kW sold x (1 + margin)); the community sells at most what the "
                                    "homes are asked for / (1 + margin), and never more than the buyer asked for"}

    def clear(self, event_id: str) -> dict[str, Any]:
        ev = self.events[event_id]
        if ev["phase"] not in ("offers", "clearing"):     # cleared already (e.g. C3 run again): nothing changes
            return ev.get("clear_summary") or {}
        n = ev["need"]
        offers = self._sellable(ev)
        margin = self.margin_of(ev)
        sids = [s["id"] for s in n["slots"]]
        get = lambda o, sid, k: next((x.get(k, 0.0) for x in o["slots"] if x["id"] == sid), 0.0)  # noqa: E731
        cap, evk, bat, share = {}, {}, {}, {}
        for s in n["slots"]:
            sid = s["id"]
            # reputation enters once, through the home's confidence (DATA_SOURCES section 5), not again here
            weighted = sum(get(o, sid, "kw_offered") * self._confidence(o) for o in offers.values())
            cap[sid] = round(min(self._asked(ev, s), weighted), 2)
            evk[sid] = sum(get(o, sid, "ev_pause_kw") for o in offers.values())
            bat[sid] = sum(get(o, sid, "battery_kw") + get(o, sid, "ac_kw") + get(o, sid, "light_kw") + get(o, sid, "app_kw") for o in offers.values())
            share[sid] = 0.0 if bat[sid] <= 0 else max(0.0, min(1.0, (cap[sid] * (1 + margin) - evk[sid]) / bat[sid]))
        # rebound guard
        homes = [h for h, o in offers.items() if o.get("evening")]
        steps = min((len(offers[h]["evening"]["import_cf_kw"]) for h in homes), default=0)
        no_event = [sum(offers[h]["evening"]["import_cf_kw"][k] for h in homes) for k in range(steps)]
        limit = max(no_event) * (1 + float(self.cfg.get("rebound_tolerance", 0.0))) if steps else None
        evs = [(h, offers[h]["ev_flex"]) for h in homes if offers[h].get("ev_flex")]
        t0 = n["window_end"]

        def evening(g: float):
            base = [0.0] * steps
            for h in homes:
                o = offers[h]
                bsum = sum(get(o, sid, "battery_kw") for sid in sids)
                frac = 0.0 if bsum <= 0 else sum(share[sid] * g * get(o, sid, "battery_kw") for sid in sids) / bsum
                e = o["evening"]
                reb = _rebound_at(e["rebound_kw"], frac)
                for k in range(steps):
                    base[k] += e["import_noev_kw"][k] + reb[k]
            return self._place_evs(base, evs, t0)

        g, guard = 1.0, {"applied": False}
        if steps:
            total, starts = evening(1.0)
            limit = max(limit, max(evening(0.0)[0]))      # benchmark fix: no event at all is the least the guard can ask for; rounding must not cancel the event
            guard = {"limit_kw": round(limit, 1), "no_event_peak_kw": round(max(no_event), 1),
                     "peak_without_guard_kw": round(max(total), 1), "applied": False}
            while max(total) > limit + 1e-6 and g > 0:
                g = round(max(0.0, g - 0.05), 2)
                total, starts = evening(g)
                guard["applied"] = True
            guard.update({"battery_factor": g, "peak_forecast_kw": round(max(total), 1),
                          "met": max(total) <= limit + 1e-6, "ev_restarts": starts,
                          "no_event_kw": [round(x, 1) for x in no_event], "with_event_kw": [round(x, 1) for x in total]})
        else:
            starts = {}
        alloc: dict[str, dict[int, float]] = {}
        for h, o in offers.items():
            for sid in sids:
                kw = get(o, sid, "ev_pause_kw") + share[sid] * g * (get(o, sid, "battery_kw") + get(o, sid, "ac_kw")
                                                                    + get(o, sid, "light_kw") + get(o, sid, "app_kw"))
                if kw >= 0.05:
                    alloc.setdefault(h, {})[sid] = round(kw, 3)
        for sid in sids:                                   # promise no more than the homes are asked for, less margin
            cap[sid] = round(min(cap[sid], sum(a.get(sid, 0.0) for a in alloc.values()) / (1 + margin)), 2)
        ev.update({"alloc": alloc, "capacity_offered": cap, "phase": "cleared", "cleared_at": self.now(),
                   "ev_restart": starts, "guard": guard, "battery_share": {sid: share[sid] * g for sid in sids}})
        if self.pooled:
            ev["sold_offers"] = offers
            self._fill_orders(ev)
        summary = {"homes": len(alloc), "capacity_offered_kw": cap,
                   "requested_kw": {s["id"]: self._asked(ev, s) for s in n["slots"]},
                   "offered_kw_raw": {sid: round(sum(get(o, sid, "kw_offered") for o in offers.values()), 2) for sid in sids},
                   "ev_pause_kw": {sid: round(evk[sid], 2) for sid in sids},
                   "battery_share": {sid: round(share[sid] * g, 3) for sid in sids},
                   "rebound_guard": {k: v for k, v in guard.items() if k not in ("ev_restarts", "no_event_kw", "with_event_kw")},
                   "offers": len(offers), "margin": margin}
        if self.pooled:
            summary["pooled_offer"] = ev["pooled_offer"]
            summary["orders"] = {od["buyer"]: {"share": {k: round(v, 4) for k, v in od["share"].items()},
                                               "firm_kw": {k: round(v, 1) for k, v in od["firm"].items()},
                                               "reserved_kw": {k: round(v, 1) for k, v in od["reserved"].items()},
                                               "deposit_held_usd": round(od["deposit_held_usd"], 2),
                                               "deposit_back_usd": od["deposit_back_usd"]} for od in self._orders(ev)}
        if ev.get("margin_decision"):
            d = ev["margin_decision"]
            summary["margin_by"] = "rules" if d.get("fallback") else "the coordinator's AI agent"
            summary["margin_reason"] = str(d.get("reason") or "")[:300]
        ev["clear_summary"] = summary
        self.log("v2_cleared", event_id=event_id, **{k: v for k, v in summary.items() if k != "offered_kw_raw"})
        return summary

    def commit_to_utility(self, event_id: str) -> dict[str, Any]:
        ev = self.events[event_id]
        if ev["phase"] not in ("cleared", "utility"):     # not cleared yet, or contracted already
            return {"sent": None, "event_id": event_id, "phase": ev["phase"]}
        n = ev["need"]
        if not any(float(v) > 0 for v in (ev.get("capacity_offered") or {}).values()):
            ev["phase"] = "no_offer"                      # nothing to sell: no contract, no retries
            self.log("v2_no_capacity", event_id=event_id, offers=len(ev["offers"]))
            return {"sent": None, "event_id": event_id, "phase": "no_offer"}
        ev["phase"] = "utility"
        ev["utility_asked_at"] = self.now()
        ev["utility_asked_wall"] = time.time()
        ev["utility_tries"] = ev.get("utility_tries", 0) + 1
        for od in self._orders(ev):
            if not od.get("active"):
                self._to_order(ev, od, "select", {"contract": {"commitments": [{"resources": [{"id": od["need"]["event_id"]}],
                                                                                "offer": {"id": od["need"]["offer_id"]}}]}})
        return {"sent": "select", "event_id": event_id}

    def clear_and_commit(self, event_id: str) -> dict[str, Any]:
        s = self.clear(event_id)
        self.commit_to_utility(event_id)
        return s

    def _contract_for(self, ev: dict[str, Any], home: str, slots: dict[int, float], k_ev: int) -> dict[str, Any]:
        n = ev["need"]
        sold = (ev.get("sold_offers") or {}).get(home) or {}

        def part(sid: int, kw: float) -> dict[str, float]:         # the pool only: how much of this commitment is firm
            x = self._slot_of(sold, sid) if sold else {}
            return {"firm_kw": round(min(kw, float(x.get("firm_kw", 0.0))), 3)} if x else {}
        return {"id": f"{n['event_id']}:{home}", "event_id": n["event_id"], "home": home,
                "window_start": n["window_start"], "window_end": n["window_end"],
                "commitments": [{"id": sid, "start": next(s["start"] for s in n["slots"] if s["id"] == sid), "kw": kw, **part(sid, kw)}
                                for sid, kw in sorted(slots.items())],
                "price_usd_per_kwh": n["slots"][0]["price"], "opt_out_by": n["commit_by"],
                "ev_resume_after": ev.get("ev_restart", {}).get(home) or n["window_end"],
                "precharge_minutes": 120, "settlement_rule": "pro rata to measured delivery"}

    def commit_homes(self, ev: dict[str, Any], homes: list[str], resend: bool = False) -> None:
        """Send each home its commitment. resend: the same contract again to homes that have not answered (the
        confirm or the answer was lost on the way); a home that did answer simply repeats its answer."""
        items, now, wall = [], self.now(), time.time()
        for k, h in enumerate(homes):
            cm = ev["commitments"].get(h) if resend else None
            if cm:
                c = cm["contract"]
                cm["tries"] = cm.get("tries", 1) + 1
                cm["sent_at"], cm["sent_wall"] = now, wall
            else:
                c = self._contract_for(ev, h, ev["alloc"][h], k)
                ev["commitments"][h] = {"status": "SENT", "contract": c, "sent_at": now, "sent_wall": wall, "tries": 1}
            items.append((h, "confirm", self._msg("confirm", h, ev["ctxn"], {"contract": c}),
                          f"{self.name}:commit:{c['id']}:{uuid.uuid4().hex[:6]}"))
        self._broadcast(items)
        self.log("v2_commitments_resent" if resend else "v2_commitments_sent", event_id=ev["need"]["event_id"],
                 homes=len(items))

    def resend_commitments(self, ev: dict[str, Any], now: str) -> None:
        """Homes still SENT after `confirm_retry_minutes` get their commitment again, at most `confirm_retries`
        times (bounded: a home that never answers is simply left out at finalize). A live demo replays a day in
        minutes (12 s per 15-minute step, so 30 replay minutes are 24 s), so a re-send also waits
        `confirm_retry_wall_seconds` of real time (120): a slow tunnel or a slow H4 flow is not a lost message. The
        offline simulation (sync_broadcast) has no real time to wait for."""
        wait = float(self.cfg.get("confirm_retry_minutes", 30))
        tries = int(self.cfg.get("confirm_retries", 3))
        floor = 0.0 if self.cfg.get("sync_broadcast") else float(self.cfg.get("confirm_retry_wall_seconds", 120))
        t = time.time()
        again = [h for h, c in ev["commitments"].items() if c["status"] == "SENT" and c.get("tries", 1) <= tries
                 and now >= beckn.add_minutes(c.get("sent_at", now), wait)
                 and (floor <= 0 or t - c.get("sent_wall", 0.0) >= floor)]
        if again:
            self.commit_homes(ev, again, resend=True)

    def top_up(self, ev: dict[str, Any], decliner: str) -> None:
        """Replace a declined share with spare offers from homes not yet asked (in roster order), once per decliner,
        and only with what the rebound guard lets each of those homes deliver (its EV pause plus the cleared share
        of its battery offer): a replacement must not empty a battery the guard did not plan for."""
        if ev.get("finalized") or decliner in ev.setdefault("topped_up", []):
            return
        ev["topped_up"].append(decliner)
        missing = dict(ev["alloc"].get(decliner, {}))
        share = ev.get("battery_share") or {}
        new = []
        for h, o in sorted((ev.get("sold_offers") or ev["offers"]).items()):
            if not missing or h in ev["commitments"]:
                continue
            take = {}
            for sid, kw in list(missing.items()):
                x = next((x for x in o["slots"] if x["id"] == sid), {})
                allowed = x.get("ev_pause_kw", 0.0) + share.get(sid, 0.0) * x.get("battery_kw", 0.0)
                avail = min(x.get("kw_offered", 0.0), allowed)
                t = round(min(avail, kw), 3)
                if t >= 0.05:
                    take[sid] = t
                    missing[sid] = round(kw - t, 3)
                    if missing[sid] < 0.05:
                        missing.pop(sid)
            if take:
                ev["alloc"][h] = take
                new.append(h)
        if new:
            self.commit_homes(ev, new)
        self.log("v2_top_up", event_id=ev["need"]["event_id"], decliner=decliner, replacements=new,
                 still_missing_kw=missing)

    def finalize(self, ev: dict[str, Any]) -> None:
        """Freeze who takes part and the capacity sold (once), then tell the utility. The update is sent again on
        every tick until Town takes it: the utility computes its baselines from it, so it must not be lost."""
        if not ev.get("finalized"):
            acc = {h: c for h, c in ev["commitments"].items() if c["status"] == "ACCEPTED"}
            cap = {}
            for s in ev["need"]["slots"]:
                got = sum(next((x["kw"] for x in c["contract"]["commitments"] if x["id"] == s["id"]), 0.0) for c in acc.values())
                cap[s["id"]] = round(min(ev["capacity_offered"][s["id"]], got / (1 + self.margin_of(ev))), 2)
            ev["capacity_offered_final"] = cap
            ev["final_meters"] = sorted(acc)
            ev["final_slots"] = {h: sorted(x["id"] for x in c["contract"]["commitments"]) for h, c in acc.items()}
            ev["finalized"] = True
            self.log("v2_final", event_id=ev["need"]["event_id"], accepted=len(acc),
                     declined=sum(1 for c in ev["commitments"].values() if c["status"] == "DECLINED"),
                     capacity_offered_kw=cap)
        cap = ev["capacity_offered_final"]
        msg = lambda od: {"contract": {"participatingMeters": ev["final_meters"],  # noqa: E731
                                       "capacityOffered": {str(k): v for k, v in self._tranche(ev, od, cap).items()},
                                       "meterSlots": ev["final_slots"], **self._terms(ev, od, cap)}}
        if self._to_utility(ev, "update", msg):
            ev["final_sent"] = True

    def notify_homes(self, ev: dict[str, Any]) -> None:
        ev["notice_sent"] = True
        if self.quiet:                                    # the contract already carries the pre-charge time
            return
        n = ev["need"]
        items = [(h, "update", self._msg("update", h, ev["ctxn"], {"notice": "event_tomorrow_precharge_now",
                                                                    "event_id": n["event_id"], "window_start": n["window_start"]}),
                  f"{self.name}:notice:{n['event_id']}:{h}")
                 for h, c in ev["commitments"].items() if c["status"] == "ACCEPTED"]
        self._broadcast(items)
        self.log("v2_event_notice", event_id=n["event_id"], homes=len(items))

    def send_telemetry(self, ev: dict[str, Any]) -> None:
        """Tell the utility the community's readings are in. Each home already sent its own signed readings to the
        utility; the coordinator does not relay them (it could not alter them, and 100 x 4 signed bodies are ~600 kB,
        too big for one message over a slow tunnel). Retried on the next tick if the send fails."""
        got = sum(len(v) for v in ev["readings"].values())
        ok = self._to_utility(ev, "update", {"telemetry_done": {"meters": len(ev["readings"]), "readings": got,
                                                                "called_slots": self.rt.called_slots(ev),
                                                                "abnormal": self.rt.abnormal_slots(ev)}}, ttl_s=beckn.TTL_EVIDENCE_S)
        ev["telemetry_tries"] = ev.get("telemetry_tries", 0) + 1
        if ok or ev["telemetry_tries"] >= 10:
            ev["telemetry_sent"] = True
        self.log("v2_telemetry_to_utility", event_id=ev["need"]["event_id"], meters=len(ev["readings"]), readings=got,
                 sent=ok)

    # ------------------------------------------------------------------ settlement (C2)
    def settle(self, event_id: str) -> dict[str, Any]:
        """Split the utility's payment and update reputations, once per event: a repeated settlement message, a re-run
        of C4 or POST /v2/settle returns the stored result and pays nobody again."""
        from . import settlement as st
        ev = self.events[event_id]
        if ev.get("result"):
            return ev["result"]
        s = ev["settlement"]
        meters = ev.get("actuals")
        if meters is None:
            meters = s.get("actuals")
        if meters is None:                                  # an older utility: wait for its separate actuals message
            ev["phase"] = "settling"
            ev["settle_waiting"] = True
            self.log("v2_settle_waiting", event_id=event_id, reason="no actuals yet")
            return {"waiting_for": "actuals", "event_id": event_id}
        paid = lambda x: next(f["value"] for f in x["revenue_flows"] if f["role"] == "seller")  # noqa: E731
        seller = sum(paid(od["settlement"]) for od in self._orders(ev)) if self.pooled else paid(s)
        delivered = {h: sum(st.clamp0(v["BASELINE"] - v["USAGE"]) * SLOT_H for v in m.values()) for h, m in meters.items()}
        committed_kwh = lambda h: (sum(x["kw"] for x in ev["commitments"][h]["contract"]["commitments"]) * SLOT_H  # noqa: E731
                                   if h in ev["commitments"] else 0.0)
        per_slot = lambda h: {str(x["id"]): x["kw"] for x in ((ev["commitments"].get(h) or {}).get("contract") or {})  # noqa: E731
                              .get("commitments", [])}
        # a building node (its signed offer says so): counted per half hour at most 1.2x its commitment, as the utility
        # counted it (settlement.building_counted_kwh); the homes as measured
        caps = {h: st.building_counted_kwh(meters[h], per_slot(h), SLOT_H) for h in delivered
                if (ev["offers"].get(h) or {}).get("node") == "building"}
        sp = st.split(seller, delivered, self.fee, caps or None)
        if st.MARKET:                                       # each node's share follows what its delivery is worth
            # the buyer's caps first: its readings carry a gap a building took on, which this contract copy doesn't
            kwcap = {h: {str(k): float(v) for k, v in ((s.get("caps_kw") or {}).get(h) or
                                                       {k: st.BUILDING_CAP_X * float(v) for k, v in per_slot(h).items()}).items()}
                     for h in caps}
            val = {h: st.meter_kwh(m, SLOT_H, kwcap.get(h), h in kwcap, h) for h, m in meters.items()}
            st.trim_called(val, s.get("capped_kw") or {}, SLOT_H)
            delivered = {h: sum(v[0].values()) for h, v in val.items()}
            sp = st.share_out(seller, val)
            if caps:
                sp["caps_kwh"] = {h: round(v, 3) for h, v in caps.items()}
            self.rt.premiums(ev, sp, delivered, committed_kwh)
        items = []
        for h, usd in sp["per_home"].items():
            committed = committed_kwh(h)
            ratio = delivered[h] / committed if committed > 0 else 0.0
            self.reputation[h] = round(0.7 * self.reputation.get(h, 1.0) + 0.3 * min(1.2, ratio), 3)
            msg = {"settlement": {"event_id": event_id, "payment": usd, "currency": s.get("currency", "USD"),
                                  "delivered_kwh": round(delivered[h], 3), "committed_kwh": round(committed, 3),
                                  "reputation": self.reputation[h], "rule": sp["rule"]}}
            items.append((h, "on_status", self._msg("on_status", h, ev["ctxn"], msg, ttl_s=beckn.TTL_EVIDENCE_S),
                          f"{self.name}:pay:{event_id}:{h}"))
        # participating homes the utility has no readings from delivered nothing it can see: reputation x 0.7
        participants = ev.get("final_meters") or [h for h, c in ev["commitments"].items() if c["status"] == "ACCEPTED"]
        silent = sorted(h for h in participants if h not in delivered)
        for h in silent:
            self.reputation[h] = round(0.7 * self.reputation.get(h, 1.0), 3)
            msg = {"settlement": {"event_id": event_id, "payment": 0.0, "currency": s.get("currency", "USD"),
                                  "delivered_kwh": 0.0, "committed_kwh": round(committed_kwh(h), 3),
                                  "reputation": self.reputation[h], "rule": sp["rule"], "note": "no meter readings"}}
            items.append((h, "on_status", self._msg("on_status", h, ev["ctxn"], msg, ttl_s=beckn.TTL_EVIDENCE_S),
                          f"{self.name}:pay:{event_id}:{h}"))
        self._broadcast(items)
        truth = sum(r["truth"]["delivered_true_kw"] * SLOT_H for rs in ev["readings"].values() for r in rs.values())
        ev["result"] = {"seller_usd": seller, "paid_homes": len(sp["per_home"]), "split": sp,
                        "delivered_kwh_measured": round(sum(delivered.values()), 3),
                        "delivered_kwh_true": round(truth, 3), "silent_homes": silent,
                        "components": s.get("components"), "net_zero_ok": s.get("net_zero_ok")}
        if self.pooled:
            ev["result"]["by_buyer"] = {od["buyer"]: {
                "order": od["need"]["event_id"], "paid_usd": paid(od["settlement"]),
                "sold_kwh": round(sum(self._tranche(ev, od, ev.get("capacity_offered_final") or {}).values()) * SLOT_H, 3),
                "delivered_kwh": round(sum(float(r.get("delivered", 0.0)) for r in od["settlement"].get("components", [])) * SLOT_H, 3),
                "deposit": od["settlement"].get("deposit")} for od in self._orders(ev)}
        ev["phase"] = "done"
        self._report(ev)
        self.log("v2_settled", event_id=event_id, seller_usd=seller, homes_paid=len(sp["per_home"]),
                 silent_homes=len(silent), measured_kwh=ev["result"]["delivered_kwh_measured"],
                 true_kwh=ev["result"]["delivered_kwh_true"])
        return ev["result"]

    def summary(self) -> dict[str, Any]:
        out = {}
        for eid, ev in self.events.items():
            cs = ev["commitments"].values()
            sl = ev["need"].get("slots") or []
            out[eid] = {"phase": ev["phase"], "window": [ev["need"]["window_start"], ev["need"]["window_end"]],
                        "commit_by": ev["need"].get("commit_by"), "final_sent": bool(ev.get("final_sent")),
                        "kw_requested": [s.get("kw_requested") for s in sl], "price": sl[0].get("price") if sl else None,
                        "margin": ev.get("margin"), "margin_thinking": bool(ev.get("margin_asked")) and "margin" not in ev,
                        "offers": len(ev["offers"]), "committed": len(ev["commitments"]),
                        "accepted": sum(1 for c in cs if c["status"] == "ACCEPTED"),
                        "declined": sum(1 for c in cs if c["status"] == "DECLINED"),
                        "capacity_offered_kw": ev.get("capacity_offered_final") or ev.get("capacity_offered"),
                        "readings": sum(len(v) for v in ev["readings"].values()),
                        "baselines": len(ev["baselines"]), "result": ev.get("result"),
                        **({"outlook": ev.get("outlook"), "buyers": sorted(ev.get("orders") or [])} if self.pooled else {})}
        return out
