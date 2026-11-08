"""The coordinator's real-time desk: calls from the buyer during the day, gaps reported by homes, and the dispatch that
moves power between homes within minutes. Beside the day-ahead market (market.py), which books the event the day before.

Messages (signed Beckn bodies on the same transport as the day-ahead ones; actions FlexGrid adds to the DEG set):
  rt_call      buyer -> coordinator   kW wanted from `start` to `end`, the price, why (a short-notice need, an emergency)
  rt_ask       coordinator -> homes   one broadcast: how much can you add from start to end, at this price?
  rt_answer    home -> coordinator    the kW it can add evenly over the call (its 5-minute plan's answer; 0 if none)
  rt_dispatch  coordinator -> home    kW owed on top of its booking, per 15 minutes, and the price
  rt_capacity  home -> coordinator    "I will be short by kW at these steps" (an owner's wish, a storm, a battery trip):
                                      only when it changes materially; the desk adds them up every 5 minutes, lets the
                                      margin (homes committed more than was sold) absorb what it can and moves only the
                                      rest, in one round for the whole community
  rt_confirm   coordinator -> buyer   kW confirmed per 15 minutes and what is short (sent at once, again when it changes)
  rt_end       buyer -> coordinator   the call is over early: the remaining dispatch is released
Readings for dispatched steps go out like the event's (homes' signed on_status, to the coordinator and the meter
owner), under the call's id. The buyer pays the call's price for what its meters measured on top of the booking.

On the quiet path (cfg report_rule "exception", report_rule.py) the desk works from what it already knows: it asks only the members whose
offers still have room, biggest first, until their room covers the need, instead of everyone. A member with nothing to add isn't asked
again for half an hour. No readings reach it during the window, so a report beyond a node's limits counts as it stands.
"""
from __future__ import annotations

import math
import time
import uuid
from typing import Any

from . import beckn
from . import settlement as st

ANSWER_WAIT_MIN = 5          # dataset minutes the desk waits for answers before it allocates
ANSWER_WAIT_S = 25.0         # live, also this many real seconds (NANDA Town brings the answers in 2-20 s; the fleet
                             # checks an idle home's inbox every 15 s); cfg rt_answer_wait_s
MARGIN = 0.10                # asks the homes for this much more than it confirms (as the day-ahead clearing does)
PRICE_LADDER = (1.0, 1.25, 1.5)   # a gap the first round cannot fill is offered again at these multiples
ASK_COVER = {"gap": 1.5, "call": 2.0}   # the quiet path asks members whose known room covers the need this many times
NO_ROOM_MIN = 30             # the quiet path doesn't ask a member with nothing to add again for this long
WARN_AGAIN_SHARE = 0.25      # the quiet path warns a buyer again only when the shortfall changes by this share
MIN_GAP_SHARE = 0.005        # the quiet path moves no gap under this share of what's sold: it costs more messages than it's worth
QUIET_CUSHION = 0.5          # the quiet path counts on this part of the margin, because small shortfalls go unreported and spend the rest
# what a node may say about a shortfall on the pool. Free text here would reach the buyers and the coordinator's model, so anything else reads "not given"
KNOWN_REASONS = {"the forecast changed", "the owner left the event", "the owner needs the car now", "its battery tripped", "the owner changed the comfort limit",
                 "the building closed for the day", "an event was added in a sold zone", "an alarm is on"}


def real(v: Any) -> float | None:
    """A real number, or None. NaN and infinity count as no number."""
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def steps_between(start: str, end: str) -> list[str]:
    out, t = [], start
    while t < end:
        out.append(t)
        t = beckn.add_minutes(t, 15)
    return out


class RtDesk:
    def __init__(self, market):
        self.m = market
        self.calls: dict[str, dict[str, Any]] = {}      # call or gap id -> record
        self.reports: dict[str, dict[str, float]] = {}  # home -> {ts: kW short} as last reported
        self.reasons: dict[str, str] = {}
        self.dirty = False                              # a report came since the last check
        self.checked_at: str | None = None
        self.absorbed: dict[str, float] = {}            # ts -> kW of shortfall the margin covered (the record)
        self.no_room: dict[str, str] = {}               # member -> when it may be asked again (the quiet path)
        self.warned: dict[str, float] = {}              # first step short -> kW the buyers were last warned of
        self.abnormal: dict[str, str] = {}              # member -> when it said its meter stopped measuring a cut (a building closed)

    # ------------------------------------------------------------------ helpers
    def _send(self, to: str, action: str, txn: str, message: dict[str, Any]) -> None:
        body = self.m._msg(action, to, txn, message)
        self.m._send(to, action, body, f"{self.m.name}:{action}:{txn}")

    def _item(self, to: str, action: str, txn: str, message: dict[str, Any]) -> tuple:
        """One message for Market._broadcast: sent in the background (live), so the market lock is not held through
        a Town round trip per home while the homes' answers wait to be handled."""
        return (to, action, self.m._msg(action, to, txn, message), f"{self.m.name}:{action}:{txn}")

    def _buyers(self) -> list[str]:
        return list(getattr(self.m, "buyers", None) or [self.m.utility])

    def _quiet(self) -> bool:
        return bool(getattr(self.m, "quiet", False))

    def _room(self, h: str, rec: dict[str, Any]) -> float | None:
        """kW this member has free through the request, from its offer and what it owes: known without a message. None: a step
        outside any sale, where nothing is on file."""
        steps = sorted(rec["short_by_ts"]) if rec["kind"] == "gap" else steps_between(rec["start"], rec["end"])
        room = None
        for t in steps:
            half = t[:14] + ("00" if t[14:16] < "30" else "30")
            ev = next((e for e in self.m.events.values() if e["need"]["window_start"] <= t < e["need"]["window_end"]), None)
            sid = next((s["id"] for s in ev["need"]["slots"] if s["start"] == half), None) if ev else None
            if sid is None:
                return None
            x = next((x for x in (ev["offers"].get(h) or {}).get("slots", []) if x["id"] == sid), {})
            cm = ev["commitments"].get(h) or {}
            owed = next((c["kw"] for c in (cm.get("contract") or {}).get("commitments", []) if c["id"] == sid), 0.0) \
                if cm.get("status") == "ACCEPTED" else 0.0
            owed += sum(g["alloc"].get(h, {}).get(t, 0.0) for g in self.calls.values() if g is not rec)
            free = max(0.0, min(float(x.get("kw_offered") or 0.0), getattr(self.m, "rated", {}).get(h, float("inf"))) - owed)
            room = free if room is None else min(room, free)
        return room or 0.0

    def _owed(self, h: str, t: str) -> float:
        """kW this member owes at a step: its booking and what the desk dispatched on top."""
        half = t[:14] + ("00" if t[14:16] < "30" else "30")
        kw = sum(g["alloc"].get(h, {}).get(t, 0.0) for g in self.calls.values())
        for ev in self.m.events.values():
            cm = ev["commitments"].get(h) or {}
            if ev["need"]["window_start"] <= t < ev["need"]["window_end"] and cm.get("status") == "ACCEPTED":
                sid = next((s["id"] for s in ev["need"]["slots"] if s["start"] == half), None)
                kw += next((c["kw"] for c in cm["contract"]["commitments"] if c["id"] == sid), 0.0)
        return kw

    def _checked_short(self, h: str, short: Any) -> dict[str, float]:
        """A report counts for what this member owes and no more: a bigger claim would move power that nobody is short of."""
        out = {}
        for ts, kw in (short.items() if isinstance(short, dict) else ()):
            v, owed = real(kw), self._owed(h, str(ts))
            if v is not None and v > 0.0 and owed > 0.0:
                out[ts] = min(v, owed)
        return out

    def _call_problem(self, c: dict[str, Any]) -> str | None:
        """Why the desk can't take this call, or None."""
        kw, price = real(c.get("kw")), real(c.get("price_usd_per_kwh"))
        if kw is None or kw <= 0 or price is None or price <= 0:
            return "its kW and price must be real numbers above zero"
        if c.get("kind") == "gap":
            return "a buyer can't open a gap"
        try:
            if beckn.add_minutes(c["start"], 0) >= beckn.add_minutes(c["end"], 0):
                return "it ends before it starts"
        except (KeyError, TypeError, ValueError):
            return "its times can't be read"
        over = next((r for r in self.calls.values() if r["kind"] != "gap" and c["start"] < (r.get("end_early") or r["end"]) and r["start"] < c["end"]), None)
        if over:                                        # both would be told the same room is free, and each would pay on the same meters
            return f"it overlaps call {over['id']}"
        return None

    def _short_list(self, rec: dict[str, Any], homes: list[str]) -> list[str]:
        """Who to ask. Everyone, except on the quiet path."""
        if not self._quiet():
            return homes
        now = self.m.now()
        homes = [h for h in homes if self.no_room.get(h, "") <= now]
        rooms = {h: self._room(h, rec) for h in homes}
        if any(v is None for v in rooms.values()):
            return homes                                # no offers on file for these steps: nothing to choose by
        want = float(rec["kw"]) * ASK_COVER["gap" if rec["kind"] == "gap" else "call"]
        out, got = [], 0.0
        for h in sorted(rooms, key=lambda x: (-rooms[x], x)):
            if rooms[h] < 0.05 or got >= want:
                break
            out.append(h)
            got += rooms[h]
        return out

    def _ask(self, rec: dict[str, Any], homes: list[str]) -> None:
        homes = self._short_list(rec, homes)
        if not homes and self._quiet():                 # nobody has room: no point in asking
            self.allocate(rec)
            return
        msg = {"ask": {"id": rec["id"], "start": rec["start"], "end": rec["end"], "price_usd_per_kwh": rec["price"],
                       "kind": rec["kind"]}}
        items = [(h, "rt_ask", self.m._msg("rt_ask", h, rec["id"], msg), f"{self.m.name}:rt_ask:{rec['id']}")
                 for h in homes]
        self.m._broadcast(items)
        rec["asked"] = sorted(set(rec.get("asked", [])) | set(homes))
        rec["asked_at"] = self.m.now()
        rec["asked_wall"] = time.time()

    # ------------------------------------------------------------------ inbox
    def handle(self, act: str, sender: str, body: dict[str, Any]) -> str | None:
        m = body.get("message") or {}
        if act == "rt_call" and sender in self._buyers():
            c = m["call"]
            if c["id"] in self.calls:
                return f"call {c['id']} already known"
            why = self._call_problem(c) if getattr(self.m, "pooled", False) else None
            if why:
                self.m.log("rt_call_refused", call_id=c["id"], why=why)
                self.m._broadcast([self._item(sender, "rt_confirm", str(c["id"]), {"confirm": {
                    "id": c["id"], "kw": 0.0, "short_kw": real(c.get("kw")) or 0.0, "homes": [], "refused": why}})])
                return f"call {c['id']} refused: {why}"
            rec = {"id": c["id"], "kind": c.get("kind", "short_notice"), "start": c["start"], "end": c["end"],
                   "kw": float(c["kw"]), "price": float(c["price_usd_per_kwh"]), "why": c.get("why", ""),
                   "from": sender, "answers": {}, "alloc": {}, "round": 0, "phase": "asking",
                   "received_at": self.m.now(), "readings": {}}
            self.calls[c["id"]] = rec
            self.m.log("rt_call", call_id=c["id"], call_kind=rec["kind"], kw=rec["kw"], start=rec["start"], end=rec["end"],
                       price=rec["price"], why=rec["why"])
            if rec["end"] <= rec["received_at"]:        # it reached the desk after its window (a backlog on the way):
                rec.update({"phase": "dispatched", "confirmed_kw": 0.0, "short_kw": rec["kw"], "homes": 0,   # say so
                            "allocated_at": rec["received_at"], "too_late": True})          # at once; ask nobody
                self.m.log("rt_call_too_late", call_id=c["id"], end=rec["end"], received_at=rec["received_at"])
                self.m._broadcast([self._item(sender, "rt_confirm", c["id"], {"confirm": {
                    "id": c["id"], "kw": 0.0, "short_kw": rec["kw"], "homes": [], "too_late": True}})])
                return f"call {c['id']} arrived after its window ({rec['end']}): confirmed 0 kW"
            self._ask(rec, list(self.m.roster))
            return f"call {c['id']}: asked {len(self.m.roster)} homes"
        if act == "rt_answer":
            a = m["answer"]
            rec = self.calls.get(a["id"])
            if not rec or sender not in rec.get("asked", []):
                return "answer ignored"
            kw = max(0.0, real(a.get("kw", 0.0)) or 0.0)
            rec["answers"][sender] = {"kw": min(kw, getattr(self.m, "rated", {}).get(sender, kw)), "why": a.get("why")}      # no more than its nameplate
            if rec["answers"][sender]["kw"] < 0.05 and "price" not in str(a.get("why") or ""):
                self.no_room[sender] = beckn.add_minutes(self.m.now(), NO_ROOM_MIN)
            if rec["phase"] == "asking" and (len(rec["answers"]) >= len(rec["asked"]) or self._covered_by_answers(rec)):
                self.allocate(rec)
            return f"{sender} can add {a.get('kw')} kW"
        if act == "rt_capacity":
            pooled = getattr(self.m, "pooled", False)
            if pooled and sender not in self.m.roster:
                self.m.log("rt_report_refused", sender=sender)
                return "report ignored: not a member"
            short = self._checked_short(sender, m.get("short")) if pooled \
                else {ts: float(kw) for ts, kw in (m.get("short") or {}).items() if float(kw) > 0.0}
            self.reports[sender] = short
            if m.get("abnormal") and self.abnormal.get(sender, "")[:10] != self.m.now()[:10]:
                self.abnormal[sender] = self.m.now()
            why = m.get("reason", "")
            self.reasons[sender] = why if not pooled or (isinstance(why, str) and why in KNOWN_REASONS) else "not given"
            self.dirty = True
            if not short:
                return f"{sender}: no shortfall any more"
            return f"{sender} short {round(max(short.values()), 2)} kW from {min(short)}: checked at the next 5 minutes"
        if act == "rt_end" and sender in self._buyers():
            rec = self.calls.get((m.get("end") or {}).get("id"))
            if not rec or rec.get("from") != sender:
                return "end for an unknown call"
            if rec.get("end_early"):                        # at-least-once delivery: a second release would take the node below its booking
                return f"call {rec['id']} already ended"
            at = (m.get("end") or {}).get("at") or self.m.now()
            rec["end_early"] = at
            items = []
            for h, slots in rec["alloc"].items():
                rest = {t: -kw for t, kw in slots.items() if t >= at}
                if rest:
                    items.append(self._item(h, "rt_dispatch", rec["id"], {"dispatch": {"id": rec["id"], "steps": rest,
                                                                                       "release": True}}))
            if items:
                self.m._broadcast(items)
            self.m.log("rt_end", call_id=rec["id"], at=at)
            return f"call {rec['id']} ended at {at}"
        if act == "on_status" and sender in self._buyers() and (m.get("settlement") or {}).get("call_id") in self.calls:
            return self.settle(m["settlement"])
        if act == "on_status" and "performance" in m and m["performance"].get("event_id") in self.calls:
            r = m["performance"]
            self.calls[r["event_id"]]["readings"].setdefault(sender, {})[r["slot_id"]] = r
            return f"call reading {sender} slot {r['slot_id']}"
        return None

    def _parent(self, ts: str, sender: str) -> tuple[dict[str, Any] | None, list[str]]:
        """What the home owes at `ts` and who can cover it where it is paid for: the booked event (its accepted homes,
        whose readings in the window count in its settlement) or a call (any home: readings go under the call)."""
        for eid, ev in self.m.events.items():
            n = ev["need"]
            if n["window_start"] <= ts < n["window_end"]:
                acc = ev.get("final_meters") or [h for h, c in ev["commitments"].items() if c["status"] == "ACCEPTED"]
                return {"kind": "event", "id": eid}, [h for h in acc if h != sender]
        for rec in self.calls.values():
            if rec["kind"] != "gap" and rec["start"] <= ts < rec["end"]:
                return {"kind": "call", "id": rec["id"]}, [h for h in self.m.roster if h != sender]
        return None, [h for h in self.m.roster if h != sender]

    def _gap_price(self, ts: str) -> float:
        """A gap is paid what the slot earns: the booked event's price there, or an open call's."""
        for ev in self.m.events.values():
            n = ev["need"]
            if n["window_start"] <= ts < n["window_end"]:
                return float(n["slots"][0]["price"])
        for rec in self.calls.values():
            if rec["kind"] != "gap" and rec["start"] <= ts < rec["end"]:
                return rec["price"]
        return 0.0

    # ------------------------------------------------------------------ allocation
    def _covered_by_answers(self, rec: dict[str, Any]) -> bool:
        """Live: the answers in hand already cover the call and its margin (or the gap), so the desk dispatches now
        instead of waiting for the last homes (over NANDA Town the last of 100 answers came 30 s after the first, three
        15-minute steps at the demo's pace, 1 Nov 2026). The offline simulation delivers every answer at once and
        keeps waiting for all of them, so its results do not change."""
        if self.m.cfg.get("sync_broadcast"):
            return False
        want = rec["kw"] * (1.0 + (MARGIN if rec["kind"] != "gap" else 0.0))
        return sum(a["kw"] for a in rec["answers"].values() if a["kw"] >= 0.05) >= want - 1e-9

    def allocate(self, rec: dict[str, Any]) -> None:
        """Fill the call (plus the margin) from the biggest answers first, evenly over its steps; tell each home what
        it owes on top, tell the buyer what is confirmed. A gap: the same, for the steps it was short."""
        want = rec["kw"] * (1.0 + (MARGIN if rec["kind"] != "gap" else 0.0))
        left, alloc = want, {}
        for h, a in sorted(rec["answers"].items(), key=lambda x: (-x[1]["kw"], x[0])):
            if left <= 1e-3:
                break
            kw = min(a["kw"], left)
            if kw >= 0.05:
                alloc[h] = round(kw, 3)
                left -= kw
        steps = steps_between(rec["start"], rec["end"])
        if rec["kind"] == "gap":                        # only the steps the home said it would miss, at its size
            steps = sorted(rec["short_by_ts"])
        parent = rec.get("parent") or {}
        items = []
        for h, kw in alloc.items():
            per = {t: (kw * min(1.0, rec["short_by_ts"][t] / rec["kw"]) if rec["kind"] == "gap" else kw) for t in steps}
            rec["alloc"][h] = {t: round(v, 3) for t, v in per.items() if v > 1e-3}
            rec.setdefault("asked_price", {})[h] = rec["price"]
            did = parent["id"] if parent.get("kind") == "call" else rec["id"]     # covering a call: owed under the call
            if parent.get("kind") == "call":
                self.calls[did]["alloc"].setdefault(h, {})
                for t, v in rec["alloc"][h].items():
                    self.calls[did]["alloc"][h][t] = round(self.calls[did]["alloc"][h].get(t, 0.0) + v, 3)
            items.append(self._item(h, "rt_dispatch", did, {"dispatch": {"id": did, "steps": rec["alloc"][h],
                                                                         "price_usd_per_kwh": rec["price"], "kind": rec["kind"],
                                                                         "start": rec["start"], "end": rec["end"],
                                                                         **({"event_id": parent["id"]} if parent.get("kind") == "event" else {})}}))
        got = sum(alloc.values())
        confirmed = round(min(rec["kw"], got / (1.0 + (MARGIN if rec["kind"] != "gap" else 0.0))), 3)
        rec.update({"phase": "dispatched", "confirmed_kw": confirmed, "short_kw": round(max(0.0, rec["kw"] - confirmed), 3),
                    "allocated_at": self.m.now(), "homes": len(alloc)})
        self.m.log("rt_dispatched", call_id=rec["id"], call_kind=rec["kind"], homes=len(alloc), confirmed_kw=confirmed,
                   short_kw=rec["short_kw"], answers=len(rec["answers"]))
        if rec["kind"] != "gap":
            items.append(self._item(rec.get("from") or self.m.utility, "rt_confirm", rec["id"],
                                    {"confirm": {"id": rec["id"], "kw": confirmed, "short_kw": rec["short_kw"], "homes": sorted(alloc)}}))
        if items:                                       # the dispatches (and the buyer's confirm) in the background
            self.m._broadcast(items)
        if rec["kind"] == "gap" and rec["short_kw"] > 0.05:
            if rec["round"] + 1 < len(PRICE_LADDER):
                self._raise(rec)                       # the first round did not cover it: once more, at a higher price
            else:
                self._warn(rec)

    def _raise(self, rec: dict[str, Any]) -> None:
        """Offer what is left once more, at the next price on the ladder, to the homes that could cover it and have not
        been dispatched for it yet (a home below its owner's minimum price may now say yes). Nobody left: warn."""
        rec["round"] += 1
        rec["price"] = round(rec["price"] * PRICE_LADDER[rec["round"]] / PRICE_LADDER[rec["round"] - 1], 4)
        rec["kw"] = rec["short_kw"]
        rec["short_by_ts"] = {t: v * rec["short_kw"] / max(1e-6, max(rec["short_by_ts"].values())) for t, v in rec["short_by_ts"].items()}
        rec["answers"], rec["phase"] = {}, "asking"
        short_now = {h for h, r in self.reports.items() if any(t > self.m.now() for t in r)}
        others = [h for h in self._parent(rec["start"], rec["from"])[1] if h not in rec["alloc"] and h not in short_now]
        self.m.log("rt_price_up", gap_id=rec["id"], price=rec["price"], kw=round(rec["kw"], 3), asked=len(others))
        if not others:
            rec["phase"] = "dispatched"
            self._warn(rec)
            return
        self._ask(rec, others)

    def _warn(self, rec: dict[str, Any]) -> None:
        """What the community will be short of: the buyer hears it now, not from the meters tomorrow."""
        if self._quiet():                               # a buyer hears a change, not every five minutes
            last = self.warned.get(rec["start"][:13])
            if last is not None and abs(rec["short_kw"] - last) < WARN_AGAIN_SHARE * max(last, 1e-6):
                return
            self.warned[rec["start"][:13]] = rec["short_kw"]
        self.m.log("rt_early_warning", gap_id=rec["id"], short_kw=rec["short_kw"], start=rec["start"], end=rec["end"])
        for b in self._buyers():
            self._send(b, "rt_confirm", rec["id"], {"warning": {"id": rec["id"], "short_kw": rec["short_kw"], "start": rec["start"],
                                                                "end": rec["end"], "why": rec["why"]}})

    def settle(self, s: dict[str, Any]) -> str:
        """Pay each home the share of the call's payment its measured extra earned (the coordinator keeps its fee)."""
        rec = self.calls[s["call_id"]]
        if rec.get("settled"):
            return f"call {rec['id']} already paid"
        rec["settled"] = s
        sp = st.split(float(s.get("paid_usd", 0.0)), {h: float(v) for h, v in (s.get("by_home_kwh") or {}).items()}, self.m.fee)
        if st.MARKET and getattr(self.m, "pooled", False):    # the price book's shares, as in the booking
            sp = st.call_shares(float(s.get("paid_usd", 0.0)), s.get("by_home_kwh") or {}, s.get("by_home_batt_kwh") or {},
                                {h for ev in self.m.events.values() for h, o in ev["offers"].items() if o.get("node") == "building"})
        items = []
        for h, usd in sp["per_home"].items():
            msg = {"settlement": {"event_id": rec["id"], "payment": usd, "currency": s.get("currency", "USD"),
                                  "delivered_kwh": round(float(s["by_home_kwh"].get(h, 0.0)), 3), "kind": rec["kind"],
                                  "rule": sp["rule"]}}
            items.append((h, "on_status", self.m._msg("on_status", h, rec["id"], msg, ttl_s=beckn.TTL_EVIDENCE_S),
                          f"{self.m.name}:pay:{rec['id']}:{h}"))
        self.m._broadcast(items)
        rec["result"] = {"paid_usd": s.get("paid_usd"), "delivered_kwh": s.get("delivered_kwh"), "homes_paid": len(items),
                         "fee_usd": sp.get("fee")}
        self.m.log("rt_paid", call_id=rec["id"], usd=s.get("paid_usd"), delivered_kwh=s.get("delivered_kwh"), homes=len(items))
        self.explain(rec)
        return f"call {rec['id']}: {s.get('paid_usd')} USD to {len(items)} homes"

    def abnormal_slots(self, ev: dict[str, Any]) -> dict[str, list[int]]:
        """Half hours a member's meter can't be paid on: load that fell because a building closed isn't a cut it made."""
        return {h: [s["id"] for s in ev["need"]["slots"] if beckn.add_minutes(s["start"], beckn.SLOT_MINUTES) > t and s["start"][:10] == t[:10]]
                for h, t in self.abnormal.items()}          # closed for that day: the next day's sale pays as usual

    def called_slots(self, ev: dict[str, Any]) -> list[int]:
        """The sale's half hours that a buyer's call covers, whoever placed it."""
        calls = [c for c in self.calls.values() if c["kind"] != "gap"]
        return [s["id"] for s in ev["need"]["slots"]
                if any(c["start"] < beckn.add_minutes(s["start"], beckn.SLOT_MINUTES) and s["start"] < c["end"] for c in calls)]

    def premiums(self, ev: dict[str, Any], sp: dict[str, Any], delivered: dict[str, float], committed_kwh) -> None:
        """A gap's helper was asked at a rising price, so it's paid it. The event pays the slot's price. The step up on the kWh it delivered
        beyond its booking comes out of the coordinator's keep, and never takes more than the keep. On the pool only."""
        if not getattr(self.m, "pooled", False):
            return
        book, add, left = st.price_book()["x"], {}, {}
        for g in self.calls.values():
            if g["kind"] != "gap" or (g.get("parent") or {}).get("id") != ev["need"]["event_id"]:
                continue
            base = float(g.get("base_price") or 0.0)
            for h, steps in g["alloc"].items():
                beyond = left.setdefault(h, st.clamp0(delivered.get(h, 0.0) - committed_kwh(h)))
                kwh = min(sum(steps.values()) * 0.25, beyond)
                left[h] = beyond - kwh                      # a kWh beyond the booking earns the step up once, in the order the gaps came
                up = float((g.get("asked_price") or {}).get(h, base)) / base - 1.0 if base > 0 else 0.0
                if kwh > 0 and up > 0:
                    x = float(book[st.node_class(h, (ev["offers"].get(h) or {}).get("node") == "building")])
                    add[h] = add.get(h, 0.0) + kwh * float(sp.get("usd_per_kwh") or 0.0) * x * up
        if not add:
            return
        k = min(1.0, float(sp.get("fee") or 0.0) / sum(add.values()))
        sp["premium_usd"] = {h: round(v * k, 4) for h, v in add.items()}
        for h, v in sp["premium_usd"].items():
            sp["per_home"][h] = round(sp["per_home"].get(h, 0.0) + v, 4)
        sp["fee"] = round(float(sp["fee"]) - sum(sp["premium_usd"].values()), 4)
        sp["pool"] = round(float(sp["pool"]) + sum(sp["premium_usd"].values()), 4)
        self.m.log("rt_gap_premiums", event_id=ev["need"]["event_id"], homes=len(add), usd=round(sum(sp["premium_usd"].values()), 2))

    def facts(self, rec: dict[str, Any]) -> dict[str, Any]:
        """What the code recorded about one call (or gap), for the AI agent's explanation."""
        ans = sorted((a["kw"] for a in rec["answers"].values()), reverse=True)
        gaps = [g for g in self.calls.values() if g["kind"] == "gap" and rec["start"] <= g["start"] < rec["end"]]
        return {"id": rec["id"], "now": self.m.now(), "kind": rec["kind"], "why": rec.get("why"), "window": [rec["start"], rec["end"]],
                "kw_asked": rec["kw"], "price_usd_per_kwh": rec["price"], "homes_asked": len(rec.get("asked", [])),
                "homes_answered": len(rec["answers"]), "homes_that_could_add": sum(1 for k in ans if k >= 0.05),
                "biggest_answers_kw": [round(k, 2) for k in ans[:5]], "kw_confirmed": rec.get("confirmed_kw"),
                "kw_short": rec.get("short_kw"), "homes_dispatched": rec.get("homes"), "price_rounds": rec.get("round", 0) + 1,
                "gaps_during_it": [{"home": g["from"], "kw": round(g["kw"], 2), "why": g.get("why"),
                                    "covered_kw": g.get("confirmed_kw"), "short_kw": g.get("short_kw")} for g in gaps],
                "measured_extra_kwh": (rec.get("settled") or {}).get("delivered_kwh"),
                "paid_usd": (rec.get("settled") or {}).get("paid_usd")}

    def explain(self, rec: dict[str, Any]) -> None:
        f = self.facts(rec)
        tmpl = (f"The buyer asked for {f['kw_asked']:.0f} kW more ({f['why'] or rec['kind']}); {f['homes_that_could_add']} of "
                f"{f['homes_answered']} homes could add something, {f['homes_dispatched'] or 0} were dispatched and "
                f"{float(f['kw_confirmed'] or 0):.1f} kW was confirmed. The meters show "
                f"{float(f['measured_extra_kwh'] or 0):.1f} kWh beyond the booking: ${float(f['paid_usd'] or 0):.2f}.")
        brain = getattr(self.m, "brain", None)
        if brain is None or not hasattr(brain, "explain_call"):
            rec["explanation"] = {"text": tmpl, "fallback": True}
            self.m.log("rt_explained", call_id=rec["id"], text=tmpl, fallback=True)
            return
        pool = getattr(self.m, "brain_pool", None)

        def run():
            try:
                rec["explanation"] = brain.explain_call(f, tmpl)
            except Exception as exc:                        # noqa: BLE001  the template stands
                rec["explanation"] = {"text": tmpl, "fallback": True, "error": str(exc)[:120]}
        if pool is not None:
            pool.submit(run)
        else:
            run()

    # ------------------------------------------------------------------ the community's total, every 5 minutes
    def _cushion(self, ts: str) -> tuple[float, dict[str, Any] | None]:
        """kW the booked event's homes committed beyond what was sold at ts (the margin), and that event."""
        for ev in self.m.events.values():
            n = ev["need"]
            if not (n["window_start"] <= ts < n["window_end"]) or not ev.get("finalized"):
                continue
            half = ts[:14] + ("00" if ts[14:16] < "30" else "30")
            sid = next((s["id"] for s in n["slots"] if s["start"] == half), None)
            if sid is None:
                return 0.0, ev
            committed = sum(next((x["kw"] for x in c["contract"]["commitments"] if x["id"] == sid), 0.0)
                            for c in ev["commitments"].values() if c["status"] == "ACCEPTED")
            sold = float((ev.get("capacity_offered_final") or {}).get(sid, 0.0))
            return max(0.0, committed - sold) + self._surplus(ev, sid), ev
        return 0.0, None

    @staticmethod
    def _measured(ev: dict[str, Any], before: int) -> tuple[float, float] | None:
        """(measured, sold) kW in the latest half hour before slot `before` that the meters have read (80 % of the
        event's homes), in the buyer's own measure (baseline minus use); None before the first one is read."""
        bases, reads = ev.get("baselines") or {}, ev.get("readings") or {}
        meters = ev.get("final_meters") or []
        for s in sorted((x for x in ev["need"]["slots"] if x["id"] < before), key=lambda x: -x["id"]):
            sid = s["id"]
            got = [h for h in meters if sid in (reads.get(h) or {}) or str(sid) in (reads.get(h) or {})]
            if not meters or len(got) < 0.8 * len(meters):
                continue
            measured = 0.0
            for h in got:
                r = reads[h].get(sid) or reads[h].get(str(sid)) or {}
                b = ((bases.get(h) or {}).get(str(sid)) or {}).get("BASELINE")
                if b is not None and r.get("usage_kw") is not None:
                    measured += max(0.0, float(b) - float(r["usage_kw"]))
            return measured, float((ev.get("capacity_offered_final") or {}).get(sid, 0.0))
        return None

    @staticmethod
    def _surplus(ev: dict[str, Any], before: int) -> float:
        """Half of what the meters showed beyond what was sold in the latest half hour already read (the buyer's own
        measure: baseline minus use, as it pays): when the community is running over its booking, a home's forecast
        of a shortfall is not worth moving power for."""
        bases, reads = ev.get("baselines") or {}, ev.get("readings") or {}
        meters = ev.get("final_meters") or []
        for s in sorted((x for x in ev["need"]["slots"] if x["id"] < before), key=lambda x: -x["id"]):
            sid = s["id"]
            got = [h for h in meters if sid in (reads.get(h) or {}) or str(sid) in (reads.get(h) or {})]
            if not meters or len(got) < 0.8 * len(meters):
                continue                                  # not read yet (readings come after each half hour)
            measured = 0.0
            for h in got:
                r = reads[h].get(sid) or reads[h].get(str(sid)) or {}
                b = ((bases.get(h) or {}).get(str(sid)) or {}).get("BASELINE")
                if b is not None and r.get("usage_kw") is not None:
                    measured += max(0.0, float(b) - float(r["usage_kw"]))
            sold = float((ev.get("capacity_offered_final") or {}).get(sid, 0.0))
            return max(0.0, measured - sold) * 0.5
        return 0.0

    def _sold(self, ts: str) -> float:
        """kW sold at this step."""
        for ev in self.m.events.values():
            n = ev["need"]
            if n["window_start"] <= ts < n["window_end"]:
                half = ts[:14] + ("00" if ts[14:16] < "30" else "30")
                sid = next((s["id"] for s in n["slots"] if s["start"] == half), None)
                return float((ev.get("capacity_offered_final") or {}).get(sid, 0.0))
        return 0.0

    def _covered(self, ts: str) -> float:
        return sum(v.get(ts, 0.0) for g in self.calls.values() if g["kind"] == "gap" for v in g["alloc"].values())

    def check(self) -> dict[str, float]:
        """The community's shortfall beyond its margin: the homes' reported shortfalls, less what the margin already
        covers (homes were asked for more than was sold) and less what other homes were dispatched to cover. Only
        that is moved, in one round for the whole community."""
        now = self.m.now()
        self.dirty, self.checked_at = False, now
        steps = sorted({t for r in self.reports.values() for t in r if t > now})
        need: dict[str, float] = {}
        drift = {h for h, why in self.reasons.items() if why == "the forecast changed"}
        for t in steps:
            _, ev0 = self._cushion(t)
            m = None
            if ev0 is not None:
                half = t[:14] + ("00" if t[14:16] < "30" else "30")
                sid = next((x["id"] for x in ev0["need"]["slots"] if x["start"] == half), None)
                m = self._measured(ev0, sid) if sid is not None else None
            tight = m is not None and m[0] < 1.1 * m[1]       # the meters say the community is near what it sold
            if self._quiet() and m is None:
                tight = True                                  # no meters on the desk in the window: a report beyond limits counts
            total = sum(r.get(t, 0.0) for h, r in self.reports.items() if h not in drift or tight)
            cushion, ev = self._cushion(t)
            if self._quiet():
                cushion *= QUIET_CUSHION                      # shortfalls under each node's limit don't reach the desk, and they use the margin too
            if ev is None and not any(c["kind"] != "gap" and c["start"] <= t < c["end"] for c in self.calls.values()):
                continue                                # owed nowhere the desk sells (a home's own plan)
            left = total - cushion - self._covered(t)
            floor = max(0.5, MIN_GAP_SHARE * self._sold(t)) if self._quiet() else 0.5
            if left > floor:
                need[t] = round(left, 3)
            elif total > 0:
                self.absorbed[t] = round(min(total, cushion), 3)
        if not need:
            if steps:
                self.m.log("rt_gap_absorbed", homes=sum(1 for r in self.reports.values() if any(t > now for t in r)),
                           short_kw=round(max(sum(r.get(t, 0.0) for r in self.reports.values()) for t in steps), 2),
                           why="within the margin and what other homes already cover")
            return {}
        who = sorted((h for h, r in self.reports.items() if any(t in need for t in r)),
                     key=lambda h: -max(self.reports[h].values(), default=0.0))
        ts = sorted(need)
        gid = f"gap-{ts[0].replace(' ', 'T')}-{uuid.uuid4().hex[:4]}"
        reporters = {h for h, r in self.reports.items() if any(t > now for t in r)}
        parent, helpers = self._parent(ts[0], who[0] if who else "")
        helpers = [h for h in helpers if h not in reporters]
        rec = {"id": gid, "kind": "gap", "start": ts[0], "end": beckn.add_minutes(ts[-1], 15), "kw": max(need.values()),
               "price": self._gap_price(ts[0]), "base_price": self._gap_price(ts[0]),
               "why": "; ".join(sorted({self.reasons.get(h, "") for h in who if self.reasons.get(h)}))[:160],
               "from": who[0] if who else "", "homes_short": who, "answers": {}, "alloc": {}, "round": 0,
               "phase": "asking", "received_at": now, "readings": {}, "short_by_ts": need, "parent": parent}
        self.calls[gid] = rec
        self.m.log("rt_gap", gap_id=gid, home=rec["from"], homes=len(who), kw=round(rec["kw"], 2), start=rec["start"],
                   end=rec["end"], reason=rec["why"], parent=(parent or {}).get("id"))
        if not helpers:
            rec.update({"phase": "dispatched", "confirmed_kw": 0.0, "short_kw": round(rec["kw"], 3), "homes": 0})
            self._warn(rec)
        else:
            self._ask(rec, helpers)
        return need

    def tick(self) -> None:
        now = self.m.now()
        live = not self.m.cfg.get("sync_broadcast")
        wait_s = float(self.m.cfg.get("rt_answer_wait_s", ANSWER_WAIT_S))
        if live:                                        # NANDA Town behind: the answers take that much longer too
            try:
                wait_s += min(float(self.m.cfg.get("rt_answer_wait_max_extra_s", 90.0)), float(self.m.town_lag()))
            except Exception:                           # noqa: BLE001  no lag known: the plain wait
                pass
        for rec in list(self.calls.values()):
            if rec["phase"] == "asking" and now >= beckn.add_minutes(rec["asked_at"], ANSWER_WAIT_MIN) \
                    and (not live or time.time() - rec.get("asked_wall", 0.0) >= wait_s):
                self.allocate(rec)                      # some homes did not answer in time: go with what came
        busy = any(r["kind"] == "gap" and r["phase"] == "asking" for r in self.calls.values())
        if self.dirty and not busy and (self.checked_at is None or now >= beckn.add_minutes(self.checked_at, 5)):
            self.check()

    def summary(self) -> list[dict[str, Any]]:
        return [{**{k: r.get(k) for k in ("id", "kind", "start", "end", "kw", "price", "phase", "confirmed_kw", "short_kw",
                                          "homes", "why", "from", "round", "result")},
                 "answers": len(r["answers"]), "asked": len(r.get("asked", [])),
                 "parent": (r.get("parent") or {}).get("id")} for r in self.calls.values()]
