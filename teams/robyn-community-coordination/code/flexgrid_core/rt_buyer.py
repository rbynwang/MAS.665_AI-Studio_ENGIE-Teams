"""The buyer's grid watch: turns what the buyer sees during the day into short-notice calls, and pays for what the
meters measured on top of the booking (the second part of a two-part settlement).

What it watches (rules in code; the buyer's AI agent may raise or skip a call, brain.py, but the rule decides first):
  * alerts: an emergency (a Power Caution) the buyer is told about -> at once, a call from the next 15-minute step
    until the alert ends, for the kW the alert names, at twice the program's price
  * prices: every 15 minutes between 12:00 and 21:00, if the real-time price of the 15 minutes just gone was more
    than twice the day-ahead price and above $100/MWh -> a short-notice call for the next hour for half the program's
    kW at 1.5 times its price (at most twice a day, never while another call is on)
Payment for a call (per half hour, for the community as one unit, as the DEG settlement pays the booking):
  extra = the measured reduction of every meter the buyer has a reading from (baseline - use, never below 0 per meter)
          - the kW the booking already pays for in that half hour (the contract's capacity offered; 0 outside it)
  paid  = min(extra, the kW called) x 0.5 h x the call's price (x the share of the half hour inside the call)
The coordinator splits it between the homes it dispatched, by what each measured beyond its own booking.
On the quiet path (report_rule.py) readings come in one batch after the window, so a call waits for them (cfg call_settle_after_min).
Live, a call is settled an hour (dataset) after it ends, once no reading has come for `utility_settle_quiet_s` real
seconds (30) and the coordinator has confirmed it (or 6 dataset hours have passed): through a busy NANDA Town the
readings, and the confirm itself, can arrive minutes late (seen 1 Nov 2026: a call settled at $0 before the
coordinator had even heard of it).
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from . import beckn
from . import settlement as st

PRICE_RULE = {"ratio": 2.0, "min_usd_mwh": 100.0, "from": "12:00", "to": "21:00", "hours": 1, "kw_share": 0.5,
              "price_x": 1.5, "per_day": 2}
EMERGENCY_PRICE_X = 2.0
SETTLE_AFTER_MIN = 60          # dataset minutes after a call ends: its last readings are in
CONFIRM_WAIT_MIN = 360         # live: a call the coordinator has not confirmed yet waits this long before it settles


def beckn_minutes(a: str, b: str) -> float:
    """Minutes from a to b (dataset local times; negative if b is earlier)."""
    from datetime import datetime
    f = "%Y-%m-%d %H:%M"
    return (datetime.strptime(b, f) - datetime.strptime(a, f)).total_seconds() / 60.0


def half_hour(ts: str) -> str:
    return ts[:14] + ("00" if ts[14:16] < "30" else "30")


class GridWatch:
    def __init__(self, util):
        self.u = util
        self.alerts: list[dict[str, Any]] = []          # {"at", "end", "kw", "why"}: told to the buyer (live or replay)
        self.calls: dict[str, dict[str, Any]] = {}
        self.price_calls: dict[str, int] = {}           # day -> price calls that went out
        self.enabled = bool(util.cfg.get("grid_watch", True))
        self.price_rule = True                          # a fixed-order buyer calls only on an alert (stand_in.py)

    # ------------------------------------------------------------------ what the buyer runs on
    def program(self, day: str | None = None) -> tuple[float, float]:
        """The program's kW and price per kWh: that day's booked need if there is one, else the catalog's first."""
        pairs = self.u._pairs()
        needs = [beckn.need_from_catalog(r, o) for r, o in pairs]
        pick = next((n for n in needs if day and n["window_start"][:10] == day), needs[0] if needs else None)
        return (float(pick["slots"][0]["kw_requested"]), float(pick["slots"][0]["price"])) if pick else (100.0, 2.04)

    def _row(self, ts: str) -> dict[str, Any] | None:
        i = self.u.clock.idx_of(ts)
        return self.u.clock.row(i) if i is not None else None

    @staticmethod
    def _q(ts: str) -> str:
        """Round up to the next 15-minute step (the steps the homes plan in)."""
        return beckn.add_minutes(ts, (15 - int(ts[14:16]) % 15) % 15)

    @staticmethod
    def _next(ts: str) -> str:
        """The first 15-minute step after ts. The live clock stands on whole steps, so a call 'from the next step'
        rounded up from 21:15 began at 21:15 itself, before any home could hear of it."""
        return GridWatch._q(beckn.add_minutes(ts, 1))

    # ------------------------------------------------------------------ calls
    def alert(self, kw: float, minutes: float, why: str = "grid emergency", at: str | None = None) -> dict[str, Any]:
        """An emergency the buyer is told about now (the audience's 'grid emergency', or a replayed Power Caution)."""
        at = at or self.u.now()
        a = {"at": at, "end": beckn.add_minutes(self._next(at), max(15.0, float(minutes))), "kw": float(kw), "why": why}
        self.alerts.append(a)
        return a

    def call(self, kind: str, start: str, end: str, kw: float, price: float, why: str) -> dict[str, Any]:
        cid = f"call-{start[:10]}-{start[11:13]}{start[14:16]}-{kind}-{uuid.uuid4().hex[:4]}"
        rec = {"id": cid, "kind": kind, "start": start, "end": end, "kw": round(kw, 3), "price_usd_per_kwh": round(price, 4),
               "why": why, "issued_at": self.u.now(), "confirmed_kw": None, "warnings": [], "readings": {},
               "status": "OPEN", "issued_wall": time.time()}
        self.calls[cid] = rec
        self.u.event_days.add(start[:10])               # a call day is a curtailment day: never in a later baseline
        to = sorted(self.u.aggregators)[0]
        rec["to"] = to
        body = beckn.make("rt_call", sender=self.u.name, receiver=to, transaction_id=cid,
                          message={"call": {k: rec[k] for k in ("id", "kind", "start", "end", "kw", "price_usd_per_kwh", "why")}},
                          key=self.u.key, sender_role="bpp")
        self.u.town.send(to, beckn.kind("rt_call"), body, f"{self.u.name}:rt_call:{cid}")
        self.u.event("rt_call", call_id=cid, call_kind=kind, kw=rec["kw"], start=start, end=end, price=rec["price_usd_per_kwh"], why=why)
        return rec

    def end_call(self, cid: str, at: str | None = None) -> dict[str, Any]:
        rec = self.calls[cid]
        at = self._q(at or self.u.now())
        if at < rec["end"]:
            rec["end"], rec["ended_early"] = at, True
            body = beckn.make("rt_end", sender=self.u.name, receiver=rec["to"], transaction_id=cid,
                              message={"end": {"id": cid, "at": at}}, key=self.u.key, sender_role="bpp")
            self.u.town.send(rec["to"], beckn.kind("rt_end"), body, f"{self.u.name}:rt_end:{cid}")
            self.u.event("rt_end", call_id=cid, at=at)
        return rec

    def tick(self) -> None:
        if not self.enabled:
            return
        now = self.u.now()
        for a in self.alerts:
            if not a.get("called") and now >= a["at"]:
                a["called"] = True
                kw0, p0 = self.program(now[:10])
                start = max(self._next(a["at"]), self._q(now))
                self.call("emergency", start, a["end"], float(a.get("kw") or kw0), p0 * EMERGENCY_PRICE_X, a.get("why", "emergency"))
        r = PRICE_RULE
        day, hm = now[:10], now[11:16]
        last = self._row(beckn.add_minutes(now, -15))          # the 15 minutes just gone: its real-time price is known
        if self.price_rule and last and r["from"] <= hm <= r["to"] and self.price_calls.get(day, 0) < r["per_day"] \
                and not any(c["status"] == "OPEN" and c["start"] <= now < c["end"] for c in self.calls.values()):
            rt, da = last["price_usd_mwh"], last["price_da_usd_mwh"]
            if rt > r["min_usd_mwh"] and rt > r["ratio"] * max(1.0, da):
                self.price_calls[day] = self.price_calls.get(day, 0) + 1
                kw0, p0 = self.program(day)
                start = self._q(now)
                self.call("short_notice", start, beckn.add_minutes(start, 60 * r["hours"]), kw0 * r["kw_share"],
                          p0 * r["price_x"], f"real-time ${rt:.0f}/MWh at {last['ts_local'][11:]} against ${da:.0f} planned")
        quiet = float(self.u.cfg.get("utility_settle_quiet_s", 0.0))
        after = float(self.u.cfg.get("call_settle_after_min", SETTLE_AFTER_MIN))
        for rec in self.calls.values():
            ref = rec["end"]
            if after > SETTLE_AFTER_MIN:                # the quiet path: a call inside a sale gets its readings in the sale's one envelope, after the window
                ref = max([ref] + [c["need"]["window_end"] for c in self.u.contracts.values()
                                   if c.get("need") and c["need"]["window_start"] < rec["end"] and rec["start"] < c["need"]["window_end"]])
            if rec["status"] != "OPEN" or now < beckn.add_minutes(ref, after):
                continue
            if quiet > 0:      # live: readings (the call's and the booking's) may still be on their way through a busy
                last = max([rec.get("last_reading_wall", 0.0), rec.get("issued_wall", 0.0)]     # Town, and the
                           + [ct.get("last_reading_wall", 0.0) for ct in self.u.contracts.values()])  # coordinator's
                unconfirmed = rec.get("confirmed_kw") is None and now < beckn.add_minutes(rec["end"], CONFIRM_WAIT_MIN)
                if time.time() - last < quiet or unconfirmed:     # confirm may not be in yet: wait for a quiet inbox
                    continue
            self.settle(rec)

    # ------------------------------------------------------------------ inbox
    def handle(self, act: str, sender: str, body: dict[str, Any]) -> str | None:
        if act != "rt_confirm":
            return None
        m = body.get("message") or {}
        if "confirm" in m:
            c = m["confirm"]
            rec = self.calls.get(c.get("id"))
            if not rec:
                return "confirm for an unknown call"
            rec.update({"confirmed_kw": c.get("kw"), "short_kw": c.get("short_kw"), "confirmed_homes": c.get("homes", [])})
            if c.get("refused"):                        # the desk took nothing on: nobody was asked, and nothing will be read or paid
                rec.update({"status": "REFUSED", "refused": c["refused"]})
            self.u.event("rt_confirmed", call_id=rec["id"], kw=c.get("kw"), short_kw=c.get("short_kw"),
                         homes=len(c.get("homes", [])))
            return f"call {rec['id']}: {c.get('kw')} kW confirmed"
        if "warning" in m:
            w = m["warning"]
            for rec in self.calls.values():
                if rec["start"] <= w.get("start", "") < rec["end"]:
                    rec["warnings"].append(w)
            self.u.event("rt_warning", **{k: w.get(k) for k in ("id", "short_kw", "start", "end", "why")})
            return f"warning: {w.get('short_kw')} kW short from {w.get('start')}"
        return None

    def reading(self, sender: str, body: dict[str, Any]) -> str | None:
        """A home's signed reading for a half hour of a call it was dispatched for (checked like the event's)."""
        r = (body.get("message") or {}).get("performance") or {}
        rec = self.calls.get(r.get("event_id"))
        if rec is None:
            return None
        pin = self.u.pins.setdefault(sender, body.get("signer", ""))
        if not beckn.check(body, pin) or r.get("home") != sender:
            rec["rejected"] = rec.get("rejected", 0) + 1
            self.u.event("reading_rejected", home=sender, event_id=rec["id"])
            return "rejected: bad signature"
        rec["readings"].setdefault(sender, {})[r["start"]] = {"usage_kw": float(r["usage_kw"]),
                                                              "dispatched_kw": float(r.get("committed_kw", 0.0)),
                                                              "battery_kw": float(r.get("battery_kw") or 0.0)}
        rec["last_reading_wall"] = time.time()
        return f"call reading {sender} {r['start'][11:]}"

    # ------------------------------------------------------------------ the second part of the settlement
    def _booking(self, slots: list[str]) -> tuple[dict[str, float], dict[str, dict[str, tuple[float, float, float]]]]:
        """For these half hours: the kW the booking pays for (its capacity offered), and each enrolled meter's
        (baseline, use, booked kW) from the event's own readings."""
        offered: dict[str, float] = {}
        per: dict[str, dict[str, tuple[float, float, float]]] = {}
        for ct in self.u.contracts.values():
            if ct.get("status") not in ("ACTIVE", "COMPLETE") or not ct.get("baselines"):
                continue
            bases = self.u.settlement_baselines(ct)
            for s in ct["need"]["slots"]:
                if s["start"] not in slots:
                    continue
                sid = int(s["id"])
                offered[s["start"]] = offered.get(s["start"], 0.0) + float((ct.get("pool_offered") or ct.get("offered") or {}).get(sid, 0.0))
                for m in ct.get("meters", []):
                    if sid in (ct.get("abnormal") or {}).get(m, ()):      # a closed building's fall isn't a cut, here as in the booking
                        continue
                    use = ct.get("meter_readings", {}).get(m, {}).get(sid)
                    base = bases.get(m, {}).get(str(sid), {}).get("BASELINE")
                    if base is not None and st.MARKET:    # the booking's own measure, with the same-day adjustment
                        base = st.adjusted(base, ct.get("meter_pre", {}).get(m), ct.get("pre_baselines", {}).get(m))
                    if use is not None and base is not None:
                        booked = float(ct.get("meter_committed", {}).get(m, {}).get(sid, 0.0))
                        per.setdefault(s["start"], {})[m] = (float(base), float(use), booked)
        return offered, per

    def settle(self, rec: dict[str, Any]) -> dict[str, Any]:
        slots = sorted({s for per in rec["readings"].values() for s in per})
        if not slots:
            rec.update({"status": "COMPLETE", "paid_usd": 0.0, "delivered_kwh": 0.0, "by_home": {}, "rows": []})
            self._tell_coordinator(rec)
            return rec
        offered, per = self._booking(slots)
        fleet = getattr(self.u, "fleet", None) or {}       # the meters the utility knows (a building node's kind)
        need = {"window_start": slots[0], "slots": [{"id": k, "start": s} for k, s in enumerate(slots)]}
        call_bases = self.u.baselines({"need": need}, sorted(rec["readings"]))
        rows, by_home, by_batt, total = [], {}, {}, 0.0
        for k, s in enumerate(slots):
            meters = dict(per.get(s, {}))
            for h, rd in rec["readings"].items():         # dispatched homes not in the booking (or not reading in it)
                if s in rd and h not in meters:
                    b = call_bases.get(h, {}).get(str(k), {}).get("BASELINE")
                    if b is not None:
                        meters[h] = (float(b), rd[s]["usage_kw"], 0.0)
            # a building node's meter counts at most 1.2x what it owed (booked + dispatched): its baseline picks days
            # without its own routine evening cut (settlement.building_counted_kwh)
            def meas(h, b, u, bk):
                v = st.clamp0(b - u)
                if (fleet.get(h) or {}).get("kind") != "building":
                    return v
                disp = ((rec["readings"].get(h) or {}).get(s) or {}).get("dispatched_kw", 0.0)
                return min(v, st.BUILDING_CAP_X * (bk + float(disp or 0.0)))
            measured = sum(meas(h, b, u, bk) for h, (b, u, bk) in meters.items())
            booked = offered.get(s, 0.0)
            # the part of this half hour inside the call (a call from :15 covers half of its first half hour)
            frac = max(0.0, min(30.0, beckn_minutes(max(s, rec["start"]), min(beckn.add_minutes(s, 30), rec["end"])))) / 30.0
            extra = min(st.clamp0(measured - booked), rec["kw"]) * frac
            beyond = {h: st.clamp0(meas(h, *meters[h]) - meters[h][2]) for h, rd in rec["readings"].items()
                      if s in rd and h in meters}
            beyond = {h: min(v, rec["readings"][h][s]["dispatched_kw"]) for h, v in beyond.items()}
            tot_b = sum(beyond.values())
            for h, v in beyond.items():                   # who made the extra: the dispatched homes, pro rata
                kwh = (extra * v / tot_b if tot_b > 0 else 0.0) * 0.5
                by_home[h] = by_home.get(h, 0.0) + kwh
                by_batt[h] = by_batt.get(h, 0.0) + min(kwh, st.clamp0(rec["readings"][h][s].get("battery_kw", 0.0)) * 0.5)
            total += extra * 0.5
            rows.append({"start": s, "measured_kw": round(measured, 3), "booked_kw": round(booked, 3),
                         "extra_kw": round(extra, 3), "share_of_half_hour": round(frac, 3), "meters": len(meters)})
        paid = round(total * rec["price_usd_per_kwh"], 4)
        rec.update({"status": "COMPLETE", "paid_usd": paid, "delivered_kwh": round(total, 3), "rows": rows,
                    "by_home": {h: round(v, 4) for h, v in by_home.items()},
                    "by_home_batt": {h: round(v, 4) for h, v in by_batt.items()}})
        self.u.event("rt_settled", call_id=rec["id"], paid_usd=paid, delivered_kwh=round(total, 3), homes=len(by_home))
        self._tell_coordinator(rec)
        return rec

    def _tell_coordinator(self, rec: dict[str, Any]) -> None:
        to = rec.get("to") or sorted(self.u.aggregators)[0]
        msg = {"settlement": {"call_id": rec["id"], "kind": rec["kind"], "price_usd_per_kwh": rec["price_usd_per_kwh"],
                              "paid_usd": rec["paid_usd"], "delivered_kwh": rec["delivered_kwh"],
                              "by_home_kwh": rec["by_home"], "by_home_batt_kwh": rec.get("by_home_batt", {}),
                              "rows": rec.get("rows", []), "currency": "USD"}}
        self.u.reply(to, "on_status", rec["id"], msg)

    def summary(self) -> list[dict[str, Any]]:
        keys = ("id", "kind", "start", "end", "kw", "price_usd_per_kwh", "why", "issued_at", "confirmed_kw", "short_kw",
                "status", "paid_usd", "delivered_kwh", "ended_early")
        return [{**{k: r.get(k) for k in keys}, "warnings": len(r["warnings"]),
                 "homes_reading": len(r["readings"])} for r in self.calls.values()]
