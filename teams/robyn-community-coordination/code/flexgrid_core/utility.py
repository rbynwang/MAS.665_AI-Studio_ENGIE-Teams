"""flexgrid-utility - the stand-in grid buyer, "Utility or ENGIE (mock)", built from public data only.

It plays the provider node of the DEG demand-flex flow (beckn/DEG docs/implementation-guides/v2/Demand_Flexibility):
  discover(date)  -> on_discover   the next day's flex needs from data/grid_events_2023.json, once its notice time
                                   (10:00 the day before) has passed
  select / init / confirm -> on_select / on_init / on_confirm (contract DRAFT -> ACTIVE)
  update (meters) -> on_status baselines   settlement.baseline_rule(), from each meter's history
  on_status (meter reading) from each home -> checked against the home's pinned key and kept
  update (readings done) -> on_status actuals, then on_status settled (the DEG demand-flex rego, ported)
Meter history = each enrolled home replayed in its default mode over the days the baseline needs, plus one day of
warm-up (the utility's own meter data in this simulation).
Actual usage comes from each home's signed readings, sent by the home straight to the utility (the coordinator never
relays them, so it cannot alter them); a real utility reads its own meters.
Who may do what: only the aggregators in `utility_aggregators` (config, default ["coordinator"]) discover and
contract, every follow-up must come from the aggregator that selected, a meter is enrolled in at most one open
contract per event, and a reading counts only for the contract its meter is enrolled in, active or settled. A select, init,
confirm or update delivered again (NANDA Town delivers at least once) changes nothing: a confirmed contract keeps its
meters and capacity, and a settled one is settled again only on readings that came after it (a top-up).

  flexgrid-utility run        # Town agent "utility" + status API on 127.0.0.1:7873
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
import time
from collections import Counter, deque
from typing import Any

from . import beckn, config, doors, ids, keep, settlement as st, signing, wire
from .adapters.simulated import SimulatedHome
from .replay import Replay
from . import meter_check as mc
from .town import InboxPoller, LagMeter, TownSession

log = logging.getLogger("flexgrid-utility")


def _meter_cfg(home: dict[str, Any], ev_max: float) -> dict[str, Any]:
    b = home["battery"]
    return {"battery_kwh": b["kwh"], "battery_kw": b["kw"], "reserve_pct": b["reserve_pct"],
            "soc_max_pct": b["soc_max_pct"], "soc0_pct": b.get("soc0_pct", 55.0), "eta": b["eta"],
            "ev_max_kw": ev_max, "has_ev": home["has_ev"], "export_limit_kw": home["export_limit_kw"]}


class Utility:
    def __init__(self, key_file: str = "utility.key"):
        """key_file: each buyer loads its own key and no other's, so two buyers starting together share no file."""
        self.cfg = config.load()
        self.name = self.cfg.get("utility_name", "utility")
        self.key = signing.load_or_create(config.CONFIG_DIR / "secrets" / key_file)
        self.pub = signing.public_hex(self.key)
        tok = config.secret("utility_town_token") or config.secret("town_token")
        self.town = TownSession(self.cfg["town_url"], self.cfg["town_run_id"], self.name, tok)
        self.catalog = st.catalog_terms(json.loads(config.events_file().read_text())["message"]["catalogs"][0])
        self.fleet = {h["name"]: h for h in json.loads(config.data_path("fleet_v2.json").read_text())["homes"]}
        from . import building                   # the building nodes this config enables: their meters' history too
        self.fleet.update({b["name"]: building.meter_entry(b) for b in building.enabled(self.cfg)})
        c = self.cfg
        self.clock = Replay(c.get("clock_profile", "tesla_calhoun"), c["replay_start"], c["replay_step_seconds"],
                            c["replay_loop_days"])
        self.pins: dict[str, str] = {}
        self.orders: dict[str, dict[str, Any]] = {}          # event_id -> the buyer agent's order, signed by us
        self.contracts: dict[str, dict[str, Any]] = {}       # transactionId -> contract state
        self.events = deque(maxlen=300)
        self.unsent: deque = deque()                         # replies Town didn't take: sent again on each tick, under the same id
        self.event_days = {beckn.need_from_catalog(r, o)["window_start"][:10]
                           for r, o in self._pairs()}
        self.lock = threading.RLock()
        self.traffic_homes: Counter = Counter()
        self.lag = LagMeter(window=float(c.get("town_lag_window_s", 20.0)))   # how far behind the inbox is
        # Town names allowed to discover and contract with this buyer (the aggregators it has enrolled). Homes only
        # send meter readings, and a reading counts only for the ACTIVE contract its meter is enrolled in.
        self.aggregators = set(c.get("utility_aggregators") or ["coordinator"])
        self.keeper = None                                   # keep.for_buyer: live only
        self.run_wall = 0.0                                  # when this run started (POST /demo/reset): older messages
                                                             # still in the inbox belong to the run before
        self.play = ids.new_play()                           # this play of the day, in every id it sends (ids.py): new at each reset, kept through a restart
        from .rt_buyer import GridWatch                      # calls during the day, and their settlement
        self.watch = GridWatch(self)

    def _pairs(self):
        """The DEG needs this buyer publishes: the catalog's, with a buyer agent's order in place of that day's."""
        offers = {o["resourceIds"][0]: o for o in self.catalog["offers"]}
        pairs = {r["id"]: (r, offers[r["id"]]) for r in self.catalog["resources"] if r["id"] in offers}
        for eid, o in self.orders.items():
            pairs[eid] = (o["resource"], o["offer"])
        return list(pairs.values())

    # ------------------------------------------------------------------ the buyer agent's orders (POST /needs)
    # Rebuilt on 2026-10-29 to the contract its callers state (tests/test_verify_run.py, flexgrid_core/mcp_server.py,
    # flexgrid_core/verify.py, tools/local_stack.py, tools/demo_event.py) after a sync overwrote the first version.
    ORDER_MAX_KW, ORDER_MAX_PRICE, ORDER_MAX_H = 20000.0, 5.0, 4.0
    ORDER_PREFIX = "flex-need-"                         # the event id of a day's order (the catalog's ids)
    PARTICIPANT = "utility-or-engie-mock"               # who the contract names as the buyer
    CLASS_RATES = False                                 # a call at each class's rate. On the one-buyer path a call pays its nodes all of it, pro rata, so there it stays at the call's price

    @staticmethod
    def notice_of(date: str) -> str:
        """When the community's coordinator asks for a day's needs: 10:00 the day before."""
        return f"{beckn.add_minutes(f'{date} 00:00', -1440)[:10]} 10:00"

    def needs_list(self) -> list[dict[str, Any]]:
        out = []
        for r, o in self._pairs():
            n = beckn.need_from_catalog(r, o)
            day = n["window_start"][:10]
            order = self.orders.get(n["event_id"])
            out.append({"event_id": n["event_id"], "window": [n["window_start"], n["window_end"]],
                        "kw": n["slots"][0]["kw_requested"], "price_usd_per_kwh": n["slots"][0]["price"],
                        "notice": self.notice_of(day), "commit_by": n["commit_by"],
                        "source": "buyer agent order" if order else "catalog",
                        **({"buyer": order["buyer"], "note": order["order"].get("note", "")} if order else {})})
        return sorted(out, key=lambda x: x["window"][0])

    def place_order(self, o: dict[str, Any], at: str | None = None) -> dict[str, Any]:
        """A buyer agent's order for one day's window becomes that day's DEG need (DemandFlexNeed + BuyOffer), signed by
        the utility. Raises ValueError("400: ...") for a malformed order and ValueError("409: ...") for one that comes
        too late (the call for offers has gone out) or twice."""
        def bad(msg: str):
            raise ValueError(f"400: {msg}")
        date, start, end = str(o.get("date", "")), str(o.get("start", "")), str(o.get("end", ""))
        try:
            d0 = beckn.add_minutes(f"{date} 00:00", 0)
        except ValueError:
            bad("date must be YYYY-MM-DD")
        meta = config.dataset_meta()                          # a baseline needs 14 days of data before the day
        lo, hi = meta.get("baseline_from", meta["range"][0]), meta["range"][1]
        if meta["name"] == "il2023":
            lo, hi = "2023-01-21", "2023-12-30"
        if not lo <= date <= hi:
            bad(f"{date} is outside the data this buyer runs on ({lo} to {hi})")
        def minutes(hhmm: str) -> int:
            try:
                h, m = (int(x) for x in hhmm.split(":"))
            except ValueError:
                bad("start and end are local times HH:MM")
            if not (0 <= h < 24 and m in (0, 30)):
                bad("start and end must be on the half hour (HH:00 or HH:30)")
            return h * 60 + m
        m0, m1 = minutes(start), minutes(end)
        if m1 <= m0:
            bad("the window must end after it starts")
        if m1 - m0 > self.ORDER_MAX_H * 60:
            bad(f"the window may be at most {self.ORDER_MAX_H:g} h")
        try:
            kw, price = float(o.get("kw")), float(o.get("price_usd_per_kwh"))
        except (TypeError, ValueError):
            bad("kw and price_usd_per_kwh must be numbers")
        if not 0 < kw <= self.ORDER_MAX_KW:
            bad(f"kw must be more than 0 and at most {self.ORDER_MAX_KW:g} per slot")
        if not 0 < price <= self.ORDER_MAX_PRICE:
            bad(f"price_usd_per_kwh must be more than 0 and at most {self.ORDER_MAX_PRICE:g}")
        # at: the buyer's clock in an offline replay (tools/sim_event.py), which starts the day at 00:00; never taken
        # from the order itself, so a live order cannot be backdated past its notice time
        eid, notice, now = f"{self.ORDER_PREFIX}{date}", self.notice_of(date), (at or self.now())
        if eid in self.orders:
            raise ValueError(f"409: an order for {date} was already placed by {self.orders[eid]['buyer']}")
        if now >= notice:
            raise ValueError(f"409: too late for {date}: the call for offers went out at {notice} (now {now})")
        if any(c.get("need", {}).get("event_id") == eid for c in self.contracts.values()):
            raise ValueError(f"409: {date} is already contracted")
        # the DEG resource + offer, in the catalog's own shape (tools/make_grid_events.py)
        tr = self.catalog["resources"][0]
        to = next(x for x in self.catalog["offers"] if x["resourceIds"][0] == tr["id"])
        n_slots = (m1 - m0) // beckn.SLOT_MINUTES
        ws = f"{date} {start}"
        utc = beckn.utc_from_local                       # the dataset's local time, as the catalog
        tzl = config.dataset_meta().get("tz_label", "CDT")
        resource = json.loads(json.dumps(tr))
        resource.update({"id": eid, "descriptor": {"name": f"Buyer order {date}",
                                                    "shortDesc": f"{kw:g} kW, {start}-{end} {tzl}"},
                         "quantity": {**tr.get("quantity", {}), "unitQuantity": kw}})
        ra = resource["resourceAttributes"]
        ra["intervalPeriod"] = {**ra["intervalPeriod"], "start": utc(ws)}
        ra["intervals"] = [{"id": k, "payloads": [{"type": "CAPACITY_REQUESTED", "values": [kw]},
                                                  {"type": "PRICE", "values": [price]},
                                                  {"type": "SHORTFALL_PENALTY", "values": [0.0]}]}
                           for k in range(n_slots)]
        offer = json.loads(json.dumps(to))
        offer.update({"id": f"offer-{date}", "resourceIds": [eid],
                      "descriptor": {"name": f"Buyer order {date}", "shortDesc": f"{st.up_to()}${price:g}/kWh reduced, measured against baseline"},
                      "validity": {"endDate": utc(f"{beckn.add_minutes(f'{date} 00:00', -1440)[:10]} 16:00")}})
        for x in offer["offerAttributes"].get("inputs") or []:
            if x.get("role") == "buyer":
                x["participantId"] = self.PARTICIPANT
        deposit = max(0.0, float(o.get("deposit_usd") or 0.0))
        if deposit > 0:                                  # it holds tentative power: the coordinator reads it from the signed offer
            offer["offerAttributes"]["deposit"] = {"usd": deposit}
        need = beckn.need_from_catalog(resource, offer)
        rec = signing.sign({"event_id": eid, "buyer": str(o.get("buyer") or "buyer agent")[:80], "placed_at": now,
                            "order": {"date": date, "start": start, "end": end, "kw": kw, "price_usd_per_kwh": price,
                                      "note": str(o.get("note") or "")[:200], **({"deposit_usd": deposit} if deposit > 0 else {})},
                            "notice": notice, "commit_by": need["commit_by"], "resource": resource, "offer": offer},
                           self.key)
        self.orders[eid] = rec
        self.event_days.add(need["window_start"][:10])     # an event day never counts in a later PTR baseline
        self.event("need_posted", event_id=eid, buyer=rec["buyer"], kw=kw, window=[need["window_start"], need["window_end"]],
                   price_usd_per_kwh=price, **({"sim": at} if at else {}))   # an offline replay: the buyer's own clock
        return rec

    def contract_of_event(self, event_id: str) -> dict[str, Any] | None:
        """The latest contract for an event (a settled one first)."""
        cs = [c for c in self.contracts.values() if c.get("need", {}).get("event_id") == event_id]
        return next((c for c in reversed(cs) if c.get("status") == "COMPLETE"), cs[-1] if cs else None)

    def evidence(self, event_id: str) -> dict[str, Any]:
        """What the utility holds about one event, for the independent check (flexgrid_core/verify.py)."""
        ct = self.contract_of_event(event_id)
        if not ct:
            raise KeyError(event_id)
        bases = self.settlement_baselines(ct) if ct.get("baselines") else {}
        return {"event_id": event_id, "need": ct["need"], "order": self.orders.get(event_id), "utility_pubkey": self.pub,
                "contract_id": ct.get("contract", {}).get("id"), "status": ct.get("status"),
                "meters": ct.get("meters", []), "pins": {m: self.pins[m] for m in ct.get("meters", []) if m in self.pins},
                "offered": {str(k): v for k, v in (ct.get("offered") or {}).items()},
                "baselines": {m: {s: v for s, v in d.items() if s != "_days"} for m, d in bases.items()},
                "pre_baselines": ct.get("pre_baselines") or {},
                "meter_readings": {m: {str(s): v for s, v in d.items()} for m, d in ct.get("meter_readings", {}).items()},
                "rejected_readings": ct.get("rejected_readings", 0), "settlement": ct.get("settlement"), "owed": ct.get("owed"), "solar": ct.get("solar"), "meter_pre": ct.get("meter_pre") or {},
                "buyer_review": ct.get("buyer_review"),
                **({"share": {str(k): v for k, v in ct["share"].items()}} if ct.get("share") else {}),
                **({"capped": {str(k): v for k, v in ct["capped"].items()}} if ct.get("capped") else {}),
                **({"abnormal": ct["abnormal"]} if ct.get("abnormal") else {})}

    def review(self, event_id: str, approve: bool, reason: str, checks: Any, by: str) -> dict[str, Any]:
        """The buyer's sign-off on a settled event, signed by the utility and kept on the contract."""
        ct = self.contract_of_event(event_id)
        if not ct or ct.get("status") != "COMPLETE":
            raise ValueError(f"409: {event_id} is not settled yet")
        rec = signing.sign({"event_id": event_id, "decision": "approved" if approve else "disputed",
                            "reason": str(reason)[:500], "checks": checks, "by": str(by)[:80], "at": self.now(),
                            "settlement_usd": ct["settlement"]["total"]}, self.key)
        ct["buyer_review"] = rec
        self.event("buyer_review", event_id=event_id, decision=rec["decision"], by=rec["by"])
        return rec

    def event(self, kind: str, **d) -> None:
        self.events.append({"at": time.time(), "sim": self.now(), "kind": kind, **d})
        log.info("%s %s", kind, json.dumps(d, default=str)[:300])

    def now(self) -> str:
        return self.clock.ts(self.clock.current())

    def reply(self, to: str, action: str, txn: str, message: dict[str, Any], *key: Any, asked: str | None = None) -> None:
        """What's sent to the coordinator. `asked`: what this answers (ids.asked). `key`: what it is, or by default what it says, as with a call's settlement, where each top-up says
        something new. A settlement names its round: a second one in a round that says otherwise would be a fault, and Town refuses it, loudly."""
        # a status (baselines, actuals, the settlement) is a statement of fact: valid for a day (beckn.TTL_EVIDENCE_S)
        body = beckn.make(action, sender=self.name, receiver=to, transaction_id=txn, message=message, key=self.key,
                          sender_role="bpp", ttl_s=beckn.TTL_EVIDENCE_S if action == "on_status" else None)
        item = (to, beckn.kind(action), body, ids.sealed(None if asked else self.key, asked or self.play, self.name, to, action, txn, *(key or (ids.digest(message),))))
        if not self._post(*item):
            self.unsent.append(item)

    def _post(self, to: str, kind: str, body: dict[str, Any], mid: str) -> bool:
        """One send to Town. False if it failed, and it's sent again on a later tick: the first day of 6 Nov lost $21,388.67 to one failed send of a settlement."""
        try:
            out = self.town.send(to, kind, body, mid)
        except Exception as exc:                            # noqa: BLE001  Town away for a moment, or its tunnel
            self.event("send_failed", to=to, message_kind=kind, error=str(exc)[:160])
            return False
        if isinstance(out, dict) and out.get("duplicate"):    # Town refused it: see town.send
            self.event("send_dropped", to=to, message_kind=kind, message_id=mid)
        return True

    # ------------------------------------------------------------------ baselines
    def baselines(self, ct: dict[str, Any], meters: list[str]) -> dict[str, dict[str, dict[str, float]]]:
        """Each meter's baseline per slot from its own history (the utility's meter data, before the event).
        Always the baseline in force (settlement.baseline_rule). Under the 2026 rules that's the last 10 days of the same kind, otherwise the
        Ameren Rider PTR baseline (mean of the 3 highest of the 14 previous non-holiday, non-event days).
        With baseline_method "weather_regression" the contract also fixes, per meter and slot, a line fitted to the
        meter's usage against the day's highest temperature over the 30 previous such days; the settlement then uses
        that line at the event day's measured highest temperature (tools/baseline_eval.py: it pays for 1.06x the
        energy truly delivered over the 10 events, where Rider PTR pays for 2.38x)."""
        need = ct["need"]
        day = need["window_start"][:10]
        ev_days = {st.day_of(x) for x in self.event_days}
        days = st.similar_days(st.day_of(day), ev_days) if st.MARKET else st.ptr_baseline_days(st.day_of(day), ev_days)
        regression = self.cfg.get("baseline_method", "ptr") == "weather_regression"
        reg_days = st.ptr_baseline_days(st.day_of(day), ev_days, 30, 60) if regression else []
        first = min(days + reg_days).isoformat() if days else beckn.add_minutes(f"{day} 00:00", -14 * 1440)[:10]
        out, models = {}, {}
        for m in meters:
            h = self.fleet.get(m)
            if not h:
                continue
            rp = Replay(h["profile"], need["window_start"], 60, 0)
            ev_max = rp.sessions[0]["max_kw"] if rp.sessions else 0.0
            sim = SimulatedHome(_meter_cfg(h, ev_max), rp)
            i_from = rp.idx_of(beckn.add_minutes(f"{first} 00:00", -1440))   # one day of warm-up before the first
            i_to = rp.idx_of(f"{day} 00:00")
            series = dict(sim.run_default(i_from, i_to))
            def window(d: str) -> list[float] | None:
                vals = []
                for s in need["slots"]:
                    t0 = d + s["start"][10:]
                    t1 = beckn.add_minutes(t0, 15)
                    if t0 in series and t1 in series:
                        vals.append(-(series[t0] + series[t1]) / 2)
                return vals if len(vals) == len(need["slots"]) else None
            usage_by_day = {}
            for dd in days:
                v = window(dd.isoformat())
                if v:
                    usage_by_day[dd] = v
            b = (st.ten_of_ten if st.MARKET else st.ameren_baseline)(usage_by_day, st.day_of(day), ev_days)
            out[m] = {str(s["id"]): {"BASELINE": b["per_slot"][j] if b["per_slot"] else 0.0}
                      for j, s in enumerate(need["slots"])}
            if st.MARKET:                                   # the same-day adjustment's hour: 2 hours before the event
                p0 = beckn.add_minutes(need["window_start"], -120)[10:]
                pre = []
                for dd in days:
                    xs = [series.get(beckn.add_minutes(dd.isoformat() + p0, 15 * q)) for q in range(4)]
                    if all(x is not None for x in xs):
                        pre.append(-sum(xs) / 4)
                ct.setdefault("pre_baselines", {})[m] = round(sum(pre) / len(pre), 3) if pre else None
                ct.setdefault("solar", {})[m] = mc.recent_solar(rp, days, need["slots"])
            out[m]["_days"] = b["days"]
            if regression:
                tmax = self._tmax(rp, [d.isoformat() for d in reg_days])
                pts = [(tmax[d.isoformat()], window(d.isoformat())) for d in reg_days if d.isoformat() in tmax]
                pts = [(t, v) for t, v in pts if v]
                if len(pts) >= 10:
                    models[m] = {str(s["id"]): st.line_fit([t for t, _ in pts], [v[j] for _, v in pts])
                                 for j, s in enumerate(need["slots"])}
        if regression:
            ct["baseline_models"] = models
        return out

    @staticmethod
    def _tmax(rp: Replay, days: list[str]) -> dict[str, float]:
        out: dict[str, float] = {}
        for d in days:
            try:
                i0 = rp.idx_of(f"{d} 00:00")
            except Exception:
                continue
            out[d] = max(rp.row(i0 + k)["temp_c"] for k in range(96))
        return out

    def settlement_baselines(self, ct: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
        """The baselines the settlement uses: those sent with the contract, or, under weather_regression, each
        meter's line at the event day's measured highest temperature (known only once the day is over)."""
        models = ct.get("baseline_models")
        if not models:
            return ct.get("baselines", {})
        day = ct["need"]["window_start"][:10]
        out = {}
        for m, per in models.items():
            h = self.fleet.get(m)
            t = self._tmax(Replay(h["profile"], ct["need"]["window_start"], 60, 0), [day]).get(day)
            if t is None:
                continue
            out[m] = {sid: {"BASELINE": round(f["a"] + f["b"] * t, 4)} for sid, f in per.items()}
        return {**ct.get("baselines", {}), **out}

    # ------------------------------------------------------------------ inbox
    def _enroll(self, ct: dict[str, Any], requested: list[str]) -> list[str]:
        """The meters this contract may count: a meter is enrolled in at most one open contract per event."""
        eid = ct["need"]["event_id"]
        held = {m for c in self.contracts.values() if c is not ct and c.get("status") in ("DRAFT", "ACTIVE")
                and c.get("need", {}).get("event_id") == eid for m in c.get("meters", [])}
        refused = sorted(set(requested) & held)
        if refused:
            self.event("meters_refused", event_id=eid, aggregator=ct.get("coordinator"), meters=refused,
                       reason="enrolled in another contract for this event")
        ct["refused_meters"] = refused
        return [m for m in requested if m not in held]

    def _contract_of_meter(self, meter: str, event_id: str | None) -> dict[str, Any] | None:
        """The ACTIVE contract for this event that the meter is enrolled in (at most one, see _enroll). Else the settled one, since a reading
        that comes after the settlement is still owed for."""
        cs = [c for c in self.contracts.values() if c.get("status") in ("ACTIVE", "COMPLETE")
              and event_id in (c.get("need", {}).get("event_id"), c.get("community_event"))
              and meter in c.get("meters", [])]
        return next((c for c in cs if c["status"] == "ACTIVE"), cs[-1] if cs else None)

    @staticmethod
    def _pool_terms(ct: dict[str, Any], c: dict[str, Any]) -> None:
        """The coordinator's terms when two buyers fill from one pool: this buyer's share of each half hour on the same meters, the
        pool's capacity (a call pays only for what goes beyond all of it), the sale's id on the readings, and the deposit it holds."""
        if "share" not in c:
            return
        ct["share"] = {int(k): float(v) for k, v in c["share"].items()}
        ct["pool_offered"] = {int(k): float(v) for k, v in (c.get("poolCapacity") or {}).items()}
        ct["community_event"] = c.get("community_event")
        ct["deposit_held_usd"] = float(c.get("deposit_held_usd") or 0.0)

    def _repeat(self, ct: dict[str, Any], sender: str, txn: str, action: str, asked: str | None = None) -> str:
        """A select / init / confirm / update for a contract that has moved past it: NANDA Town delivered it again, or
        the aggregator is retrying because a reply was lost. Nothing changes; an open contract is answered as it
        stands (so a retrying aggregator can carry on), a settled or superseded one gets no answer."""
        status = ct.get("status")
        if status in ("DRAFT", "ACTIVE") and action != "on_status":
            self.reply(sender, action, txn, {"contract": ct["contract"]}, "contract", asked=asked)
            return f"{action} {status} (repeat, nothing changed)"
        if status == "COMPLETE":
            return "already settled"
        return beckn.Refused(f"contract {str(status).lower()}: nothing to do")

    def handle(self, kind: str, sender: str, body: dict[str, Any]) -> str:
        late = beckn.expired(body, default_ttl=float(self.cfg.get("max_message_age_s", 900)))
        if late:                                            # waited in the inbox past its validity: drop it
            self.event("expired_skipped", sender=sender, message_kind=kind, late_s=round(late))
            return beckn.Refused("expired: not acted on")
        sent = ((body or {}).get("context") or {}).get("sentWall") if isinstance(body, dict) else None
        if isinstance(sent, (int, float)) and sent < self.run_wall:
            # a run before this one (a demo reset since): its question about another day must not be answered into
            # this run's record (live 1 Nov 2026: "Sat 4 Jul: no event that day" at the start of the next run)
            self.event("stale_skipped", sender=sender, message_kind=kind)
            return beckn.Refused("sent before this run started: not acted on")
        if beckn.action_of(kind) == "on_status" and "batch" in (body.get("message") or {}):
            return beckn.of_batch([self._one_of_batch(kind, sender, b) for b in body["message"]["batch"]])
        if beckn.action_of(kind) == "on_status" and "performance" in (body.get("message") or {}):
            return self.meter_reading(sender, body)
        if sender not in self.aggregators:                # only enrolled aggregators discover and contract
            self.event("sender_refused", sender=sender, message_kind=kind)
            return beckn.Refused("rejected: not an enrolled aggregator")
        signer = body.get("signer", "")
        self.pins.setdefault(sender, signer)
        if not beckn.check(body, self.pins[sender]):
            return beckn.Refused("rejected: bad signature")
        act = beckn.action_of(kind)
        txn = body["context"]["transactionId"]
        m = body["message"]
        asked = ids.asked(body)
        rt = self.watch.handle(act, sender, body)
        if rt is not None:
            return rt
        if act == "discover":
            date = m["intent"]["date"]
            res, offs = [], []
            for r, o in self._pairs():
                n = beckn.need_from_catalog(r, o)
                notice = beckn.add_minutes(f"{beckn.add_minutes(n['window_start'], -1440)[:10]} 10:00", 0)
                if n["window_start"][:10] == date and self.now() >= notice:
                    res.append(r); offs.append(o)
            cat = {**{k: v for k, v in self.catalog.items() if k not in ("resources", "offers")},
                   "resources": res, "offers": offs}
            self.reply(sender, "on_discover", txn, {"catalogs": [cat] if res else []}, "catalogs", asked=asked)
            self.event("on_discover", date=date, needs=[r["id"] for r in res])
            return f"{len(res)} needs for {date}"
        if act == "select":
            prev = self.contracts.get(txn)
            if prev and prev.get("coordinator", sender) != sender:
                return beckn.Refused("rejected: this transaction belongs to another aggregator")
            if prev and prev.get("status") in ("DRAFT", "ACTIVE", "COMPLETE", "SUPERSEDED"):
                return self._repeat(prev, sender, txn, "on_select", asked)    # delivered again: the contract stays as it is
            ct = self.contracts.setdefault(txn, {"coordinator": sender, "status": "NEW"})
            rid = m["contract"]["commitments"][0]["resources"][0]["id"]
            r, o = next((r, o) for r, o in self._pairs() if r["id"] == rid)
            for t, other in self.contracts.items():       # one open contract per aggregator and event: a new
                if t != txn and other.get("coordinator") == sender and other.get("status") in ("DRAFT", "ACTIVE") \
                        and other.get("need", {}).get("event_id") == rid:   # transaction replaces the old one
                    other["status"] = "SUPERSEDED"
                    self.event("contract_superseded", event_id=rid, transaction=t)
            ct.update({"need": beckn.need_from_catalog(r, o), "resource": r, "offer": o, "status": "DRAFT"})
            contract = {"id": f"contract-{rid}", "status": "DRAFT", "event_id": rid,
                        "contractAttributes": {**o["offerAttributes"]["contractAttributes"],
                                               "roles": [{"role": "buyer", "participantId": "utility-or-engie-mock"},
                                                         {"role": "seller", "participantId": None}]}}
            ct["contract"] = contract
            self.reply(sender, "on_select", txn, {"contract": contract}, "contract", asked=asked)
            return "on_select DRAFT"
        ct = self.contracts.get(txn)
        if not ct or "need" not in ct:
            return beckn.Refused("unknown transaction")
        if ct.get("coordinator") != sender:                # follow-ups only from the aggregator that selected
            self.event("sender_refused", sender=sender, message_kind=kind, transaction=txn)
            return beckn.Refused("rejected: not this contract's aggregator")
        if ct.get("status") == "SUPERSEDED":
            return beckn.Refused("contract superseded by a newer transaction")
        if act == "init":
            if ct.get("status") != "DRAFT":               # init again after confirm (redelivery or the aggregator's
                return self._repeat(ct, sender, txn, "on_init", asked)    # retry): the confirmed meters and capacity stay
            c = m["contract"]
            ct["contract"]["contractAttributes"]["roles"][1]["participantId"] = c.get("seller")
            ct.update({"meters": self._enroll(ct, c.get("participatingMeters", [])),
                       "offered": {int(k): float(v) for k, v in c.get("capacityOffered", {}).items()}})
            self._pool_terms(ct, c)
            self.reply(sender, "on_init", txn, {"contract": ct["contract"]}, "contract", asked=asked)
            return "on_init"
        if act == "confirm":
            if ct.get("status") == "COMPLETE":
                return self._repeat(ct, sender, txn, "on_confirm", asked)  # a settled contract is never reopened
            ct["contract"]["status"] = ct["status"] = "ACTIVE"
            self.reply(sender, "on_confirm", txn, {"contract": ct["contract"]}, "contract", asked=asked)
            self.event("contract_active", event_id=ct["need"]["event_id"], meters=len(ct.get("meters", [])),
                       offered=ct.get("offered"))
            return "on_confirm ACTIVE"
        if act == "update" and "telemetry_done" in m:
            if ct.get("status") == "COMPLETE":
                if ct.get("settled_msg"):                 # the coordinator is still waiting for it, or this is a repeat, which it ignores
                    self.reply(sender, "on_status", txn, ct["settled_msg"], "settled", ct.get("top_ups", 0), asked=asked)
                return "already settled"
            ct["settle_to"] = (sender, txn)
            ct["called_slots"] = m["telemetry_done"].get("called_slots") or []
            ct["abnormal"] = {h: [int(x) for x in v] for h, v in (m["telemetry_done"].get("abnormal") or {}).items()}
            ct["owed"] = mc.owed_of(m["telemetry_done"].get("owed"))
            if self._readings_complete(ct):
                return self.settle(ct)
            if not ct.get("settle_requested"):
                ct["settle_requested"] = True             # settle anyway once the wait is over (replay clock, see tick)
                ct["settle_by"] = beckn.add_minutes(self.now(), float(self.cfg.get("utility_settle_wait_minutes", 30)))
                ct["settle_requested_wall"] = time.time()
                self.reply(sender, "on_status", txn, {"readings_lacking": self._lacking(ct)}, "lacking", asked=asked)   # only this buyer knows what reached it
            return "waiting for the last meter readings"
        if act == "update":
            if ct.get("status") == "COMPLETE":
                return self._repeat(ct, sender, txn, "on_status", asked)
            c = m["contract"]
            ct["meters"] = self._enroll(ct, c.get("participatingMeters", ct.get("meters", [])))
            ct["offered"] = {int(k): float(v) for k, v in c.get("capacityOffered", {}).items()} or ct.get("offered", {})
            self._pool_terms(ct, c)
            if c.get("meterSlots"):                       # the slots each meter committed to (it reports only those)
                ct["meter_slots"] = {h: [int(x) for x in v] for h, v in c["meterSlots"].items()}
            ct["baselines"] = self.baselines(ct, ct["meters"])
            self.reply(sender, "on_status", txn, {"contract": {"id": ct["contract"]["id"]},
                                                  "performance": {"kind": "baselines", "methodology":
                                                                  st.baseline_rule() if not ct.get("baseline_models") else
                                                                  "weather-normalized: per meter and slot, usage against the day's "
                                                                  "highest temperature over the 30 previous non-holiday, non-event "
                                                                  "days, settled at the event day's measured highest temperature "
                                                                  f"(the values below, {st.baseline_rule()}, are for reference)",
                                                                  "models": ct.get("baseline_models") or None,
                                                                  "meters": {k: {s: v for s, v in d.items() if s != "_days"}
                                                                             for k, d in ct["baselines"].items()}}}, "baselines", asked=asked)
            self.event("baselines_sent", event_id=ct["need"]["event_id"], meters=len(ct["baselines"]))
            return f"baselines for {len(ct['baselines'])} meters"
        return beckn.Refused(f"ignored {kind}")

    # ------------------------------------------------------------------ meter data and settlement
    def _one_of_batch(self, kind: str, sender: str, body: dict[str, Any]) -> str:
        """A reading can set off the settlement. Whatever goes wrong there, the readings behind it in the envelope still count."""
        try:
            out = self.handle(kind, sender, body)
            return out if isinstance(out, str) else str(out)
        except Exception as exc:                            # noqa: BLE001
            self.event("reading_error", home=sender, error=str(exc)[:200])
            log.exception("a reading from %s", sender)
            return beckn.Refused(f"error: {str(exc)[:80]}")

    def meter_reading(self, sender: str, body: dict[str, Any]) -> str:
        """A home's own signed reading for one slot (the utility's meter data). Checked against the key the utility
        pinned for that meter on first contact; a reading changed on the way is counted and thrown out."""
        r = body["message"]["performance"]
        self.traffic_homes["on_status"] += 1                 # the monitor counts messages; homes -> utility only here
        if not isinstance(r, dict) or not isinstance(r.get("event_id"), str):
            self.event("reading_unmatched", home=sender, event_id=None)
            return beckn.Refused("reading ignored: it names no event")
        with self.lock:
            if r.get("event_id") in self.watch.calls:        # a half hour of a call the home was dispatched for
                ok = self.watch.reading(sender, body)
                wire.note("recv", sender, self.name, "on_status", body, ok=not isinstance(ok, beckn.Refused), sim=self.now())
                return ok
            ct = self._contract_of_meter(sender, r.get("event_id"))   # the ACTIVE contract this meter is enrolled in
            if not ct:
                self.event("reading_unmatched", home=sender, event_id=r.get("event_id"))
                return beckn.Refused("reading ignored: this meter is in no active contract for that event")
            pin = self.pins.setdefault(sender, body.get("signer", ""))
            ok = beckn.check(body, pin) and r.get("home") == sender
            wire.note("recv", sender, self.name, "on_status", body, ok=ok, sim=self.now())
            if not ok:
                ct["rejected_readings"] = ct.get("rejected_readings", 0) + 1
                self.event("reading_rejected", home=sender, event_id=r.get("event_id"))
                return beckn.Refused("rejected: bad signature")
            why = mc.buyer_refuses(ct, sender, r, self.event)   # what can't be true never reaches the money, and it's counted
            if why:
                return why
            pre = mc.pre_kw(ct, sender, r, self.event)
            if ct.get("status") == "COMPLETE" and int(r["slot_id"]) not in ct.get("meter_readings", {}).get(sender, {}):
                ct["late_readings"], ct["late_wall"] = ct.get("late_readings", 0) + 1, time.time()
            ct.setdefault("meter_readings", {}).setdefault(sender, {}).setdefault(int(r["slot_id"]), float(r["usage_kw"]))     # the first of a slot stands: small repeats can't walk it anywhere
            ct.setdefault("meter_committed", {}).setdefault(sender, {})[int(r["slot_id"])] = float(r.get("committed_kw") or 0.0)
            ct.setdefault("meter_batt", {}).setdefault(sender, {}).setdefault(int(r["slot_id"]), float(r.get("battery_kw") or 0.0))
            if pre is not None:
                ct.setdefault("meter_pre", {})[sender] = pre
            ct["last_reading_wall"] = time.time()
            if ct.get("settle_requested") and ct.get("status") != "COMPLETE" and self._readings_complete(ct):
                return self.settle(ct)
            return f"reading {sender} slot {r['slot_id']}"

    def _lacking(self, ct: dict[str, Any]) -> dict[str, list[int]]:
        """The slots each participating meter committed to that this buyer has no reading for (a home with no commitment in a
        slot sends no reading for it). Without the coordinator's per-meter slots: every slot of the window."""
        got = ct.get("meter_readings", {})
        every = [int(s["id"]) for s in ct["need"]["slots"]]
        slots = ct.get("meter_slots") or {}
        out = {m: sorted(set(slots.get(m, every)) - set(got.get(m, {})) - set((ct.get("rejected_slots") or {}).get(m, {}))) for m in ct.get("meters", [])}   # a reading refused came: nobody waits for another
        return {m: v for m, v in out.items() if v}

    def _readings_complete(self, ct: dict[str, Any]) -> bool:
        return not self._lacking(ct)

    def tick(self) -> None:
        """Settle a contract whose readings are still incomplete once its wait (replay minutes) is over and no reading
        has come for `utility_settle_quiet_s` real seconds: readings still on their way through a busy Town, or waiting
        in the inbox of a machine that was asleep (live, 29 Oct 2026), are counted before the money is split, however
        fast the replay clock runs or however far it jumped. 30 s for the live service (main below), 0 in-process
        (the tests and the simulations deliver every message at once, on a virtual clock)."""
        with self.lock:
            while self.unsent and self._post(*self.unsent[0]):
                self.unsent.popleft()
            try:
                self.watch.tick()
            except Exception as exc:                      # noqa: BLE001  the sale's settlement below doesn't wait on the calls
                self.event("tick_error", error=str(exc)[:200])
            now = self.now()
            quiet = float(self.cfg.get("utility_settle_quiet_s", 0.0))
            for ct in self.contracts.values():
                last = max(ct.get("last_reading_wall", 0.0), ct.get("settle_requested_wall", 0.0))
                try:                                      # one contract that can't be settled doesn't hold up the next
                    if ct.get("settle_requested") and ct.get("status") == "ACTIVE" and now >= ct.get("settle_by", now) \
                            and time.time() - last >= quiet:
                        self.event("settle_late", event_id=ct["need"]["event_id"],
                                   readings=sum(len(v) for v in ct.get("meter_readings", {}).values()))
                        self.settle(ct)
                    if ct.get("status") == "COMPLETE" and ct.get("late_readings") and time.time() - ct.get("late_wall", 0.0) >= quiet:
                        self.settle(ct, top_up=True)      # readings came after the settlement: the sellers are owed for them
                except Exception as exc:                  # noqa: BLE001
                    self.event("settle_error", event_id=(ct.get("need") or {}).get("event_id"), error=str(exc)[:200])
                    log.exception("settling a contract")

    def tick_loop(self) -> None:
        while True:
            try:
                self.tick()
            except Exception as exc:
                self.event("tick_error", error=str(exc)[:200])
            time.sleep(2)

    def settle(self, ct: dict[str, Any], top_up: bool = False) -> str:
        """DEG demand-flex settlement on the participating meters' own readings against their baselines. `top_up`: settled again on readings
        that came in after the first settlement. The whole result goes again, and the coordinator pays what it adds."""
        sender, txn = ct["settle_to"]
        prior = float((ct.get("settlement") or {}).get("total") or 0.0)
        need = ct["need"]
        meters, notes = mc.counted(ct, self.settlement_baselines(ct))
        # a building node's meter counts at most 1.2x its commitment in each half hour (the coordinator's word for what it
        # committed): its baseline picks days without its own routine evening cut (settlement.building_counted_kwh)
        caps = {m: {int(k): st.BUILDING_CAP_X * mc.committed(ct, m, k) for k in ct.get("meter_committed", {}).get(m, {})}
                for m in meters if (self.fleet.get(m) or {}).get("kind") == "building"}
        ct["capped"] = self._called_slots(ct) if st.MARKET else {}
        result = st.deg_settle(need["slots"], ct.get("offered", {}), meters, hours=beckn.SLOT_MINUTES / 60,
                               currency=need["currency"], caps=caps or None, share=ct.get("share"), capped=ct["capped"])
        if ct["capped"]:
            result["capped_kw"] = {str(k): v for k, v in ct["capped"].items()}
        if ct.get("deposit_held_usd"):
            result["deposit"] = st.deposit_statement(result["total"], ct["deposit_held_usd"])
        if caps:
            result["caps_kw"] = {m: {str(k): round(v, 6) for k, v in c.items()} for m, c in caps.items()}   # the split reads these
            result["cap_rule"] = f"a building's meter counts at most {st.BUILDING_CAP_X:g} x its commitment per half hour"
        refused = {h: {str(k): v for k, v in d.items()} for h, d in (ct.get("rejected_slots") or {}).items() if d}
        checks = {**notes, **({"refused": refused} if refused else {}), **({"pre_refused": ct["pre_refused"]} if ct.get("pre_refused") else {})}
        if checks:                                        # never silent: on the settlement, on the record and in the log
            result["meter_checks"] = checks
            self.event("readings_checked", event_id=need["event_id"], **{k: (len(v) if hasattr(v, "__len__") else v) for k, v in checks.items()})
        ct["settlement"] = result
        ct["status"] = "COMPLETE"
        ct["late_readings"] = 0
        ct["top_ups"] = ct.get("top_ups", 0) + (1 if top_up else 0)
        bad = ct.get("rejected_readings", 0)
        actuals = {h: {str(k): v for k, v in d.items()} for h, d in meters.items()}
        self.reply(sender, "on_status", txn, {"contract": {"id": ct["contract"]["id"]},
                                              "performance": {"kind": "actuals", "meters": actuals}}, "actuals", ct["top_ups"])
        # the actuals travel inside the settlement too, so the split never depends on which message arrives first
        ct["settled_msg"] = {"contract": {"id": ct["contract"]["id"], "status": "COMPLETE"},
                             "settlement": {**result, "currency": need["currency"], "rejected_readings": bad, "actuals": actuals, "lacking": self._lacking(ct),
                                            **({"top_up": ct["top_ups"], "prior_total": prior} if top_up else {})}}
        self.reply(sender, "on_status", txn, ct["settled_msg"], "settled", ct["top_ups"])
        self.event("settle_top_up" if top_up else "settled", event_id=need["event_id"], seller_usd=result["total"], prior_usd=prior, meters=len(meters), bad=bad)
        self.watch.settle_again_for(ct)                   # a call settled first was measured against fewer of these readings
        return f"settled {result['total']} {need['currency']}"

    def _called_slots(self, ct: dict[str, Any]) -> dict[int, float]:
        """Half hours a call covers, with the capacity sold in them: the booking stops there and the call pays the rest, so nothing is
        paid twice. A call from this buyer is known here. Another buyer's comes from the coordinator (update, telemetry_done)."""
        sold = ct.get("pool_offered") or ct.get("offered") or {}
        out = {int(k): float(sold.get(int(k), 0.0)) for k in ct.get("called_slots") or []}
        for s in ct["need"]["slots"]:
            end = beckn.add_minutes(s["start"], beckn.SLOT_MINUTES)
            if any(c["start"] < end and s["start"] < c["end"] and c.get("status") != "REFUSED" and c.get("confirmed_kw") is not None for c in self.watch.calls.values()):   # a call the desk never heard of cuts nothing
                out[int(s["id"])] = float(sold.get(int(s["id"]), 0.0))
        return out

    def inbox_loop(self) -> None:
        """The Town inbox, watched with short checks (town.InboxPoller: no held long-poll, which costs NANDA Town 0.2
        about 20 database scans a second): every `town_poll_seconds` (1 s) while messages flow, every
        `town_poll_idle_seconds` (3 s) after a quiet minute. A pass that finds a message brings in
        `utility_inbox_workers` claimers: each claim and each ack is a round trip to Town; live on 2026-10-29 a single
        claimer handled about 0.2 messages a second, so the 400 meter readings of one event took over half an hour."""
        self.poller = InboxPoller(self.town, self._handle_claimed, every=float(self.cfg.get("town_poll_seconds", 1.0)),
                                  idle_every=float(self.cfg.get("town_poll_idle_seconds", 3.0)),
                                  claimers=max(1, int(self.cfg.get("utility_inbox_workers", 12))),
                                  on_error=lambda where, exc: self.event("town_error", error=f"{where}: {str(exc)[:200]}"))
        self.poller.run()

    def _handle_claimed(self, msg: dict[str, Any]) -> None:
        self.lag.seen(msg)                                # how far behind this inbox is (GET /traffic, /health)
        with self.lock:
            try:
                note = self.handle(msg.get("kind", ""), msg.get("from"), msg.get("body") or {})
            except Exception as exc:
                note = beckn.Refused(f"error: {exc}")
                log.exception("handle")
        if self.keeper is not None and not keep.from_a_node(msg):
            self.keeper.save()                            # on disk before Town hears it: nobody sends a contract step twice
        self.town.ack(msg["message_id"], msg["fence"], *beckn.ack_of(note, 300))


def build_app(u: Utility):
    from fastapi import Body, FastAPI, HTTPException

    app = FastAPI(title="flexgrid-utility", version="0.2.0", dependencies=[doors.door()])

    @app.get("/health")
    def health():
        p = getattr(u, "poller", None)
        lag = getattr(u, "lag", None)
        return {"ok": True, "agent": u.name, "pubkey": u.pub, "now": u.now(), "contracts": len(u.contracts),
                "label": "Utility or ENGIE (mock, public data)", "town_calls": p.stats() if p else None,
                "inbox_lag_s": round(lag.value(), 1) if lag else 0.0, "doors": doors.state(),
                "town_ms": u.town.timing() if getattr(u, "town", None) is not None and hasattr(u.town, "timing") else None, **keep.health(u.keeper)}

    @app.get("/contracts")
    def contracts():
        return {t: {**{k: v for k, v in c.items() if k in ("status", "meters", "offered", "settlement", "buyer_review", "rejected_readings", "rejected_slots", "pre_refused")},
                    "event_id": c.get("need", {}).get("event_id")}
                for t, c in u.contracts.items()}

    def _fail(exc: ValueError):
        code, _, msg = str(exc).partition(": ")
        raise HTTPException(int(code) if code in ("400", "404", "409") else 400, msg or str(exc))

    @app.get("/needs")
    def needs():
        with u.lock:
            return {"now": u.now(), "needs": u.needs_list(), "orders": u.orders}

    @app.post("/needs")
    def post_need(body: dict = Body(...)):
        with u.lock:
            try:
                return u.place_order(body)
            except ValueError as exc:
                _fail(exc)

    @app.get("/forecast/{day}")
    def forecast(day: str):
        """The community's demand tomorrow by hour (10/50/90 %), from the meters' own history and the issued weather forecast:
        what the buyer's desk reads before it orders (MCP tool forecast_community_demand)."""
        from . import community_forecast
        try:
            issue = min(f"{beckn.add_minutes(f'{day} 00:00', -1440)[:10]} 10:00", u.now())   # never ahead of the clock
            return community_forecast.forecast(day, issued_at=issue)
        except (ValueError, OSError) as exc:
            raise HTTPException(404, str(exc)[:300])

    @app.get("/evidence/{event_id}")
    def evidence(event_id: str):
        with u.lock:
            try:
                return u.evidence(event_id)
            except KeyError:
                raise HTTPException(404, f"no contract for {event_id}")

    @app.post("/reviews/{event_id}")
    def review(event_id: str, body: dict = Body(...)):
        with u.lock:
            try:
                return u.review(event_id, bool(body.get("approve")), body.get("reason", ""), body.get("checks"),
                                body.get("by", "buyer"))
            except ValueError as exc:
                _fail(exc)

    @app.get("/events")
    def events(n: int = 50):
        return list(u.events)[-n:]

    @app.get("/traffic")
    def traffic():
        lag = getattr(u, "lag", None)                  # the coordinator's clock waits while this inbox is behind
        return {"received_from_homes": dict(u.traffic_homes), "inbox_lag_s": round(lag.value(), 1) if lag else 0.0}

    @app.get("/calls")
    def calls():
        """Short-notice and emergency calls during the day (rt_buyer.py): what was asked, confirmed, measured, paid."""
        with u.lock:
            return {"now": u.now(), "calls": u.watch.summary(), "alerts": u.watch.alerts}

    @app.post("/alert")
    def alert(body: dict = Body(...)):
        """The grid operator tells the buyer of an emergency (the audience's control, or a replayed Power Caution):
        {"kw": extra kW wanted, "minutes": how long, "why": ...}. The buyer calls the community at once."""
        try:
            if body.get("kw") is None and body.get("pct") is not None:      # "10-100 % more than the program"
                body = {**body, "kw": u.watch.program(u.now()[:10])[0] * float(body["pct"]) / 100.0}
            kw, minutes = float(body.get("kw")), float(body.get("minutes", 60))
        except (TypeError, ValueError):
            raise HTTPException(400, "kw (or pct) and minutes must be numbers")
        if not (0 < kw <= 2000 and 15 <= minutes <= 240):
            raise HTTPException(400, "kw must be in (0, 2000] and minutes in [15, 240]")
        with u.lock:                                     # "at": a step a test runner names, for an alert told at that step
            a = u.watch.alert(kw, minutes, str(body.get("why") or "grid emergency")[:120], at=body.get("at") or None)
            if a["at"] <= u.now():
                u.watch.tick()
            return a

    @app.post("/demo/price")
    def demo_price(body: dict = Body(default={})):
        """What if the real-time price spikes: {"factor": 2..4, "hours": 1} from now on ({"factor": 1}: off). Only the
        buyer's view of the price changes; its grid watch decides whether to call the community."""
        from . import replay as replay_mod
        f = float(body.get("factor") or 1.0)
        if not 1.0 <= f <= 10.0:
            raise HTTPException(400, "factor must be in [1, 10]")
        now = u.now()
        to = beckn.add_minutes(now, 60 * max(0.25, min(4.0, float(body.get("hours") or 1.0))))
        w = replay_mod.set_price(f if f > 1.0 else None, now, to)
        u.event("what_if_price", factor=f, start=now, end=to)
        with u.lock:
            u.watch.tick()
        return {"now": now, "price": {k: w.get(k) for k in ("price", "price_from", "price_to")}}

    @app.post("/calls/{call_id}/end")
    def end_call(call_id: str):
        with u.lock:
            if call_id not in u.watch.calls:
                raise HTTPException(404, f"no call {call_id}")
            return u.watch.end_call(call_id)

    @app.post("/demo/reset")
    def demo_reset(body: dict = Body(default={})):
        """A new run. A buyer agent's order placed before the run is kept, unless clear_orders (the Mac's own buyer
        agent is about to order again for the same day)."""
        with u.lock:
            u.contracts.clear()
            u.unsent.clear()
            u.traffic_homes.clear()
            u.run_wall = time.time()
            u.play = ids.new_play()
            from .rt_buyer import GridWatch
            from . import replay as replay_mod
            u.watch, u.watch.price_rule = GridWatch(u), u.watch.price_rule   # a stand-in buyer stays one that calls only on an alert
            replay_mod.set_price(None)
            u.orders = {} if (body or {}).get("clear_orders") else \
                {k: v for k, v in u.orders.items() if not str(v.get("buyer", "")).startswith("Control Room")}
        return {"ok": True}

    @app.get("/demo/clock")
    def demo_clock_get():
        return u.clock.clock()

    @app.post("/demo/clock")
    def demo_clock(body: dict = Body(default={})):
        u.clock.configure(body.get("start", u.clock.start), float(body.get("step_seconds", u.clock.step_seconds)),
                          int(body.get("loop_days", u.clock.loop_days)), body.get("at"), body.get("at_wall"), hold=bool(body.get("hold")))
        return u.clock.clock()

    return app


def main() -> None:
    ap = argparse.ArgumentParser(prog="flexgrid-utility")
    ap.add_argument("cmd", choices=["run", "pubkey"])
    a = ap.parse_args()
    if a.cmd == "pubkey":
        print(signing.public_hex(signing.load_or_create(config.CONFIG_DIR / "secrets" / "utility.key")))
        return
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    u = Utility()
    u.cfg.setdefault("utility_settle_quiet_s", 30.0)      # live: wait for a quiet inbox before settling (see tick)
    logs = __import__("pathlib").Path(__file__).resolve().parents[1] / "logs"
    logs.mkdir(exist_ok=True)
    wire.open_log(str(logs / "town_wire_utility.jsonl"))       # every reading a home sends the utility (live_talk)
    keeper = keep.for_buyer(u)                                 # a contract that was running carries on
    u.town.join_until_up(then=lambda: threading.Thread(target=u.inbox_loop, daemon=True).start(),
                         log=lambda m: (print(m, flush=True), u.event("town_error", error=m)))
    threading.Thread(target=u.tick_loop, daemon=True).start()
    try:
        uvicorn.run(build_app(u), host="127.0.0.1", port=int(u.cfg.get("utility_port", 7873)), log_level="warning")
    finally:
        keeper.close()


if __name__ == "__main__":
    main()
