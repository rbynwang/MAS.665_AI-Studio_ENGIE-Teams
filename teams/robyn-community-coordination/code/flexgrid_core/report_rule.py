"""When a node speaks to the coordinator, on the new path. One coordinator can't hear thousands of nodes every few minutes.

The rule: a node sends nothing while things stay inside expected limits. A step is inside limits when the plan's shortfall at every step still
owed is under LIMIT_SHARE of what is owed there, the part the coordinator's margin already covers. A node speaks when:
  * it first leaves those limits
  * it gets worse by WORSE_BY of what it last reported
  * it is back inside them
A cause someone chose or a fault goes at once. Forecast drift waits DRIFT_EVERY_MIN between reports.
Each report covers the rest of the window, so a step sliding into view is no news.

Readings aren't exceptions, so they don't go out one half hour at a time: Outbox holds a node's signed readings and sends them in one envelope
after the window. Each reading stays signed on its own, so the buyer's count and the independent check read them as before.

The coordinator's side is in rt_market.py: it asks only the members with room, it warns a buyer only when the shortfall changes, and it
sends no event notice, because the contract already says when to prepare.
"""
from __future__ import annotations

from typing import Any, Callable

from . import beckn, ids
from .home_v2 import HomeV2
from .market import PARTS, firm_part
from .node import loans
from .replay import Replay

LIMIT_SHARE = 0.10        # of what is owed at a step: about what a 10 % margin covers
WORSE_BY = 0.5            # a new report needs the shortfall this much above the last one
DRIFT_EVERY_MIN = 60      # forecast drift, at most this often
HOLD_MIN = 45             # readings go out this long after the last one came in: longer than a half hour, so a window goes as one
MAX_HOLD_MIN = 225        # and none waits longer than this. A call running past the window keeps readings coming, and the market stops waiting 2 h after it
                          # ends: a 3-hour window still goes as one, and the rest of a late call follows on its own


class Outbox:
    """What a node sends, with its readings held back for one envelope."""

    def __init__(self, name: str, key: Any, send: Callable, hold_min: float = HOLD_MIN):
        self.name, self.key, self.send, self.hold_min = name, key, send, hold_min
        self.held: dict[str, list[dict[str, Any]]] = {}
        self.first: str | None = None
        self.last: str | None = None
        self.now: Callable[[], str] = lambda: ""
        self.play: Callable[[], str] = ids.new_play      # QuietNode gives it the node's

    def __call__(self, to: str, kind: str, body: dict[str, Any], mid: str) -> Any:
        if beckn.action_of(kind) == "on_status" and "performance" in (body.get("message") or {}):
            if not self.held:
                self.first = self.now()
            self.held.setdefault(to, []).append(body)
            self.last = self.now()
            return True
        return self.send(to, kind, body, mid)

    def tick(self, now: str) -> None:
        if not self.held or (Replay.minutes_between(self.last or now, now) < self.hold_min
                             and Replay.minutes_between(self.first or now, now) < MAX_HOLD_MIN):
            return
        self.flush(now)

    def flush(self, now: str, asked: str | None = None) -> None:
        """Everything held goes now, one envelope per receiver. Its id names the play and the readings in it, not the moment it went: a day played again has a new play, so its
        batches go (live, 6 Nov 2026: the ids repeated, Town refused them, and 192 homes and 3 buildings went unpaid), and a node that restarts with these readings still held
        names the same batch. `asked`: the chase this batch answers, so a chase's batch is delivered even when it's the same readings."""
        for to, bodies in self.held.items():
            env = beckn.make("on_status", sender=self.name, receiver=to, transaction_id=f"readings-{self.name}",
                             message={"batch": bodies}, key=self.key, sender_role="bpp", ttl_s=beckn.TTL_EVIDENCE_S)
            self.send(to, beckn.kind("on_status"), env, ids.sealed(None if asked else self.key, asked or self.play(), self.name, to, "on_status", "readings", ids.readings(bodies)))
        self.held = {}


def offer_nothing(offer: dict[str, Any]) -> dict[str, Any]:
    """What a node an alarm holds offers: no kW, and an evening with no rebound and no car to restart. Its devices have gone back to their own day, so the offer says so, and the
    pool and the guard count nothing it can't give. Nothing held back either, so the second round doesn't ask it."""
    from .brain import scale_offer
    out = scale_offer(offer, 0.0, False, False)
    out["slots"] = [{**x, **{k: 0.0 for k in PARTS + ("firm_kw", "tentative_kw") if k in x}} for x in out["slots"]]
    out["energy_kwh"] = 0.0
    out.pop("held_back_kw", None)
    return out


class PoolNode(HomeV2):
    """A node on the new path: every offer says what's firm and what's tentative. Its own estimate where it has one (node/sellable.py)."""

    def reset(self) -> None:
        super().reset()
        loans.forget(self)

    def build_offer(self, need: dict[str, Any], fc: dict[str, Any]) -> dict[str, Any]:
        return self.tag_offer(super().build_offer(need, fc))

    def alarm_holds(self) -> bool:
        """An alarm holds this node, until a person's signed word ends it."""
        ov = getattr(self.adapter, "overrides", None) or {}
        return bool(ov.get("alarm")) or "alarm_hold" in ov

    def send_offer(self, txn: str, offer: dict[str, Any], asked: str | None = None) -> dict[str, Any]:
        """Every offer leaves through here: the first, one asked for again and a second round's. While an alarm holds the node it offers nothing. On a dry day three held nodes offered,
        were sold and accepted, delivered nothing and were paid nothing, and the sellers came in $52.41 short of the honest day: their share could have gone to the others."""
        if self.alarm_holds():
            offer = self.offers[offer["event_id"]] = offer_nothing(offer)
            self.log("v2_offer_held", event_id=offer["event_id"], why="an alarm holds this node")
        return super().send_offer(txn, offer, asked)

    def verify_contract(self, sender: str, body: dict[str, Any]) -> dict[str, Any]:
        """An alarm that came after the offer: the node says no, and the coordinator gives its share to an offer it has spare (Market.top_up)."""
        v = super().verify_contract(sender, body)
        if v["decision"] == "ACCEPTED" and self.alarm_holds():
            return {"decision": "DECLINED", "reason": "an_alarm_holds_this_node", "contract_id": v.get("contract_id")}
        return v

    def tag_offer(self, offer: dict[str, Any]) -> dict[str, Any]:
        for x in offer["slots"]:
            f = firm_part(offer, x)
            x["firm_kw"], x["tentative_kw"] = round(f, 3), round(float(x.get("kw_offered") or 0.0) - f, 3)
        return offer

    def rt_answer(self, txn: str, ask: dict[str, Any], asked: str | None = None) -> str:
        """A node that has left the sale adds nothing: its plan has stopped, so a yes here would be a promise nobody keeps."""
        ov = self.adapter.overrides or {}
        if not (ov.get("alarm") or ov.get("stop") or ov.get("closed")):
            return super().rt_answer(txn, ask, asked)
        why = self._rt_reason()
        body = beckn.make("rt_answer", sender=self.name, receiver=self.coordinator, transaction_id=txn,
                          message={"answer": {"id": ask["id"], "kw": 0.0, "why": why}}, key=self.key, sender_role="bpp")
        self.send(self.coordinator, beckn.kind("rt_answer"), body, self.mid(self.coordinator, "rt_answer", ask["id"], asked=asked))
        self.log("rt_answer", call_id=ask["id"], kw=0.0, why=why)
        return f"can add nothing for {ask['id']}: {why}"

    def _rt_reason(self) -> str:
        return "an alarm is on" if (self.adapter.overrides or {}).get("alarm") else super()._rt_reason()

    def tick(self) -> list[str]:
        """Life safety (SPEC section 6, step 1): on an alarm nothing more is sent and every limit releases. The sale's plan writes through the controller, not the
        agent's guard, so it has to stop here: the devices go back to their own day, and the node tells the desk at once. Only a person clears it."""
        ov = self.adapter.overrides or {}
        if (ov.get("alarm") or "alarm_hold" in ov) and not ov.get("stop"):
            self.adapter.rt_override("stop", True)
        if ov.get("note") and getattr(self, "agent", None) is None:
            loans.ask(self)                                 # rules can't read a note, so a node with no agent borrows the model for it
        return super().tick()


class QuietNode(PoolNode):
    """A node under this rule. Its contracts and readings are HomeV2's."""

    def __init__(self, name: str, cfg: dict[str, Any], adapter: Any, replay: Any, key: Any, send: Callable,
                 trusted: Callable[[], list[str]], log: Callable[..., None], coordinator: str = "coordinator"):
        self.outbox = Outbox(name, key, send)
        super().__init__(name, cfg, adapter, replay, key, self.outbox, trusted, log, coordinator)
        self.outbox.now = self.now_ts
        self.outbox.play = lambda: self.play

    def rt_report(self) -> None:
        ctl = getattr(self.adapter, "ctl", None)
        if ctl is None:
            return
        now, i = self.now_ts(), self.now_idx()
        if self.adapter.overrides.get("stop"):              # it left the event: everything still owed is short
            short = {}
            for k in range(1, 96):                          # all of it, however far off: the sooner the desk knows, the more it can do
                t = self.replay.ts(i + k)
                if ctl.target_at(t) > 0.1:
                    short[t] = round(ctl.target_at(t), 3)
        elif not ctl.last or not ctl.last.get("ok"):
            return
        else:
            short = {t: v for t, v in (ctl.last.get("short_by_ts") or {}).items()
                     if t > now and v >= max(self.REPORT_MIN_KW, LIMIT_SHARE * ctl.target_at(t))}
        prev = getattr(self, "_rt_reported", {})
        top = max(prev.values(), default=0.0)
        worse = any(v - prev.get(t, top) >= max(self.REPORT_MIN_KW, WORSE_BY * prev.get(t, top)) for t, v in short.items())
        back = bool(prev) and not short and any(t > now for t in prev)
        if not worse and not back:
            return
        why = self._rt_reason()
        at_once = why != "the forecast changed" and why != getattr(self, "_rt_why", None)
        last = getattr(self, "_rt_at", None)
        if not at_once and last and self.replay.minutes_between(last, now) < DRIFT_EVERY_MIN:
            return
        msg = {"short": {t: round(v, 3) for t, v in short.items()}, "reason": why}
        if (self.adapter.overrides or {}).get("closed"):
            msg["abnormal"] = True                          # its meter no longer measures a cut: the settlement must leave it out
        body = beckn.make("rt_capacity", sender=self.name, receiver=self.coordinator, transaction_id=f"cap-{self.name}",
                          message=msg, key=self.key, sender_role="bpp")
        self.send(self.coordinator, beckn.kind("rt_capacity"), body, self.mid(self.coordinator, "rt_capacity", now, ids.digest(msg)))
        self._rt_reported, self._rt_at, self._rt_why = short, now, why
        self.log("rt_capacity", short_kw=round(max(short.values(), default=0.0), 3), steps=len(short), reason=why)

    def _rt_reason(self) -> str:
        ov = self.adapter.overrides or {}
        if ov.get("closed") and not ov.get("alarm"):
            return "the building closed for the day"
        if ov.get("zone_event") and not ov.get("alarm"):
            return "an event was added in a sold zone"
        return super()._rt_reason()

    def tick(self) -> list[str]:
        out = super().tick()
        self.outbox.tick(self.now_ts())
        return out

    def handle(self, kind: str, sender: str, body: dict[str, Any]) -> str:
        if beckn.action_of(kind) == "status" and "readings_wanted" in (body.get("message") or {}):
            return self.gate(kind, sender, body) or self.send_again(body["message"]["readings_wanted"], ids.asked(body))
        return super().handle(kind, sender, body)

    def send_again(self, want: dict[str, Any], asked: str | None = None) -> str:
        """The coordinator or a buyer lacks these readings. Each is made again from this node's own meter rows, signed afresh and sent at once. Money follows
        the readings, so a seller must not go unpaid for a message nobody saw go missing. No contract id: a call's readings alone."""
        cid = want.get("contract_id")
        c = self.contracts.get(cid) or {}
        rows, now_i = getattr(self.adapter, "recent", {}), self.now_idx()
        sent, gone = [], []
        ok = c.get("status") == "ACCEPTED"
        extra = [x["id"] for x in c.get("commitments", []) if x.get("gap")] if ok else []    # a gap's half hour: this node knows it, the contract copies don't
        for sid in sorted(set(want.get("slots") or []) | set(extra)):
            s = next((x for x in c["commitments"] if x["id"] == sid), None) if ok else None
            i0 = self.replay.idx_of(s["start"]) if s else None
            if i0 is None or now_i <= i0 + 1 or i0 not in rows or i0 + 1 not in rows:
                gone.append(sid)
                continue
            self.send_reading(c.get("transaction_id", ""), self.build_reading(cid, sid))
            sent.append(sid)
        for k, rt in self.contracts.items():                # a call's own readings were in the same lost envelope
            if rt.get("rt") and rt.get("status") == "ACCEPTED":
                for s in rt["commitments"]:
                    i0 = self.replay.idx_of(s["start"])
                    if s["id"] in rt["reported"] and i0 in rows and i0 + 1 in rows:
                        self.send_reading(rt.get("transaction_id", ""), self.build_reading(k, s["id"]))
        self.outbox.flush(self.now_ts(), asked)
        self.log("v2_readings_sent_again", contract_id=cid, slots=sent, gone=gone)
        return f"sent again {sent}, gone {gone}"
