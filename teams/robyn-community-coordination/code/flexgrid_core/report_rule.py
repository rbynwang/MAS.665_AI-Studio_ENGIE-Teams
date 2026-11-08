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

import uuid
from typing import Any, Callable

from . import beckn
from .home_v2 import HomeV2
from .market import firm_part
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
        for to, bodies in self.held.items():
            env = beckn.make("on_status", sender=self.name, receiver=to, transaction_id=f"readings-{self.name}",
                             message={"batch": bodies}, key=self.key, sender_role="bpp", ttl_s=beckn.TTL_EVIDENCE_S)
            self.send(to, beckn.kind("on_status"), env, f"{self.name}:readings:{now}")
        self.held = {}


class PoolNode(HomeV2):
    """A node on the new path: every offer says what's firm and what's tentative. Its own estimate where it has one (node/sellable.py)."""

    def build_offer(self, need: dict[str, Any], fc: dict[str, Any]) -> dict[str, Any]:
        offer = super().build_offer(need, fc)
        for x in offer["slots"]:
            f = firm_part(offer, x)
            x["firm_kw"], x["tentative_kw"] = round(f, 3), round(float(x.get("kw_offered") or 0.0) - f, 3)
        return offer

    def rt_answer(self, txn: str, ask: dict[str, Any]) -> str:
        """A node that has left the sale adds nothing: its plan has stopped, so a yes here would be a promise nobody keeps."""
        ov = self.adapter.overrides or {}
        if not (ov.get("alarm") or ov.get("stop") or ov.get("closed")):
            return super().rt_answer(txn, ask)
        why = self._rt_reason()
        body = beckn.make("rt_answer", sender=self.name, receiver=self.coordinator, transaction_id=txn,
                          message={"answer": {"id": ask["id"], "kw": 0.0, "why": why}}, key=self.key, sender_role="bpp")
        self.send(self.coordinator, beckn.kind("rt_answer"), body, f"{self.name}:rt_answer:{ask['id']}:{uuid.uuid4().hex[:6]}")   # a second round asks under the same id
        self.log("rt_answer", call_id=ask["id"], kw=0.0, why=why)
        return f"can add nothing for {ask['id']}: {why}"

    def _rt_reason(self) -> str:
        return "an alarm is on" if (self.adapter.overrides or {}).get("alarm") else super()._rt_reason()

    def tick(self) -> list[str]:
        """Life safety (SPEC section 6, step 1): on an alarm nothing more is sent and every limit releases. The sale's plan writes through the controller, not the
        agent's guard, so it has to stop here: the devices go back to their own day, and the node tells the desk at once. Only a person clears it."""
        ov = self.adapter.overrides or {}
        if ov.get("alarm") and not ov.get("stop"):
            self.adapter.rt_override("stop", True)
        return super().tick()


class QuietNode(PoolNode):
    """A node under this rule. Its contracts and readings are HomeV2's."""

    def __init__(self, name: str, cfg: dict[str, Any], adapter: Any, replay: Any, key: Any, send: Callable,
                 trusted: Callable[[], list[str]], log: Callable[..., None], coordinator: str = "coordinator"):
        self.outbox = Outbox(name, key, send)
        super().__init__(name, cfg, adapter, replay, key, self.outbox, trusted, log, coordinator)
        self.outbox.now = self.now_ts

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
        self.send(self.coordinator, beckn.kind("rt_capacity"), body, f"{self.name}:rt_capacity:{now}")
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
