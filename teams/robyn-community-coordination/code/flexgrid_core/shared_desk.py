"""Benchmark only: one desk over the two markets. Homes and data centers run on one clock in one process.

The desk decides only from what a real coordinator has: each side's own real-time desk, fed by the signed shortfall reports nodes
send (rt_capacity). When a side's desk has tried its own nodes and a gap is still left, that gap goes to the other side's buyer as a
call, in the same step. It never reads the no-event twin (grid_cf_kw, import_cf_kw) or any truth field: rule R5.

Each buyer's settlement stays as it is: the utility pays for homes, the grid operator for data centers. A cross-served call goes
through that side's own real-time desk, so its normal settlement applies; the buyer who placed it pays."""
from __future__ import annotations
from typing import Any

MIN_GAP_KW = 1.0


class SharedDesk:
    def __init__(self, day: str, w0: str, w1: str, gen, ctx: dict[str, Any], cross: bool = True):
        self.day, self.w0, self.w1, self.gen, self.ctx, self.cross = day, w0, w1, gen, ctx, cross
        self.score_rows: list[dict[str, Any]] = []         # for the Judge only: the desk never reads these
        self.calls: list[dict[str, Any]] = []
        self.seen: set[str] = set()

    def left_over(self, market) -> list[dict[str, Any]]:
        """Gaps this side's own desk could not fill, from its own records."""
        out = []
        for rec in market.rt.calls.values():
            if rec.get("kind") == "gap" and rec["id"] not in self.seen and rec.get("phase") != "asking" and float(rec.get("short_kw") or 0.0) >= MIN_GAP_KW:
                out.append(rec)
        return out

    def step(self, now, homes, market, util, traffic, log) -> None:
        next(self.gen)                                     # the data centers take the same step
        self.score(now, homes)
        if not self.cross:
            return
        mins = lambda x: int(x[11:13]) * 60 + int(x[14:16])  # noqa: E731
        for side, m, other, watch in (("homes", market, "datacenters", self.ctx["op"].watch), ("datacenters", self.ctx["market"], "homes", util.watch)):
            for rec in self.left_over(m):
                self.seen.add(rec["id"])
                if rec["end"] <= now:
                    continue
                kw = float(rec["short_kw"])
                watch.alert(kw, max(15.0, float(mins(rec["end"]) - mins(now))), f"shared desk: {side} shortfall", at=now)
                self.calls.append({"at": now, "reported_at": rec.get("received_at"), "short_side": side, "asked_side": other, "kw": round(kw, 1), "gap_id": rec["id"], "why": str(rec.get("why") or "")[:80]})

    def score(self, now, homes) -> None:
        """The Judge's record of each step. Uses the no-event twin, so nothing here may feed a decision."""
        if not (self.w0 <= now < self.w1):
            return
        hg = sum(float(h.adapter.tel.get("grid_kw", 0.0)) - float(h.adapter.tel.get("grid_cf_kw", 0.0)) for h in homes.values())
        dg = sum(float(a.site.tel.get("import_cf_kw", 0.0)) - float(a.site.tel.get("import_kw", 0.0)) for a in self.ctx["agents"].values())
        self.score_rows.append({"ts": now, "homes_delivered_kw": round(hg, 1), "dc_delivered_kw": round(dg, 1)})

    def finish(self) -> dict[str, Any]:
        try:
            while True:
                next(self.gen)
        except StopIteration as e:
            return e.value or {}
        return {}
