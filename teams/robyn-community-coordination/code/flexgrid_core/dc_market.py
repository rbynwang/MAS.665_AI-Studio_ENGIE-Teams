"""The coordinator's market for data centers: the household market (market.py) unchanged, plus the one step large
loads need on top: when the paused work comes back.

A data center's paused batch work must run again within its deferral window (4.5 to 10.5 h). Run straight after the
window, it lands on the evening shoulder of the region's peak and can make a new, higher peak (seen in the first run:
+12.6 MW at 18:00 on 2023-08-24, above the day's no-event peak). The homes' version of this is the EV restart, placed by
valley filling over the community's own evening (market._place_evs). For data centers the load that matters is the
region's, so after clearing, the coordinator:
  1. forecasts the region's load for the evening (the same step on the 3 latest days seen: history only),
  2. splits each data center's commitment per 15-minute step into paused work (up to its offered compute pause)
     and battery, the paused part with its deadline (step + deferral window),
  3. fills the lowest steps of the forecast first, earliest deadline first, never past a deadline and never above
     what the site said it can take (`evening.room_kw` in its offer), largest data center first,
  4. sends each data center its own resume plan inside its contract (`resume_plan`: kW of paused work per step).
The data center runs that plan; work at its deadline still runs whatever the plan says (the plan cannot make a job
late). Battery refills wait for the night (22:00-06:00, datacenter.py).
"""
from __future__ import annotations

import json
from typing import Any, Callable

from . import beckn, config
from .market import Market


class DCMarket(Market):
    def __init__(self, *args, region_forecast: Callable[[list[str], str], list[float]] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.region_forecast = region_forecast
        self.fill_step_kw = float(self.cfg.get("dc_fill_step_kw", 250.0))

    def clear(self, event_id: str) -> dict[str, Any]:
        s = super().clear(event_id)
        ev = self.events[event_id]
        if ev.get("phase") == "cleared" and "resume_plans" not in ev:
            self._plan_resume(ev)
        return s

    def _contract_for(self, ev: dict[str, Any], home: str, slots: dict[int, float], k_ev: int) -> dict[str, Any]:
        c = super()._contract_for(ev, home, slots, k_ev)
        plan = (ev.get("resume_plans") or {}).get(home)
        if plan:
            c["resume_plan"] = plan
        return c

    def _plan_resume(self, ev: dict[str, Any]) -> None:
        n = ev["need"]
        w0, w1 = n["window_start"], n["window_end"]
        horizon = [beckn.add_minutes(w1, 15 * k) for k in range(96)]      # up to 24 h after the window
        base = self.region_forecast(horizon, self.now()) if self.region_forecast else [0.0] * len(horizon)
        total = [x * 1000 for x in base]                                  # MW -> kW
        placed_all = [0.0] * len(horizon)
        plans: dict[str, dict[str, float]] = {}
        sizes = []
        for h, alloc in ev["alloc"].items():
            o = ev["offers"].get(h) or {}
            if o.get("node_type") != "datacenter":
                continue
            w = int(round(float(o.get("deferral_window_h", 0)) * 4))
            pause = {x["id"]: x.get("compute_pause_kw", 0.0) for x in o["slots"]}
            chunks = []                                                   # (deadline step in horizon, kW to place)
            for sid, kw in sorted(alloc.items()):
                start = next(s["start"] for s in n["slots"] if s["id"] == sid)
                for k in (0, 1):
                    t = beckn.add_minutes(start, 15 * k)
                    p = min(kw, pause.get(sid, 0.0))
                    k_since = int(round(self.clock.minutes_between(w0, t) / 15))
                    win_steps = int(round(self.clock.minutes_between(w0, w1) / 15))
                    dl = k_since + w - win_steps                          # deadline, in steps after the window end
                    if p > 1e-6 and dl >= 0:
                        chunks.append((min(dl, len(horizon) - 1), p))
            room = list((o.get("evening") or {}).get("room_kw") or [])
            room = (room + [room[-1] if room else 0.0] * len(horizon))[:len(horizon)]
            sizes.append((sum(c[1] for c in chunks), h, chunks, room))
        for _, h, chunks, room in sorted(sizes, key=lambda x: -x[0]):
            mine = [0.0] * len(horizon)
            for dl, kw in sorted(chunks):
                left = kw                                                 # kW for one step = kWh / 0.25
                while left > 1e-6:
                    cand = [j for j in range(dl + 1) if mine[j] < room[j] - 1e-6]
                    if not cand:                                          # no room before the deadline: the site
                        break                                             # runs it at the deadline itself
                    j = min(cand, key=lambda j: (total[j], -j))
                    x = min(self.fill_step_kw, left, room[j] - mine[j])
                    mine[j] += x
                    total[j] += x
                    placed_all[j] += x
                    left -= x
            plans[h] = {horizon[j]: round(v, 1) for j, v in enumerate(mine) if v > 1e-6}
        ev["resume_plans"] = plans
        if self.region_forecast:
            ev["region_guard"] = {"forecast_no_event_peak_mw": round(max(base), 1),
                                  "forecast_with_resume_peak_mw": round(max(total) / 1000, 1),
                                  "resume_kwh": round(sum(placed_all) * 0.25, 1),
                                  "last_resume_step": max((t for p in plans.values() for t in p), default=None)}
        self.log("dc_resume_planned", event_id=n["event_id"], datacenters=len(plans),
                 **{k: v for k, v in (ev.get("region_guard") or {}).items()})

    # ------------------------------------------------------------------ the desk's AI agent (CoordinatorBrain, DESK_DECIDE)
    def _backtest(self, day: str) -> dict[str, Any] | None:
        """The same ten data centers and clearing replayed offline on the season's earlier event days
        (tools/sim_datacenters.py event --event all -> dc_events_all.json in results/<dataset>_dc/, the Illinois year's
        in results/): what was sold against what the grid operator's meters measured. The grid operator pays per kWh
        measured against each meter's baseline, at most what was sold: on a day the baseline runs low, a bigger margin
        is what fills it."""
        if day in self._backtests:
            return self._backtests[day]
        out = None
        try:
            ds = config.DATASET
            f = config.REPO_DIR / "results" / (f"{ds}_dc" if ds != "il2023" else "") / "dc_events_all.json"
            rows = []
            for r in json.loads(f.read_text()) if f.exists() else []:
                if str(r.get("event")) >= day:
                    continue
                w0, w1 = str(r.get("window") or "-").split("-")[:2]
                hours = (int(w1[:2]) * 60 + int(w1[3:5]) - int(w0[:2]) * 60 - int(w0[3:5])) / 60 if w0 and w1 else 0
                sold = float(r.get("capacity_mw") or 0) * 1000 * hours
                if sold > 0 and r.get("measured_mwh") is not None:
                    mk = float(r["measured_mwh"]) * 1000
                    rows.append({"date": r["event"], "sold_kwh": round(sold), "measured_kwh": round(mk),
                                 "measured_pct_of_sold": round(100 * mk / sold),
                                 "true_kwh": round(float(r.get("true_mwh") or 0) * 1000), "paid_usd": r.get("paid_usd")})
            if rows:
                pct = [x["measured_pct_of_sold"] for x in rows]
                out = {"what": "the same ten data centers and clearing replayed on this season's earlier event days "
                               "(simulation, on rules, a 10 % margin)",
                       "events": len(rows), "measured_pct_of_sold_mean": round(sum(pct) / len(pct)),
                       "measured_pct_of_sold_worst": min(pct), "days_measured_below_sold": sum(1 for p in pct if p < 100),
                       "last_events": rows[-5:]}
        except (OSError, ValueError, TypeError):
            out = None
        self._backtests[day] = out
        return out

    def margin_facts(self, ev: dict[str, Any]) -> dict[str, Any]:
        """What the desk knows when its AI agent picks the margin: the request, the data centers' offers (structured:
        batch work paused and battery, never a site's free text), what each margin would sell (sell_by_margin) and this
        run's finished events."""
        n, offers = ev["need"], ev["offers"]
        get = lambda o, sid, k: next((x.get(k, 0.0) for x in o["slots"] if x["id"] == sid), 0.0)  # noqa: E731
        per = [{"start": s["start"][11:], "kw_asked": s["kw_requested"],
                "batch_work_pause_kw_offered": round(sum(get(o, s["id"], "compute_pause_kw") for o in offers.values()), 1),
                "battery_kw_offered": round(sum(get(o, s["id"], "dc_battery_kw") for o in offers.values()), 1),
                "confidence_weighted_kw": round(sum(get(o, s["id"], "kw_offered") * self._confidence(o)
                                                    for o in offers.values()), 1)} for s in n["slots"]]
        conf = sorted(self._confidence(o) for o in offers.values()) or [0.0]
        done = []
        for e in self.events.values():
            res = e.get("result") or {}
            if e is ev or not res:
                continue
            cap = e.get("capacity_offered_final") or e.get("capacity_offered") or {}
            done.append({"event_id": e["need"]["event_id"], "promised_kwh": round(sum(cap.values()) * 0.5, 1),
                         "measured_kwh": res.get("delivered_kwh_measured"), "paid_usd": res.get("seller_usd")})
        return {"now": self.now(), "default_margin": self.margin,
                "request": {"event_id": n["event_id"], "date": n["window_start"][:10],
                            "window": f"{n['window_start'][11:]}-{n['window_end'][11:]}",
                            "price_usd_per_kwh": n["slots"][0].get("price"), "reply_to_buyer_by": n.get("commit_by")},
                "offers": {"data_centers": len(offers),
                           "offering_nothing": sum(1 for o in offers.values() if (o.get("energy_kwh") or 0) <= 0.01),
                           "with_ai_agent": sorted(h for h, o in offers.items() if (o.get("agent") or {}).get("decided_by")),
                           "confidence": {"min": round(conf[0], 2), "median": round(conf[len(conf) // 2], 2),
                                          "max": round(conf[-1], 2)},
                           "per_half_hour": per},
                "sell_by_margin": self.sell_by_margin(ev),
                "this_run_finished_events": done[-5:], "backtest": self._backtest(n["window_start"][:10]),
                "how_payment_works": "the grid operator pays per kWh its meters measure against each site's baseline, "
                                     "at most what was sold in each half hour; asking the data centers for more raises "
                                     "what is measured on a day the baseline runs low",
                "how_margin_works": "data centers are asked for (kW sold x (1 + margin)); the desk sells at most what "
                                    "they are asked for / (1 + margin), and never more than the grid operator asked for"}
