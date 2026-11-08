"""The test bench's scorecard: one run of a day (tools/sim_event.py), scored the way each party would judge it.

  booking   what the community sold the day before vs what really changed in the window (the twin's measure)
  calls     each call during the day: asked, confirmed, how fast, measured extra, paid, and the true extra
  cars      every car that left that day: did it leave as charged as its owner's own schedule would have left it?
  comfort   the homes whose air conditioning took part: degree-hours warmer than their own day, the worst minute,
            and any 15 minutes past the owner's limit
  money     what the buyer paid (booking + calls) and what reached the homes
  grid      the community's peak (the day, after the window), its energy at real-time prices, against the same day
            with no event and no calls (every home's twin)
  messages  how many messages it took, by kind
"""
from __future__ import annotations

from typing import Any

from flexgrid_core import beckn
from flexgrid_core.replay import Replay


def _steps(homes: dict[str, Any], day: str, stop: str) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for n, h in homes.items():
        tel = [t for _, t in sorted(h.adapter.recent.items()) if day <= t["ts"] <= stop]
        out[n] = tel
    return out


def score(day: str, stop: str, homes: dict[str, Any], market, util, feeder: dict[str, dict[str, float]],
          traffic: dict[str, Any], log: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted((ts, r) for ts, r in feeder.items() if ts >= f"{day} 00:00")
    ev = next(iter(market.events.values()), None)
    n = ev["need"] if ev else None
    res = (ev or {}).get("result") or {}
    offered = {s["start"]: float((ev.get("capacity_offered_final") or {}).get(s["id"], 0.0)) for s in n["slots"]} if ev else {}

    def booked_at(ts: str) -> float:
        return offered.get(beckn.add_minutes(ts, -15) if ts[14:16] in ("15", "45") else ts, 0.0)

    # booking: sold vs truly delivered in the window
    win = [(ts, r) for ts, r in rows if n and n["window_start"] <= ts < n["window_end"]]
    true_win = sum(r["import_cf_kw"] - r["import_kw"] for _, r in win) * 0.25
    promised = sum(offered.values()) * 0.5
    booking = {"promised_kwh": round(promised, 1), "true_kwh": round(true_win, 1),
               "measured_kwh": res.get("delivered_kwh_measured"), "paid_usd": res.get("seller_usd"),
               "true_vs_promised": round(true_win / promised, 3) if promised > 0 else None}

    # calls: what the buyer asked during the day and what came of it
    calls = []
    for c in util.watch.calls.values():
        ev_log = [x for x in log if x.get("who") == "utility" and x.get("call_id") == c["id"]]
        issued = next((x["sim"] for x in ev_log if x["kind"] == "rt_call"), c["issued_at"])
        conf = next((x["sim"] for x in ev_log if x["kind"] == "rt_confirmed"), None)
        sel = [(ts, r) for ts, r in rows if c["start"] <= ts < c["end"]]
        extra = [(r["import_cf_kw"] - r["import_kw"]) - booked_at(ts) for ts, r in sel]
        calls.append({"id": c["id"], "kind": c["kind"], "why": c["why"], "start": c["start"], "end": c["end"],
                      "kw_asked": c["kw"], "kw_confirmed": c.get("confirmed_kw"),
                      "minutes_to_confirm": (Replay.minutes_between(issued, conf) if conf else None),
                      "measured_extra_kwh": c.get("delivered_kwh"), "paid_usd": c.get("paid_usd"),
                      "true_extra_kw_avg": round(sum(extra) / len(extra), 1) if extra else None,
                      "true_extra_kw_min": round(min(extra), 1) if extra else None,
                      "warnings": len(c.get("warnings", []))})
    gaps = [g for g in market.rt.calls.values() if g["kind"] == "gap"]
    gap_rows = [{"home": g["from"], "start": g["start"], "end": g["end"], "kw": round(g["kw"], 2), "why": g["why"],
                 "covered_kw": g.get("confirmed_kw"), "short_kw": g.get("short_kw"), "rounds": g.get("round", 0) + 1,
                 "helpers": g.get("homes")} for g in gaps]

    # cars: every car that left between the day's start and the end of the run
    left, short, unmet, cf_short = 0, 0, 0.0, 0
    for h in homes.values():
        for e in h.adapter.ev_log:
            if f"{day} 00:00" <= e["left_at"] <= stop:
                left += 1
                unmet += e["unmet_kwh"]
                short += e["unmet_kwh"] > 0.1
        cf_short += sum(1 for e in h.adapter.cf_ev_log if f"{day} 00:00" <= e["left_at"] <= stop and e["unmet_kwh"] > 0.1)
    cars = {"left": left, "left_short": short, "unmet_kwh": round(unmet, 2), "left_short_on_own_schedule": cf_short}

    # comfort: rolling homes only (the fixed plan never touches the air conditioning)
    tel = _steps(homes, f"{day} 00:00", stop)
    deg_h, worst, over, ac_homes = 0.0, 0.0, 0, 0
    for nm, ts in tel.items():
        h = homes[nm]
        if getattr(h.adapter, "mode", "fixed") != "rolling" or not h.adapter.house \
                or getattr(h.adapter, "name", "") == "building":          # a building is scored on its own (below)
            continue
        ac_homes += 1
        lim = h.adapter.ctl.limits
        ov = (h.adapter.overrides or {}).get("comfort") or {}
        hi_occ, hi_away = ov.get("max_rise_c", lim.max_rise_c), ov.get("max_rise_away_c", lim.max_rise_away_c)
        for t in ts:
            x = t.get("t_dev_c") or 0.0
            if t.get("occupied"):
                deg_h += max(0.0, x) * 0.25
            worst = max(worst, x)
            if x > (hi_occ if t.get("occupied") else hi_away) + 0.05:
                over += 1
    comfort = {"homes_with_ac_in_play": ac_homes, "degree_hours_warmer_occupied": round(deg_h, 2),
               "worst_c": round(worst, 2), "steps_past_limit": over}

    # money
    calls_paid = sum(float(c.get("paid_usd") or 0.0) for c in util.watch.calls.values())
    homes_got = sum(float(p.get("payment") or 0.0) for h in homes.values() for p in h.payments)
    money = {"buyer_paid_booking_usd": res.get("seller_usd"), "buyer_paid_calls_usd": round(calls_paid, 2),
             "homes_received_usd": round(homes_got, 2)}

    # grid: the community against its twin (the same day with no event and no calls)
    day_rows = [(ts, r) for ts, r in rows if ts < beckn.add_minutes(f"{day} 00:00", 1440)]
    price = {}
    clock = util.clock
    for ts, _ in day_rows:
        i = clock.idx_of(ts)
        price[ts] = clock.row(i)["price_usd_mwh"] / 1000.0 if i is not None else 0.0
    imp = lambda r: r.get("import_homes_kw", r["import_kw"])            # noqa: E731  the homes' feeder: a building's
    imp_cf = lambda r: r.get("import_cf_homes_kw", r["import_cf_kw"])   # noqa: E731  own 1 MW would hide their peaks
    cost = sum(imp(r) * price[ts] * 0.25 for ts, r in day_rows)
    cost_cf = sum(imp_cf(r) * price[ts] * 0.25 for ts, r in day_rows)
    post = [(ts, r) for ts, r in rows if n and n["window_end"] <= ts < beckn.add_minutes(n["window_end"], 180)]
    grid = {"day_peak_kw": round(max(imp(r) for _, r in day_rows), 1),
            "day_peak_no_event_kw": round(max(imp_cf(r) for _, r in day_rows), 1),
            "post_3h_peak_kw": round(max((imp(r) for _, r in post), default=0.0), 1),
            "post_3h_peak_no_event_kw": round(max((imp_cf(r) for _, r in post), default=0.0), 1),
            "energy_at_rt_price_usd": round(cost, 2), "energy_at_rt_price_no_event_usd": round(cost_cf, 2),
            "import_kwh": round(sum(imp(r) for _, r in day_rows) * 0.25, 1),
            "import_kwh_no_event": round(sum(imp_cf(r) for _, r in day_rows) * 0.25, 1)}
    buildings = {}
    for nm, h in homes.items():                       # each building node on its own: booked, delivered, its people
        if getattr(h.adapter, "name", "") != "building":
            continue
        ts_b = tel.get(nm) or []
        cm = (ev or {}).get("commitments", {}).get(nm) or {}
        booked = sum(x["kw"] for x in (cm.get("contract") or {}).get("commitments", [])) * 0.5 \
            if cm.get("status") == "ACCEPTED" else 0.0
        win_b = [t for t in ts_b if n and n["window_start"] <= t["ts"] < n["window_end"]]
        occ_dh = sum(max(0.0, t.get("t_dev_c") or 0.0) * 0.25 for t in ts_b if t.get("occupied"))
        lim_b = h.adapter.ctl.limits
        over_b = sum(1 for t in ts_b if (t.get("t_dev_c") or 0.0) > (lim_b.max_rise_c if t.get("occupied")
                                                                      else lim_b.max_rise_away_c) + 0.05)
        buildings[nm] = {"accepted": cm.get("status") == "ACCEPTED", "booked_kwh": round(booked, 1),
                         "true_kwh_in_window": round(sum(t["grid_kw"] - t["grid_cf_kw"] for t in win_b) * 0.25, 1),
                         "lighting_dimmed_kwh": round(sum(t.get("light_dim_kw", 0.0) for t in ts_b) * 0.25, 1),
                         "cooling_change_kwh": round(sum(t.get("cool_kw", 0.0) - t.get("cool_own_day_kw", 0.0)
                                                         for t in ts_b) * 0.25, 1),
                         "rt_extra_kwh": round(sum(t.get("rt_extra_kw", 0.0) for t in ts_b) * 0.25, 1),
                         "comfort_degree_hours_open": round(occ_dh, 2),
                         "worst_c": round(max((t.get("t_dev_c") or 0.0 for t in ts_b), default=0.0), 2),
                         "steps_past_limit": over_b,
                         "paid_usd": round(sum(float(p.get("payment") or 0.0) for p in h.payments), 2)}
    rt_kinds = {k: v for k, v in traffic["by_kind"].items() if "rt_" in k}
    out = {"booking": booking, "calls": calls, "gaps": gap_rows, "cars": cars, "comfort": comfort, "money": money,
           "grid": grid, "messages": {"total": traffic["messages"], "real_time": rt_kinds}}
    if buildings:
        out["buildings"] = buildings
    return out
