"""Independent check of one settled event, from what the coordinator and the utility each hold.

Nothing here trusts either party's totals: every number is recomputed from the homes' own signed meter readings.
  order        the need the community contracted on is exactly the buyer agent's signed order (POST /needs)
  signatures   every reading verifies against the home key that the coordinator and the utility each pinned on
               first contact (the two pins agree), and the utility counted exactly the usage the home signed
  delivery     the DEG demand-flex settlement re-run on the signed readings and the utility's baselines gives the
               utility's total to the cent, slot by slot
  split        the coordinator's pro-rata split adds up to what the utility paid, and each home's share matches its
               measured delivery
  replay       (optional) the offline replay of the same order with the same homes pays the same to the cent; with
               AI agents, it plays what they decided (their signed offers say it) and the homes' answers; within
               0.5 % for an event that a live run reached through an earlier one (the replay starts fresh)

  bundle(coordinator_event, utility_evidence)   from GET :7871/v2/events/{id}?signed=true and GET :7873/evidence/{id}
  fetch_live(event_id)                          the same, fetched from the running agents
  check(bundle) -> {"ok": bool, "checks": [{"name", "ok", "detail"}], "summary": {...}}
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from . import beckn, config, settlement as st, signing
from .rt_market import PRICE_LADDER

SLOT_H = beckn.SLOT_MINUTES / 60
REPLAY_TOLERANCE = 0.005                # of the total: what a live run carried over from an earlier event (see check)


def bundle(coordinator_event: dict[str, Any], utility_evidence: dict[str, Any], source: str = "live",
           participants: str = "homes", buyer: str = "utility") -> dict[str, Any]:
    """One event's evidence in one JSON object (save it next to the video; tools/verify_run.py --bundle re-checks it).
    participants / buyer: what the sellers and the buyer are called in the report ("homes" and "utility"; "data
    centers" and "grid operator" for flexgrid_core/dc_fleet.py). On the pool, buyer also names whose order this bundle checks."""
    ev = coordinator_event
    need = ((ev.get("orders") or {}).get(buyer) or {}).get("need") or ev["need"]
    readings = {h: {str(s): r.get("signed_body") for s, r in rs.items()} for h, rs in ev.get("readings", {}).items()}
    return {"schema": "flexgrid.evidence.v1", "source": source, "event_id": ev["need"]["event_id"],
            "participants": participants, "buyer": buyer,
            "coordinator": {"need": need, "phase": ev.get("phase"), "result": ev.get("result"),
                            "commitments": {h: {"status": c.get("status"), "kw": [x["kw"] for x in
                                                c.get("contract", {}).get("commitments", [])]}
                                            for h, c in ev.get("commitments", {}).items()},
                            "offered_homes": sorted(ev.get("offers", {})),
                            "buildings": sorted(h for h, o in ev.get("offers", {}).items() if o.get("node") == "building"),
                            "pins": ev.get("pins", {}), "signed_readings": readings,
                            # what the AI agents decided (their signed offers carry it; the margin is the market's)
                            "agents": {"margin": ev.get("margin"),
                                       "offers": {h: o["agent"] for h, o in ev.get("offers", {}).items()
                                                  if isinstance(o.get("agent"), dict)},
                                       # the conditions of the day each seller declared in its signed offer (a data
                                       # center's urgent work, a battery out of service: the Control Room's what if)
                                       "conditions": {h: c for h, c in ((h, offer_conditions(o))
                                                                        for h, o in ev.get("offers", {}).items()) if c}}},
            "utility": utility_evidence}


def offer_conditions(o: dict[str, Any]) -> dict[str, Any]:
    """The conditions a seller declared in its signed offer: a data center's share of batch work that was urgent that
    day and a battery out of service (datacenter.DataCenterAgent.safe_offer); a home's what-if weather (home_v2).
    {} for a normal day."""
    c: dict[str, Any] = {}
    if isinstance(o, dict):
        if float(o.get("urgent_work_share") or 0) > 0:
            c["urgent_share"] = float(o["urgent_work_share"])
        if ((o.get("battery") or {}) if isinstance(o.get("battery"), dict) else {}).get("out_of_service"):
            c["battery_out"] = True
        w = o.get("weather")
        if isinstance(w, dict) and w.get("from"):          # a home's what-if weather (home_v2.HomeV2.build_offer)
            c["weather"] = {"pv": float(w.get("pv", 1.0)), "load": float(w.get("load", 1.0)), "from": str(w["from"])}
    return c


def fetch_live(event_id: str, coordinator: str = "http://127.0.0.1:7871",
               utility: str = "http://127.0.0.1:7873") -> dict[str, Any]:
    """The bundle for one event from the running coordinator and utility (each asked separately)."""
    import httpx
    ev = httpx.get(f"{coordinator}/v2/events/{event_id}", params={"signed": "true"}, timeout=30)
    ev.raise_for_status()
    ue = httpx.get(f"{utility}/evidence/{event_id}", timeout=30)
    ue.raise_for_status()
    return bundle(ev.json(), ue.json(), "live")


def replay_agents(b: dict[str, Any]) -> dict[str, Any] | None:
    """What the replay must play as it was decided live: the AI agents' offer decisions (from their signed offers),
    the coordinator's margin, and the homes that declined or did not answer. None when nothing was decided by an
    agent and every home accepted (the plain replay). Raises ValueError when an agent decided but its offer does not
    say what (an offer from before 1 Nov 2026): such an event cannot be replayed."""
    c = b["coordinator"]
    ag = c.get("agents") or {}
    offers = {h: d for h, d in (ag.get("offers") or {}).items() if d.get("decided_by") == "llm"}
    unknown = sorted(h for h, d in offers.items()       # a home's, a data center's or a building's decision
                     if all(d.get(k) is None for k in ("battery_share", "cooling_share", "lighting_share")))
    if unknown:
        raise ValueError(f"the AI agents of {', '.join(unknown)} decided their offers, and their offers do not record "
                         f"how (made before the decisions were signed into the offer): not replayable")
    declines = sorted(h for h, x in (c.get("commitments") or {}).items() if x.get("status") not in ("ACCEPTED", None))
    conditions = ag.get("conditions") or {}
    if not offers and not declines and ag.get("margin") is None and not conditions:
        return None
    # a home's decision: take part, battery share, EV pause; a data center's: take part, batch work and battery shares
    keys = ("take_part", "battery_share", "pause_ev", "pause_share", "cooling_share", "lighting_share")
    out = {"event_id": b["event_id"], "margin": ag.get("margin"), "declines": declines,
           "offers": {h: {k: d.get(k) for k in keys if k in d} for h, d in offers.items()}}
    if conditions:                       # the day's conditions the sellers declared: the replay runs under them too
        out["conditions"] = conditions
    return out


def replay_total(b: dict[str, Any]) -> float:
    """The same order with the same homes, offline (tools/sim_event.py in a fresh process); what the utility pays.
    With AI agents in the loop the replay plays what they decided live (replay_agents): the same decisions through
    the same code must pay the same, to the cent."""
    return replay_run(b)["utility_pays_usd"]


def replay_run(b: dict[str, Any]) -> dict[str, Any]:
    """replay_total's whole offline result (tools/sim_event.py's JSON for the day)."""
    u = b["utility"]
    names = b["coordinator"].get("offered_homes") or sorted(b["coordinator"]["signed_readings"])
    out = Path(tempfile.mkdtemp(prefix="flexgrid-verify-"))
    cmd = [sys.executable, str(config.REPO_DIR / "tools" / "sim_event.py"), "--event", u["need"]["window_start"][:10],
           "--only", ",".join(names), "--tag", "verify", "--out", str(out)]
    from . import building
    bl = [n for n in names if n in {x["name"] for x in building.registry()}]
    if bl:                                   # building nodes took part: the replay runs them too (their own levers)
        cmd += ["--buildings", ",".join(bl)]
    if u.get("order"):
        cmd += ["--order", json.dumps(u["order"]["order"] | {"buyer": u["order"]["buyer"]})]
    ag = replay_agents(b)
    if ag:
        cmd += ["--agents", json.dumps(ag)]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300)
    return json.loads(next(out.glob("sim_*.json")).read_text())


def _row(name: str, ok: bool | None, detail: str, **data) -> dict[str, Any]:
    return {"name": name, "ok": ok, "detail": detail, **data}


def check(b: dict[str, Any], replay_total: float | None = None) -> dict[str, Any]:
    u, c = b["utility"], b["coordinator"]
    who, buyer = b.get("participants") or "homes", b.get("buyer") or "utility"
    need = u["need"]
    out = []

    # 1. order: the contracted need is the agent's signed order
    o = u.get("order")
    if o:
        sig_ok = signing.verify(o, u["utility_pubkey"])
        same = beckn.need_from_catalog(o["resource"], o["offer"]) == need == c["need"]
        out.append(_row("order", sig_ok and same,
                        f"{o['buyer']} ordered {o['order']['kw']:g} kW {o['order']['start']}-{o['order']['end']} at "
                        f"${o['order']['price_usd_per_kwh']:g}/kWh; order signature {'valid' if sig_ok else 'INVALID'}; "
                        f"contracted need {'is' if same else 'is NOT'} that order"))
    else:
        out.append(_row("order", None, "no agent order: the day's catalog need was used"))

    # 2. signatures: each reading against both parties' pins, and the utility counted what the home signed
    signed = c.get("signed_readings", {})
    n = good = 0
    bad, pin_mismatch, miscounted = [], [], []
    for h, rs in signed.items():
        pc, pu = c.get("pins", {}).get(h), u.get("pins", {}).get(h)
        if pc and pu and pc != pu:
            pin_mismatch.append(h)
        for sid, body in rs.items():
            n += 1
            perf = ((body or {}).get("message") or {}).get("performance") or {}
            ok = bool(body) and signing.verify(body, pu or pc or "") and perf.get("home") == h \
                and str(perf.get("slot_id")) == sid and perf.get("event_id") == b["event_id"]
            if not ok:
                bad.append(f"{h}/{sid}")
                continue
            good += 1
            counted = u.get("meter_readings", {}).get(h, {}).get(sid, u.get("meter_readings", {}).get(h, {}).get(int(sid)))
            if counted is None or abs(float(counted) - float(perf["usage_kw"])) > 1e-9:
                miscounted.append(f"{h}/{sid}")
    counted_n = sum(len(v) for v in u.get("meter_readings", {}).values())
    sig_ok = n > 0 and good == n and not pin_mismatch and not miscounted and counted_n == n
    out.append(_row("signatures", sig_ok,
                    f"{good}/{n} signed readings valid against the {who}' pinned keys; coordinator and {buyer} pinned "
                    f"the same key for {len(signed) - len(pin_mismatch)}/{len(signed)} {who}; {buyer} counted "
                    f"{counted_n} readings, {len(miscounted)} differ from what was signed; "
                    f"{u.get('rejected_readings', 0)} rejected in transit",
                    bad=bad[:10], pin_mismatch=pin_mismatch[:10], miscounted=miscounted[:10]))

    # 3. delivery: re-run the DEG settlement on the signed readings and the utility's baselines
    meters: dict[str, dict[int, dict[str, float]]] = {}
    for h, rs in signed.items():
        for sid, body in rs.items():
            if f"{h}/{sid}" in bad:
                continue
            base = u.get("baselines", {}).get(h, {}).get(sid, {}).get("BASELINE")
            if base is not None and int(sid) not in (u.get("abnormal") or {}).get(h, ()):
                meters.setdefault(h, {})[int(sid)] = {"BASELINE": base,
                                                      "USAGE": float(body["message"]["performance"]["usage_kw"])}
                if st.MARKET:
                    perf = body["message"]["performance"]
                    meters[h][int(sid)]["BATT"] = float(perf.get("battery_kw") or 0.0)
                    meters[h][int(sid)]["BASELINE"] = st.adjusted(base, perf.get("pre_kw"), (u.get("pre_baselines") or {}).get(h))
    offered = {int(k): float(v) for k, v in (u.get("offered") or {}).items()}
    # a building node's meter: at most 1.2x what it committed in each half hour, from its own signed readings
    bl = set(c.get("buildings") or [])
    caps = {h: {int(sid): st.BUILDING_CAP_X * float(((body or {}).get("message") or {}).get("performance", {})
                                                    .get("committed_kw") or 0.0)
                for sid, body in rs.items() if f"{h}/{sid}" not in bad} for h, rs in signed.items() if h in bl}
    mine = st.deg_settle(need["slots"], offered, meters, hours=SLOT_H, currency=need["currency"], caps=caps or None,
                         share=u.get("share"), capped=u.get("capped"))
    theirs = u.get("settlement") or {}
    slots = []
    for s, r in zip(need["slots"], mine["components"]):
        slots.append({"slot": s["start"][11:], "requested_kw": s["kw_requested"], "contracted_kw": r["offered"],
                      "delivered_kw": r["delivered"], "paid_kw": r["eligible"], "usd": r["net"]})
    same_total = abs(mine["total"] - float(theirs.get("total", -1))) < 0.005
    short = [x for x in slots if x["delivered_kw"] < x["contracted_kw"] - 1e-6]
    out.append(_row("delivery", same_total,
                    f"recomputed ${mine['total']:.2f} vs {buyer} ${float(theirs.get('total', 0)):.2f}; "
                    f"{len(slots) - len(short)}/{len(slots)} slots delivered at least the contracted kW "
                    f"(measured against the utility's 3-of-14 baseline)", slots=slots))

    # 4. split: the community's payout adds up and follows measured delivery
    res = c.get("result") or {}
    sp = (res.get("split") or {}).get("per_home", {})
    paid_total = sum(sp.values())
    seller = float(res.get("seller_usd", float("nan")))
    this_buyer = float(((res.get("by_buyer") or {}).get(buyer) or {}).get("paid_usd", seller))    # the pool: this buyer's part of it
    fee = float((res.get("split") or {}).get("fee", 0.0))
    delivered = {h: sum(st.clamp0(v["BASELINE"] - v["USAGE"]) * SLOT_H for v in m.values()) for h, m in meters.items()}
    cm = c.get("commitments") or {}
    sids = [str(s["id"]) for s in need["slots"]]
    caps_kwh = {h: st.building_counted_kwh({str(k): v for k, v in meters[h].items()},
                                           dict(zip(sids, cm.get(h, {}).get("kw") or [])), SLOT_H)
                for h in (c.get("buildings") or []) if h in delivered and (cm.get(h) or {}).get("status") == "ACCEPTED"}
    expect = st.split(seller, delivered, fee / seller if seller > 0 else 0.0, caps_kwh or None)["per_home"]
    if st.MARKET:
        kwcap = {h: {str(k): float(v) for k, v in caps[h].items()} for h in caps_kwh if h in caps}   # as the buyer counted them: its readings
        val = {h: st.meter_kwh(m, SLOT_H, kwcap.get(h), h in kwcap, h) for h, m in meters.items()}
        st.trim_called(val, u.get("capped") or {}, SLOT_H)
        expect = st.share_out(seller, val)["per_home"]
        x_of = st.price_book()["x"]
        rate = float((res.get("split") or {}).get("usd_per_kwh") or 0.0)
        for h, v in ((res.get("split") or {}).get("premium_usd") or {}).items():    # a gap's helper, paid its rising price from the keep
            beyond = st.clamp0(sum((val.get(h) or ({}, {}))[0].values()) - sum((cm.get(h) or {}).get("kw") or []) * SLOT_H)
            if float(v) <= beyond * rate * float(x_of[st.node_class(h, h in bl)]) * (PRICE_LADDER[-1] - 1.0) + 0.01:   # at most the top of the ladder, on what the meters show beyond its booking
                expect[h] = round(expect.get(h, 0.0) + v, 4)
    off = [h for h in set(sp) | set(expect) if abs(sp.get(h, 0.0) - expect.get(h, 0.0)) > 0.0001]
    split_ok = bool(sp) and abs(paid_total + fee - seller) < 0.01 and abs(this_buyer - mine["total"]) < 0.005 \
        and not off and c.get("phase") == "done" and u.get("status") == "COMPLETE"
    out.append(_row("split", split_ok,
                    f"{len(sp)} {who} paid ${paid_total:.2f} (+ fee ${fee:.2f}) of the ${seller:.2f} the buyers paid; "
                    f"{len(sp) - len(off)}/{len(sp)} shares match measured delivery pro rata; settled once "
                    f"(coordinator {c.get('phase')}, {buyer} {u.get('status')})"))

    # 5. replay (optional): the same order, the same homes (and what their AI agents decided), offline. To the cent
    # for a run that started at this event; a live run that went through an earlier event carries that day's state
    # (batteries, plans) into this one and the replay starts fresh, so up to REPLAY_TOLERANCE of the total is allowed
    # (the AI agents' decisions are ten times that: 5 % on 3 Jul 2026)
    if replay_total is not None:
        diff = abs(replay_total - mine["total"])
        ag = (c.get("agents") or {})
        llm = sorted(h for h, d in (ag.get("offers") or {}).items() if d.get("decided_by") == "llm")
        what = f"the same order with the same {len(c.get('offered_homes') or signed)} {who}" + \
            (f" and what the AI agents decided ({', '.join(llm)}" + (f"; margin {ag['margin']:.0%}" if ag.get("margin") is not None
                                                                     else "") + ")" if llm else "")
        cond = ag.get("conditions") or {}
        if cond:                         # the day's conditions the sellers declared in their offers
            urg = sorted({v.get("urgent_share") for v in cond.values() if v.get("urgent_share")})
            out_b = sorted(h for h, v in cond.items() if v.get("battery_out"))
            wx = sorted({(v["weather"]["pv"], v["weather"]["load"]) for v in cond.values() if v.get("weather")})
            what += ", under the conditions they declared (" + "; ".join(
                ([f"{', '.join(f'{u:.0%}' for u in urg)} of the batch work urgent at {len([1 for v in cond.values() if v.get('urgent_share')])}"]
                 if urg else []) + ([f"batteries out at {', '.join(out_b)}"] if out_b else []) +
                ([f"weather: sunshine {pv:.0%}, household load {ld:.0%} at {len([1 for v in cond.values() if v.get('weather')])}"
                  for pv, ld in wx])) + ")"
        exact = diff < 0.005
        out.append(_row("replay", exact or diff <= REPLAY_TOLERANCE * mine["total"],
                        f"offline replay of {what} pays ${replay_total:.2f} (live ${mine['total']:.2f})" +
                        ("" if exact else f": {diff / max(mine['total'], 1e-9):.2%} apart (allowed {REPLAY_TOLERANCE:.1%}: the "
                                          f"replay starts fresh at this event, a live run carries the day before)")))

    true_kwh = res.get("delivered_kwh_true")
    summary = {"event_id": b["event_id"], "window": [need["window_start"], need["window_end"]],
               "requested_kw": need["slots"][0]["kw_requested"], "price_usd_per_kwh": need["slots"][0]["price"],
               "contracted_kwh": round(sum(offered.values()) * SLOT_H, 2),
               "delivered_kwh_measured": round(sum(delivered.values()), 2),
               "paid_kwh": round(sum(r["eligible"] for r in mine["components"]) * SLOT_H, 2),
               "delivered_kwh_true": true_kwh, "utility_pays_usd": mine["total"], "homes_paid": len(sp),
               "readings": n, "participants": who}
    return {"ok": all(r["ok"] is not False for r in out), "checks": out, "summary": summary}


def report(r: dict[str, Any]) -> str:
    """The check as a few lines of text (terminal, MCP tool result)."""
    lines = []
    for x in r["checks"]:
        mark = "PASS" if x["ok"] else ("SKIP" if x["ok"] is None else "FAIL")
        lines.append(f"{mark:4s}  {x['name']:10s} {x['detail']}")
    s = r["summary"]
    lines.append(f"{'ALL CHECKS PASS' if r['ok'] else 'CHECK FAILED'}: {s['event_id']} {s['window'][0]}-"
                 f"{s['window'][1][11:]}, contracted {s['contracted_kwh']} kWh, measured {s['delivered_kwh_measured']} "
                 f"kWh, paid for {s['paid_kwh']} kWh, ${s['utility_pays_usd']:.2f} to {s['homes_paid']} "
                 f"{s.get('participants') or 'homes'}")
    return "\n".join(lines)
