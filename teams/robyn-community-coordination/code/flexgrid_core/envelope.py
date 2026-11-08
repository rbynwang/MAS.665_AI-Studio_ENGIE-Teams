"""Build the flexibility envelope: how far this home can move its grid power, for how long.

up_kw   = extra export / less import available now (discharge headroom + pausing the EV)
down_kw = extra import available now (charge headroom + starting a waiting EV)
Only these numbers leave the home - never raw readings, schedules or device ids.
"""
from __future__ import annotations

from typing import Any


def build(tel: dict[str, Any], limits: dict[str, Any], health: dict[str, Any],
          forecast: list[dict[str, Any]], agent: str, site_label: str, seq: int,
          brand: str, valid_minutes: int = 15) -> dict[str, Any]:
    E, P = limits["battery_kwh"], limits["battery_kw"]
    R, SM, eta = limits["reserve_pct"] / 100, limits["soc_max_pct"] / 100, limits.get("eta", 0.95)
    soc = tel["soc_pct"] / 100
    b = tel["batt_kw"]
    e_up = max(0.0, (soc - R) * E * eta)              # kWh that can still be discharged
    e_dn = max(0.0, (SM - soc) * E / eta)             # kWh that can still be absorbed
    up_batt = max(0.0, min(P - max(b, 0.0), e_up / 0.25))
    dn_batt = max(0.0, min(P - max(-b, 0.0), e_dn / 0.25))
    up = up_batt + tel.get("ev_kw", 0.0)              # pausing EV charging also frees power
    ev_wait = limits["ev_max_kw"] - tel.get("ev_kw", 0.0) if tel.get("ev_need_kwh", 0) > 0.01 else 0.0
    dn = dn_batt + max(0.0, ev_wait)
    cf = health.get("confidence_factor", 1.0)
    base_conf = 0.9 if limits.get("control", "auto") == "auto" else 0.6
    conf = round(base_conf * cf, 2)
    if health.get("status") in ("frozen", "jump"):    # never promise power we cannot verify
        up, dn = 0.0, 0.0
    sustain_up = (e_up / up_batt) if up_batt > 0.05 else 0.0
    sustain_dn = (e_dn / dn_batt) if dn_batt > 0.05 else 0.0
    baseline = [{"slot": f["slot"], "grid_p50": round(f["pv_p50"] - f["load_p50"], 3),
                 "grid_p10": round(f["pv_p10"] - f["load_p90"], 3),
                 "grid_p90": round(f["pv_p90"] - f["load_p10"], 3)} for f in forecast]
    return {
        "type": "flex.report", "version": "flexgrid.envelope.v1",
        "agent": agent, "site": site_label, "brand": brand, "seq": seq,
        "slot_start": tel["ts"], "valid_minutes": valid_minutes,
        "grid_kw_now": tel["grid_kw"],
        "up_kw": round(up, 2), "down_kw": round(dn, 2),
        "sustain_up_h": round(min(sustain_up, 12), 2), "sustain_down_h": round(min(sustain_dn, 12), 2),
        "soc_band_pct": [round(max(R * 100, tel["soc_pct"] - 2), 1), round(min(SM * 100, tel["soc_pct"] + 2), 1)],
        "ev_deadline": tel.get("ev_deadline"), "ev_need_kwh": tel.get("ev_need_kwh", 0.0),
        "confidence": conf, "health": health.get("status"), "control": limits.get("control", "auto"),
        "baseline_next": baseline,
    }
