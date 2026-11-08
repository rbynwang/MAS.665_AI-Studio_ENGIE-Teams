"""The AI agents' conversation for the Control Room's chat box and the phone app: GET /monitor/state -> "agents".

One list of lines, oldest first, in plain words, from what the agents really did in the current run:
  message   a protocol step (the Beckn messages over NANDA Town: the coordinator's and the utility's logs, and the two
            Jetson homes' and the building nodes' own offers, answers and signed readings as the market recorded them)
  decision  an AI agent's decision with its reason, the model and the seconds it took, or "rules" when it fell back
            (the two Jetson homes' and the building nodes' offers and answers, the coordinator's margin, the buyer's
            order and sign-off)
  owner     a homeowner's chat with their home's agent in the phone app: from "owner:<home>" to the home, and the
            agent's answer from the home to "owner:<home>"
Each line has its phase ("role": need, offer, commit, pre, report, money). Lines that move money carry it: "money":
{"usd", "what"}, + to the one who receives it (or a decision-maker's own expected income), - paid by the buyer, and
usd null when the effect is not dollars (the margin). "money" (top level) sums up the event so far: the price, kW
asked / offered / sold, what the buyer pays and what the homes earn. Field names: viz/live (fg-agents.js) and its
mock, viz/live/mock/agents_2026-07-02.json.
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime
from typing import Any

from . import beckn, config

SLOT_H = beckn.SLOT_MINUTES / 60
HOME_NAMES = {"home-tesla": "Tesla home", "home-enphase": "Enphase home"}
HOME_HOSTS = {"home-tesla": "jetson-a", "home-enphase": "jetson-b"}
JETSONS = tuple(HOME_NAMES)
_BLD: dict[str, dict[str, Any]] | None = None


def _buildings() -> dict[str, dict[str, Any]]:
    """The dataset's building nodes by name (building.registry()), {} when it has none."""
    global _BLD
    if _BLD is None:
        try:
            from . import building
            _BLD = {b["name"]: b for b in building.registry()}
        except Exception:                                # noqa: BLE001  the chat never breaks over a name
            _BLD = {}
    return _BLD


def node_name(n: Any) -> Any:
    """A node's name in the conversation: "Tesla home", "Library" (a building node's short name), else as given."""
    return HOME_NAMES.get(n) or (_buildings().get(n) or {}).get("short") or n


def model_label(name: str | None) -> str | None:
    """'hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q3_K_XL', 'ollama/qwen3.8-27b-q3' -> 'Qwen3.8-27B (3-bit)'."""
    if not name:
        return None
    n = str(name)
    if n.startswith("openclaw/"):
        n = mac_model() or n
    base = n.split("/")[-1].split(":")[0].replace("-GGUF", "")
    if base.lower().startswith("qwen3.8") and "27" in n:
        base = "Qwen3.8-27B"
    bits = next((f"({b}-bit)" for b, keys in (("3", ("q3", "3bit", "3-bit")), ("4", ("q4", "4bit", "mlx", "4-bit")),
                                                    ("8", ("q8", "8bit"))) if any(k in n.lower() for k in keys)), "")
    if not bits and n.lower().startswith("qwen3.8:27b"):
        bits = "(4-bit)"                                  # Ollama's own qwen3.8:27b is a 4-bit build
    return f"{base} {bits}".strip()


def mac_model() -> str | None:
    """The model the Mac serves (mac/setup_llm.sh writes ~/.config/flexgrid/llm.json; elsewhere: llm_model)."""
    try:
        return json.loads((config.CONFIG_DIR / "llm.json").read_text()).get("model")
    except (OSError, ValueError):
        return config.load().get("llm_model") or "qwen3.8-27b-q3"


def _day(ts: str | None) -> str:
    if not ts:
        return ""
    try:
        return datetime.strptime(ts[:10], "%Y-%m-%d").strftime("%a %-d %b")
    except ValueError:
        return ts[:10]


def _kw(x: Any) -> str:
    try:
        return f"{float(x):.0f}" if float(x) >= 10 else f"{float(x):.1f}"
    except (TypeError, ValueError):
        return "?"


def _kws(v: list[Any]) -> str:
    return " · ".join(f"{float(x):.2f}" for x in v)


def _mean(v: Any) -> float | None:
    vals = list(v.values()) if isinstance(v, dict) else list(v or [])
    vals = [float(x) for x in vals if x is not None]
    return round(sum(vals) / len(vals), 1) if vals else None


def _usd(x: Any) -> float | None:
    try:
        return round(float(x), 2)
    except (TypeError, ValueError):
        return None


def _llm(d: dict[str, Any]) -> dict[str, Any] | None:
    m = d.get("llm") or {}
    name = str(m.get("model") or "")
    if d.get("fallback") and not name:
        return {"model": None, "ms": None, "fallback": True, "runtime": None, "error": m.get("error")}
    return {"model": model_label(name) or model_label(mac_model()), "ms": m.get("ms"), "fallback": bool(d.get("fallback")),
            "runtime": m.get("runtime"), "error": m.get("error")}


def what_if_line(ch: dict[str, Any]) -> dict[str, Any] | None:
    """The presenter's what if (the coordinator's `what_if` log entry) as a line of the conversation: what changed in
    the world the agents live in; the agents' reactions follow as their own lines."""
    from .scenarios import PRESETS
    base = {"from": "presenter", "to": None, "kind": "whatif", "role": "whatif"}
    if ch.get("preset"):
        p = PRESETS.get(str(ch["preset"])) or {}
        if ch["preset"] == "normal":
            return {**base, "text": "A normal day: the weather, the region and the data centers as in the dataset"}
        return {**base, "text": f"{p.get('label') or ch['preset']}: {p.get('what') or ''}".strip(),
                "reason": p.get("expect")}
    parts = []
    if ch.get("reset"):
        parts.append("everything back to normal")
    w = ch.get("weather") or {}
    if w.get("pv") is not None:
        parts.append(f"sunshine {float(w['pv']):.0%} on every roof")
    if w.get("load") is not None:
        parts.append(f"household load {float(w['load']):.0%}")
    rg = ch.get("region") or {}
    if rg.get("note"):
        parts.append(f"the weather service: {rg['note']}")
    elif rg.get("load_x") is not None:
        parts.append(f"the region's load forecast ×{float(rg['load_x']):g}")
    d = ch.get("datacenters") or {}
    if d.get("urgent_share") is not None:
        u = float(d["urgent_share"])
        parts.append(f"{u:.0%} of the data centers' batch work is urgent today" if u > 0 else "no urgent work at the data centers")
    if isinstance(d.get("urgent_job"), dict):
        uj = d["urgent_job"]
        parts.append(f"an urgent job at {uj.get('site')}: {float(uj.get('mw') or 0):g} MW of its batch work can no longer "
                     f"wait, for {float(uj.get('hours') or 0):g} h")
    if "battery_out" in d:
        parts.append(f"batteries out of service at {', '.join(d['battery_out'])}" if d["battery_out"]
                     else "the data centers' batteries back in service")
    bf = ch.get("battery_fault") or {}
    if bf:
        who = {"fleet": f"{bf.get('n') or 10} fleet homes" if bf.get("on", True) else "the fleet",
               "home-tesla": "the Tesla home", "home-enphase": "the Enphase home"}.get(bf.get("home"), str(bf.get("home")))
        parts.append(("batteries repaired: " if bf.get("on") is False else "battery fault: ") + who)
    off = ch.get("offline") or {}
    if off:
        name = "jetson-a (the Tesla home)" if off.get("home") == "home-tesla" else "the Enphase home"
        parts.append(f"{name} goes offline" if float(off.get("seconds") or 0) > 0 else f"{name} is back online")
    if ch.get("note"):
        parts.append(f"the utility's buyer hears: {ch['note']}")
    wn = ch.get("weather_now") or {}
    if wn.get("kind") == "storm":
        parts.append(f"a storm now: solar down 80 % and 6 °C cooler for {float(wn.get('hours') or 2):g} h")
    elif wn.get("kind") == "heat":
        parts.append(f"a heat spike now: 3 °C hotter for {float(wn.get('hours') or 3):g} h")
    elif wn.get("kind") == "clear":
        parts.append("the weather is back to what was measured")
    ps = ch.get("price_spike") or {}
    if ps:
        parts.append(f"the real-time price jumps ×{float(ps.get('factor') or 3):g} for {float(ps.get('hours') or 1):g} h")
    em = ch.get("emergency") or {}
    if em:
        parts.append(f"the grid operator declares an emergency: {float(em.get('pct') or 50):.0f} % more relief wanted "
                     f"for {float(em.get('minutes') or 60):.0f} min")
    ow = ch.get("owner") or {}
    if ow and ow.get("home") in _buildings():            # a building's facility manager (building.py)
        who = node_name(ow.get("home"))
        parts.append({"comfort": f"the facility manager of the {who} allows "
                                 f"{float(ow.get('add_c') if ow.get('add_c') is not None else 1.5):g} °C warmer while open",
                      "lights": f"the facility manager of the {who} lets "
                                f"{float(ow.get('share') if ow.get('share') is not None else 0.3):.0%} of the lights be dimmed",
                      "stop": f"the facility manager of the {who} takes it out of the event",
                      "clear": f"the facility manager of the {who} goes back to the plan"}.get(ow.get("kind"),
                                                                                           f"{who}: {ow.get('kind')}"))
    elif ow:
        who = HOME_NAMES.get(ow.get("home"), ow.get("home"))
        parts.append({"ev_now": f"the owner of {who}: \"I need the car now\"",
                      "comfort": f"the owner of {who} allows {float(ow.get('add_c') or 0):g} °C more",
                      "stop": f"the owner of {who} leaves the event",
                      "clear": f"the owner of {who} goes back to the plan"}.get(ow.get("kind"), f"{who}: {ow.get('kind')}"))
    if not parts:
        return None
    t = "; ".join(parts)
    return {**base, "text": t[:1].upper() + t[1:]}


def protocol_line(x: dict[str, Any], price: float | None) -> dict[str, Any] | None:
    """A coordinator / utility log entry (LiveRecorder.log) -> a chat line, or None for the ones the chat skips."""
    k, who = x.get("kind"), x.get("who")
    if k == "what_if" and who != "utility":
        return what_if_line(x.get("change") or {})

    def L(frm, to, role, text, **kw):
        return {"from": frm, "to": to, "kind": "message", "role": role, "text": text, **kw}
    if who == "utility":
        if k == "need_posted":
            w = x.get("window") or [None, None]
            try:
                hours = (datetime.strptime(w[1], "%Y-%m-%d %H:%M") - datetime.strptime(w[0], "%Y-%m-%d %H:%M")).seconds / 3600
            except (TypeError, ValueError):
                hours = 0.0
            kw, p = x.get("kw"), x.get("price_usd_per_kwh")
            agent = "agent" in str(x.get("buyer", "")).lower()
            kwh = float(kw or 0) * hours
            return {"from": "buyer", "to": None if agent else "coordinator", "kind": "decision" if agent else "message",
                    "role": "need",
                    "text": f"Orders {_kw(kw)} kW less grid import for each half hour, {str(w[0])[11:16]}–{str(w[1])[11:16]} "
                            f"on {_day(w[0])}, at ${float(p or 0):.2f} per kWh",
                    "kw": kw, "price": p, "by": x.get("buyer"),
                    "money": {"usd": -round(kwh * float(p or 0), 2),
                              "what": f"at most, if all {kwh:.0f} kWh are delivered"} if hours else None}
        if k == "on_discover":
            return L("buyer", "coordinator", "need", f"{_day(x.get('date'))}: "
                     + ("the request is on its way (a signed DEG demand-flex need)" if x.get("needs") else "no event that day"))
        if k == "contract_active":
            off = x.get("offered") or {}
            return L("buyer", "coordinator", "commit",
                     f"Contract signed and active: {x.get('meters')} meters enrolled, {_kw(_mean(off))} kW in each half hour",
                     kw=_mean(off), price=price)
        if k == "baselines_sent":
            return L("buyer", "coordinator", "report", f"Baselines for the {x.get('meters')} meters (what each would have "
                                                       f"used without the event)")
        if k == "settled":
            return L("buyer", "coordinator", "money", f"Settlement on the signed meter readings: pays ${float(x.get('seller_usd') or 0):.2f}",
                     money={"usd": -_usd(x.get("seller_usd") or 0), "what": "paid to the community"})
        if k == "buyer_review":
            return None                                         # the buyer agent's sign-off decision says it
        if k == "rt_call":
            what = "Grid emergency" if x.get("call_kind") == "emergency" else "Short-notice call"
            return L("buyer", "coordinator", "rt", f"{what}: {_kw(x.get('kw'))} kW more from {str(x.get('start'))[11:16]} "
                                               f"to {str(x.get('end'))[11:16]} at ${float(x.get('price') or 0):.2f} per kWh "
                                               f"({x.get('why')})", kw=x.get("kw"), price=x.get("price"))
        if k == "what_if_price":
            return None                                         # the presenter's line says it
        if k == "rt_settled":
            return L("buyer", "coordinator", "money", f"Call settled on the meters: {float(x.get('delivered_kwh') or 0):.1f} kWh "
                                                      f"beyond the booking, pays ${float(x.get('paid_usd') or 0):.2f}",
                     money={"usd": -_usd(x.get("paid_usd") or 0), "what": "for the call, to the community"})
        return None
    if k == "v2_discover":
        return L("coordinator", "buyer", "need", f"Asks the utility for {_day(x.get('date'))}'s needs")
    if k == "v2_call_for_offers":
        return L("coordinator", "all", "offer", f"Call for offers to all {x.get('homes')} homes: what can you give, half "
                                                f"hour by half hour?", price=price)
    if k == "v2_cleared":
        cap, raw = x.get("capacity_offered_kw") or {}, x.get("requested_kw") or {}
        sold = sum(float(v) for v in cap.values()) * SLOT_H
        return L("coordinator", "buyer", "commit",
                 f"Market cleared: sells {_kw(_mean(cap))} kW in each half hour (asked {_kw(_mean(raw))}); "
                 f"{x.get('homes')} homes asked for a share, {float(x.get('margin') or 0):.0%} more than sold",
                 kw=_mean(cap), price=price,
                 money={"usd": round(sold * float(price or 0), 2), "what": "for the community, if delivered"}
                 if cap and price else None)
    if k == "v2_commitments_sent":
        return L("coordinator", "all", "commit", f"Commitments sent to {x.get('homes')} homes")
    if k == "v2_final":
        cap = x.get("capacity_offered_kw") or {}
        return L("coordinator", "buyer", "commit", f"Final: {x.get('accepted')} homes accepted, {x.get('declined')} "
                                                   f"declined; {_kw(_mean(cap))} kW in each half hour stands", kw=_mean(cap))
    if k == "v2_event_notice":
        return L("coordinator", "all", "pre", f"Event notice to the {x.get('homes')} committed homes: top up your "
                                              f"batteries now")
    if k == "v2_telemetry_to_utility":
        return L("coordinator", "buyer", "report", f"Readings are in: {x.get('readings')} signed readings from "
                                                   f"{x.get('meters')} homes")
    if k == "v2_settled":
        return L("coordinator", "all", "money", f"Pays ${float(x.get('seller_usd') or 0):.2f} to {x.get('homes_paid')} "
                                                f"homes, pro rata to what each delivered",
                 money={"usd": _usd(x.get("seller_usd")), "what": f"to {x.get('homes_paid')} homes"})
    if k == "rt_call":
        return L("coordinator", "all", "rt", f"Asks every home how much more it can give, {str(x.get('start'))[11:16]}–"
                                             f"{str(x.get('end'))[11:16]}, at ${float(x.get('price') or 0):.2f} per kWh")
    if k == "rt_dispatched":
        if x.get("call_kind") == "gap":
            return L("coordinator", "all", "rt", f"Moves the gap to {x.get('homes')} other homes: {_kw(x.get('confirmed_kw'))} kW covered"
                                                 + (f", {_kw(x.get('short_kw'))} kW still short" if float(x.get("short_kw") or 0) > 0.05 else ""),
                     kw=x.get("confirmed_kw"))
        return L("coordinator", "buyer", "rt", f"{x.get('answers')} homes answered; dispatches {x.get('homes')} of them and "
                                               f"confirms {_kw(x.get('confirmed_kw'))} kW"
                                               + (f" ({_kw(x.get('short_kw'))} kW short)" if float(x.get("short_kw") or 0) > 0.05 else ""),
                 kw=x.get("confirmed_kw"))
    if k == "rt_gap":
        who = node_name(x.get("home"))
        return L(x.get("home"), "coordinator", "rt", f"{who} will be {_kw(x.get('kw'))} kW short from {str(x.get('start'))[11:16]} "
                                                     f"({x.get('reason') or 'its plan changed'})", kw=x.get("kw"))
    if k == "rt_price_up":
        return L("coordinator", "all", "rt", f"Not covered yet: offers the last {_kw(x.get('kw'))} kW again at "
                                             f"${float(x.get('price') or 0):.2f} per kWh")
    if k == "rt_early_warning":
        return L("coordinator", "buyer", "rt", f"Warns the buyer now: {_kw(x.get('short_kw'))} kW short from "
                                               f"{str(x.get('start'))[11:16]} to {str(x.get('end'))[11:16]}")
    if k == "rt_paid":
        return L("coordinator", "all", "money", f"Pays ${float(x.get('usd') or 0):.2f} for the call to {x.get('homes')} homes, "
                                                f"by what each measured beyond its booking",
                 money={"usd": _usd(x.get("usd")), "what": f"to {x.get('homes')} homes"})
    if k == "rt_gap_absorbed":
        return L("coordinator", None, "rt", f"{x.get('homes')} {'home says it' if x.get('homes') == 1 else 'homes say they'} will be "
                                            f"short (up to {_kw(x.get('short_kw'))} kW in all): the margin and what the meters show cover it, nobody "
                                            f"else needs to move")
    if k == "rt_explained":
        return {"from": "coordinator", "to": None, "kind": "message", "role": "rt", "text": x.get("text")}
    if k == "rt_end":
        return L("buyer", "coordinator", "rt", f"The call ends early at {str(x.get('at'))[11:16]}: the rest is released")
    if k == "rt_call_too_late":
        return L("coordinator", "buyer", "rt", f"The call reached the coordinator at {str(x.get('received_at'))[11:16]}, after it "
                                               f"ended ({str(x.get('end'))[11:16]}): 0 kW confirmed, no home asked")
    if k == "v2_no_capacity":
        return L("coordinator", "buyer", "commit", "No capacity to sell for this event")
    if k == "v2_utility_gave_up":
        return L("coordinator", None, "commit", "The utility did not answer before its deadline: no contract")
    return None


def decision_line(d: dict[str, Any], who: str) -> dict[str, Any] | None:
    """A decision from an agent's decision log -> a chat line (to: null: it is the agent's own decision)."""
    k = d.get("kind")
    base = {"from": who, "to": None, "kind": "decision", "reason": d.get("reason") or None, "llm": _llm(d),
            "guard": d.get("guard") or None, "event_id": d.get("event_id"), "price": d.get("price")}
    if k == "offer" and ("cooling_share" in d or d.get("node") == "building"):   # a building node (BuildingBrain)
        kw, safe = d.get("kw") or [], d.get("safe_kw") or []
        if not d.get("take_part"):
            return {**base, "role": "offer", "text": "Sits out this event", "kw": 0.0}
        cs, ls = float(d.get("cooling_share") or 0), float(d.get("lighting_share") or 0)
        ck, lk = _mean(d.get("cooling_kw") or []) or 0.0, _mean(d.get("lighting_kw") or []) or 0.0
        parts = ([f"{'its' if cs >= 0.995 else f'{cs:.0%} of its'} cooling ({ck:.0f} kW)"] if cs > 0 and ck > 0 else []) + \
            ([f"{'its' if ls >= 0.995 else f'{ls:.0%} of its'} dimmable lighting ({lk:.0f} kW)"] if ls > 0 and lk > 0 else [])
        lo, hi = (min(kw), max(kw)) if kw else (0, 0)
        full = all(abs(a - b) < 0.051 for a, b in zip(kw, safe))
        text = (f"Offers {'all of its safe offer, ' if full else ''}{_power(lo, hi)} per half hour "
                f"({_energy(sum(float(x) for x in kw) * SLOT_H)}): {' and '.join(parts) or 'nothing'}"
                + ("" if full else f"; the safe offer was {_energy(sum(safe) * SLOT_H)}"))
        return {**base, "role": "offer", "text": text, "kw": _mean(kw),
                "money": {"usd": _usd(d.get("if_delivered_usd")), "what": "at most, if all of it is used"}}
    if k == "offer" and "pause_share" in d:              # a data center (brain.DataCenterBrain): batch work and battery
        kw, safe = d.get("kw") or [], d.get("safe_kw") or []
        if not d.get("take_part"):
            return {**base, "role": "offer", "text": "Sits out this event", "kw": 0.0}
        ps, bs = float(d.get("pause_share") or 0), float(d.get("battery_share") or 0)
        parts = ([("all its batch work paused" if ps >= 0.995 else f"{ps:.0%} of its batch work paused")] if ps > 0 else []) + \
            ([("all its battery" if bs >= 0.995 else f"{bs:.0%} of its battery")] if bs > 0 else [])
        lo, hi = (min(kw), max(kw)) if kw else (0, 0)
        full = all(abs(a - b) < 0.051 for a, b in zip(kw, safe))
        text = (f"Offers {'all of its safe offer, ' if full else ''}{_power(lo, hi)} per half hour "
                f"({_energy(sum(float(x) for x in kw) * SLOT_H)}): {' and '.join(parts) or 'nothing'}"
                + ("" if full else f"; the safe offer was {_energy(sum(safe) * SLOT_H)}"))
        return {**base, "role": "offer", "text": text, "kw": _mean(kw),
                "money": {"usd": _usd(d.get("if_delivered_usd")), "what": "at most, if all of it is used"}}
    if k == "offer":
        kw, safe = d.get("kw") or [], d.get("safe_kw") or []
        kwh = sum(float(x) for x in kw) * SLOT_H
        if not d.get("take_part"):
            return {**base, "role": "offer", "text": "Sits out this event", "kw": 0.0}
        parts = []
        if d.get("battery_share", 0) > 0:
            parts.append("battery" if d.get("battery_share", 0) >= 0.995 else f"{d['battery_share']:.0%} of the battery")
        if d.get("pause_ev"):
            parts.append("EV pause")
        lo, hi = (min(kw), max(kw)) if kw else (0, 0)
        full = all(abs(a - b) < 0.011 for a, b in zip(kw, safe))
        text = (f"Offers {'all of its safe offer: ' if full else ''}{lo:.1f}–{hi:.1f} kW per half hour ({kwh:.1f} kWh), "
                f"{' and '.join(parts) or 'nothing'}" + ("" if full else f"; the safe offer was {sum(safe) * SLOT_H:.1f} kWh"))
        return {**base, "role": "offer", "text": text, "kw": _mean(kw),
                "money": {"usd": _usd(d.get("if_delivered_usd")), "what": "at most, if all of it is used"}}
    if k == "accept":
        ok = d.get("accept") if "accept" in d else d.get("decision") == "ACCEPTED"
        kw = d.get("kw") or []
        kwh = sum(float(x) for x in kw) * SLOT_H
        return {**base, "role": "commit", "kw": _mean(kw),
                "text": f"Accepts its share ({_energy(kwh)})" if ok else "Declines its share"}
    if k == "margin":
        m = float(d.get("margin") or 0)
        sellers = "data centers" if who == "desk" else "homes"
        return {**base, "role": "commit", "text": f"Holds back {m:.0%}: asks the {sellers} for {1 + m:.2f}× what it sells",
                "money": {"usd": None, "what": f"holds back {m:.0%}: covers {sellers} that deliver less than they offered"}}
    if k == "order" and "mw" in d:                       # the grid operator's AI agent (brain.GridOperatorBrain)
        if not d.get("placed"):
            return {**base, "role": "need", "text": "No order of its own: the program's request stands"
                    + (f" (refused: {str(d['refused']).partition(': ')[2] or d['refused']})" if d.get("refused") else "")}
        mw, price = float(d.get("mw") or 0), float(d.get("price") or 0)
        try:
            h = (int(d["end"][:2]) * 60 + int(d["end"][3:5]) - int(d["start"][:2]) * 60 - int(d["start"][3:5])) / 60
        except (KeyError, TypeError, ValueError):
            h = 0.0
        cat = d.get("catalog") or {}
        was = (f" (the program's request: {float(cat['mw']):g} MW {cat.get('start')}–{cat.get('end')})"
               if cat.get("mw") is not None and (abs(float(cat["mw"]) - mw) > 0.05 or cat.get("start") != d.get("start")
                                                 or cat.get("end") != d.get("end")) else " (as the program's request)")
        return {**base, "role": "need", "kw": mw * 1000,
                "text": f"Orders {mw:g} MW less grid import for each half hour, {d.get('start')}–{d.get('end')} on "
                        f"{_day(d.get('day'))}, at ${price:.2f} per kWh, the window's day-ahead price{was}",
                "money": {"usd": -round(mw * 1000 * h * price, 2), "what": f"at most, if all {mw * h:g} MWh are delivered"}}
    if k == "order":
        if not d.get("placed"):
            return {**base, "role": "need", "text": "No order of its own: the program's request stands"}
        return None                                          # the order itself is the utility's need_posted line
    if k == "sign_off":
        dec = d.get("decision")
        paid = (f" (${float(d['usd']):,.2f} for {_energy(float(d['paid_kwh']))})" if d.get("usd") and d.get("paid_kwh")
                else f" (${float(d['usd']):.2f})" if d.get("usd") else "")
        return {**base, "role": "money", "text": ("Signs off the settlement" if dec == "approved" else
                                                 "Disputes the settlement" if dec == "disputed" else "Could not sign off")
                + paid + (f": failed {', '.join(d['failed'])}" if dec == "disputed" and d.get("failed") else "")}
    return None


def _power(lo: float, hi: float) -> str:
    """kW, or MW for a data center's."""
    return f"{lo / 1000:.1f}–{hi / 1000:.1f} MW" if hi >= 1000 else f"{lo:.1f}–{hi:.1f} kW"


def _energy(kwh: float) -> str:
    return f"{kwh / 1000:.1f} MWh" if kwh >= 1000 else f"{kwh:.1f} kWh"


def decision_text(d: dict[str, Any]) -> str | None:
    """The one-line text of a home's decision (the phone app's decisions list)."""
    ln = decision_line(d, "home")
    return ln["text"] if ln else None


def market_lines(ev: dict[str, Any] | None, seen: dict[tuple, tuple[float, str]], phones_v2: dict[str, dict[str, Any]],
                 jetsons: tuple[str, ...] = JETSONS) -> list[dict[str, Any]]:
    """The Jetson homes' (and the building nodes', `jetsons` = every node shown one by one) own messages as the market
    recorded them (their offer, answer, signed readings), the
    simulated homes' in one line per step, and each Jetson home's commitment and payment. Stamped with when the live
    recorder first saw them (`seen`: key -> (wall, sim), kept by LiveRecorder; a key starts with the event id, as
    market_keys gives it, so two events' lines never share a stamp)."""
    if not ev:
        return []
    n = ev["need"]
    price = float(n["slots"][0].get("price") or 0) if n.get("slots") else 0.0
    out = []

    def add(key, frm, to, role, text, **kw):
        key = (n["event_id"],) + key
        if key in seen:
            w, s = seen[key]
            out.append({"wall": w, "sim": s, "from": frm, "to": to, "kind": "message", "role": role, "text": text,
                        "event_id": n["event_id"], **kw})
    offers = ev.get("offers") or {}
    for h in jetsons:
        o = offers.get(h)
        if o:
            add(("offer", h), h, "coordinator", "offer",
                f"Offer: {_kws([s['kw_offered'] for s in o['slots']])} kW, confidence {float(o.get('confidence') or 0):.2f}",
                kw=_mean([s["kw_offered"] for s in o["slots"]]))
    fleet = {h: o for h, o in offers.items() if h not in jetsons}
    if fleet:
        tot = [sum(s["kw_offered"] for s in o["slots"]) / max(1, len(o["slots"])) for o in fleet.values()]
        add(("fleet_offers",), "fleet", "coordinator", "offer",
            f"{len(fleet)} offers from the simulated homes (rules, no LLM): {min(tot):.0f}–{max(tot):.0f} kW each")
    cms = ev.get("commitments") or {}
    for h in jetsons:
        cm = cms.get(h)
        if cm:
            kw = [x["kw"] for x in (cm.get("contract") or {}).get("commitments", [])]
            add(("share", h), "coordinator", h, "commit", f"Your share: {_kws(kw)} kW",
                money={"usd": round(sum(kw) * SLOT_H * price, 2), "what": "if delivered"}, kw=_mean(kw))
            if cm.get("status") in ("ACCEPTED", "DECLINED", "LATE"):
                add(("answer", h), h, "coordinator", "commit",
                    "Accepts the commitment (signed)" if cm["status"] == "ACCEPTED" else
                    f"Declines the commitment ({cm.get('reason') or 'no reason given'})")
    fcm = {h: c for h, c in cms.items() if h not in jetsons}
    if fcm:
        add(("fleet_commit",), "coordinator", "fleet", "commit", f"Commitments to {len(fcm)} simulated homes")
        if ev.get("finalized"):
            acc = sum(1 for c in fcm.values() if c.get("status") == "ACCEPTED")
            add(("fleet_answer",), "fleet", "coordinator", "commit",
                f"{acc} of {len(fcm)} simulated homes accept (rules: within each offer)")
    rd = ev.get("readings") or {}
    for h in jetsons:
        for sid, r in sorted((rd.get(h) or {}).items(), key=lambda kv: int(kv[0])):
            t0 = str(r.get("start", ""))[11:16]
            add(("reading", h, int(sid)), h, "coordinator", "report",
                f"Signed meter reading, {t0}–{beckn.add_minutes(r['start'], 30)[11:16] if r.get('start') else ''}: "
                f"committed {float(r.get('committed_kw') or 0):.2f} kW less; meter at {float(r.get('usage_kw') or 0):.2f} kW"
                + (", EV waiting" if r.get("ev_held") else ""))
    fr = sum(len(v) for h, v in rd.items() if h not in jetsons)
    if fr and ev.get("telemetry_sent"):
        add(("fleet_readings",), "fleet", "coordinator", "report", f"{fr} signed readings from the simulated homes")
    split = ((ev.get("result") or {}).get("split") or {}).get("per_home") or {}
    for h in jetsons:
        pay = next((p for p in (phones_v2.get(h) or {}).get("payments") or [] if p.get("event_id") == n["event_id"]), None)
        if pay:
            add(("paid", h), "coordinator", h, "money",
                f"Payment: ${float(pay.get('payment') or 0):.2f} for {float(pay.get('delivered_kwh') or 0):.1f} kWh measured",
                money={"usd": _usd(pay.get("payment")), "what": "to this home"})
    fp = [v for h, v in split.items() if h not in jetsons]
    if fp:
        add(("fleet_paid",), "coordinator", "fleet", "money",
            f"Payments to {len(fp)} simulated homes: ${sum(fp):.2f} (median ${statistics.median(fp):.2f})",
            money={"usd": round(sum(fp), 2), "what": f"to {len(fp)} homes"})
    return out


def market_keys(ev: dict[str, Any] | None, phones_v2: dict[str, dict[str, Any]], jetsons: tuple[str, ...] = JETSONS) -> list[tuple]:
    """Every key market_lines can show now (the recorder stamps the new ones with the time it first saw them), each
    starting with the event id."""
    if not ev:
        return []
    return [(ev["need"]["event_id"],) + k for k in _market_keys(ev, phones_v2, jetsons)]


def _market_keys(ev: dict[str, Any], phones_v2: dict[str, dict[str, Any]], jetsons: tuple[str, ...]) -> list[tuple]:
    keys: list[tuple] = []
    offers, cms, rd = ev.get("offers") or {}, ev.get("commitments") or {}, ev.get("readings") or {}
    keys += [("offer", h) for h in jetsons if h in offers]
    if any(h not in jetsons for h in offers) and (ev.get("phase") != "offers" or len(offers) >= 100):
        keys.append(("fleet_offers",))
    for h in jetsons:
        if h in cms:
            keys.append(("share", h))
            if cms[h].get("status") in ("ACCEPTED", "DECLINED", "LATE"):
                keys.append(("answer", h))
        keys += [("reading", h, int(s)) for s in (rd.get(h) or {})]
    if any(h not in jetsons for h in cms):
        keys.append(("fleet_commit",))
        if ev.get("finalized"):
            keys.append(("fleet_answer",))
    if ev.get("telemetry_sent"):
        keys.append(("fleet_readings",))
    eid = ev["need"]["event_id"]
    keys += [("paid", h) for h in jetsons if any(p.get("event_id") == eid for p in (phones_v2.get(h) or {}).get("payments") or [])]
    if (ev.get("result") or {}).get("split"):
        keys.append(("fleet_paid",))
    return keys


def build(log: list[dict[str, Any]], home_states: dict[str, dict[str, Any]], ev: dict[str, Any] | None,
          coord: dict[str, Any] | None, buyer: dict[str, Any] | None, phones_v2: dict[str, dict[str, Any]],
          cfg: dict[str, Any], seen: dict[tuple, tuple[float, str]] | None = None,
          events: list[dict[str, Any]] | None = None, dc: dict[str, Any] | None = None,
          buildings: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The "agents" block. log: LiveRecorder.chat_log (coordinator + utility, this run); home_states: each Jetson's
    /agent/state (decisions and owner chat, this run); coord/buyer: their brains' state; seen: market_lines' stamps;
    events: every event of the run whose market lines belong in the conversation (default: `ev` only); dc: the data
    centers' record (flexgrid-dc's GET /dc/state, as the monitor last read it): their market's steps, the AI site's
    decisions and the large-load desk's margins join the conversation, and their agents the roster; buildings: the
    building nodes in the community's market (building.py), each with its own daemon and AI agent like a Jetson
    home's (home_states holds theirs too). The money strip and the roster are about `ev`, the event being shown
    ("money_dc": the data centers' event of the same day)."""
    own = JETSONS + tuple(b["name"] for b in buildings or [])    # the nodes whose own messages show one by one
    n = (ev or {}).get("need") or {}
    price = (n.get("slots") or [{}])[0].get("price") if n else None
    eid = n.get("event_id")
    lines: list[dict[str, Any]] = []
    for x in log:
        ln = _agent_decision_line(x) if x.get("kind") == "agent_decision" else protocol_line(x, price)
        if ln:
            ln.setdefault("event_id", x.get("event_id") or eid)
            lines.append({"wall": x.get("wall"), "sim": x.get("sim"), **ln})
    for home, st in home_states.items():
        for d in st.get("decisions") or []:
            ln = decision_line(d, home)
            if ln:
                lines.append({"wall": d.get("wall"), "sim": d.get("sim"), **ln})
        for m in st.get("chat") or []:
            owner = m.get("role") == "owner"
            lines.append({"wall": m.get("wall"), "sim": m.get("sim"), "kind": "owner", "role": None, "text": m.get("text"),
                          "from": f"owner:{home}" if owner else home, "to": home if owner else f"owner:{home}",
                          "llm": None if owner else {"model": model_label(m.get("model")) or model_label(mac_model()),
                                                     "ms": m.get("ms"), "fallback": bool(m.get("error"))},
                          "tools": None if owner else [t.get("tool") for t in m.get("trace") or []] or None})
    if seen is not None:
        for e in (events if events is not None else [ev]):
            lines += market_lines(e, seen, phones_v2, own)
    if isinstance(dc, dict):
        lines += dc_lines(dc, seen)
    lines = [x for x in lines if x.get("text")]
    lines.sort(key=lambda x: (x.get("wall") or 0.0))
    for i, x in enumerate(lines):
        x["id"] = f"c-{i + 1}"
        x.setdefault("event_id", eid)
        for k in ("kw", "price", "money", "reason", "llm", "to", "role"):
            x.setdefault(k, None)
    label = model_label(mac_model())
    roster = [{"id": "buyer", "kind": "buyer", "name": "Grid buyer", "org": "Utility (mock ENGIE)", "host": "the Mac",
               "llm": bool(buyer), "runtime": _rt(buyer), "model": label if buyer else None,
               **({"state": "thinking"} if (buyer or {}).get("busy") else {})},
              {"id": "coordinator", "kind": "coordinator", "name": "Community coordinator", "host": "the Mac",
               "llm": bool(coord), "runtime": _rt(coord), "model": label if coord else None,
               **({"state": "thinking"} if ev and ev.get("margin_asked") and "margin" not in ev else {})}]
    for home in JETSONS:
        st = home_states.get(home) or {}
        roster.append({"id": home, "kind": "home", "name": HOME_NAMES[home], "host": HOME_HOSTS[home],
                       "llm": bool(st.get("enabled")), "runtime": _rt(st if st.get("enabled") else None),
                       "model": label if st.get("enabled") else None,
                       "wishes": st.get("wishes") if st.get("enabled") else None})
    for b in buildings or []:                          # a building node: its agent on its host, the Mac's model
        st = home_states.get(b["name"]) or {}
        roster.append({"id": b["name"], "kind": "building", "name": b.get("short") or b["name"], "label": b.get("label"),
                       "host": b.get("host"), "llm": bool(st.get("enabled")),
                       "runtime": _rt(st if st.get("enabled") else None), "model": label if st.get("enabled") else None,
                       "wishes": st.get("wishes") if st.get("enabled") else None})
    roster.append({"id": "fleet", "kind": "home", "name": "98 simulated homes", "host": "jetson-b", "llm": False,
                   "runtime": None, "model": None})
    for r in roster:
        r["market"] = "homes"
    out_dc = None
    if isinstance(dc, dict):
        roster += dc_roster(dc, label)
        out_dc = dc_money(dc)
    st = (coord or {}).get("llm") or next((h.get("llm") for h in home_states.values() if h.get("llm")), None) or {}
    llm = {"ok": bool(st.get("ok")), "model": label, "where": "the Mac",
           "runtime": {"openclaw": "OpenClaw", "ollama": "Ollama"}.get(st.get("runtime") or "", st.get("runtime")),
           "median_ms": st.get("median_ms"), "calls": st.get("calls"), "error": st.get("error")}
    return {"llm": llm, "roster": roster, "chat": lines[-300:], "money": money(ev), "money_dc": out_dc}


def _rt(st: dict[str, Any] | None) -> str | None:
    if not st:
        return None
    rt = ((st.get("llm") or {}).get("runtime"))
    return {"openclaw": "OpenClaw", "ollama": "Ollama"}.get(rt or "", "OpenClaw")


def _agent_decision_line(x: dict[str, Any]) -> dict[str, Any] | None:
    """A coordinator log entry `agent_decision` (what = margin: the coordinator's; order / sign_off: the buyer's).
    The homes' decisions come from their own daemons (decision_line), not from this log."""
    k = x.get("what")
    if k == "explain":                                   # the coordinator's AI agent explains a call afterwards
        llm = None if x.get("fallback") and not x.get("runtime") else \
            {"model": model_label(mac_model()), "ms": x.get("ms"), "runtime": x.get("runtime"),
             "fallback": bool(x.get("fallback"))}
        return {"from": "coordinator", "to": None, "kind": "decision", "role": "rt", "text": x.get("reason"),
                "llm": llm, "reason": None, "event_id": x.get("event_id")}
    if k not in ("margin", "order", "sign_off"):
        return None
    who = "coordinator" if k == "margin" else "buyer"
    llm = None if x.get("fallback") and not x.get("runtime") else \
        {"model": mac_model(), "ms": x.get("ms"), "runtime": x.get("runtime")}
    return decision_line({**x, "kind": k, "llm": llm or {}}, who)


def money(ev: dict[str, Any] | None) -> dict[str, Any] | None:
    """The event so far: price, kW asked / offered / sold (a half hour, mean over the window), the buyer's payment and
    the homes' earnings once settled."""
    if not ev:
        return None
    n = ev["need"]
    sl = n.get("slots") or []
    sids = [s["id"] for s in sl]
    offered = None
    if ev.get("offers"):
        tot = {sid: 0.0 for sid in sids}
        for o in ev["offers"].values():
            for s in o.get("slots") or []:
                if s["id"] in tot:
                    tot[s["id"]] += float(s.get("kw_offered") or 0)
        offered = _mean(tot)
    cap = ev.get("capacity_offered_final") or ev.get("capacity_offered") or {}
    sold = _mean(cap) if cap else None
    res = ev.get("result") or {}
    split = (res.get("split") or {}).get("per_home") or {}
    price = sl[0].get("price") if sl else None
    return {"event_id": n.get("event_id"), "price": price,
            "kw_asked": _mean([s.get("kw_requested") for s in sl]), "kw_offered": offered, "kw_sold": sold,
            "margin": ev.get("margin"), "up_to_usd": round(sum(cap.values()) * SLOT_H * float(price or 0), 2) if cap else None,
            "utility_pays_usd": res.get("seller_usd"),
            "homes_earned_usd": round(sum(split.values()), 2) if split else None,
            "homes_paid": len(split) if split else None}


# --------------------------------------------------------------------------------------------- the data centers
DC_SITES_HOST = "jetson-a"


def _mw(kw: Any) -> str:
    try:
        return f"{float(kw) / 1000:.1f} MW"
    except (TypeError, ValueError):
        return "?"


def _mean_list(v: list[Any] | None) -> float | None:
    vals = [float(x) for x in v or [] if x is not None]
    return sum(vals) / len(vals) if vals else None


def _dc_ai_sites(dc: dict[str, Any]) -> list[str]:
    return sorted(((dc.get("agents") or {}).get("sites") or {}))


def _dc_steps(dc: dict[str, Any] | None) -> list[tuple[tuple, dict[str, Any]]]:
    """(key, line) for each step the data centers' market has taken, as flexgrid-dc's record shows it: every event the
    large-load desk holds (the day being shown in detail: its sites' offers, shares and payments). A key starts with
    "dc" and the event id; the recorder stamps a key when it first sees it (dc_keys)."""
    if not isinstance(dc, dict):
        return []
    det = dc.get("event") if isinstance(dc.get("event"), dict) else None
    ai = _dc_ai_sites(dc)
    n_sites = len((det or {}).get("sites") or {}) or 10
    out: list[tuple[tuple, dict[str, Any]]] = []
    evs = dc.get("events") if isinstance(dc.get("events"), dict) else {}
    if det and det.get("id") not in evs:                    # a record without the desk's summary: the day shown only
        evs = {**evs, det["id"]: {"phase": det.get("phase"), "window": det.get("window"), "offers": det.get("offers"),
                                  "accepted": det.get("accepted"), "declined": det.get("declined")}}
    for eid, sm in sorted(evs.items(), key=lambda kv: str((kv[1].get("window") or [""])[0])):
        d = det if det and det.get("id") == eid else None
        w = sm.get("window") or (d or {}).get("window") or [None, None]
        price = sm.get("price") if sm.get("price") is not None else (d or {}).get("price_usd_per_kwh")
        kwr = [x for x in (sm.get("kw_requested") or [s.get("kw_requested") for s in (d or {}).get("slots") or []])
               if x is not None]
        res = sm.get("result") or {}
        ph = sm.get("phase")

        def L(step: tuple, frm: str, to: str | None, role: str, text: str, **kw) -> None:
            out.append((("dc", eid) + step, {"from": frm, "to": to, "kind": "message", "role": role, "text": text,
                                             "event_id": eid, "market": "dc", **kw}))
        if w[0]:
            kwh = sum(kwr) * SLOT_H
            L(("request",), "grid-op", "desk", "need",
              (f"Asks the data centers for {_mw(_mean_list(kwr))} less grid import" if kwr else
               "Asks the data centers to draw less from the grid") +
              f" for each half hour, {str(w[0])[11:16]}–{str(w[1])[11:16]} on {_day(w[0])}"
              + (f", at ${float(price):.2f} per kWh" if price else "")
              + ((" (its AI agent's order, signed)" if "agent" in str(((d or {}).get("order") or {}).get("by"))
                  else " (the presenter's request, signed)") if (d or {}).get("order") else ""),
              kw=_mean_list(kwr), price=price,
              money={"usd": -round(kwh * float(price), 2), "what": f"at most, if all {_energy(kwh)} are delivered"}
              if price and kwr else None)
            L(("call",), "desk", "datacenters", "offer",
              f"Call for offers for {_day(w[0])} to all {n_sites} data centers: batch work they can pause and battery "
              f"above the backup reserve, half hour by half hour", price=price)
        offers = int(sm.get("offers") or 0)
        if offers and (offers >= n_sites or ph not in (None, "offers")):
            tot = None
            if d:
                per = [v.get("offer_kw") or [] for v in (d.get("sites") or {}).values()]
                k = max((len(x) for x in per), default=0)
                tot = _mean_list([sum(float(x[i] or 0) for x in per if i < len(x)) for i in range(k)]) if k else None
            rules = max(0, offers - len(ai))                 # an AI site always answers (0 kW if it sits out)
            L(("offers",), "datacenters", "desk", "offer",
              f"{offers} offers for {_day(w[0])} from the data centers"
              + (f": {_mw(tot)} in each half hour in all" if tot is not None else "")
              + (f" ({rules} by their controllers' rules, no LLM)" if ai and rules > 0 else ""), kw=tot)
        cap = sm.get("capacity_offered_kw") or dict(enumerate((d or {}).get("capacity_kw") or []))
        if cap:
            vals = [float(v) for v in cap.values() if v is not None]
            sold, m = _mean_list(vals), sm.get("margin")
            L(("cleared",), "desk", "grid-op", "commit",
              f"Market cleared for {_day(w[0])}: sells {_mw(sold)} in each half hour"
              + (f" (asked {_mw(_mean_list(kwr))})" if kwr else "")
              + (f"; asks the data centers for {1 + float(m):.2f}× that" if m is not None else ""),
              kw=sold, price=price,
              money={"usd": round(sum(vals) * SLOT_H * float(price), 2), "what": "for the data centers, if delivered"}
              if price else None)
        committed = int(sm.get("committed") or 0) or len([1 for v in ((d or {}).get("sites") or {}).values()
                                                           if v.get("status")])
        acc, dec = int(sm.get("accepted") or 0), int(sm.get("declined") or 0)
        if d and cap:
            for a in ai:
                st = (d.get("sites") or {}).get(a) or {}
                ck = [float(x) for x in st.get("commit_kw") or []]
                if ck:
                    L(("share", a), "desk", a, "commit", f"Your share: {' · '.join(_mw(x) for x in ck)}",
                      kw=_mean_list(ck), money={"usd": round(sum(ck) * SLOT_H * float(price or 0), 2), "what": "if delivered"})
        if committed and acc + dec >= committed:
            L(("answers",), "datacenters", "desk", "commit",
              f"{acc} of {committed} data centers accept their commitments for {_day(w[0])}"
              + (f", {dec} decline" if dec else ""))
        split = (res.get("split") or {}).get("per_home") or {}
        if res.get("seller_usd") is not None:
            L(("settled",), "grid-op", "desk", "money",
              f"Settlement for {_day(w[0])} on the signed meter readings: pays ${float(res['seller_usd']):,.2f}"
              + (f" for {_energy(float(res['delivered_kwh_measured']))} measured" if res.get("delivered_kwh_measured") else ""),
              money={"usd": -_usd(res["seller_usd"]), "what": "paid for the data centers' flexibility"})
            if split:
                L(("paid",), "desk", "datacenters", "money",
                  f"Pays ${sum(split.values()):,.2f} to {len(split)} data centers, pro rata to what each delivered",
                  money={"usd": round(sum(split.values()), 2), "what": f"to {len(split)} data centers"})
                for a in ai:
                    if a in split:
                        L(("paid", a), "desk", a, "money", f"Payment: ${float(split[a]):,.2f}",
                          money={"usd": _usd(split[a]), "what": "to this data center"})
    return out


def dc_keys(dc: dict[str, Any] | None) -> list[tuple]:
    """Every data center market line dc_lines can show now (the recorder stamps the new ones)."""
    return [k for k, _ in _dc_steps(dc)]


def dc_lines(dc: dict[str, Any], seen: dict[tuple, tuple[float, str]] | None) -> list[dict[str, Any]]:
    """The data centers' part of the conversation: their market's steps (stamped by the recorder: `seen`), the AI
    sites' decisions (brain.DataCenterBrain, on the campus) and the large-load desk's margins (the coordinator's AI
    agent, brain.CoordinatorBrain with DESK_DECIDE)."""
    lines = []
    for key, ln in _dc_steps(dc):
        if seen is not None and key in seen:
            w, sim = seen[key]
            lines.append({"wall": w, "sim": sim, **ln})
    ag = dc.get("agents") or {}
    for name, st in (ag.get("sites") or {}).items():
        for d in st.get("decisions") or []:
            ln = decision_line(d, name)
            if ln:
                lines.append({"wall": d.get("wall"), "sim": d.get("sim"), "market": "dc", **ln})
    for d in (ag.get("desk") or {}).get("decisions") or []:
        ln = decision_line(d, "desk")
        if ln:
            lines.append({"wall": d.get("wall"), "sim": d.get("sim"), "market": "dc", **ln})
    for d in (ag.get("operator") or {}).get("decisions") or []:
        ln = decision_line(d, "grid-op")
        if ln:
            lines.append({"wall": d.get("wall"), "sim": d.get("sim"), "market": "dc", **ln})
    return lines


def dc_roster(dc: dict[str, Any], label: str | None) -> list[dict[str, Any]]:
    """The data centers' cast: the grid operator (rules), the large-load desk (the coordinator's AI agent picks its
    margin), each site with an AI agent and the rest of the sites (their controllers' rules)."""
    ag = dc.get("agents") or {}
    sites, desk, op = ag.get("sites") or {}, ag.get("desk"), ag.get("operator")
    det = dc.get("event") if isinstance(dc.get("event"), dict) else {}
    n_all = len(det.get("sites") or {}) or 10
    where = dc.get("where") or {}

    def host(name: str) -> str:
        return (where.get(name) or DC_SITES_HOST) if dc.get("role") == "market" else "the Mac"
    thinking = any((v or {}).get("margin_thinking") for v in (dc.get("events") or {}).values()
                   if isinstance(v, dict))
    out = [{"id": "grid-op", "kind": "buyer", "name": "Grid operator", "org": "Large-load program (mock)",
            "host": "the Mac", "llm": bool(op), "runtime": _rt(op), "model": label if op else None, "market": "dc",
            **({"state": "thinking"} if op and op.get("busy") else {})},
           {"id": "desk", "kind": "desk", "name": "Large-load desk", "org": "The coordinator's market for data centers",
            "host": "the Mac", "llm": bool(desk), "runtime": _rt(desk), "model": label if desk else None, "market": "dc",
            **({"state": "thinking"} if desk and thinking else {})}]
    for name, st in sorted(sites.items()):
        on = bool(st.get("enabled"))
        out.append({"id": name, "kind": "datacenter", "name": f"Data center {name}", "host": host(name), "llm": on,
                    "runtime": _rt(st if on else None), "model": label if on else None, "market": "dc",
                    "policy": st.get("policy"), **({"state": "thinking"} if st.get("thinking") else {})})
    if n_all - len(sites) > 0:
        out.append({"id": "dc-fleet", "kind": "datacenter", "name": f"{n_all - len(sites)} data centers",
                    "host": host(next(iter(sorted(det.get("sites") or {})), "")), "llm": False, "runtime": None,
                    "model": None, "market": "dc"})
    return out


def dc_money(dc: dict[str, Any]) -> dict[str, Any] | None:
    """The data centers' event of the day shown: price, MW asked / offered / sold (a half hour, mean over the window),
    the desk's margin, what the grid operator pays and what the data centers earn."""
    d = dc.get("event") if isinstance(dc.get("event"), dict) else None
    if not d:
        return None
    sm = (dc.get("events") or {}).get(d.get("id")) or {}
    sl = d.get("slots") or []
    per = [v.get("offer_kw") or [] for v in (d.get("sites") or {}).values()]
    offered = _mean_list([sum(float(x[i] or 0) for x in per if i < len(x)) for i in range(len(sl))]) \
        if any(per) else None
    cap = [float(x) for x in d.get("capacity_kw") or [] if x is not None]
    price = d.get("price_usd_per_kwh")
    paid = [v.get("paid_usd") for v in (d.get("sites") or {}).values() if v.get("paid_usd")]
    return {"event_id": d.get("id"), "price": price, "kw_asked": _mean_list([s.get("kw_requested") for s in sl]),
            "kw_offered": round(offered, 1) if offered is not None else None,
            "kw_sold": round(_mean_list(cap), 1) if cap else None, "margin": sm.get("margin"),
            "up_to_usd": round(sum(cap) * SLOT_H * float(price or 0), 2) if cap else None,
            "operator_pays_usd": d.get("seller_usd"), "sites_earned_usd": round(sum(paid), 2) if paid else None,
            "sites_paid": len(paid) or None, "delivered_kwh_true": d.get("delivered_kwh_true"),
            "promised_kwh": d.get("promised_kwh")}
