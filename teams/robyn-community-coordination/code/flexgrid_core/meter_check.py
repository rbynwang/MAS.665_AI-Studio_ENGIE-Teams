"""What a meter reading may say before money moves on it.

The buyer pays on what each node says it used, and a signature proves who spoke, not that the number is true. So every number is held here against what the data says the
node is (its enrolment), never against what it says about itself. A reading gets one of three answers:

  refused   it can't be read, or it can't be true: not a real number, no such slot or contract, more than the meter can send out or take in. It never reaches the money.
            It's counted and logged, its slot isn't asked for again, and the node is paid nothing for that half hour unless a true reading comes
  capped    it may be true, but not all of it can count (see below). What's cut is shown on the settlement
  repeated  the first reading of a slot stands, and every repeat is held to it. The same numbers again change nothing. Different numbers are refused: no seller gets a second try
            after seeing the settlement. A repeat inside the tolerance is taken and not kept, or thousands of small steps could carry a reading anywhere

What can count, and why a home's limit isn't a multiple of its commitment as a building's is. Across the debug days an honest home showed up to 3.5 times its commitment at the
full ask, and over 100 times at a small one: its battery runs on past what was sold, and that's paid. So the limits are physical, and an honest home can sit right on the first:
  meter     a meter can't send out more than its export limit, and it can't take in more than its baseline and its nameplate. A home at its export limit is honest
  energy    over a window a home can give what it stored (battery and two-way car, above their reserve), its solar, and its own usual load going to zero, and no more
  nameplate in one half hour nothing counts past all it could shed or give at once
  battery   what's paid at the battery rate stops at the battery and the car's power, at what they stored, and at what the meter delivered in that half hour
A building counts at most 1.2 x its commitment, as it always has (settlement.building_counted_kwh), and that commitment is now the coordinator's, not the reading's.
fix_notes/fix_money.md has the honest margins to each limit.
"""
from __future__ import annotations

import csv
import json
import logging
import math
from functools import lru_cache
from typing import Any, Callable

from . import beckn, config
from . import settlement as st

TOL_KW = 0.02             # readings carry three decimals, and a meter at its limit reads its limit
SOLAR_SLACK = 1.25        # today can be sunnier than the best of the days just gone
ENERGY_SLACK_KWH = 1.0
TOP = 1e9                 # past anything a sale can say in kW, kWh or dollars
BATTERY_SLACK = 1.1       # solar can put some back into a battery inside the window. Honest homes reach 0.97 of what was stored
SLOT_H = beckn.SLOT_MINUTES / 60
MONEY_FIELDS = ("usage_kw", "battery_kw")      # what a repeat may not change. A commitment isn't the meter's: it's held to the coordinator's numbers when it's read
_FACTS: dict[str, dict[str, Any]] = {}
log = logging.getLogger("flexgrid-meters")


def real(v: Any) -> bool:
    """A real number. Text is not one, even text float() would take: a node signed "nan" on 6 Nov and it took a buyer's whole bill with it."""
    try:
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    except OverflowError:                                  # JSON's integers have no top
        return False


def sane(v: Any) -> bool:
    """A real number a sum of them can't take past a float's top: two of 1e308 add up to infinity, a payment of NaN goes nowhere, and Town can't carry either."""
    return real(v) and abs(v) <= TOP


# ------------------------------------------------------------------ what the data says a node is
@lru_cache(maxsize=1)
def _enrolled() -> dict[str, dict[str, Any]]:
    from . import building
    out = {h["name"]: h for h in json.loads(config.data_path("fleet_v2.json").read_text()).get("homes", [])}
    return {**out, **{b["name"]: {**b, "kind": "building"} for b in building.registry()}}


def _car_kw(profile: str) -> float:
    """The home's car charger: its first session's, as the node's own adapter takes it."""
    name = config.dataset_meta().get("profile_alias", {}).get(profile, profile)
    try:
        with open(config.data_path(f"{name}_ev.csv" if "/" in name else f"ev_{name}.csv")) as f:
            return next((float(r["max_kw"]) for r in csv.DictReader(f)), 0.0)
    except (OSError, ValueError, KeyError):
        return 0.0


def _owner_car_kwh(name: str, default: float) -> float:
    return float(_owners().get(name, {}).get("v2h_kwh", default))


@lru_cache(maxsize=1)
def _owners() -> dict[str, dict[str, Any]]:
    try:
        return json.loads(config.data_path("owners.json").read_text())["homes"]
    except (OSError, ValueError, KeyError):
        return {}


def _stored(b: dict[str, Any]) -> float:
    """kWh a battery can give from full, above its reserve."""
    return float(b.get("kwh", 0.0)) * max(0.0, float(b.get("soc_max_pct", 100.0)) - float(b.get("reserve_pct", 0.0))) / 100.0 * float(b.get("eta", 0.95))


def _home(h: dict[str, Any]) -> dict[str, Any]:
    from . import devices, hvac
    name, b = h["name"], h.get("battery") or {}
    power = float(b.get("kw", 0.0))
    kw = power + (_car_kw(h["profile"]) if h.get("has_ev", True) else 0.0)
    p = hvac.params_for(name)
    kw += float(p["cool_cap_kw"]) / 2.0 if p else 0.0              # kW of heat removed, at a low COP
    spec, arr, row = devices.for_node("home", name)
    dev = {k: float(arr[k][row].max()) for k in spec}
    car = dev.get("v2h", 0.0)
    rated_kw = round(kw + sum(dev.values()), 1)
    return {"kind": "home", "rated_kw": rated_kw, "intake_kw": rated_kw - car, "export_kw": float(h["export_limit_kw"]), "battery_kw": power + car,
            "stored_kwh": _stored(b) + (_owner_car_kwh(name, float(spec["v2h"].get("event_kwh", 0.0))) if car > 0 else 0.0)}


def _building(b: dict[str, Any]) -> dict[str, Any]:
    from . import building, devices
    p = building.params(b["name"])
    bat = p.get("battery") or {}
    kw = float(bat.get("kw", 0.0)) + float(p["thermal"]["cool_cap_kw"]) / 2.0
    try:
        cat = devices.catalog()
    except (FileNotFoundError, OSError, ValueError):       # a dataset without the devices: as SimulatedBuilding._own_devices
        cat = {}
    row = (cat.get("rows", {}).get("building") or {}).get(b["name"])
    if row is not None:
        kw += sum(float(devices.array("building", k)[row].max()) for k in cat["devices"].get("building") or {})
    kw += max(building._light_series(b["profile"]).values(), default=0.0)
    return {"kind": "building", "rated_kw": round(kw, 1), "intake_kw": round(kw, 1), "export_kw": float(p.get("export_limit_kw", 5000.0 if float(bat.get("kwh", 0.0)) > 0 else 2000.0)),
            "battery_kw": float(bat.get("kw", 0.0)), "stored_kwh": _stored(bat)}


def facts(name: str) -> dict[str, Any] | None:
    """A node's nameplate, what it can take in above its usual load (a two-way car charges on the charger it already counts), its export limit, the power and energy it can
    give from storage. From the data. None for a name the data doesn't hold (a data center on the old path, a test's stand-in): its readings are checked as numbers only. A
    pooled coordinator refuses such a node's offers (market.rated), so it never reaches a sale."""
    if name in _FACTS:
        return _FACTS[name]
    try:
        e = _enrolled().get(name)
        if not e:
            return None
        _FACTS[name] = _building(e) if e.get("kind") == "building" else _home(e)
        return _FACTS[name]
    except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:     # not kept: the next reading asks again
        log.error("no enrolment facts for %s: %s", name, str(exc)[:200])
        return None


def rated(names: list[str]) -> dict[str, float]:
    """The live coordinator's market.rated: each member's nameplate from its enrolment. tools/sim_event.py reads the same number off its adapters."""
    return {n: f["rated_kw"] for n in names if (f := facts(n))}


def own_kind(o: Any, sender: str) -> Any:
    """A signed offer says "node": "building", and a building is paid at another rate and capped differently, so what it is comes from its enrolment, not its offer."""
    f = facts(sender)
    if f is None or not isinstance(o, dict):
        return o
    return {**o, "node": "building"} if f["kind"] == "building" else {k: v for k, v in o.items() if k != "node"}


# ------------------------------------------------------------------ one reading
def slot_of(r: dict[str, Any]) -> int | None:
    s = r.get("slot_id")
    return s if isinstance(s, int) and not isinstance(s, bool) else None


def _base_of(baselines: Any, home: str, sid: Any) -> Any:
    return (((baselines or {}).get(home) or {}).get(str(sid)) or {}).get("BASELINE")


def value_problem(r: dict[str, Any], fact: dict[str, Any] | None, base: Any = None) -> str | None:
    """Why these numbers can't be true, or None."""
    u = r.get("usage_kw")
    if not real(u):
        return "its usage isn't a real number"
    for k in ("committed_kw", "battery_kw"):
        if r.get(k) is not None and not real(r[k]):
            return f"its {k} isn't a real number"
    if real(r.get("committed_kw")) and r["committed_kw"] < 0:
        return "it committed less than nothing"
    if fact:
        if u < -fact["export_kw"] - TOL_KW:
            return f"{-u:g} kW going out is past its meter's {fact['export_kw']:g} kW"
        top = max(float(base), 0.0) if real(base) else fact["rated_kw"]
        if st.MARKET and u > top + fact["intake_kw"] + TOL_KW:      # the old path's nameplate has no cooling or devices in it, and too much use earns nothing there anyway
            return f"{u:g} kW coming in is past its baseline and the {fact['intake_kw']:g} kW it can add"
        if real(r.get("battery_kw")) and r["battery_kw"] < -fact["battery_kw"] - TOL_KW:
            return f"{-r['battery_kw']:g} kW into a battery that takes {fact['battery_kw']:g}"
    return None


def reading_problem(r: Any, slots: Any = None, fact: dict[str, Any] | None = None, base: Any = None) -> str | None:
    """Why this reading of a sale can't count, or None. slots: the sale's half hours."""
    if not isinstance(r, dict):
        return "it isn't a reading"
    sid = slot_of(r)
    if sid is None:
        return "it names no real slot"
    if slots is not None and sid not in slots:
        return f"this sale has no slot {sid}"
    return value_problem(r, fact, base)


def _differs(a: Any, b: Any) -> bool:
    return real(a) and real(b) and abs(float(a) - float(b)) > 2 * TOL_KW / 10


def _bad_pre(pre: Any, fact: dict[str, Any] | None, pre_base: Any) -> bool:
    """Its load before the event. It raises the node's own baseline, so it can't pass what the meter could have taken in."""
    if not real(pre):
        return True
    if not fact:
        return False
    return not -fact["export_kw"] - TOL_KW <= pre <= (max(float(pre_base), 0.0) if real(pre_base) else 0.0) + fact["intake_kw"] + TOL_KW


def no_bad_truth(r: dict[str, Any]) -> dict[str, Any]:
    """The twin's answer rides on a simulated node's reading. A real node has none, and no money reads it. What can't be read is left out here, so nothing later trips on it."""
    t = r.get("truth")
    ok = isinstance(t, dict) and all(real(v) for v in t.values())
    return r if ok else {k: v for k, v in r.items() if k != "truth"}


def true_kw(r: dict[str, Any]) -> float:
    """The delivered kW the twin saw, for the record. 0 where there is no twin."""
    v = (r.get("truth") or {}).get("delivered_true_kw") if isinstance(r.get("truth"), dict) else None
    return float(v) if real(v) else 0.0


# ------------------------------------------------------------------ the buyer's door
def buyer_refuses(ct: dict[str, Any], sender: str, r: dict[str, Any], event: Callable[..., None]) -> str | None:
    """None if this reading may be kept. Else why not, once it's counted and logged. A contract's slots are all a reading can name."""
    need = ct.get("need") or {}
    slots = {int(s["id"]) for s in need.get("slots") or []} or None
    sid = slot_of(r)
    why = reading_problem(r, slots, facts(sender), _base_of(ct.get("baselines"), sender, sid))
    have = (ct.get("meter_readings") or {}).get(sender) or {}
    if why is None and sid in have and (_differs(r.get("usage_kw"), have[sid]) or _differs(r.get("battery_kw") or 0.0, (ct.get("meter_batt") or {}).get(sender, {}).get(sid))):
        why = "it changes a reading already in"
    if why is None:
        (ct.get("rejected_slots") or {}).get(sender, {}).pop(sid, None)
        return None
    ct["rejected_readings"] = ct.get("rejected_readings", 0) + 1
    if sid is not None and sid not in have and (slots is None or sid in slots):
        ct.setdefault("rejected_slots", {}).setdefault(sender, {})[sid] = why     # no later true reading: this slot goes unpaid, and nobody waits for it
    event("reading_refused", home=sender, event_id=need.get("event_id"), slot=str(r.get("slot_id"))[:24], why=why)
    return beckn.Refused(f"rejected: {why}")


def pre_kw(ct: dict[str, Any], sender: str, r: dict[str, Any], event: Callable[..., None]) -> float | None:
    """The node's load before the event, if it may count. Ask before keeping the reading. The first stands, every honest reading of a sale carries the same one, and a repeat of a
    half hour can't bring one in: that's a second try at a settlement already seen."""
    pre = r.get("pre_kw")
    if pre is None:
        return None
    was = (ct.get("meter_pre") or {}).get(sender)
    repeat = slot_of(r) in ((ct.get("meter_readings") or {}).get(sender) or {})
    bad = _bad_pre(pre, facts(sender), (ct.get("pre_baselines") or {}).get(sender))
    if not bad and was is None and not repeat:
        return float(pre)
    if bad or _differs(pre, was) or was is None:
        ct["pre_refused"] = ct.get("pre_refused", 0) + 1
        event("reading_pre_refused", home=sender, event_id=(ct.get("need") or {}).get("event_id"), pre_kw=str(pre)[:24], kept=was)
    return None


def committed(ct: dict[str, Any], home: str, sid: Any) -> float:
    """What this meter owed in this half hour. A reading says, and the coordinator says too (ct["owed"], with "readings are in"), so a reading moves it only inside what the coordinator allows."""
    return against_owed(float(((ct.get("meter_committed") or {}).get(home) or {}).get(int(sid), 0.0)), (ct.get("owed") or {}).get(home), sid)


def against_owed(said: float, owed: dict[Any, Any] | None, sid: Any) -> float:
    """owed: {slot: (its contract's kW, with the gaps it was dispatched for)}. A dispatch that reached a node after it read the half hour isn't on its reading, so the node's word stands anywhere between."""
    o = (owed or {}).get(int(sid), (owed or {}).get(str(sid)))
    if o is None:
        return said
    lo, hi = float(o[0]), float(o[1])
    room = 0.01 + 0.002 * hi
    return lo if said < lo - room else hi if said > hi + room else said


def _limit_home(name: str, rows: dict[Any, dict[str, float]], fact: dict[str, Any], solar: dict[str, float] | None) -> dict[str, Any]:
    """Cuts this home's slots in place to what could count. Returns what was cut: nothing, for an honest home."""
    cut: dict[str, Any] = {}
    # the battery rate first: per half hour, then over the window
    for i, v in rows.items():
        if "BATT" in v and v["BATT"] > fact["battery_kw"] + TOL_KW:
            cut.setdefault("battery_kw", {})[str(i)] = v["BATT"]
            v["BATT"] = fact["battery_kw"]
    kwh = sum(max(v.get("BATT", 0.0), 0.0) for v in rows.values()) * SLOT_H
    room = BATTERY_SLACK * fact["stored_kwh"] + ENERGY_SLACK_KWH
    if kwh > room:
        cut["battery_kwh"] = {"said": round(kwh, 3), "stored": round(room, 3)}
        for v in rows.values():
            if v.get("BATT", 0.0) > 0:
                v["BATT"] *= room / kwh
    # what counts: its nameplate in each half hour
    for i, v in rows.items():
        if v["BASELINE"] - v["USAGE"] > fact["rated_kw"] + TOL_KW:
            cut.setdefault("nameplate_kw", {})[str(i)] = round(v["BASELINE"] - v["USAGE"], 3)
            v["USAGE"] = v["BASELINE"] - fact["rated_kw"]
            if "BATT" in v:
                v["BATT"] = min(v["BATT"], fact["rated_kw"])
    # and over the window: what it stored, its solar, its own load going to zero
    if solar is not None:
        got = sum(max(v["BASELINE"] - v["USAGE"], 0.0) for v in rows.values()) * SLOT_H
        room = fact["stored_kwh"] + ENERGY_SLACK_KWH + sum(max(v["BASELINE"], 0.0) + SOLAR_SLACK * float(solar.get(str(i), 0.0)) for i, v in rows.items()) * SLOT_H
        if got > room:
            cut["energy_kwh"] = {"said": round(got, 3), "could": round(room, 3)}
            for v in rows.values():
                x = v["BASELINE"] - v["USAGE"]
                if x > 0:
                    v["USAGE"] = v["BASELINE"] - x * room / got
                    if "BATT" in v:
                        v["BATT"] = min(v["BATT"], x * room / got)
    # and the battery rate is paid on what the meter delivered: a claim past that only moves kWh to the better rate, with nothing behind them. It isn't shown as a cut, because an
    # honest battery runs past its meter whenever the house's own load rises (44 of 240 half hours on 2 Jul), so a note here would say nothing
    for v in rows.values():
        if "BATT" in v:
            v["BATT"] = min(v["BATT"], max(v["BASELINE"] - v["USAGE"], 0.0))
    return cut


def limit(meters: dict[str, dict[Any, dict[str, float]]], solar: dict[str, Any] | None = None) -> dict[str, Any]:
    """Every home's slots cut to what could count, in place (a building has its cap). Returns what was cut, by home."""
    cut = {}
    for h, rows in meters.items():
        f = facts(h)
        if f and f["kind"] == "home":
            c = _limit_home(h, rows, f, (solar or {}).get(h))
            if c:
                cut[h] = c
    return cut


def counted(ct: dict[str, Any], bases: dict[str, Any]) -> tuple[dict[str, dict[int, dict[str, float]]], dict[str, Any]]:
    """The contract's meters as settlement counts them, and what was left out or cut. The utility's settlement and the call's measure read this one view."""
    meters: dict[str, dict[int, dict[str, float]]] = {}
    notes: dict[str, Any] = {}
    owed = ct.get("owed")
    for home in ct.get("meters", []):
        for sid, usage in (ct.get("meter_readings") or {}).get(home, {}).items():
            base = bases.get(home, {}).get(str(sid), {}).get("BASELINE")
            if base is None or int(sid) in (ct.get("abnormal") or {}).get(home, ()):
                continue
            if owed is not None and not {int(sid), str(sid)} & set(owed.get(home) or ()):      # it never owed this half hour: nobody asked for it
                notes.setdefault("not_owed", {}).setdefault(home, []).append(int(sid))
                continue
            row = meters.setdefault(home, {})[int(sid)] = {"BASELINE": base, "USAGE": usage}
            if st.MARKET:
                row["BATT"] = float((ct.get("meter_batt") or {}).get(home, {}).get(sid, 0.0))
                row["BASELINE"] = st.adjusted(base, (ct.get("meter_pre") or {}).get(home), (ct.get("pre_baselines") or {}).get(home))
    cut = limit(meters, ct.get("solar")) if st.MARKET else {}      # the old path stops at the capacity sold, and its nameplates hold no cooling
    if cut:
        notes["capped"] = cut
    return meters, notes


def recent_solar(rp: Any, days: list[Any], slots: list[dict[str, Any]]) -> dict[str, float]:
    """The most solar this meter's home made in each half hour of the window on the days just gone: what it could have today."""
    out: dict[str, float] = {}
    for s in slots:
        best = 0.0
        for d in days:
            t = d.isoformat() + s["start"][10:]
            i, j = rp.idx_of(t), rp.idx_of(beckn.add_minutes(t, 15))
            if i is not None and j is not None:
                best = max(best, (float(rp.row(i)["pv_kw"]) + float(rp.row(j)["pv_kw"])) / 2.0)
        out[str(s["id"])] = round(best, 3)
    return out


def at_most_nameplate(home: str, kw: float) -> float:
    f = facts(home) if st.MARKET else None
    return min(kw, f["rated_kw"]) if f else kw


# ------------------------------------------------------------------ a call's reading, at the buyer
def _half_hours(rec: dict[str, Any]) -> tuple[str, str]:
    """The first half hour a call's reading can name, and the end of what it can reach: a gap's helper under a call can owe past the call's end, and earns nothing there."""
    first = rec["start"][:14] + ("00" if rec["start"][14:16] < "30" else "30")
    return first, beckn.add_minutes(rec["end"], 360)


def _start_problem(rec: dict[str, Any], start: Any) -> str | None:
    try:
        ok = isinstance(start, str) and beckn.add_minutes(start, 0) == start and start[14:16] in ("00", "30")
    except (TypeError, ValueError):
        ok = False
    if not ok:
        return "its half hour can't be read"
    first, last = _half_hours(rec)
    return None if first <= start < last else f"{start} is no half hour of this call"


def allowed_kw(rec: dict[str, Any], home: str, start: str) -> float | None:
    """kW the desk dispatched this node for in this half hour. None: the desk's list hasn't come (its confirm was lost)."""
    a = rec.get("alloc")
    if a is None:
        return None
    end = beckn.add_minutes(start, 30)
    return sum(float(kw) for t, kw in (a.get(home) or {}).items() if start <= t < end) / 2.0


def call_refuses(rec: dict[str, Any], sender: str, r: dict[str, Any], event: Callable[..., None]) -> str | None:
    """None if this reading of a call may be kept. A call pays only those it dispatched, and they share what it pays, so a stranger's reading takes from them."""
    why = _start_problem(rec, r.get("start")) or value_problem(r, facts(sender))
    if why is None and rec.get("alloc") is not None and sender not in rec["alloc"]:
        why = "the desk didn't dispatch this node for this call"
    have = (rec.get("readings") or {}).get(sender, {}).get(r.get("start")) if why is None else None
    if have and (_differs(r.get("usage_kw"), have["usage_kw"]) or _differs(r.get("battery_kw") or 0.0, have.get("said_battery_kw", have["battery_kw"]))):
        why = "it changes a reading already in"
    if why is None:
        return None
    rec["rejected"] = rec.get("rejected", 0) + 1
    event("reading_refused", home=sender, event_id=rec.get("id"), start=str(r.get("start"))[:24], why=why)
    return beckn.Refused(f"rejected: {why}")


def take_list(rec: dict[str, Any], alloc: Any, event: Callable[..., None]) -> None:
    """The desk's list of who owes what under this call. A node asked and not dispatched knows the call's id, so what came before the list is held to it too."""
    rec["alloc"] = alloc if isinstance(alloc, dict) else {}
    for h in [h for h in rec.get("readings") or {} if h not in rec["alloc"]]:
        rec["rejected"] = rec.get("rejected", 0) + 1
        del rec["readings"][h]
        event("reading_refused", home=h, event_id=rec.get("id"), why="the desk didn't dispatch this node for this call")
    for h, rows in (rec.get("readings") or {}).items():
        for start, row in rows.items():
            row["dispatched_kw"] = min(row["dispatched_kw"], allowed_kw(rec, h, start) + TOL_KW)


def call_row(rec: dict[str, Any], sender: str, r: dict[str, Any]) -> dict[str, float]:
    """A call's reading as the buyer keeps it: what it was dispatched for is the desk's word, and the battery rate stops at its battery and car."""
    said = float(r.get("committed_kw") or 0.0)
    ok = allowed_kw(rec, sender, r["start"])
    disp = said if ok is None or said <= ok + TOL_KW else ok
    batt = float(r.get("battery_kw") or 0.0)
    f = facts(sender)
    return {"usage_kw": float(r["usage_kw"]), "dispatched_kw": disp, "battery_kw": min(batt, f["battery_kw"]) if f else batt, "said_battery_kw": batt}


def keep_call_row(rec: dict[str, Any], sender: str, r: dict[str, Any]) -> None:
    """The first of a half hour stands for what the meter said, as in a sale. What it was dispatched for follows the desk's word, which can come after the first reading."""
    rows = rec["readings"].setdefault(sender, {})
    row, first = call_row(rec, sender, r), rows.get(r["start"])
    rows[r["start"]] = {**row, **{k: first[k] for k in ("usage_kw", "battery_kw", "said_battery_kw") if k in first}} if first else row


# ------------------------------------------------------------------ the coordinator's door
def owed_slots(market: Any, ev: dict[str, Any], home: str) -> dict[int, tuple[float, float]]:
    """What this node owes in each half hour of this sale: (its contract's kW, and that with what the desk dispatched under the sale to cover a gap). Nothing without an accepted contract."""
    c = ev["commitments"].get(home) or {}
    if c.get("status") != "ACCEPTED":
        return {}
    lo = {x["id"]: float(x["kw"]) for x in c["contract"]["commitments"]}
    hi = dict(lo)
    n = ev["need"]
    sid_of = {s["start"]: s["id"] for s in n["slots"]}
    for g in market.rt.calls.values():
        if g.get("kind") == "gap" and (g.get("parent") or {}).get("id") == n["event_id"]:
            for t, kw in (g["alloc"].get(home) or {}).items():
                sid = sid_of.get(t[:14] + ("00" if t[14:16] < "30" else "30"))
                if sid is not None:
                    hi[sid] = hi.get(sid, 0.0) + float(kw) / 2.0
    return {k: (lo.get(k, 0.0), v) for k, v in hi.items()}


def owed_kw(market: Any, ev: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    """The same for every node, as "readings are in" tells the buyers: they settle on it, and a reading's own word can't move it."""
    return {h: {str(k): [round(a, 3), round(b, 3)] for k, (a, b) in owed_slots(market, ev, h).items()} for h in ev["commitments"] if ev["commitments"][h].get("status") == "ACCEPTED"}


def owed_of(told: Any) -> dict[str, dict[int, tuple[float, float]]] | None:
    """The buyer's copy of owed_kw. None from a coordinator that doesn't say."""
    return {h: {int(k): (float(v[0]), float(v[1])) for k, v in d.items()} for h, d in told.items()} if isinstance(told, dict) else None


def _refused(market: Any, ev: dict[str, Any] | None, sender: str, r: Any, why: str) -> str:
    sid = slot_of(r) if isinstance(r, dict) else None
    if ev is not None:
        ev["readings_refused"] = ev.get("readings_refused", 0) + 1
        if sid is not None and sid not in (ev["readings"].get(sender) or {}):
            ev.setdefault("rejected_slots", {}).setdefault(sender, {})[sid] = why
    market.log("v2_reading_refused", event_id=(ev or {}).get("need", {}).get("event_id"), home=sender, slot=str((r or {}).get("slot_id") if isinstance(r, dict) else "")[:24], why=why)
    return beckn.Refused(f"reading refused: {why}")


def market_refuses(market: Any, sender: str, r: Any, body: dict[str, Any]) -> str | None:
    """None if a node's reading goes on to be kept or ignored as before. Else why not, once it's counted and logged. The buyers hold the same line on their own copies."""
    if not isinstance(r, dict):
        return _refused(market, None, sender, r, "it isn't a reading")
    sent = (body.get("context") or {}).get("sentWall")
    if isinstance(sent, (int, float)) and sent < market.run_wall:
        market.log("v2_reading_ignored", home=sender, why="sent before this run started")
        return beckn.Refused("reading ignored: sent before this run started")
    eid = r.get("event_id")
    if not isinstance(eid, str):
        return _refused(market, None, sender, r, "it names no event")
    rec = market.rt.calls.get(eid)
    if rec is not None:
        why = reading_problem(r, None, facts(sender))
        have = ((rec.get("readings") or {}).get(sender) or {}).get(slot_of(r))
        if why is None and have and any(_differs(r.get(k) or 0.0, have.get(k) or 0.0) for k in MONEY_FIELDS):
            why = "it changes a reading already in"
        if why:
            rec["refused"] = rec.get("refused", 0) + 1
            return _refused(market, None, sender, r, why)
        return None
    ev = market.events.get(eid)
    if ev is None or (ev.get("finalized") and sender not in (ev.get("final_meters") or [])):
        market.log("v2_reading_ignored", event_id=eid[:80], home=sender)
        return None                                        # the market's own answer: nothing here to keep
    sid = slot_of(r)
    why = "it's another node's reading" if r.get("home") != sender else None
    why = why or reading_problem(r, {s["id"] for s in ev["need"]["slots"]}, facts(sender), _base_of(ev.get("baselines"), sender, sid))
    if why is None and market.pooled and sid not in owed_slots(market, ev, sender):
        why = f"it owes nothing in slot {sid}"
    have = (ev["readings"].get(sender) or {}).get(sid)
    if why is None and have and any(_differs(r.get(k) or 0.0, have.get(k) or 0.0) for k in MONEY_FIELDS):
        why = "it changes a reading already in"
    if why is None:
        (ev.get("rejected_slots") or {}).get(sender, {}).pop(sid, None)
        return None
    return _refused(market, ev, sender, r, why)


def clean_actuals(meters: dict[str, Any], log: Callable[..., None], event_id: str) -> dict[str, dict[str, dict[str, float]]]:
    """The buyers' meters as the split reads them. What can't be read is left out, so one node's bad row costs that node and nobody else."""
    out: dict[str, dict[str, dict[str, float]]] = {}
    bad = set()
    for h, m in (meters.items() if isinstance(meters, dict) else ()):
        for k, v in (m.items() if isinstance(m, dict) else ()):
            if isinstance(v, dict) and sane(v.get("BASELINE")) and sane(v.get("USAGE")) and (v.get("BATT") is None or sane(v["BATT"])):
                out.setdefault(h, {})[k] = v
            else:
                bad.add(h)
        if not isinstance(m, dict):
            bad.add(h)
    if bad:
        log("v2_bad_actuals", event_id=event_id, homes=sorted(bad)[:8], count=len(bad))
    return out


def keep_lower(meters: dict[str, dict[Any, dict[str, float]]], h: str, k: Any, v: dict[str, float]) -> None:
    """One buyer's row for the split. A node that told the buyers different things is counted on the copy that pays it less: a bigger copy at one buyer would otherwise be shared
    out at the rate of what both paid, and every other seller would pay for it. Honest copies are the same row."""
    have = meters.setdefault(h, {}).setdefault(k, v)
    if v["BASELINE"] - v["USAGE"] < have["BASELINE"] - have["USAGE"]:
        meters[h][k] = v


def _kw_of(x: Any) -> bool:
    return x is None or (isinstance(x, dict) and all(sane(v) for v in x.values()))


def lacking_problem(x: Any) -> str | None:
    """A buyer's list of the half hours it has no reading for: {node: [slot, ...]}. The tick reads it every second, so what it can't read would stop the market for every sale."""
    ok = x is None or (isinstance(x, dict) and all(isinstance(v, list) and all(isinstance(k, int) and not isinstance(k, bool) for k in v) for v in x.values()))
    return None if ok else "a list of what it lacks that can't be read"


def baselines_problem(x: Any) -> str | None:
    """{node: {slot: {"BASELINE": kW}}}. Every reading is held to these, and the desk's tick reads them."""
    ok = isinstance(x, dict) and all(isinstance(d, dict) and all(isinstance(b, dict) and (b.get("BASELINE") is None or sane(b["BASELINE"])) for b in d.values()) for d in x.values())
    return None if ok else "baselines that can't be read"


def settlement_problem(s: Any) -> str | None:
    """Why the split can't work from this settlement, or None. Each field the split, the sale's record or the tick goes on to read is held here, at the door. A settlement that gets past
    is stored and answered for, so a split that stops on it later pays nobody and can't run again: one junk field held every seller's pay for good on 7 Nov. A settlement refused here
    is asked for again (ask_settlement), and the alarm names the buyer (keep.market_risks). A row in `actuals` that can't be read isn't refused: it costs that node (clean_actuals)."""
    if not isinstance(s, dict):
        return "it isn't a settlement"
    rows = s.get("revenue_flows")
    paid = next((f.get("value") for f in rows if isinstance(f, dict) and f.get("role") == "seller"), None) if isinstance(rows, list) else None
    if not isinstance(rows, list) or not all(isinstance(f, dict) for f in rows) or not sane(paid):
        return "no real sum for the sellers"
    if st.MARKET and paid < 0:
        return "less than nothing for the sellers"         # the market's rules have no penalty, and a split of a negative sum would bill the nodes
    if "full_price_usd" in s and not (sane(s["full_price_usd"]) and s["full_price_usd"] >= 0):
        return "a full price that can't be read"           # every node's rate reads it
    comps = s.get("components")
    if comps is not None and not (isinstance(comps, list) and all(isinstance(r, dict) and sane(r.get("delivered", 0.0)) for r in comps)):
        return "components that can't be read"
    caps = s.get("caps_kw")
    if not _kw_of(s.get("capped_kw")) or not (caps is None or (isinstance(caps, dict) and all(_kw_of(v) for v in caps.values()))):
        return "caps that can't be read"
    top = s.get("top_up")
    if top is not None and not (isinstance(top, int) and not isinstance(top, bool) and top >= 0):
        return "a top-up round that can't be read"
    if s.get("currency") is not None and not isinstance(s["currency"], str):
        return "a currency that can't be read"
    meters = s.get("actuals")
    if meters is not None and not isinstance(meters, dict):
        return "actuals that can't be read"
    if meters is not None and paid > 0 and not any(v["BASELINE"] - v["USAGE"] > 0 or (v.get("BATT") or 0.0) > 0 for m in clean_actuals(meters, lambda *a, **k: None, "").values() for v in m.values()):
        return "a sum for the sellers and no meter that shows them delivering"      # it would pay nobody, with the sale marked done
    try:
        json.dumps({k: v for k, v in s.items() if k != "actuals"}, allow_nan=False)     # whatever else it carries goes on the record, and a NaN there stops the routes that serve it
    except (TypeError, ValueError, RecursionError):
        return "a number that isn't real"
    return lacking_problem(s.get("lacking"))


def call_settlement_problem(s: Any, rec: dict[str, Any], buyer: str, most_kwh: float) -> str | None:
    """settlement_problem, for a call: why this one can't be paid on, or None. What gets past here is taken for paid, and only a top-up could mend it, so one junk number held all that the call's
    nodes were owed, or paid one of them less than nothing. A call refused here is on the alarm (keep.market_risks) until one that can be read comes. most_kwh: what the call could bring in, at its kW."""
    if not isinstance(s, dict):
        return "it isn't a settlement"
    if rec.get("from") != buyer:
        return "it isn't this buyer's call"
    money, kwh, top = [s.get("paid_usd", 0.0)] + ([s["full_price_usd"]] if "full_price_usd" in s else []), s.get("delivered_kwh", 0.0), s.get("top_up")
    if not all(sane(v) and v >= 0 for v in money):
        return "no real sum"
    if not (sane(kwh) and kwh >= 0) or kwh > most_kwh + 0.01:
        return "a delivery the call can't have brought in"
    homes, batt = s.get("by_home_kwh"), s.get("by_home_batt_kwh")
    if not all(x is not None and isinstance(x, dict) and all(sane(v) and v >= 0 for v in x.values()) for x in (homes, {} if batt is None else batt)):
        return "kWh that can't be read"
    if sum(homes.values()) > kwh + 0.01 + 1e-4 * len(homes):
        return "more kWh for its nodes than it says the call brought in"     # a node's rate is the sum over them all, so a stranger's kWh would take from each of them
    read = s.get("read")
    if read is not None and not (isinstance(read, dict) and all(isinstance(v, list) for v in read.values())):
        return "a list of what it read that can't be read"
    if top is not None and not (isinstance(top, int) and not isinstance(top, bool) and top >= 0):
        return "a top-up round that can't be read"
    if s.get("currency") is not None and not isinstance(s["currency"], str):
        return "a currency that can't be read"
    try:
        json.dumps(s, allow_nan=False)                     # it stays on the record, and a NaN there stops the routes that serve it
    except (TypeError, ValueError, RecursionError):
        return "a number that isn't real"
    return None


def buyers_disagree(orders: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Half hours where two buyers hold different meters for one node: it told them different things, and each pays on its own copy. Each is held to its limits, and this says so."""
    seen: dict[tuple[str, str], float] = {}
    out: dict[str, list[str]] = {}
    for od in orders:
        for h, m in ((od.get("settlement") or {}).get("actuals") or {}).items():
            for k, v in (m.items() if isinstance(m, dict) else ()):
                u = v.get("USAGE") if isinstance(v, dict) else None
                if real(u) and _differs(u, seen.setdefault((h, str(k)), u)):
                    out.setdefault(h, []).append(str(k))
    return out
