"""Measurement and settlement, written to match published rules exactly.

deg_settle  - a line-by-line port of beckn/DEG specification/policies/demand-flex-contractpolicy.rego:
                delivered_i = sum over meters of clamp0(BASELINE_i - USAGE_i)        (kW)
                eligible_i  = min(delivered_i, CAPACITY_OFFERED_i)
                pay_i       = eligible_i x hours x PRICE_i
                penalty_i   = clamp0(CAPACITY_OFFERED_i - delivered_i) x hours x SHORTFALL_PENALTY_i
              buyer pays -(sum of net_i), seller receives +(sum of net_i): net zero.
ameren_baseline - Ameren Illinois Peak Time Rewards baseline (Rider PTR, Sheet 51.001): of "the 14 previous days
              that are not holidays or Curtailment Event days", the 3 with the highest usage in the event hours,
              averaged. Holidays are the tariff's own six (New Year's Day, Memorial Day, Fourth of July, Labor Day,
              Thanksgiving, Christmas; when one falls on a Sunday, the Monday after it is a holiday too). The lookback walks
              back past holidays and event days until it has 14 days (ptr_baseline_days).
              The tariff also drops days whose Temperature Humidity Index is not within 10 % of the event day's; we
              skip that step because the dataset's community load follows 2018 weather while prices are 2023.
split       - the coordinator's rule for sharing the seller's revenue: pro rata to each home's measured
              delivery (clamp0(BASELINE - USAGE) summed over the event), minus a coordinator fee (default 0).
USAGE is consumption at the meter in kW (import positive); a home exporting has negative USAGE.

Under FG_MARKET=2026, the rules every run now uses, three of those give way:
  ten_of_ten stands in for ameren_baseline, with the same-day adjustment on top (adjusted)
  deg_settle pays for all that was delivered: its cap at the offer and its penalty are off
  share_out stands in for split. A node gets the order's price on each kWh it delivered, times its class's share in price_book.json, and a buyer pays each class's own rate: the
  price x the class's share / (1 - coordinator_keeps). So the coordinator keeps coordinator_keeps of the bill, whatever the mix. At $2.04 a battery's or two-way car's kWh costs
  $2.04, a building's $1.479 and another home device's $0.5355, and the order's price is the most a kWh can cost. A home's battery and the rest of its load keep their own shares,
  and a building's battery earns the business rate (meter_kwh).
  A settlement carries two sums: `total`, the bill, and `full_price_usd`, the same kWh at the order's price. A node's rate reads the second, so its pay doesn't hang on the bill. The coordinator works
  both out again from the buyer's meters and the order's price (market.own_sums). A buyer's word stands only when it's the same to the cent: otherwise its own sum stands, and it's an alarm.
  A book with no coordinator_keeps (data_ma's) keeps the one price: there the bill is the full price and the coordinator keeps what the shares leave.
baseline_rule, pay_rule, bill_rule and policy_rule say which rule is in force. A signed record states its rule from them, so the rule and its words can't part.
"""
from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from typing import Any

MARKET = "" if os.environ.get("FLEXGRID_DATASET") == "il2023" else os.environ.get("FG_MARKET", "")   # benchmark: "2026" = the Massachusetts rules and the price book
_BOOK: dict[str, Any] | None = None


def price_book() -> dict[str, Any]:
    global _BOOK
    if _BOOK is None:
        from .config import data_path
        _BOOK = json.loads(data_path("devices/price_book.json").read_text())
    return _BOOK


def baseline_rule() -> str:
    """The baseline in force, as a record states it."""
    from .config import dataset_meta
    if not MARKET:
        return ("Ameren PTR: 3 highest of the 14 previous non-holiday, non-event days" if dataset_meta().get("name") == "il2023"
                else "3 highest of the 14 previous non-holiday, non-event days (the Rider PTR method)")
    lift = ", plus the meter's load in the hour starting 2 hours before the event, above its baseline for that hour" if price_book().get("same_day_adjustment") else ""
    return f"mean of the last 10 days of the same kind, weekday or weekend, that aren't holidays or event days{lift}"


def catalog_terms(catalog: dict[str, Any]) -> dict[str, Any]:
    """A buyer's catalog as it goes out. The dataset's names the old rules, and its offers are signed on every step of a sale, so under the 2026 rules they say what's in force."""
    if MARKET:
        for o in catalog["offers"]:
            a = o["offerAttributes"]
            a["contractAttributes"]["policy"]["changed"] = "cap at the offer and shortfall penalty off" + ("" if keeps() is None else f"; each kWh billed at the price x its class's share / {1 - keeps():g}")
            if o.get("descriptor", {}).get("shortDesc"):
                o["descriptor"]["shortDesc"] = up_to() + o["descriptor"]["shortDesc"]
            for side in a["inputs"]:
                if side["role"] == "buyer":
                    side["inputs"]["baselineMethodology"] = {"lastOf": 10, "sameKind": True, "sameDayAdjustment": bool(price_book().get("same_day_adjustment"))}
    return catalog


def keeps() -> float | None:
    """The coordinator's part of a bill, from the price book. None: a book that states none, where a buyer pays the one price."""
    k = price_book().get("coordinator_keeps")
    return None if k is None else float(k)


def keep_words() -> str:
    """What follows a node's share in any rule the 2026 market states."""
    k = keeps()
    return "The coordinator keeps the rest" if k is None else f"The buyer pays that divided by {1 - k:g}, so the coordinator keeps {k:g} of the bill"


def up_to() -> str:
    """What goes before a price wherever one is shown: under each class's rate the order's price is the most a kWh can cost."""
    return "up to " if MARKET and keeps() is not None else ""


def class_rule() -> str:
    """share_out's rule, as a contract and a payment state it."""
    x = price_book()["x"]
    return (f"the order's price on each measured kWh a node delivered, times its class's share: home battery or two-way car {x['home_battery']:g}, building {x['building']:g}, "
            f"any other home device {x['home_other']:g}. {keep_words()} (devices/price_book.json)")


def pay_rule() -> str:
    """How a node's pay follows from its delivery, in the rules in force."""
    return class_rule() if MARKET else "pro rata to measured delivery"


def bill_rule() -> str:
    """What a buyer pays, as an order, a settlement and a model's facts state it."""
    k = keeps()
    if not MARKET or k is None:
        return "the order's price on every measured kWh"
    x = {c: f"{v / (1 - k):g}" for c, v in price_book()["x"].items()}
    return (f"each measured kWh at its class's own rate, the order's price times {x['home_battery']} for a home battery or two-way car, {x['building']} for a building and "
            f"{x['home_other']} for any other home device, so the order's price is the most a kWh can cost and the coordinator keeps {k:g} of the bill")


def policy_rule() -> str:
    """What a settlement was worked out under."""
    port = "beckn/DEG specification/policies/demand-flex-contractpolicy.rego (ported)"
    return f"{port}, with its cap and penalty off: paid for all that was delivered, stopping at the capacity sold only in a half hour a call covers. The bill: {bill_rule()}" if MARKET else port


def similar_days(event_day: date, event_days: set[date], n: int = 10, max_back: int = 70) -> list[date]:
    """ISO-NE's last 10-of-10: the last n days of the same type (weekday or weekend), no holiday, no event, newest first."""
    from .config import dataset_meta
    first = date.fromisoformat(dataset_meta().get("baseline_from", "2026-05-11"))     # the dataset's first full day after its warm-up
    days, k, hol, wk = [], 0, baseline_holidays(), event_day.weekday() >= 5
    while len(days) < n and k < max_back:
        k += 1
        d = event_day - timedelta(days=k)
        if d < first:
            break
        if d in hol or d in event_days or (d.weekday() >= 5) != wk:
            continue
        days.append(d)
    return days


def ten_of_ten(usage_by_day: dict[date, list[float]], event_day: date, event_days: set[date]) -> dict[str, Any]:
    days = [d for d in similar_days(event_day, event_days) if d in usage_by_day]
    if not days:
        return {"per_slot": [], "days": []}
    n = len(usage_by_day[days[0]])
    return {"per_slot": [round(sum(usage_by_day[d][j] for d in days) / len(days), 3) for j in range(n)],
            "days": [d.isoformat() for d in days], "candidates": len(days)}


def adjusted(base: float, pre_kw: float | None, pre_base: float | None) -> float:
    """The same-day adjustment (Massachusetts Targeted Dispatch; PJM's and ISO-NE's baselines have one too): the node's load in
    the hour starting 2 hours before the event, above its baseline for that hour, is added. Never negative."""
    if not price_book().get("same_day_adjustment") or pre_kw is None or pre_base is None:
        return base
    return base + clamp0(float(pre_kw) - float(pre_base))


def node_class(name: str, building: bool = False) -> str:
    return "datacenter" if str(name).startswith("dc-") else "building" if building else "home_other"


def meter_kwh(m: dict[Any, dict[str, float]], hours: float, cap: dict[Any, float] | None = None,
              building: bool = False, name: str = "") -> tuple[dict[str, float], dict[str, float]]:
    """Per slot: (kWh delivered, kWh the node itself is paid for at the order's price: its share of it), under the 2026 rules.
    cap: a building's kW per slot."""
    b = price_book()
    rest, batt = {}, {}
    capk = {str(k): float(v) for k, v in (cap or {}).items()}
    for i, v in m.items():
        if "BASELINE" not in v or "USAGE" not in v:
            continue
        batt[str(i)] = clamp0(float(v.get("BATT") or 0.0))
        x = float(v["BASELINE"]) - float(v["USAGE"]) - batt[str(i)]
        if str(i) in capk:                         # the cap is on the whole meter: a campus battery counts inside it, not on top
            batt[str(i)] = min(batt[str(i)], capk[str(i)])
            x = min(x, capk[str(i)] - batt[str(i)])
        rest[str(i)] = x
    if sum(rest.values()) < 0:                     # a negative event pays nothing (the season's average never goes below 0)
        rest = {k: 0.0 for k in rest}
    xr = float(b["x"][node_class(name, building)])
    xb = xr if building else float(b["x"]["home_battery"])     # a building's battery earns the business rate, as in Massachusetts
    return ({k: (batt[k] + rest[k]) * hours for k in rest}, {k: (batt[k] * xb + rest[k] * xr) * hours for k in rest})


def trim_called(val: dict[str, tuple[dict[str, float], dict[str, float]]], capped: dict[Any, float], hours: float) -> None:
    """In a half hour a call covers the booking pays only up to the capacity sold (deg_settle's capped), so each node's booked kWh there
    shrink by the same factor. Its part of the call is paid under the call."""
    for sid, kw in (capped or {}).items():
        tot = sum(v[0].get(str(sid), 0.0) for v in val.values()) / hours
        if tot > float(kw) > 0:
            f = float(kw) / tot
            for v in val.values():
                for d in v:
                    if str(sid) in d:
                        d[str(sid)] *= f


def within(per: dict[str, float], bill: float) -> dict[str, float]:
    """No split pays out more than its bills brought in. A full price the bills don't bear out could ask for it: a buyer's word, or a copy of the meters short of what the split reads."""
    pool = sum(per.values())
    return per if pool <= bill else {h: v * bill / pool for h, v in per.items()}


def share_out(seller: float, full: float, val: dict[str, tuple[dict[str, float], dict[str, float]]]) -> dict[str, Any]:
    """seller: the bills. full: the same kWh at the order's price. Each node's pay is that price, read as full over the kWh, times its class's share, so a change in the
    bill can't move it. The coordinator keeps the difference: coordinator_keeps of the bill, when the buyers' meters are the ones the split reads."""
    kwh = sum(sum(v[0].values()) for v in val.values())
    rate = full / kwh if kwh > 0 else 0.0
    per = {h: round(v, 4) for h, v in within({h: sum(v[1].values()) * rate for h, v in val.items()}, seller).items()}
    pool = sum(per.values())
    return {"pool": round(pool, 4), "fee": round(seller - pool, 4), "per_home": per, "usd_per_kwh": round(rate, 4),
            "rule": class_rule()}

# Ameren Illinois Rider PTR holidays, 2023: New Year's Day (Sun 1 Jan, so Mon 2 Jan as well), Memorial Day,
# Fourth of July, Labor Day, Thanksgiving Day, Christmas Day. Not the federal (OPM) list: no MLK, Presidents',
# Juneteenth, Columbus or Veterans Day.
PTR_HOLIDAYS_2023 = {date(2023, 1, 1), date(2023, 1, 2), date(2023, 5, 29), date(2023, 7, 4), date(2023, 9, 4),
                     date(2023, 11, 23), date(2023, 12, 25)}
HOLIDAYS_2023 = PTR_HOLIDAYS_2023                   # the name older code imports


def baseline_holidays() -> set[date]:
    """The holidays a baseline skips: Rider PTR's for the Illinois 2023 data, the dataset's own list otherwise
    (data_ma/dataset.json: the Massachusetts holidays inside the summer the data covers)."""
    from .config import dataset_meta
    m = dataset_meta()
    if m.get("name") == "il2023":
        return PTR_HOLIDAYS_2023
    return {date.fromisoformat(d) for d in m.get("holidays", [])}


def clamp0(x: float) -> float:
    return x if x >= 0 else 0.0


def deg_settle(slots: list[dict[str, Any]], offered: dict[int, float], meters: dict[str, dict[int, dict[str, float]]],
               hours: float = 0.5, currency: str = "USD",
               caps: dict[str, dict[int, float]] | None = None,
               share: dict[Any, float] | None = None, capped: dict[Any, float] | None = None) -> dict[str, Any]:
    """slots: [{id, price, penalty}]; offered: {slot_id: kW}; meters: {meter: {slot_id: {BASELINE, USAGE}}}.
    caps: {meter: {slot_id: kW}}, the most a meter's measured reduction counts in a half hour (a building node:
    BUILDING_CAP_X x its commitment, see building_counted_kwh); the homes have none, so their settlement is the port's.
    share: {slot_id: 0-1}, this buyer's part of what the pool delivered, when two buyers' orders fill from one pool (market.py).
    Each buyer settles its own part on the same meters, so no kWh is paid for twice.
    capped: {slot_id: kW}, the half hours a call covers: the booking stops at the capacity sold, because the call pays for what's beyond it.
    Under the 2026 rules `total` is the bill, each kWh at its class's rate, and `full_price_usd` the same kWh at the slot's price."""
    rows, total, full = [], 0.0, 0.0
    keep = keeps() if MARKET else None
    caps = caps or {}
    part = {str(k): float(v) for k, v in (share or {}).items()}
    lim = {str(k): float(v) for k, v in (capped or {}).items()}

    def counted(name: str, m: dict, i) -> float:
        v = clamp0(m[i]["BASELINE"] - m[i]["USAGE"])
        cap = (caps.get(name) or {}).get(i) if name in caps else None
        return v if cap is None else min(v, clamp0(cap))
    per = {n: meter_kwh(m, hours, caps.get(n), n in caps, n) for n, m in meters.items()} if MARKET else {}
    for s in slots:
        i = s["id"]
        delivered = sum(counted(n, m, i) for n, m in meters.items()
                        if i in m and "BASELINE" in m[i] and "USAGE" in m[i])
        k = part.get(str(i), 1.0)
        delivered *= k
        off = float(offered.get(i, 0.0))
        eligible = min(delivered, off)
        pay = eligible * hours * s["price"]
        penalty = clamp0(off - delivered) * hours * s.get("penalty", 0.0)
        if MARKET:                                 # paid for what was delivered: no cap at the promise, no penalty
            delivered = sum(p[0].get(str(i), 0.0) for p in per.values()) / hours
            cut = lim[str(i)] / delivered if str(i) in lim and delivered > lim[str(i)] else 1.0     # each class by the same factor, as trim_called does
            delivered = k * (min(delivered, lim[str(i)]) if str(i) in lim else delivered)
            eligible = delivered
            pay = delivered * hours * s["price"]
            full += pay
            if keep is not None:                   # the nodes' part of the half hour, over what they're left of the bill
                pay = k * cut * sum(p[1].get(str(i), 0.0) for p in per.values()) * s["price"] / (1.0 - keep)
            penalty = 0.0
        net = pay - penalty
        total += net
        rows.append({"id": i, "price": s["price"], "offered": round(off, 3), "delivered": round(delivered, 3),
                     "eligible": round(eligible, 3), "pay": round(pay, 4), "penalty": round(penalty, 4),
                     "net": round(net, 4)})
    flows = [{"role": "buyer", "value": round(-total, 4), "currency": currency},
             {"role": "seller", "value": round(total, 4), "currency": currency}]
    return {"components": rows, "total": round(total, 4), "revenue_flows": flows,
            "net_zero_ok": abs(sum(f["value"] for f in flows)) < 1e-9,
            "policy": policy_rule(), **({"full_price_usd": round(full, 4)} if MARKET else {})}


def call_shares(full_usd: float, by_home_kwh: dict[str, float], batt_kwh: dict[str, float], buildings: set[str], bill_usd: float | None = None) -> dict[str, Any]:
    """A call's payment on the price book: each node's kWh at the call's price, times its class's share, stored energy at the battery
    share, a building at its own. full_usd: the call's kWh at its price. bill_usd: what its buyer pays (call_bill), and the coordinator keeps the difference, as in the booking."""
    b = price_book()["x"]
    bill = full_usd if bill_usd is None else bill_usd
    kwh = sum(float(v) for v in by_home_kwh.values())
    rate = full_usd / kwh if kwh > 0 else 0.0
    per = {}
    for h, v in by_home_kwh.items():
        v = float(v)
        if h in buildings:
            per[h] = v * float(b["building"]) * rate
        else:
            s = min(v, clamp0(float(batt_kwh.get(h, 0.0))))
            per[h] = (s * float(b["home_battery"]) + (v - s) * float(b["home_other"])) * rate
    per = within(per, bill)
    pool = sum(per.values())
    return {"pool": round(pool, 4), "fee": round(bill - pool, 4), "per_home": {h: round(v, 4) for h, v in per.items()},
            "usd_per_kwh": round(rate, 4), "rule": f"the call's price on what each node added, times its class's share. {keep_words()} (price_book.json)"}


def call_bill(full_usd: float, by_home_kwh: dict[str, float], batt_kwh: dict[str, float], buildings: set[str]) -> float:
    """What a call costs its buyer: what call_shares gives its nodes, over what they're left of the bill. So the keep holds even where a half hour's kWh went to no node."""
    k = keeps()
    return full_usd if k is None else round(call_shares(full_usd, by_home_kwh, batt_kwh, buildings)["pool"] / (1.0 - k), 4)


def deposit_statement(total_usd: float, held_usd: float) -> dict[str, float]:
    """A buyer's deposit against its bill. It held tentative power, so it counts toward what the buyer owes. If the bill is smaller the rest goes back: credited and refunded
    always add up to what was held. The sellers' money doesn't change: they are paid on what was delivered. A deposit is set at the order's price and a bill at each class's
    rate, so on a poor delivery of cheap kWh it can pass the bill, and then nothing is due."""
    held = clamp0(float(held_usd))
    used = min(held, clamp0(float(total_usd)))
    return {"held_usd": round(held, 4), "credited_usd": round(used, 4), "refund_usd": round(held - used, 4),
            "still_due_usd": round(float(total_usd) - used, 4)}


def ptr_baseline_days(event_day: date, event_days: set[date], lookback: int = 14, max_back: int = 90) -> list[date]:
    """Rider PTR: "the 14 previous days that are not holidays or Curtailment Event days", newest first. Walks back
    past holidays and event days until it has `lookback` days (at most `max_back` calendar days)."""
    days, k = [], 0
    hol = baseline_holidays()
    while len(days) < lookback and k < max_back:
        k += 1
        d = event_day - timedelta(days=k)
        if d in hol or d in event_days:
            continue
        days.append(d)
    return days


def ameren_baseline(usage_by_day: dict[date, list[float]], event_day: date, event_days: set[date],
                    lookback: int = 14, best_of: int = 3) -> dict[str, Any]:
    """usage_by_day: {day: [kW per slot in the event window]} from the meter history, which must cover
    ptr_baseline_days(event_day, event_days, lookback); a day missing from it is left out."""
    days = [d for d in ptr_baseline_days(event_day, event_days, lookback) if d in usage_by_day]
    ranked = sorted(days, key=lambda d: -sum(usage_by_day[d]))[:best_of]
    if not ranked:
        return {"per_slot": [], "days": []}
    n = len(usage_by_day[ranked[0]])
    per_slot = [round(sum(usage_by_day[d][j] for d in ranked) / len(ranked), 3) for j in range(n)]
    return {"per_slot": per_slot, "days": [d.isoformat() for d in ranked], "candidates": len(days)}


BUILDING_CAP_X = 1.2     # a building counts at most this times its commitment in the split (the reputation's cap)


def building_counted_kwh(meter: dict[Any, dict[str, float]], committed_kw: dict[Any, float], hours: float = 0.5) -> float:
    """A building node's measured delivery as the program counts it: in each half hour its reduction against the
    baseline, at most BUILDING_CAP_X x what it committed for that half hour. The baseline (the 3 highest of 14 days)
    picks a building's highest days; for the library those are days without its own 17:00-20:00 cut, so its routine
    would otherwise count as hundreds of kW delivered (2 Jul 2026, a local run: about 450 kW "measured" in each half
    hour while it truly gave 28 kW). meter: {slot_id:
    {BASELINE, USAGE}}; committed_kw: {slot_id: kW} (keys compared as strings). Used the same way by the utility
    (utility.settle, per half hour through deg_settle's caps), the coordinator's split (market.settle), the real-time
    calls (rt_buyer.GridWatch.settle: at most 1.2 x booked + dispatched) and the independent check (verify.check)."""
    com = {str(k): float(v or 0.0) for k, v in (committed_kw or {}).items()}
    return sum(min(clamp0(v["BASELINE"] - v["USAGE"]), BUILDING_CAP_X * com.get(str(sid), 0.0)) * hours
               for sid, v in meter.items() if "BASELINE" in v and "USAGE" in v)


def split(seller_total: float, delivered_kwh: dict[str, float], fee_share: float = 0.0,
          cap_kwh: dict[str, float] | None = None) -> dict[str, Any]:
    """cap_kwh: participants whose measured delivery counts at most this much (a building: building_counted_kwh, its
    reduction capped at BUILDING_CAP_X x its commitment in each half hour, as the utility counted it). The 3-of-14
    baseline takes a building's highest days, which for the library are days without its own 17:00-20:00 cut, so
    uncapped its routine would take most of the homes' pool (2 Jul 2026, simulated: $425 of $612 for 35 kWh truly
    delivered)."""
    pool = seller_total * (1 - fee_share)
    cap_kwh = cap_kwh or {}
    counted = {h: (min(clamp0(v), clamp0(cap_kwh[h])) if h in cap_kwh else clamp0(v)) for h, v in delivered_kwh.items()}
    tot = sum(counted.values())
    pay = {h: (round(pool * v / tot, 4) if tot > 0 else 0.0) for h, v in counted.items()}
    out = {"pool": round(pool, 4), "fee": round(seller_total - pool, 4), "per_home": pay,
           "rule": "pro rata to measured delivery (clamp0(BASELINE - USAGE) x hours)"}
    if cap_kwh:
        out["caps_kwh"] = {h: round(v, 3) for h, v in cap_kwh.items()}
        out["rule"] += f"; a building counts at most {BUILDING_CAP_X:g} x its commitment in each half hour"
    return out


def day_of(ts_local: str) -> date:
    return datetime.strptime(ts_local[:10], "%Y-%m-%d").date()


def line_fit(xs: list[float], ys: list[float]) -> dict[str, float]:
    """Least-squares line y = a + b x (weather-normalized baselines: usage against the day's highest temperature)."""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx > 1e-9 else 0.0
    return {"a": round(my - b * mx, 5), "b": round(b, 5), "days": n}
