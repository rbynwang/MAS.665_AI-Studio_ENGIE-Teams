"""Real-world grid signals: what the MISO grid is doing right now, from MISO's public data (no login, no key).

The agents trade on the 2023 dataset's clock; this module is the one place where today's real grid comes in. It is
read by the coordinator (GET /grid/now, cached), the Control Room's "Grid now" panel, the buyer agent's MCP tool
get_grid_outlook, and tools/grid_now.py. Nothing here moves a device: it informs which window to relieve and which
hours suit flexible load (data center batch work, EV charging).

  snapshot()  ->  {
    fuel_mix   the latest 5-minute MISO-wide generation by fuel (MW), the share from wind and solar, and the
               carbon-free share (wind, solar, hydro, nuclear)
    load       MISO's real-time total load now, today's hourly cleared load and MISO's medium-term load forecast
    price_now  the 5-minute real-time LMP at the Illinois hub (where the FlexGrid homes are) and at the other hubs
    day_ahead  today's day-ahead LMP at the Illinois hub, hour by hour: the most expensive 2-hour window (the peak
               to relieve) and the cheapest hours (where to put flexible load)
    plan       what FlexGrid would ask for today on these signals, in plain words
    errors     which sources could not be read (each part degrades on its own)
  }

Sources (MISO public data, the same ones the open-source gridstatus library reads):
  https://public-api.misoenergy.org/api/FuelMix
  https://public-api.misoenergy.org/api/RealTimeTotalLoad
  https://public-api.misoenergy.org/api/MarketPricing/GetRealTimeFiveMinExPost/Current
  https://docs.misoenergy.org/marketreports/<YYYYMMDD>_da_expost_lmp.csv    (day-ahead LMPs, hour ending EST)
MISO's market hours are Eastern Standard Time all year; Illinois is on Central time, which equals EST while daylight
saving time is on and is one hour behind it in winter. Hours below are given in Illinois local time.
"""
from __future__ import annotations

import csv
import io
import time
from datetime import datetime, timedelta, timezone
from typing import Any

API = "https://public-api.misoenergy.org/api"
REPORTS = "https://docs.misoenergy.org/marketreports"
URLS = {"fuel_mix": f"{API}/FuelMix", "load": f"{API}/RealTimeTotalLoad",
        "price_now": f"{API}/MarketPricing/GetRealTimeFiveMinExPost/Current"}
HUB = "ILLINOIS.HUB"
CARBON_FREE = {"Wind", "Solar", "Hydro", "Nuclear"}
WIND_SOLAR = {"Wind", "Solar"}
EST = timezone(timedelta(hours=-5))


def da_url(day: str) -> str:
    return f"{REPORTS}/{day.replace('-', '')}_da_expost_lmp.csv"


# ------------------------------------------------------------------------------------------------ time
def _us_dst(d: datetime) -> bool:
    """US daylight saving time: second Sunday of March to first Sunday of November (2:00 local)."""
    y = d.year
    mar = datetime(y, 3, 8) + timedelta(days=(6 - datetime(y, 3, 8).weekday()) % 7)
    nov = datetime(y, 11, 1) + timedelta(days=(6 - datetime(y, 11, 1).weekday()) % 7)
    return mar <= d.replace(tzinfo=None) < nov


def central_minus_est_h(day: str) -> int:
    """Illinois local time minus EST, in hours: 0 under daylight saving time (CDT = UTC-5 = EST), else -1 (CST)."""
    return 0 if _us_dst(datetime.strptime(day, "%Y-%m-%d") + timedelta(hours=12)) else -1


def he_to_local(he: int, day: str) -> str:
    """Hour ending `he` (1-24, EST) -> the local start time of that hour, 'HH:00'."""
    h = (he - 1 + central_minus_est_h(day)) % 24
    return f"{h:02d}:00"


def today_central() -> str:
    now = datetime.now(timezone.utc) + timedelta(hours=-5)
    day = now.strftime("%Y-%m-%d")
    return (now + timedelta(hours=central_minus_est_h(day))).strftime("%Y-%m-%d")


# ------------------------------------------------------------------------------------------------ parsing
def _num(x: Any) -> float | None:
    try:
        return float(str(x).replace(",", ""))
    except (TypeError, ValueError):
        return None


def parse_fuel_mix(j: dict[str, Any]) -> dict[str, Any]:
    by = {}
    for t in (j.get("Fuel") or {}).get("Type") or []:
        v = _num(t.get("ACT"))
        if v is not None:
            by[t.get("CATEGORY", "?")] = by.get(t.get("CATEGORY", "?"), 0.0) + v
    gen = {k: v for k, v in by.items() if k != "Imports"}           # generation inside MISO
    total = sum(v for v in gen.values() if v > 0)
    pct = lambda keys: round(100 * sum(max(0.0, gen.get(k, 0.0)) for k in keys) / total, 1) if total else None  # noqa: E731
    return {"interval": j.get("RefId"), "total_mw": _num(j.get("TotalMW")), "by_fuel_mw": by,
            "generation_mw": round(total), "wind_solar_pct": pct(WIND_SOLAR), "carbon_free_pct": pct(CARBON_FREE)}


def parse_load(j: dict[str, Any]) -> dict[str, Any]:
    li = j.get("LoadInfo") or {}
    cleared = [_num((x.get("ClearedMWHourly") or {}).get("Value")) for x in li.get("ClearedMW") or []]
    fc = {}
    for x in li.get("MediumTermLoadForecast") or []:
        f = x.get("Forecast") or {}
        he, v = _num(f.get("HourEnding")), _num(f.get("LoadForecast"))
        if he is not None and v is not None:
            fc[int(he)] = v
    five = [(x.get("Load") or {}) for x in li.get("FiveMinTotalLoad") or []]
    five = [(f.get("Time"), _num(f.get("Value"))) for f in five if _num(f.get("Value")) is not None]
    peak = max(fc.items(), key=lambda kv: kv[1]) if fc else None
    return {"interval": li.get("RefId"), "now_mw": five[-1][1] if five else None, "now_time_est": five[-1][0] if five else None,
            "hourly_cleared_mw": [x for x in cleared if x is not None],
            "forecast_mw_by_he": [fc.get(h) for h in range(1, 25)],
            "peak_forecast": {"he": peak[0], "mw": peak[1]} if peak else None}


def parse_price_now(j: dict[str, Any]) -> dict[str, Any]:
    hubs = {}
    interval = None
    for row in j.get("data") or []:
        if len(row) >= 3 and str(row[1]).endswith(".HUB"):
            hubs[row[1]] = _num(row[2])
            interval = row[0]
    return {"interval_est": interval, "hub_usd_mwh": hubs, "illinois_hub_usd_mwh": hubs.get(HUB)}


def parse_day_ahead(text: str, node: str = HUB) -> list[float] | None:
    """The day-ahead ex-post LMP CSV -> the node's 24 hourly LMPs (hour ending 1..24 EST)."""
    lines = text.splitlines()
    head = next((i for i, l in enumerate(lines) if l.startswith("Node,")), None)
    if head is None:
        return None
    for row in csv.reader(io.StringIO("\n".join(lines[head:]))):
        if len(row) >= 27 and row[0] == node and row[2] == "LMP":
            vals = [_num(x) for x in row[3:27]]
            return vals if all(v is not None for v in vals) else None
    return None


def outlook(lmp: list[float], day: str, window_h: int = 2, cheap_n: int = 4) -> dict[str, Any]:
    """The most expensive `window_h`-hour window of the day (the peak to relieve) and the `cheap_n` cheapest hours,
    on the Illinois day's own hours. In winter (CST) MISO's hour ending 1 EST is 23:00-24:00 of the evening before,
    so the day keeps 00:00-22:00 (23 hours; its 23:00 hour is in tomorrow's report)."""
    vals = lmp[-central_minus_est_h(day):]                  # the hours that start at local 00:00, 01:00, ...
    n = len(vals)
    best = max(range(0, n - window_h + 1), key=lambda i: sum(vals[i:i + window_h]))
    w = vals[best:best + window_h]
    order = sorted(range(n), key=lambda i: vals[i])[:cheap_n]
    hhmm = lambda i: f"{i:02d}:00"                                                     # noqa: E731
    end = lambda i: f"{(i + 1) % 24:02d}:00"                                           # noqa: E731
    return {"date": day, "node": HUB, "hours": "local (Illinois)",
            "lmp_usd_mwh": [round(x, 2) for x in vals], "starts": [hhmm(i) for i in range(n)],
            "peak_window": {"start": hhmm(best), "end": end(best + window_h - 1), "avg_usd_mwh": round(sum(w) / window_h, 2),
                            "max_usd_mwh": round(max(w), 2)},
            "cheapest_hours": [{"start": hhmm(i), "end": end(i), "usd_mwh": round(vals[i], 2)} for i in sorted(order)],
            "day_avg_usd_mwh": round(sum(vals) / n, 2), "spread_usd_mwh": round(max(vals) - min(vals), 2)}


def plan_from(s: dict[str, Any]) -> dict[str, Any] | None:
    """What FlexGrid would ask for today on these signals, in plain words (the same rules as the dataset runs: relieve
    the most expensive window, move flexible load into the cheapest hours)."""
    da = s.get("day_ahead")
    if not da:
        return None
    pw, cheap = da["peak_window"], da["cheapest_hours"]
    ratio = pw["avg_usd_mwh"] / da["day_avg_usd_mwh"] if da["day_avg_usd_mwh"] > 0 else None
    fm = s.get("fuel_mix") or {}
    return {"event_window": f"{pw['start']}-{pw['end']}",
            "call_event": bool(ratio and ratio >= 1.8),
            "why": (f"MISO's day-ahead price at the Illinois hub averages ${pw['avg_usd_mwh']:.0f}/MWh from {pw['start']} "
                    f"to {pw['end']}, {ratio:.1f}x the day's average (${da['day_avg_usd_mwh']:.0f})" if ratio else ""),
            "flexible_load_hours": [f"{c['start']}-{c['end']}" for c in cheap],
            "flexible_load_why": f"the {len(cheap)} cheapest hours of the day (${min(c['usd_mwh'] for c in cheap):.0f}-"
                                 f"{max(c['usd_mwh'] for c in cheap):.0f}/MWh): data center batch work and EV charging",
            "wind_solar_now_pct": fm.get("wind_solar_pct")}


# ------------------------------------------------------------------------------------------------ fetching
def snapshot(day: str | None = None, timeout: float = 12.0, get=None) -> dict[str, Any]:
    """Read every source (each failure is reported, the rest still works). `get(url) -> (status, text)` can be
    replaced in tests."""
    if get is None:
        import httpx

        def get(url: str):
            r = httpx.get(url, timeout=timeout, headers={"User-Agent": "FlexGrid research demo (MIT AI Studio)"})
            return r.status_code, r.text
    import json
    day = day or today_central()
    out: dict[str, Any] = {"source": "MISO public data (public-api.misoenergy.org, docs.misoenergy.org)",
                           "fetched_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                           "day": day, "errors": []}
    parsers = {"fuel_mix": parse_fuel_mix, "load": parse_load, "price_now": parse_price_now}
    for key, url in URLS.items():
        try:
            code, text = get(url)
            if code != 200:
                raise ValueError(f"HTTP {code}")
            out[key] = parsers[key](json.loads(text))
        except Exception as exc:                                    # noqa: BLE001 (report and go on)
            out[key] = None
            out["errors"].append(f"{key}: {str(exc)[:120]}")
    try:
        code, text = get(da_url(day))
        lmp = parse_day_ahead(text) if code == 200 else None
        if not lmp:
            raise ValueError(f"no {HUB} day-ahead prices for {day} (HTTP {code})")
        out["day_ahead"] = outlook(lmp, day)
    except Exception as exc:                                        # noqa: BLE001
        out["day_ahead"] = None
        out["errors"].append(f"day_ahead: {str(exc)[:120]}")
    out["plan"] = plan_from(out)
    out["ok"] = not out["errors"]
    return out


class Cache:
    """snapshot() at most once every `ttl` seconds (MISO updates every 5 minutes); the coordinator holds one."""

    def __init__(self, ttl: float = 120.0):
        self.ttl, self.at, self.value = ttl, 0.0, None

    def get(self) -> dict[str, Any]:
        if self.value is None or time.time() - self.at > self.ttl:
            self.value, self.at = snapshot(), time.time()
        return self.value
