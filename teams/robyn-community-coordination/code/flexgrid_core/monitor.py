"""Control Room data: one grid event, step by step, in the shape the monitor, the household app and the talk read.

Schema flexgrid.timeline.v1 (fields: tools/timeline.py). Two producers, one shape:
  * tools/timeline.py   the offline simulation (same code as the live agents) -> results/timeline_<day>.json
  * LiveRecorder below  inside the coordinator on the Mac, while a live event runs -> GET /monitor/state

Live sources, all reached on 127.0.0.1 of the Mac (start_mac.sh opens the ssh tunnels):
  the coordinator's own market state   offers, commitments, readings, settlement, payments, its message counts
  the mock utility       :7873/traffic messages it received (the homes send their meter readings to it directly)
  the fleet (jetson-b)  :7882         the 98 replay homes' telemetry
  jetson-a's daemon  :7880         home-tesla's telemetry and its own contract view (its app's data)
  jetson-b's daemon     :7881         home-enphase's telemetry and contract view
  flexgrid-dc (the Mac)  :7874/dc/state the ten data center agents, the large-load desk and the grid operator
                                       (dc_fleet.py), passed through as `datacenters` (schema flexgrid.dc.v1)
  a building node        :7885         its own daemon (jetson-a :7875, building.py): telemetry, contract view and
                                       AI agent like a Jetson home's, recorded in `phones` and `buildings` (never in
                                       the homes' series: a 1 MW library would hide 100 homes)
The homes' telemetry is simulation truth, shown on the monitor only: the coordinator's decisions never use it
(it sees offers, commitments and signed readings, nothing else).
"""
from __future__ import annotations

import json
import statistics
import threading
import time
from collections import Counter
from typing import Any, Callable

import httpx

from . import beckn, config
from .settlement import clamp0

SCHEMA = "flexgrid.timeline.v1"
STATES = ["idle", "offered", "committed", "precharging", "delivering", "ev_paused", "reported", "paid", "declined"]
S = {k: i for i, k in enumerate(STATES)}
PHONES = ["home-tesla", "home-enphase"]
HOME_KEYS = ("state", "soc", "import_kw", "import_cf_kw", "pv_kw", "batt_kw", "ev_kw")   # home_steps, per home and step
SERIES = ("import_kw", "import_cf_kw", "pv_kw", "load_kw", "batt_kw", "ev_kw", "stored_kwh", "stored_cf_kwh",
          "price_usd_mwh")
KINDS = {
    "discover": ("coordinator", "utility|homes", "asks for tomorrow's need / calls for offers"),
    "on_discover": ("utility|homes", "coordinator", "the need (kW per slot, price) / an offer"),
    "select": ("coordinator", "utility", "picks the need"),
    "on_select": ("utility", "coordinator", "quote"),
    "init": ("coordinator", "utility", "contract draft: meters, capacity"),
    "on_init": ("utility", "coordinator", "draft accepted"),
    "confirm": ("coordinator", "utility|homes", "sign the contract / a home's commitment"),
    "on_confirm": ("utility|homes", "coordinator", "contract ACTIVE / home ACCEPTED or DECLINED"),
    "update": ("coordinator", "utility|homes", "meters and capacity / readings are in / pre-charge notice"),
    "on_status": ("utility|homes", "coordinator|utility|homes", "baselines / signed meter reading / settlement / payment"),
}


def r1(x: float) -> float:
    """One decimal, and never -0.0 (the live and offline records must compare equal)."""
    return round(x, 1) + 0.0


def state_code(ev: dict[str, Any] | None, name: str, tel: dict[str, Any], now: str) -> int:
    """A home's state for the monitor: what the coordinator knows about it (offer, commitment, readings, payment)
    plus what the home is physically doing (pre-charging, delivering, EV paused)."""
    if not ev:
        return S["idle"]
    c = ev["commitments"].get(name)
    if ev.get("result") and name in ev["result"].get("split", {}).get("per_home", {}):
        return S["paid"]
    if c and c["status"] == "DECLINED":
        return S["declined"]
    mode = tel.get("mode")
    if mode == "deliver":
        return S["ev_paused"] if tel.get("ev_held") else S["delivering"]
    if mode == "precharge":
        return S["precharging"]
    if name in ev["readings"] and now >= ev["need"]["window_end"]:
        return S["reported"]
    if c and c["status"] in ("ACCEPTED", "SENT"):
        return S["committed"]
    if name in ev["offers"]:
        return S["offered"]
    return S["idle"]


def household_money(ev: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Per home that sent readings: what FlexGrid paid it, what the same home would be paid on its own under the same
    Peak Time Rewards credit (same Rider PTR baseline, same price per kWh, no agent: its meter as in the no-event
    twin), and its real and measured reduction. The twin's usage travels in each reading (truth.usage_without_event_kw,
    simulation truth); the coordinator never uses it. Partial while readings arrive."""
    prices = {str(s["id"]): s["price"] for s in ev["need"]["slots"]}
    base = ev.get("baselines") or {}
    split = (ev.get("result") or {}).get("split", {}).get("per_home", {})
    out: dict[str, dict[str, Any]] = {}
    for n, rs in ev["readings"].items():
        b = base.get(n) or {}
        alone = true = meas = 0.0
        has_alone = False
        for sid, r in rs.items():
            truth = r.get("truth") or {}
            true += (truth.get("delivered_true_kw") or 0.0) * 0.5
            bl = (b.get(str(sid)) or {}).get("BASELINE")
            p = prices.get(str(sid))
            if bl is None or p is None:
                continue
            meas += clamp0(bl - r["usage_kw"]) * 0.5
            tw = truth.get("usage_without_event_kw")
            if tw is not None:
                alone += clamp0(bl - tw) * 0.5 * p
                has_alone = True
        out[n] = {"alone_usd": round(alone, 4) + 0.0 if has_alone else None, "true_kwh": round(true, 3) + 0.0,
                  "measured_kwh": round(meas, 3) + 0.0, "paid_usd": round(split.get(n, 0.0), 4) if split else None}
    return out


def _dc_available() -> bool:
    from . import datacenter as dc
    return dc.available()


def labels() -> dict[str, str]:
    """The words the pages use for the dataset's grid, price and program (Illinois 2023 or Massachusetts 2026)."""
    m = config.dataset_meta()
    if m.get("name") == "il2023":
        return {"price": "MISO real-time price", "price_where": "MISO real-time price, Illinois hub, 2023 (dataset)",
                "program": "Ameren Peak Time Rewards",
                "baseline": "Ameren Peak Time Rewards (Rider PTR): the 3 highest of the 14 previous days that are not "
                            "holidays or event days", "baseline_short": "the Ameren Peak Time Rewards baseline"}
    return {"price": "ISO New England real-time price", "price_where": m.get("price_label", "ISO New England real-time price"),
            "program": "a ConnectedSolutions-style program",
            "baseline": "the 3 highest of the 14 previous days that are not holidays or event days (the Rider PTR "
                        "method, kept for comparison with the Illinois runs; ConnectedSolutions itself pays a "
                        "battery on its measured output)", "baseline_short": "the 3-of-14 baseline"}


RAW_KEYS = ("import_kw", "import_cf_kw")          # unrounded per-home import, kept for the money numbers only


def household_energy(steps: list[str], raw: dict[str, dict[str, list]], price: list[float], names: list[str],
                     day: str, paid: dict[str, float] | None = None) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    """Per home, its electricity on the event day valued every 15 minutes at the MISO real-time price (the price the
    simulated homes see): + = it earned (sold more than it bought), - = it paid. 'with' is the home with FlexGrid plus
    its event payment once settled; 'without' is its no-event twin (each device on its own app). Computed from the
    unrounded per-home import (`raw`), so the community totals returned with it are exactly the sum over the homes."""
    idx = [i for i, t in enumerate(steps) if t.startswith(day)]
    out, tot = {}, {"energy_with": 0.0, "energy_without": 0.0}
    for n in names:
        imp, cf = raw["import_kw"].get(n) or [], raw["import_cf_kw"].get(n) or []
        e_with = -sum(imp[i] * price[i] for i in idx if i < len(imp)) * 0.25 / 1000
        e_wo = -sum(cf[i] * price[i] for i in idx if i < len(cf)) * 0.25 / 1000
        tot["energy_with"] += e_with
        tot["energy_without"] += e_wo
        pay = (paid or {}).get(n)
        out[n] = {"energy_with": round(e_with, 4) + 0.0, "energy_without": round(e_wo, 4) + 0.0,
                  "with": round(e_with + (pay or 0.0), 4) + 0.0, "without": round(e_wo, 4) + 0.0}
    return out, tot


def household_how() -> str:
    return (f"What the homes' electricity earned (+) or cost (-) on the event day, every 15 minutes at the {labels()['price']} "
            "the simulated homes see. Without FlexGrid: each home's no-event twin, every device "
            "on its own app. With FlexGrid: the home as it ran, plus its event payment once settled.")


def alone_how() -> str:
    return (f"What the same homes would be paid on their own under {labels()['program']}, at the same price per kWh "
            "and with the same baseline (the 3 highest of the 14 previous days that are not holidays or event days), "
            "with no agent: each meter as in its no-event twin. Their real reduction is then zero; that payment comes "
            "from the baseline sitting above what the homes would have used anyway.")


def benefits(steps: list[str], s: dict[str, list[float]], need: dict[str, Any], result: dict[str, Any] | None,
             money: dict[str, dict[str, Any]] | None = None, energy: dict[str, float] | None = None) -> dict:
    """The power-system numbers, each with its definition (shown next to it on the pages). Partial while live."""
    w0, w1 = need["window_start"], need["window_end"]
    day = w0[:10]
    iw = [i for i, t in enumerate(steps) if w0 <= t < w1]
    ie = [i for i, t in enumerate(steps) if w1 <= t <= f"{day} 23:45"]
    iday = [i for i, t in enumerate(steps) if t.startswith(day)]
    out: dict[str, Any] = {}
    if iw:
        out["window_peak_kw"] = {
            "with": round(max(s["import_kw"][i] for i in iw), 1), "without": round(max(s["import_cf_kw"][i] for i in iw), 1),
            "how": "Highest 15-minute community grid import inside the event window; 'without' is each home's "
                   "no-event twin (the same home simulated without the contract)."}
        out["energy_shifted_kwh"] = {
            "value": round(sum(s["import_cf_kw"][i] - s["import_kw"][i] for i in iw) * 0.25, 1),
            "how": "Grid energy the community did not draw inside the window, against its no-event twin (sum over "
                   "the window of twin import minus actual import)."}
        pw = [s["price_usd_mwh"][i] for i in iw]
        out["window_price_usd_mwh"] = {"mean": round(statistics.mean(pw), 1), "max": round(max(pw), 1),
                                       "how": f"{labels()['price_where']}, during the window."}
    if ie:
        out["evening_peak_kw"] = {
            "with": round(max(s["import_kw"][i] for i in ie), 1), "without": round(max(s["import_cf_kw"][i] for i in ie), 1),
            "how": "Highest community import from the end of the window to midnight. The rebound guard keeps this "
                   "from rising when batteries refill and EVs restart."}
    if iday:
        if energy:                                   # the sum over the homes (household_energy), unrounded
            cost, cost_cf = -energy["energy_with"], -energy["energy_without"]
        else:
            cost = sum(s["import_kw"][i] * s["price_usd_mwh"][i] for i in iday) * 0.25 / 1000
            cost_cf = sum(s["import_cf_kw"][i] * s["price_usd_mwh"][i] for i in iday) * 0.25 / 1000
        out["wholesale_cost_usd"] = {
            "with": round(cost, 2), "without": round(cost_cf, 2), "saved": round(cost_cf - cost, 2),
            "how": f"The community's grid energy on the event day priced at the {labels()['price']} of each 15 "
                   "minutes, with the event and for the no-event twin (negative = the community sold more than it bought)."}
        paid = (result or {}).get("utility_pays_usd")
        out["household_usd"] = {"with": round(-cost + (paid or 0.0), 2) + 0.0, "without": round(-cost_cf, 2) + 0.0,
                                "energy_with": round(-cost, 2) + 0.0, "energy_without": round(-cost_cf, 2) + 0.0,
                                "paid": paid, "how": household_how()}
    if result:
        out["paid_usd"] = {"total": result.get("utility_pays_usd"), "homes": result.get("homes_paid"),
                           "median": result.get("median"), "max": result.get("max"),
                           "how": f"DEG demand-flex settlement: ${need['slots'][0]['price']:g} per kWh of capacity offered and delivered in each "
                                  "30-minute slot, split among the homes pro rata to their measured delivery."}
        shares = {n: v.get("paid_usd") for n, v in (money or {}).items() if v.get("paid_usd") is not None}
        if shares:
            zero = sorted(n for n, v in shares.items() if not v > 0)
            out["paid_usd"].update({
                "homes": len(shares) - len(zero), "homes_in_split": len(shares), "zero_share": zero,
                "how_zero": ("Homes in the split whose share is $0: the utility's baseline (the 3 highest of the 14 "
                             "previous days) measured no reduction at their meters, so the split by measured delivery "
                             "gave them nothing, even if they truly delivered.") if zero else None})
        alone = [v["alone_usd"] for v in (money or {}).values() if v.get("alone_usd") is not None]
        if alone:
            out["paid_usd"].update({
                "alone_total": round(sum(alone), 2), "alone_median": round(statistics.median(alone), 4),
                "alone_homes": len(alone), "true_kwh_total": round(sum(v["true_kwh"] for v in money.values()), 1),
                "how_alone": alone_how()})
        out["delivered_kwh"] = {"promised": result.get("promised_kwh"), "true": result.get("true_kwh"),
                                "measured": result.get("measured_kwh"),
                                "how": "Promised = capacity sold x 0.5 h per slot. True = twin minus actual meter. "
                                       f"Measured = {labels()['baseline_short']} (the 3 highest of the 14 "
                                       "previous days that are not holidays or event days) minus the meter: what the "
                                       "utility pays on, capped at the capacity sold."}
    return out


def market_step(ev: dict[str, Any] | None) -> dict[str, Any]:
    m = {"phase": ev["phase"] if ev else "waiting", "offers": 0, "accepted": 0, "declined": 0, "sent": 0,
         "readings": 0, "capacity_kw": None, "seller_usd": None, "paid_homes": 0}
    if ev:
        m["offers"] = len(ev["offers"])
        for c in ev["commitments"].values():
            m["accepted" if c["status"] == "ACCEPTED" else "declined" if c["status"] == "DECLINED" else "sent"] += 1
        m["readings"] = sum(len(v) for v in ev["readings"].values())
        cap = ev.get("capacity_offered_final") or ev.get("capacity_offered")
        m["capacity_kw"] = [cap.get(s["id"], cap.get(str(s["id"]))) for s in ev["need"]["slots"]] if cap else None
        if ev.get("result"):
            m["seller_usd"] = round(ev["result"]["seller_usd"], 2)
            per_home = ev["result"].get("split", {}).get("per_home", {})
            m["paid_homes"] = sum(1 for v in per_home.values() if v > 0)      # homes that got a share of the payment
            m["split_homes"] = len(per_home)                                  # homes in the split, a $0 share included
    return m


def event_static(ev: dict[str, Any], fleet: dict[str, dict], names: list[str], times: dict[str, dict]) -> dict:
    need = ev["need"]
    split = (ev.get("result") or {}).get("split", {}).get("per_home", {})
    money = household_money(ev)
    homes = []
    for n in names:
        f = fleet[n]
        o = ev["offers"].get(n) or {}
        c = ev["commitments"].get(n) or {}
        cc = c.get("contract") or {}
        hm = money.get(n) or {}
        homes.append({"name": n, "county": f["county"], "county_id": f["county_id"], "host": f["host"], "kind": f["kind"],
                      "battery_kwh": f["battery"]["kwh"], "battery_kw": f["battery"]["kw"], "has_ev": f["has_ev"],
                      "offer_kw": [s["kw_offered"] for s in o.get("slots", [])], "offer_kwh": o.get("energy_kwh"),
                      "commit_kw": {str(s["id"]): s["kw"] for s in cc.get("commitments", [])}, "status": c.get("status"),
                      "reason": c.get("reason"), "ev_resume_after": cc.get("ev_resume_after"),
                      "paid_usd": round(split.get(n, 0.0), 4) if split else None,
                      "alone_usd": hm.get("alone_usd"), "true_kwh": hm.get("true_kwh"), "measured_kwh": hm.get("measured_kwh"),
                      "times": times.get(n, {})})
    cs = ev.get("clear_summary") or {}
    return {
        "event": {"id": need["event_id"], "day": need["window_start"][:10], "window": [need["window_start"], need["window_end"]],
                  "commit_by": need["commit_by"], "price_usd_per_kwh": need["slots"][0]["price"],
                  "slots": [{"id": s["id"], "start": s["start"], "kw_requested": s["kw_requested"]} for s in need["slots"]],
                  "buyer": "Utility or ENGIE (mock, public data)",
                  "baseline": labels()["baseline"]},
        "homes": homes,
        "clearing": {k: cs.get(k) for k in ("capacity_offered_kw", "requested_kw", "offered_kw_raw", "ev_pause_kw",
                                            "battery_share", "rebound_guard", "margin")},
        "settlement": {"components": (ev.get("settlement") or {}).get("components"),
                       "seller_usd": (ev.get("result") or {}).get("seller_usd"),
                       "policy": "beckn/DEG demand-flex contract policy (rego), ported"},
    }


# ---------------------------------------------------------------------------------------------- live recorder
LOG_SKIP = {"h1_langflow", "c1_langflow", "c2_langflow", "report_skipped_stale", "langflow_pruned"}
TEL_KEYS = ("pv_kw", "load_kw", "batt_kw", "ev_kw", "grid_kw", "grid_cf_kw", "soc_pct", "soc_cf_pct", "mode",
            "ev_held", "ev_plugged", "ev_need_kwh", "ev_deadline", "price_usd_mwh", "ts",
            # the real-time design (control_mode "rolling"): what it owes now, any dispatch on top, the house's warmth
            "control", "owed_kw", "rt_extra_kw", "t_dev_c", "t_in_c", "cool_kw", "override")
BLD_KEYS = ("light_kw", "light_dim_kw", "light_on_kw", "light_dim_max", "node", "cool_own_day_kw", "occupied")
# a building node's own (building.py): its lighting, its limit, its cooling on its own day, open or closed


class LiveRecorder:
    """Records the coordinator's current event every replay step (polled every 2 s), for GET /monitor/state."""

    def __init__(self, market, cfg: dict[str, Any], event: Callable, log_source: Callable[[], list[dict]]):
        self.m, self.cfg, self.event, self.log_source = market, cfg, event, log_source
        fleet = json.loads(config.data_path("fleet_v2.json").read_text())["homes"]
        self.fleet = {h["name"]: h for h in fleet}
        self.names = [h["name"] for h in fleet]
        self.src = {"fleet": cfg.get("monitor_fleet_url", "http://127.0.0.1:7882"),
                    "home-tesla": cfg.get("monitor_tesla_url", "http://127.0.0.1:7880"),
                    "home-enphase": cfg.get("monitor_enphase_url", "http://127.0.0.1:7881"),
                    "utility": f"http://127.0.0.1:{cfg.get('utility_port', 7873)}",
                    "datacenters": cfg.get("monitor_dc_url", f"http://127.0.0.1:{cfg.get('dc_port', 7874)}")}
        from . import building                         # the building nodes this config enables (cfg "buildings")
        self.bnodes = building.enabled(cfg)
        self.bnames = [b["name"] for b in self.bnodes]
        self.bparams = {}
        for b in self.bnodes:
            self.src[b["name"]] = building.monitor_url(b, cfg)
            try:
                self.bparams[b["name"]] = building.params(b["name"])
            except (KeyError, OSError, ValueError):
                self.bparams[b["name"]] = {}
        self.polled = [p for p in PHONES if p in self.fleet] + self.bnames     # the nodes with a daemon of their own, polled one by one: the new data has no named Jetson homes
        self.http = httpx.Client(timeout=8)
        self.lock = threading.Lock()
        self.dc: dict[str, Any] | None = None          # the data centers' record (flexgrid-dc), passed through
        self.coord_brain = None                        # the coordinator's AI agent (brain.CoordinatorBrain), if any
        self.buyer = None                              # the buyer's AI agent (buyer_agent.BuyerAgent), if any
        self.coord_cached: dict[str, Any] | None = None   # their state, refreshed by poll() outside the locks
        self.buyer_cached: dict[str, Any] | None = None
        self.util_lag, self.util_lag_at = 0.0, 0.0      # the utility's inbox: seconds behind, when last read
        self.reset()

    def reset(self, event_id: str | None = None, keep_chat: bool = False) -> None:
        """A fresh record. keep_chat (the next event began, the run goes on): the agents' conversation is kept, since
        the next day's market happens during this day's run and the chat is one timeline; a restarted run starts it
        afresh."""
        kept = (self.chat_log[-3000:], self.log_at, self.agents_since, self.home_agents, self.pay_wall, self.seen) \
            if keep_chat and hasattr(self, "chat_log") else None
        self.event_id = event_id
        self.steps: list[str] = []
        self.series: dict[str, list[float]] = {k: [] for k in SERIES}
        self.h = {k: {n: [] for n in self.names} for k in HOME_KEYS}
        self.raw = {k: {n: [] for n in self.names} for k in RAW_KEYS}
        self.market_steps: list[dict] = []
        self.messages: list[dict] = []
        self.phones: dict[str, list[dict]] = {p: [] for p in self.polled}
        self.phone_raw: dict[str, dict] = {}
        self.times: dict[str, dict[str, str]] = {}
        self.log: list[dict] = []
        self.prev_traffic: Counter = Counter()
        self.tel_seen: dict[str, dict[str, dict]] = {}       # home -> {ts: telemetry}, the last few steps
        self.fresh = 0                                       # homes whose telemetry for the current step has arrived
        self.fleet_at = 0.0
        self.fleet_ts: str | None = None                     # the fleet's dataset time at its last answer
        self.sources_ok: dict[str, bool] = {}
        self.last_poll = 0.0
        self.log_at = {"coordinator": time.time(), "utility": time.time()}
        # the AI agents of the nodes with a daemon of their own (the two Jetson homes, the building nodes) in this run:
        # their decisions and their owners' / facility managers' chats (agents_feed)
        self.agents_since = time.time()
        self.home_agents: dict[str, dict[str, Any]] = {p: {"enabled": False, "decisions": {}, "chat": {}}
                                                       for p in self.polled}
        self.pay_wall: dict[tuple[str, str], float] = {}
        self.seen: dict[tuple, tuple[float, str]] = {}     # agents_feed.market_lines: when each line was first seen
        self.chat_log: list[dict] = []                     # self.log's entries for the chat, kept across events
        if kept:
            self.chat_log, self.log_at, self.agents_since, self.home_agents, self.pay_wall, self.seen = kept

    # -- sources
    def _get(self, key: str, path: str) -> Any:
        try:
            r = self.http.get(self.src[key] + path)
            r.raise_for_status()
            self.sources_ok[key] = True
            return r.json()
        except Exception:
            self.sources_ok[key] = False
            return None

    def _telemetry(self, step: str) -> dict[str, dict[str, Any]]:
        """Each home's telemetry for `step`: the reading stamped with that step if one has arrived, else the newest
        older one (a home a few seconds behind the coordinator's clock), never a newer one."""
        f = None                                  # ~15 kB for 98 homes over jetson-b's slow link: every 5 s at most
        if time.time() - self.fleet_at >= float(self.cfg.get("monitor_fleet_seconds", 5.0)):
            self.fleet_at = time.time()
            # with the steps since the last answer (a fast-forwarded live clock moves several steps in 5 s)
            f = self._get("fleet", "/telemetry" + (f"?since={self.fleet_ts}" if self.fleet_ts else ""))
        if f:
            keys = f.get("keys")
            for n, rows in (f.get("recent") or {}).items():
                for t in rows:
                    self._keep(n, dict(zip(keys, t)) if keys else t)
            for n, t in (f.get("homes") or {}).items():
                self._keep(n, dict(zip(keys, t)) if keys else t)
            if f.get("now"):
                self.fleet_ts = f["now"]
        for p in self.polled:
            keys = TEL_KEYS + BLD_KEYS if p in self.bnames else TEL_KEYS
            d = self._get(p, "/telemetry")
            if d and d.get("telemetry"):
                E = (d.get("limits") or {}).get("battery_kwh")
                for h0 in d.get("history") or []:            # the Jetson's last readings, one per step it saw
                    if isinstance(h0, dict) and h0.get("ts"):
                        self._keep(p, {**{k: h0.get(k) for k in keys}, "E": E})
                t = {k: d["telemetry"].get(k) for k in keys}
                t["E"] = E
                self._keep(p, t)
                v2 = self._get(p, "/v2/status")
                if v2:
                    self.phone_raw[p] = v2
                    for pay in v2.get("payments") or []:             # when each payment reached the home (the chat)
                        self.pay_wall.setdefault((p, str(pay.get("event_id"))), time.time())
            self._agent_poll(p)
        out = {}
        for n, by_ts in self.tel_seen.items():
            ok = [ts for ts in by_ts if ts <= step]
            if ok:
                out[n] = by_ts[max(ok)]
        self.fresh = sum(1 for n, by_ts in self.tel_seen.items() if step in by_ts and n not in self.bnames)
        return out

    def _agent_poll(self, p: str) -> None:
        """This Jetson home's AI agent: its new decisions and owner-chat lines since the last poll (a few hundred bytes
        on most polls). The home forgets its decisions when a run starts (demo/reset); the owner's chat is kept there,
        so only the lines from this run are taken."""
        ha = self.home_agents[p]
        with self.lock:
            walls = [d.get("wall", 0.0) for d in ha["decisions"].values()] + [m.get("wall", 0.0) for m in ha["chat"].values()]
        since = max(walls) if walls else 0.0
        try:
            r = self.http.get(self.src[p] + f"/agent/state?since={since:.3f}", timeout=4)
            st = r.json() if r.status_code == 200 else None
        except Exception:                                      # an older daemon, or the Jetson is away
            st = None
        if not isinstance(st, dict):
            return
        with self.lock:
            ha["enabled"] = bool(st.get("enabled"))
            for k in ("llm", "wishes", "label", "why"):
                ha[k] = st.get(k)
            for d in st.get("decisions") or []:
                ha["decisions"][d.get("n")] = d
            for m in st.get("chat") or []:
                if m.get("wall", 0.0) >= self.agents_since - 30:
                    ha["chat"][(m.get("wall"), m.get("role"))] = m

    def agents_state(self, ev: dict[str, Any] | None) -> dict[str, Any]:
        """GET /monitor/state -> "agents" (callers hold self.lock)."""
        from . import agents_feed
        homes = {p: {**{k: v for k, v in ha.items() if k not in ("decisions", "chat")},
                     "decisions": sorted(ha["decisions"].values(), key=lambda d: d.get("wall", 0.0)),
                     "chat": sorted(ha["chat"].values(), key=lambda m: m.get("wall", 0.0))}
                 for p, ha in self.home_agents.items()}
        out = agents_feed.build(self.chat_log, homes, ev, self.coord_cached, self.buyer_cached, self._phones_v2(),
                                self.cfg, seen=self.seen, events=list(self.m.events.values()), dc=self.dc,
                                buildings=self.bnodes)
        # each Jetson home's agent as GET /agent/{home}/state gives it (the phone's Agent tab), for a recorded replay
        out["homes"] = {p: {"enabled": bool(st.get("enabled")), "home": p, "label": st.get("label"), "llm": st.get("llm"),
                            "wishes": st.get("wishes"), "decisions": st["decisions"][-30:], "chat": st["chat"][-30:]}
                        for p, st in homes.items() if st.get("enabled")}
        return out

    def _phones_v2(self) -> dict[str, dict[str, Any]]:
        return {p: {"payments": [{**pay, "_wall": self.pay_wall.get((p, str(pay.get("event_id"))))}
                                 for pay in (v2.get("payments") or [])]} for p, v2 in self.phone_raw.items()}

    def _keep(self, name: str, t: dict[str, Any]) -> None:
        by_ts = self.tel_seen.setdefault(name, {})
        if t.get("ts"):
            by_ts[t["ts"]] = t
            if len(by_ts) > 16:
                for old in sorted(by_ts)[:-16]:
                    by_ts.pop(old, None)

    def _tel_at(self, step: str) -> dict[str, dict[str, Any]]:
        """Each home's reading for `step`, or the newest older one."""
        out = {}
        for n, by_ts in self.tel_seen.items():
            ok = [ts for ts in by_ts if ts <= step]
            if ok:
                out[n] = by_ts[max(ok)]
        return out

    def _traffic(self) -> Counter:
        c = Counter()
        for k, v in getattr(self.m, "traffic", {}).items():
            c[k] += v
        u = self._get("utility", "/traffic")
        for k, v in (u or {}).get("received_from_homes", {}).items():
            c[k] += v
        if isinstance(u, dict) and u.get("inbox_lag_s") is not None:     # how far behind the utility's inbox is: the
            self.util_lag, self.util_lag_at = float(u["inbox_lag_s"]), time.time()   # live clock waits for it too
        return c

    def _current_event(self, now: str) -> dict[str, Any] | None:
        """The event being recorded: the earliest one whose evening (6 h after its window) is not over yet. On an
        event day the next day's need arrives at 10:00; it waits its turn instead of resetting the record mid-event
        (six of the ten event days are followed by another: 29 Jun, 26-27 Jul, 21-23 Aug)."""
        evs = list(self.m.events.values())
        live = [e for e in evs if now <= beckn.add_minutes(e["need"]["window_end"], 6 * 60)]
        if live:
            return min(live, key=lambda e: e["need"]["window_start"])
        return evs[-1] if evs else None

    # -- one poll
    def poll(self) -> None:
        now = self.m.now()
        step = now[:14] + f"{int(now[14:16]) // 15 * 15:02d}"
        with self.m.lock:
            ev = self._current_event(now)
            ev_id = ev["need"]["event_id"] if ev else None
        with self.lock:
            if self.steps and (step < self.steps[-1] or (self.event_id and ev_id != self.event_id)):
                # demo restarted (clock went back) or the next event began (the chat goes on)
                self.reset(ev_id, keep_chat=step >= self.steps[-1])
            self.event_id = ev_id
            new = not self.steps or step != self.steps[-1]
        tel = self._telemetry(step)
        traffic = self._traffic()
        coord_log = list(self.log_source())
        util_log = self._get("utility", "/events?n=100") or []
        try:                                         # the Mac's AI agents (their LLM status may need a quick HTTP check)
            self.coord_cached = self.coord_brain.state() if self.coord_brain is not None else None
            self.buyer_cached = self.buyer.status() if self.buyer is not None else None
        except Exception as exc:                     # noqa: BLE001
            self.event("monitor_error", error=f"agents: {str(exc)[:160]}")
        if self.cfg.get("monitor_datacenters", True) and _dc_available():
            # the data centers' record about the same day (when this dataset has the data center node)
            day = ev_id and ev and ev["need"]["window_start"][:10]
            d = self._get("datacenters", "/dc/state" + (f"?day={day}" if day else ""))
            self.dc = d if isinstance(d, dict) and d.get("schema") == "flexgrid.dc.v1" else None
        with self.m.lock, self.lock:
            ev = self.m.events.get(ev_id) if ev_id else None
            if not self.steps or step != self.steps[-1]:
                new = True               # reset meanwhile (a run started from the Control Room): a fresh first step
            if new:
                gap = []
                if self.steps:
                    s0 = beckn.add_minutes(self.steps[-1], 15)
                    while s0 < step and len(gap) < 48:
                        gap.append(s0)
                        s0 = beckn.add_minutes(s0, 15)
                for s0 in gap:           # steps no poll saw (a fast-forwarded live clock): from the readings kept
                    self._add_step(s0)
                    self._fill(len(self.steps) - 1, s0, self._tel_at(s0), ev, ev_id)
                    self.messages[-1] = {}
                self._add_step(step)
            i = len(self.steps) - 1
            self._fill(i, step, tel, ev, ev_id)
            delta = {k: traffic[k] - self.prev_traffic.get(k, 0) for k in traffic if traffic[k] > self.prev_traffic.get(k, 0)}
            if new:
                self.messages[i] = delta
            else:
                for k, v in delta.items():
                    self.messages[i][k] = self.messages[i].get(k, 0) + v
            self.prev_traffic = traffic
            # the agents' chat: the market's own lines (the Jetson homes' offers, answers, readings ...), stamped
            # with when this recorder first saw each
            try:                                     # every event's: the next day's market runs during this day's
                from . import agents_feed
                pv2 = self._phones_v2()
                for e in self.m.events.values():
                    for key in agents_feed.market_keys(e, pv2, agents_feed.JETSONS + tuple(self.bnames)):
                        self.seen.setdefault(key, (time.time(), step))
                for key in agents_feed.dc_keys(self.dc):     # the data centers' market (flexgrid-dc's record)
                    self.seen.setdefault(key, (time.time(), step))
            except Exception as exc:                 # noqa: BLE001  never let the chat break the record
                self.event("monitor_error", error=f"agents chat: {str(exc)[:160]}")
            # what the coordinator and the utility did since the last poll, stamped with this step
            for who, rows in (("coordinator", coord_log), ("utility", util_log)):
                for x in rows:
                    if not isinstance(x.get("at"), (int, float)):
                        continue
                    if x["at"] > self.log_at[who] and x.get("kind") not in LOG_SKIP:
                        entry = {**{k: v for k, v in x.items() if k != "at"}, "who": who, "sim": step, "wall": x["at"]}
                        self.log.append(entry)
                        self.chat_log.append(entry)
                        self.log_at[who] = x["at"]
        self.last_poll = time.time()

    def _add_step(self, step: str) -> None:
        self.steps.append(step)
        for k in SERIES:
            self.series[k].append(0.0)
        for k in self.h:
            for n in self.names:
                self.h[k][n].append(0)
        for k in self.raw:
            for n in self.names:
                self.raw[k][n].append(0.0)
        self.market_steps.append({})
        self.messages.append({})
        for p in self.polled:
            self.phones[p].append({})

    def _fill(self, i: int, step: str, tel: dict[str, dict[str, Any]], ev: dict[str, Any] | None, ev_id: str | None) -> None:
        """Step i's numbers from each home's reading (callers hold both locks)."""
        sums = dict.fromkeys(SERIES, 0.0)
        for n in self.names:
            t = tel.get(n)
            if not t:
                continue
            E = t.get("E") or self.fleet[n]["battery"]["kwh"]
            sums["import_kw"] += -(t.get("grid_kw") or 0.0)
            sums["import_cf_kw"] += -(t.get("grid_cf_kw") or 0.0)
            sums["pv_kw"] += t.get("pv_kw") or 0.0
            sums["load_kw"] += t.get("load_kw") or 0.0
            sums["batt_kw"] += t.get("batt_kw") or 0.0
            sums["ev_kw"] += t.get("ev_kw") or 0.0
            sums["stored_kwh"] += (t.get("soc_pct") or 0.0) / 100 * E
            sums["stored_cf_kwh"] += (t.get("soc_cf_pct") or 0.0) / 100 * E
            if t.get("price_usd_mwh") is not None:
                sums["price_usd_mwh"] = t["price_usd_mwh"]
            code = state_code(ev, n, t, step)
            self.h["state"][n][i] = code
            self.h["soc"][n][i] = int(round(t.get("soc_pct") or 0))
            self.h["import_kw"][n][i] = r1(-(t.get("grid_kw") or 0.0))
            self.h["import_cf_kw"][n][i] = r1(-(t.get("grid_cf_kw") or 0.0))
            self.h["pv_kw"][n][i] = r1(t.get("pv_kw") or 0.0)
            self.raw["import_kw"][n][i] = -(t.get("grid_kw") or 0.0)
            self.raw["import_cf_kw"][n][i] = -(t.get("grid_cf_kw") or 0.0)
            self.h["batt_kw"][n][i] = r1(t.get("batt_kw") or 0.0)
            self.h["ev_kw"][n][i] = r1(t.get("ev_kw") or 0.0)
            at = self.times.setdefault(n, {})
            for key, cond in (("offer", code >= S["offered"]), ("accept", S["committed"] <= code < S["declined"]),
                              ("precharge", code == S["precharging"]),
                              ("deliver", code in (S["delivering"], S["ev_paused"])), ("paid", code == S["paid"])):
                if cond and key not in at:
                    at[key] = step
        for k in SERIES:
            self.series[k][i] = round(sums[k], 2) + 0.0
        self.market_steps[i] = market_step(ev)
        for p in self.polled:
            t = tel.get(p) or {}
            v2 = self.phone_raw.get(p) or {}
            cid = f"{ev_id}:{p}"
            c = (v2.get("contracts") or {}).get(cid) or {}
            pay = next((x for x in v2.get("payments") or [] if x.get("event_id") == ev_id), None)
            bld = p in self.bnames                     # a building node: not one of the 100 homes' series
            code = (state_code(ev, p, t, step) if t else S["idle"]) if bld else self.h["state"][p][i]
            if bld and t:
                at = self.times.setdefault(p, {})
                for key, cond in (("offer", code >= S["offered"]), ("accept", S["committed"] <= code < S["declined"]),
                                  ("deliver", code == S["delivering"]), ("paid", code == S["paid"])):
                    if cond and key not in at:
                        at[key] = step
            self.phones[p][i] = {
                **{k: t.get(k) for k in (TEL_KEYS + BLD_KEYS if bld else TEL_KEYS) if k not in ("ts", "price_usd_mwh")},
                "state": STATES[code], "contract": c.get("status"),
                "reported": len(c.get("reported") or []),
                "paid_usd": round(float(pay.get("payment") or 0), 2) if pay else None,
                "reputation": v2.get("reputation")}

    def loop(self, dump: str | None = None) -> None:
        """Poll every 2 s. With `dump`, also write the state to that file once per step (a record of the live run
        that can be replayed or compared with the offline simulation later), and when the next event begins, keep
        the finished one's last record (keep_record)."""
        import os
        last, last_ev = None, None
        while True:
            try:
                self.poll()
                if dump and last_ev and self.event_id != last_ev:
                    self.keep_record(dump)                   # the dump still holds the finished event's last step
                    last_ev = None
                if dump and self.steps and self.steps[-1] != last:
                    last = self.steps[-1]
                    tmp = dump + ".tmp"
                    with open(tmp, "w") as f:
                        json.dump(self.state(), f, separators=(",", ":"))
                    os.replace(tmp, dump)
                    last_ev = self.event_id
            except Exception as exc:
                self.event("monitor_error", error=str(exc)[:200])
            time.sleep(float(self.cfg.get("monitor_poll_seconds", 2.0)))

    def keep_record(self, dump: str) -> str | None:
        """A live event that ran to its settlement, kept for the Control Room's replay: results/live_timeline_<day>.json
        (python3 viz/live/build.py --timeline results/live_timeline_<day>.json). The live run itself, with the AI
        agents' real conversation, their decisions, the homes' agents for the phone, and the data centers; not a
        re-simulation. A run stopped before its settlement is not kept (an earlier full record stays)."""
        import os
        if not self.cfg.get("monitor_keep_runs", True):
            return None
        try:
            st = json.loads(open(dump).read())
            ev = st.get("event") or {}
            day = str(ev.get("day") or "")[:10]
            settled = (st.get("summary") or {}).get("utility_pays_usd") is not None
            if not (st.get("live") and day and settled and st.get("steps")):
                return None
            st["live"] = False                               # a record now: the pages replay it
            st["recorded"] = {"when": time.strftime("%-d %b %Y, %H:%M"),
                              "how": "the coordinator's live recorder (flexgrid_core/monitor.py), its last step of the "
                                     "event, kept when the next event began"}
            path = config.REPO_DIR / "results" / f"live_timeline_{day}.json"
            tmp = str(path) + ".tmp"
            with open(tmp, "w") as f:
                json.dump(st, f, separators=(",", ":"))
            os.replace(tmp, path)
            self.event("monitor_kept", path=f"results/{path.name}", steps=len(st["steps"]),
                       agents_lines=len((st.get("agents") or {}).get("chat") or []))
            return str(path)
        except Exception as exc:                             # noqa: BLE001  never let this stop the recorder
            self.event("monitor_error", error=f"keep record: {str(exc)[:160]}")
            return None

    # -- the JSON the pages read
    def state(self) -> dict[str, Any]:
        with self.m.lock, self.lock:
            ev = self.m.events.get(self.event_id) if self.event_id else None
            out: dict[str, Any] = {
                "schema": SCHEMA, "live": True, "now": self.steps[-1] if self.steps else self.m.now(),
                "source": "live: the coordinator on the Mac (market state) + the homes' telemetry from the fleet on "
                          "jetson-b, the two Jetson daemons and any building node's daemon (simulation truth, never "
                          "used by the coordinator)",
                "sources_ok": dict(self.sources_ok), "homes_current": self.fresh, "states": STATES,
                "steps": list(self.steps),
                "series": {k: list(v) for k, v in self.series.items()},
                "home_steps": {k: [list(self.h[k][n]) for n in self.names] for k in self.h},
                "market": list(self.market_steps), "messages": list(self.messages), "message_kinds": KINDS,
                "log": list(self.log[-400:]), "phones": {p: list(v) for p, v in self.phones.items()},
                "event": None, "homes": [{"name": n, "county": self.fleet[n]["county"], "county_id": self.fleet[n]["county_id"],
                                           "host": self.fleet[n]["host"], "kind": self.fleet[n]["kind"],
                                           "battery_kwh": self.fleet[n]["battery"]["kwh"],
                                           "battery_kw": self.fleet[n]["battery"]["kw"], "has_ev": self.fleet[n]["has_ev"]}
                                          for n in self.names]}
            if self.dc:
                out["datacenters"] = self.dc
            if self.bnodes:                                    # the building nodes: who they are, their event so far
                out["buildings"] = [self._building_card(b, ev) for b in self.bnodes]
            try:                                               # the AI agents' conversation (the chat box)
                out["agents"] = self.agents_state(ev)
            except Exception as exc:                           # never let the chat break the record
                out["agents_error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            if ev:
                out.update(event_static(ev, self.fleet, self.names, self.times))
                split = (ev.get("result") or {}).get("split", {}).get("per_home", {})
                earn, energy = household_energy(self.steps, self.raw, self.series["price_usd_mwh"], self.names,
                                                ev["need"]["window_start"][:10], split)
                for hm in out["homes"]:
                    hm["earn_usd"] = earn.get(hm["name"])
                out["phone_detail"] = self._phone_detail(ev)
                out["summary"] = self._summary(ev, energy)
            return out

    def _building_card(self, b: dict[str, Any], ev: dict[str, Any] | None) -> dict[str, Any]:
        """A building node for the Control Room: who it is (building.py's constants and their sources), its AI agent,
        and in an event its offer (cooling and lighting parts), commitment, signed readings, baseline and payment.
        Its 5-minute record is phones[name], step by step (callers hold both locks)."""
        n = b["name"]
        p = self.bparams.get(n) or {}
        th, li, sc = p.get("thermal") or {}, p.get("lighting") or {}, p.get("schedule") or {}
        ha = self.home_agents.get(n) or {}
        card: dict[str, Any] = {
            "name": n, "kind": "building", "label": b.get("label"), "short": b.get("short"), "host": b.get("host"),
            "place": b.get("place"), "floor_area_sqft": b.get("floor_area_sqft"),
            "electricity_kwh_year": p.get("electricity_kwh_2025"), "open": sc.get("open"),
            "lighting_open_kw": li.get("open_kw"), "cooling_cap_kw": th.get("cool_cap_kw"),
            "cooling_kw_per_c": th.get("cool_kw_per_c"), "tau_h": th.get("tau_h"),
            "limits": p.get("limits"), "sources": p.get("sources"), "reachable": self.sources_ok.get(n),
            "agent": {"enabled": bool(ha.get("enabled")), "label": ha.get("label"), "policy": ha.get("wishes"),
                      "why": ha.get("why")}}
        if not ev:
            return card
        o = ev["offers"].get(n) or {}
        c = ev["commitments"].get(n) or {}
        cc = c.get("contract") or {}
        split = (ev.get("result") or {}).get("split", {}).get("per_home", {})
        hm = household_money(ev).get(n) or {}
        card.update({
            "offer": {"kw": [s["kw_offered"] for s in o.get("slots", [])],
                      "cooling_kw": [s.get("ac_kw", 0.0) for s in o.get("slots", [])],
                      "lighting_kw": [s.get("light_kw", 0.0) for s in o.get("slots", [])],
                      "energy_kwh": o.get("energy_kwh"), "confidence": o.get("confidence"),
                      "agent": o.get("agent"), "cooling": o.get("air_conditioning"), "lighting": o.get("lighting")}
            if o else None,
            "commit_kw": {str(s["id"]): s["kw"] for s in cc.get("commitments", [])}, "status": c.get("status"),
            "reason": c.get("reason"),
            "readings": {str(k): {kk: v.get(kk) for kk in ("start", "committed_kw", "usage_kw")}
                         for k, v in (ev["readings"].get(n) or {}).items()},
            "baseline_kw": {k: v.get("BASELINE") for k, v in (ev.get("baselines") or {}).get(n, {}).items()},
            "paid_usd": round(split.get(n, 0.0), 4) if split else None,
            "true_kwh": hm.get("true_kwh"), "measured_kwh": hm.get("measured_kwh"), "times": self.times.get(n, {})})
        return card

    def _phone_detail(self, ev: dict[str, Any]) -> dict[str, Any]:
        out = {}
        split = (ev.get("result") or {}).get("split", {}).get("per_home", {})
        for p in (x for x in PHONES if x in self.fleet):
            f = self.fleet[p]
            o = ev["offers"].get(p) or {}
            cc = (ev["commitments"].get(p) or {}).get("contract") or {}
            out[p] = {"home": p, "county": f["county"], "host": f["host"], "battery": f["battery"], "has_ev": f["has_ev"],
                      "offer": {"slots": o.get("slots"), "energy_kwh": o.get("energy_kwh"), "confidence": o.get("confidence"),
                                "battery_kwh_at_window": (o.get("battery") or {}).get("window_start_kwh")},
                      "contract": {k: cc.get(k) for k in ("id", "window_start", "window_end", "commitments",
                                                           "price_usd_per_kwh", "opt_out_by", "ev_resume_after",
                                                           "precharge_minutes", "settlement_rule")},
                      "readings": {str(k): {kk: v.get(kk) for kk in ("start", "committed_kw", "usage_kw", "soc_end_pct",
                                                                       "ev_held")}
                                   for k, v in (ev["readings"].get(p) or {}).items()},
                      "baseline_kw": {k: v.get("BASELINE") for k, v in (ev.get("baselines") or {}).get(p, {}).items()},
                      "paid_usd": round(split.get(p, 0.0), 4) if split else None, "times": self.times.get(p, {})}
        return out

    def _summary(self, ev: dict[str, Any], energy: dict[str, float] | None = None) -> dict[str, Any]:
        cap = ev.get("capacity_offered_final") or ev.get("capacity_offered") or {}
        res = ev.get("result") or {}
        truth = sum(r.get("truth", {}).get("delivered_true_kw", 0.0) * 0.5 for rs in ev["readings"].values()
                    for r in rs.values())
        split = res.get("split", {}).get("per_home", {})
        pays = sorted(split.values())
        result = None
        if res:
            result = {"utility_pays_usd": round(res["seller_usd"], 4), "homes_paid": res.get("paid_homes"),
                      "median": round(statistics.median(pays), 4) if pays else None,
                      "max": round(max(pays), 4) if pays else None,
                      "promised_kwh": round(sum(cap.values()) * 0.5, 1), "true_kwh": round(truth, 3),
                      "measured_kwh": res.get("delivered_kwh_measured")}
        return {"homes": len(self.names), "offers": len(ev["offers"]),
                "contracts": {"sent": len(ev["commitments"]),
                              "accepted": sum(1 for c in ev["commitments"].values() if c["status"] == "ACCEPTED"),
                              "declined": sum(1 for c in ev["commitments"].values() if c["status"] == "DECLINED")},
                "capacity_offered_kw": {str(k): v for k, v in cap.items()}, "promised_kwh": round(sum(cap.values()) * 0.5, 1),
                "delivered_kwh_true": round(truth, 3) if ev["readings"] else None,
                "delivered_kwh_measured": res.get("delivered_kwh_measured"),
                "utility_pays_usd": res.get("seller_usd"), "messages": sum(sum(m.values()) for m in self.messages),
                "benefits": benefits(self.steps, self.series, ev["need"], result, household_money(ev), energy)}
