"""A building as a FlexGrid node: the Boston Public Library's Central Library (data_ma/buildings/, made by
tools/make_ma_building.py from its measured 5-minute meter data).

It runs the homes' real-time machinery (adapters/simulated.py, rt_home.py, rt_control.py) with a building's levers:
  * cooling: the building's one-node physics (hvac.House on the building's constants), eased off or pre-cooled within
    the facility manager's limits: at most 1.5 C warmer than its own day while open, 3 C while closed, 1 C cooler
  * lighting: dimmed by at most `light_dim_max` of what is on (30 %: the facility manager's default, 0-50 %). It goes
    first, because it leaves nothing to catch up afterwards; the cooling covers the rest
  * no battery, no car, no solar (the library has none in the data)
What it forecasts (only from what it knows now): its load on the weather forecast (an hour-of-day line over its
previous 21 days, the homes' day-ahead method), its cooling the same way from its own record (which holds its own
17:00-20:00 cut, so it does not offer cooling it would not run), its lighting from its opening hours.
The guard, every 5 minutes: never warmer than the limit, the lights never dimmed past the limit, the facility manager's
"stop" first. The fans and the fresh air are never touched (FlexGrid has no lever on them).

The 2026 nodes (university, shopping center, office; stage3/nodes3.py) add a battery at the campus and three devices: fans, plugs and
EV ports, each the part the sale counts. The node agent's tools reach all of it through sim.levers (flexgrid_core/node/building_agent.py):
None means the controller may use every lever, a dict means only what the agent's calls opened. The campus battery waits for sales, so
it needs FG_BATTERY_MODE=backup.
"""
from __future__ import annotations

import csv
import gzip
import json
import threading
from collections import deque
from typing import Any

from . import devices, hvac, rt_control, rt_home
from .adapters.simulated import BACKUP, SimulatedHome
from .config import data_path

LIGHT_DIM_HARD_MAX = 0.5          # no agent or facility manager can dim more than half of what is on
DRIFT_C = ((3, 1.1), (6, 1.7), (12, 2.2))   # the cards' drift limits: 5-minute parts back, C. A plant cap or stop releases past them
LIMIT_KEYS = ("max_rise_c", "max_rise_away_c", "precool_c", "ev_keep_h", "call_battery_share")
_LOCK = threading.Lock()
_LIGHT: dict[str, dict[str, float]] = {}
_REG: dict[str, Any] | None = None


# ---------------------------------------------------------------------------------------------- the building nodes
def registry() -> list[dict[str, Any]]:
    """The building nodes of this dataset (data_ma/buildings.json), [] where there are none."""
    global _REG
    with _LOCK:
        if _REG is None:
            try:
                _REG = json.loads(data_path("buildings.json").read_text())
            except (FileNotFoundError, OSError, ValueError):
                _REG = {"buildings": []}
    return list(_REG.get("buildings", []))


def enabled(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """The building nodes this process takes part with: the names listed in cfg "buildings" (the live config, or
    tools/sim_event.py --buildings). Nothing by default, so the reference checks run as before."""
    want = cfg.get("buildings") or []
    if want in (True, "all"):
        return registry()
    return [b for b in registry() if b["name"] in want]


def params(name: str) -> dict[str, Any]:
    """A building's constants and where each comes from (data_ma/buildings/<id>.json)."""
    b = next((x for x in registry() if x["name"] == name or x["profile"] == name), None)
    if b is None:
        raise KeyError(f"no building node {name!r} in this dataset")
    return json.loads(data_path(b["params"]).read_text())


def meter_entry(b: dict[str, Any]) -> dict[str, Any]:
    """A building in the shape of a fleet_v2.json home (the utility's meter history, the simulations): no battery,
    no car, no solar."""
    return {"name": b["name"], "kind": "building", "profile": b["profile"], "battery": b["battery"],
            "has_ev": False, "export_limit_kw": b.get("export_limit_kw", 2000.0), "host": b.get("host"),
            "label": b.get("label"), "lat": (b.get("place") or {}).get("lat"), "lon": (b.get("place") or {}).get("lon")}


def nodes(cfg: dict[str, Any], homes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The community's nodes: the homes, then the building nodes this config enables."""
    names = {h["name"] for h in homes}
    return list(homes) + [meter_entry(b) for b in enabled(cfg) if b["name"] not in names]


def monitor_url(b: dict[str, Any], cfg: dict[str, Any] | None = None) -> str:
    """Where the coordinator's machine reaches this building node's daemon: cfg monitor_building_urls[name], else the
    registry's monitor_url (the Mac's tunnel to its host), else 127.0.0.1:7885."""
    return str(((cfg or {}).get("monitor_building_urls") or {}).get(b["name"]) or b.get("monitor_url")
               or "http://127.0.0.1:7885")


def _light_series(profile: str) -> dict[str, float]:
    with _LOCK:
        if profile not in _LIGHT:
            out: dict[str, float] = {}
            try:
                with gzip.open(data_path(f"{profile}_hvac.csv.gz"), "rt") as f:
                    for r in csv.DictReader(f):
                        out[r["ts_local"]] = float(r.get("light_kw") or 0.0)
            except (FileNotFoundError, OSError):
                pass
            _LIGHT[profile] = out
    return _LIGHT[profile]


class BuildingHouse(hvac.House):
    """The building's cooling lever (hvac.House on its constants) and its lighting by step (its own day)."""

    def __init__(self, p: dict[str, Any], profile: str):
        super().__init__(p["thermal"], profile)
        self.light = _light_series(profile)
        self.cut = 0.0                      # the share of its own day's cooling a plant cap or stop holds off
        self.cut_span = ("", "9")           # the steps it holds: a call made at 17:30 for 18:00 to 21:00 holds only then

    def light_kw(self, ts: str) -> float:
        return float(self.light.get(ts, 0.0))

    def cutting(self, ts: str) -> bool:
        return self.cut > 0 and self.cut_span[0] <= ts < self.cut_span[1]

    def dq_bounds(self, ts: str) -> tuple[float, float]:
        lo, hi = super().dq_bounds(ts)
        return (lo, min(hi, lo * self.cut)) if self.cutting(ts) else (lo, hi)

    def thermostat(self, x: float, ts: str, dt_h: float) -> float:
        """Stored cold sits in the building's mass and leaks away at the building's own rate. A home's air warms in minutes: with
        hours between a pre-cool and the sale, a house's rule would throw the pre-cool away."""
        if x < 0 and self.p.get("keeps_cold"):
            return 0.0
        return super().thermostat(x, ts, dt_h)


# ---------------------------------------------------------------------------------------------- its 5-minute loop
class BuildingController(rt_home.RollingController):
    """rt_home's controller with the building's lighting first: what is owed beyond what the dimmed lights give is
    planned on the cooling by rt_control's linear program, as for a home's air conditioning."""

    def __init__(self, sim, house, limits=None, forecast_fn=None):
        super().__init__(sim, house, limits, forecast_fn)
        self._dims: dict[str, float] = {}
        self._fit_cache: dict[tuple, tuple[float, float] | None] = {}
        self._dev_fc: dict[tuple[str, str], float] = {}

    def afresh(self) -> None:
        super().afresh()
        self._dims, self._fit_cache, self._dev_fc = {}, {}, {}

    def active(self, i: int) -> bool:
        """The plan stays out between a pre-cool and the sale: left alone, it spends the stored cold on the afternoon's bill."""
        return False if self.sim.hold_cold else super().active(i)

    def comfort_weight(self) -> float:
        """How many homes' worth of people the building's comfort stands for in the plan's costs: its cooling
        capacity over a home's (15 kW of heat), so 1 C warmer costs what it would in that many homes."""
        return max(1.0, (self.house.qmax if self.house else 15.0) / 15.0)

    def light_cap(self, ts: str) -> float:
        """kW of lighting that may be dimmed at this step: what is on x the facility manager's limit."""
        return self.house.light_kw(ts) * self.sim.dim_max() * self.sim.light_scale(ts) if self.house else 0.0

    def app_fc(self, ts: str) -> float:
        """rt_home's forecast of the devices, kept per device: an agent's call changes a device's share at once."""
        s, rp = self.sim, self.sim.replay
        c = self._dev_fc
        j, out = rp.idx_of(ts), 0.0
        if j is None:
            return 0.0
        now = rp.current() if s.idx is None else s.idx
        for k in s.app:
            if k in s.dev_off:
                continue
            if (ts, k) not in c:
                v = [float(s.dev_arr[k][s.dev_row, j - 96 * d]) for d in range(1, 42) if 0 <= j - 96 * d <= now][:14]
                c[(ts, k)] = sum(v) / len(v) if v else 0.0
            out += c[(ts, k)] * s.dev_share(k, ts)
        if len(c) > 6000:
            c.clear()
        return out

    def problem(self, i: int, first_dt: float = 0.25):
        p = super().problem(i, first_dt)
        p.comfort_weight = self.comfort_weight()
        self._dims = {}
        for k, t in enumerate(p.ts):
            d = min(p.target[k], self.light_cap(t)) if p.target[k] > 1e-6 else 0.0
            if d > 0:
                self._dims[t] = d
                p.load[k] = max(0.0, p.load[k] - d)           # the lights give this much of what is owed
        return p

    def decide(self, i: int, first_dt: float = 0.25) -> dict[str, Any]:
        out = super().decide(i, first_dt)
        if out.get("ok"):
            t0 = out.get("at")
            out["dim"] = round(self._dims.get(t0, 0.0), 3)
            out["dim_plan"] = {t: round(v, 2) for t, v in list(self._dims.items())[:8]}
        return out

    def capacity(self, i: int, call_from: str, call_to: str) -> dict[str, Any]:
        """The cooling's answer (rt_home) plus the lighting that is still free over the call, evenly."""
        out = super().capacity(i, call_from, call_to)
        if out.get("why") and not out.get("steps"):
            return out
        rp = self.sim.replay
        now = rp.ts(i)
        steps, t = [], max(now, call_from)
        while t < call_to and len(steps) < 64:
            steps.append(t)
            t = rt_home.add_minutes(t, 15)
        spare = min((self.light_cap(s) - self._dims.get(s, 0.0) for s in steps), default=0.0)
        cool = float(out.get("kw", 0.0))
        out.update({"kw": round(cool + max(0.0, spare), 3), "kw_cooling": round(cool, 3),
                    "kw_lighting": round(max(0.0, spare), 3)})
        return out

    def light_offer(self, window_start: str, window_end: str, confidence: float = 0.95) -> dict[str, Any]:
        """How many kW the lighting can give through tomorrow's window, evenly: its schedule is known, so the only
        uncertainty is the building's own day."""
        rp = self.sim.replay
        w0, w1 = rp.idx_of(window_start), rp.idx_of(window_end)
        if w0 is None or w1 is None:
            return {"kw": 0.0, "why": "window outside the dataset"}
        caps = [self.light_cap(rp.ts(k)) for k in range(w0, w1)]
        kw = min(caps) if caps else 0.0
        return {"kw": round(kw * confidence, 3), "raw_kw": round(kw, 3), "share": round(self.sim.dim_max(), 2),
                "lighting_on_kw": round(sum(self.house.light_kw(rp.ts(k)) for k in range(w0, w1)) / max(1, w1 - w0), 1)}

    def ac_offer(self, window_start: str, window_end: str, confidence: float = 0.8) -> dict[str, Any]:
        """rt_home's offer with the building's comfort weight. At a home's weight the plan takes a building past its limit to
        sell more: on 14 Jul 2026 the shopping center offered 540 kW of cooling and delivered 118."""
        rp = self.sim.replay
        w0, w1 = rp.idx_of(window_start), rp.idx_of(window_end)
        if w0 is None or w1 is None:
            return {"kw": 0.0, "why": "window outside the dataset"}
        ts = [rp.ts(k) for k in range(max(0, w0 - rt_home.LEAD_STEPS), min(len(rp.rows), w1 + rt_home.POST_STEPS))]
        cool, cop, occ = [], [], []
        for t in ts:
            j = rp.idx_of(t)
            t_fc = rp.row(j).get("fc_temp_c", rp.row(j)["temp_c"])
            fit = self._cool_fit(j)
            cf = self._cop(t, t_fc)
            cool.append(min(max(0.0, fit[0] + fit[1] * t_fc) if fit else 0.0, self.house.qmax / max(1.0, cf)))
            cop.append(cf)
            occ.append(int(self.house.at(t)[3]))
        n = len(ts)
        p = rt_control.Problem(ts=ts, dt=[0.25] * n, pv=[0.0] * n, load=list(cool), cool=cool, cop=cop, occ=occ, ref=list(cool), target=[0.0] * n,
                               watch=[True] * n, ua=self.house.ua, cap=self.house.cap, qmax=self.house.qmax, limits=self.said_limits(), x0=0.0,
                               comfort_weight=self.comfort_weight())
        call = [window_start <= t < window_end for t in ts]
        kw = rt_control.capacity(p, call, None)
        return {"kw": round(kw * confidence, 3), "raw_kw": round(kw, 3),
                "cooling_forecast_kw": round(sum(c for c, w in zip(cool, call) if w) / max(1, sum(call)), 3)}

    FIRM_QUANTILE = 25            # the cooling a building counts on at a step: its lower quartile over 14 days
    FIRM_DAYS = 14

    def _cool_fit(self, j: int):
        """The cooling the building can count on at this 15-minute step, as rt_home's line (a + b x outdoor C): the
        lower quartile of the same step over its previous 14 days, moved by its weather response. A home's mean line
        would offer cooling the library does not run: it cuts its own 17:00-20:00 load on about half of the summer
        days (on 2 Jul 2026 it would have offered 35 kW of cooling that was off). Only days before now; cached per step
        and day."""
        rp, h = self.sim.replay, self.house
        now = rp.current() if self.sim.idx is None else self.sim.idx
        key = (rp.ts(j)[11:16], rp.ts(now)[:10])
        if key in self._fit_cache:
            return self._fit_cache[key]
        vals, temps = [], []
        for d in range(1, 3 * self.FIRM_DAYS):
            k = j - 96 * d
            if k < 0 or len(vals) >= self.FIRM_DAYS:
                break
            if k >= now:
                continue
            r = rp.row(k)
            vals.append(h.at(r["ts_local"])[0])
            temps.append(r["temp_c"])
        out = None
        if len(vals) >= 5:
            vs = sorted(vals)
            q = vs[int(round(self.FIRM_QUANTILE / 100 * (len(vs) - 1)))]
            slope = float((self.sim.bparams.get("thermal") or {}).get("cool_kw_per_c", 0.0)) if q > 1e-6 else 0.0
            out = (q - slope * sum(temps) / len(temps), slope) if q > 1e-6 else (0.0, 0.0)
        if len(self._fit_cache) > 2000:
            self._fit_cache.clear()
        self._fit_cache[key] = out
        return out

    def _extend(self, p, ts: list[str]) -> None:
        """Lengthen the plan to a call that reaches past it: rt_home copies the last step's cooling forward, which for
        a building whose cooling follows a daily schedule (off 17:00-20:00, back on at night) is wrong; here each new
        step gets its own firm forecast, and the no-event import is the load itself (no battery)."""
        n0 = len(p.ts)
        super()._extend(p, ts)
        rp, h = self.sim.replay, self.house
        for k in range(n0, len(p.ts)):
            j = rp.idx_of(p.ts[k])
            r = rp.row(j) if j is not None else {}
            t_fc = float(r.get("fc_temp_c", r.get("temp_c", 25.0)))
            fit = self._cool_fit(j) if j is not None else None
            c = max(0.0, fit[0] + fit[1] * t_fc) if fit else 0.0
            p.cool[k] = min(p.load[k], c, h.qmax / max(1.0, self._cop(p.ts[k], t_fc)))
            p.cop[k] = self._cop(p.ts[k], t_fc)
            p.ref[k] = p.load[k] - p.pv[k]

    def _cooling(self, i: int, ts: list[str], load: list[float]) -> tuple[list[float], list[float], list[int]]:
        """The cooling's own day ahead from its own record: the hour-of-day line on the forecast temperature (its own
        evening cut included), moved by the error seen now (fading with a 2-hour half-life)."""
        rp, h = self.sim.replay, self.house
        row = rp.row(i)
        t_now = row.get("temp_c", 25.0)
        c_now = h.at(ts[0])[0]
        fit0 = self._cool_fit(i)
        err = c_now - (max(0.0, fit0[0] + fit0[1] * t_now) if fit0 else c_now)
        cool, cop, occ = [], [], []
        for k, t in enumerate(ts):
            j = rp.idx_of(t)
            r = rp.row(j) if j is not None else row
            t_fc = r.get("fc_temp_c", t_now) if k else t_now
            if k:
                t_fc += (t_now - row.get("fc_temp_c", t_now)) * 0.5 ** (k * 0.25 / rt_home.HALF_LIFE_H)
                fit = self._cool_fit(j) if j is not None else None
                c = max(0.0, (fit[0] + fit[1] * t_fc) if fit else c_now) + err * 0.5 ** (k * 0.25 / rt_home.HALF_LIFE_H)
            else:
                c = c_now
            c = max(0.0, min(c, h.qmax / max(1.0, self._cop(t, t_fc))))
            cool.append(min(load[k], c))
            cop.append(self._cop(t, t_fc))
            occ.append(int(h.at(t)[3]))
        return cool, cop, occ

    def _cop(self, ts: str, t_fc: float) -> float:
        """The 2026 nodes' plant runs at the COP in their file, so the plan and the meter agree: the outdoor rule made the plan
        think a kW of heat was worth more kW of electricity, and delivery came 4% short. The library keeps the rule."""
        return self.house.at(ts)[1] if self.sim.sell else rt_home.cop_cool(t_fc)


# ---------------------------------------------------------------------------------------------- the building itself
class SimulatedBuilding(SimulatedHome):
    """The building on its measured day (the replay), with its cooling and lighting as FlexGrid's levers. Always the
    real-time design (control_mode "rolling"): a building has no day-ahead fixed script to fall back on."""
    name = "building"

    def __init__(self, cfg: dict[str, Any], replay):
        p = params(cfg.get("building") or cfg.get("agent_name") or getattr(replay, "profile", ""))
        lim = {**(p.get("limits") or {}), **(cfg.get("owner_limits") or {})}
        self.light_default = max(0.0, min(LIGHT_DIM_HARD_MAX, float(lim.pop("light_dim_max", 0.3))))
        b = p.get("battery") or {}
        kwh = float(b.get("kwh", 0.0))
        if kwh > 0 and not BACKUP:
            raise ValueError("a building's battery waits for sales: set FG_BATTERY_MODE=backup")
        cfg = {**cfg, "battery_kwh": kwh, "battery_kw": float(b.get("kw", 0.0)), "reserve_pct": float(b.get("reserve_pct", 0.0)),
               "soc_max_pct": float(b.get("soc_max_pct", 100.0)), "soc0_pct": float(b.get("soc0_pct", 0.0)), "eta": float(b.get("eta", 0.95)),
               "ev_max_kw": 0.0, "has_ev": False, "export_limit_kw": float(p.get("export_limit_kw", 5000.0 if kwh > 0 else 2000.0)),
               "control_mode": "rolling", "owner_limits": {k: v for k, v in lim.items() if k in LIMIT_KEYS}}
        super().__init__(cfg, replay)
        self.bparams = p
        self.sell = dict(p.get("sell") or {})
        self._zones = None
        self.hold_cold = False               # set by the agent's pre-cool, until the sale's levers open
        self.x_recent: deque[float] = deque(maxlen=13)
        if self.sell:
            from .node import buildings
            self._zones = buildings.load(cfg.get("name") or cfg.get("agent_name") or "")
        self.house = BuildingHouse(p, replay.profile)
        self.ctl = BuildingController(self, self.house, rt_control.Limits(**cfg["owner_limits"]))
        self._own_devices(cfg.get("name") or cfg.get("agent_name") or "")

    def afresh(self) -> None:
        super().afresh()
        self.hold_cold = False
        self.x_recent.clear()
        self.house.cut, self.house.cut_span = 0.0, ("", "9")

    def _own_devices(self, name: str) -> None:
        """A 2026 building's fans, plugs and EV ports are always on: FG_DEVICES names the homes' devices, not these."""
        try:
            cat = devices.catalog()
        except (FileNotFoundError, OSError, ValueError):
            return
        row = (cat.get("rows", {}).get("building") or {}).get(name)
        if row is not None:
            spec = cat["devices"].get("building") or {}
            arr = {k: devices.array("building", k) for k in spec}
            self.app = {k: v for k, v in spec.items() if float(arr[k][row].max()) > 0.0}
            self.dev_arr = {k: arr[k] for k in self.app}
            self.dev_row = row

    def free_scale(self, ts: str) -> float:
        """The sale counts free rooms: with fewer free than it assumed, as on a winter evening, there is less to give (the winter sheet's factor)."""
        zm = self._zones
        if zm is None or not zm.m.get("sale_free"):
            return 1.0
        free = sum(s["share"]["light"] * (1.0 if s.get("closed") else 1.0 - zm.plan(k, ts)) for k, s in zm.spaces.items())
        return min(1.0, free / zm.m["sale_free"])

    def dev_share(self, k: str, ts: str) -> float:
        """The part of this device that may go at this step: the sale's share, or what the agent's calls opened."""
        if self.levers is None:
            return float(self.sell.get(k, self.app[k].get("share", 1.0))) * (1.0 if k == "ev" else self.free_scale(ts))
        return float((self.levers.get("dev") or {}).get(k, 0.0))

    def dev_now(self, ts: str) -> dict[str, float]:
        j = self.replay.idx_of(ts) if self.app else None
        if j is None:
            return {}
        out = {}
        for k in self.app:
            if k not in self.dev_off:
                x = float(self.dev_arr[k][self.dev_row, j]) * self.dev_share(k, ts)
                if x > 1e-6:
                    out[k] = x
        return out

    def _default(self, pv: float, base: float, ev: float, soc: float, ts=None) -> float:
        """A building's battery does no self-consumption: it waits for sales."""
        return self._backup_refill(ts, soc, self._caps(soc)[1]) if self.E > 0 else 0.0

    def _backup_refill(self, ts, soc: float, can_ch: float) -> float:
        """What a sale took comes back overnight at full power, not at a home's 2 kW, so the rebound shows."""
        night = bool(ts) and (ts[11:13] >= "23" or ts[11:13] < "06")
        return -min(can_ch, self.P) if night and soc < self.SM - 1e-3 else 0.0

    def light_limit(self) -> float:
        """The share of the lights a person allows dimmed now, whoever runs the node: the facility manager's override, else the default."""
        v = (self.overrides or {}).get("lights")
        v = self.light_default if v is None else v
        try:
            return max(0.0, min(LIGHT_DIM_HARD_MAX, float(v)))
        except (TypeError, ValueError):
            return self.light_default

    def dim_max(self) -> float:
        """The share of the lights that may be dimmed now: what the agent's calls opened, never past the limit. With no levers, the limit."""
        lim = self.light_limit()
        if self.levers is None:
            return 0.0 if "lighting" in self.held_out else lim
        return max(0.0, min(lim, float(self.levers.get("light_share", 0.0))))

    def light_scale(self, ts: str) -> float:
        return 1.0 if self.levers is not None else self.free_scale(ts)

    def _light_dim(self, ts: str, decided: dict[str, Any] | None) -> float:
        """The guard for the lights: what the plan dims, never past the facility manager's limit, nothing once the
        building has left the event."""
        self._release_plant(ts)
        other = super()._light_dim(ts, decided)                 # benchmark: its other smart systems
        if decided is None or (self.overrides or {}).get("stop"):
            return other
        want, cap = float(decided.get("dim", 0.0)), self.ctl.light_cap(ts)
        self._clip("light_kw", ts, want, cap)
        return other + max(0.0, min(cap, want))

    def _release_plant(self, ts: str) -> None:
        """A cap or a stop ends when the zones reach their limit: the plant comes back, and the agent hears why (the card's step 5)."""
        lim = (self.overrides.get("comfort") or {}).get("max_rise_c")
        h = self.house
        self.x_recent.append(self.x)
        nxt = h.step(self.x, h.dq_bounds(ts)[1], 0.25 / 3)                  # the next part, so the plant comes back before a limit passes, not after
        fast = any(len(self.x_recent) >= n and nxt - self.x_recent[-n] > c for n, c in DRIFT_C)
        if h.cutting(ts) and lim is not None and (fast or nxt >= lim):
            h.cut = 0.0
            if self.levers is not None:
                self.levers["released"] = ts

    def _extra_tel(self, ts: str, acc: dict[str, float]) -> dict[str, Any]:
        on = self.house.light_kw(ts)
        return {"node": "building", "light_kw": round(on, 1), "light_dim_kw": round(acc.get("dim", 0.0), 2),
                "light_on_kw": round(on - acc.get("dim", 0.0), 1), "light_dim_max": round(self.dim_max(), 2)}

    def rt_override(self, kind: str, value: Any = True) -> None:
        """The facility manager: comfort ({max_rise_c, max_rise_away_c}), lights (the share that may be dimmed,
        0-0.5), stop (leave the event), clear (back to the plan). No car here."""
        if kind == "lights":
            try:
                value = max(0.0, min(LIGHT_DIM_HARD_MAX, float(value)))
            except (TypeError, ValueError):
                return
            self.overrides["lights"] = value
            return
        if kind in ("ev_now", "depart_at"):
            return
        super().rt_override(kind, value)
        if kind == "stop" and value not in (None, False):
            self.house.cut = 0.0                  # a cap or a stop on the plant goes with the node: otherwise cooling stays down after a person's stop
        if kind == "alarm" and value not in (None, False):
            # Life safety, whoever runs the node: levers set to None would let the plan keep every device, so the plan stands down until a person clears it
            self.house.cut = 0.0
            super().rt_override("stop", True)

    def limits(self) -> dict[str, Any]:
        out = super().limits()
        out.update({"node": "building", "light_dim_max": round(self.light_limit(), 2),
                    "comfort_max_rise_c": self.ctl.limits.max_rise_c if self.ctl else None})
        return out
