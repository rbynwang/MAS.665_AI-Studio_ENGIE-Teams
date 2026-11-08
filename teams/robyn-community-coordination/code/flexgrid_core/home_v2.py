"""The home agent's v2 behaviour: offer, accept or decline, deliver, report. Shared by the Jetson daemons and the
replay fleet on jetson-b, so every home runs the same code.

H3 Offer      beckn.discover (call for offers)  -> forecast the event window -> build offer -> beckn.on_discover
H4 Accept     beckn.confirm (commitment)        -> verify -> install the plan -> beckn.on_confirm (ACCEPTED/DECLINED)
H5 Report     each finished event slot          -> signed meter reading (USAGE kW) -> beckn.on_status, to the
              coordinator (for the split) and to the utility (the meter owner settles on it)

How an offer is computed (all from data the home has; no peeking at the event day):
  * only data from before the moment the offer is made (10:00 the day before): no peeking
  * PV per 15-minute step: the mean of the same step on the 3 latest days already seen (seasonal naive)
  * load per 15-minute step: a straight line of load against temperature for that hour of day, fitted on the
    previous 21 days, applied to a day-ahead temperature forecast (the dataset's temperature plus a realistic
    forecast error); cfg forecast = "naive" uses the previous-3-days mean instead
  * EV: the latest charging session seen is expected again (same arrival time, energy, departure)
  * the battery is run forward from its state now to the window opening (no event), plus the 2 h pre-charge (event);
    in the window it keeps doing its default job of covering the house; what it can add on top, up to its power
    rating and the export limit, is offered, plus what pausing the EV really saves at the meter
  * the battery part is scaled down if the window needs more energy than the battery holds between its
    reserve and its maximum charge
  * confidence = 0.9 in automatic control (0.6 advisory) x the sensor-health factor x the home's reputation
  * the evening after the window (8 h) is forecast the same way, three times: no event; no EV and no delivery;
    no EV and 25 / 50 / 75 / 100 % of the battery offer delivered. The offer carries the net-import forecasts and
    the differences (`rebound_kw`, the extra import because the battery is emptier), so the coordinator can check that the event
    does not move the community's peak into the evening. The EV's flexibility (energy, power, latest finish) is
    offered separately: the coordinator picks when it restarts. These are net-meter forecasts and an EV envelope,
    the same level of detail as v1's flexibility envelope, not device data.
"""
from __future__ import annotations

import random
import threading
import time
import uuid
from collections import deque
from concurrent.futures import Future
from typing import Any, Callable

from . import beckn, signing, telemetry as tcheck
from .forecaster import HomeForecaster, HomeForecaster2026
from .node import sellable

SLOT = beckn.SLOT_MINUTES
POST_STEPS = 32                     # the evening after the window that offers forecast (8 h)
TRAINED = ("trained", "trained_2026")   # forecast settings that run models: the 2025-26 ones, or forecaster.HomeForecaster2026
REBOUND_FRACTIONS = (0.25, 0.5, 0.75, 1.0)   # battery delivery levels the evening rebound is forecast for


TEMP_FC_SIGMA_C = 2.1    # day-ahead temperature forecast error (see temp_forecast)


def temp_forecast(ts: str, actual_c: float) -> float:
    """Day-ahead temperature forecast for a step = the dataset's temperature plus one error per day, drawn from a
    normal distribution with sigma 2.1 C (mean absolute error 1.7 C = 3 F; NWS verification for Wichita, summer 2000:
    next-day maximum temperature MAE 2.4 F for forecasters and 3.5 F for MOS guidance). Same error for every home
    (weather errors are regional), seeded by the date so every run is repeatable."""
    return actual_c + random.Random(f"temp-forecast-{ts[:10]}").gauss(0.0, TEMP_FC_SIGMA_C)


import os as _os
BACKUP = _os.environ.get("FG_BATTERY_MODE") == "backup"

class HomeV2:
    def __init__(self, name: str, cfg: dict[str, Any], adapter, replay, key, send: Callable,
                 trusted: Callable[[], list[str]], log: Callable[..., None], coordinator: str = "coordinator"):
        self.name, self.cfg, self.adapter, self.replay, self.key = name, cfg, adapter, replay, key
        self.pub = signing.public_hex(key)
        self.send, self.trusted, self.log, self.coordinator = send, trusted, log, coordinator
        self.is_building = getattr(adapter, "name", None) == "building"
        if self.is_building:                  # the homes' trained models are for homes: a building forecasts its load
            self.cfg = {**cfg, "forecast": "temp_regression", "building_forecast": cfg.get("forecast")}   # on its own record (hour-of-day line, weather), or its own trained models
        self.offers: dict[str, dict[str, Any]] = {}          # event_id -> offer we made
        self.contracts: dict[str, dict[str, Any]] = {}       # contract_id -> contract + our status
        self.payments: list[dict[str, Any]] = []             # one per event_id (a repeated payment notice is ignored)
        self.reputation = float(cfg.get("reputation", 1.0))
        self.history: deque = deque(maxlen=8)
        self._fits: dict[tuple, tuple[float, float] | None] = {}
        self._fc_now: int | None = None
        self._forecaster: HomeForecaster | None = None       # forecast = "trained" (flexgrid_core/forecaster.py)
        self._tf: dict[str, Any] | None = None               # the trained forecast being used, by step
        self._bfc: dict[str, Any] = {}                       # a building's trained forecast, by day
        self._lock = threading.Lock()                        # readings marked by send callbacks (fleet threads)
        # the AI agent (brain.HomeBrain), attached by a Jetson's daemon only: the 98 simulated homes and the offline
        # simulations keep their rules. It may lower the safe offer and decline a commitment the guard found safe
        self.brain = None
        self.agent = None                    # a thinking node's workflow, for its record of steps 1 to 6
        self._rt_fc: dict[str, Any] = {"hour": None, "vals": {}}
        if getattr(adapter, "ctl", None) is not None:            # the real-time design: its forecasts come from here
            adapter.ctl.forecast_fn = self.rt_forecast

    def rt_forecast(self, i: int, ts_list: list[str]) -> dict[str, tuple[float, float]]:
        """The 5-minute loop's forecast of solar and house load for the next steps (rt_home corrects it with what
        the meter reads now): the trained models (forecast = "trained") made from the history up to now, once an
        hour; else today's day-ahead method run from now."""
        hour = i // 4
        if self._rt_fc["hour"] != hour:
            self._rt_fc = {"hour": hour, "vals": {}}
        vals = self._rt_fc["vals"]
        todo = [t for t in ts_list if t not in vals]
        if todo:
            if self.cfg.get("forecast") in TRAINED:
                try:
                    f = self._models().forecast(i, todo) or {}
                    for t, v in f.items():
                        vals[t] = (float(v["pv"][1]), float(v.get("load_mean", v["load"][1])))
                except Exception as exc:                       # no model files: the day-ahead method below
                    self.log("rt_forecast_fallback", error=str(exc)[:160])
            saved, self._fc_now = self._fc_now, i
            try:
                for t in todo:
                    if t not in vals:
                        vals[t] = self._naive(t)
            finally:
                self._fc_now = saved
        return {t: vals[t] for t in ts_list if t in vals}

    def _models(self):
        """The home's trained forecaster, made on first use. Raises where its files or the dataset's columns are missing."""
        if self._forecaster is None:
            self._forecaster = (HomeForecaster2026 if self.cfg.get("forecast") == "trained_2026" else HomeForecaster)(self.name, self.replay)
        return self._forecaster

    def reset(self) -> None:
        """Forget offers, contracts and plans (demo: replay the same day again)."""
        self.offers.clear(); self.contracts.clear(); self.payments.clear()
        if self.brain is not None:
            self.brain.reset()
        if hasattr(self.adapter, "plan"):
            self.adapter.plan.clear()
            self.adapter.ev_holds.clear()
        if hasattr(self.adapter, "rt_reset"):
            self.adapter.rt_reset()
        self._rt_reported, self._rt_at, self._rt_why = {}, None, None

    # ------------------------------------------------------------------ helpers
    def now_idx(self) -> int:
        return self.replay.current()

    def now_ts(self) -> str:
        return self.replay.ts(self.now_idx())

    def observe(self) -> dict[str, Any]:
        tel = self.adapter.telemetry()
        if not self.history or self.history[-1].get("idx") != tel.get("idx"):
            self.history.append(tel)
        return tel

    def health(self) -> dict[str, Any]:
        return tcheck.check(list(self.history), self.adapter.limits())

    def _ev_latest(self, day: str) -> dict[str, Any] | None:
        """The latest charging session that has already started, moved to `day`: the EV forecast."""
        if not self.adapter.has_ev:
            return None
        now = self.now_ts()
        for s in reversed(self.replay.sessions):
            if s["arrival"] <= now:
                arrival = day + s["arrival"][10:]
                return {"arrival": arrival, "deadline": beckn.add_minutes(arrival, self.replay.minutes_between(
                            s["arrival"], s["deadline"])), "kwh": s["kwh"],
                        "max_kw": min(self.adapter.ev_max, s["max_kw"])}
        return None

    @staticmethod
    def _ev_profile(ev: dict[str, Any] | None, ts_list: list[str]) -> list[float]:
        """Unmanaged charging (full power from arrival until done) on the given 15-minute steps."""
        out, left = [], (ev or {}).get("kwh", 0.0)
        if ev:
            t = ev["arrival"]
            while left > 1e-6 and t < ev["deadline"] and t < ts_list[0]:      # energy charged before the steps
                left -= min(ev["max_kw"] * 0.25, left)
                t = beckn.add_minutes(t, 15)
        for ts in ts_list:
            kw = 0.0
            if ev and ev["arrival"] <= ts < ev["deadline"] and left > 1e-6:
                kw = min(ev["max_kw"], left / 0.25)
                left -= kw * 0.25
            out.append(kw)
        return out

    def _naive(self, ts: str) -> tuple[float, float]:
        """(PV, load) forecast for one 15-minute step. PV: mean of the same step on the previous 3 days.
        Load: the same, or (forecast = "temp_regression", the default) a straight-line fit of load on temperature
        for that hour of day over the previous 21 days, applied to the day-ahead temperature forecast."""
        if self._tf and ts in self._tf:                     # forecast = "trained": PV median, load expected value
            f = self._tf[ts]
            return f["pv"][1], f.get("load_mean", f["load"][1])
        i = self.replay.idx_of(ts)
        if i is None:
            return 0.0, 0.0
        days = self._past_days(i, 3)
        pv = [self.replay.row(i - 96 * d)["pv_kw"] for d in days]
        ld = [self.replay.row(i - 96 * d)["base_kw"] for d in days]
        if not days:
            return 0.0, 0.0
        load = sum(ld) / len(ld)
        if self.cfg.get("forecast", "temp_regression") == "temp_regression":
            fit = self._load_fit(i)
            if fit:
                row = self.replay.row(i)
                tf = row.get("fc_temp_c")                  # the forecast as issued, where the dataset has it
                load = max(0.0, fit[0] + fit[1] * (tf if tf is not None else temp_forecast(ts, row["temp_c"])))
        trained = self._building_load(ts)
        return sum(pv) / len(pv), load if trained is None else trained

    def _building_load(self, ts: str) -> float | None:
        """A building's own trained forecast of its total load, an hour at a time. None past what the models or the calendar reach, and the rule stands in."""
        if not self.is_building or self.cfg.get("building_forecast") != "trained_2026":
            return None
        day = ts[:10]
        if day not in self._bfc:
            from .buildings_forecast import for_node
            rp = self.replay
            hours = [(h, rp.row(j).get("fc_temp_c")) for h in range(24) if (j := rp.idx_of(f"{day} {h:02d}:00")) is not None]
            try:
                self._bfc[day] = for_node(self.name).forecast(day, [(h, float(t)) for h, t in hours if t is not None])
            except (ValueError, OSError) as exc:             # no calendar for that day, no temperature, or no model files
                self.log("building_forecast_fallback", day=day, error=str(exc)[:160])
                self._bfc[day] = None
        f = self._bfc[day]
        return f[int(ts[11:13])]["total"][1] if f and int(ts[11:13]) in f else None

    def _range(self, ts: str) -> dict[str, float]:
        """The trained forecast's 10 % and 90 % values for a step (empty under the other methods)."""
        f = (self._tf or {}).get(ts)
        if not f:
            return {}
        return {"pv_p10": round(f["pv"][0], 3), "pv_p90": round(f["pv"][2], 3),
                "load_p10": round(f["load"][0], 3), "load_p90": round(f["load"][2], 3)}

    def _past_days(self, i: int, n: int) -> list[int]:
        """The n most recent whole days d for which step i - 96 d is already in the past (before the forecast was
        made): an offer made at 10:00 for tomorrow 16:00 cannot use today's 16:00."""
        now = getattr(self, "_fc_now", None)
        now = self.now_idx() if now is None else now
        first = 1
        while i - 96 * first >= now:
            first += 1
        return [d for d in range(first, first + n) if i - 96 * d >= 0]

    def _load_fit(self, i: int) -> tuple[float, float] | None:
        """Ordinary least squares load = a + b x temperature, on the same hour of day over the previous 21 days
        (84 points). History only: nothing from the forecast day is used."""
        h0 = i - (i % 4)                                    # first 15-minute step of this hour
        key = (self.replay.ts(h0)[11:13], self.replay.ts(i)[:10], self._fc_now)
        if key in self._fits:
            return self._fits[key]
        xs, ys = [], []
        for d in self._past_days(h0 + 3, 21):
            for q in range(4):
                j = h0 + q - 96 * d
                if j >= 0:
                    r = self.replay.row(j)
                    xs.append(r["temp_c"]); ys.append(r["base_kw"])
        fit = None
        if len(xs) >= 20:
            mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
            sxx = sum((x - mx) ** 2 for x in xs)
            if sxx > 1e-6:
                b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
                fit = (my - b * mx, b)
        self._fits[key] = fit
        return fit

    # ------------------------------------------------------------------ H3 Offer
    def forecast_window(self, need: dict[str, Any]) -> dict[str, Any]:
        day = need["window_start"][:10]
        self._fc_now = self.now_idx()
        ev = self._ev_latest(day)
        win = [(s["id"], beckn.add_minutes(s["start"], 15 * k)) for s in need["slots"] for k in (0, 1)]
        post_ts = [beckn.add_minutes(need["window_end"], 15 * k) for k in range(POST_STEPS)]
        ev_win, ev_post = self._ev_profile(ev, [t for _, t in win]), self._ev_profile(ev, post_ts)
        # from now to the window: where the battery will stand when the window opens (no event)
        lead_ts, t = [], self.replay.ts(self._fc_now + 1)
        while t < need["window_start"] and len(lead_ts) < 96 * 3:
            lead_ts.append(t)
            t = beckn.add_minutes(t, 15)
        ev_lead = [0.0] * len(lead_ts)
        for d in sorted({x[:10] for x in lead_ts}):               # each day's expected session before the window
            prof = self._ev_profile(self._ev_latest(d), lead_ts)
            ev_lead = [x + y for x, y in zip(ev_lead, prof)]
        self._tf = None
        if self.cfg.get("forecast") in TRAINED:
            try:
                self._tf = self._models().forecast(self._fc_now, lead_ts + [t for _, t in win] + post_ts) or None
            except Exception as exc:                     # no model files etc.: today's method, and say so
                self.log("v2_forecast_fallback", error=str(exc)[:200])
            if self._tf and ev:                          # the EV: the latest session, scaled to what the EV model
                x = self._forecaster.ev_expected(day)    # expects in 15:00-20:00 (never above the session itself)
                if x is not None and "kw_by_hour" in x:  # the 2026 models say kW by hour of the day itself, for the whole fleet to add up
                    part = x["takes_part"] or (int(need["window_start"][11:13]), int(need["window_end"][11:13])) != x["window"]   # the chance is for its own window
                    kw = lambda t: x["kw_by_hour"][int(t[11:13])]                      # noqa: E731  (past midnight: the night before, the nearest forecast)
                    ev_win, ev_post = [kw(t) if part else 0.0 for _, t in win], [kw(t) for t in post_ts]   # a car out of the sale still charges after it
                    ev = {**ev, "expected_kwh_window": round(x["mean_kwh"], 2), "p_charging": round(x["p_charging"], 2), "in_sale": part}
                elif x is not None:
                    span = [f"{day} {h:02d}:{m:02d}" for h in range(15, 20) for m in (0, 15, 30, 45)]
                    full = sum(self._ev_profile(ev, span)) * 0.25
                    k = min(1.0, x["mean_kwh"] / full) if full > 0.1 else 1.0
                    ev_win, ev_post = [v * k for v in ev_win], [v * k for v in ev_post]
                    ev = {**ev, "expected_kwh_15_20": round(x["mean_kwh"], 2), "p_charging": round(x["p_charging"], 2),
                          "scale": round(k, 3)}
        lead = []
        for ts, evk in zip(lead_ts, ev_lead):
            pv, ld = self._naive(ts)
            lead.append({"ts": ts, "pv_p50": round(pv, 3), "load_p50": round(ld, 3), "ev_kw": round(evk, 2)})
        steps, post = [], []
        for (sid, ts), evk in zip(win, ev_win):
            pv, ld = self._naive(ts)
            steps.append({"slot_id": sid, "ts": ts, "pv_p50": round(pv, 3), "load_p50": round(ld, 3), "ev_kw": round(evk, 2),
                          **self._range(ts)})
        for ts, evk in zip(post_ts, ev_post):
            pv, ld = self._naive(ts)
            post.append({"ts": ts, "pv_p50": round(pv, 3), "load_p50": round(ld, 3), "ev_kw": round(evk, 2)})
        method = ("load: hour-of-day regression on day-ahead temperature, previous 21 days"
                  if self.cfg.get("forecast", "temp_regression") == "temp_regression" else "load: previous 3 days")
        ev_how = "" if self.cfg.get("forecast") == "trained_2026" else "; EV: latest session"
        method = f"{self._forecaster.method}{ev_how}" if self._tf else f"{method}; PV: previous 3 days; EV: latest session"
        self._tf = None
        return {"event_id": need["event_id"], "method": method,
                "steps": steps, "post": post, "ev": ev, "lead": lead,
                "soc_now_pct": self.observe().get("soc_pct")}

    @staticmethod
    def _battery_run(steps: list[dict[str, Any]], E: float, Emax: float, P: float, eta: float, ev_on: bool,
                     extra: list[float] | None = None) -> tuple[list[float], float]:
        """Net import per step with the battery in its default job (cover deficit, store surplus), plus extra[k] kW
        of delivery on top at step k. E, Emax: kWh the battery can still deliver (AC side).
        build_offer starts it where the forecast says it will stand when the window opens."""
        imp = []
        for k, st in enumerate(steps):
            net = st["load_p50"] + (st["ev_kw"] if ev_on else 0.0) - st["pv_p50"]
            b = (0.0 if BACKUP else min(P, net)) if net > 0 else -min(P, -net)
            if extra and k < len(extra):
                b += extra[k]
            b = max(-min(P, (Emax - E) / 0.25 / (eta * eta)), min(b, P, E / 0.25))
            E -= b * 0.25 if b > 0 else b * 0.25 * eta * eta
            imp.append(net - b)
        return imp, E

    def _weather(self) -> dict[str, Any] | None:
        """The what-if weather in force for this home ({pv, load, from}), or None on the dataset's own weather."""
        from . import replay as replay_mod
        w = getattr(self.replay, "weather", None)
        w = w if w is not None else replay_mod.WEATHER
        if not w.get("from"):
            return None
        return {"pv": round(float(w.get("pv", 1.0)), 4), "load": round(float(w.get("load", 1.0)), 4), "from": w["from"]}

    def build_offer(self, need: dict[str, Any], fc: dict[str, Any]) -> dict[str, Any]:
        """Everything is read off the same forecast runs the coordinator will rely on:
        no event (EV on, battery default)  vs  EV paused  vs  EV paused + battery discharging extra.
        EV part of the offer = what pausing the EV really saves at the meter (less than the EV's kW when the battery
        would have covered part of the EV anyway); battery part = what the extra discharge saves on top."""
        lim = self.adapter.limits()
        P, E, R, SM, eta = lim["battery_kw"], lim["battery_kwh"], lim["reserve_pct"] / 100, lim["soc_max_pct"] / 100, lim["eta"]
        usable = (SM - R) * E * eta                                   # kWh between reserve and full, AC side
        win, both, n_w = fc["steps"], fc["steps"] + fc["post"], len(fc["steps"])
        # battery energy when the window opens: no event = run the battery's default job forward from now;
        # with the event = that plus the 2 h pre-charge, at most full
        soc = fc.get("soc_now_pct")
        e_now = usable if soc is None else max(0.0, min(usable, (soc / 100 - R) * E * eta))
        _, e_cf = self._battery_run(fc.get("lead", []), e_now, usable, P, eta, ev_on=True)
        e_ev = min(usable, e_cf + P * 2.0 * eta * eta)
        # most the battery can add on top of its default job, per window step (power and export limits)
        incr, need_def = [], 0.0
        for st in win:
            b0 = 0.0 if BACKUP else min(P, max(0.0, st["load_p50"] - st["pv_p50"]))
            incr.append(max(0.0, min(P, lim["export_limit_kw"] - st["pv_p50"] + st["load_p50"]) - b0))
            need_def += b0 * 0.25
        need_inc = sum(incr) * 0.25
        scale = 1.0 if need_inc <= 1e-9 else max(0.0, min(1.0, (e_ev - need_def) / need_inc))
        x = [scale * v for v in incr]
        cf, _ = self._battery_run(both, e_cf, usable, P, eta, ev_on=True)
        noev, _ = self._battery_run(both, e_ev, usable, P, eta, ev_on=False)
        per_slot: dict[int, dict[str, float]] = {}
        runs = {}
        for f in REBOUND_FRACTIONS:                              # the rebound is not linear in the delivery
            runs[f], _ = self._battery_run(both, e_ev, usable, P, eta, ev_on=False, extra=[f * v for v in x])
        full = runs[1.0]
        for k, st in enumerate(win):
            d = per_slot.setdefault(st["slot_id"], {"ev": 0.0, "batt": 0.0, "n": 0})
            d["ev"] += max(0.0, cf[k] - noev[k]); d["batt"] += max(0.0, noev[k] - full[k]); d["n"] += 1
        slots = []
        for sid, d in sorted(per_slot.items()):
            ev_kw, batt = d["ev"] / d["n"], d["batt"] / d["n"]
            slots.append({"id": sid, "kw_offered": round(batt + ev_kw, 2), "battery_kw": round(batt, 2),
                          "ev_pause_kw": round(ev_kw, 2)})
        ac = None
        ctl = getattr(self.adapter, "ctl", None)
        if ctl is not None and getattr(self.adapter, "house", None) is not None:   # the real-time design: + the AC
            try:
                saved, self._fc_now = self._fc_now, self.now_idx()
                ac = ctl.ac_offer(need["window_start"], need["window_end"])
            except Exception as exc:
                self.log("v2_ac_offer_error", error=str(exc)[:200])
            finally:
                self._fc_now = saved
            if ac and ac.get("kw", 0.0) > 0.01:
                slots = [{**x, "ac_kw": ac["kw"], "kw_offered": round(x["kw_offered"] + ac["kw"], 2)} for x in slots]
        sell, firm_share = None, None
        if self.cfg.get("forecast") == "trained_2026" and not self.is_building and ctl is not None:
            try:                                                     # kW by device from the forecasts: rides on the offer for the fleet range
                sell = sellable.home_window(self.adapter, self._models(), need["window_start"], need["window_end"], self.now_idx())
                firm_share = sellable.fleet_range([sell], need["window_start"][:10], need["window_start"], need["window_end"])["firm_share"]
            except Exception as exc:                                 # too little history: the offer goes out on the rule
                self.log("v2_sellable_error", error=str(exc)[:200])
        app = None
        if ctl is not None and getattr(self.adapter, "app", None):   # benchmark: the node's other smart systems
            app = ctl.app_offer(need["window_start"], need["window_end"])
            if app and sell:
                app = sellable.offer_app(app, sell)
            if app and app.get("kw", 0.0) > 0.01:
                slots = [{**x, "app_kw": app["kw"], "kw_offered": round(x["kw_offered"] + app["kw"], 2)} for x in slots]
        light = None
        if ctl is not None and hasattr(ctl, "light_offer"):        # a building: its lighting too (building.py)
            try:
                light = ctl.light_offer(need["window_start"], need["window_end"])
            except Exception as exc:
                self.log("v2_light_offer_error", error=str(exc)[:200])
            if light and light.get("kw", 0.0) > 0.01:
                slots = [{**x, "light_kw": light["kw"], "kw_offered": round(x["kw_offered"] + light["kw"], 2)} for x in slots]
        if firm_share is not None:                                   # this home's part of a firm sale: its kW at the fleet's firm share
            slots = [{**x, "firm_kw": round(x["kw_offered"] * firm_share, 2)} for x in slots]
        h = self.health()
        base_conf = 0.9 if lim.get("control", "auto") == "auto" else 0.6
        conf = round(base_conf * h.get("confidence_factor", 1.0) * self.reputation, 3)
        if h.get("status") in ("frozen", "jump"):
            slots = [{**s, "kw_offered": 0.0, "battery_kw": 0.0, "ev_pause_kw": 0.0} for s in slots]
        rebound = {str(f): [round(r - z, 2) for r, z in zip(runs[f][n_w:], noev[n_w:])] for f in REBOUND_FRACTIONS}
        ev = fc.get("ev")
        ev_kwh = sum(st["ev_kw"] for st in both) * 0.25 if ev else 0.0
        evening = {"start": need["window_end"], "step_min": 15,
                   "import_cf_kw": [round(v, 2) for v in cf[n_w:]],
                   "import_noev_kw": [round(v, 2) for v in noev[n_w:]],
                   "rebound_kw": rebound}
        ev_flex = None
        if ev and ev.get("in_sale") is False:                        # a car the sale won't touch charges as it always does: the evening holds it as it is
            evening["import_noev_kw"] = [round(v + st["ev_kw"], 2) for v, st in zip(evening["import_noev_kw"], fc["post"])]
        elif ev and ev_kwh > 0.05:                                   # charging expected in the window or evening
            ev_flex = {"kwh": round(ev_kwh, 2), "max_kw": round(ev["max_kw"], 2), "latest_finish": ev["deadline"],
                       "arrival": ev["arrival"]}
        if app and app.get("kw", 0.0) > 0.01:                        # their catch-up goes in the evening it declares
            left = sum(v for k, v in (app.get("by_device") or {}).items()
                       if self.adapter.app[k]["kind"] == "shift") * n_w * 0.25
            k = int(ctl.app_delay_min() // 15)
            row = evening["import_noev_kw"]
            while left > 1e-6 and k < len(row):
                kw = min(3.0, left / 0.25)
                row[k] = round(row[k] + kw, 2)
                left -= kw * 0.25
                k += 1
        if self.is_building:
            # the community's evening check is about the homes' batteries and cars refilling at once; a building's
            # lights leave nothing to catch up and its cooling cools back at most RECOVER_C_PER_H, so its evening
            # (a 1 MW building on a 300 kW community) is not added to it
            evening, ev_flex = None, None
        offer = {"event_id": need["event_id"], "home": self.name, "slots": slots,
                 "energy_kwh": round(sum(s["kw_offered"] for s in slots) * SLOT / 60, 2),
                 "confidence": conf, "health": h.get("status"), "price_floor": 0.0,
                 "battery": {"kw": P, "kwh": E, "usable_kwh": round(usable, 2), "energy_scale": round(scale, 3),
                             "window_start_kwh": {"no_event": round(e_cf, 2), "event": round(e_ev, 2)}},
                 "evening": evening, "ev_flex": ev_flex, "forecast_method": fc.get("method"),
                 **({"air_conditioning": ac} if ac else {}), **({"lighting": light} if light else {}), **({"sellable": sell} if sell else {}),
                 **({"appliances": app} if app and app.get("kw", 0.0) > 0.01 else {}),
                 **({"node": "building"} if self.is_building else {}),
                 "valid_until": need["commit_by"], "made_at": self.now_ts()}
        w = self._weather()
        if w:                                # the what-if weather this home lives in (the Control Room's scenario),
            offer["weather"] = w             # declared so that the independent check's replay runs under it too
        if P <= 0 and E > 0:                 # a battery out of service (its inverter tripped): declared the same way
            offer["battery"]["out_of_service"] = True
        if self.agent is not None:
            self.agent.market("5 offer", self.now_ts())
        if self.brain is not None:                                   # the AI agent decides how much of it to offer
            try:                                                     # (it falls back to the plan by itself; should it
                offer = self.brain.decide_offer(need, fc, offer)     # raise anyway, the plan's offer goes out)
            except Exception as exc:
                self.log("agent_error", where="offer", error=f"{type(exc).__name__}: {str(exc)[:200]}")
        self.offers[need["event_id"]] = offer
        return offer

    def send_offer(self, txn: str, offer: dict[str, Any]) -> dict[str, Any]:
        body = beckn.make("on_discover", sender=self.name, receiver=self.coordinator, transaction_id=txn,
                          message={"offer": offer}, key=self.key, sender_role="bpp")
        out = self.send(self.coordinator, beckn.kind("on_discover"), body, f"{self.name}:offer:{offer['event_id']}:{uuid.uuid4().hex[:6]}")
        self.log("v2_offer_sent", event_id=offer["event_id"], kw=[s["kw_offered"] for s in offer["slots"]],
                 confidence=offer["confidence"])
        return {"sent": out, "summary": f"Offer for {offer['event_id']}: {[s['kw_offered'] for s in offer['slots']]} kW "
                                        f"per slot, {offer['energy_kwh']} kWh, confidence {offer['confidence']}."}

    # ------------------------------------------------------------------ H4 Accept
    def verify_contract(self, sender: str, body: dict[str, Any]) -> dict[str, Any]:
        c = (body.get("message") or {}).get("contract") or {}
        def no(reason):
            return {"decision": "DECLINED", "reason": reason, "contract_id": c.get("id")}
        if body.get("signer") not in self.trusted() or not beckn.check(body, body.get("signer", "")):
            return no("bad_signature")
        if c.get("home") != self.name or body["context"].get("bppId") != self.name:
            return no("wrong_recipient")
        offer = self.offers.get(c.get("event_id"))
        if not offer:
            return no("no_offer_for_event")
        if self.now_ts() >= c.get("opt_out_by", "9999"):
            return no("past_opt_out_deadline")
        mine = {s["id"]: s["kw_offered"] for s in offer["slots"]}
        for s in c.get("commitments", []):
            if s["kw"] > mine.get(s["id"], 0.0) + 0.01:
                return no(f"slot_{s['id']}_above_offer")
        h = self.health().get("status")
        if h in ("frozen", "jump"):
            return no("telemetry_unverified")
        if c.get("event_id") in (self.cfg.get("opt_out_events") or []) or \
                c.get("window_start", "")[:10] in (self.cfg.get("opt_out_days") or []):
            return no("homeowner_opted_out")
        if self.cfg.get("control", "auto") == "advisory":
            return no("homeowner_approval_needed")
        verdict = {"decision": "ACCEPTED", "reason": "within_offer", "contract_id": c.get("id")}
        if self.agent is not None:
            self.agent.market("6 commitment", self.now_ts())
        if self.brain is not None:                    # the guard said yes: the AI agent may still say no (never yes to a no)
            prev = self.contracts.get(c.get("id"))
            if not (prev and prev.get("status") in ("ACCEPTED", "DECLINED")):
                try:
                    out = self.brain.decide_accept(c, verdict)
                    if out.get("decision") in ("ACCEPTED", "DECLINED"):
                        verdict = out if out["decision"] == "DECLINED" else verdict
                except Exception as exc:
                    self.log("agent_error", where="accept", error=f"{type(exc).__name__}: {str(exc)[:200]}")
        return verdict

    def install_plan(self, c: dict[str, Any], verdict: dict[str, Any]) -> dict[str, Any]:
        prev = self.contracts.get(c.get("id"))
        if prev and prev.get("status") in ("ACCEPTED", "DECLINED"):      # answered before: nothing changes
            return {"installed": prev["status"] == "ACCEPTED", "repeat": True}
        if verdict["decision"] != "ACCEPTED":
            return {"installed": False}
        cid = c["id"]
        actions: dict[str, dict[str, Any]] = {}
        for s in c["commitments"]:
            for k in (0, 1):
                actions[beckn.add_minutes(s["start"], 15 * k)] = {"deliver_kw": float(s["kw"]), "contract_id": cid}
        w0 = c["window_start"]
        by_idx = self.replay.idx_of(w0)
        for k in range(8, 0, -1):                                    # pre-charge over the 2 h before the window
            actions.setdefault(beckn.add_minutes(w0, -15 * k),
                               {"precharge_to": self.adapter.SM, "by_idx": by_idx, "contract_id": cid})
        hold = (w0, c["ev_resume_after"]) if (self.adapter.has_ev and c.get("ev_resume_after")) else None
        self.adapter.set_plan(actions, hold)
        self.contracts[cid] = {**c, "status": "ACCEPTED", "reported": [], "accepted_at": self.now_ts()}
        return {"installed": True, "steps": len(actions), "ev_hold": hold}

    def answer(self, txn: str, c: dict[str, Any], verdict: dict[str, Any]) -> dict[str, Any]:
        prev = self.contracts.get(c.get("id")) if c.get("id") else None
        if prev and prev.get("status") in ("ACCEPTED", "DECLINED"):
            # an answer is final: a contract this home accepted (its plan is installed) is never turned into DECLINED,
            # and a repeat of the same commitment gets the same answer again
            verdict = {"decision": prev["status"], "reason": prev.get("reason", "within_offer"), "contract_id": c.get("id")}
        elif verdict["decision"] != "ACCEPTED" and c.get("id"):
            self.contracts[c["id"]] = {**c, "status": "DECLINED", "reason": verdict["reason"], "reported": []}
        body = beckn.make("on_confirm", sender=self.name, receiver=self.coordinator, transaction_id=txn,
                          message={"contract": {"id": c.get("id"), "event_id": c.get("event_id"), "home": self.name,
                                                "status": verdict["decision"], "reason": verdict["reason"]}},
                          key=self.key, sender_role="bpp")
        out = self.send(self.coordinator, beckn.kind("on_confirm"), body, f"{self.name}:answer:{c.get('id')}:{uuid.uuid4().hex[:6]}")
        self.log("v2_contract_" + verdict["decision"].lower(), contract_id=c.get("id"), reason=verdict["reason"])
        return {"sent": out, "summary": f"Contract {c.get('id')}: {verdict['decision']} ({verdict['reason']})."}

    # ------------------------------------------------------------------ H5 Report
    def due_readings(self) -> list[tuple[str, int]]:
        cur = self.now_idx()
        due = []
        for cid, c in self.contracts.items():
            if c.get("status") != "ACCEPTED":
                continue
            busy = c.get("in_flight") or []
            for s in c["commitments"]:
                i0 = self.replay.idx_of(s["start"])
                if i0 is not None and cur > i0 + 1 and s["id"] not in c["reported"] and s["id"] not in busy \
                        and i0 in self.adapter.recent and i0 + 1 in self.adapter.recent:
                    due.append((cid, s["id"]))
        return due

    # ------------------------------------------------------------------ the agent's tools, logged with input and output
    def _tool(self, name: str, inp: str, out: str, t0: float) -> None:
        self.log("tool", tool=name, input=inp, output=out, ms=round((time.perf_counter() - t0) * 1000, 1))

    def _fc_words(self, fc: dict[str, Any]) -> str:
        st = fc.get("steps") or []
        if not st:
            return "no forecast"
        pv = [x.get("pv_p50", 0.0) for x in st]
        ld = [x.get("load_p50", 0.0) for x in st]
        ev = fc.get("ev") or {}
        m = str(fc.get("method", ""))
        src = "trained models" if m.startswith("trained") else m.split(":")[0][:40] or "forecast"
        out = f"{src}: solar {min(pv):.1f}-{max(pv):.1f} kW, house load {min(ld):.1f}-{max(ld):.1f} kW in the window"
        if ev.get("expected_kwh_15_20") is not None:
            out += f"; EV {ev['expected_kwh_15_20']:.1f} kWh expected 15-20 h (charging chance {ev.get('p_charging', 0):.0%})"
        elif ev:
            out += "; EV session expected"
        return out

    def _pre_kw(self, c: dict[str, Any]) -> float | None:
        """benchmark: the meter's mean import in the hour starting 2 hours before the event (the same-day adjustment)."""
        w0 = c.get("window_start") or min((x["start"] for x in c.get("commitments") or []), default=None)
        if w0 is None:
            return None
        j = self.replay.idx_of(beckn.add_minutes(w0, -120))
        rec = getattr(self.adapter, "recent", {}) or {}
        vals = [rec[k]["grid_kw"] for k in range(j, j + 4) if k in rec] if j is not None else []
        return round(-sum(vals) / len(vals), 3) if len(vals) == 4 else None

    def build_reading(self, cid: str, slot_id: int) -> dict[str, Any]:
        c = self.contracts[cid]
        s = [x for x in c["commitments"] if x["id"] == slot_id][0]
        i0 = self.replay.idx_of(s["start"])
        t0, t1 = self.adapter.recent[i0], self.adapter.recent[i0 + 1]
        usage = -(t0["grid_kw"] + t1["grid_kw"]) / 2
        usage_cf = -(t0["grid_cf_kw"] + t1["grid_cf_kw"]) / 2
        self.log("tool", tool="meter_reading", input=f"slot {slot_id} ({s['start'][11:]}), committed {s['kw']:g} kW less",
                 output=f"grid use {usage:.2f} kW, battery {t1['soc_pct']:.0f} %" + (", EV paused" if t0["ev_held"] or t1["ev_held"] else ""), ms=0.0)
        return {"contract_id": cid, "event_id": c["event_id"], "home": self.name, "slot_id": slot_id,
                "start": s["start"], "committed_kw": s["kw"], "usage_kw": round(usage, 3),
                "soc_end_pct": t1["soc_pct"], "modes": [t0["mode"], t1["mode"]], "ev_held": t0["ev_held"] or t1["ev_held"],
                "battery_kw": round((t0.get("batt_kw", 0.0) + t1.get("batt_kw", 0.0) + t0.get("v2h_kw", 0.0) + t1.get("v2h_kw", 0.0)) / 2, 3), "pre_kw": self._pre_kw(c),
                "truth": {"usage_without_event_kw": round(usage_cf, 3),
                          "delivered_true_kw": round(usage_cf - usage, 3)}}

    def send_reading(self, txn: str, reading: dict[str, Any]) -> dict[str, Any]:
        body = beckn.make("on_status", sender=self.name, receiver=self.coordinator, transaction_id=txn,
                          message={"performance": reading}, key=self.key, sender_role="bpp",
                          ttl_s=beckn.TTL_EVIDENCE_S)
        out = self.send(self.coordinator, beckn.kind("on_status"), body,
                        f"{self.name}:reading:{reading['contract_id']}:{reading['slot_id']}:{uuid.uuid4().hex[:6]}")
        util = self.cfg.get("utility_name", "utility")
        uout = None
        if util:                     # the meter owner gets the reading straight from the home, signed by the home
            meter = {k: v for k, v in reading.items() if k != "truth"}
            ubody = beckn.make("on_status", sender=self.name, receiver=util, transaction_id=txn,
                               message={"performance": meter}, key=self.key, sender_role="bpp",
                               ttl_s=beckn.TTL_EVIDENCE_S)
            uout = self.send(util, beckn.kind("on_status"), ubody,
                             f"{self.name}:meter:{reading['contract_id']}:{reading['slot_id']}:{uuid.uuid4().hex[:6]}")
        self._mark_reported(reading["contract_id"], reading["slot_id"], [out, uout])
        return {"sent": out, "summary": f"Slot {reading['slot_id']} of {reading['event_id']}: committed "
                                        f"{reading['committed_kw']} kW, meter {reading['usage_kw']} kW used."}

    def _mark_reported(self, cid: str, sid: int, results: list[Any]) -> None:
        """A slot counts as reported only once both sends went through. A send that raised never gets here (the
        reading is due again on the next tick); the fleet sends in the background and hands back futures: the slot
        stays in flight until they finish, and goes back to due if one of them failed."""
        c = self.contracts[cid]
        futs = [r for r in results if isinstance(r, Future)]
        if not futs:
            with self._lock:
                if sid not in c["reported"]:
                    c["reported"].append(sid)
            return
        with self._lock:
            c.setdefault("in_flight", []).append(sid)

        def done(_f: Future) -> None:
            if not all(f.done() for f in futs):
                return
            ok = all(f.exception() is None and f.result() is not False for f in futs)
            with self._lock:
                if sid in c.get("in_flight", []):
                    c["in_flight"].remove(sid)
                    if ok and sid not in c["reported"]:
                        c["reported"].append(sid)
                    elif not ok:
                        self.log("v2_reading_send_failed", contract_id=cid, slot_id=sid)
        for f in futs:
            f.add_done_callback(done)

    # ------------------------------------------------------------------ in-process path (and the fleet)
    def gate(self, kind: str, sender: str, body: dict[str, Any]) -> str | None:
        """None if the message comes from the coordinator this home trusts (Town sender, trusted key, valid signature
        over the whole body); else why not. Every path applies it first: in-process, the fleet, and the daemon before
        it hands a message to the Langflow flows H3 / H4."""
        if sender != self.coordinator or body.get("signer") not in self.trusted() or \
                not beckn.check(body, body.get("signer", "")):
            self.log("v2_rejected_message", message_kind=kind, sender=sender)
            return "rejected: not from a trusted coordinator"
        return None

    def repeat_answer(self, body: dict[str, Any]) -> str | None:
        """A commitment this home has answered already (Town redelivery, or the coordinator re-sending it because
        the answer was lost): the same answer again, without checking or installing anything. None if it is new."""
        txn = body.get("context", {}).get("transactionId", "")
        c = {**((body.get("message") or {}).get("contract") or {}), "transaction_id": txn}
        prev = self.contracts.get(c.get("id"))
        if prev and prev.get("status") in ("ACCEPTED", "DECLINED"):
            v = {"decision": prev["status"], "reason": prev.get("reason", "within_offer"), "contract_id": c.get("id")}
            return self.answer(txn, c, v)["summary"]
        return None

    def handle(self, kind: str, sender: str, body: dict[str, Any]) -> str:
        act = beckn.action_of(kind)
        txn = body.get("context", {}).get("transactionId", "")
        late = beckn.expired(body, default_ttl=float(self.cfg.get("max_message_age_s", 900)))
        if late:                                            # e.g. a call for offers that waited while this home was
            return f"expired {round(late)} s ago: not acted on"   # off: the coordinator has moved on
        rejected = self.gate(kind, sender, body)
        if rejected:
            return rejected
        if act == "discover":
            need = body["message"]["intent"]["need"]
            sl = need.get("slots") or []
            self.log("recv", what="call for offers", sender=sender, event=need.get("event_id"),
                     window=f"{need['window_start'][5:]}-{need['window_end'][11:]}", half_hours=len(sl),
                     kw_wanted=sl[0].get("kw_requested") if sl else None, price=sl[0].get("price") if sl else None)
            t0 = time.perf_counter()
            fc = self.forecast_window(need)
            self._tool("forecast_window", f"event {need['window_start'][5:]}-{need['window_end'][11:]} ({len(sl)} half hours), "
                       f"as of {self.now_ts()[5:]}", self._fc_words(fc), t0)
            t0 = time.perf_counter()
            offer = self.build_offer(need, fc)
            lim = self.adapter.limits()
            soc = fc.get("soc_now_pct")
            self._tool("plan_offer", f"battery {lim['battery_kwh']:g} kWh" + (f" at {soc:.0f} %" if soc is not None else "")
                       + f", reserve {lim['reserve_pct']:.0f} %" + (f"; EV charger {lim['ev_max_kw']:g} kW" if lim.get("ev_max_kw") else ""),
                       " · ".join(f"{x['kw_offered']:g}" for x in offer["slots"]) + f" kW per half hour = {offer['energy_kwh']} kWh, "
                       f"confidence {offer['confidence']}", t0)
            return self.send_offer(txn, offer)["summary"]
        if act == "confirm":
            again = self.repeat_answer(body)
            if again is not None:
                return again
            c = {**body["message"]["contract"], "transaction_id": txn}
            t0 = time.perf_counter()
            v = self.verify_contract(sender, body)
            self._tool("check_commitment", "share " + " · ".join(f"{x['kw']:g}" for x in c.get("commitments", [])) + " kW, "
                       f"opt out by {str(c.get('opt_out_by', ''))[5:]}", f"{v['decision']} ({v['reason']})", t0)
            self.install_plan(c, v)
            return self.answer(txn, c, v)["summary"]
        if act == "cancel":                                 # the coordinator froze the deal without this home's late yes
            cid = (body["message"].get("contract") or {}).get("id")
            c = self.contracts.get(cid)
            if not c:
                return "cancel for a contract this home does not hold: nothing to undo"
            if c.get("status") == "ACCEPTED":
                if hasattr(self.adapter, "clear_plan"):
                    self.adapter.clear_plan(cid)
                if hasattr(self.adapter, "ev_holds"):
                    self.adapter.ev_holds = [x for x in self.adapter.ev_holds
                                             if tuple(x) != (c.get("window_start"), c.get("ev_resume_after"))]
                c["status"], c["reason"] = "CANCELLED", (body["message"].get("contract") or {}).get("reason")
                self.log("v2_contract_cancelled", contract_id=cid, reason=c["reason"])
            return f"contract {cid} cancelled"
        if act == "update":
            self.log("v2_event_notice", **{k: v for k, v in body["message"].items() if k != "contract"})
            return "event notice noted"
        if act == "rt_ask":                                 # the coordinator asks, during the day: how much more?
            return self.rt_answer(txn, body["message"]["ask"])
        if act == "rt_dispatch":                            # what this home owes on top of its booking
            return self.rt_take(txn, body["message"]["dispatch"])
        if act == "on_status" and "settlement" in body["message"]:
            st = body["message"]["settlement"]
            if any(p.get("event_id") == st.get("event_id") for p in self.payments):
                return f"payment for {st.get('event_id')} already noted"      # at-least-once delivery: count it once
            self.payments.append(st)
            self.reputation = float(st.get("reputation", self.reputation))
            self.log("v2_paid", event_id=st.get("event_id"), usd=st.get("payment"), delivered_kwh=st.get("delivered_kwh"))
            return f"payment noted: {st.get('payment')}"
        return f"ignored {kind}"

    # ------------------------------------------------------------------ the real-time design (rt_market.py)
    def rt_answer(self, txn: str, ask: dict[str, Any]) -> str:
        if getattr(self.adapter, "mode", "fixed") == "rolling":
            cap = self.agent.could_give(ask["start"], ask["end"]) if self.agent is not None else self.adapter.rt_capacity(ask["start"], ask["end"])
        else:
            cap = {"kw": 0.0, "why": "fixed plan: this home cannot change its plan during the day"}
        kw = float(cap.get("kw", 0.0))
        if ask.get("price_usd_per_kwh", 0.0) < float(self.cfg.get("rt_min_price", 0.0)):
            kw, cap["why"] = 0.0, "below the owner's minimum price"
        body = beckn.make("rt_answer", sender=self.name, receiver=self.coordinator, transaction_id=txn,
                          message={"answer": {"id": ask["id"], "kw": round(kw, 3), "why": cap.get("why")}},
                          key=self.key, sender_role="bpp")
        self.send(self.coordinator, beckn.kind("rt_answer"), body, f"{self.name}:rt_answer:{ask['id']}:{uuid.uuid4().hex[:6]}")
        self.log("rt_answer", call_id=ask["id"], kw=round(kw, 3), why=cap.get("why"))
        return f"can add {kw:.2f} kW for {ask['id']}"

    def rt_take(self, txn: str, d: dict[str, Any]) -> str:
        """Owe the dispatched kW on top of the booking (a release: less). Covering another home's gap in the booked
        event, the kW is owed under the event (its readings count in the event's settlement); for a call, readings for
        its half hours go out under the call's id, so the buyer can pay for them."""
        applied = getattr(self.adapter, "mode", "fixed") == "rolling"
        if applied:
            self.adapter.rt_dispatch({t: float(v) for t, v in d["steps"].items()})
            if self.agent is not None:
                self.agent.called({t: float(v) for t, v in d["steps"].items()})
        halves: dict[str, float] = {}
        for t, v in d["steps"].items():                       # half hours, as the meter readings go
            h = t[:14] + ("00" if t[14:16] < "30" else "30")
            halves[h] = halves.get(h, 0.0) + float(v) / 2.0
        ev_c = next((c for c in self.contracts.values() if d.get("event_id") and c.get("event_id") == d["event_id"]
                     and c.get("status") == "ACCEPTED" and not c.get("rt")), None)
        if ev_c is not None:
            for h, v in halves.items():
                sid = int(self.replay.minutes_between(ev_c["window_start"], h) // 30)
                cm = next((x for x in ev_c["commitments"] if x["id"] == sid), None)
                if sid in ev_c.get("reported", []):
                    continue                                   # its reading went out already
                if cm is None:
                    ev_c["commitments"].append({"id": sid, "start": h, "kw": round(v, 3), "gap": True})
                else:
                    cm["kw"] = round(cm["kw"] + v, 3)
            ev_c["commitments"].sort(key=lambda x: x["id"])
            where = ev_c["id"]
        else:
            cid = f"{d['id']}:{self.name}"
            c = self.contracts.setdefault(cid, {"id": cid, "event_id": d["id"], "home": self.name, "status": "ACCEPTED",
                                                "commitments": [], "reported": [], "rt": True, "transaction_id": txn,
                                                "price_usd_per_kwh": d.get("price_usd_per_kwh"), "steps": {}})
            for t, v in d["steps"].items():
                c["steps"][t] = round(c["steps"].get(t, 0.0) + float(v), 4)
            per: dict[str, float] = {}
            for t, v in c["steps"].items():
                h = t[:14] + ("00" if t[14:16] < "30" else "30")
                per[h] = per.get(h, 0.0) + v / 2.0
            done = {x["start"]: x for x in c["commitments"] if x["id"] in c["reported"]}
            c["commitments"] = [done.get(h) or {"id": k, "start": h, "kw": round(v, 3)}
                                for k, (h, v) in enumerate(sorted(per.items())) if v > 1e-3 or h in done]
            where = cid
        self.log("rt_dispatch", call_id=d["id"], call_kind=d.get("kind"), steps=len(d["steps"]),
                 kw=round(max(d["steps"].values(), default=0.0), 3), release=bool(d.get("release")), applied=applied,
                 owed_under=where)
        return f"dispatch {d['id']}: {len(d['steps'])} steps"

    REPORT_MIN_KW = 0.5            # a shortfall below this (or 10 % of what is owed) is the plan's noise, not news
    REPORT_AHEAD_STEPS = 12        # report what the next 3 hours look like
    REPORT_EVERY_MIN = 30          # forecast drift is reported at most every 30 dataset minutes; an owner's wish, a
                                   # battery trip or leaving the event at once

    def rt_report(self) -> None:
        """Tell the coordinator when this home's plan says it will fall short of what it owes (an owner's wish, a storm,
        a battery trip), and only when that changes materially: one message, not one every 5 minutes."""
        ctl = getattr(self.adapter, "ctl", None)
        if ctl is None:
            return
        now = self.now_ts()
        i = self.now_idx()
        horizon = self.replay.ts(i + self.REPORT_AHEAD_STEPS)
        if self.adapter.overrides.get("stop"):              # the owner left the event: everything still owed is short
            short = {}
            for k in range(1, self.REPORT_AHEAD_STEPS + 1):
                t = self.replay.ts(i + k)
                if ctl.target_at(t) > 0.1:
                    short[t] = round(ctl.target_at(t), 3)
        elif not ctl.last or not ctl.last.get("ok"):
            return
        else:
            short = {t: v for t, v in (ctl.last.get("short_by_ts") or {}).items()
                     if now < t <= horizon and v >= max(self.REPORT_MIN_KW, 0.1 * ctl.target_at(t))}
        prev = getattr(self, "_rt_reported", {})
        grew = any(v - prev.get(t, 0.0) >= max(self.REPORT_MIN_KW, 0.25 * prev.get(t, 0.0)) for t, v in short.items())
        healed = bool(prev) and not short and any(t > now for t in prev)
        if not grew and not healed:
            return
        why = self._rt_reason()
        urgent = why != "the forecast changed" and why != getattr(self, "_rt_why", None)
        last = getattr(self, "_rt_at", None)
        if not urgent and last and self.replay.minutes_between(last, now) < self.REPORT_EVERY_MIN:
            return
        body = beckn.make("rt_capacity", sender=self.name, receiver=self.coordinator, transaction_id=f"cap-{self.name}",
                          message={"short": {t: round(v, 3) for t, v in short.items()}, "reason": why},
                          key=self.key, sender_role="bpp")
        self.send(self.coordinator, beckn.kind("rt_capacity"), body, f"{self.name}:rt_capacity:{uuid.uuid4().hex[:6]}")
        self._rt_reported, self._rt_at, self._rt_why = short, now, why
        self.log("rt_capacity", short_kw=round(max(short.values(), default=0.0), 3), steps=len(short), reason=why)

    def _rt_reason(self) -> str:
        ov = self.adapter.overrides or {}
        if ov.get("stop"):
            return "the owner left the event"
        if ov.get("ev_now"):
            return "the owner needs the car now"
        if self.adapter.P <= 0 < self.adapter.E:
            return "its battery tripped"
        if ov.get("comfort"):
            return "the owner changed the comfort limit"
        return "the forecast changed"

    def tick(self) -> list[str]:
        self.observe()
        self.rt_report()
        out = []
        for cid, sid in self.due_readings():
            txn = self.contracts[cid].get("transaction_id", "")
            out.append(self.send_reading(txn, self.build_reading(cid, sid))["summary"])
        return out

    def status(self) -> dict[str, Any]:
        tel = self.adapter.tel or {}
        return {"home": self.name, "now": self.now_ts(), "soc_pct": tel.get("soc_pct"), "grid_kw": tel.get("grid_kw"),
                "mode": tel.get("mode"), "offers": len(self.offers),
                "contracts": {k: v.get("status") for k, v in self.contracts.items()},
                "paid_usd": round(sum(float(p.get("payment") or 0) for p in self.payments), 4),
                "reputation": self.reputation, "time": time.time()}
