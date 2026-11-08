"""The home's 5-minute loop (control_mode "rolling"): forecasts, the plan and the owner's limits, for one home.

adapters/simulated.py runs the physics and the guard; this module decides. Every time it is asked (each 15-minute
step while the home owes something, and whenever something changes inside a step: a call, a dispatch, the owner):

1. Forecast the next hours from what the home knows now: the day-ahead forecast for each step (the trained models when
   the home has them, `forecast_fn`), corrected by what the meter and the thermometer read now (the error now,
   fading with a 2-hour half-life). The air conditioning's own day comes from its cooling now and the weather
   forecast; the car from its charger (energy still needed, departure).
2. Run the no-event twin forward on the same forecasts: the reference the home's promises are measured against.
3. Solve rt_control's linear program: battery, car and cooling for every step to the end of the horizon (the last
   thing owed plus 3 hours, so the catch-up is planned too). The first step is executed; the rest is re-planned.
"""
from __future__ import annotations

import math
import os
from typing import Any, Callable

from . import beckn
from .rt_control import Limits, Problem, capacity, solve

BACKUP = os.environ.get("FG_BATTERY_MODE") == "backup"
LEAD_STEPS = 16          # start planning 4 h before the first thing owed (pre-charge, pre-cool)
POST_STEPS = 12          # and keep planning 3 h after the last (the catch-up)
MAX_STEPS = 64           # 16 h at most
HALF_LIFE_H = 2.0        # how fast the error seen now fades from the forecast


def cop_cool(t_out_c: float) -> float:
    return max(2.0, min(4.5, 4.5 - 0.1 * (t_out_c - 25.0)))


def cop_heat(t_out_c: float) -> float:
    return max(1.75, min(4.0, 1.5 + 0.08 * (t_out_c + 10.0)))          # NEEP floor: COP 1.75 at 5 F


class RollingController:
    def __init__(self, sim, house, limits: Limits | None = None,
                 forecast_fn: Callable[[int, list[str]], dict[str, tuple[float, float]]] | None = None):
        self.sim, self.house = sim, house
        self.limits = limits or Limits()
        self.forecast_fn = forecast_fn                    # (now_idx, ts list) -> {ts: (pv_kw, base_kw)}
        self.last: dict[str, Any] | None = None           # the latest plan (for the record and the coordinator)
        self.replans = 0
        self.solve_ms: list[float] = []

    # ------------------------------------------------------------------ what the home owes
    def target_at(self, ts: str) -> float:
        act = self.sim.plan.get(ts) or {}
        return float(act.get("deliver_kw", 0.0)) + float(self.sim.rt_extra.get(ts, 0.0))

    def _owed_span(self, i: int) -> tuple[int, int] | None:
        """First and last step index with something owed, among the steps near i."""
        idx = [j for j in range(max(0, i - POST_STEPS), i + LEAD_STEPS + 1) if self.target_at(self.sim.replay.ts(j)) > 1e-6]
        if not idx:
            return None
        last = idx[-1]
        j = last + 1
        while self.target_at(self.sim.replay.ts(j)) > 1e-6 and j < i + MAX_STEPS:
            j += 1
        return idx[0], j - 1

    def active(self, i: int) -> bool:
        if self._owed_span(i):
            return True
        if self.house and abs(self.sim.x) > 0.05:
            return True                                    # the house is still away from its own day
        s = self.sim
        if s.has_ev and s.ev_left > 0.01 and s.ev_deadline and self.ev_controlled(s.replay.ts(i)):
            return True                                    # its car is still waiting for its staggered restart
        return False

    def ev_controlled(self, ts: str) -> bool:
        """May the plan pause the car at this step? Inside a hold the coordinator gave (the window to its
        staggered restart) or a step with a real-time dispatch on top."""
        s = self.sim
        if s.levers is not None and not s.levers.get("ev"):
            return False                                   # the node agent hasn't opened the car
        return any(a <= ts < b for a, b in s.ev_holds) or s.rt_extra.get(ts, 0.0) > 1e-6

    # ------------------------------------------------------------------ forecasts
    def _forecast(self, i: int, ts: list[str]) -> tuple[list[float], list[float]]:
        rp = self.sim.replay
        fc = self.forecast_fn(i, ts) if self.forecast_fn else {}
        pv, load = [], []
        for t in ts:
            if t in fc:
                pv.append(float(fc[t][0])); load.append(float(fc[t][1]))
                continue
            j = rp.idx_of(t)                               # yesterday at the same time, when nothing better
            y = rp.row(j - 96) if j is not None and j >= 96 else None
            pv.append(y["pv_kw"] if y else 0.0); load.append(y["base_kw"] if y else 0.5)
        # correction from what the meter reads now
        row = rp.row(i)
        pv_ratio = (row["pv_kw"] + 0.05) / (pv[0] + 0.05)
        pv_ratio = max(0.2, min(2.0, pv_ratio))
        load_err = row["base_kw"] - load[0]
        out_pv, out_load = [], []
        for k in range(len(ts)):
            w = 0.5 ** (k * 0.25 / HALF_LIFE_H)
            out_pv.append(max(0.0, pv[k] * (1.0 + (pv_ratio - 1.0) * w)))
            out_load.append(max(0.05, load[k] + load_err * w))
        out_pv[0], out_load[0] = row["pv_kw"], row["base_kw"]       # step 0: what it reads now
        return out_pv, out_load

    def _cooling(self, i: int, ts: list[str], load: list[float]) -> tuple[list[float], list[float], list[int]]:
        """The cooling's own day ahead: its kW now, moved with the forecast temperature; COP from the forecast."""
        rp, h = self.sim.replay, self.house
        row = rp.row(i)
        t_now = row.get("temp_c", 25.0)
        c_now = h.at(ts[0])[0]
        cool, cop, occ = [], [], []
        sg = h.sign(ts[0])                                  # +1 cooling, -1 a heat pump heating
        for k, t in enumerate(ts):
            j = rp.idx_of(t)
            r = rp.row(j) if j is not None else row
            t_fc = r.get("fc_temp_c", t_now) if k else t_now
            if k:                                           # the forecast as issued, moved by today's error now
                t_fc += (t_now - row.get("fc_temp_c", t_now)) * 0.5 ** (k * 0.25 / HALF_LIFE_H)
            same_hour_c = h.at(t)[0] if k == 0 else max(0.0, c_now * (1.0 + 0.12 * sg * (t_fc - t_now)))
            cool.append(min(load[k], same_hour_c))
            cop.append(cop_cool(t_fc) if sg > 0 else cop_heat(t_fc))
            occ.append(int(h.at(t)[3]))                     # the household's routine is known to its home
        return cool, cop, occ

    def _twin_ref(self, ts: list[str], dt: list[float], pv: list[float], load: list[float], ev_kw: list[float]) -> list[float]:
        """The no-event twin's net import on the forecasts, from the twin's battery state now (default job)."""
        s = self.sim
        soc = s.cf["soc"]
        out = []
        for k in range(len(ts)):
            net = load[k] + ev_kw[k] - pv[k]
            can_dis, can_ch = s._caps(soc, dt[k])
            b = (0.0 if BACKUP else min(net, can_dis)) if net > 0 else -min(-net, can_ch)
            soc = s._soc_after(soc, b, dt[k])
            out.append(net - b)
        return out

    def _twin_ev(self, ts: list[str], dt: list[float]) -> list[float]:
        s = self.sim
        left, dl = s.cf["ev_left"], s.cf["ev_deadline"]
        out = []
        for k, t in enumerate(ts):
            if dl and t >= dl:
                left, dl = 0.0, None
            kw = min(s.ev_max, left / dt[k]) if (s.has_ev and left > 0.01 and dl) else 0.0
            left = max(0.0, left - kw * dt[k])
            out.append(kw)
        return out

    # ------------------------------------------------------------------ the plan
    def problem(self, i: int, first_dt: float = 0.25) -> Problem:
        s, rp = self.sim, self.sim.replay
        span = self._owed_span(i)
        end = (span[1] + POST_STEPS) if span else i + POST_STEPS
        n = max(4, min(MAX_STEPS, end - i + 1))
        ts = [rp.ts(i + k) for k in range(n)]
        dt = [first_dt] + [0.25] * (n - 1)
        pv, load = self._forecast(i, ts)
        if self.house:
            cool, cop, occ = self._cooling(i, ts, load)
        else:
            cool, cop, occ = [0.0] * n, [3.0] * n, [1] * n
        ref = self._twin_ref(ts, dt, pv, load, self._twin_ev(ts, dt))
        target = [self.target_at(t) for t in ts]
        if getattr(s, "app", None):                  # benchmark: what the other devices will not draw, and their catch-up
            left = sum(s.app_q.values())
            last = None
            for k, t in enumerate(ts):
                if target[k] > 1e-6:
                    a = s.app_now(t) if k == 0 else self.app_fc(t)
                    load[k] = max(0.05, load[k] - a)
                    left += a * dt[k]
                    last = k
            k = (last + 1 if last is not None else 0) + int(self.app_delay_min() // 15)
            while left > 1e-6 and k < n:
                kw = min(3.0, left / dt[k])
                load[k] += kw
                left -= kw * dt[k]
                k += 1
        # the car: the plan may pause it only where the coordinator asked (the window to its staggered restart) and
        # in the steps of a real-time dispatch; elsewhere it charges as it always does (full power until done), which
        # the plan takes as given: a home must not push its car's catch-up past its horizon on its own
        ev_until, ev_need, hold, ctrl = None, 0.0, None, [False] * n
        if s.has_ev and s.ev_left > 0.01 and s.ev_deadline and not s.overrides.get("ev_now"):
            ctrl = [self.ev_controlled(t) for t in ts]
            left, dflt = s.ev_left, []
            for k, t in enumerate(ts):
                if t >= s.ev_deadline or left <= 0.01 or ctrl[k]:
                    dflt.append(0.0)
                    continue
                kw = min(s.ev_max, left / dt[k])
                left -= kw * dt[k]
                dflt.append(kw)
            load = [x + y for x, y in zip(load, dflt)]
            last_ctrl = max((k for k in range(n) if ctrl[k] and ts[k] < s.ev_deadline), default=None)
            if last_ctrl is not None:
                ev_until = last_ctrl + 1
                after_h = max(0.0, rp.minutes_between(ts[-1], s.ev_deadline) / 60.0 - 0.25)
                ev_need = max(0.0, left - s.ev_max * after_h * 0.95)       # what the free hours cannot cover
                hold = [not c for c in ctrl]                                 # outside its steps the plan leaves it
        elif s.has_ev and s.ev_left > 0.01 and s.ev_deadline:            # the owner's "now": full power, given
            left = s.ev_left
            for k in range(n):
                kw = min(s.ev_max, left / dt[k]) if (left > 0.01 and ts[k] < s.ev_deadline) else 0.0
                left -= kw * dt[k]
                load[k] += kw
        lim = self.limits
        if s.overrides.get("comfort"):
            lim = Limits(**{**lim.as_dict(), **s.overrides["comfort"]})
        h = self.house
        price = [max(0.0, float(rp.row(rp.idx_of(t) if rp.idx_of(t) is not None else i).get("price_da_usd_mwh", 50.0))) / 1000.0
                 for t in ts]
        return Problem(ts=ts, dt=dt, pv=pv, load=load, cool=cool, cop=cop, occ=occ, ref=ref, target=target, price=price,
                       watch=[t <= 1e-6 for t in target], soc=s.soc, E=s.E, P=s.P, R=s.R, SM=s.SM, eta=s.eta,
                       export_limit=s.export_limit, ev_max=s.ev_max if s.has_ev else 0.0,
                       ev_left=ev_need if ev_until else 0.0, ev_until=ev_until, ev_hold=hold, x0=s.x,
                       ua=h.ua if h else 0.0, cap=h.cap if h else 1.0, qmax=h.qmax if h else 0.0, limits=lim, P_ch=s.P_ch,
                       P_dis=None if s.levers is None else float(s.levers.get("battery_kw", 0.0)))

    # --- benchmark: the node's other smart systems (flexgrid_core/devices.py) -------------------------------------------
    def app_holding(self, ts: str) -> bool:
        return self.target_at(ts) > 1e-6

    def app_delay_min(self) -> int:
        """Catch-up start after the hold ends: past the evening's own peak, then fixed per node over the catalog's
        stagger, so nodes do not all start at once."""
        import zlib
        from . import devices
        n = devices.stagger_min() // 15 + 1
        return devices.catch_up_after_min() + 15 * (zlib.crc32(str(self.sim.cfg.get("name") or self.sim.cfg.get("agent_name") or "").encode()) % n)

    def _app_days(self, j0: int, j1: int, days: int = 14, device: str | None = None) -> list[float]:
        """Mean kW the devices can give over steps j0..j1 on each of the last `days` days, only days before now."""
        rp, s = self.sim.replay, self.sim
        now = rp.current() if s.idx is None else s.idx
        out, d = [], 1
        while len(out) < days and d < 3 * days:
            a, b = j0 - 96 * d, j1 - 96 * d
            d += 1
            if a < 0:
                break
            if b > now:
                continue
            vals = [s.dev_now(rp.ts(k)) for k in range(a, b)]
            out.append(sum(sum(v.values()) if device is None else v.get(device, 0.0) for v in vals) / max(1, b - a))
        return out

    def app_fc(self, ts: str) -> float:
        """What the devices are expected to give at this step: the same step's mean over the last 14 days."""
        c = getattr(self, "_app_cache", None)
        if c is None:
            c = self._app_cache = {}
        if ts not in c:
            j = self.sim.replay.idx_of(ts)
            v = self._app_days(j, j + 1) if j is not None else []
            c[ts] = sum(v) / len(v) if v else 0.0
            if len(c) > 3000:
                c.clear()
        return c.get(ts, 0.0)

    def app_offer(self, window_start: str, window_end: str, confidence: float = 0.8) -> dict[str, Any]:
        """How many kW the devices can give through tomorrow's window, evenly: what they gave in the same clock window
        on the last 14 days, on average, times the same confidence as the air conditioning's offer. A dryer runs some
        days and not others, so no device is promised on its own: the node's meter settles what it really gave."""
        rp, s = self.sim.replay, self.sim
        w0, w1 = rp.idx_of(window_start), rp.idx_of(window_end)
        if w0 is None or w1 is None or not getattr(s, "app", None):
            return {"kw": 0.0, "why": "no controllable devices"}
        v = self._app_days(w0, w1)
        if len(v) < 5:
            return {"kw": 0.0, "why": "not enough history"}
        by = {}
        for k in s.app:
            if k not in s.dev_off and k not in s.kept_out:
                d = self._app_days(w0, w1, device=k)
                by[k] = round(sum(d) / len(d), 3) if d else 0.0
        m = sum(v) / len(v)
        return {"kw": round(m * confidence, 3), "mean_kw": round(m, 3), "days": len(v), "by_device": by}

    def decide(self, i: int, first_dt: float = 0.25) -> dict[str, Any]:
        import time
        t0 = time.perf_counter()
        p = self.problem(i, first_dt)
        res = solve(p)
        self.solve_ms.append((time.perf_counter() - t0) * 1000)
        self.replans += 1
        if not res.get("ok"):
            self.last = {"ok": False, "at": p.ts[0], "why": res.get("message")}
            return self.last
        owed = [(t, x, sh) for t, x, sh in zip(p.ts, p.target, res["short"]) if x > 1e-6]
        self.last = {"ok": True, "at": p.ts[0], "b": res["b"][0], "e": res["e"][0], "u": res["u"][0],
                     "x_plan": res["x"][:8], "short_kwh": res["short_kwh"],
                     "short_by_ts": {t: round(sh, 3) for t, x, sh in owed if sh > 1e-3},
                     "owed_kw": {t: round(x, 3) for t, x, _ in owed}, "steps": len(p.ts), "problem": p, "plan": res}
        return self.last

    def capacity(self, i: int, call_from: str, call_to: str) -> dict[str, Any]:
        """What this home can add, evenly, between call_from and call_to on top of what it owes (kW), with the
        owner's call limits. Planned from the state now, as a re-plan would."""
        rp = self.sim.replay
        p = self.problem(i)
        end = rp.idx_of(call_to)
        if end is not None:
            n = min(MAX_STEPS, end - i + POST_STEPS)
            if n > len(p.ts):                              # the horizon must cover the call and its catch-up
                self._extend(p, [rp.ts(i + k) for k in range(n)])
        plan = solve(p)
        call = [call_from <= t < call_to for t in p.ts]
        if not any(call):
            return {"kw": 0.0, "why": "the call is outside the planning horizon"}
        kw = capacity(p, call, plan if plan.get("ok") else None)
        return {"kw": round(kw, 3), "steps": sum(call),
                "short_kwh_now": round(plan.get("short_kwh", 0.0), 3) if plan.get("ok") else None}

    # ------------------------------------------------------------------ the day-ahead offer of the air conditioning
    def _cool_fit(self, j: int) -> tuple[float, float] | None:
        """cooling kW = a + b x outdoor C for this hour of day, on the previous 21 days of the house's own record."""
        rp, h = self.sim.replay, self.house
        h0 = j - (j % 4)
        xs, ys = [], []
        now = self.sim.replay.current() if self.sim.idx is None else self.sim.idx
        for d in range(1, 30):
            k = h0 - 96 * d
            if k < 0 or len(xs) >= 84:
                break
            if k >= now:
                continue
            for q in range(4):
                r = rp.row(k + q)
                xs.append(r["temp_c"]); ys.append(h.at(r["ts_local"])[0])
        if len(xs) < 20:
            return None
        mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
        sxx = sum((x - mx) ** 2 for x in xs)
        if sxx < 1e-6:
            return my, 0.0
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
        return my - b * mx, b

    def ac_offer(self, window_start: str, window_end: str, confidence: float = 0.8) -> dict[str, Any]:
        """How many kW the air conditioning can give through tomorrow's window, evenly, within the owner's
        limits: the house's physics on a forecast of its cooling (its own record against the temperature, the
        forecast as issued), pre-cooling allowed before. Only data from before now."""
        if not self.house:
            return {"kw": 0.0, "why": "no air conditioning in the dataset"}
        rp = self.sim.replay
        w0, w1 = rp.idx_of(window_start), rp.idx_of(window_end)
        if w0 is None or w1 is None:
            return {"kw": 0.0, "why": "window outside the dataset"}
        i0 = max(0, w0 - LEAD_STEPS)
        ts = [rp.ts(k) for k in range(i0, min(len(rp.rows), w1 + POST_STEPS))]
        cool, cop, occ = [], [], []
        for t in ts:
            j = rp.idx_of(t)
            r = rp.row(j)
            t_fc = r.get("fc_temp_c", r["temp_c"])
            fit = self._cool_fit(j)
            c = max(0.0, fit[0] + fit[1] * t_fc) if fit else 0.0
            cf = cop_cool(t_fc) if self.house.sign(t) > 0 else cop_heat(t_fc)
            c = min(c, self.house.qmax / max(1.0, cf))
            cool.append(c); cop.append(cf); occ.append(int(self.house.at(t)[3]))
        n = len(ts)
        p = Problem(ts=ts, dt=[0.25] * n, pv=[0.0] * n, load=list(cool), cool=cool, cop=cop, occ=occ, ref=list(cool),
                    target=[0.0] * n, watch=[True] * n, ua=self.house.ua, cap=self.house.cap, qmax=self.house.qmax,
                    limits=self.limits, x0=0.0)
        call = [window_start <= t < window_end for t in ts]
        kw = capacity(p, call, None)
        return {"kw": round(kw * confidence, 3), "raw_kw": round(kw, 3), "cooling_forecast_kw":
                round(sum(c for c, w in zip(cool, call) if w) / max(1, sum(call)), 3)}

    def _extend(self, p: Problem, ts: list[str]) -> None:
        """Lengthen a problem to these steps (the call reaches past the plan): yesterday's values, nothing owed."""
        rp = self.sim.replay
        while len(p.ts) < len(ts):
            t = ts[len(p.ts)]
            j = rp.idx_of(t)
            y = rp.row(j - 96) if j is not None and j >= 96 else {"pv_kw": 0.0, "base_kw": 0.5}
            p.ts.append(t); p.dt.append(0.25); p.pv.append(y["pv_kw"]); p.load.append(y["base_kw"])
            p.cool.append(p.cool[-1] if p.cool else 0.0); p.cop.append(p.cop[-1] if p.cop else 3.0)
            p.occ.append(int(self.house.at(t)[3]) if self.house else 1)
            p.ref.append(p.ref[-1] if p.ref else y["base_kw"]); p.target.append(0.0); p.watch.append(True)
            if p.price is not None:                        # the day-ahead price of that step (known the day before)
                r = rp.row(j) if j is not None else {}
                p.price.append(max(0.0, float(r.get("price_da_usd_mwh", 50.0))) / 1000.0)
            if p.ev_hold is not None:
                p.ev_hold.append(False)


def to_minutes(ts: str) -> int:
    return int(ts[11:13]) * 60 + int(ts[14:16])


def add_minutes(ts: str, m: float) -> str:
    return beckn.add_minutes(ts, m)
