"""A home driven by the dataset replay: real 2023 solar and load, simulated battery and EV.

Default mode is self-consumption (what a Powerwall or IQ Battery does out of the box): the battery covers the
home's deficit and soaks up its solar surplus; the EV charges at full power from arrival (unmanaged, as in the
dataset's `ev_load_kw`).

Two ways to change that:
  * v1: a one-shot dispatch (`apply`) sets a temporary override that expires on its own.
  * v2: a contract plan (`set_plan`): per 15-minute step, "deliver X kW less import than I would have used"
    (EV paused first, then the battery), "pre-charge to SOC s by the event start", and an EV hold window (the
    coordinator picks the restart time; the hold always ends early enough to finish charging by departure, and the
    restarted EV charges from the grid so the battery's energy stays with the house for the evening).

A twin of the home runs alongside in default mode, so every step also knows what the meter would have read
with no event (`grid_cf_kw`). That counterfactual is truth for evaluation, like the dataset's *_true columns:
it is never used for settlement, which uses the utility's baseline method.

control_mode "rolling" (the real-time design, flexgrid_core/rt_home.py): while the home owes something, every 15
minutes and whenever something changes, a linear program re-plans the battery, the car and the air conditioning
(the house's own physics, hvac.py), and the step runs in three 5-minute parts, each checked by a guard: battery
within its limits, car charged by departure and never held in its last hours, house within the owner's limits, the
owner's "I need it now" first. "fixed" (the default) is the day-ahead design above, step for step as before.
"""
from __future__ import annotations

import json
from typing import Any

import os
from ..config import state_dir
from .. import devices, hvac, rt_control, rt_home

DONE_BY = {"dish": "23:30", "dryer": "22:45"}      # the owner's done-by times: a held machine finishes by then

BACKUP = os.environ.get("FG_BATTERY_MODE") == "backup"   # benchmark: the battery is held for grid events
SUB = 3                     # 5-minute parts of a 15-minute step (control_mode rolling)


class SimulatedHome:
    name = "simulated"

    def __init__(self, cfg: dict[str, Any], replay):
        self.cfg, self.replay = cfg, replay
        self.E = float(cfg["battery_kwh"]); self.P = float(cfg["battery_kw"])
        self.R = float(cfg["reserve_pct"]) / 100; self.SM = float(cfg["soc_max_pct"]) / 100
        self.eta = float(cfg.get("eta", 0.95)); self.ev_max = float(cfg["ev_max_kw"])
        self.P_ch = float(os.environ.get("FG_CHARGE_KW") or 1e9)      # benchmark: Powerwall 3 charges at 5 kW
        self.has_ev = bool(cfg.get("has_ev", True))
        self.export_limit = float(cfg["export_limit_kw"])
        self.soc0 = float(cfg["soc0_pct"]) / 100
        self.soc = self.soc0
        self.ev_left = 0.0; self.ev_deadline = None; self.ev_session = None
        self.cf = {"soc": self.soc, "ev_left": 0.0, "ev_deadline": None}      # the no-event twin
        self.override: dict[str, Any] | None = None
        self.ev_log: list[dict[str, Any]] = []       # every car that left: what it still lacked (the owner's measure)
        self.cf_ev_log: list[dict[str, Any]] = []
        self.plan: dict[str, dict[str, Any]] = {}                               # ts -> action (v2 contract)
        self.ev_holds: list[tuple[str, str]] = []                              # [(from_ts, until_ts)], one per contract
        self.fault: dict[str, Any] | None = None
        self.idx: int | None = None
        self.prev = None          # state before the current step (for re-simulating after apply)
        self.cf_before: dict[str, Any] = dict(self.cf)
        self.tel: dict[str, Any] = {}
        self.frozen_reading: dict[str, Any] | None = None
        self.recent: dict[int, dict[str, Any]] = {}                            # idx -> telemetry, last ~4 days
        # the real-time design (control_mode "rolling")
        self.mode = cfg.get("control_mode", "fixed")
        self.x = 0.0                                                            # indoor C minus the house's own day
        self.rt_extra: dict[str, float] = {}                                    # ts -> kW dispatched on top (calls)
        self.overrides: dict[str, Any] = {}                                     # the owner: ev_now, comfort, stop
        self.events: list[tuple[str, Any]] = []                                 # (minute "YYYY-MM-DD HH:MM", fn)
        self.house = hvac.House.of(cfg.get("name") or cfg.get("agent_name"), getattr(replay, "profile", "")) \
            if self.mode == "rolling" else None
        self.ctl = rt_home.RollingController(self, self.house, rt_control.Limits(**(cfg.get("owner_limits") or {}))) \
            if self.mode == "rolling" else None
        self.prev_x = 0.0
        # benchmark: the node's other smart systems (flexgrid_core/devices.py)
        node = "building" if getattr(self, "name", "") == "building" else "home"
        self.app, self.dev_arr, self.dev_row = devices.for_node(node, cfg.get("name") or cfg.get("agent_name") or "")
        self.dev_off: set[str] = set()                         # what the offer left out: replaced at each offer
        self.kept_out: set[str] = set()                        # a person's word: it stays through offers until the person takes it back
        self.app_q: dict[str, float] = {}                      # kWh waiting, by device
        self.app_log: dict[str, float] = {}                    # kWh it gave, by device
        self.app_hold_end: str | None = None
        self.v2h_day, self.v2h_used = None, 0.0
        self.ev_wait: dict[str, Any] | None = None       # a car plugged in and waiting for its timer: the session to come
        self.ev_skip: dict[str, Any] | None = None       # a session already dealt with: the owner started it early, or its car left first
        self.prev_wait: tuple[Any, Any] = (None, None)
        self.v2h_owed = self.prev_owed = 0.0             # kWh a two-way car gave and must get back before it leaves
        self.e_part = self.meter_room = 0.0              # in this 5-minute part: what the car draws, and what the meter can still send out
        # new data holds a two-way car to its card: the home's car circuit, the owner's kWh counted leaving the car, no giving while
        # it draws. Old data has no plug-in times and stays as signed off.
        self.car_rules = bool(getattr(replay, "plugins", None))
        self.v2h_kw = 0.0                     # what a two-way car gave this step: it earns the battery rate
        # the node agent's device tools (flexgrid_core/node): None = the controller may use every device. A dict = only what
        # the agent's calls opened, until they end: battery_kw, ev (the plan may pause the car), ev_kw (its floor), v2h_kw
        self.levers: dict[str, Any] | None = None

    # --- physics -------------------------------------------------------------------
    def _caps(self, soc: float, dt: float = 0.25) -> tuple[float, float]:
        P = self.P if self.levers is None else min(self.P, float(self.levers.get("battery_kw", 0.0)))
        can_dis = max(0.0, min(P, (soc - self.R) * self.E * self.eta / dt))
        can_ch = max(0.0, min(self.P, self.P_ch, (self.SM - soc) * self.E / self.eta / dt))
        return can_dis, can_ch

    def _soc_after(self, soc: float, b: float, dt: float = 0.25) -> float:
        if self.E <= 0:                                            # a home without a battery (b is always 0)
            return soc
        soc += (-b) * dt * self.eta / self.E if b < 0 else -b * dt / self.eta / self.E
        return min(self.SM, max(self.R * 0.999, soc))

    def _ev_demand(self, ev_left: float, ev_deadline: str | None) -> float:
        return min(self.ev_max, ev_left / 0.25) if (self.has_ev and ev_left > 0.01 and ev_deadline) else 0.0

    def _backup_refill(self, ts, soc: float, can_ch: float) -> float:
        """Benchmark's held-for-the-grid mode: no discharge for the house; surplus solar charges it, and overnight it
        refills from the grid at a gentle 2 kW (backup batteries keep themselves full)."""
        if ts and (ts[11:13] >= "22" or ts[11:13] < "06") and soc < self.SM - 1e-3:
            return -min(can_ch, 2.0)
        return 0.0

    def _default(self, pv: float, base: float, ev: float, soc: float, ts=None) -> float:
        can_dis, can_ch = self._caps(soc)
        net = base + ev - pv                                       # >0 deficit
        if BACKUP and net > 0:
            return self._backup_refill(ts, soc, can_ch)
        return min(net, can_dis) if net > 0 else -min(-net, can_ch)

    def _clip_export(self, grid: float) -> tuple[float, float]:
        return (self.export_limit, grid - self.export_limit) if grid > self.export_limit else (grid, 0.0)

    def _step(self, i: int) -> None:
        row = self.replay.row(i)
        ts = row["ts_local"]
        self._plug_in(i, ts)
        sess = self.replay.session_starting(i)
        if sess:
            if self.ev_skip and self.ev_skip["arrival"] == sess["arrival"]:
                self.ev_skip = None                                        # it already started, or its car has gone
            else:
                carry = self._owed_now() if sess.get("plug_in") else 0.0   # a new session must not wipe what the car gave
                leaves = min(self.ev_deadline, sess["deadline"]) if self.ev_wait and self.ev_deadline else sess["deadline"]   # the owner may have moved it up
                self.ev_wait, self.v2h_owed = None, carry
                self.ev_left, self.ev_deadline, self.ev_session = sess["kwh"] + carry, leaves, sess["arrival"]
            self.cf["ev_left"], self.cf["ev_deadline"] = sess["kwh"], sess["deadline"]
        if self.ev_deadline and ts >= self.ev_deadline:
            self._car_left(ts)
        if self.cf["ev_deadline"] and ts >= self.cf["ev_deadline"]:
            self.cf_ev_log.append({"left_at": ts, "unmet_kwh": round(self.cf["ev_left"], 3)})
            self.cf["ev_left"], self.cf["ev_deadline"] = 0.0, None
        if self.override and i > self.override["until_idx"]:
            self.override = None                                    # expiry = automatic revert
        self.prev = (self.soc, self.ev_left, self.ev_deadline, self.ev_session)
        self.prev_owed, self.prev_wait = self.v2h_owed, (self.ev_wait, self.ev_skip)
        self.prev_x = self.x
        self.cf_before = dict(self.cf)
        self._simulate(i, row)

    def _plug_in(self, i: int, ts: str) -> None:
        """A car on a timer is plugged in hours before its session starts: from here it can give, and it draws nothing yet.
        It reads as plugged in (ev_deadline) with nothing to charge (ev_left 0) until the session starts."""
        plug = self.replay.session_plugging(i)
        if not plug or not self.has_ev:
            return
        if self.ev_deadline and ts >= self.ev_deadline:
            self._car_left(ts)                                      # it left and came back in the same step
        self.ev_deadline = max(self.ev_deadline or "", plug["deadline"])
        if plug["arrival"] > ts:
            self.ev_wait = plug

    def _car_left(self, ts: str) -> None:
        w = self.ev_wait
        self.ev_log.append({"arrival": w["arrival"] if w else self.ev_session, "left_at": ts, "unmet_kwh": round(self.ev_left + (w["kwh"] + self.v2h_owed if w else 0.0), 3)})
        if w and w["arrival"] > ts:
            self.ev_skip = w                                        # gone before its timer: that session never runs
        self.ev_left, self.ev_deadline, self.ev_wait, self.v2h_owed = 0.0, None, None, 0.0

    def _car_now(self) -> None:
        """The owner's "now" beats the timer: the session waiting for it starts here, with what the car gave."""
        w = self.ev_wait
        if w and self.overrides.get("ev_now"):
            self.ev_left += w["kwh"] + self.v2h_owed
            self.ev_session, self.ev_wait, self.ev_skip = w["arrival"], None, w

    def _rewind(self) -> None:
        """Back to the start of the step, for running it again."""
        self.soc, self.ev_left, self.ev_deadline, self.ev_session = self.prev
        self.v2h_owed, (self.ev_wait, self.ev_skip) = self.prev_owed, self.prev_wait

    def _owed_now(self) -> float:
        """What the car still has to get back. A session already drawing carries it in ev_left, and owed energy goes in last."""
        return self.v2h_owed if self.ev_wait else min(self.v2h_owed, self.ev_left)

    def _twin(self, row: dict[str, Any]) -> float:
        pv, base = row["pv_kw"], row["base_kw"]
        ev = self._ev_demand(self.cf["ev_left"], self.cf["ev_deadline"])
        b = self._default(pv, base, ev, self.cf["soc"], row["ts_local"])
        self.cf["soc"] = self._soc_after(self.cf["soc"], b)
        self.cf["ev_left"] = max(0.0, self.cf["ev_left"] - ev * 0.25)
        return self._clip_export(pv + b - base - ev)[0]

    def _simulate(self, i: int, row: dict[str, Any]) -> None:
        self._car_now()
        if self.mode == "rolling":
            return self._simulate_rolling(i, row)
        pv, base, ts = row["pv_kw"], row["base_kw"], row["ts_local"]
        grid_cf = self._twin(row)
        ov, act = self.override, self.plan.get(ts)
        ev = self._ev_demand(self.ev_left, self.ev_deadline)
        held = ev > 0 and any(a <= ts < b for a, b in self.ev_holds)
        if held and self.ev_deadline:                               # never hold an EV past the point where it
            need_min = self.ev_left / max(0.1, self.ev_max) * 60    # could no longer finish by its departure
            held = self.replay.minutes_between(ts, self.ev_deadline) - 15 > need_min
        ev_now, stop = bool(self.overrides.get("ev_now")), bool(self.overrides.get("stop"))
        if held and not ev_now:
            ev = 0.0                                                # staggered restart after an event
        if ov and ov.get("ev_kw_max") is not None and not ev_now:   # v1 dispatch throttles or pauses the EV
            ev = min(ev, ov["ev_kw_max"])
        held = held and not ev_now
        can_dis, can_ch = self._caps(self.soc)
        # a managed restart after an event charges the EV from the grid in its assigned slot; the battery keeps its
        # energy for the house's evening load (that is what the coordinator's evening forecast assumes)
        managed = bool(self.ev_session and any(self.ev_session[:10] == a[:10] for a, _ in self.ev_holds))
        b = self._default(pv, base, 0.0 if managed else ev, self.soc, ts)
        mode, ev_cut = "self_consumption", 0.0
        if ov:
            target = ov["battery_kw"]
            b = min(target, can_dis) if target >= 0 else -min(-target, can_ch)
            mode = "dispatch"
        elif act and "deliver_kw" in act and not stop:              # v2: X kW less import than the twin
            extra = (grid_cf + act["deliver_kw"]) - (pv + b - base - ev)
            if extra > 0 and ev > 0 and not ev_now:                 # EV first: the energy is only deferred
                ev_cut = min(ev, extra)
                ev -= ev_cut
                extra -= ev_cut
            if extra > 0:
                nb = min(can_dis, b + extra)
                extra -= nb - b
                b = nb
            mode = "deliver"
        elif act and "precharge_to" in act:                         # v2: be full when the event starts
            steps_left = max(1, act["by_idx"] - i)
            need_kwh = (act["precharge_to"] - self.soc) * self.E
            if need_kwh > 1e-3:
                req = need_kwh / (steps_left * 0.25) / self.eta
                if req > max(0.0, -b):
                    b = -min(can_ch, req)
                mode = "precharge"
        self.soc = self._soc_after(self.soc, b)
        self.ev_left = max(0.0, self.ev_left - ev * 0.25)
        grid, curtailed = self._clip_export(pv + b - base - ev)
        self.idx = i
        self.tel = {
            "ts": ts, "idx": i, "pv_kw": round(pv, 3), "load_kw": round(base, 3),
            "ev_kw": round(ev, 3), "batt_kw": round(b, 3), "soc_pct": round(self.soc * 100, 2),
            "grid_kw": round(grid, 3), "curtailed_kw": round(curtailed, 3), "mode": mode,
            "ev_plugged": bool(self.ev_deadline), "ev_need_kwh": round(self.ev_left, 2),
            "ev_deadline": self.ev_deadline, "ev_held": held or ev_cut > 0,
            "price_usd_mwh": row["price_usd_mwh"], "override": dict(ov) if ov else None,
            "plan": dict(act) if act else None, "grid_cf_kw": round(grid_cf, 3), "soc_cf_pct": round(self.cf["soc"] * 100, 2),
            "source": "replay",
        }
        self.recent[i] = self.tel
        if len(self.recent) > 400:
            for k in sorted(self.recent)[:-384]:
                self.recent.pop(k, None)

    # --- the real-time design (control_mode "rolling") -------------------------------------------
    def _sub_ts(self, ts: str, j: int) -> str:
        return rt_home.add_minutes(ts, j * 15 // SUB)

    def _due_events(self, ts: str, j: int) -> bool:
        """Apply the events (a dispatch, an owner's wish, a what-if) whose minute has come by this 5-minute part."""
        t_end = self._sub_ts(ts, j + 1)
        due = [e for e in self.events if e[0] < t_end]
        if not due:
            return False
        self.events = [e for e in self.events if e[0] >= t_end]
        for _, fn in sorted(due, key=lambda e: e[0]):
            fn(self)
        return True

    def _simulate_rolling(self, i: int, row: dict[str, Any]) -> None:
        pv, base, ts = row["pv_kw"], row["base_kw"], row["ts_local"]
        grid_cf = self._twin(row)
        h, lim = self.house, self.ctl.limits
        if self.overrides.get("comfort"):
            lim = rt_control.Limits(**{**lim.as_dict(), **self.overrides["comfort"]})
        cool0, cop0, tin0, occ0 = h.at(ts) if h else (0.0, 3.0, None, 1)
        dt = 0.25 / SUB
        acc = {"b": 0.0, "e": 0.0, "du": 0.0, "grid": 0.0, "cur": 0.0, "dim": 0.0, "v2h": 0.0}
        decided, held_any, mode = None, False, "self_consumption"
        stop = bool(self.overrides.get("stop"))
        for j in range(SUB):
            changed = self._due_events(ts, j)
            self._car_now()
            active = (not stop) and self.ctl.active(i)
            if active and (decided is None or changed):
                decided = self.ctl.decide(i, first_dt=(SUB - j) * dt)
                if not decided.get("ok"):
                    decided = None
            ev_dem = min(self.ev_max, self.ev_left / dt) if (self.has_ev and self.ev_left > 0.01 and self.ev_deadline) else 0.0
            # what the plan says for this part (or the home's default job when it owes nothing)
            if active and decided:
                b, u = decided["b"], decided["u"]
                e = min(decided["e"], ev_dem) if self.ctl.ev_controlled(ts) else ev_dem
                mode = "rolling"
            else:
                e = ev_dem
                b = self._default(pv, base, e, self.soc, ts) if dt == 0.25 else None
                if b is None:
                    can_dis, can_ch = self._caps(self.soc, dt)
                    net = base + e - pv
                    if BACKUP and net > 0:
                        b = self._backup_refill(ts, self.soc, can_ch)
                    else:
                        b = min(net, can_dis) if net > 0 else -min(-net, can_ch)
                u = h.thermostat(self.x, ts, dt) if h else 0.0
            # the guard, every 5 minutes
            if ev_dem > 0:
                mins = self.replay.minutes_between(self._sub_ts(ts, j), self.ev_deadline)
                need_min = self.ev_left / max(0.1, self.ev_max) * 60
                if self.overrides.get("ev_now") or mins - lim.ev_keep_h * 60 <= need_min + 5:
                    e = ev_dem                                       # the owner's car first; never late
            if self.levers is not None and self.levers.get("ev_kw") is not None:
                e = max(e, min(ev_dem, float(self.levers["ev_kw"])))     # a limit, not a pause: the agent's amps
            held_any = held_any or (ev_dem > 0 and e < ev_dem - 0.01)
            can_dis, can_ch = self._caps(self.soc, dt)
            b = max(-can_ch, min(can_dis, b))
            if h:
                lo, hi = h.dq_bounds(ts)
                u = max(lo, min(hi, u))
                a = h.a(dt)
                x_hi = lim.max_rise_c if occ0 else lim.max_rise_away_c
                if h.step(self.x, u, dt) > x_hi + 1e-6:              # too warm: cool at least enough
                    u = min(hi, max(u, (a * self.x - x_hi) * h.ua / (1.0 - a)))
                if h.step(self.x, u, dt) < -lim.precool_c - 1e-6:    # too cold: cool less
                    u = max(lo, min(u, (a * self.x + lim.precool_c) * h.ua / (1.0 - a)))
                self.x = h.step(self.x, u, dt)
                du = u / max(1.0, cop0)
            else:
                du = 0.0
            self.e_part, self.meter_room = e, self.export_limit - (pv + b - base - du - e)
            dim = self._light_dim(ts, decided if (active and decided) else None)   # a building's lighting; 0 at home
            grid, cur = self._clip_export(pv + b - base - du - e + dim)
            self.soc = self._soc_after(self.soc, b, dt)
            self.ev_left = max(0.0, self.ev_left - e * dt)
            for k, v in (("b", b), ("e", e), ("du", du), ("grid", grid), ("cur", cur), ("dim", dim), ("v2h", self.v2h_kw)):
                acc[k] += v / SUB
        act = self.plan.get(ts)
        self.idx = i
        last = (self.ctl.last or {}) if mode == "rolling" else {}
        self.tel = {
            "ts": ts, "idx": i, "pv_kw": round(pv, 3), "load_kw": round(base + acc["du"] - acc["dim"], 3),
            "ev_kw": round(acc["e"], 3), "batt_kw": round(acc["b"], 3), "v2h_kw": round(acc["v2h"], 3), "soc_pct": round(self.soc * 100, 2),
            "grid_kw": round(acc["grid"], 3), "curtailed_kw": round(acc["cur"], 3), "mode": mode,
            "ev_plugged": bool(self.ev_deadline), "ev_need_kwh": round(self.ev_left, 2),
            "ev_deadline": self.ev_deadline, "ev_held": held_any,
            "price_usd_mwh": row["price_usd_mwh"], "override": dict(self.overrides) if self.overrides else None,
            "plan": dict(act) if act else None, "grid_cf_kw": round(grid_cf, 3), "soc_cf_pct": round(self.cf["soc"] * 100, 2),
            "source": "replay", "control": "rolling",
            "owed_kw": round(self.ctl.target_at(ts), 3), "rt_extra_kw": round(self.rt_extra.get(ts, 0.0), 3),
            "cool_kw": round(cool0 + acc["du"], 3), "cool_own_day_kw": round(cool0, 3),
            "t_in_c": round(tin0 + self.x * (h.sign(ts) if h else 1), 2) if tin0 is not None else None, "t_dev_c": round(self.x, 2),
            "occupied": bool(occ0), "short_kwh_plan": round(last.get("short_kwh", 0.0), 3) if last else 0.0,
            "replans": self.ctl.replans,
        }
        self.tel.update(self._extra_tel(ts, acc))
        self.recent[i] = self.tel
        if len(self.recent) > 400:
            for k in sorted(self.recent)[:-384]:
                self.recent.pop(k, None)

    def dev_now(self, ts: str) -> dict[str, float]:
        """kW each device can give at this step on the node's own day: its share of what it draws."""
        j = self.replay.idx_of(ts) if self.app else None
        if j is None:
            return {}
        out = {}
        held = None if self.levers is None else self.levers.get("devices")     # the agent's levers: only what its calls opened
        for k, spec in self.app.items():
            if k not in self.dev_off and k not in self.kept_out and (held is None or k in held):
                x = self._dev_kw(k, spec, j)
                if held is not None and spec.get("kind") == "source":           # a car gives what its call opened. The plan reads this too, and must not count a closed car
                    x = min(x, float(self.levers.get("v2h_kw", 0.0)))
                if x > 1e-6:
                    out[k] = x
        return out

    def _dev_kw(self, k: str, spec: dict[str, Any], j: int) -> float:
        x = float(self.dev_arr[k][self.dev_row, j]) * float(spec.get("share", 1.0))
        return min(x, self.ev_max) if self.car_rules and spec["kind"] == "source" else x     # the car circuit carries no more back

    def dev_all(self, ts: str) -> dict[str, float]:
        """dev_now without the switches: what each device would draw at this step (what the node agent reads)."""
        j = self.replay.idx_of(ts) if self.app else None
        if j is None:
            return {}
        return {k: self._dev_kw(k, s, j) for k, s in self.app.items()}

    def app_now(self, ts: str) -> float:
        return sum(self.dev_now(ts).values())

    def _v2h_give(self, x: float, ts: str, dt: float) -> float:
        """A plugged-in car gives from its battery; what it gives is charged back before it leaves, and it never
        gives more than the time left can put back (the owner's car first)."""
        spec = self.app["v2h"]
        if not (self.has_ev and self.ev_deadline) or ts >= self.ev_deadline or self.overrides.get("ev_now"):
            return 0.0                                              # the owner's now stops it at once
        if self.car_rules and self.e_part > 0.01:
            return 0.0                                              # one charger: it can't draw and give at once
        if self.v2h_day != ts[:10]:
            self.v2h_day, self.v2h_used = ts[:10], 0.0
        eff = float(spec.get("eff", 0.9))
        out = 1.0 / eff if self.car_rules else 1.0                  # kWh leaving the car for each kWh reaching the home
        give = max(0.0, min(x, (float(spec.get("event_kwh", 15.0)) - self.v2h_used) / out / dt, self.v2h_room(ts) * eff / dt))
        lim = (self.levers or {}).get("v2h_kwh")                    # the call's own max_kwh
        if lim is not None:
            give = min(give, max(0.0, float(lim)) / out / dt)
            self.levers["v2h_kwh"] = max(0.0, float(lim) - give * dt * out)
        self.v2h_owed += give * dt / eff
        if not self.ev_wait:
            self.ev_left += give * dt / eff                         # a timer car takes it on when its session starts
        self.v2h_used += give * dt * out
        return give

    def v2h_room(self, ts: str) -> float:
        """kWh the time left can still put back, after what the car must get anyway. A car on a timer charges from its timer."""
        if not self.ev_deadline:
            return 0.0
        eff = float(self.app["v2h"].get("eff", 0.9)) if "v2h" in self.app else 0.9
        w = self.ev_wait
        keep = self.ctl.limits.ev_keep_h if self.ctl else 0.0
        hours = max(0.0, self.replay.minutes_between(max(ts, w["arrival"]) if w else ts, self.ev_deadline) / 60.0 - keep)
        need = self.ev_left + (w["kwh"] + self.v2h_owed if w else 0.0)
        return max(0.0, self.ev_max * eff * hours - need)

    def _due(self, ts: str) -> bool:
        """A machine's done-by is the owner's: when the time left is what it and those before it still need, the stagger gives way."""
        need = 0.0
        for k in sorted(DONE_BY, key=lambda d: DONE_BY[d]):
            left = self.app_q.get(k, 0.0)
            if left > 1e-6:
                need += left / max(1e-6, float(self.app[k].get("kw", 1.0))) * 60
                if self.replay.minutes_between(ts, f"{ts[:11]}{DONE_BY[k]}") <= need + 15:
                    return True
        return False

    def _light_dim(self, ts: str, decided: dict[str, Any] | None) -> float:
        """A node's other smart systems (benchmark, FG_DEVICES). While something is owed they wait, turn down or give;
        afterwards what waited catches up one device at a time at its own power, staggered per node. Positive = less
        load now; negative = the catch-up."""
        self.v2h_kw = 0.0
        if not self.app:
            return 0.0
        dt = 0.25 / SUB
        stop = bool(self.overrides.get("stop"))
        if not stop and self.ctl.app_holding(ts):
            kw = 0.0
            for k, x in sorted(self.dev_now(ts).items(), key=lambda d: self.app[d[0]]["kind"] == "source"):   # the car last: it takes what the meter has left
                spec = self.app[k]
                if spec["kind"] == "source":
                    if self.levers is not None:
                        x = min(x, float(self.levers.get("v2h_kw", 0.0)))
                    if self.car_rules:
                        x = min(x, max(0.0, self.meter_room - kw))  # past the export limit it would leave the car and be thrown away
                    x = self._v2h_give(x, ts, dt)
                    self.v2h_kw += x
                elif spec["kind"] == "shift":
                    self.app_q[k] = self.app_q.get(k, 0.0) + x * dt
                self.app_log[k] = self.app_log.get(k, 0.0) + x * dt
                kw += x
            self.app_hold_end = ts
            return kw
        if not any(v > 1e-6 for v in self.app_q.values()):
            return 0.0
        if self.app_hold_end and not stop and self.replay.minutes_between(self.app_hold_end, ts) < self.ctl.app_delay_min() and not self._due(ts):
            return 0.0
        for k in sorted(self.app_q, key=lambda d: (d not in DONE_BY, self.app[d].get("order", 9))):
            left = self.app_q.get(k, 0.0)
            if left > 1e-6:
                kw = min(float(self.app[k].get("kw", 1.0)), left / dt)
                self.app_q[k] = max(0.0, left - kw * dt)
                return -kw
        return 0.0

    def _extra_tel(self, ts: str, acc: dict[str, float]) -> dict[str, Any]:
        """More telemetry for a node with more levers (a building's lighting); nothing for a home."""
        return {}

    def rerun_step(self) -> dict[str, Any]:
        """Run the current step again from its start (an event arrived during it: the parts already played come out
        the same, the event takes over from its 5-minute part on)."""
        if self.idx is None or self.prev is None:
            return self.tel
        self._rewind()
        self.x = self.prev_x
        self.cf = dict(self.cf_before)
        self._simulate(self.idx, self.replay.row(self.idx))
        return self.tel

    def rt_at(self, minute: str, fn) -> None:
        """Schedule a change at a dataset minute (offline: the surprise schedule; live: the minute it arrived)."""
        self.events.append((minute, fn))

    def rt_capacity(self, call_from: str, call_to: str) -> dict[str, Any]:
        """How many kW this home can add from call_from to call_to, on top of what it owes (a call's question)."""
        if self.mode != "rolling" or self.ctl is None:
            return {"kw": 0.0, "why": "fixed plan: this home cannot change its plan during the day"}
        i = self.idx if self.idx is not None else self.replay.current()
        return self.ctl.capacity(i, call_from, call_to)

    def rt_dispatch(self, extra: dict[str, float]) -> None:
        """kW owed on top of the booking, per 15-minute step (the coordinator's dispatch for a call or a gap)."""
        for ts, kw in extra.items():
            self.rt_extra[ts] = round(self.rt_extra.get(ts, 0.0) + float(kw), 4)

    def rt_reset(self) -> None:
        """A new run: no dispatch on top, no owner's wish, the house at its own temperature."""
        self.rt_extra.clear()
        self.overrides.clear()
        self.events.clear()
        self.x = self.prev_x = 0.0
        if self.ctl is not None:
            self.ctl.last = None

    def rt_override(self, kind: str, value: Any = True) -> None:
        """The owner: ev_now (charge the car at full power, now), comfort ({max_rise_c: ...}), stop (leave the event),
        depart_at (the car leaves earlier: "YYYY-MM-DD HH:MM"), clear (back to the plan)."""
        if kind == "clear":
            self.overrides.clear()
        elif kind == "depart_at":
            if self.ev_deadline and value < self.ev_deadline:
                self.ev_deadline = value
        elif value in (None, False):
            self.overrides.pop(kind, None)
        else:
            self.overrides[kind] = value
            if kind == "alarm":                                   # life safety, whoever runs the node: the plan stands down until a person clears it
                self.overrides["stop"] = True

    def advance(self) -> dict[str, Any]:
        cur = self.replay.current()
        if self.idx is None or cur < self.idx or cur - self.idx > 96 * 2:
            self.idx = cur - 1                                    # (re)start: no catch-up across days
            self.cf = {"soc": self.soc, "ev_left": self.ev_left, "ev_deadline": self.ev_deadline}
        while self.idx < cur:
            self._step(self.idx + 1)
        return self.tel

    def run_default(self, i_from: int, i_to: int, soc0: float | None = None) -> list[tuple[str, float]]:
        """Meter readings (ts, grid_kw) of this home in default mode over [i_from, i_to): what the utility's
        meter history would hold. Used for baselines; does not touch the live state."""
        soc = self.soc0 if soc0 is None else soc0
        ev_left, ev_deadline, out = 0.0, None, []
        for i in range(i_from, i_to):
            row = self.replay.row(i)
            ts = row["ts_local"]
            sess = self.replay.session_starting(i)
            if sess:
                ev_left, ev_deadline = sess["kwh"], sess["deadline"]
            if ev_deadline and ts >= ev_deadline:
                ev_left, ev_deadline = 0.0, None
            ev = self._ev_demand(ev_left, ev_deadline)
            b = self._default(row["pv_kw"], row["base_kw"], ev, soc, row["ts_local"])
            soc = self._soc_after(soc, b)
            ev_left = max(0.0, ev_left - ev * 0.25)
            out.append((ts, self._clip_export(row["pv_kw"] + b - row["base_kw"] - ev)[0]))
        return out

    # --- v2 contract plan -------------------------------------------------------------
    def set_battery_fault(self, on: bool) -> dict[str, Any]:
        """What-if accident: the battery's inverter trips (0 kW either way) until it is cleared. The home's offers and
        delivery follow from its limits, as they would with a real fault."""
        if on and self.P > 0:
            self._P0, self.P = self.P, 0.0
        elif not on and getattr(self, "_P0", None):
            self.P, self._P0 = self._P0, None
        return {"battery_fault": self.P == 0 and self.E > 0, "battery_kw": self.P}

    def set_plan(self, actions: dict[str, dict[str, Any]], ev_hold: tuple[str, str] | None = None) -> None:
        self.plan.update(actions)
        if ev_hold:
            self.ev_holds = (self.ev_holds + [tuple(ev_hold)])[-6:]

    def clear_plan(self, contract_id: str) -> None:
        self.plan = {ts: a for ts, a in self.plan.items() if a.get("contract_id") != contract_id}

    # --- adapter interface ------------------------------------------------------------
    def telemetry(self) -> dict[str, Any]:
        tel = dict(self.advance())
        if self.fault and self.fault["until_idx"] >= tel["idx"]:
            if self.frozen_reading is None:
                self.frozen_reading = {"soc_pct": tel["soc_pct"]}
            tel.update({"soc_pct": self.frozen_reading["soc_pct"], "batt_kw": 0.0,
                        "fault_injected": self.fault["kind"]})
        else:
            self.fault, self.frozen_reading = None, None
        return tel

    def limits(self) -> dict[str, Any]:
        return {"battery_kwh": self.E, "battery_kw": self.P, "reserve_pct": self.R * 100,
                "soc_max_pct": self.SM * 100, "ev_max_kw": self.ev_max if self.has_ev else 0.0,
                "export_limit_kw": self.export_limit, "eta": self.eta, "control": self.cfg.get("control", "auto")}

    def apply(self, delta_kw: float, minutes: float, dispatch_id: str) -> dict[str, Any]:
        self.advance()
        # Baseline for this step = what the home would do with no command (a new dispatch replaces the old one)
        self._rewind()
        self.override = None
        self.cf = dict(self.cf_before)                     # re-running a step must not advance the twin twice
        self._simulate(self.idx, self.replay.row(self.idx))
        base = dict(self.tel)
        steps = max(1, int(round(minutes / 15.0)))
        b0, ev0 = base["batt_kw"], base["ev_kw"]
        can_dis, can_ch = self._caps(self.prev[0])        # what the battery can really do this step (SOC-limited)
        ev_cap = None
        if delta_kw >= 0:                                  # more export / less import
            bt = max(b0, min(can_dis, b0 + delta_kw))
            rest = delta_kw - (bt - b0)
            if rest > 1e-6 and ev0 > 0:                     # battery maxed: throttle the EV for the rest
                ev_cap = max(0.0, ev0 - rest)
        else:                                              # more import: charge harder
            bt = min(b0, max(-can_ch, b0 + delta_kw))
        self.override = {"battery_kw": round(bt, 3), "until_idx": self.idx + steps - 1,
                         "dispatch_id": dispatch_id, "ev_kw_max": ev_cap}
        self._rewind()                                     # re-run the step (read-back)
        self.cf = dict(self.cf_before)
        self._simulate(self.idx, self.replay.row(self.idx))
        after = dict(self.tel)
        keys = ("ts", "batt_kw", "soc_pct", "grid_kw", "ev_kw", "mode")
        return {"applied_kw": round(after["grid_kw"] - base["grid_kw"], 3),
                "battery_target_kw": self.override["battery_kw"],
                "ev_action": None if ev_cap is None else f"limit EV to {ev_cap:.1f} kW",
                "readback": {k: after[k] for k in keys}, "before": {k: base[k] for k in keys}}

    def release(self) -> None:
        self.override = None

    def inject_fault(self, kind: str, steps: int) -> dict[str, Any]:
        self.advance()
        self.fault = {"kind": kind, "until_idx": self.idx + steps}
        self.frozen_reading = None
        return self.fault

    def snapshot(self) -> None:
        (state_dir() / "sim.json").write_text(json.dumps({"soc": self.soc, "idx": self.idx}))
