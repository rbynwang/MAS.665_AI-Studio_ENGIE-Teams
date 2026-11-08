"""The home's real-time controller: one small linear program over the battery, the car's charging and the air
conditioning, solved again whenever something changes and at least every 15 minutes (the 5-minute loop).

It replaces the fixed order of the day-ahead design (car first, then battery, decided once): the plan is a
starting point, and every re-plan uses the latest meter, weather and owner's wishes. What it minimises, in dollars:

  shortfall     kWh the home owes (the booking plus any real-time dispatch) and would not deliver     $10 / kWh
  new peaks     kWh imported above the no-event day outside the event (pre-charging from the grid
                before it, the catch-up after it): $0.4 plus twice that hour's day-ahead price per kWh, so
                the plan charges when power is cheap and keeps the battery for the expensive hours
  discomfort    C x hours the house is warmer than its own day while someone is home                   $0.6
  over limit    C x hours above the owner's limit (soft, so a forecast miss never makes it unsolvable)   $50
  energy        every kWh imported, a little ($0.05), so nothing runs without a reason; stored battery energy at
                the end of the horizon keeps a value ($0.10 / kWh), so it is not drained for nothing

The hard limits, which no answer can break: the battery between its reserve and full at its power rating; the car
charged by its departure, and never held in its last `ev_keep_h` hours; the house no cooler than `precool_c` below
its own day; the export limit. The guard in the home (adapters/simulated.py) checks them again on every 5 minutes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

W_SHORT, W_REBOUND, W_DISCOMFORT, W_VIOL, W_ENERGY, V_STORED, W_CYCLE, W_TERM = 10.0, 0.4, 2.0, 50.0, 0.05, 0.05, 0.01, 2.0
# (tuned on 2 Jul 2026, 1 Nov 2026: with discomfort at 0.6 and stored energy at 0.10 the plan let houses warm up to save cooling
#  energy and put it in the battery, then cooled them all at once at 23:00; now the air conditioning eases off only
#  where it helps what is owed, and the battery is never charged from the grid for its own sake)
RECOVER_C_PER_H = 0.3              # after the last step owed: the house cools back at most this fast (a spread rebound)
W_HARD = 500.0
W_XTRA = 0.06                      # delivering more than owed: more than the battery saves by discharging (0.04), so a
                                   # battery keeps what the window does not need for the house's own evening


@dataclass
class Limits:
    """The owner's limits (defaults: the plan's emergency defaults). FlexGrid's agent sets them with the owner."""
    max_rise_c: float = 2.0          # warmer than the house's own day, while someone is home
    max_rise_away_c: float = 4.0     # ... while nobody is home
    precool_c: float = 1.5           # cooler than its own day, at most
    ev_keep_h: float = 2.0           # the car is never held in its last hours before departure
    call_battery_share: float = 0.5  # a short-notice call may use at most this share of the usable battery

    def as_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


@dataclass
class Problem:
    """One home over the next steps. Lists are per step; step 0 starts now (dt[0] may be the rest of 15 minutes)."""
    ts: list[str]
    dt: list[float]
    pv: list[float]
    load: list[float]                # the house without the car, cooling as in its own day (base_kw forecast)
    cool: list[float]                # its own day's cooling electricity inside `load`
    cop: list[float]
    occ: list[int]
    ref: list[float]                 # no-event net import forecast (kW, + = import)
    target: list[float]              # kW below ref the home owes this step (0 = nothing owed)
    watch: list[bool]                # after the event: import above ref is rebound
    price: list[float] | None = None  # day-ahead price, $/kWh: a new peak at an expensive hour costs more
    soc: float = 0.5                 # battery state (0-1)
    E: float = 0.0
    P: float = 0.0
    P_ch: float = 1e9                # benchmark: the battery's charging limit, when below its output
    P_dis: float | None = None       # the node agent's limit on discharging, when its tools set one
    R: float = 0.2
    SM: float = 1.0
    eta: float = 0.95
    export_limit: float = 10.0
    ev_max: float = 0.0
    ev_left: float = 0.0             # kWh still to charge
    ev_until: int | None = None      # steps the car is plugged in for (its departure), None if not plugged in
    ev_hold: list[bool] | None = None   # steps the coordinator asked the car to wait (a staggered restart)
    x0: float = 0.0                  # indoor temperature minus its own day, C
    ua: float = 0.0                  # house constants (0: no air conditioning to move)
    cap: float = 1.0
    qmax: float = 0.0
    limits: Limits = field(default_factory=Limits)
    call: list[bool] | None = None   # capacity(): the steps of a call to answer
    battery_floor: list[float] | None = None   # kWh (DC side) the battery must keep at the end of each step
    short_cap: list[float] | None = None       # capacity(): no step may fall shorter than the plan already does
    comfort_weight: float = 1.0      # discomfort is per C, the energy per kWh: a building (building.py) weighs its
                                     # people's comfort in proportion to its size, or the plan warms it to save energy


def _var_index(K: int) -> dict[str, slice]:
    names = ("d", "c", "e", "u", "x", "S", "short", "over", "dis", "viol", "xtra")
    return {n: slice(i * K, (i + 1) * K) for i, n in enumerate(names)}


def solve(p: Problem, extra_r: bool = False) -> dict[str, Any]:
    """The plan for every step: battery (+ discharging), car, cooling change (kW of heat; + = more cooling), the
    net import, the shortfall and the indoor temperature path. With extra_r, one more variable r is maximised: the
    most kW the home can add, evenly over the call's steps, on top of everything it owes (capacity for a call)."""
    K = len(p.ts)
    V = _var_index(K)
    n = 11 * K + (1 if extra_r else 0)
    ir = 11 * K
    dt = np.array(p.dt, dtype=float)
    cop = np.maximum(1.0, np.array(p.cop, dtype=float))
    has_house = p.ua > 0 and p.qmax > 0
    a = np.array([math.exp(-h * p.ua / p.cap) if has_house else 1.0 for h in dt])
    lim = p.limits
    cost = np.zeros(n)
    # objective
    cost[V["short"]] = W_SHORT * dt
    pr = np.array(p.price if p.price else [0.0] * K, dtype=float)
    cost[V["over"]] = (W_REBOUND + 2.0 * np.clip(pr, 0.0, 2.0)) * dt
    cw = float(p.comfort_weight)
    cost[V["dis"]] = W_DISCOMFORT * cw * dt * np.array(p.occ, dtype=float)
    # someone home: an owner's limit comes before the sale, so the plan falls short and says so. W_HARD is above anything a kWh short can cost
    cost[V["viol"]] = np.where(np.array(p.occ) > 0, W_HARD, W_VIOL) * cw * dt
    cost[V["xtra"]] = W_XTRA * dt
    cost[V["d"]] = (-W_ENERGY + W_CYCLE) * dt
    cost[V["c"]] = (W_ENERGY + W_CYCLE) * dt
    cost[V["e"]] = W_ENERGY * dt
    cost[V["u"]] = W_ENERGY * dt / cop
    cost[V["S"].start + K - 1] = -V_STORED
    if has_house:                                   # warmer than its day at the end costs, home or not (rebound later)
        cost[V["dis"].start + K - 1] += W_TERM * cw * dt[-1]
    if extra_r:
        cost[ir] = -100.0 * sum(h for h, c in zip(dt, p.call or []) if c)
    rows, cols, vals, b_ub = [], [], [], []
    erows, ecols, evals, b_eq = [], [], [], []

    def ub(entries: list[tuple[int, float]], rhs: float) -> None:
        r = len(b_ub)
        for j, v in entries:
            rows.append(r); cols.append(j); vals.append(v)
        b_ub.append(rhs)

    def eq(entries: list[tuple[int, float]], rhs: float) -> None:
        r = len(b_eq)
        for j, v in entries:
            erows.append(r); ecols.append(j); evals.append(v)
        b_eq.append(rhs)

    S0 = p.soc * p.E
    last_owed = max((k for k in range(K) if p.target[k] > 0 or (extra_r and p.call and p.call[k])), default=-1)
    for k in range(K):
        d, c, e, u, x, S = (V[s].start + k for s in ("d", "c", "e", "u", "x", "S"))
        # battery energy (DC side): S_k = S_{k-1} - d dt / eta + c dt eta
        ent = [(S, 1.0), (d, dt[k] / p.eta), (c, -dt[k] * p.eta)]
        if k:
            ent.append((S - 1, -1.0))
        eq(ent, S0 if k == 0 else 0.0)
        # house: x_k = a x_{k-1} - (1 - a) / UA u_k
        if has_house:
            ent = [(x, 1.0), (u, (1.0 - a[k]) / p.ua)]
            if k:
                ent.append((x - 1, -a[k]))
            eq(ent, a[k] * p.x0 if k == 0 else 0.0)
        else:
            eq([(x, 1.0)], 0.0)
        # net import g = load - pv + u / cop + e + c - d
        base = p.load[k] - p.pv[k]
        g = [(u, 1.0 / cop[k]), (e, 1.0), (c, 1.0), (d, -1.0)]
        if p.target[k] > 0 or (extra_r and p.call and p.call[k]):
            ent = g + [(V["short"].start + k, -1.0)]
            if extra_r and p.call and p.call[k]:
                ent.append((ir, 1.0))
            if extra_r:
                ub(ent, p.ref[k] - p.target[k] - base)
            else:                                   # exactly what is owed, any shortfall or extra counted
                eq(ent + [(V["xtra"].start + k, 1.0)], p.ref[k] - p.target[k] - base)
        if p.watch[k]:
            ub(g + [(V["over"].start + k, -1.0)], p.ref[k] - base)
        ub([(j, -v) for j, v in g], max(0.0, base + p.export_limit))  # export limit: g >= -limit (solar beyond
        #                                                                 it is curtailed whatever the home does)
        if has_house:
            free = 0.0 if (p.occ[k] or k == K - 1) else lim.max_rise_away_c   # warmer than its day: comfort
            ub([(x, 1.0), (V["dis"].start + k, -1.0)], free)
            hi = lim.max_rise_c if p.occ[k] else lim.max_rise_away_c
            ub([(x, 1.0), (V["viol"].start + k, -1.0)], hi)
            if k > last_owed:                       # nothing owed from here on: never warmer than the step before,
                prev = [(x - 1, -1.0)] if k else []  # and cool back no faster than RECOVER_C_PER_H
                ub([(x, 1.0)] + prev + [(V["viol"].start + k, -1.0)], 0.0 if k else max(0.0, p.x0))
                if k:
                    ub([(x - 1, 1.0), (x, -1.0)], RECOVER_C_PER_H * dt[k])
                else:
                    ub([(x, -1.0)], RECOVER_C_PER_H * dt[k] - p.x0)
    # the car: charged by departure, never held in its last hours, never more than it needs
    if p.ev_until is not None and p.ev_left > 0.01 and p.ev_max > 0:
        upto = min(K, p.ev_until)
        hours_after = max(0.0, sum(dt[K:]) if p.ev_until <= K else (p.ev_until - K) * 0.25)
        need_now = max(0.0, p.ev_left - p.ev_max * hours_after * 0.98) if p.ev_until > K else p.ev_left
        if upto > 0:
            ub([(V["e"].start + k, -dt[k]) for k in range(upto)], -min(need_now, p.ev_max * sum(dt[:upto]) * 0.999))
            ub([(V["e"].start + k, dt[k]) for k in range(upto)], p.ev_left)
            keep = int(round(lim.ev_keep_h / 0.25))
            cut = p.ev_until - keep
            if 0 < cut <= K:
                need_cut = max(0.0, p.ev_left - p.ev_max * lim.ev_keep_h * 0.98)
                if need_cut > 0:
                    ub([(V["e"].start + k, -dt[k]) for k in range(cut)],
                       -min(need_cut, p.ev_max * sum(dt[:cut]) * 0.999))
    bounds: list[tuple[float | None, float | None]] = [(0.0, 0.0)] * n
    for k in range(K):
        bounds[V["d"].start + k] = (0.0, (p.P if p.P_dis is None else min(p.P, p.P_dis)) if p.E > 0 else 0.0)
        bounds[V["c"].start + k] = (0.0, min(p.P, p.P_ch) if p.E > 0 else 0.0)
        plugged = p.ev_until is not None and k < p.ev_until and p.ev_left > 0.01
        held = bool(p.ev_hold and k < len(p.ev_hold) and p.ev_hold[k])
        bounds[V["e"].start + k] = (0.0, 0.0 if (not plugged or held) else p.ev_max)
        if has_house:
            q = p.cool[k] * cop[k]
            bounds[V["u"].start + k] = (-q, max(0.0, p.qmax - q))
            bounds[V["x"].start + k] = (-lim.precool_c if p.x0 >= -lim.precool_c else p.x0, None)
        else:
            bounds[V["u"].start + k] = (0.0, 0.0)
            bounds[V["x"].start + k] = (None, None)
        lo = p.R * p.E
        if p.battery_floor is not None and k < len(p.battery_floor):
            lo = max(lo, p.battery_floor[k])
        bounds[V["S"].start + k] = (min(lo, S0), p.SM * p.E)
        for s in ("short", "over", "dis", "viol", "xtra"):
            bounds[V[s].start + k] = (0.0, None)
        if p.short_cap is not None and k < len(p.short_cap):
            bounds[V["short"].start + k] = (0.0, max(0.0, p.short_cap[k]) + 1e-6)
    if extra_r:
        bounds[ir] = (0.0, None)
    A_ub = coo_matrix((vals, (rows, cols)), shape=(len(b_ub), n)).tocsr() if b_ub else None
    A_eq = coo_matrix((evals, (erows, ecols)), shape=(len(b_eq), n)).tocsr()
    res = linprog(cost, A_ub=A_ub, b_ub=np.array(b_ub) if b_ub else None, A_eq=A_eq, b_eq=np.array(b_eq),
                  bounds=bounds, method="highs")
    if res.status != 0:
        return {"ok": False, "status": int(res.status), "message": str(res.message)[:200]}
    z = res.x
    get = lambda s: z[V[s]]                                                              # noqa: E731
    d, c, e, u = get("d"), get("c"), get("e"), get("u")
    g = np.array(p.load) - np.array(p.pv) + u / cop + e + c - d
    out = {"ok": True, "b": (d - c).tolist(), "e": e.tolist(), "u": u.tolist(), "cool_kw": (u / cop).tolist(),
           "g": g.tolist(), "x": get("x").tolist(), "S": get("S").tolist(), "short": get("short").tolist(),
           "over": get("over").tolist(),
           "short_kwh": float(np.dot(get("short"), dt)), "cost": float(res.fun)}
    if extra_r:
        out["r"] = float(z[ir])
    return out


def capacity(p: Problem, call: list[bool], plan: dict[str, Any] | None = None) -> float:
    """The most kW this home can add, evenly over the call's steps, on top of everything it owes, without owing
    more than it already does and keeping the owner's call limit on the battery. 0 if nothing."""
    q = Problem(**{**p.__dict__, "call": call})
    q.short_cap = [0.0] * len(q.ts)                         # nothing owed yet: the answer may not fall short at all
    if plan and plan.get("ok"):
        q.short_cap = list(plan["short"])                   # adding r may not make the home owe more elsewhere
        if q.E > 0:
            usable = (q.SM - q.R) * q.E
            q.battery_floor = [max(q.R * q.E, s - q.limits.call_battery_share * usable) for s in plan["S"]]
    res = solve(q, extra_r=True)
    if not res.get("ok"):
        return 0.0
    return max(0.0, res["r"])
