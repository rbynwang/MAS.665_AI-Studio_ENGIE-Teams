"""What a home's agent knows at 10:00 the day before, for each hour of tomorrow, with numpy only.

The same features stage4/code/homes_features.py, homes_store.py and homes_solar.py built for training, for one home and one day.
tau is the hour of 10:00 local on the day before. Every series goes through cut() first, so nothing from tau on can be read:
no_leak in tests/forecast_checks.py scrambles those hours and asks for equal features.

Hours are real hours: hour t is replay rows 4t to 4t + 3, so a lag of 24 is the agents' i - 96 across a clock change as well.
"""
from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np

GROUPS = ((0, 7), (7, 15), (15, 21), (21, 24))        # night, day, evening, late: one cooling and one heating slope each
GROUP_OF = np.zeros(24, dtype=int)
for _g, (_a, _b) in enumerate(GROUPS):
    GROUP_OF[_a:_b] = _g
SUN_LIMIT = 20.0                                      # W/m2 of clear sky: below this the solar model says nothing


class Clock:
    """The replay's year: real hours for the home models, and the local 96-slot days the car models and the plan read."""

    def __init__(self, stamps: list[str], holidays: list[str], tz_offset_h: float):
        hs = stamps[::4]
        self.T = len(hs)
        first: dict[str, int] = {}
        for k, s in enumerate(stamps):
            first.setdefault(s, k)
        self.days = sorted({s[:10] for s in stamps})
        self.ND = len(self.days)
        self.day_of = {d: i for i, d in enumerate(self.days)}
        self.t = np.arange(self.T)
        self.day_i = np.array([self.day_of[s[:10]] for s in hs])
        self.hour = np.array([int(s[11:13]) for s in hs])
        ds = [date.fromisoformat(d) for d in self.days]
        hol = set(holidays)
        self.day_dow = np.array([d.weekday() for d in ds])
        self.day_holiday = np.array([d in hol for d in self.days])
        self.day_off = (self.day_dow >= 5) | self.day_holiday
        self.dow = self.day_dow[self.day_i]
        self.weekend = (self.dow >= 5).astype(int)
        self.holiday = self.day_holiday[self.day_i].astype(int)
        self.off = (self.weekend | self.holiday).astype(int)
        self.month = np.array([d.month for d in ds])[self.day_i]
        tau_day = np.full(self.ND, -1)
        for i in range(1, self.ND):
            k = first.get(f"{self.days[i - 1]} 10:00")
            tau_day[i] = -1 if k is None else k // 4
        self.tau = tau_day[self.day_i]
        self.k0 = np.where(self.t - 24 < self.tau, 1, 2)        # first whole-day lag already seen at 10:00
        self.rows_of = [np.flatnonzero(self.day_i == i) for i in range(self.ND)]
        self.utc = (np.datetime64(stamps[0].replace(" ", "T")) - np.timedelta64(int(tz_offset_h), "h")) + self.t.astype("timedelta64[h]")
        self.grid_day = np.array([self.day_of[s[:10]] for s in stamps])
        self.grid_slot = np.array([int(s[11:13]) * 4 + int(s[14:16]) // 15 for s in stamps])

    def grid(self, x: np.ndarray, fill: float = 0.0) -> np.ndarray:
        """A year of 15-minute values as days x 96 local slots. The hour repeated in the autumn keeps its second copy."""
        g = np.full((self.ND, 96), fill, dtype="float64")
        g[self.grid_day, self.grid_slot] = x
        return g


def cop(temp: np.ndarray, cooling: bool) -> np.ndarray:
    """The heat a kW of electricity moves at this outdoor temperature, as in rt_home, which needs scipy and so stays out of the forecasts."""
    t = np.asarray(temp, dtype=float)
    return np.clip(4.5 - .1 * (t - 25), 2, 4.5) if cooling else np.clip(1.5 + .08 * (t + 10), 1.75, 4.0)


def cut(x: np.ndarray, tau: int) -> np.ndarray:
    """The series as known at hour tau: nothing from tau on."""
    y = np.array(x, dtype=float)
    y[max(tau, 0):] = np.nan
    return y


def at(x: np.ndarray, i: np.ndarray) -> np.ndarray:
    out = np.full(i.shape, np.nan)
    ok = i >= 0
    out[ok] = x[i[ok]]
    return out


def nan_mean(*xs: np.ndarray) -> np.ndarray:
    """The mean of what is known, without numpy's warning when nothing is."""
    a = np.stack(xs)
    n = np.isfinite(a).sum(axis=0)
    return np.where(n > 0, np.nansum(a, axis=0) / np.maximum(n, 1), np.nan)


def same_hours(x: np.ndarray, t: np.ndarray, k0: np.ndarray, m: int) -> np.ndarray:
    """The same hour on the m latest whole days already seen: rows x m."""
    return at(x, t[:, None] - 24 * (k0[:, None] + np.arange(m)))


def same_kind(x: np.ndarray, ck: Clock, t: np.ndarray) -> np.ndarray:
    """The same hour on the last 4 days of the same kind (working day, or day off), already seen. 3 will do."""
    s, got = np.zeros(len(t)), np.zeros(len(t), dtype=int)
    tau = ck.tau[t]
    for q in range(1, 40):
        src = t - 24 * q
        m = (src >= 0) & (src < tau) & (tau >= 0)
        m[m] = ck.off[src[m]] == ck.off[t[m]]
        m &= got < 4
        got += m
        s[m] += x[src[m]]
    return np.where(got >= 3, s / np.maximum(got, 1), np.nan)


def back_mean(x: np.ndarray, tau: int, hours: int, min_hours: int) -> float:
    """Mean of the hours before tau."""
    a, b = max(tau - hours, 0), max(tau, 0)
    return float(x[a:b].mean()) if b - a >= min_hours and tau >= 0 else float("nan")


def rule_line(y: np.ndarray, xm: np.ndarray, xf: np.ndarray, t: np.ndarray, k0: np.ndarray, days: int = 21) -> np.ndarray:
    """The simple rule, home_v2._load_fit: for this hour, load = a + b x temperature over the 21 latest days seen, read at the forecast."""
    Y, X = same_hours(y, t, k0, days), same_hours(xm, t, k0, days)
    sy, sx, sxx0, sxy0 = Y.sum(1), X.sum(1), (X * X).sum(1), (X * Y).sum(1)
    sxx, sxy = sxx0 - sx ** 2 / days, sxy0 - sx * sy / days
    with np.errstate(invalid="ignore", divide="ignore"):
        b = np.where(sxx > 1e-6, sxy / sxx, 0.0)
    return np.maximum(0.0, (sy - b * sx) / days + b * xf)


def degrees(temp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cooling and heating degrees: half the hour's temperature, half the mean of the last 24 h, because a house answers to both."""
    c = np.concatenate([[0.0], np.cumsum(np.nan_to_num(temp))])
    i = np.arange(len(temp))
    a = np.maximum(i - 23, 0)
    te = .5 * temp + .5 * (c[i + 1] - c[a]) / (i + 1 - a)
    return np.maximum(0.0, te - 18.0), np.maximum(0.0, 15.0 - te)


def degree_slopes(y: np.ndarray, cm: np.ndarray, wm: np.ndarray, tau: int) -> tuple[np.ndarray, np.ndarray]:
    """kW per degree, per part of the day, from the days over at tau, with one level per hour of day."""
    K, L = np.zeros(len(GROUPS)), np.zeros(len(GROUPS))
    m = max(tau, 0) // 24
    if m < 14:
        return K, L
    Y, C, W = (a[:24 * m].reshape(m, 24) for a in (y, cm, wm))
    sy, sc, sw = Y.sum(0), C.sum(0), W.sum(0)
    scc, sww, scw, syc, syw = (C * C).sum(0), (W * W).sum(0), (C * W).sum(0), (Y * C).sum(0), (Y * W).sum(0)
    for g, (a, b) in enumerate(GROUPS):
        s = slice(a, b)
        A11 = (scc[s] - sc[s] ** 2 / m).sum() + 5.0            # a light ridge: no cooling seen yet means no slope
        A22 = (sww[s] - sw[s] ** 2 / m).sum() + 5.0
        A12 = (scw[s] - sc[s] * sw[s] / m).sum()
        b1 = (syc[s] - sc[s] * sy[s] / m).sum()
        b2 = (syw[s] - sw[s] * sy[s] / m).sum()
        det = A11 * A22 - A12 ** 2
        K[g], L[g] = max(0.0, (A22 * b1 - A12 * b2) / det), max(0.0, (A11 * b2 - A12 * b1) / det)
    return K, L


def history(y: np.ndarray, xm: np.ndarray, xf: np.ndarray, ck: Clock, t: np.ndarray) -> dict[str, np.ndarray]:
    """The series' own past as seen at tau, for these hours of one day, in its own units. y, xm: series and measured temperature.
    xf: the forecast as issued, which holds no more than the target day."""
    tau = int(ck.tau[t[0]])
    y, xm, xf = cut(y, tau), cut(xm, tau), cut(xf, int(t[-1]) + 1)
    k0 = ck.k0[t]
    lag = {k: at(y, t - k) for k in (24, 48, 72, 96, 168)}
    Y14 = same_hours(y, t, k0, 14)
    m14 = Y14.mean(1)
    naive3 = nan_mean(lag[48], lag[72], lag[96])
    h = {"scale": np.full(len(t), back_mean(y, tau, 336, 269)), "last24": np.full(len(t), back_mean(y, tau, 24, 19)), "last6": np.full(len(t), back_mean(y, tau, 6, 5)),
         "lag24": np.where(k0 == 1, lag[24], np.nan), "lag48": lag[48], "lag72": lag[72], "lag96": lag[96], "lag168": lag[168], "naive3": naive3,
         "seen3": same_hours(y, t, k0, 3).mean(1), "m7": same_hours(y, t, k0, 7).mean(1), "m14": m14, "sd14": np.sqrt(np.maximum((Y14 ** 2).mean(1) - m14 ** 2, 0)), "kind4": same_kind(y, ck, t),
         "rule": rule_line(y, xm, xf[t], t, k0)}
    cm, wm = degrees(xm)
    cf, wf = degrees(xf)
    K, L = degree_slopes(y, cm, wm, tau)
    gt = GROUP_OF[t % 24]
    Kc, Kw = K[gt], L[gt]
    m14c, m14w = same_hours(cm, t, k0, 14).mean(1), same_hours(wm, t, k0, 14).mean(1)
    k4c, k4w = same_kind(cm, ck, t), same_kind(wm, ck, t)
    lvl14 = m14 - Kc * m14c - Kw * m14w
    lvlk = h["kind4"] - Kc * k4c - Kw * k4w
    h.update({"phys14": np.maximum(0.0, lvl14 + Kc * cf[t] + Kw * wf[t]), "physk": np.maximum(0.0, lvlk + Kc * cf[t] + Kw * wf[t]), "Kc": Kc, "Kw": Kw,
              "dcool": Kc * (cf[t] - m14c), "dheat": Kw * (wf[t] - m14w), "lvl14": lvl14})
    return h


def table(names: list[str], h: dict[str, np.ndarray], scale: np.ndarray, ck: Clock, t: np.ndarray, static: dict[str, float], plan: dict[str, np.ndarray], ratio: bool = True) -> np.ndarray:
    """The rows a model reads, in its own column order."""
    X = np.empty((len(t), len(names)))
    sc = scale if ratio else 1.0
    for j, c in enumerate(names):
        if c in ("hour", "dow", "weekend", "holiday"):
            X[:, j] = getattr(ck, c)[t]
        elif c == "log_scale":
            X[:, j] = np.log(np.maximum(scale, 1e-6))
        elif c in static:
            X[:, j] = static[c]
        elif c in plan:
            X[:, j] = plan[c]
        elif c in h:
            X[:, j] = h[c] / sc
        else:
            raise KeyError(c)
    return X


# --------------------------------------------------------------------------------------------------- solar
def cos_zenith(ck: Clock, lat: float, lon: float) -> np.ndarray:
    """cos(solar zenith) at the middle of each hour, the formulas of forecaster.cos_zenith."""
    u = ck.utc + np.timedelta64(30, "m")
    d0 = u.astype("datetime64[D]")
    doy = (d0 - u.astype("datetime64[Y]").astype("datetime64[D]")).astype(int) + 1
    hour = (u - d0) / np.timedelta64(1, "h")
    g = 2 * np.pi / 365.0 * (doy - 1 + (hour - 12) / 24)
    decl = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g) - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    ha = np.radians((hour * 60 + eqt + 4 * lon) / 4 - 180)
    la = np.radians(lat)
    return np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha)


def clear_sky(cz: np.ndarray) -> np.ndarray:
    """Haurwitz clear-sky sunlight, W/m2."""
    c = np.clip(cz, 0, None)
    return np.where(c > 0.01, 1098 * c * np.exp(-0.057 / np.maximum(c, 0.01)), 0.0)


class Solar:
    """solar_per_kw_homes_2026 for one array. Its trees learned on one south roof, so they run on the array's own clock: the home's
    history moves by shift_h, the model predicts, the answer moves back. Then the 14-day per-hour correction, as the old agents did."""

    def __init__(self, ck: Clock, cols: dict[str, np.ndarray], lat: float, lon: float, shift_h: float, scale_kw: float):
        self.ck, self.c = ck, cols
        self.shift, self.sk = float(shift_h), float(scale_kw)
        self.x = np.arange(ck.T, dtype=float)
        self.cz = cos_zenith(ck, lat, lon)
        self.cs = clear_sky(self.cz)
        self.cs_array = np.interp(self.x - self.shift, self.x, self.cs)

    def block(self, model: Any, t: np.ndarray, pv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """10/50/90 % in kW over hours t, on the home's clock, before the correction. pv: the home's meter, cut at tau."""
        c, cz, cs = self.c, self.cz[t], self.cs[t]
        pa = np.interp(self.x + self.shift, self.x, pv) / self.sk
        lag = {k: at(pa, t - k) for k in (48, 72, 96)}
        n3 = nan_mean(lag[48], lag[72], lag[96])
        keep = (cz > 0.05) & (cs > SUN_LIMIT)
        k = cs[keep] / 1000
        f = {"hour": self.ck.hour[t][keep], "cosz": cz[keep], "sw": c["sw"][t][keep], "direct": c["direct"][t][keep], "diffuse": c["diffuse"][t][keep], "cloud": c["cloud"][t][keep], "temp": c["temp"][t][keep],
             "csi_fc": c["sw"][t][keep] / np.maximum(cs[keep], 1), "lag48_csi": lag[48][keep] / k, "naive3_csi": n3[keep] / k}
        out = [np.zeros(len(t)) for _ in range(3)]
        if keep.any():
            X = np.column_stack([f[n] for n in model.features])
            for a, v in zip(out, model.predict(X)):
                a[keep] = np.maximum(0.0, v * k * self.sk)
        xb = t.astype(float)
        return tuple(np.interp(xb - self.shift, xb, a) for a in out)

    def day(self, model: Any, D: int, pv_meter: np.ndarray) -> dict[str, np.ndarray]:
        """Day D as known at 10:00 the day before: kW at 10/50/90 %, with the fill for hours under the model's sun limit."""
        ck = self.ck
        rows = ck.rows_of[D]
        tau = int(ck.tau[rows[0]])
        pv = cut(pv_meter, tau)
        d0 = max(D - 15, 0)
        first = int(ck.rows_of[d0][0])
        t = np.arange(max(first - 3, 0), min(int(rows[-1]) + 4, ck.T))
        lo, mid, hi = self.block(model, t, pv)
        num, den = np.zeros(24), np.zeros(24)
        if D >= 3:
            for d in range(d0, D - 1):                         # finished days D-15 .. D-2
                r = ck.rows_of[d]
                j = r - t[0]
                ok = mid[j] > 0
                np.add.at(num, ck.hour[r][ok], pv[r][ok])
                np.add.at(den, ck.hour[r][ok], mid[j][ok])
        fac = np.where(den > 0.05 * self.sk * 14 / 7, np.clip(num / np.maximum(den, 1e-9), 0.5, 1.5), 1.0)
        j = rows - t[0]
        f = fac[ck.hour[rows]]
        p10, p50, p90 = lo[j] * f, mid[j] * f, hi[j] * f
        k0 = ck.k0[rows]
        rule = same_hours(pv, rows, k0, 3).mean(1)              # the last 3 days already seen
        low = (self.cs_array[rows] <= SUN_LIMIT) & np.isfinite(rule)
        return {"p10": np.where(low, .5 * rule, p10), "p50": np.where(low, rule, p50), "p90": np.where(low, 1.5 * rule, p90), "factor": f, "low_sun": low}
