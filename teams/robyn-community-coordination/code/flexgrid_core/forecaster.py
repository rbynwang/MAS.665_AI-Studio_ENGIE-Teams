"""Trained day-ahead forecasts for a home agent (forecast = "trained"): its base load and its solar, each as 10/50/90 %
quantiles, from the models in models/forecast_2025_26 (exported gradient-boosted trees, evaluated with numpy only).

The features are rebuilt here from what the home itself has, exactly as tools/train/forecast_models.py built them for
training: its own meter history (the replay's base_kw and pv_kw, only hours already past when the forecast is made),
the measured temperature, and the weather forecast as it was issued about 48 h ahead (the replay's fc_* columns).
  load  home_base_load.json.gz   target = base load / its last 14 days' mean
  solar solar_per_kw.json.gz     target = PV per kW of panels / clear-sky sunlight (kW/m2); 0 at night
The home's roster entry (fleet_v2.json) gives heating type, EV, the array's size in the solar model's kW
(pv_scale_kw) and where it is. Hourly values are used for each 15-minute step of the hour.
"""
from __future__ import annotations

import json
import math
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Any

import numpy as np

from . import car_features as cf
from . import config
from . import home_features as hf
from .beckn import add_minutes
from .plan_past import PastPlan, planned_run
from .treemodel import TreeModel

LAGS = (48, 72, 96, 168)


@lru_cache(maxsize=24)
def _model(path: str) -> TreeModel:
    return TreeModel.load(path)


@lru_cache(maxsize=1)
def _roster() -> dict[str, dict[str, Any]]:
    try:
        return {h["name"]: h for h in json.loads(config.data_path("fleet_v2.json").read_text())["homes"]}
    except (OSError, ValueError, KeyError):
        return {}


def cos_zenith(utc: datetime, lat: float, lon: float) -> float:
    """cos(solar zenith) at the middle of the hour starting at `utc` (NOAA simplified formulas, as in training)."""
    t = utc + timedelta(minutes=30)
    doy = t.timetuple().tm_yday
    hour = t.hour + t.minute / 60.0
    g = 2 * math.pi / 365.0 * (doy - 1 + (hour - 12) / 24)
    decl = (0.006918 - 0.399912 * math.cos(g) + 0.070257 * math.sin(g) - 0.006758 * math.cos(2 * g)
            + 0.000907 * math.sin(2 * g) - 0.002697 * math.cos(3 * g) + 0.00148 * math.sin(3 * g))
    eqt = 229.18 * (0.000075 + 0.001868 * math.cos(g) - 0.032077 * math.sin(g) - 0.014615 * math.cos(2 * g)
                    - 0.040849 * math.sin(2 * g))
    ha = math.radians((hour * 60 + eqt + 4 * lon) / 4 - 180)
    la = math.radians(lat)
    return math.sin(la) * math.sin(decl) + math.cos(la) * math.cos(decl) * math.cos(ha)


def clear_sky(cz: float) -> float:
    """Haurwitz clear-sky sunlight, W/m2."""
    return 1098 * cz * math.exp(-0.057 / max(cz, 0.01)) if cz > 0.01 else 0.0


class HomeForecaster:
    """forecast(now_idx, ts_list) -> {ts: {"pv": (p10, p50, p90), "load": (p10, p50, p90), "load_mean": kW}}.
    The agent's energy balance uses load_mean (an offer adds hours up; a skewed load's median is low) and the PV
    median; the 10-90 % values travel with the forecast."""

    def __init__(self, name: str, replay, home: dict[str, Any] | None = None, correct_pv: bool = True):
        meta = config.dataset_meta()
        self.replay = replay
        self.home = home or _roster().get(name, {})
        mdir = config.REPO_DIR / meta.get("models", "models/forecast_2025_26")
        self.load_m = _model(str(mdir / "home_base_load.json.gz"))
        self.pv_m = _model(str(mdir / "solar_per_kw.json.gz"))
        self.ev_m = _model(str(mdir / "ev_window.json.gz")) if (mdir / "ev_window.json.gz").exists() else None
        self._evd: tuple[dict, dict] | None = None             # each day's unmanaged EV kWh (15-20 h, whole day)
        comm = meta.get("community", {})
        self.lat = float(self.home.get("lat", comm.get("lat", 42.26)))
        self.lon = float(self.home.get("lon", comm.get("lon", -71.80)))
        self.pv_kw = float(self.home.get("pv_scale_kw", self.home.get("pv_kw_dc", 0.0)) or 0.0)
        self.static = {"is_hp": int(self.home.get("heating_type") == "heat_pump"),
                       "has_ev": int(bool(self.home.get("has_ev", False)))}
        self.tz = int(meta.get("tz_offset_h", -4))
        self.tz_name = meta.get("tz_name")
        self.holidays = {date.fromisoformat(d) for d in meta.get("holidays", [])}
        self._corr: dict[str, dict[int, float]] = {}          # per-hour solar correction, for the current day
        self.correct_pv = correct_pv
        self.method = ("trained: home_base_load (gradient-boosted trees on 100 synthetic homes) and solar_per_kw "
                       "(trained on HouseZero's measured PV, corrected per hour of day by this array's last 14 days), "
                       "10/50/90 %, ECMWF forecasts issued 48 h ahead; EV: ev_window (expected kWh 15-20 h, trained on "
                       "synthetic EV homes) scales the latest session")

    # ------------------------------------------------------------------ the home's own hourly history
    def _row(self, i: int) -> dict[str, Any] | None:
        return self.replay.row(i) if 0 <= i < len(self.replay.rows) else None

    def _val(self, i: int, col: str) -> float:
        r = self._row(i)
        return float(r[col]) if r is not None and col in r else float("nan")

    def _mean(self, i0: int, i1: int, col: str) -> tuple[float, int]:
        """mean of `col` over hour starts in [i0, i1) (steps of 4)."""
        xs = [self._val(j, col) for j in range(max(0, i0), i1, 4)]
        xs = [x for x in xs if np.isfinite(x)]
        return (sum(xs) / len(xs), len(xs)) if xs else (float("nan"), 0)

    def _hour_start(self, i: int) -> int:
        return i - int(self.replay.ts(i)[14:16]) // 15

    def _context(self, now_idx: int) -> dict[str, float]:
        """What is known at the forecast's issue time: the last complete hour ends where the current hour starts."""
        tau = self._hour_start(now_idx)
        scale, n14 = self._mean(tau - 336 * 4, tau, "base_kw")
        last24, n24 = self._mean(tau - 24 * 4, tau, "base_kw")
        return {"tau": tau, "scale": scale if n14 >= 269 else float("nan"), "last24": last24 if n24 >= 19 else float("nan")}

    def _day_temps(self, i: int) -> tuple[float, float, float]:
        ts = self.replay.ts(i)
        start = i - (int(ts[11:13]) * 4 + int(ts[14:16]) // 15)
        xs = [self._val(j, "fc_temp_c") for j in range(start, start + 96, 4)]
        xs = [x for x in xs if np.isfinite(x)]
        return (sum(xs) / len(xs), max(xs), min(xs)) if xs else (float("nan"),) * 3

    def _known(self, j: int, tau: int, col: str) -> float:
        return self._val(j, col) if j < tau else float("nan")

    # ------------------------------------------------------------------ features, as in training

    def _offset(self, d) -> float:
        """UTC offset of a local time: the dataset's fixed one, or the real one when the dataset crosses the clock changes."""
        if not self.tz_name:
            return self.tz
        from zoneinfo import ZoneInfo
        return d.replace(tzinfo=ZoneInfo(self.tz_name)).utcoffset().total_seconds() / 3600
    def _features(self, h0: int, ctx: dict[str, float]) -> tuple[dict[str, float], dict[str, float], float, float]:
        r = self._row(h0)
        ts = r["ts_local"]
        d = datetime.strptime(ts, "%Y-%m-%d %H:%M")
        doy = d.timetuple().tm_yday
        tmean, tmax, tmin = self._day_temps(h0)
        scale, tau = ctx["scale"], ctx["tau"]
        lags = {k: self._known(h0 - 4 * k, tau, "base_kw") for k in LAGS}
        base = {"hour": d.hour, "dow": d.weekday(), "weekend": int(d.weekday() >= 5), "holiday": int(d.date() in self.holidays),
                "doy_sin": math.sin(2 * math.pi * doy / 365.25), "doy_cos": math.cos(2 * math.pi * doy / 365.25),
                "temp": r.get("fc_temp_c", float("nan")), "rh": r.get("fc_rh_pct", float("nan")),
                "cloud": r.get("fc_cloud_pct", float("nan")), "sw": r.get("fc_sw_wm2", float("nan")),
                "wind": r.get("fc_wind_ms", float("nan")), "tday_mean": tmean, "tday_max": tmax, "tday_min": tmin}
        load = dict(base)
        for k, v in lags.items():
            load[f"lag{k}"] = v / scale
        n3 = [lags[k] for k in (48, 72, 96) if np.isfinite(lags[k])]
        load["naive3"] = (sum(n3) / len(n3) / scale) if n3 else float("nan")
        load["last24"] = ctx["last24"] / scale
        load["log_scale"] = math.log(max(scale, 1e-6)) if np.isfinite(scale) else float("nan")
        tm48 = self._known(h0 - 192, tau, "temp_c")
        load["temp_meas_lag48"] = tm48
        load["dtemp48"] = load["temp"] - tm48
        load.update(self.static)
        # solar: per kW of panels, relative to clear-sky sunlight at this hour
        utc = d - timedelta(hours=self._offset(d))
        cz = cos_zenith(utc, self.lat, self.lon)
        cs = clear_sky(max(cz, 0.0))
        pv = {"hour": d.hour, "cosz": cz, "sw": base["sw"], "direct": r.get("fc_direct_wm2", float("nan")),
              "diffuse": r.get("fc_diffuse_wm2", float("nan")), "cloud": base["cloud"], "temp": base["temp"]}
        pv["csi_fc"] = pv["sw"] / max(cs, 1.0) if cs > 20 else float("nan")
        per_kw = {k: self._known(h0 - 4 * k, tau, "pv_kw") / self.pv_kw if self.pv_kw > 0 else float("nan") for k in (48, 72, 96)}
        pv["lag48_csi"] = per_kw[48] / (cs / 1000) if cs > 20 else float("nan")
        p3 = [v for v in per_kw.values() if np.isfinite(v)]
        pv["naive3_csi"] = (sum(p3) / len(p3)) / (cs / 1000) if (cs > 20 and p3) else float("nan")
        return load, pv, cz, cs

    def _pv(self, feats: list, h0s: list[int]) -> dict[int, tuple[float, float, float]]:
        """The solar model's 10/50/90 % in kW for each hour start (0 at night)."""
        pv = {h: (0.0, 0.0, 0.0) for h in h0s}
        day = [k for k, f in enumerate(feats) if f[2] > 0.05 and f[3] > 20 and self.pv_kw > 0]
        if day:
            plo, pmid, phi = self.pv_m.predict([feats[k][1] for k in day])
            for j, k in enumerate(day):
                f = feats[k][3] / 1000 * self.pv_kw
                pv[h0s[k]] = (max(0.0, plo[j] * f), max(0.0, pmid[j] * f), max(0.0, phi[j] * f))
        return pv

    def _pv_correction(self, now_idx: int, days: int = 14) -> dict[int, float]:
        """Per hour of day: what this array made over the last `days` days / what the solar model said for them, each
        day forecast as it would have been (at 10:00 the day before). It corrects what the model cannot know about
        these panels (orientation, shading: it learned on another roof). Only days already over are used; an hour with
        too little sun keeps 1.0; the factor stays within 0.5-1.5."""
        if self.pv_kw <= 0:
            return {}
        day0 = now_idx - (int(self.replay.ts(now_idx)[11:13]) * 4 + int(self.replay.ts(now_idx)[14:16]) // 15)
        key = self.replay.ts(day0)
        if key in self._corr:
            return self._corr[key]
        num: dict[int, float] = {}
        den: dict[int, float] = {}
        h0s, feats = [], []
        for d in range(1, days + 1):
            p0 = day0 - 96 * d                              # 00:00 of a past day, already over
            issue = p0 - 96 + 40                            # 10:00 the day before it
            if issue < 0:
                break
            ctx = self._context(issue)
            if not np.isfinite(ctx["scale"]) or ctx["scale"] <= 0:
                continue
            hs = list(range(p0, p0 + 96, 4))
            h0s += hs
            feats += [self._features(h, ctx) for h in hs]
        pred = self._pv(feats, h0s) if h0s else {}             # all the past days in one pass of the trees
        for h in h0s:
            if pred[h][1] > 0:
                hr = int(self.replay.ts(h)[11:13])
                num[hr] = num.get(hr, 0.0) + self._val(h, "pv_kw")
                den[hr] = den.get(hr, 0.0) + pred[h][1]
        out = {hr: float(min(1.5, max(0.5, num[hr] / den[hr]))) for hr in den if den[hr] > 0.05 * self.pv_kw * days / 7}
        self._corr = {key: out}                             # one day's factors (the next day recomputes)
        return out

    # ------------------------------------------------------------------ the EV in the event window
    def _ev_days(self) -> tuple[dict[str, float], dict[str, float]]:
        """kWh the car(s) drew each day if nobody intervened (full power from arrival until done or departure, as the
        home charges): (15:00-20:00 by day, whole day by day), from the home's own charging sessions."""
        if self._evd is None:
            win: dict[str, float] = {}
            day: dict[str, float] = {}
            for x in self.replay.sessions:
                t, left = x["arrival"], float(x["kwh"])
                while left > 1e-6 and t < x["deadline"]:
                    e = min(float(x["max_kw"]) * 0.25, left)
                    left -= e
                    day[t[:10]] = day.get(t[:10], 0.0) + e
                    if "15:00" <= t[11:16] < "20:00":
                        win[t[:10]] = win.get(t[:10], 0.0) + e
                    t = add_minutes(t, 15)
            self._evd = (win, day)
        return self._evd

    def ev_expected(self, day: str) -> dict[str, float] | None:
        """The EV model for `day`: kWh the car(s) will draw 15:00-20:00 if nobody intervenes (expected value, 10/50/90 %)
        and the chance the car charges then, from what was known at 10:00 the day before (its charging up to two days
        back, the calendar, the day's forecast temperature). None without an EV, the model or 9 days of history."""
        if self.ev_m is None or not self.home.get("has_ev") or not self.replay.sessions:
            return None
        D = date.fromisoformat(day)
        first = self.replay.ts(0)[:10]
        if str(D - timedelta(days=9)) < first:
            return None
        win, dayk = self._ev_days()
        past = [str(D - timedelta(days=k)) for k in range(2, 9)]       # lags 2..8, all over before the offer
        i0 = self.replay.idx_of(f"{day} 00:00")
        temps = [self._val(j, "fc_temp_c") for j in range(i0, i0 + 96, 4)] if i0 is not None else []
        temps = [x for x in temps if np.isfinite(x)]
        row = {"dow": D.weekday(), "weekend": int(D.weekday() >= 5), "holiday": int(D in self.holidays),
               "tday_fc": sum(temps) / len(temps) if temps else float("nan"),
               "win_lag2": win.get(past[0], 0.0), "win_lag7": win.get(str(D - timedelta(days=7)), 0.0),
               "win_mean7": sum(win.get(d, 0.0) for d in past) / 7, "win_share7": sum(win.get(d, 0.0) > 0.1 for d in past) / 7,
               "day_mean7": sum(dayk.get(d, 0.0) for d in past) / 7,
               "charger_kw": max(float(x["max_kw"]) for x in self.replay.sessions), "n_ev": float(self.home.get("n_ev", 1))}
        lo, mid, hi = self.ev_m.predict([row], calibrated=False)
        mean = float(self.ev_m.predict_extra("mean", [row])[0])
        logit = float(self.ev_m.predict_extra("charging_logit", [row])[0])
        return {"mean_kwh": max(0.0, mean), "p10_kwh": max(0.0, float(lo[0])), "p50_kwh": max(0.0, float(mid[0])),
                "p90_kwh": max(0.0, float(hi[0])), "p_charging": 1.0 / (1.0 + math.exp(-logit))}

    def forecast(self, now_idx: int, ts_list: list[str]) -> dict[str, dict[str, tuple[float, float, float]]]:
        ctx = self._context(now_idx)
        hours: dict[int, str] = {}
        for ts in ts_list:
            i = self.replay.idx_of(ts)
            if i is not None:
                hours.setdefault(self._hour_start(i), ts)
        if not hours or not np.isfinite(ctx["scale"]) or ctx["scale"] <= 0:
            return {}
        h0s = sorted(hours)
        feats = [self._features(h, ctx) for h in h0s]
        rows = [f[0] for f in feats]
        lo, mid, hi = self.load_m.predict(rows)
        s = ctx["scale"]
        load = {h: (max(0.0, lo[k] * s), max(0.0, mid[k] * s), max(0.0, hi[k] * s)) for k, h in enumerate(h0s)}
        mean = self.load_m.predict_extra("mean", rows) * s if self.load_m.has("mean") else mid * s
        load_mean = {h: max(0.0, float(mean[k])) for k, h in enumerate(h0s)}
        pv = self._pv(feats, h0s)
        corr = self._pv_correction(now_idx) if self.correct_pv else {}
        if corr:
            pv = {h: tuple(v * corr.get(int(self.replay.ts(h)[11:13]), 1.0) for v in x) for h, x in pv.items()}
        out = {}
        for ts in ts_list:
            i = self.replay.idx_of(ts)
            if i is not None and self._hour_start(i) in load:
                h = self._hour_start(i)
                out[ts] = {"pv": pv[h], "load": load[h], "load_mean": load_mean[h]}
        return out


# --------------------------------------------------------------------------------------------------- the 2026 models
CAR_WINDOWS = {"summer": (18, 21), "winter": (17, 19)}
CAR_FIRST_DAY = 16                 # two weeks of history behind it
CHARGE_CHANCE = 0.10               # below this chance a car takes no part in an offer (stage4 reports/cars.json)
COOLING_MONTHS = (5, 6, 7, 8, 9)   # the warming model is used here only: in heating it loses to the simple rule
CAP_EVENT = .6                     # the part of the house's heat store a sale can use, as in stage2/tiers2.py
WATER_KIND = {"gas or oil": 0.0, "electric tank": 1.0, "heat pump": 2.0}


@lru_cache(maxsize=2)
def _clock(stamps: tuple[str, ...]) -> hf.Clock:
    meta = config.dataset_meta()
    return hf.Clock(list(stamps), meta.get("holidays", []), float(meta.get("tz_offset_h", -4)))


def _fill_day_gaps(g: np.ndarray) -> np.ndarray:
    """The spring-forward day has a missing hour: carry the last value forward, then the first one back."""
    for i in np.flatnonzero(np.isnan(g).any(axis=1)):
        r = g[i]
        ok = np.flatnonzero(~np.isnan(r))
        r[:] = r[ok[np.clip(np.searchsorted(ok, np.arange(96), side="right") - 1, 0, None)]]
    return g


def car_grids(sessions: list[dict[str, Any]], ck: hf.Clock) -> tuple[np.ndarray, np.ndarray]:
    """kW the car draws if nobody intervenes, and plugged in or not, as days x 96 local slots: the adapter's steps, from the replay's
    sessions. A session starts at its charge start and draws until its kWh are in or the car leaves. A new session replaces the last."""
    n = ck.ND * 96
    day0 = date.fromisoformat(ck.days[0])
    slot = lambda ts: (date.fromisoformat(ts[:10]) - day0).days * 96 + int(ts[11:13]) * 4 + int(ts[14:16]) // 15   # noqa: E731
    s = sorted(sessions, key=lambda x: x["arrival"])
    charge, plug = np.zeros(n, dtype="float32"), np.zeros(n, dtype=np.int8)
    if not s:
        return charge.reshape(ck.ND, 96), plug.reshape(ck.ND, 96)
    kw = float(s[0]["max_kw"])
    c = [slot(x["arrival"]) for x in s]
    for k, x in enumerate(s):
        d = slot(x["deadline"])
        a, b = max(slot(x.get("plug_in") or x["arrival"]), 0), min(d, n)
        if a < b:
            plug[a:b] = 1
        end = min(d, c[k + 1] if k + 1 < len(s) else n, n)
        t, left = c[k], float(x["kwh"])
        while left > 0.01 and t < end:
            e = min(kw, left / 0.25)
            charge[t] = e
            left -= e * 0.25
            t += 1
    return charge.reshape(ck.ND, 96), plug.reshape(ck.ND, 96)


class HomeForecaster2026:
    """One home's new forecasts, numpy only, from the models in models/forecast_2026. They need a b dataset: true plug-in times, sunlight at
    the start of its hour, direct and diffuse sunlight, the home's facts and its schedule record.

    Every forecast for a day is made as at 10:00 the day before, the moment the models were trained for, and is kept for that day.
      forecast(now_idx, ts)   the same answer as HomeForecaster, for the offer and the 5-minute loop
      day(D)                  load, solar, hot water and house drift by hour of day D
      cars(D)                 the car's window and hour models for day D
      plan                    the day-before plan, past data only: plan.blocks(D), plan.occupied(D)
    Load and solar sums use the expected value and the median. For a fleet sum of hot water use `water_kind4`, not the model:
    the plain 4-day average beats it there, 11.62 % against 13.41 %."""

    def __init__(self, name: str, replay, home: dict[str, Any] | None = None, miss_trips: float = 0.0):
        meta = config.dataset_meta()
        self.name, self.replay = name, replay
        self.home = home or _roster().get(name, {})
        self.facts = self.home.get("facts") or {}
        if not self.facts or "fc_direct_wm2" not in replay.rows[0] or "models_2026" not in meta:
            raise ValueError("the 2026 models need a b dataset (ma_year_t1b, t2b, t3b)")
        self.mdir = config.REPO_DIR / meta["models_2026"]
        stamps = tuple(r["ts_local"] for r in replay.rows)
        self.ck = ck = _clock(stamps)
        c = replay.columns(("base_kw", "pv_kw", "temp_c", "fc_temp_c", "fc_cloud_pct", "fc_sw_wm2", "fc_direct_wm2", "fc_diffuse_wm2"))
        per_hour = lambda x: np.asarray(x, dtype=float).reshape(-1, 4).mean(axis=1)   # noqa: E731
        self.load, self.pv, self.temp, self.fc_temp = per_hour(c["base_kw"]), per_hour(c["pv_kw"]), per_hour(c["temp_c"]), c["fc_temp_c"][::4]
        self.tier = int(self.home.get("tier", 1))
        f = self.facts
        self.static = {"is_hp": float(f["heating"] == "heat_pump"), "central_ac": float(f["cooling"] != "window_ac"), "people": float(f["people"]), "log_sqft": math.log(f["sqft"]),
                       "pool": float(f["pool"]), "tub": float(f["hot_tub"]), "ua": float(f["ua_kw_per_c"]), "cap": float(f["cap_kwh_per_c"]), "cool_cap": float(f["cool_cap_kw"]),
                       "water_kind": 2.0 if self.tier < 3 else WATER_KIND[f["water_heater"]]}
        self.solar = hf.Solar(ck, {"sw": c["fc_sw_wm2"][::4], "direct": c["fc_direct_wm2"][::4], "diffuse": c["fc_diffuse_wm2"][::4], "cloud": c["fc_cloud_pct"][::4], "temp": self.fc_temp},
                              float(self.home.get("lat", 42.36)), float(self.home.get("lon", -71.06)), f["pv_shift_h"], f["pv_scale_kw"])
        z = np.load(config.data_path(f"{self.home['profile']}_sched.npz"))
        self.family = f["archetype"] == "family"
        self.plan = PastPlan(ck, z["occ"], z["away"], meta.get("school_breaks", []), self.family, miss_trips, self.name)
        self.water_rec = self._device_record("hpwh")
        self.drift_rec = self._drift_record()
        self.wx = {"t": _fill_day_gaps(ck.grid(c["fc_temp_c"], np.nan)).reshape(1, ck.ND, 24, 4).mean(3).astype("float32"),
                   "c": _fill_day_gaps(ck.grid(c["fc_cloud_pct"], np.nan)).reshape(1, ck.ND, 24, 4).mean(3).astype("float32")}
        self._days: dict[int, dict[str, Any] | None] = {}
        self._cars: dict[int, dict[str, Any] | None] = {}
        self._grids: tuple[np.ndarray, np.ndarray] | None = None
        self.method = ("trained 2026: home_load_2026 and solar_per_kw_homes_2026, 10/50/90 %, made at 10:00 the day before; cars: car_window and car_hour on the true plug-in time; "
                       "the plan from past data only")

    def _m(self, name: str) -> TreeModel:
        return _model(str(self.mdir / f"{name}.json.gz"))

    def _device_record(self, key: str) -> np.ndarray | None:
        """A device's own record, by hour. None where the tier has no such device."""
        from . import devices
        row = (devices.catalog()["rows"].get("home") or {}).get(self.name)
        if row is None or not config.data_path(f"devices/home_{key}.npy").exists():
            return None
        return np.asarray(devices.array("home", key)[row], dtype=float).reshape(-1, 4).mean(axis=1)

    def _drift_record(self) -> np.ndarray | None:
        """C per hour the house would have drifted with its heating or cooling off: the heat moved each hour over the usable heat store."""
        from . import hvac
        rows = hvac._series(self.home["profile"])
        if rows is None:
            return None
        kw, mode = rows.cool.astype(float), rows.mode.astype(int)
        hour = lambda x: x.reshape(-1, 4).mean(axis=1)                                 # noqa: E731
        return (hour(np.where(mode > 0, kw, 0.0)) * hf.cop(self.temp, True) + hour(np.where(mode < 0, kw, 0.0)) * hf.cop(self.temp, False)) / (float(self.facts["cap_kwh_per_c"]) * CAP_EVENT)

    def d_of(self, ts: str) -> int | None:
        return self.ck.day_of.get(ts[:10])

    def known(self, now_idx: int, D: int) -> bool:
        """Is day D's forecast made yet? It is made at 10:00 the day before, and anything later than that is not known then."""
        tau = int(self.ck.tau[self.ck.rows_of[D][0]])
        return tau >= 0 and now_idx // 4 >= tau

    def day(self, D: int) -> dict[str, Any] | None:
        """Everything for day D by hour. Rows are the real hours of the day, so 23 or 25 on a clock-change day."""
        if D in self._days:
            return self._days[D]
        ck = self.ck
        t = ck.rows_of[D]
        if ck.tau[t[0]] < 0:
            return None
        plan = self.plan.features(t)
        h = hf.history(self.load, self.temp, self.fc_temp, ck, t)
        scale = np.maximum(h["scale"], 1e-6)
        if not (np.isfinite(scale).all() and np.isfinite(h["physk"]).all()):
            self._days[D] = None
            return None
        m = self._m("home_load_2026")
        X = hf.table(m.features, h, scale, ck, t, self.static, plan)
        kw = lambda v: np.maximum(0.0, h["physk"] + v * scale)                          # noqa: E731
        lo, mid, hi = m.predict(X)
        pv = self.solar.day(self._m("solar_per_kw_homes_2026"), D, self.pv)
        out: dict[str, Any] = {"D": D, "t": t, "hours": ck.hour[t], "load": (kw(lo), kw(mid), kw(hi)), "load_mean": kw(m.predict_extra("mean", X)), "load_rule": h["rule"],
                               "pv": (pv["p10"], pv["p50"], pv["p90"]), "pv_low_sun": pv["low_sun"], "plan": plan}
        if self.water_rec is not None:
            out.update(self._water(t, plan))
        if self.drift_rec is not None:
            out.update(self._drift(t, plan, D))
        if len(self._days) > 6:
            self._days.pop(min(self._days))
        self._days[D] = out
        return out

    def _water(self, t: np.ndarray, plan: dict[str, np.ndarray]) -> dict[str, Any]:
        """The heat pump water heater: the model for one home's energy, kind4 for what a fleet adds up."""
        ck = self.ck
        h = hf.history(self.water_rec, self.temp, self.fc_temp, ck, t)
        scale = np.maximum(h["scale"], .02)
        m = self._m("home_hot_water_2026")
        X = hf.table(m.features, h, scale, ck, t, self.static, plan)
        kw = lambda v: np.maximum(0.0, h["kind4"] + v * scale)                          # noqa: E731
        lo, mid, hi = m.predict(X)
        return {"water": (kw(lo), kw(mid), kw(hi)), "water_mean": kw(m.predict_extra("mean", X)), "water_kind4": h["kind4"]}

    def _drift(self, t: np.ndarray, plan: dict[str, np.ndarray], D: int) -> dict[str, Any]:
        """How fast the house drifts if its cooling stops, C per hour. In heating, the simple rule: the model loses to it there."""
        ck = self.ck
        h = hf.history(self.drift_rec, self.temp, self.fc_temp, ck, t)
        if int(ck.month[t[0]]) not in COOLING_MONTHS:
            r = np.nan_to_num(h["rule"])
            return {"drift": (r, r, r), "drift_mean": r, "drift_source": "rule"}
        m = self._m("home_warming_2026")
        X = hf.table(m.features, h, np.maximum(h["scale"], 1e-6), ck, t, self.static, plan, ratio=False)
        c = lambda v: np.maximum(0.0, h["physk"] + v)                                   # noqa: E731
        lo, mid, hi = m.predict(X)
        return {"drift": (c(lo), c(mid), c(hi)), "drift_mean": c(m.predict_extra("mean", X)), "drift_source": "home_warming_2026"}

    def cars(self, D: int) -> dict[str, Any] | None:
        """The car on day D, as known at 10:00 the day before. Expected kWh and the chance of charging are for offers. One home's 10-90 % is not."""
        if D in self._cars:
            return self._cars[D]
        if D < CAR_FIRST_DAY or not self.replay.sessions:
            return None
        ck = self.ck
        if self._grids is None:
            self._grids = car_grids(self.replay.sessions, ck)
        n = (D - 1) * 96 + cf.TAU                                # the decision moment: nothing later is read
        charge, plug = self._grids[0].copy().reshape(1, -1), self._grids[1].copy().reshape(1, -1)
        charge[:, n:], plug[:, n:] = 0.0, 0
        charge, plug = charge.reshape(1, ck.ND, 96), plug.reshape(1, ck.ND, 96)
        cal = {"dow": ck.day_dow, "holiday": ck.day_holiday, "off": ck.day_off, "school_out": self.plan.school}
        s = cf.series(charge, plug, CAR_WINDOWS)
        static = {"max_kw": np.array([self.facts["car_max_kw"]], "float32"), "pack_kwh": np.array([self.facts["car_pack_kwh"]], "float32")}
        occ, away = (self.plan.grid_true[None] > 0).astype("int8"), self.plan.away.copy()[None]
        away[0, D] = self.plan.told[D]                          # past trips are known. Tomorrow's is known only if the plan was told
        day = cf.build_day(D, s, {"charge": charge, "plug": plug}, cal, self.wx, static)
        pl = cf.plan_asof(D, occ, away, s["dep"], cal, np.array([self.family]), CAR_WINDOWS)
        out: dict[str, Any] = {"D": D, "window": {}}
        sig = lambda x: 1.0 / (1.0 + math.exp(-float(x)))                               # noqa: E731
        for w, hrs in CAR_WINDOWS.items():
            m = self._m(f"car_window_{w}")
            f = {**day, **cf.build_window(D, s["win"][w], hrs, cal, self.wx), **pl, "pl_occ": pl[f"pl_occ_{w}"]}
            x = np.array([[float(f[c][0]) for c in m.features]])
            lo, mid, hi = m.predict(x, calibrated=False)
            ex = lambda k: float(m.predict_extra(k, x)[0])                              # noqa: E731
            out["window"][w] = {"hours": hrs, "mean_kwh": max(0.0, ex("mean")), "p10_kwh": max(0.0, float(lo[0])), "p50_kwh": max(0.0, float(mid[0])), "p90_kwh": max(0.0, float(hi[0])),
                                "p_charging": sig(ex("charging_logit")), "p_plugged": sig(ex("plugged_logit")), "p_plugged_whole": sig(ex("plugged_whole_logit")),
                                "plugged_share": float(np.clip(ex("plugged_share_mean"), 0, 1)), "plugged_share_p10": float(np.clip(ex("plugged_share_p10"), 0, 1)), "plugged_share_p90": float(np.clip(ex("plugged_share_p90"), 0, 1))}
        hk = cf.build_hours(D, s, cal, self.wx)
        f = {**{k: np.repeat(np.asarray(v, "float32"), 24) for k, v in day.items()}, **{k: np.asarray(v, "float32").ravel() for k, v in hk.items()},
             **{k: np.repeat(np.asarray(pl[k], "float32"), 24) for k in ("pl_away", "pl_away_prev", "pl_leave", "pl_back", "pl_dep")}}
        mk, mp = self._m("car_hour_kwh"), self._m("car_hour_plugged")
        x = np.column_stack([f[c] for c in mk.features]).astype("float64")
        out["hour_kwh"] = np.maximum(0.0, mk.predict_extra("mean", x))
        out["hour_plugged"] = np.clip(mp.predict_extra("mean", x), 0, 1)
        out["plan"] = {k: float(pl[k][0]) for k in ("pl_away", "pl_leave", "pl_back", "pl_dep")}
        if len(self._cars) > 6:
            self._cars.pop(min(self._cars))
        self._cars[D] = out
        return out

    def forecast(self, now_idx: int, ts_list: list[str]) -> dict[str, dict[str, tuple[float, float, float]]]:
        """{ts: {"pv": (p10, p50, p90), "load": (p10, p50, p90), "load_mean": kW}} for the steps whose day is already forecast. Hours further off
        are left out: nothing later than now may be read, and the caller's rule covers them."""
        out = {}
        for ts in ts_list:
            i = self.replay.idx_of(ts)
            if i is None:
                continue
            t, D = i // 4, int(self.ck.day_i[i // 4])
            if not self.known(now_idx, D):
                continue
            d = self.day(D)
            if d is None:
                continue
            j = t - int(d["t"][0])
            out[ts] = {"pv": tuple(float(v[j]) for v in d["pv"]), "load": tuple(float(v[j]) for v in d["load"]), "load_mean": float(d["load_mean"][j])}
        return out

    def ev_expected(self, day: str) -> dict[str, Any] | None:
        """The car's expected draw on `day`, kW by local hour, for an offer. A car under a 10 % chance of charging in the window takes no part."""
        D = self.ck.day_of.get(day)
        c = self.cars(D) if D is not None else None
        if c is None:
            return None
        w = c["window"]["summer" if int(day[5:7]) in COOLING_MONTHS else "winter"]
        return {"mean_kwh": w["mean_kwh"], "p10_kwh": w["p10_kwh"], "p50_kwh": w["p50_kwh"], "p90_kwh": w["p90_kwh"], "p_charging": w["p_charging"], "kw_by_hour": [float(x) for x in c["hour_kwh"]],
                "takes_part": w["p_charging"] >= CHARGE_CHANCE, "window": w["hours"]}

    def appliance_run(self, key: str, D: int, record: np.ndarray) -> dict[str, Any] | None:
        """A dryer or dishwasher tomorrow from the plan. record: its 15-minute kW for the year."""
        return planned_run(self.plan, self.ck.grid(np.asarray(record, dtype=float)), D)
