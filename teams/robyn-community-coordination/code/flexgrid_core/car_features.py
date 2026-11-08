#!/usr/bin/env python3
"""Features for the car forecasts, numpy only so an agent can run the same code.

For day D the decision moment is 10:00 on D-1. build_day(D) reads only:
  complete days up to D-2 (the day series below), the first 10 hours of D-1, the calendar, the weather forecast as
  issued for D, the plan for D. leak_check() poisons everything after the decision moment and asks for equal features.
Arrays are homes x days (x hours or x 96 slots)."""
from __future__ import annotations

import warnings

import numpy as np

TAU = 40                                     # 10:00 as a 15-minute slot
EVE = 40                                     # the day's first charge is looked for from 10:00, past the night's carry-over


def series(charge: np.ndarray, plug: np.ndarray, windows: dict) -> dict:
    """Per-day series from the raw grids. Each day's value uses that day's slots, and the slot before midnight."""
    n, nd, _ = charge.shape
    kwh = charge.astype("float32") * 0.25
    hk = kwh.reshape(n, nd, 24, 4).sum(3)
    hp = plug.astype("float32").reshape(n, nd, 24, 4).mean(3)
    flat = plug.reshape(n, -1).astype(bool)
    rise = np.zeros_like(flat)
    rise[:, 1:] = flat[:, 1:] & ~flat[:, :-1]
    rise = rise.reshape(n, nd, 96)
    pi = np.where(rise.any(2), rise.argmax(2).astype("float32"), np.nan)      # first plug-in of the day
    on = charge[:, :, EVE:] > 0
    ce = np.where(on.any(2), (on.argmax(2) + EVE).astype("float32"), np.nan)  # first charge from 10:00 on
    late = (pi >= 56) & (pi < 84)                                             # plug-ins between 14:00 and 21:00
    delay = np.where(late, (ce - pi) >= 1, np.nan).astype("float32")          # charging waited: a timer
    delay[late & np.isnan(ce)] = 1.0
    fall = np.zeros_like(flat)
    fall[:, 1:] = ~flat[:, 1:] & flat[:, :-1]
    fall = fall.reshape(n, nd, 96)
    dep = np.where(fall.any(2), fall.argmax(2).astype("float32"), np.nan)     # the car's first departure of the day
    pos = charge > 0
    last = np.where(pos.any(2), 95 - pos[:, :, ::-1].argmax(2).astype("float32"), np.nan)
    out = {"hk": hk, "hp": hp, "dk": hk.sum(2), "pi": pi / 4, "ce": ce / 4, "delay": delay, "dep": dep / 4, "last": last, "win": {}}
    for w, (a, b) in windows.items():
        out["win"][w] = {"k": hk[:, :, a:b].sum(2), "p": hp[:, :, a:b].mean(2)}
    return out


def _m(x: np.ndarray, axis: int = 1) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanmean(x, axis=axis)


def _sd(x: np.ndarray) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nanstd(x, axis=1)


def _lag(x: np.ndarray, D: int, k: int) -> np.ndarray:
    return x[:, D - k] if D - k >= 0 else np.full(x.shape[0], np.nan, dtype="float32")


def _past(x: np.ndarray, D: int, n: int) -> np.ndarray:
    """The n complete days before the decision: D-1-n .. D-2."""
    return x[:, max(0, D - 1 - n):D - 1]


def _same(x: np.ndarray, D: int, off: np.ndarray, n: int = 28) -> np.ndarray:
    """Those of the last n complete days that are the same kind of day as D (working day or day off)."""
    d0 = max(0, D - 1 - n)
    return x[:, d0:D - 1][:, off[d0:D - 1] == off[D]]


def _dow(x: np.ndarray, D: int) -> np.ndarray:
    idx = [D - 7 * k for k in (1, 2, 3, 4) if D - 7 * k >= 0]
    return x[:, idx]


DAY = ["dow", "weekend", "holiday", "prev_off", "next_off", "tday_fc", "dtemp14", "tmax_fc", "tmin_fc", "cloud_fc", "max_kw", "pack_kwh",
       "plugged_now", "kwh_morn", "days_since_charge", "day_lag2", "day_lag3", "day_mean7", "day_mean28", "charge_days14",
       "kwh_per_charge28", "need_est", "pi_same", "pi_same_sd", "pi_share_same", "pi_all", "delay28", "ce_same"]
WIN = ["twin_fc", "w_lag2", "w_lag3", "w_lag7", "w_lag14", "w_mean7", "w_mean14", "w_mean28", "w_share7", "w_share14", "w_share28",
       "w_same", "w_share_same", "w_dow", "w_share_dow", "w_on28", "p_lag2", "p_lag7", "p_mean7", "p_mean14", "p_mean28", "p_same", "p_dow"]
HOUR = ["hour", "thour_fc", "h_lag1m", "h_lag2", "h_lag7", "h_mean7", "h_mean14", "h_mean28", "h_same", "h_dow", "hp_lag1m", "hp_lag2",
        "hp_lag7", "hp_mean7", "hp_mean14", "hp_mean28", "hp_same", "hp_dow"]


def build_day(D: int, s: dict, raw: dict, cal: dict, wx: dict, static: dict) -> dict:
    """Day-level features for day D, one value per home. s: series(); raw: charge and plug grids (only D-1's first 10
    hours are read); cal: dow, off, holiday by day; wx: forecast as issued, homes x days x 24; static: max_kw, pack_kwh."""
    k, dk = raw["charge"], s["dk"]
    n = dk.shape[0]
    morn = k[:, D - 1, :TAU]
    seen = np.isfinite(s["last"][:, :D - 1])                                # days up to D-2 with any charging
    lastday = D - 2 - seen[:, ::-1].argmax(1)
    last = np.where(seen.any(1), lastday + (np.take_along_axis(s["last"], lastday[:, None], 1)[:, 0] + 1) / 96, np.nan)
    since = np.where((morn > 0).any(1), 0.0, (D - 1) + TAU / 96 - last)
    d28 = _past(dk, D, 28)
    day_mean28 = _m(d28)
    days_on = (d28 > 0.5)
    f = {"dow": np.full(n, cal["dow"][D], dtype="float32"), "weekend": np.full(n, float(cal["dow"][D] >= 5), dtype="float32"),
         "holiday": np.full(n, float(cal["holiday"][D]), dtype="float32"),
         "prev_off": np.full(n, float(cal["off"][D - 1]), dtype="float32"),
         "next_off": np.full(n, float(cal["off"][min(D + 1, len(cal["off"]) - 1)]), dtype="float32"),
         "tday_fc": wx["t"][:, D].mean(1), "dtemp14": wx["t"][:, D].mean(1) - _m(_past(wx["t"], D, 14).mean(2)), "tmax_fc": wx["t"][:, D].max(1), "tmin_fc": wx["t"][:, D].min(1), "cloud_fc": wx["c"][:, D].mean(1),
         "max_kw": static["max_kw"], "pack_kwh": static["pack_kwh"],
         "plugged_now": raw["plug"][:, D - 1, TAU - 1].astype("float32"), "kwh_morn": morn.sum(1) * 0.25,
         "days_since_charge": since, "day_lag2": _lag(dk, D, 2), "day_lag3": _lag(dk, D, 3), "day_mean7": _m(_past(dk, D, 7)),
         "day_mean28": day_mean28, "charge_days14": _m((_past(dk, D, 14) > 0.5).astype("float32")),
         "kwh_per_charge28": _m(np.where(days_on, d28, np.nan)), "need_est": day_mean28 * since,
         "pi_same": _m(_same(s["pi"], D, cal["off"])), "pi_same_sd": _sd(_same(s["pi"], D, cal["off"])),
         "pi_share_same": _m(np.isfinite(_same(s["pi"], D, cal["off"])).astype("float32")), "pi_all": _m(_past(s["pi"], D, 28)),
         "delay28": _m(_past(s["delay"], D, 28)), "ce_same": _m(_same(s["ce"], D, cal["off"]))}
    return f


def build_window(D: int, w: dict, hours: tuple, cal: dict, wx: dict) -> dict:
    """Window-level features for day D. w: series()['win'][name]."""
    k, p = w["k"], w["p"]
    a, b = hours
    on = (k > 0.1).astype("float32")
    return {"twin_fc": wx["t"][:, D, a:b].mean(1), "w_lag2": _lag(k, D, 2), "w_lag3": _lag(k, D, 3), "w_lag7": _lag(k, D, 7), "w_lag14": _lag(k, D, 14),
            "w_mean7": _m(_past(k, D, 7)), "w_mean14": _m(_past(k, D, 14)), "w_mean28": _m(_past(k, D, 28)),
            "w_share7": _m(_past(on, D, 7)), "w_share14": _m(_past(on, D, 14)), "w_share28": _m(_past(on, D, 28)),
            "w_same": _m(_same(k, D, cal["off"])), "w_share_same": _m(_same(on, D, cal["off"])), "w_dow": _m(_dow(k, D)), "w_share_dow": _m(_dow(on, D)),
            "w_on28": _m(np.where(_past(k, D, 28) > 0.1, _past(k, D, 28), np.nan)),
            "p_lag2": _lag(p, D, 2), "p_lag7": _lag(p, D, 7), "p_mean7": _m(_past(p, D, 7)), "p_mean14": _m(_past(p, D, 14)), "p_mean28": _m(_past(p, D, 28)),
            "p_same": _m(_same(p, D, cal["off"])), "p_dow": _m(_dow(p, D))}


def build_hours(D: int, s: dict, cal: dict, wx: dict) -> dict:
    """Hour-level features for day D, homes x 24."""
    hk, hp = s["hk"], s["hp"]
    n = hk.shape[0]
    early = np.arange(24) < 10                       # these hours of D-1 are already over at 10:00
    f = {"hour": np.broadcast_to(np.arange(24, dtype="float32"), (n, 24)), "thour_fc": wx["t"][:, D]}
    for name, x in (("h", hk), ("hp", hp)):
        f[f"{name}_lag1m"] = np.where(early, x[:, D - 1], np.nan)
        f[f"{name}_lag2"] = x[:, D - 2]
        f[f"{name}_lag7"] = x[:, D - 7] if D >= 7 else np.full((n, 24), np.nan, dtype="float32")
        for m in (7, 14, 28):
            f[f"{name}_mean{m}"] = _m(_past(x, D, m))
        f[f"{name}_same"] = _m(_same(x, D, cal["off"]))
        f[f"{name}_dow"] = _m(_dow(x, D))
    return f


def leave_back(on: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """on: homes x 96, someone home. First time out after 06:00, first time back after that, in hours."""
    n = on.shape[0]
    leave, back = np.full(n, np.nan, "float32"), np.full(n, np.nan, "float32")
    out = ~on[:, 24:]
    has = out.any(1)
    l0 = out.argmax(1) + 24
    leave[has] = l0[has] / 4
    for i in np.flatnonzero(has):
        r = np.flatnonzero(on[i, l0[i]:])
        if len(r):
            back[i] = (l0[i] + r[0]) / 4
    return leave, back


def plan_asof(D: int, occ: np.ndarray, away: np.ndarray, dep: np.ndarray, cal: dict, fam: np.ndarray, windows: dict) -> dict:
    """The plan for day D as it can be made at 10:00 the day before: the home's own last 28 days of the same kind of day,
    the family's trips, the school calendar. No snow closure: that notice comes the night before."""
    off, so = cal["off"], cal["school_out"]
    n, nd = away.shape
    d0 = max(0, D - 29)
    typ = np.zeros((n, 96), "float32")
    for grp, kind in ((fam, off | so), (~fam, off)):                        # families: school breaks count as days off
        same = np.flatnonzero(kind[d0:D - 1] == kind[D]) + d0
        if len(same) and grp.any():
            o = (occ[grp][:, same] > 0).astype("float32")
            w = (~away[grp][:, same])[:, :, None].astype("float32")
            typ[grp] = (o * w).sum(1) / np.maximum(w.sum(1), 1)
    on = typ >= 0.5
    gone = away[:, D]
    lv, bk = leave_back(on)
    f = {"pl_away": gone.astype("float32"), "pl_away_prev": away[:, D - 1].astype("float32"),
         "pl_leave": np.where(gone, np.nan, lv), "pl_back": np.where(gone, np.nan, bk), "pl_dep": _m(_same(dep, D, off))}
    for w, (a, b) in windows.items():
        f[f"pl_occ_{w}"] = np.where(gone, 0.0, on[:, a * 4:b * 4].mean(1))
    return f
