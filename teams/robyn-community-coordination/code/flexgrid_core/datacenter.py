"""The data center node: a data center joins FlexGrid the way a home does, through its own agent.

What the agent may change, and what it never touches
  * critical load (servers answering users, their cooling): never touched
  * deferrable batch work: the dataset's `dc_flexible_kw`, "load that could be postponed right now". Each piece may
    wait up to the site's deferral window (`max_defer_steps`: 4.5 to 10.5 h for the ten sites) and then must run.
    A job is never late: work that reaches its deadline runs, whatever else is going on
  * the on-site battery, but only above its backup reserve (the UPS ride-through, 50 % by default, is never used)
What it shares with the coordinator: offers (kW it can shed per 30-minute slot, with a confidence), answers, signed
meter readings. Never its job list, never its servers' state.

How the site runs (DataCenterSite.step, one 15-minute dataset step)
  default  every arriving piece of work runs at once; the battery sits at its normal charge (90 %, the BMS ceiling).
           This is the no-event twin (`import_cf_kw`) and the grid operator's meter history
  event    a contract slot says "use X kW less than default": arriving work is paused first (it goes to the queue
           with its deadline), the battery covers the rest down to the backup reserve
  clean    the coordinator's day-ahead clean allowance (its forecast of community solar the data centers would
           otherwise leave unused, shared out by flexible work): arriving work is held for allowance steps inside its
           own deferral window and runs there
After an event (or work held for an allowance that did not come) the queue runs at the lowest constant rate that
still meets every deadline (earliest deadline first); the battery recharges slowly once the queue is empty.

Data: data/datacenters/ (tools: the extraction notes in data/datacenters/README.md). The same 35,040 local-time
15-minute steps as the home profiles, so one replay clock drives homes and data centers alike.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import threading
import time
from collections import deque
from concurrent.futures import Future
from typing import Any, Callable

from . import beckn, config, ids, signing

STEP_H = 0.25
POST_STEPS = 32                          # the evening after the window that offers forecast (8 h), as homes do
REBOUND_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
_CACHE: dict[str, Any] = {}
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------------------------------------- data
def _csv(name: str) -> list[dict[str, str]]:
    with gzip.open(config.data_path(f"datacenters/{name}"), "rt") as f:
        return list(csv.DictReader(f))


def files() -> dict[str, Any]:
    """The data center node's files and words for the active dataset (dataset.json's "datacenters"): Illinois 2023
    (data/datacenters/) or Massachusetts 2026 (data_ma/datacenters/, tools/make_ma_datacenters.py)."""
    return {**config.IL2023_META["datacenters"], **(config.dataset_meta().get("datacenters") or {})}


def available() -> bool:
    """Whether the active dataset has the data center node (its own sites' data, not the Illinois fallback)."""
    try:
        return config.data_path("datacenters/datacenters.json").exists() and \
            config.data_path("datacenters/" + files()["events"]).exists()
    except Exception:
        return False


def meta() -> dict[str, Any]:
    with _LOCK:
        if "meta" not in _CACHE:
            _CACHE["meta"] = json.loads(config.data_path("datacenters/datacenters.json").read_text())
        return _CACHE["meta"]


def site_ids() -> list[str]:
    return [m["entity_id"] for m in meta()["datacenters"]]


def site_meta(entity_id: str) -> dict[str, Any]:
    return next(m for m in meta()["datacenters"] if m["entity_id"] == entity_id)


def node_name(entity_id: str) -> str:
    """DC_03 -> dc-03 (the NANDA Town participant name of that data center's agent)."""
    return "dc-" + entity_id.split("_")[-1]


def entity_of(name: str) -> str:
    return "DC_" + name.split("-")[-1]


def _columns(name: str, cols: tuple[str, ...]) -> dict[str, Any]:
    with _LOCK:
        if name not in _CACHE:
            rows = _csv(name)
            d: dict[str, Any] = {"ts": [r["ts_local"] for r in rows]}
            for c in cols:
                d[c] = [float(r[c]) for r in rows]
            idx: dict[str, int] = {}
            for i, t in enumerate(d["ts"]):
                idx.setdefault(t, i)            # the repeated hour when clocks fall back keeps its first copy
            d["index"] = idx
            _CACHE[name] = d
        return _CACHE[name]


def community_solar() -> dict[str, Any]:
    """The 50 community solar plants (PV + battery), summed: `solar_kw` = output before export curtailment."""
    return _columns(files()["solar"], ("solar_kw", "solar_ac_kw", "curtailed_kw", "bess_kw"))


def region() -> dict[str, Any]:
    """The grid region the community sits in: total demand (all agents included) and its prices (MISO Illinois hub
    in 2023; ISO New England's WCMA zone in 2026)."""
    return _columns(files()["region"], ("system_load_mw", "merchant_pv_mw", "net_load_dispatched_mw",
                                        "price_usd_mwh", "price_da_usd_mwh"))


def events_catalog() -> dict[str, Any]:
    """The mock grid operator's large-load events (Beckn catalog) for the active dataset."""
    return json.loads(config.data_path("datacenters/" + files()["events"]).read_text())["message"]["catalogs"][0]


def _levers() -> dict[str, Any]:
    """benchmark: the facility levers switched on (FG_DC_LEVERS = "all" or a comma list), with their numbers."""
    import os
    on = os.environ.get("FG_DC_LEVERS", "")
    if not on or os.environ.get("FLEXGRID_DATASET") == "il2023":
        return {}
    from . import devices
    cat = devices.catalog()["devices"]["datacenter"]
    return dict(cat) if on == "all" else {k: v for k, v in cat.items() if k in on.split(",")}


LEVERS = _levers()
# a site's own lever table (datacenters.json "levers"). The catalog stays the default for its numbers and FG_DC_LEVERS the outer
# switch; a site with no table runs the catalog as before. A lever needs what the operator controls (operator_controls), so data
# cannot grant a lever to someone who does not own what it moves: a colocation operator cannot cap its tenants' servers.
LEVER_NEEDS = {"power_cap": "servers", "compute_pause": "workload", "chiller_setpoint": "cooling", "lighting": "building"}
LEVER_BASIS = {"power_cap": "it", "chiller_setpoint": "mech_cooling", "lighting": "load"}


def site_levers(m: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], bool, dict[str, dict[str, Any]] | None]:
    """(facility levers, may it pause work, its stores of energy) for a site's entry. Stores: None means the bess_* fields."""
    table = m.get("levers")
    if table is None:
        return dict(LEVERS), True, None
    controls = set(m.get("operator_controls") or ())

    def allowed(k: str, p: dict[str, Any]) -> bool:
        return bool(p.get("on", True)) and p.get("needs", "storage" if p.get("kind") == "store" else LEVER_NEEDS.get(k)) in controls
    levers = {}
    for k, p in table.items():
        if k == "compute_pause" or p.get("kind") == "store" or k not in LEVERS or not allowed(k, p):
            continue
        q = {**LEVERS[k], **p}
        if q.get("basis", LEVER_BASIS.get(k)) not in ("it", "mech_cooling", "load"):
            raise ValueError(f"{m.get('entity_id')}: lever {k} has no known basis")
        levers[k] = q
    pause = table.get("compute_pause")
    return levers, bool(pause) and allowed("compute_pause", pause), \
        {k: p for k, p in table.items() if p.get("kind") == "store" and allowed(k, p)}


class Series:
    """One data center's series (the dataset's year): facility load, the part of it that could wait, capacity,
    on-site solar."""

    def __init__(self, entity_id: str):
        d = _columns(f"dc_{entity_id}.csv.gz", ("load_kw", "flex_kw", "cap_kw", "pv_kw", "temp_c", "it_kw", "pue",
                                                "dr_ref_kw"))
        self.entity_id = entity_id
        self.ts, self.index = d["ts"], d["index"]
        self.load, self.flex, self.cap, self.pv, self.temp = d["load_kw"], d["flex_kw"], d["cap_kw"], d["pv_kw"], d["temp_c"]
        self.dr_ref = d["dr_ref_kw"]
        self.it, self.pue = d["it_kw"], d["pue"]

    def __len__(self) -> int:
        return len(self.ts)

    def idx_of(self, ts: str) -> int | None:
        return self.index.get(ts)

    def default_import(self, i: int) -> float:
        """What the meter reads in default operation (every job runs on arrival, battery idle at its normal charge)."""
        return self.load[i] - self.pv[i]


class FlatSeries:
    """A new data center that does not exist yet (the headroom study): a flat facility load, with the fleet's
    share of deferrable work at each step and the fleet's capacity margin."""

    def __init__(self, load_kw: float, flex_share: list[float], cap_ratio: float, ts: list[str]):
        n = len(flex_share)
        self.entity_id, self.ts, self.index = "NEW", ts, {}
        self.load = [load_kw] * n
        self.flex = [load_kw * f for f in flex_share]
        self.cap = [load_kw * cap_ratio] * n
        self.pv = [0.0] * n
        self.temp = [0.0] * n

    def __len__(self) -> int:
        return len(self.load)

    def default_import(self, i: int) -> float:
        return self.load[i] - self.pv[i]


# ---------------------------------------------------------------------------------------------------------- site
def beckn_minutes(a: str, b: str) -> float:
    """Minutes from a to b (dataset local times)."""
    from datetime import datetime
    f = "%Y-%m-%d %H:%M"
    return (datetime.strptime(b, f) - datetime.strptime(a, f)).total_seconds() / 60.0


class DataCenterSite:
    """The physical site, simulated from the dataset: what the agent controls and what the meter reads."""

    def __init__(self, entity_id: str, cfg: dict[str, Any] | None = None, clock=None, series=None,
                 site: dict[str, Any] | None = None):
        cfg = cfg or {}
        m = site or site_meta(entity_id)
        self.id = entity_id
        self.s = series if series is not None else Series(entity_id)
        self.clock = clock
        self.w = int(cfg.get("defer_steps", m["max_defer_steps"]))
        self.P = float(cfg.get("battery_kw", m["bess_power_kw"]))
        self.E = float(cfg.get("battery_kwh", m["bess_energy_kwh"]))
        self.eta = math.sqrt(float(m.get("eta_battery_rt") or 0.92))           # one way
        # the site's own number if it has one, else the run's
        self.backup = float(m.get("backup_pct", cfg.get("backup_pct", 50.0))) / 100   # UPS reserve: never used for flex
        self.hold = float(m.get("hold_pct", cfg.get("hold_pct", 90.0))) / 100    # normal charge (BMS ceiling 90 %)
        self.recharge_hours = tuple(m.get("recharge_hours", cfg.get("recharge_hours", (22, 6))))   # refill at night only
        self.levers, self.can_pause, stores = site_levers(m)
        self.store_share: dict[str, float] = {"battery": 1.0}                    # how a discharge divides between stores
        if stores is not None:
            self._own_stores(stores)
        self.soc = self.hold
        # contracts' resume plans: from the window end to the last paused job's deadline, kW of paused work per step
        self.resumes: list[dict[str, Any]] = []                                  # [{"from", "until", "plan", "left"}]
        self.queue: deque = deque()          # [deadline_idx, kW for one step]; arrivals in time order = EDF order
        self.plan: dict[str, dict[str, Any]] = {}                               # ts -> {"deliver_kw", "contract_id"}
        self.allowance: dict[int, float] = {}                                    # idx -> kW clean allowance
        self.reserved: dict[int, float] = {}                                     # idx -> kW held for that step
        self.idx: int | None = None
        self.tel: dict[str, Any] = {}
        self.recent: dict[int, dict[str, Any]] = {}
        self.late_kwh = 0.0                  # work that reached its deadline with no room left under the site's
        self.over_cap_kw = 0.0               # capacity (it would be late in reality): must stay 0
        # what if (the Control Room; flexgrid-dc POST /dc/conditions): conditions of the day the operator knows of
        self.urgent = 0.0                    # the share of arriving batch work that is urgent today: runs on arrival
        self.battery_out = False             # the battery out of service: 0 kW either way
        self.urgent_jobs: list[dict[str, Any]] = []   # the operator's urgent jobs: {"from", "to" (idx), "kw"} of the
                                                      # batch work that can no longer wait (it cannot be paused)
        # benchmark: the facility levers. Mechanical cooling is the part of the site's PUE that follows the weather; the
        # rest of its overhead is fans, pumps, power losses and lighting
        self.pue_base = float(m.get("pue_base") or 1.2)
        self.pue_k, self.econ = float(m.get("pue_temp_k") or 0.0), float(m.get("economiser_c") or 10.0)
        self.fac_debt = 0.0                  # kWh of chilled-water cold borrowed, to be made back after the window
        self.fac_log: dict[str, float] = {}  # kWh given, by lever
        self.levers_off: set[str] = set()    # the operator's or the agent's "not this one"
        self.levers_all = tuple(self.levers)
        self.levers_forbidden = set(LEVERS) - set(self.levers)   # off at this site whatever an agent answers

    def _own_stores(self, stores: dict[str, dict[str, Any]]) -> None:
        """The stores of the site's table (a share of the UPS, an installed battery) as one: each keeps its own reserve, and a
        discharge divides between them in proportion to what each may give."""
        self.P = sum(float(p["kw"]) for p in stores.values())
        self.E = sum(float(p["kwh"]) for p in stores.values())
        self.store_share = {}
        if self.E <= 0:
            self.P, self.E, self.backup, self.hold = 0.0, 0.0, 0.0, 0.0
            return
        use = {k: (float(p.get("hold_pct", 100.0)) - float(p["reserve_pct"])) / 100 * float(p["kwh"]) for k, p in stores.items()}
        self.hold = sum(float(p.get("hold_pct", 100.0)) / 100 * float(p["kwh"]) for p in stores.values()) / self.E
        self.backup = self.hold - sum(use.values()) / self.E
        self.eta = math.sqrt(sum(float(p.get("round_trip", 0.92)) * float(p["kwh"]) for p in stores.values()) / self.E)
        self.store_share = {k: v / max(1e-9, sum(use.values())) for k, v in use.items()}

    def facility_kw(self, i: int, paused_kw: float = 0.0, forecast: bool = False) -> dict[str, float]:
        """kW each facility lever can take off the meter at step i, on what is left running (work that is paused has
        already taken its cooling with it). forecast: without the ride-through's own limit."""
        if not self.levers or i is None or isinstance(self.s, FlatSeries):
            return {}
        s = self.s
        it = max(0.0, float(s.it[i]) * (1.0 - min(1.0, paused_kw / max(1.0, float(s.load[i])))))
        chill = self.pue_k * max(0.0, float(s.temp[i]) - self.econ) * it
        out = {}
        for k, p in self.levers.items():
            if k in self.levers_off or p.get("kind") == "reserve":
                continue
            basis = p.get("basis", LEVER_BASIS.get(k))
            if basis == "mech_cooling":
                out[k] = chill * float(p["share"])
            elif basis == "load":
                out[k] = float(s.load[i]) * float(p["load_share"]) * float(p["share"])
            elif basis == "it":
                out[k] = it * self.pue_base * float(p["share"])
            if k in out and p.get("max_load_share") is not None:     # measured ceiling, as a share of the site
                out[k] = min(out[k], float(s.load[i]) * float(p["max_load_share"]))
        return out

    def must_run(self, i: int) -> float:
        """kW of arriving batch work that an urgent job keeps from being paused at step i."""
        return sum(j["kw"] for j in self.urgent_jobs if j["from"] <= i < j["to"])

    # --- state ----------------------------------------------------------------------------------------------
    def queued_kwh(self) -> float:
        return sum(q[1] for q in self.queue) * STEP_H

    def can_discharge(self) -> float:
        if self.battery_out:
            return 0.0
        return max(0.0, min(self.P, (self.soc - self.backup) * self.E * self.eta / STEP_H))

    def can_charge(self) -> float:
        if self.battery_out:
            return 0.0
        return max(0.0, min(self.P, (self.hold - self.soc) * self.E / self.eta / STEP_H))

    def usable_kwh(self, soc: float | None = None) -> float:
        """Battery energy above the backup reserve, at the meter (none while the battery is out of service)."""
        if self.battery_out:
            return 0.0
        return max(0.0, ((self.soc if soc is None else soc) - self.backup) * self.E * self.eta)

    def can_wait(self, kw: float, i: int | None = None) -> float:
        """The part of arriving batch work that may be held: all of it, less today's urgent share and any urgent job."""
        if not self.can_pause:
            return 0.0
        return max(0.0, kw * (1.0 - self.urgent) - (self.must_run(i) if i is not None else 0.0))

    def min_rate(self, i: int, kind: str | None = None) -> float:
        """Lowest constant kW from step i on (inclusive) that still runs every queued job (of `kind`, or all) by its
        deadline. Work held for clean-allowance steps has its own steps reserved, so it is left out of the
        after-event resume rate."""
        need, best = 0.0, 0.0
        for d, kw, k in self.queue:
            if kind and k != kind:
                continue
            need += kw
            best = max(best, need / max(1, d - i + 1))
        return best

    def has_queued(self, kind: str) -> bool:
        return any(k == kind for _, _, k in self.queue)

    def peek(self, i: int) -> tuple[float, float]:
        """(work arriving at step i, queued work that must run at step i)."""
        due = sum(kw for d, kw, _ in self.queue if d <= i)
        return self.s.flex[i], due

    def _run_queue(self, kw: float, kind: str | None = None) -> float:
        """Run up to kw of queued work (only of `kind` if given), earliest deadline first."""
        done = 0.0
        for item in self.queue:
            if kw - done <= 1e-9:
                break
            if kind and item[2] != kind:
                continue
            take = min(item[1], kw - done)
            item[1] -= take
            done += take
        if done > 0:
            self.queue = deque(x for x in self.queue if x[1] > 1e-9)
        return done

    # --- physics ----------------------------------------------------------------------------------------------
    def step(self, i: int, act: dict[str, Any] | None = None) -> dict[str, Any]:
        """One 15-minute step. act (all optional):
             deliver_kw   use this much less than default (event slot): pause arriving work, battery for the rest
             hold_kw      hold this much of the arriving work in the queue (clean allowance, coordinator dispatch)
             run_kw       run up to this much queued work early (default: the lowest rate that meets every deadline)
             run_event_kw / run_clean_kw   the agent's own plan: run this much paused (event) / held (clean) work at
                          this step in total, the part that is at its deadline included
             battery_kw   + discharge / - charge (coordinator dispatch); default: recharge slowly when the queue is empty
        """
        act = act or {}
        s = self.s
        arr = s.flex[i]
        crit = s.load[i] - arr
        cap = s.cap[i]
        pv = s.pv[i]
        due_k = {"event": 0.0, "clean": 0.0}
        while self.queue and self.queue[0][0] <= i:            # at its deadline: runs now, whatever else happens
            _, kw, k = self.queue.popleft()
            due_k[k] = due_k.get(k, 0.0) + kw
        due = sum(due_k.values())
        deliver = act.get("deliver_kw")
        hold, kind = 0.0, "event"
        if deliver is not None:
            hold = min(self.can_wait(arr, i), max(0.0, float(deliver)))
        elif act.get("hold_kw"):
            hold = min(self.can_wait(arr, i), max(0.0, float(act["hold_kw"])))
            kind = act.get("hold_kind", "event")
        if hold > 1e-9:
            self.queue.append([i + self.w, hold, kind])
        run = due + arr - hold
        extra = 0.0
        if deliver is None:                                    # queued work early; never inside an event slot
            room = cap - crit - run
            if "run_event_kw" in act or "run_clean_kw" in act:  # the agent's plan: totals, due work counted in
                for k in ("event", "clean"):
                    x = self._run_queue(max(0.0, min(float(act.get(f"run_{k}_kw", 0.0)) - due_k.get(k, 0.0), room)), k)
                    extra += x
                    room -= x
            else:
                want = act.get("run_kw")
                if want is None:
                    want = self.min_rate(i) if self.queue else 0.0
                extra = self._run_queue(max(0.0, min(float(want), room)))
        run += extra
        if crit + run > cap + 1e-6:                            # only work at its deadline can get here
            self.over_cap_kw = max(self.over_cap_kw, crit + run - cap)
            self.late_kwh += (crit + run - cap) * STEP_H
        can_dis, can_ch = self.can_discharge(), self.can_charge()
        b = 0.0
        mode = "default"
        fac, rec = 0.0, 0.0
        lever_kw: dict[str, float] = {}
        if deliver is not None:
            need_kw = max(0.0, float(deliver) - hold + due)
            for k, v in self.facility_kw(i, hold).items():      # the levers that store nothing go before the battery
                x = min(v, need_kw - fac)
                if x <= 1e-9:
                    continue
                fac += x
                lever_kw[k] = x
                if k == "chilled_water":
                    self.fac_debt += x * STEP_H
            b = min(can_dis, max(0.0, need_kw - fac))
            mode = "deliver"
            if abs(hold - due) > 1e-9:                          # true delivery = paused - due + levers + stores
                lever_kw["compute_pause"] = hold - due
            if b > 1e-9:
                lever_kw.update({k: b * f for k, f in self.store_share.items()})
            for k, x in lever_kw.items():
                self.fac_log[k] = self.fac_log.get(k, 0.0) + x * STEP_H
        elif self.fac_debt > 1e-6:                              # the chilled water cools back down at a gentle rate
            rec = min(self.fac_debt / STEP_H, max(1.0, 0.25 * self.pue_k * max(0.0, float(s.temp[i]) - self.econ) * float(s.it[i])))
            self.fac_debt -= rec * STEP_H
        elif act.get("battery_kw") is not None:
            v = float(act["battery_kw"])
            b = min(v, can_dis) if v >= 0 else -min(-v, can_ch)
            mode = "dispatch"
        elif not self.has_queued("event") and self.soc < self.hold - 1e-6 and self._night(i):
            b = -min(can_ch, self.P * float(act.get("recharge_share", 0.25)))
        if hold > 1e-9 and mode == "default":
            mode = "hold"
        if extra > 1e-9 and mode == "default":
            mode = "run_held"
        if self.E > 0:                                          # a site with no store has no charge to change
            self.soc += (-b * STEP_H * self.eta / self.E) if b < 0 else (-b * STEP_H / self.eta / self.E)
        imp = crit + run - pv - b - fac + rec
        self.idx = i
        self.tel = {"ts": s.ts[i], "idx": i, "import_kw": round(imp, 2), "import_cf_kw": round(s.load[i] - pv, 2),
                    "critical_kw": round(crit, 2), "flex_arrived_kw": round(arr, 2), "flex_run_kw": round(run, 2),
                    "held_kw": round(hold, 2), "due_kw": round(due, 2), "early_kw": round(extra, 2),
                    "queue_kwh": round(self.queued_kwh(), 2), "battery_kw": round(b, 2), "facility_kw": round(fac - rec, 2),
                    "lever_kw": {k: round(x, 2) for k, x in lever_kw.items()},
                    "soc_pct": round(self.soc * 100, 2), "pv_kw": round(pv, 2), "cap_kw": round(cap, 1), "mode": mode}
        return self.tel

    def _night(self, i: int) -> bool:
        a, b = self.recharge_hours
        h = int(self.s.ts[i][11:13])
        return (a <= h or h < b) if a > b else (a <= h < b)

    # --- agent mode: the clock drives the site, the agent's plan and allowance decide each step ----------------
    def _act_for(self, i: int) -> dict[str, Any]:
        a = self.plan.get(self.s.ts[i]) or {}
        if "deliver_kw" in a:
            return {"deliver_kw": a["deliver_kw"]}
        act: dict[str, Any] = {}
        active = [r for r in self.resumes if r["from"] <= i <= r["until"]]
        if active:                                             # the coordinator's resume plan (valley of the region),
            p = sum(r["plan"].pop(i, 0.0) for r in active)     # scaled to what is really queued (the plan was made
            left = sum(r["left"] for r in active)              # from forecasts); work at its deadline counts in it
            q = sum(kw for _, kw, k in self.queue if k == "event")
            act["run_event_kw"] = p * q / left if left > 1e-9 else self.min_rate(i, "event")
            for r in active:
                r["left"] = max(0.0, r["left"] - p * (r["left"] / left if left > 1e-9 else 0.0))
        self.resumes = [r for r in self.resumes if r["until"] >= i - 1]
        if self.allowance:
            arr, held = self.s.flex[i], 0.0
            if self.allowance.get(i, 0.0) <= 1e-9:             # no surplus forecast now: hold arriving work for a
                for j in range(i + 1, i + self.w + 1):         # later step that has some, inside its window (the
                    if arr - held <= 1e-9:                     # allowance is surplus beyond the default load, so
                        break                                  # work arriving in a surplus step runs as it is)
                    left = self.allowance.get(j, 0.0) - self.reserved.get(j, 0.0)
                    if left > 1e-9:
                        take = min(left, arr - held)
                        self.reserved[j] = self.reserved.get(j, 0.0) + take
                        held += take
            act.update({"hold_kw": held, "hold_kind": "clean", "run_clean_kw": self.reserved.pop(i, 0.0)})
            act.setdefault("run_event_kw", self.min_rate(i, "event"))
        return act

    def advance(self) -> dict[str, Any]:
        cur = self.clock.current()
        if self.idx is None or cur < self.idx or cur - self.idx > 96 * 2:
            self.idx = cur - 1                                  # (re)start: no catch-up across days
        while self.idx < cur:
            i = self.idx + 1
            self.recent[i] = self.step(i, self._act_for(i))
            if len(self.recent) > 400:
                for k in sorted(self.recent)[:-384]:
                    self.recent.pop(k, None)
        return self.tel

    def set_allowance(self, start_idx: int, kw: list[float]) -> None:
        for k, v in enumerate(kw):
            if v > 0:
                self.allowance[start_idx + k] = float(v)
        cur = self.idx or 0
        for j in [j for j in self.allowance if j < cur - 96]:   # forget old days
            self.allowance.pop(j, None)


# ---------------------------------------------------------------------------------------------------------- agent
class DataCenterAgent:
    """The data center's own agent: offers, accepts or declines, delivers, reports, follows clean allowances.
    Same Beckn v2 messages and signatures as a home (home_v2.py), so the coordinator's market runs it unchanged."""
    node_type = "datacenter"

    def __init__(self, name: str, cfg: dict[str, Any], site: DataCenterSite, clock, key, send: Callable,
                 trusted: Callable[[], list[str]], log: Callable[..., None], coordinator: str = "coordinator",
                 lock=None):
        self.name, self.cfg, self.site, self.clock, self.key = name, cfg, site, clock, key
        self.pub = signing.public_hex(key)
        self.send, self.trusted, self.log, self.coordinator = send, trusted, log, coordinator
        self.offers: dict[str, dict[str, Any]] = {}
        self.contracts: dict[str, dict[str, Any]] = {}
        self.payments: list[dict[str, Any]] = []
        self.reputation = float(cfg.get("reputation", 1.0))
        self.play = ids.new_play()                           # in every id it sends (ids.py). dc_fleet builds new agents for each run, so each run has its own
        self._fc_now: int | None = None
        self._lock = threading.Lock()
        # the AI agent (brain.DataCenterBrain), attached by flexgrid-dc for the sites in dc_llm_sites only: it decides
        # how much of the safe offer to give and may decline a safe commitment; the code below still decides.
        # cfg think_in_background (flexgrid-dc): the LLM's seconds run in a thread of their own, never inside the
        # message handler, which holds the process's lock (`lock`: taken again to store and send what was decided)
        # and the Town claim (its 30 s lease)
        self.brain = None
        self.think_lock = lock if lock is not None else threading.RLock()
        self._thinking: set[tuple[str, Any]] = set()
        self.closed = False                          # set when flexgrid-dc starts a new run: a late decision is dropped

    def now_idx(self) -> int:
        return self.clock.current()

    def now_ts(self) -> str:
        return self.clock.ts(self.now_idx())

    def observe(self) -> dict[str, Any]:
        return self.site.advance()

    # --- forecasts: history only (the same rule as the homes: only days already seen when the offer is made) ----
    def _past_days(self, i: int, n: int = 3) -> list[int]:
        now = self.now_idx() if self._fc_now is None else self._fc_now
        first = 1
        while i - 96 * first >= now:
            first += 1
        return [d for d in range(first, first + n) if i - 96 * d >= 0]

    def _fc(self, arr: list[float], i: int, n: int = 3) -> float:
        days = self._past_days(i, n)
        return sum(arr[i - 96 * d] for d in days) / len(days) if days else 0.0

    def forecast(self, i: int) -> dict[str, float]:
        s = self.site.s
        f = self._fc(s.flex, i)
        load = self._fc(s.load, i)
        return {"flex": f, "critical": max(0.0, load - f), "pv": self._fc(s.pv, i), "cap": s.cap[i]}

    # --- H3 Offer -------------------------------------------------------------------------------------------
    def build_offer(self, need: dict[str, Any]) -> dict[str, Any]:
        """The offer this site sends: the controller's safe offer, lowered by the AI agent if it has one."""
        offer = self.safe_offer(need)
        if self.brain is not None:                   # the AI agent decides how much of it to offer (never more)
            try:
                offer = self.brain.decide_offer(need, offer)
            except Exception as exc:                 # noqa: BLE001  the safe offer goes out
                self.log("agent_error", where="dc offer", error=f"{type(exc).__name__}: {str(exc)[:200]}")
        self.offers[need["event_id"]] = offer
        return offer

    def safe_offer(self, need: dict[str, Any]) -> dict[str, Any]:
        """The site controller's safe offer: forecast batch work it can pause (none if a job paused at the window start
        would fall due inside it) plus the battery above its backup reserve, and the evening after the window."""
        self._fc_now = self.now_idx()
        site, s = self.site, self.site.s
        win = [(sl["id"], beckn.add_minutes(sl["start"], 15 * k)) for sl in need["slots"] for k in (0, 1)]
        steps = []
        for sid, ts in win:
            i = s.idx_of(ts)
            fc = self.forecast(i) if i is not None else {"flex": 0.0, "critical": 0.0, "pv": 0.0, "cap": 0.0}
            steps.append({"slot_id": sid, "ts": ts, **fc})
        hours = len(win) * STEP_H
        usable = site.usable_kwh(max(site.soc, site.hold))     # held at its normal charge until the window
        batt = min(site.P, usable / hours) if hours > 0 else 0.0
        if site.w < len(win) + 1 or not site.can_pause:        # work paused at the window start would be due inside
            steps = [{**st, "flex": 0.0} for st in steps]      # the window (or the site may not pause): pause nothing
        per: dict[int, list[float]] = {}
        fac_by: dict[str, list[float]] = {}
        for st in steps:
            per.setdefault(st["slot_id"], []).append(st["flex"])
            j = s.idx_of(st["ts"])
            if j is not None and site.levers:                  # the same step on the 3 latest days seen: history only
                days = self._past_days(j)
                for d in days:
                    for k, v in site.facility_kw(j - 96 * d, st["flex"], forecast=True).items():
                        fac_by[k] = fac_by.get(k, []) + [v / len(days)]
        n_st = max(1, len(steps))
        fac_map = {k: round(sum(v) / n_st, 1) for k, v in fac_by.items()}
        fac_kw = round(0.9 * sum(fac_map.values()), 1)         # the same 10 % margin the evening keeps
        slots = []
        for sid, fl in sorted(per.items()):
            pause = site.can_wait(sum(fl) / len(fl))           # less today's urgent work, if any
            kw = pause + batt + fac_kw
            # the market's clearing reads `battery_kw` as the part it may scale down (homes: battery); for a data
            # center all of it is scalable, so the whole offer goes there and `ev_pause_kw` (taken in full) is 0
            slots.append({"id": sid, "kw_offered": round(kw, 1), "battery_kw": round(kw, 1), "ev_pause_kw": 0.0,
                          "compute_pause_kw": round(pause, 1), "dc_battery_kw": round(batt, 1), "facility_kw": fac_kw})
        # evening after the window: no-event import forecast and the rebound of delivering a share of the offer
        w_end = need["window_end"]
        post, room = [], []
        for k in range(max(POST_STEPS, 48)):
            i = s.idx_of(beckn.add_minutes(w_end, 15 * k))
            fc = self.forecast(i) if i is not None else {"flex": 0.0, "critical": 0.0, "pv": 0.0, "cap": 0.0}
            if k < POST_STEPS:
                post.append(round(fc["critical"] + fc["flex"] - fc["pv"], 1))
            room.append(round(max(0.0, 0.9 * fc["cap"] - fc["critical"] - fc["flex"]), 1))   # 10 % margin kept
        paused_kwh = sum(st["flex"] for st in steps) * STEP_H
        n_res = max(1, site.w - len(win) + 1)                  # steps between the window end and the first deadline
        rebound = {}
        for frac in REBOUND_FRACTIONS:
            r = [0.0] * POST_STEPS
            for k in range(min(n_res, POST_STEPS)):
                r[k] += frac * paused_kwh / (n_res * STEP_H)
            refill = frac * batt * hours / site.eta ** 2       # battery refill after the queue is empty
            k = n_res
            while refill > 1e-6 and k < POST_STEPS:
                kw = min(site.P * 0.25, refill / STEP_H)
                r[k] += kw
                refill -= kw * STEP_H
                k += 1
            rebound[str(frac)] = [round(x, 1) for x in r]
        conf = round(0.9 * self.reputation, 3)
        offer = {"event_id": need["event_id"], "home": self.name, "node_type": self.node_type, "slots": slots,
                 "energy_kwh": round(sum(x["kw_offered"] for x in slots) * beckn.SLOT_MINUTES / 60, 1),
                 "confidence": conf, "health": "ok", "price_floor": 0.0,
                 "battery": {"kw": 0.0 if site.battery_out else site.P, "kwh": site.E, "usable_kwh": round(usable, 1),
                             "backup_reserve_pct": round(site.backup * 100),
                             **({"out_of_service": True} if site.battery_out else {})},
                 **({"urgent_work_share": round(site.urgent, 2)} if site.urgent > 0 else {}),
                 "deferral_window_h": site.w * STEP_H, **({"facility": fac_map} if fac_map else {}),
                 "evening": {"start": w_end, "step_min": 15, "import_cf_kw": post, "import_noev_kw": post,
                             "rebound_kw": rebound, "room_kw": room},
                 "ev_flex": None,
                 "forecast_method": "flexible work, facility load and on-site PV: same step on the 3 latest days seen",
                 "valid_until": need["commit_by"], "made_at": self.now_ts()}
        return offer

    def mid(self, to: str, kind: str, *key: Any, asked: str | None = None) -> str:
        """The id of one of this site's messages, as a home's (home_v2.HomeV2.mid)."""
        return ids.sealed(None if asked else self.key, asked or self.play, self.name, to, kind, *key)

    def send_offer(self, txn: str, offer: dict[str, Any], asked: str | None = None) -> dict[str, Any]:
        body = beckn.make("on_discover", sender=self.name, receiver=self.coordinator, transaction_id=txn,
                          message={"offer": offer}, key=self.key, sender_role="bpp")
        out = self.send(self.coordinator, beckn.kind("on_discover"), body, self.mid(self.coordinator, "on_discover", offer["event_id"], asked=asked))
        self.log("v2_offer_sent", event_id=offer["event_id"], kw=[x["kw_offered"] for x in offer["slots"]],
                 confidence=offer["confidence"])
        return {"sent": out, "summary": f"Offer for {offer['event_id']}: {[x['kw_offered'] for x in offer['slots']]} kW"}

    # --- H4 Accept: every commitment is checked against what this site can do safely -------------------------
    def verify_contract(self, sender: str, body: dict[str, Any], ask_agent: bool = True) -> dict[str, Any]:
        """The guard's verdict on a commitment, then (ask_agent) the AI agent's: it may decline a safe one, never accept
        one the guard declined."""
        c = (body.get("message") or {}).get("contract") or {}

        def no(reason: str) -> dict[str, Any]:
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
        mine = {x["id"]: x for x in offer["slots"]}
        for x in c.get("commitments", []):
            if x["kw"] > mine.get(x["id"], {}).get("kw_offered", 0.0) + 0.01:
                return no(f"slot_{x['id']}_above_offer")
        # the battery part may never reach into the backup reserve
        batt_kwh = sum(max(0.0, x["kw"] - mine.get(x["id"], {}).get("compute_pause_kw", 0.0) - mine.get(x["id"], {}).get("facility_kw", 0.0))
                       for x in c.get("commitments", [])) * beckn.SLOT_MINUTES / 60
        if batt_kwh > 1e-6 and self.site.battery_out:
            return no("battery_out_of_service")
        if batt_kwh > self.site.usable_kwh(max(self.site.soc, self.site.hold)) * 1.01 + 1e-6:
            return no("would_use_backup_reserve")
        # only work that is paused has to outlast the event: an offer of levers and stored energy has no deadline to meet
        paused = any(mine.get(x["id"], {}).get("compute_pause_kw", 0.0) > 0.0 for x in c.get("commitments", []))
        if paused and self.site.w < 2 * len(c.get("commitments", [])) + 1:
            return no("deferral_window_shorter_than_event")
        if c.get("window_start", "")[:10] in (self.cfg.get("opt_out_days") or []):
            return no("operator_opted_out")
        if self.cfg.get("control", "auto") == "advisory":
            return no("operator_approval_needed")
        verdict = {"decision": "ACCEPTED", "reason": "within_offer_and_safe", "contract_id": c.get("id")}
        if self.brain is not None and ask_agent:     # the guard said yes: the AI agent may still say no (never yes to a no)
            prev = self.contracts.get(c.get("id"))
            if not (prev and prev.get("status") in ("ACCEPTED", "DECLINED")):
                try:
                    out = self.brain.decide_accept(c, verdict)
                    if out.get("decision") == "DECLINED":
                        verdict = out
                except Exception as exc:             # noqa: BLE001
                    self.log("agent_error", where="dc accept", error=f"{type(exc).__name__}: {str(exc)[:200]}")
        return verdict

    def install_plan(self, c: dict[str, Any], verdict: dict[str, Any]) -> dict[str, Any]:
        prev = self.contracts.get(c.get("id"))
        if prev and prev.get("status") in ("ACCEPTED", "DECLINED"):
            return {"installed": prev["status"] == "ACCEPTED", "repeat": True}
        if verdict["decision"] != "ACCEPTED":
            return {"installed": False}
        actions = {}
        for x in c["commitments"]:
            for k in (0, 1):
                actions[beckn.add_minutes(x["start"], 15 * k)] = {"deliver_kw": float(x["kw"]), "contract_id": c["id"]}
        self.site.plan.update(actions)
        rp = c.get("resume_plan") or {}
        i_end = self.site.s.idx_of(c["window_end"])
        if rp and i_end is not None:                           # when the paused work comes back (coordinator's plan)
            plan = {}
            for ts, kw in rp.items():
                j = self.site.s.idx_of(ts)
                if j is not None and j >= i_end:
                    plan[j] = plan.get(j, 0.0) + float(kw)
            self.site.resumes.append({"from": i_end, "until": i_end + self.site.w,      # every paused job's deadline
                                      "plan": plan, "left": sum(plan.values()), "contract_id": c["id"]})
        self.contracts[c["id"]] = {**c, "status": "ACCEPTED", "reported": [], "accepted_at": self.now_ts()}
        return {"installed": True, "steps": len(actions)}

    def answer(self, txn: str, c: dict[str, Any], verdict: dict[str, Any], asked: str | None = None) -> dict[str, Any]:
        prev = self.contracts.get(c.get("id")) if c.get("id") else None
        if prev and prev.get("status") in ("ACCEPTED", "DECLINED"):
            verdict = {"decision": prev["status"], "reason": prev.get("reason", "within_offer_and_safe"),
                       "contract_id": c.get("id")}
        elif verdict["decision"] != "ACCEPTED" and c.get("id"):
            self.contracts[c["id"]] = {**c, "status": "DECLINED", "reason": verdict["reason"], "reported": []}
        body = beckn.make("on_confirm", sender=self.name, receiver=self.coordinator, transaction_id=txn,
                          message={"contract": {"id": c.get("id"), "event_id": c.get("event_id"), "home": self.name,
                                                "status": verdict["decision"], "reason": verdict["reason"]}},
                          key=self.key, sender_role="bpp")
        out = self.send(self.coordinator, beckn.kind("on_confirm"), body, self.mid(self.coordinator, "on_confirm", c.get("id"), asked=asked))
        self.log("v2_contract_" + verdict["decision"].lower(), contract_id=c.get("id"), reason=verdict["reason"])
        return {"sent": out, "summary": f"Contract {c.get('id')}: {verdict['decision']} ({verdict['reason']})."}

    # --- H5 Report ------------------------------------------------------------------------------------------
    def due_readings(self) -> list[tuple[str, int]]:
        cur, due = self.now_idx(), []
        for cid, c in self.contracts.items():
            if c.get("status") != "ACCEPTED":
                continue
            for x in c["commitments"]:
                i0 = self.site.s.idx_of(x["start"])
                if i0 is not None and cur > i0 + 1 and x["id"] not in c["reported"] \
                        and x["id"] not in c.get("in_flight", ()) \
                        and i0 in self.site.recent and i0 + 1 in self.site.recent:
                    due.append((cid, x["id"]))
        return due

    def build_reading(self, cid: str, slot_id: int) -> dict[str, Any]:
        c = self.contracts[cid]
        x = [y for y in c["commitments"] if y["id"] == slot_id][0]
        i0 = self.site.s.idx_of(x["start"])
        t0, t1 = self.site.recent[i0], self.site.recent[i0 + 1]
        usage = (t0["import_kw"] + t1["import_kw"]) / 2
        usage_cf = (t0["import_cf_kw"] + t1["import_cf_kw"]) / 2
        return {"contract_id": cid, "event_id": c["event_id"], "home": self.name, "node_type": self.node_type,
                "slot_id": slot_id, "start": x["start"], "committed_kw": x["kw"], "usage_kw": round(usage, 2),
                "soc_end_pct": t1["soc_pct"], "modes": [t0["mode"], t1["mode"]],
                "queue_kwh": t1["queue_kwh"],
                "truth": {"usage_without_event_kw": round(usage_cf, 2), "delivered_true_kw": round(usage_cf - usage, 2)}}

    def send_reading(self, txn: str, reading: dict[str, Any]) -> dict[str, Any]:
        body = beckn.make("on_status", sender=self.name, receiver=self.coordinator, transaction_id=txn,
                          message={"performance": reading}, key=self.key, sender_role="bpp",
                          ttl_s=beckn.TTL_EVIDENCE_S)
        out = self.send(self.coordinator, beckn.kind("on_status"), body, self.mid(self.coordinator, "on_status", reading["contract_id"], reading["slot_id"]))
        util = self.cfg.get("utility_name", "grid-operator")
        meter = {k: v for k, v in reading.items() if k != "truth"}
        ubody = beckn.make("on_status", sender=self.name, receiver=util, transaction_id=txn,
                           message={"performance": meter}, key=self.key, sender_role="bpp",
                               ttl_s=beckn.TTL_EVIDENCE_S)
        uout = self.send(util, beckn.kind("on_status"), ubody, self.mid(util, "on_status", reading["contract_id"], reading["slot_id"]))
        self._mark_reported(reading["contract_id"], reading["slot_id"], [out, uout])
        return {"sent": out, "summary": f"Slot {reading['slot_id']}: committed {reading['committed_kw']} kW, "
                                        f"meter {reading['usage_kw']} kW"}

    def _mark_reported(self, cid: str, sid: int, results: list[Any]) -> None:
        """A slot counts as reported once both sends went through (as home_v2.py). In one process the sends are done
        when they return; on NANDA Town (flexgrid-dc) they go out in the background and hand back futures: the slot
        stays in flight until both finish, and is due again if one of them failed."""
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

    # --- the AI agent, in the background (flexgrid-dc) ----------------------------------------------------------
    def _think(self, key: tuple[str, Any], work: Callable[[], None]) -> bool:
        """Runs `work` (the AI agent's decision, then storing and sending it) in a thread of its own, once per key at a
        time. False if that decision is already being made (the desk sent its message again meanwhile)."""
        with self._lock:
            if key in self._thinking:
                return False
            self._thinking.add(key)

        def run() -> None:
            try:
                work()
            except Exception as exc:                 # noqa: BLE001
                self.log("agent_error", where=f"dc {key[0]} (background)", error=f"{type(exc).__name__}: {str(exc)[:200]}")
            finally:
                with self._lock:
                    self._thinking.discard(key)
        threading.Thread(target=run, name=f"{self.name}-agent-{key[0]}", daemon=True).start()
        return True

    def thinking(self) -> list[str]:
        with self._lock:
            return sorted(str(k[0]) for k in self._thinking)

    def _offer_in_background(self, txn: str, need: dict[str, Any], asked: str | None = None) -> str:
        eid = need["event_id"]
        if eid in self.offers:                       # decided already (the call for offers came again): the same offer
            return self.send_offer(txn, self.offers[eid], asked)["summary"]
        safe = self.safe_offer(need)

        def work() -> None:
            try:
                offer = self.brain.decide_offer(need, safe)
            except Exception as exc:                 # noqa: BLE001  the safe offer goes out
                self.log("agent_error", where="dc offer", error=f"{type(exc).__name__}: {str(exc)[:200]}")
                offer = safe
            with self.think_lock:
                if self.closed or eid in self.offers:
                    return
                self.offers[eid] = offer
                self.send_offer(txn, offer, asked)
        if self._think(("offer", eid), work):
            return f"the AI agent decides how much of its safe offer ({safe['energy_kwh']:.0f} kWh) to give"
        return "the AI agent is still deciding its offer"

    def _accept_in_background(self, txn: str, c: dict[str, Any], sender: str, body: dict[str, Any]) -> str:
        v = self.verify_contract(sender, body, ask_agent=False)
        asked = ids.asked(body)
        if v["decision"] != "ACCEPTED":              # the guard said no: the agent is not asked
            self.install_plan(c, v)
            return self.answer(txn, c, v, asked)["summary"]

        def work() -> None:
            try:
                out = self.brain.decide_accept(c, v)
            except Exception as exc:                 # noqa: BLE001  the guard's yes stands
                self.log("agent_error", where="dc accept", error=f"{type(exc).__name__}: {str(exc)[:200]}")
                out = v
            verdict = out if out.get("decision") == "DECLINED" else v
            with self.think_lock:
                prev = self.contracts.get(c.get("id"))
                if self.closed or (prev and prev.get("status") in ("ACCEPTED", "DECLINED")):
                    return
                if self.now_ts() >= c.get("opt_out_by", "9999"):
                    verdict = {"decision": "DECLINED", "reason": "past_opt_out_deadline", "contract_id": c.get("id")}
                self.install_plan(c, verdict)
                self.answer(txn, c, verdict, asked)
        if self._think(("accept", c.get("id")), work):
            return "the guard found it safe: the AI agent decides whether to accept"
        return "the AI agent is still deciding whether to accept"

    # --- inbox ----------------------------------------------------------------------------------------------
    def gate(self, kind: str, sender: str, body: dict[str, Any]) -> str | None:
        if sender != self.coordinator or body.get("signer") not in self.trusted() or \
                not beckn.check(body, body.get("signer", "")):
            self.log("v2_rejected_message", message_kind=kind, sender=sender)
            return beckn.Refused("rejected: not from a trusted coordinator")
        return None

    def handle(self, kind: str, sender: str, body: dict[str, Any]) -> str:
        act = beckn.action_of(kind)
        txn = body.get("context", {}).get("transactionId", "")
        asked = ids.asked(body)
        late = beckn.expired(body, default_ttl=float(self.cfg.get("max_message_age_s", 900)))
        if late:
            return beckn.Refused(f"expired {round(late)} s ago: not acted on")
        rejected = self.gate(kind, sender, body)
        if rejected:
            return rejected
        m = body.get("message") or {}
        background = self.brain is not None and bool(self.cfg.get("think_in_background"))
        if act == "discover":
            if background:
                return self._offer_in_background(txn, m["intent"]["need"], asked)
            return self.send_offer(txn, self.build_offer(m["intent"]["need"]), asked)["summary"]
        if act == "confirm":
            c = {**m["contract"], "transaction_id": txn}
            prev = self.contracts.get(c.get("id"))
            if prev and prev.get("status") in ("ACCEPTED", "DECLINED"):
                v = {"decision": prev["status"], "reason": prev.get("reason", "within_offer_and_safe"),
                     "contract_id": c.get("id")}
                return self.answer(txn, c, v, asked)["summary"]
            if background:
                return self._accept_in_background(txn, c, sender, body)
            v = self.verify_contract(sender, body)
            self.install_plan(c, v)
            return self.answer(txn, c, v, asked)["summary"]
        if act == "cancel":                                 # the desk froze the deal without this site's late yes
            cid = (m.get("contract") or {}).get("id")
            c = self.contracts.get(cid)
            if not c:
                return "cancel for a contract this data center does not hold: nothing to undo"
            if c.get("status") == "ACCEPTED":
                self.site.plan = {ts: a for ts, a in self.site.plan.items() if a.get("contract_id") != cid}
                self.site.resumes = [r for r in self.site.resumes if r.get("contract_id") != cid]
                c["status"], c["reason"] = "CANCELLED", (m.get("contract") or {}).get("reason")
                self.log("v2_contract_cancelled", contract_id=cid, reason=c["reason"])
            return f"contract {cid} cancelled"
        if act == "rt_ask":                                 # the desk asks, during the day: how much more?
            return self.rt_answer(txn, m["ask"], asked)
        if act == "rt_dispatch":
            return self.rt_take(txn, m["dispatch"])
        if act == "update" and "clean_allowance" in m:
            a = m["clean_allowance"]
            i0 = self.site.s.idx_of(a["start"])
            if i0 is None:
                return beckn.Refused("clean allowance ignored: unknown start")
            self.site.set_allowance(i0, [max(0.0, float(v)) for v in a["kw"]])
            self.log("dc_clean_allowance", start=a["start"], kwh=round(sum(a["kw"]) * STEP_H, 1))
            return f"clean allowance for {a['start'][:10]}: {round(sum(a['kw']) * STEP_H)} kWh"
        if act == "update":
            self.log("v2_event_notice", **{k: v for k, v in m.items() if k != "contract"})
            return "event notice noted"
        if act == "on_status" and "settlement" in m:
            st = m["settlement"]
            if any(p.get("event_id") == st.get("event_id") for p in self.payments):
                return f"payment for {st.get('event_id')} already noted"
            self.payments.append(st)
            self.reputation = float(st.get("reputation", self.reputation))
            self.log("v2_paid", event_id=st.get("event_id"), usd=st.get("payment"), delivered_kwh=st.get("delivered_kwh"))
            return f"payment noted: {st.get('payment')}"
        return beckn.Refused(f"ignored {kind}")

    # --- the real-time design (rt_market.py): calls and gaps during the day, as the homes ----------------------
    def _rt_steps(self, start: str, end: str) -> list[int]:
        i0, i1 = self.site.s.idx_of(start), self.site.s.idx_of(end)
        return list(range(i0, i1)) if i0 is not None and i1 is not None else []

    def _levers_at(self, i: int, paused_kw: float) -> float:
        """kW the facility levers can give at step i: the step itself if it is here, else the same step on the 3 latest days
        seen, with the offer's 10 % margin."""
        s = self.site
        if not s.levers:
            return 0.0
        if i <= (s.idx or 0) + 1:
            return sum(s.facility_kw(i, paused_kw).values())
        days = self._past_days(i)
        return 0.9 * sum(sum(s.facility_kw(i - 96 * d, paused_kw).values()) for d in days) / max(1, len(days))

    def _cover(self, steps: list[int], extra: float) -> list[float]:
        """Per step, what the site would be short if it owed its plan plus `extra` kW: batch work it can hold first
        (the forecast arrivals, less urgent work), then its facility levers, the battery above its backup reserve for the rest."""
        s = self.site
        e = s.usable_kwh()
        out = []
        for i in steps:
            ts = s.s.ts[i]
            owe = float((s.plan.get(ts) or {}).get("deliver_kw", 0.0)) + extra
            arr = s.s.flex[i] if i <= (s.idx or 0) + 1 else self.forecast(i)["flex"]
            hold = min(s.can_wait(arr, i), owe)
            need = max(0.0, owe - hold - self._levers_at(i, hold))
            b = min(need, s.P if not s.battery_out else 0.0, e / STEP_H)
            e -= b * STEP_H
            out.append(max(0.0, need - b))
        return out

    def rt_capacity(self, start: str, end: str) -> dict[str, Any]:
        steps = [i for i in self._rt_steps(start, end) if i > (self.site.idx or 0)]
        if not steps or sum(self._cover(steps, 0.0)) > 0.5:
            return {"kw": 0.0, "why": "nothing to spare: it is short of what it owes already" if steps else "outside the day"}
        lo, hi = 0.0, self.site.P + max(self.site.s.flex[i] for i in steps) + max(self._levers_at(i, 0.0) for i in steps)
        for _ in range(18):                                   # the most it can add evenly with nothing short
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if sum(self._cover(steps, mid)) < 1e-3 else (lo, mid)
        return {"kw": round(lo * 0.95, 1), "why": "batch work it can hold and battery above its backup reserve"}

    def rt_answer(self, txn: str, ask: dict[str, Any], asked: str | None = None) -> str:
        cap = self.rt_capacity(ask["start"], ask["end"])
        body = beckn.make("rt_answer", sender=self.name, receiver=self.coordinator, transaction_id=txn,
                          message={"answer": {"id": ask["id"], "kw": cap["kw"], "why": cap.get("why")}},
                          key=self.key, sender_role="bpp")
        self.send(self.coordinator, beckn.kind("rt_answer"), body, self.mid(self.coordinator, "rt_answer", ask["id"], asked=asked))
        self.log("rt_answer", call_id=ask["id"], kw=cap["kw"], why=cap.get("why"))
        return f"can add {cap['kw']} kW for {ask['id']}"

    def rt_take(self, txn: str, d: dict[str, Any]) -> str:
        """Owe the dispatched kW on top: more batch work held and more battery in those steps. Covering another site's
        gap in the event, it is owed under the event (its readings count there); a call: under the call's id."""
        s = self.site
        for ts, kw in d["steps"].items():
            a = s.plan.setdefault(ts, {"deliver_kw": 0.0, "contract_id": d["id"]})
            a["deliver_kw"] = round(max(0.0, float(a.get("deliver_kw", 0.0)) + float(kw)), 2)
        halves: dict[str, float] = {}
        for ts, kw in d["steps"].items():
            h = ts[:14] + ("00" if ts[14:16] < "30" else "30")
            halves[h] = halves.get(h, 0.0) + float(kw) / 2.0
        ev_c = next((c for c in self.contracts.values() if d.get("event_id") and c.get("event_id") == d["event_id"]
                     and c.get("status") == "ACCEPTED" and not c.get("rt")), None)
        if ev_c is not None:
            for h, kw in halves.items():
                sid = int(beckn_minutes(ev_c["window_start"], h) // 30)
                if sid in ev_c.get("reported", []):
                    continue
                cm = next((x for x in ev_c["commitments"] if x["id"] == sid), None)
                if cm is None:
                    ev_c["commitments"].append({"id": sid, "start": h, "kw": round(kw, 2), "gap": True})
                else:
                    cm["kw"] = round(cm["kw"] + kw, 2)
            ev_c["commitments"].sort(key=lambda x: x["id"])
        else:
            cid = f"{d['id']}:{self.name}"
            c = self.contracts.setdefault(cid, {"id": cid, "event_id": d["id"], "home": self.name, "status": "ACCEPTED",
                                                "commitments": [], "reported": [], "rt": True, "transaction_id": txn})
            have = {x["start"]: x for x in c["commitments"]}
            for h, kw in sorted(halves.items()):
                if h in have:
                    if have[h]["id"] not in c["reported"]:
                        have[h]["kw"] = round(have[h]["kw"] + kw, 2)
                else:
                    c["commitments"].append({"id": len(c["commitments"]), "start": h, "kw": round(kw, 2)})
        self.log("rt_dispatch", call_id=d["id"], call_kind=d.get("kind"), steps=len(d["steps"]),
                 kw=round(max(d["steps"].values(), default=0.0), 2))
        return f"dispatch {d['id']}: {len(d['steps'])} steps"

    def rt_report(self) -> None:
        """The site will fall short of what it owes (an urgent job, its battery out): tell the desk, once per change."""
        s = self.site
        cur = s.idx if s.idx is not None else self.now_idx()
        steps = [j for j in range(cur + 1, cur + 17) if (s.plan.get(s.s.ts[j]) or {}).get("deliver_kw", 0.0) > 0]
        short = {s.s.ts[j]: round(v, 1) for j, v in zip(steps, self._cover(steps, 0.0)) if v > 50.0}
        prev = getattr(self, "_rt_reported", {})
        if not short or (set(short) == set(prev) and all(abs(short[t] - prev[t]) < 50.0 for t in short)):
            return
        reason = "an urgent job that cannot wait" if s.urgent_jobs else ("its battery is out" if s.battery_out else
                                                                          "less batch work to hold than forecast")
        msg = {"short": short, "reason": reason}
        body = beckn.make("rt_capacity", sender=self.name, receiver=self.coordinator, transaction_id=f"cap-{self.name}", message=msg, key=self.key, sender_role="bpp")
        self.send(self.coordinator, beckn.kind("rt_capacity"), body, self.mid(self.coordinator, "rt_capacity", self.now_ts(), ids.digest(msg)))
        self._rt_reported = short
        self.log("rt_capacity", short_kw=max(short.values()), steps=len(short), reason=reason)

    def urgent_job(self, kw: float, hours: float, at: str | None = None) -> dict[str, Any]:
        """The operator's urgent job: kw of this site's batch work can no longer wait, for `hours` from `at` (now)."""
        i0 = self.site.s.idx_of(at) if at else self.now_idx()
        job = {"from": i0, "to": i0 + int(round(hours * 4)), "kw": float(kw)}
        self.site.urgent_jobs.append(job)
        self.log("dc_urgent_job", kw=round(kw, 1), hours=hours, start=self.site.s.ts[i0])
        return {**job, "start": self.site.s.ts[i0], "end": self.site.s.ts[min(job["to"], len(self.site.s.ts) - 1)]}

    def tick(self) -> list[str]:
        self.observe()
        try:
            self.rt_report()
        except Exception as exc:                              # noqa: BLE001  never let a report stop the readings
            self.log("rt_report_error", error=str(exc)[:160])
        out = []
        for cid, sid in self.due_readings():
            out.append(self.send_reading(self.contracts[cid].get("transaction_id", ""), self.build_reading(cid, sid))["summary"])
        return out

    def status(self) -> dict[str, Any]:
        t = self.site.tel or {}
        return {"node": self.name, "type": self.node_type, "now": self.now_ts(), "import_kw": t.get("import_kw"),
                "queue_kwh": t.get("queue_kwh"), "soc_pct": t.get("soc_pct"), "mode": t.get("mode"),
                "offers": len(self.offers), "contracts": {k: v.get("status") for k, v in self.contracts.items()},
                "paid_usd": round(sum(float(p.get("payment") or 0) for p in self.payments), 2),
                "late_kwh": round(self.site.late_kwh, 3), "reputation": self.reputation, "time": time.time()}


# ---------------------------------------------------------------------------------------------------------- a day on real prices
def price_plan(s: "Series", i0: int, n: int, w: int, price_mwh: list[float], margin: float = 0.10,
               min_gain: float = 5.0) -> dict[str, Any]:
    """The site's own plan for one day of real prices: when to run its deferrable work so it costs least, inside every
    deadline and under its capacity. Work arriving at step t may run in any step of [t, t + w] that is still in the day
    (the day ahead's prices are all it knows; nothing is pushed past midnight); critical load never moves; every step
    keeps `margin` of the site's capacity free. Each arriving piece goes, in arrival order (so earliest deadline first),
    to the cheapest steps of its window that still have room and are at least `min_gain` $/MWh cheaper (work is not
    moved for pennies); a piece with no such room runs on arrival, as it would without a plan. Returns the per-step load with and without the plan and what it changes, at `price_mwh` ($/MWh per
    15-minute step). The battery is left at its normal charge (it is kept for the grid's events and for backup)."""
    idx = [i0 + k for k in range(n)]
    arr = [s.flex[i] for i in idx]
    crit = [s.load[i] - s.flex[i] for i in idx]
    pv = [s.pv[i] for i in idx]
    # room for moved work: the capacity (less the margin) above critical load and the step's own arriving work, which
    # keeps its place unless it moves itself (so a step that takes moved work never pushes its own work out)
    spare = [max(0.0, (1 - margin) * s.cap[i] - c - a) for i, c, a in zip(idx, crit, arr)]
    run = [0.0] * n
    moves = []                                                     # (from step, to step, kW for one step)
    for t in range(n):
        left = arr[t]
        cand = sorted(range(t + 1, min(n, t + w + 1)), key=lambda j: (price_mwh[j], j))
        for j in cand:
            if left <= 1e-9 or price_mwh[j] > price_mwh[t] - min_gain:   # only to a clearly cheaper step
                break
            x = min(left, spare[j])
            if x > 1e-9:
                run[j] += x
                spare[j] -= x
                left -= x
                moves.append((t, j, x))
        run[t] += left                                             # what found no cheaper room runs on arrival
    over = max(0.0, max(run[k] - (s.cap[i] - crit[k]) for k, i in enumerate(idx)))
    imp = [crit[k] + run[k] - pv[k] for k in range(n)]
    imp0 = [crit[k] + arr[k] - pv[k] for k in range(n)]
    cost = sum(max(0.0, x) * p for x, p in zip(imp, price_mwh)) * STEP_H / 1000
    cost0 = sum(max(0.0, x) * p for x, p in zip(imp0, price_mwh)) * STEP_H / 1000
    return {"import_kw": [round(x, 1) for x in imp], "import_cf_kw": [round(x, 1) for x in imp0],
            "moved_kwh": round(sum(m[2] for m in moves) * STEP_H, 1),
            "moves": moves, "cost_usd": round(cost, 2), "cost_cf_usd": round(cost0, 2),
            "flex_kwh": round(sum(arr) * STEP_H, 1), "late_kwh": 0.0, "over_capacity_kw": round(over, 3),
            "max_wait_steps": max((m[1] - m[0] for m in moves), default=0)}


# ---------------------------------------------------------------------------------------------------------- coordinator side
def region_forecast(ts_list: list[str], made_at: str, days_back: int = 3) -> list[float]:
    """The coordinator's forecast of the region's load (MW) for these steps, made at `made_at`: the same step on the
    3 latest days already seen (history only; a real coordinator would use the grid operator's published forecast)."""
    r = region()
    now = r["index"].get(made_at, 0)
    out = []
    for t in ts_list:
        i = r["index"].get(t)
        if i is None:
            out.append(out[-1] if out else 0.0)
            continue
        first = 1
        while i - 96 * first >= now:
            first += 1
        ds = [d for d in range(first, first + days_back) if i - 96 * d >= 0]
        out.append(sum(r["system_load_mw"][i - 96 * d] for d in ds) / len(ds) if ds else r["system_load_mw"][i])
    return out


def clean_allowance(day: str, made_idx: int, sites: list[DataCenterSite], solar: list[float],
                    days_back: int = 3) -> dict[str, list[float]]:
    """The coordinator's day-ahead clean allowance for `day`, made at step `made_idx` (10:00 the day before):
    community solar the data centers would leave unused = forecast solar minus the forecast of the sites' own
    metered import, per 15-minute step, shared out in proportion to each site's forecast flexible work that day.
    Forecasts: the same step on the 3 latest days already seen (seasonal naive, history only)."""
    s0 = sites[0].s
    i0 = s0.idx_of(f"{day} 00:00")
    if i0 is None:
        return {}
    i1 = s0.idx_of(f"{beckn.add_minutes(f'{day} 00:00', 1440)[:10]} 00:00")
    n = (i1 - i0) if i1 else 96                                  # 92 / 100 steps on the clock-change days

    def past(i: int) -> list[int]:
        first = 1
        while i - 96 * first >= made_idx:
            first += 1
        return [d for d in range(first, first + days_back) if i - 96 * d >= 0]

    def fc(arr: list[float], i: int) -> float:
        ds = past(i)
        return sum(arr[i - 96 * d] for d in ds) / len(ds) if ds else 0.0
    surplus, flex = [], {x.id: 0.0 for x in sites}
    for k in range(n):
        i = i0 + k
        imp = sum(fc(x.s.load, i) - fc(x.s.pv, i) for x in sites)
        surplus.append(max(0.0, fc(solar, i) - imp))
        for x in sites:
            flex[x.id] += fc(x.s.flex, i)
    tot = sum(flex.values()) or 1.0
    return {x.id: [round(v * flex[x.id] / tot, 1) for v in surplus] for x in sites}
