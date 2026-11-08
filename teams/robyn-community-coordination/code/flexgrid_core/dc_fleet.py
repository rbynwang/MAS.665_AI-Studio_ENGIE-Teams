"""flexgrid-dc - the data centers of the large-load program, live: ten data center agents, the coordinator's
large-load desk and the grid operator, on the same replay clock as the homes.

The parties (status API on 127.0.0.1:7874 wherever this runs):
  ten data center agents  datacenter.py (DataCenterAgent + DataCenterSite), one per site in data/datacenters/, each
                          with its own Ed25519 key. Each one offers from its own history, checks every commitment
                          against what its site can do safely, pauses batch work and draws its battery (never below
                          the backup reserve) in the window, runs the paused work again before its deadline, and sends
                          its signed meter readings to the grid operator.
  the large-load desk     the coordinator's market for data centers (dc_market.py: the household market plus a resume
                          plan for the paused work), signing with the coordinator's own key. Every day at 10:00 it
                          also sends each data center its clean allowance for the next day (community solar it
                          forecasts the data centers would otherwise leave unused); not for an event day.
  the grid operator       grid_operator.py: the stand-in buyer of large-load flexibility (the utility's code with its
                          own key, the data centers' meters, the Rider PTR baseline from each site's history).
Every message is a signed Beckn body that its receiver checks, exactly as between the homes and the coordinator.

Where they run (`dc_role` in agent.json):
  all     (the default) all twelve parties in this process, as on the Mac until 29 Oct 2026; the keys are pinned at
          enrollment (this process creates the agents)
  sites   the data center campus (jetson-a): the ten data center agents and their sites only. Their keys are made
          here and never leave. It joins NANDA Town only once it trusts a desk: the Mac's flexgrid-dc (market role)
          hands over the desk's and the grid operator's public keys through its SSH tunnel (POST /dc/trust), and the
          first desk key is pinned (another one is refused until ~/.config/flexgrid/dc/campus_trust.json is removed)
  market  the Mac's side once the campus runs elsewhere: the desk and the grid operator only. It reaches the campus at
          `dc_campus_url` (mac/start_mac.sh opens 127.0.0.1:7884 -> jetson-a 127.0.0.1:7874) for three things that
          are not market messages: the trust handover above, which returns the data centers' public keys (pinned on
          first sight in ~/.config/flexgrid/dc/campus_pins.json; a changed key is not accepted and says so), the
          sites' telemetry for the Control Room (simulation truth, never read by the desk or the grid operator), and
          the demo clock. Offers, contracts, meter readings and payments go between the machines only over NANDA Town.
POST /dc/role switches the Mac between all and market at run time (between events), and says so everywhere.

Transport: NANDA Town, once each party here holds a Town identity (mac/town_add_datacenters.sh adds the twelve to the
team's run; their join tokens live in ~/.config/flexgrid/secrets/dc_town_tokens.json on the machine that runs them).
Each party sends and claims through its own Town session, checking its inbox with short polls (town.InboxPoller: no
held long-poll); the desk has its own name because a second session under "coordinator" would end the coordinator's
(it still signs with the coordinator's key). In the all role without tokens, or with `dc_transport: "memory"`
(tools/dc_timeline.py, the tests), an in-memory queue in this process stands in for Town, and every surface says so.
The campus and the market roles have no in-memory stand-in: across two machines there is only Town.

What else is a stand-in, and every surface says so:
  * the sites: simulated from the team dataset's 2023 series (facility load, deferrable work, on-site PV, battery),
    like the 98 fleet homes. The agents decide from what a site's own controller would know (its history, its state)
Clock: the replay clock in agent.json, like every agent. The coordinator's /demo/start (and tools/demo_event.py) moves
it through /demo/reset and /demo/clock (the market role passes both on to the campus), so a demo event runs the homes
and the data centers on the same steps. A reset waits for the new clock before the next step; a clock that goes back
starts a fresh run; one that goes on (a change of speed) keeps the run, and every site still plays each 15-minute step
exactly once.

GET /dc/state is the record the Control Room reads (tools/dc_timeline.py builds the same record offline with this
class on a virtual clock): every step since the clock was set, per site and in total, against each site's no-event
twin. GET /verify/{event_id} runs flexgrid_core/verify.py, the independent check of the homes' events, on a settled
data center event. GET /dc/today is the one part on today's real grid: each agent's plan for its batch work on MISO's
day-ahead prices for today (advice; the coordinator's GET /grid/now passes it to the Control Room).

  flexgrid-dc run          # python -m flexgrid_core.dc_fleet run
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from collections import Counter, OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any, Callable

try:            # at module level: the routes' annotations are strings (postponed), resolved in this module's globals,
    from fastapi import Request     # so a Request imported inside build_app would read as a query parameter
except ImportError:                 # pragma: no cover
    Request = Any

import httpx

from . import beckn, config, doors, ids, signing, wire
from . import datacenter as dc
from .dc_market import DCMarket
from .replay import Replay
from .town import InboxPoller, TownSession, _once_more

log = logging.getLogger("flexgrid-dc")
TRANSPORT = ("in-memory stand-in for NANDA Town (flexgrid-dc, one process: the data centers and the grid operator "
             "have no Town identities yet)")
TOKENS_FILE = "dc_town_tokens.json"          # ~/.config/flexgrid/secrets: party -> join token (town_add_datacenters.sh)
TRUST_FILE = "campus_trust.json"             # ~/.config/flexgrid/dc on the campus: the desk it trusts (first one wins)
PINS_FILE = "campus_pins.json"               # ~/.config/flexgrid/dc on the Mac: the campus's data center keys
ROLES = ("all", "sites", "market")
SITE_KEYS = ("import_kw", "import_cf_kw", "queue_kwh", "soc_pct", "battery_kw", "held_kw")
MODES = ["default", "hold", "run_held", "deliver", "dispatch"]       # DataCenterSite.step()
RECORD_STEPS = 192                                                   # the record: at most the last 2 days
SLOT_H = beckn.SLOT_MINUTES / 60
HOW = {
    "requested": "The grid operator's large-load need: kW asked for in each 30-minute slot of the window, paid per kWh "
                 "delivered at the window's day-ahead price (the energy the grid did not have to buy).",
    "capacity": "What the coordinator sold: the ten offers weighted by each data center's confidence, capped at the "
                "request, less the 10 % margin it keeps.",
    "delivered_true": "Each data center's no-event twin (the same site with every job run on arrival and its battery "
                      "idle) minus its meter, over the window. Simulation truth: the grid operator never sees it.",
    "delivered_measured": "Ameren Rider PTR baseline (the 3 highest of the 14 previous non-holiday, non-event days) minus "
                          "the meter: what the grid operator pays on, capped at the capacity sold.",
    "late": "Batch work that reached its deadline with no room left under the site's capacity (it would be late in "
            "reality). Every paused job must run before its deadline: this must stay 0.",
    "region": "The region's load in the dataset (all its agents included) with the data centers as they ran, against "
              "the same region with the data centers' no-event twins.",
    "clean": "Community solar the coordinator forecasts the data centers would otherwise leave unused tomorrow (the "
             "same step on the 3 latest days seen), shared out by their deferrable work: each site holds batch work "
             "for those hours, inside its own deadlines.",
}


def _stats(v: list[float], nd: int) -> dict[str, Any]:
    """n, median, 95th percentile and max of a list (rounded to nd decimals)."""
    if not v:
        return {"n": 0}
    v = sorted(v)
    q = lambda p: v[min(len(v) - 1, int(round(p * (len(v) - 1))))]          # noqa: E731
    return {"n": len(v), "p50": round(q(0.5), nd), "p95": round(q(0.95), nd), "max": round(v[-1], nd)}


def _hex64(k: Any) -> bool:
    try:
        return isinstance(k, str) and len(k) == 64 and int(k, 16) >= 0
    except ValueError:
        return False


def _offer_ok(o: Any) -> bool:
    """An offer the market can clear: a list of slots, each with an id and a non-negative number of kW."""
    try:
        return isinstance(o, dict) and isinstance(o.get("slots"), list) and bool(o["slots"]) and all(
            isinstance(x, dict) and "id" in x and float(x.get("kw_offered")) >= 0 and float(x.get("battery_kw", 0)) >= 0
            and float(x.get("ev_pause_kw", 0)) >= 0 for x in o["slots"]) and isinstance(o.get("energy_kwh"), (int, float))
    except (TypeError, ValueError):
        return False


def _sum_counts(a: dict[str, int] | None, b: dict[str, int] | None) -> dict[str, int]:
    out = Counter(a or {})
    out.update(b or {})
    return dict(out)


class _Session:
    """What an agent's NANDA Town session would do here: send() queues for DCFleet.deliver()."""
    label = TRANSPORT

    def __init__(self, name: str, fleet: "DCFleet"):
        self.name, self.fleet = name, fleet

    def send(self, to: str, kind: str, body: dict[str, Any], message_id: str) -> dict[str, Any]:
        self.fleet.bus.append((self.name, to, kind, body))
        return {"message_id": message_id}

    def ack(self, *a, **k) -> dict[str, Any]:
        return {}


class _TownSender:
    """The grid operator's Town session as utility.py uses it (it only sends; flexgrid-dc claims its inbox), with the
    fleet's retries."""

    def __init__(self, fleet: "DCFleet", town: TownSession):
        self.fleet, self.town, self.name = fleet, town, town.name

    def send(self, to: str, kind: str, body: dict[str, Any], message_id: str) -> dict[str, Any]:
        return self.fleet._town_send(self.town, to, kind, body, message_id)


class _NoTown:
    """A party of a split role before Town is reached: nothing can go anywhere yet (there is no in-memory stand-in
    across two machines), so a send fails loudly and the agent sends again later (meter readings) or logs it."""

    def __init__(self, name: str):
        self.name = name

    def send(self, to: str, kind: str, body: dict[str, Any], message_id: str) -> dict[str, Any]:
        raise RuntimeError(f"{self.name}: not on NANDA Town yet, cannot reach {to}")


class DCFleet:
    def __init__(self, cfg: dict[str, Any] | None = None):
        base = cfg if cfg is not None else config.load()
        self.base = base
        c = base
        self.dc_role = str(c.get("dc_role", "all")).lower()
        if self.dc_role not in ROLES:
            raise ValueError(f"dc_role must be one of {ROLES}, not {self.dc_role!r}")
        self.campus_url = (c.get("dc_campus_url") or "").rstrip("/") or None
        self.op_name = c.get("grid_operator_name", "grid-operator")
        self.lock = threading.RLock()
        self.bus: deque = deque()
        self.events: deque = deque(maxlen=800)
        self.clock = Replay(c.get("clock_profile", "tesla_calhoun"), c["replay_start"], c["replay_step_seconds"],
                            c["replay_loop_days"])
        self.ids = dc.site_ids()
        self.names = [dc.node_name(e) for e in self.ids]
        self.solar = dc.community_solar()["solar_kw"]
        self.reg = dc.region()
        self.meta = {m["entity_id"]: m for m in dc.meta()["datacenters"]}
        self.started = time.time()
        self.hold_until = 0.0                    # after /demo/reset: no tick until the new clock arrives (or 30 s)
        from . import grid_signals
        self._grid = grid_signals.Cache(ttl=float(c.get("grid_now_cache_s", 300)))     # MISO, for /dc/today
        # the transport: NANDA Town (one session per party here) or the in-memory stand-in
        self.towns: dict[str, TownSession] = {}
        self.pollers: dict[str, InboxPoller] = {}
        self.town_pending: dict[str, TownSession] | None = None     # joined in the background, used from the next run
        self.town_in: Counter = Counter()                            # messages claimed from Town, by receiver role
        self.town_out: Counter = Counter()                           # messages Town accepted, by sender role
        self.town_errors: Counter = Counter()
        self.town_last_error: str | None = None
        self.stale_in = 0                                            # Town messages from an earlier run, dropped
        self._tlock = threading.Lock()                               # the Town counters, updated from many threads
        self.lat: deque = deque(maxlen=2000)                         # (receiver role, s from sentWall to handling)
        self.call_ms: dict[str, deque] = {}                          # Town call -> its last durations, ms
        self.retries = 0                                             # sends that needed another try
        self.redelivered = 0                                         # messages Town handed out again (acked, not reapplied)
        self._recovered0 = 0
        self.town_lease = 30.0                                       # Town's claim lease (from the join answer)
        self._seen: OrderedDict = OrderedDict()                      # Town message ids handled (at-least-once delivery)
        self._pollers = False
        self._gen = 0                                                # bumped when this process changes role
        self._joining = False
        self.pool = ThreadPoolExecutor(max_workers=int(c.get("dc_send_workers", 8)))          # data centers' sends
        # the campus link (market role) and the trust it was given (sites role)
        self.chttp = httpx.Client(timeout=float(c.get("dc_campus_timeout_s", 10.0)), headers=doors.headers())
        self.campus: dict[str, Any] = {"url": self.campus_url, "reached": None}
        self._camp_cache: tuple[float, dict[str, Any] | None] = (0.0, None)
        self.site_pins: dict[str, str] = {}
        self.history: list[SimpleNamespace] | None = None            # the sites' history (market role: allowance)
        # what if (POST /dc/conditions, from the coordinator's /demo/scenario): "operator" (its forecast of the region,
        # what the weather service says) and "sites" (urgent work, batteries out of service); kept until a reset
        self.conditions: dict[str, Any] = {}
        self._load_keys()
        if self.dc_role == "all" and self.campus_url:
            # a campus may still hold the data centers' Town sessions: NANDA Town 0.2 gives a second join the same
            # session, and the two machines would split the data centers' messages without an error. Stand it down.
            self.campus["standby"] = self._campus_post("/dc/standby", {})
        self._choose_transport()
        self._build()
        if self.dc_role == "market":
            self._start_campus_link()

    # ------------------------------------------------------------------ roles and keys
    @property
    def local_sites(self) -> bool:
        return self.dc_role in ("all", "sites")

    @property
    def local_market(self) -> bool:
        return self.dc_role in ("all", "market")

    def parties(self) -> list[str]:
        """The Town parties this process runs."""
        return ([self.desk_name, self.op_name] if self.local_market else []) + (self.names if self.local_sites else [])

    def _load_keys(self) -> None:
        """The keys of the parties that run here. The desk signs with the coordinator's key (the Mac's); a campus holds
        only its data centers' keys and the desk key it was given (the trust file)."""
        keydir = config.CONFIG_DIR / "dc" / "keys"
        if self.local_market:
            self.ckey = signing.load_or_create(config.CONFIG_DIR / "secrets" / "signing.key")   # the coordinator's key
            self.cpub = signing.public_hex(self.ckey)
            self.op_pub = signing.public_hex(signing.load_or_create(config.CONFIG_DIR / "secrets" / "grid_operator.key"))
            # the desk's Town name; in one process on the in-memory stand-in it answers as the coordinator
            self.desk_name = self.base.get("dc_desk_name", "dc-desk") if self.dc_role == "market" \
                else self.base.get("coordinator_name", "coordinator")
        else:
            self.ckey, self.op_pub = None, None
            t = self._trust()
            self.cpub = t.get("desk_pub")
            self.desk_name = t.get("desk") or self.base.get("dc_desk_name", "dc-desk")
            self.op_name = t.get("grid_operator") or self.op_name
        if self.local_sites:
            keydir.mkdir(parents=True, exist_ok=True)
            self.keys = {n: signing.load_or_create(keydir / f"{n}.key") for n in self.names}
        else:
            self.keys = {}
        if self.dc_role == "market":
            self.site_pins = {n: k for n, k in (self._pinned().get("pins") or {}).items() if n in self.names}

    def _trust(self) -> dict[str, Any]:
        p = config.CONFIG_DIR / "dc" / TRUST_FILE
        try:
            return json.loads(p.read_text()) if p.exists() else {}
        except (OSError, ValueError):
            return {}

    def _pinned(self) -> dict[str, Any]:
        p = config.CONFIG_DIR / "dc" / PINS_FILE
        try:
            return json.loads(p.read_text()) if p.exists() else {}
        except (OSError, ValueError):
            return {}

    def pins(self) -> dict[str, str]:
        """The data centers' public keys (the campus, or the all role): what the desk and the grid operator pin."""
        return {n: signing.public_hex(k) for n, k in self.keys.items()}

    # ------------------------------------------------------------------ the transport
    def _choose_transport(self) -> None:
        """NANDA Town when this machine holds a join token for every party here (TOKENS_FILE) and Town lets them join;
        else, in the all role, the in-memory stand-in, with the reason in transport_note (and, if Town did not answer, a
        background retry: the next run uses Town once it does). `dc_transport` in agent.json: auto (the default), town
        or memory. A campus waits for a desk to trust before it joins ("standby")."""
        b = self.base
        want = str(b.get("dc_transport", "auto")).lower()
        self.transport, self.transport_note = "memory", None
        if self.dc_role == "all":
            self.desk_name = b.get("coordinator_name", "coordinator")
        else:
            self.transport = "standby"
            if want == "memory":
                self.transport_note = f"dc_transport is memory, but the {self.dc_role} role has only NANDA Town"
                self._set_cfg()
                return
            if self.dc_role == "sites" and not self.cpub:
                self.transport_note = ("waiting for a desk to trust: the Mac's flexgrid-dc (market role) hands over the "
                                       "desk's key through its SSH tunnel (POST /dc/trust)")
                self._set_cfg()
                return
        if want != "memory":
            towns, why = self._town_sessions()
            if towns is not None:
                try:
                    for t in towns.values():
                        j = t.join()
                        self.town_lease = float(((j or {}).get("run") or {}).get("lease_seconds") or 30.0)
                    self._use_town(towns)
                except Exception as exc:
                    self._town_error("join", exc)
                    self.transport_note = (f"could not join NANDA Town ({str(exc)[:120]}): "
                                           + ("in-memory until it can" if self.dc_role == "all" else "waiting")
                                           + " (tried again every minute; used from the next run)")
                    threading.Thread(target=self._retry_town, args=(self._gen,), daemon=True).start()
            elif why and (want == "town" or self.dc_role != "all" or not why.startswith("no NANDA Town identities yet")):
                self.transport_note = why
        self._set_cfg()

    def _town_sessions(self) -> tuple[dict[str, TownSession] | None, str | None]:
        b = self.base
        path = config.CONFIG_DIR / "secrets" / TOKENS_FILE
        if not path.exists():
            return None, f"no NANDA Town identities yet ({TOKENS_FILE} not on this machine: mac/town_add_datacenters.sh)"
        try:
            tokens = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            return None, f"{TOKENS_FILE} unreadable: {exc}"
        desk = b.get("dc_desk_name", "dc-desk") if self.local_market else self.desk_name
        need = ([desk, self.op_name] if self.local_market else []) + (self.names if self.local_sites else [])
        missing = [n for n in need if not tokens.get(n)]
        if missing:
            return None, "NANDA Town identities incomplete, missing " + ", ".join(missing)
        if not b.get("town_url") or not b.get("town_run_id"):
            return None, "no town_url / town_run_id in agent.json"
        return {n: TownSession(b["town_url"], b["town_run_id"], n, tokens[n]) for n in need}, None

    def _retry_town(self, gen: int) -> None:
        while self.transport != "town" and gen == self._gen:
            time.sleep(float(self.base.get("dc_town_retry_seconds", 60)))
            towns, _ = self._town_sessions()
            if towns is None or gen != self._gen:
                continue
            try:
                for t in towns.values():
                    t.join()
            except Exception as exc:
                self._town_error("join", exc)
                continue
            with self.lock:
                if gen != self._gen:
                    return
                if not self._busy():                                  # nothing under way: switch now
                    self._use_town(towns)
                    self._set_cfg()
                    self._build()
                else:                                                 # an event is running in memory: next run
                    self.town_pending = towns
            return

    def _busy(self) -> bool:
        """An event is under way here (not settled yet)."""
        d = getattr(self, "desk", None)
        if d is not None:
            return any(e.get("phase") != "done" for e in d.events.values())
        return any(a.contracts for a in getattr(self, "agents", {}).values())

    def _use_town(self, towns: dict[str, TownSession]) -> None:
        self.towns, self.town_pending = towns, None
        self.transport, self.transport_note = "town", None
        if self.local_market:
            self.desk_name = self.base.get("dc_desk_name", "dc-desk")

    def _set_cfg(self) -> None:
        """The desk runs the household market's code: the buyer is the grid operator, the evening check is the
        region's (dc_market.py). In one process every send goes out at once (nothing is lost: no wall-clock retry
        floors); on NANDA Town the desk sends as the coordinator does, broadcasts in the background, with retries."""
        b = self.base
        self.cfg = {**b, "utility_name": self.op_name, "sync_broadcast": self.transport != "town",
                    "rebound_tolerance": float(b.get("dc_rebound_tolerance", 9.0))}

    def where(self) -> dict[str, str]:
        """Which machine runs which party."""
        here = self.base.get("dc_host_label") or socket.gethostname().split(".")[0]
        camp = self.campus.get("host") or "the campus"
        out = {}
        for n in [self.desk_name, self.op_name]:
            out[n] = here if self.local_market else "the Mac"
        for n in self.names:
            out[n] = here if self.local_sites else camp
        return out

    def transport_label(self) -> str:
        b = self.base
        if self.transport == "town":
            town = f"NANDA Town ({b.get('town_url')}, run {str(b.get('town_run_id'))[:8]})"
            if self.dc_role == "market":
                return (f"{town}: the desk ({self.desk_name}) and the grid operator here, the {len(self.names)} data "
                        f"centers on the data center campus ({self.campus.get('host') or self.campus_url}), each party "
                        "with its own Town identity; between the machines every message goes through Town, a signed "
                        "Beckn body its receiver checks")
            if self.dc_role == "sites":
                return (f"{town}: the {len(self.names)} data centers of this campus, each with its own Town identity; "
                        f"the desk ({self.desk_name}) and the grid operator run on the Mac and are reached only through "
                        "Town; every message is a signed Beckn body its receiver checks")
            return (f"{town}: the desk ({self.desk_name}), the {len(self.names)} data centers and the grid operator "
                    "each have their own Town identity, and every message is a signed Beckn body its receiver checks")
        if self.dc_role != "all":
            return f"not on NANDA Town yet ({self.dc_role} role): {self.transport_note or 'joining'}"
        if self.transport_note:
            return f"in-memory stand-in for NANDA Town (flexgrid-dc, one process): {self.transport_note}"
        return TRANSPORT

    def _sender(self, name: str, blocking: bool) -> Callable:
        """What `name` sends with: the in-memory queue, or its own Town session. The desk and the grid operator wait for
        Town's answer, as the coordinator and the utility do; a data center sends in the background and gets a future,
        as a fleet home does (a meter reading counts as reported only once Town has taken it)."""
        if self.transport != "town":
            return (_Session(name, self) if self.dc_role == "all" else _NoTown(name)).send
        town = self.towns[name]
        if blocking:
            return lambda to, kind, body, mid: self._town_send(town, to, kind, body, mid)
        return lambda to, kind, body, mid: self.pool.submit(self._town_send, town, to, kind, body, mid)

    def _town_send(self, town: TownSession, to: str, kind: str, body: dict[str, Any], mid: str, tries: int = 3):
        """A send, tried up to three times (a connection the server had closed, a timeout). Retries reuse the message
        id: Town takes the same message once, so a lost answer never makes a second message."""
        for k in range(tries):
            t0 = time.time()
            try:
                out = town.send(to, kind, body, mid)
                if isinstance(out, dict) and out.get("duplicate"):    # Town refused it: see town.send
                    self._town_error(f"send {town.name}->{to}", RuntimeError(f"Town refused {kind}: id {mid} holds another message"))
                self._timed("send", t0)
                with self._tlock:
                    self.town_out[self.role(town.name)] += 1
                    self.retries += 1 if k else 0
                return out
            except Exception as exc:
                self._town_error(f"send {town.name}->{to}", exc)
                if k == tries - 1:
                    raise
                time.sleep(1.0 + k)

    def _timed(self, op: str, t0: float) -> None:
        with self._tlock:
            self.call_ms.setdefault(op, deque(maxlen=500)).append((time.time() - t0) * 1000)

    def _post(self, frm: str, to: str, kind: str, body: dict[str, Any], mid: str) -> None:
        """A send of this process's own (the clean allowance): the in-memory queue, or Town in the background."""
        if self.transport == "town":
            self.pool.submit(self._town_send, self.towns[frm], to, kind, body, mid)
        else:
            self.bus.append((frm, to, kind, body))

    def _town_error(self, where: str, exc: Exception) -> None:
        with self._tlock:
            self.town_errors[where.split()[0]] += 1
            self.town_last_error = f"{where}: {str(exc)[:160]}"
        log.warning("town %s", self.town_last_error)

    def start_town(self) -> None:
        """The Town inbox pollers (once per role): one per party here, a short notify check every few seconds (no
        long-polls held open: Town re-checks each one every 50 ms inside a write transaction), then claims until the
        inbox is empty (town.InboxPoller). Measured on the team's Town (29 Oct): a call through its Cloudflare tunnel
        takes about a second, so each party polls on its own and never waits behind another party's calls. After a
        quiet minute a party checks less often (`dc_town_poll_idle_seconds`)."""
        if self._pollers or self.transport != "town":
            return
        self._pollers = True
        b = self.base
        gen = self._gen
        self._recovered0 = 0
        self.pollers = {}
        for n in self.parties():
            market = n in (self.desk_name, self.op_name)
            every = float(b.get("dc_town_poll_seconds", 2.0)) if market else float(b.get("dc_town_poll_seconds_sites", 3.0))
            p = InboxPoller(self.towns[n], lambda msg, n=n: self._handle_claimed(n, msg, gen), every=every,
                            idle_every=max(every, float(b.get("dc_town_poll_idle_seconds", 6.0))),
                            idle_after=float(b.get("dc_town_idle_after_seconds", 60.0)),
                            sweep=float(b.get("dc_town_sweep_seconds", 60)), lease=self.town_lease,
                            claimers=int(b.get("dc_town_claimers", 6)) if market else 1,
                            on_error=lambda where, exc, n=n: self._town_error(f"{where} {n}", exc),
                            alive=lambda n=n: gen == self._gen and n in self.towns, timed=self._timed)
            self.pollers[n] = p
            p.start(f"dc-town-{n}")

    def _handle_claimed(self, name: str, msg: dict[str, Any], gen: int) -> None:
        """One message claimed from `name`'s inbox: handled, then acknowledged. A message claimed just as this process
        changed role is left unacknowledged: Town hands it out again after its lease, to the party's new home."""
        if gen != self._gen or name not in self.towns:
            raise RuntimeError(f"{name} no longer runs here: left for its new home (Town hands it out again)")
        note, status = self.receive(name, msg)
        t1 = time.time()
        try:
            _once_more(self.towns[name].ack, msg["message_id"], msg["fence"], status, {"note": note[:200]})
            self._timed("ack", t1)
        except Exception as exc:
            self._town_error(f"ack {name}", exc)

    def receive(self, to: str, msg: dict[str, Any]) -> tuple[str, str]:
        """One message claimed from `to`'s Town inbox. Town delivers at least once: a message id already handled is
        acknowledged again, not applied again. A message sent before this run started (an earlier demo, or before a
        restart) is dropped. Returns (note, ack status)."""
        mid = str(msg.get("message_id", ""))
        body = msg.get("body") or {}
        with self.lock:
            if mid in self._seen:
                self.redelivered += 1
                return "already handled (redelivered)", self._seen[mid]      # what it came to the first time
            self._seen[mid] = "rejected"
            while len(self._seen) > 20000:
                self._seen.popitem(last=False)
            sent = (body.get("context") or {}).get("sentWall") if isinstance(body, dict) else None
            if isinstance(sent, (int, float)) and sent < self.epoch_wall:
                self.stale_in += 1
                return beckn.Refused("sent before this run started: not acted on"), "rejected"
            self.town_in[self.role(to)] += 1
            if isinstance(sent, (int, float)):                       # made -> sent -> waiting in Town -> claimed here
                self.lat.append((self.role(to), max(0.0, time.time() - sent)))
            note = self._dispatch(str(msg.get("from") or ""), to, str(msg.get("kind", "")), body)
            self._seen[mid] = beckn.ack_of(note)[0]
        return note, self._seen[mid]

    def _reset_town_stats(self) -> None:
        """Town traffic counts are per run (the record's run); the Town calls' own durations are kept (the link)."""
        with self._tlock:
            self.town_in.clear()
            self.town_out.clear()
            self.town_errors.clear()
            self.town_last_error = None
            self.stale_in = self.retries = self.redelivered = 0
            self.lat.clear()
            self._recovered0 = sum(p.recovered for p in self.pollers.values())

    # ------------------------------------------------------------------ the agents
    def _build(self) -> None:
        """Fresh sites, agents, desk and grid operator (the ones that run here), and a fresh record (start, a demo reset,
        a new clock, or the clock going back); the keys stay. A Town connection made in the background since the last
        run is used now."""
        from .grid_operator import GridOperator
        if self.town_pending is not None:
            self._use_town(self.town_pending)
            self._set_cfg()
        c = self.cfg
        self.epoch_wall = time.time()                                 # messages sent before this belong to an old run
        self.events.clear()                                           # the record starts again with the agents
        self._reset_town_stats()
        for old in (getattr(self, "agents", None) or {}).values():
            old.closed = True                                         # a decision still being made is dropped
        self.agents: dict[str, dc.DataCenterAgent] = {}
        if self.local_sites:
            # an AI agent's LLM call runs in a thread of its own: every message here is handled under self.lock (and a
            # Town claim lasts 30 s), so the seconds it thinks must not hold either
            acfg = {"utility_name": self.op_name, "max_message_age_s": c.get("max_message_age_s", 900),
                    "think_in_background": bool(c.get("dc_think_in_background", True))}
            for eid, name in zip(self.ids, self.names):
                site = dc.DataCenterSite(eid, c.get("dc_site_cfg"), clock=self.clock)
                self.agents[name] = dc.DataCenterAgent(name, {**acfg, **(c.get("dc_agent_cfg") or {}).get(name, {})},
                                                       site, self.clock, self.keys[name],
                                                       send=self._sender(name, blocking=False),
                                                       trusted=lambda: [self.cpub] if self.cpub else [],
                                                       log=self._logger(name), coordinator=self.desk_name,
                                                       lock=self.lock)
                self._site_conditions(self.agents[name])
                if c.get("llm_enabled") and name in self.llm_sites:      # this site's AI agent (brain.py)
                    from .brain import DataCenterBrain
                    self.agents[name].brain = DataCenterBrain(
                        self.agents[name], self._brain_backend("datacenter"),
                        policy=(c.get("dc_agent_policy") or {}).get(name), label=f"Data center {name}",
                        log=self._logger(name), offer_budget_s=float(c.get("llm_offer_budget_s", 40)),
                        accept_budget_s=float(c.get("llm_accept_budget_s", 45)))
        self.op = self.desk = None
        self.op_agent = None                                          # the grid operator's AI agent (market side)
        self.enrolled: set[str] = set()
        if self.local_market:
            pins = self.pins() if self.local_sites else dict(self.site_pins)
            op = GridOperator()
            op.clock = self.clock
            # how long the grid operator waits for a late meter reading before it settles without it (replay
            # minutes). With every reading in it settles at once; only on Town can one still be on its way when the
            # desk reports
            op.cfg = {**op.cfg, "utility_settle_wait_minutes": float(c.get("dc_settle_wait_minutes", 120)),
                      # on Town, a reading may still be on its way (or in the inbox of a machine that slept): settle an
                      # incomplete event only once none has come for this long in real time (utility.Utility.tick)
                      "utility_settle_quiet_s": float(c.get("dc_settle_quiet_s", 30.0 if self.transport == "town" else 0.0))}
            if self.transport == "town":
                op.town = _TownSender(self, self.towns[self.op_name])
            else:
                op.town = _Session(self.op_name, self) if self.dc_role == "all" else _NoTown(self.op_name)
            op.pins = {**pins, self.desk_name: self.cpub}                 # its enrolled meters and aggregator
            op.aggregators = set(c.get("grid_operator_aggregators") or []) | {self.desk_name}
            op.event = self._logger(self.op_name)
            self.op = op
            # every key the desk will ever accept is known here (enrollment; on a campus, the keys it handed over);
            # _dispatch() refuses anyone else, so the market's pin-on-first-contact never pins a stranger
            self.desk = DCMarket(self.desk_name, self.ckey, self._sender(self.desk_name, blocking=True), self.clock,
                                 list(self.names), {**pins, self.op_name: op.pub}, lambda: None,
                                 self._logger(self.desk_name), c, region_forecast=dc.region_forecast)
            if c.get("llm_enabled") and c.get("dc_desk_llm", True):     # the desk's margin: the coordinator's AI agent
                from .brain import DESK_DECIDE, CoordinatorBrain
                self.desk.brain = CoordinatorBrain(self._brain_backend("coordinator"), log=self._logger(self.desk_name),
                                                   budget_s=float(c.get("llm_margin_budget_s", 40)),
                                                   default_margin=float(c.get("clear_margin", 0.10)),
                                                   prompt=DESK_DECIDE, who="desk")
            if c.get("llm_enabled") and c.get("dc_operator_llm", True):  # the grid operator's own AI agent: tomorrow's
                from .brain import GridOperatorBrain                       # order, then the sign-off (brain.py)
                self.op_agent = GridOperatorBrain(self._brain_backend("operator"), log=self._logger(self.op_name),
                                                  order_budget_s=float(c.get("llm_operator_order_budget_s", 40)),
                                                  sign_budget_s=float(c.get("llm_operator_sign_budget_s", 30)))
            self.enrolled = set(self.names) | {self.op_name}
        self.signing: dict[str, str] = {}                                # event id -> the operator agent's sign-off state
        self.start_town()                                             # the inbox pollers, once on Town
        self.rec_from: int | None = None
        self.last_idx: int | None = None
        self.msg_steps: dict[int, Counter] = {}                          # clock step -> messages in it
        self.traffic: Counter = Counter()
        self.allowance: dict[str, dict[str, Any]] = {}                   # day sent -> what was sent
        self.verified: dict[str, dict[str, Any]] = {}
        self.errors = 0

    # ------------------------------------------------------------------ what if: conditions of the day
    def _site_conditions(self, a) -> None:
        sc = self.conditions.get("sites") or {}
        a.site.urgent = float(sc.get("urgent_share") or 0.0)
        a.site.battery_out = a.name in (sc.get("battery_out") or [])

    def set_conditions(self, body: dict[str, Any]) -> dict[str, Any]:
        """POST /dc/conditions (the coordinator's what if): {"reset": true} (all back to normal), {"operator":
        {"region_load_x": 0.8-1.25, "note": "what the weather service says"}} (the grid operator's AI agent reads both
        before its order), {"sites": {"urgent_share": 0-0.9, "battery_out": ["dc-03"]}} (the share of each site's
        batch work that cannot wait today: its safe offer and what it holds both shrink; batteries out of service: 0 kW
        either way). The agents react with their own code; the market role passes the sites' part on to the campus."""
        out: dict[str, Any] = {}
        with self.lock:
            if body.get("reset"):
                self.conditions = {}
            op = body.get("operator")
            if isinstance(op, dict):
                x = max(0.8, min(1.25, float(op.get("region_load_x") or 1.0)))
                note = str(op.get("note") or "").strip()[:300] or None
                self.conditions["operator"] = {"region_load_x": round(x, 3), "note": note}
                if abs(x - 1.0) < 1e-9 and not note:
                    self.conditions.pop("operator")
            sc = body.get("sites")
            if isinstance(sc, dict):
                cur = self.conditions.get("sites") or {}
                u = max(0.0, min(0.9, float(sc.get("urgent_share", cur.get("urgent_share", 0.0)) or 0.0)))
                bo = sorted({str(n) for n in (sc.get("battery_out", cur.get("battery_out")) or []) if str(n) in self.names})
                self.conditions["sites"] = {"urgent_share": round(u, 2), "battery_out": bo}
                if u <= 0 and not bo:
                    self.conditions.pop("sites")
            for a in self.agents.values():
                self._site_conditions(a)
                if body.get("reset"):
                    a.site.urgent_jobs.clear()
            uj = body.get("urgent_job")                    # the operator's urgent job at one site, from now on
            if isinstance(uj, dict) and uj.get("site") in self.agents:
                a = self.agents[uj["site"]]
                out["urgent_job"] = a.urgent_job(1000.0 * max(0.5, min(10.0, float(uj.get("mw") or 5))),
                                                 max(0.5, min(4.0, float(uj.get("hours") or 2))))
                out["urgent_job"]["site"] = uj["site"]
            out["conditions"] = json.loads(json.dumps(self.conditions))
            self._logger("flexgrid-dc")("what_if", conditions=out["conditions"])
        if self.dc_role == "market" and (body.get("reset") or isinstance(body.get("sites"), dict)
                                         or isinstance(body.get("urgent_job"), dict)):
            out["campus"] = self._campus_post("/dc/conditions", {"reset": bool(body.get("reset")),
                                                                **({"sites": body["sites"]} if isinstance(body.get("sites"), dict) else {}),
                                                                **({"urgent_job": body["urgent_job"]} if isinstance(body.get("urgent_job"), dict) else {})})
            self._camp_cache = (0.0, None)
        return out

    @property
    def llm_sites(self) -> set[str]:
        """The data centers whose agent thinks with the LLM (dc_llm_sites; the rest keep their rules)."""
        return set(self.cfg.get("dc_llm_sites") or ["dc-09"])

    def _brain_backend(self, agent: str):
        """One LLM backend per OpenClaw agent for this process (kept across runs: its status and timings carry on)."""
        from .brain import make_backend
        cache = self.__dict__.setdefault("_backends", {})
        if agent not in cache:
            cache[agent] = make_backend({**self.cfg, "openclaw_agent": agent}, agent)
        return cache[agent]

    def _agents_view(self) -> dict[str, Any]:
        """The AI agents here (GET /dc/state "agents"): each LLM site's decisions and the desk's margins."""
        out: dict[str, Any] = {}
        try:
            sites = {n: a.brain.state() for n, a in list((getattr(self, "agents", None) or {}).items())
                     if getattr(a, "brain", None) is not None}
            if sites:
                out["sites"] = sites
            if getattr(self, "desk", None) is not None and getattr(self.desk, "brain", None) is not None:
                out["desk"] = self.desk.brain.state()
            if getattr(self, "op_agent", None) is not None:
                out["operator"] = self.op_agent.state()
        except Exception as exc:                                   # noqa: BLE001  never let this break the record
            out["error"] = f"{type(exc).__name__}: {str(exc)[:120]}"
        return out

    def _logger(self, who: str):
        def f(kind: str, **d):
            self.events.append({"at": time.time(), "sim": self.now(), "who": who, "kind": kind, **d})
        return f

    def now(self) -> str:
        return self.clock.ts(self.clock.current())

    def role(self, name: str) -> str:
        return "C" if name == self.desk_name else "G" if name == self.op_name else "D"

    # ------------------------------------------------------------------ transport and the loop
    def deliver(self) -> int:
        """The in-memory stand-in: every queued message to its receiver (on NANDA Town the queue stays empty)."""
        n = 0
        while self.bus:
            frm, to, kind, body = self.bus.popleft()
            n += 1
            self._dispatch(frm, to, kind, body)
        return n

    def _dispatch(self, frm: str, to: str, kind: str, body: dict[str, Any]) -> str:
        """One message to its receiver, whichever way it came, counted in the record. The desk hears only enrolled
        parties and well-formed offers. Returns the receiver's note."""
        act = beckn.action_of(kind)
        if to == self.desk_name and self.desk is not None:
            why = None if frm in self.enrolled else "not enrolled"
            o = ((body or {}).get("message") or {}).get("offer")
            if why is None and o is not None and not _offer_ok(o):
                why = "malformed offer"
            if why:
                self._logger(self.desk_name)("refused", sender=frm, message_kind=kind, reason=why)
                return beckn.Refused(f"refused: {why}")
        self.msg_steps.setdefault(self.clock.current(), Counter())[f"{self.role(frm)}{self.role(to)}:{act}"] += 1
        self.traffic[act] += 1
        try:
            if to == self.desk_name and self.desk is not None:
                out = self.desk.handle(kind, frm, body)
            elif to == self.op_name and self.op is not None:
                with self.op.lock:
                    out = self.op.handle(kind, frm, body)
            elif to in self.agents:
                out = self.agents[to].handle(kind, frm, body)
            else:
                self._logger("flexgrid-dc")("undeliverable", to=to, message_kind=kind)
                out = beckn.Refused("refused: undeliverable")
            return out if isinstance(out, str) else str(out)
        except Exception as exc:                                        # one bad message must not stop the others
            self.errors += 1
            self._logger("flexgrid-dc")("handle_error", to=to, message_kind=kind, error=str(exc)[:200])
            log.error("deliver %s -> %s: %s", frm, to, traceback.format_exc())
            return beckn.Refused(f"error: {str(exc)[:160]}")

    def tick(self) -> None:
        """One pass: every site here to the clock's step (each agent sends the meter readings that are due), then the
        desk's timeline, the grid operator's, and the day's clean allowance; messages are delivered between. A split
        role waits until it is on Town (there is nobody to talk to before)."""
        with self.lock:
            if time.time() < self.hold_until:                       # reset, waiting for /demo/clock (coordinator's
                return                                               # /demo/start sends both, a moment apart)
            if self.dc_role != "all" and self.transport != "town":
                return
            cur = self.clock.current()
            if self.last_idx is not None and cur < self.last_idx:     # the clock went back (a looping replay wraps)
                self._logger("flexgrid-dc")("clock_back", now=self.now())
                self._build()
            if self.rec_from is None:
                self.rec_from = cur
            self.last_idx = cur
            for a in self.agents.values():
                self._guard(a.name, a.tick)
            self.deliver()
            if self.desk is not None:
                self._guard(self.desk_name, self.desk.tick)
                self.deliver()
                self._guard(self.op_name, self._op_tick)
                self.deliver()
                self._guard(self.desk_name, self._allowance)
                self.deliver()
                self._guard("verify", self._verify_settled)
            if len(self.msg_steps) > 600:                           # keep the message counts of the record's span
                for k in [k for k in self.msg_steps if k < cur - 2 * RECORD_STEPS]:
                    del self.msg_steps[k]

    def _op_tick(self) -> None:
        with self.op.lock:
            self.op.tick()

    def _guard(self, who: str, f) -> None:
        """One phase of the tick; a failure is counted and logged, and the next phase still runs."""
        try:
            f()
        except Exception as exc:
            self.errors += 1
            self._logger(who)("tick_error", error=str(exc)[:200])
            log.error("%s: %s", who, traceback.format_exc())

    def loop(self) -> None:
        while True:
            t0 = time.time()
            try:
                self.tick()
            except Exception as exc:
                self.errors += 1
                self._logger("flexgrid-dc")("loop_error", error=str(exc)[:200])
                log.error("tick: %s", traceback.format_exc())
            time.sleep(max(0.1, float(self.cfg.get("dc_tick_seconds", 0.5)) - (time.time() - t0)))

    # ------------------------------------------------------------------ clean allowance (measure 1, live)
    def _history(self) -> list[Any]:
        """The sites' 2023 series (history only: no state), for the desk's forecasts where the sites run elsewhere."""
        if self.agents:
            return [a.site for a in self.agents.values()]
        if self.history is None:
            self.history = [SimpleNamespace(id=e, s=dc.Series(e), w=int(self.meta[e]["max_defer_steps"]))
                            for e in self.ids]
        return self.history

    def _allowance(self) -> None:
        """At 10:00 each day, each data center's share of tomorrow's forecast unused community solar, as a signed
        Beckn update from the desk. Not for a day with a large-load event: there the contract decides. Whether tomorrow
        is one comes from the grid operator's answer to the desk's 10:00 discover, which over NANDA Town takes a few
        seconds, so the allowance waits for that answer (until noon at the latest)."""
        if not self.cfg.get("dc_clean_allowance", True):
            return
        now = self.now()
        day = now[:10]
        if now[11:16] < "10:00" or day in self.allowance:
            return
        if (self.desk.last_discover_day or "") < day and now[11:16] < "12:00":
            return                                  # the grid operator has not said yet whether tomorrow is an event day
        tomorrow = beckn.add_minutes(f"{day} 00:00", 1440)[:10]
        r = self.clock
        i_tom = r.idx_of(f"{tomorrow} 00:00")
        if r.loop_days > 0 and (i_tom is None or i_tom >= r.i0 + r.loop_days * 96):
            self.allowance[day] = {"for": tomorrow, "skipped": "the looping replay clock never reaches that day"}
            return
        if any(e["need"]["window_start"][:10] == tomorrow for e in self.desk.events.values()):
            self.allowance[day] = {"for": tomorrow, "skipped": "large-load event day: the contract decides"}
            return
        sites = self._history()
        alw = dc.clean_allowance(tomorrow, self.clock.current(), sites, self.solar)
        if not alw:
            self.allowance[day] = {"for": tomorrow, "skipped": "not in the dataset"}
            return
        for eid, name in zip(self.ids, self.names):
            kw = alw.get(eid) or []
            body = beckn.make("update", sender=self.desk_name, receiver=name, transaction_id=f"clean-{tomorrow}",
                              message={"clean_allowance": {"start": f"{tomorrow} 00:00", "kw": kw}}, key=self.ckey)
            self._post(self.desk_name, name, beckn.kind("update"), body,
                       ids.sealed(self.ckey, self.desk.play, self.desk_name, name, "update", "clean", tomorrow))
        kwh = {name: round(sum(alw.get(eid) or []) * 0.25, 1) for eid, name in zip(self.ids, self.names)}
        self.allowance[day] = {"for": tomorrow, "kwh": kwh, "total_kwh": round(sum(kwh.values()), 1), "made_at": now}
        self._logger(self.desk_name)("dc_clean_allowance_sent", date=tomorrow, kwh=round(sum(kwh.values()), 1))

    # ------------------------------------------------------------------ the independent check
    def bundle(self, event_id: str) -> dict[str, Any]:
        from . import verify
        ev = self.desk.events.get(event_id)
        if not ev:
            raise KeyError(event_id)
        with self.desk.lock, self.op.lock:
            cev = json.loads(json.dumps({**ev, "pins": dict(self.desk.pins)}, default=str))
            uev = json.loads(json.dumps(self.op.evidence(event_id), default=str))
        return verify.bundle(cev, uev, "live: flexgrid-dc", participants="data centers", buyer="grid operator")

    def verify_event(self, event_id: str, replay: bool = False) -> dict[str, Any]:
        """flexgrid_core/verify.py on this event (signatures, delivery recomputed to the cent, the split). replay:
        also run the same day offline (tools/sim_datacenters.py, a fresh process) and compare what the grid operator
        pays."""
        from . import verify
        b = self.bundle(event_id)
        o = b["utility"].get("order")
        rt = replay_total(b["utility"]["need"]["window_start"][:10], verify.replay_agents(b),
                          order={**o["order"], "buyer": o["buyer"]} if o else None) if replay else None
        return verify.check(b, rt)

    def _verify_settled(self) -> None:
        for eid, ev in self.desk.events.items():
            if eid in self.verified or ev.get("phase") != "done":
                continue
            ct = self.op.contract_of_event(eid)
            if not ct or ct.get("status") != "COMPLETE":
                continue
            try:
                r = self.verify_event(eid)
                self.verified[eid] = {"ok": r["ok"], "checks": [{k: x[k] for k in ("name", "ok", "detail")}
                                                                for x in r["checks"]], "summary": r["summary"]}
                self._logger("verify")("verified", event_id=eid, ok=r["ok"])
            except Exception as exc:
                self.verified[eid] = {"ok": False, "error": str(exc)[:200]}
            if self.op_agent is not None and eid not in self.signing:
                self._operator_sign_off(eid)

    # ------------------------------------------------------------------ the grid operator's AI agent
    def _catalog_request(self, day: str) -> dict[str, Any] | None:
        """The program's own request for `day` (the catalog's, or an order already placed): window, MW, price."""
        n = next((x for x in self.op.needs_list() if x["window"][0][:10] == day), None)
        if not n:
            return None
        return {"event_id": n["event_id"], "start": n["window"][0][11:16], "end": n["window"][1][11:16],
                "mw": round(float(n["kw"]) / 1000, 1), "price": float(n["price_usd_per_kwh"]), "source": n["source"]}

    def _price_of(self, day: str, start: str, end: str, default: float) -> float:
        """The program's price for a window: its day-ahead price, the mean over its quarter hours ($/MWh -> $/kWh), as
        the catalog's (tools/make_ma_datacenters.py). In reality fixed when the day-ahead market clears (13:30 the day
        before); the mock fills it in from the dataset at once, as its catalog does."""
        r = self.reg
        i0, i1 = r["index"].get(f"{day} {start}"), r["index"].get(f"{day} {end}")
        if i0 is None or i1 is None or i1 <= i0:
            return default
        v = [r["price_da_usd_mwh"][i] for i in range(i0, i1)]
        return round(sum(v) / len(v) / 1000, 4) if v and all(x is not None for x in v) else default

    def _region_peaks(self) -> dict[str, float]:
        """The region's daily peak load (MW), every day of the dataset (computed once)."""
        if getattr(self, "_peaks", None) is None:
            pk: dict[str, float] = {}
            for t, mw in zip(self.reg["ts"], self.reg["system_load_mw"]):
                if mw is not None and mw > pk.get(t[:10], -1.0):
                    pk[t[:10]] = float(mw)
            self._peaks = pk
        return self._peaks

    def operator_facts(self, day: str) -> dict[str, Any]:
        """What the grid operator knows the morning before `day`, history only: its forecast of the region's load (the
        same quarter hour on the 3 latest days seen, as the desk's), the recent day-ahead prices by hour, the region's
        daily peaks so far this season and its planning level (their 90th percentile), the program's own request for
        the day, the ten enrolled data centers and how they delivered on this season's earlier events (the desk's
        backtest), and any condition the presenter set (a weather service's word on tomorrow: conditions)."""
        now = self.now()
        r = self.reg
        cond = (getattr(self, "conditions", None) or {}).get("operator") or {}
        x = float(cond.get("region_load_x", 1.0))
        hours = list(range(12, 22))
        steps = [f"{day} {h:02d}:{m:02d}" for h in hours for m in (0, 15, 30, 45)]
        fc = dc.region_forecast(steps, now)
        i_now = r["index"].get(now, 0)
        by_hour = []
        for k, h in enumerate(hours):
            i = r["index"].get(f"{day} {h:02d}:00")
            back = [i - 96 * d for d in range(1, 8) if i is not None and 0 <= i - 96 * d <= i_now][:3]
            pr = [r["price_da_usd_mwh"][j] for j in back if r["price_da_usd_mwh"][j] is not None]
            by_hour.append({"hour": f"{h:02d}:00", "load_mw": round(sum(fc[4 * k:4 * k + 4]) / 4 * x),
                            "recent_day_ahead_usd_per_kwh": round(sum(pr) / len(pr) / 1000, 3) if pr else None})
        top = max(by_hour, key=lambda y: y["load_mw"])
        peaks = [v for d, v in sorted(self._region_peaks().items()) if d < now[:10]]
        srt = sorted(peaks)
        plan = srt[min(len(srt) - 1, int(0.9 * len(srt)))] if len(srt) >= 10 else (max(srt) if srt else None)
        metas = list(self.meta.values())
        cat = self._catalog_request(day) or {}
        return {
            "now": now, "order_for": day,
            "program_request": {"window": f"{cat.get('start')}-{cat.get('end')}", "mw": cat.get("mw"),
                                "price_usd_per_kwh": cat.get("price"), "what": "the program's catalog request: it "
                                "stands if you do not order"} if cat else None,
            "region_forecast": {"how": "your forecast: the same quarter hour on the 3 latest days seen (history only)"
                                       + (f", times {x:g} on the weather service's word" if abs(x - 1) > 1e-9 else ""),
                                "by_hour": by_hour, "peak": top},
            "forecast_peak_above_planning_level_mw": round(top["load_mw"] - plan) if plan else None,
            "region_so_far": {"days": len(peaks), "planning_level_mw": round(plan) if plan else None,
                              "highest_daily_peak_mw": round(srt[-1]) if srt else None,
                              "median_daily_peak_mw": round(srt[len(srt) // 2]) if srt else None,
                              "last_3_daily_peaks_mw": [round(v) for v in peaks[-3:]]},
            "data_centers": {"enrolled": len(metas),
                             "registered_flexible_mw": round(sum(m.get("flex_down_kw", 0) for m in metas) / 1000, 1),
                             "batteries_mw": round(sum(m.get("bess_power_kw", 0) for m in metas) / 1000, 1),
                             "earlier_events": self.desk._backtest(day) if self.desk is not None else None},
            "conditions": cond.get("note") or None,
            "how_it_works": "the desk sells you at most what the data centers offer (confidence-weighted) and never "
                            "more than you order; you pay per kWh your meters measure against each site's baseline, "
                            "at most what was sold, at the window's day-ahead price"}

    def operator_order(self, day: str, budget_s: float | None = None) -> dict[str, Any]:
        """POST /dc/operator/order: the grid operator's AI agent orders `day`'s flexibility, in the background (the
        coordinator calls this when a live run starts, at 09:45 the day before, before the desk's 10:00 call). The order
        goes through GridOperator.place_order's own checks; the catalog's request stands if it is late or refused."""
        if self.op is None:
            raise ValueError("409: the grid operator does not run here (the sites role)")
        agent, op = self.op_agent, self.op
        if agent is None:
            return {"ordering": None, "why": "the grid operator's AI agent is off (llm_enabled, dc_operator_llm)"}
        with self.lock:
            cat = self._catalog_request(day)
        if cat is None:
            return {"ordering": None, "why": f"no large-load event on {day} in the program: nothing to order"}
        if cat["source"] != "catalog":
            return {"ordering": None, "why": f"{day} already has an order", "request": cat}
        if agent.busy:
            raise ValueError(f"409: the grid operator's agent is busy ({agent.busy})")
        if budget_s:
            agent.order_budget_s = max(10.0, min(120.0, float(budget_s)))
        agent.busy = f"ordering flexibility for {day}"

        def run() -> None:
            try:
                with self.lock:
                    facts = self.operator_facts(day)
                res = agent.decide_order(day, facts, cat, lambda s0, s1: self._price_of(day, s0, s1, cat["price"]))
                if res["order"] is None:
                    return
                with self.lock:
                    if self.op is not op or self.op_agent is not agent:
                        agent.refused(res["decision"], "a new run started meanwhile")
                        return
                    try:
                        with op.lock:
                            rec = op.place_order(res["order"])
                        res["decision"]["event_id"] = rec["event_id"]
                    except ValueError as exc:
                        agent.refused(res["decision"], str(exc))
            except Exception as exc:                                  # noqa: BLE001  the catalog's request stands
                self.errors += 1
                self._logger(self.op_name)("tick_error", error=f"operator order: {str(exc)[:180]}")
                log.error("operator order: %s", traceback.format_exc())
            finally:
                agent.busy = None
        threading.Thread(target=run, daemon=True, name=f"operator-order-{day}").start()
        return {"ordering": day, "request": cat, "budget_s": agent.order_budget_s}

    def presenter_order(self, day: str, mw: float, start: str | None = None, end: str | None = None) -> dict[str, Any]:
        """POST /dc/needs: the Control Room's request for the data centers before a run (the presenter's what if, as
        the homes' POST /needs on the utility): the program's window and price, the presenter's MW, signed by the
        grid operator; the grid operator's own agent does not order that day."""
        if self.op is None:
            raise ValueError("409: the grid operator does not run here (the sites role)")
        with self.lock:
            cat = self._catalog_request(day)
            if cat is None:
                raise ValueError(f"404: no large-load event on {day} in the program")
            o = {"date": day, "start": start or cat["start"], "end": end or cat["end"],
                 "kw": max(1.0, min(self.op.ORDER_MAX_KW, float(mw) * 1000.0)),
                 "price_usd_per_kwh": cat["price"], "buyer": "Control Room (what if)",
                 "note": "set by the presenter before the run"}
            with self.op.lock:
                rec = self.op.place_order(o)
        return {"date": day, "start": o["start"], "end": o["end"], "mw": round(o["kw"] / 1000, 1),
                "price_usd_per_kwh": o["price_usd_per_kwh"], "event_id": rec["event_id"]}

    def _operator_sign_off(self, eid: str) -> None:
        """The settled event's full check (with the offline replay, a few seconds in a fresh process), then the grid
        operator's agent signs off or disputes; GridOperator.review signs its decision and keeps it on the contract.
        In the background: the replay and the LLM must not hold the fleet's lock. Under self.lock."""
        agent, op = self.op_agent, self.op
        self.signing[eid] = "checking"
        agent.busy = f"checking the settlement of {eid}"

        def run() -> None:
            try:
                try:
                    r = self.verify_event(eid, replay=bool(self.cfg.get("dc_operator_replay", True)))
                except Exception as exc:                              # noqa: BLE001  a check that cannot run fails
                    r = {"ok": False, "summary": (self.verified.get(eid) or {}).get("summary") or {},
                         "checks": [{"name": "check", "ok": False, "detail": f"the check could not run: {str(exc)[:160]}"}]}
                d = agent.decide_sign_off(eid, r, now=self.now())
                with self.lock:
                    if self.op is not op or self.op_agent is not agent:
                        return
                    with op.lock:
                        op.review(eid, d["decision"] == "approved", d["reason"],
                                  [{k: x.get(k) for k in ("name", "ok", "detail")} for x in r.get("checks") or []],
                                  by="grid operator's agent" + (" (rules)" if d.get("fallback") else ""))
                    self.signing[eid] = d["decision"]
            except Exception as exc:                                  # noqa: BLE001
                self.errors += 1
                self.signing[eid] = "failed"
                self._logger(self.op_name)("tick_error", error=f"operator sign-off: {str(exc)[:180]}")
                log.error("operator sign-off: %s", traceback.format_exc())
            finally:
                agent.busy = None
        threading.Thread(target=run, daemon=True, name=f"operator-sign-{eid}").start()

    # ------------------------------------------------------------------ today, on MISO's real prices
    @staticmethod
    def profile_day(day: str) -> str:
        """The 2023 day that stands in for `day`'s forecast of batch work: the same weekday in the same week."""
        d = datetime.strptime(day, "%Y-%m-%d")
        base = d.replace(year=2023, day=min(d.day, 28)) if (d.month, d.day) == (2, 29) else d.replace(year=2023)
        p = base + timedelta(days=(d.weekday() - base.weekday() + 3) % 7 - 3)
        if p.year < 2023:
            p += timedelta(days=7)
        elif p.year > 2023:
            p -= timedelta(days=7)
        return p.strftime("%Y-%m-%d")

    def today(self, snap: dict[str, Any] | None = None) -> dict[str, Any]:
        """Each data center agent's plan for today on MISO's real day-ahead prices at the Illinois hub
        (grid_signals.py): its batch work moved to the cheapest hours it can reach inside its own deadlines and under
        its capacity (datacenter.price_plan), critical load untouched, nothing pushed past midnight. Today's batch work
        is forecast from the site's own history: the same weekday of the same week in 2023 (a stand-in for the
        operator's forecast). The plan is advice: nothing moves until an operator (or a contract) says so. In the
        market role the agents on the campus make it; if the campus does not answer, the Mac makes the same plan from
        the same history and says so."""
        snap = snap if snap is not None else self._grid.get()
        da = (snap or {}).get("day_ahead")
        if not da:
            return {"ok": False, "error": "no day-ahead prices: " + "; ".join((snap or {}).get("errors") or ["none"])}
        note = None
        if self.dc_role == "market":
            try:
                r = self.chttp.post(self.campus_url + "/dc/today", json=snap)
                r.raise_for_status()
                return {**r.json(), "made_on": self.campus.get("host") or "the campus"}
            except Exception as exc:
                note = f"the campus did not answer ({str(exc)[:80]}): made on the Mac from the same site history"
        by_start = dict(zip(da["starts"], da["lmp_usd_mwh"]))         # Illinois local hours of today only
        hourly: list[float] = []
        estimated = []                                               # hours with no price of their own in the report
        for hh in range(24):
            v = by_start.get(f"{hh:02d}:00")
            if v is None:
                estimated.append(f"{hh:02d}:00")
            hourly.append(float(v) if v is not None else (hourly[-1] if hourly else float(da["lmp_usd_mwh"][0])))
        with self.lock:
            plan_sites = [SimpleNamespace(name=a.name, s=a.site.s, w=a.site.w) for a in self.agents.values()] or \
                [SimpleNamespace(name=n, s=h.s, w=h.w) for n, h in zip(self.names, self._history())]
        s0 = plan_sites[0].s

        def span(day: str):
            a = s0.idx_of(f"{day} 00:00")
            b = s0.idx_of(f"{beckn.add_minutes(f'{day} 00:00', 1440)[:10]} 00:00")
            return a, ((b - a) if a is not None and b is not None else None)
        prof = self.profile_day(da["date"])
        i0, n = span(prof)
        if n != 96:                                                  # a clock-change day in 2023: a week away instead
            for alt in (beckn.add_minutes(f"{prof} 00:00", -7 * 1440)[:10], beckn.add_minutes(f"{prof} 00:00", 7 * 1440)[:10]):
                if span(alt)[1] == 96:
                    prof = alt
                    i0, n = span(alt)
                    break
        if i0 is None or not n:
            return {"ok": False, "error": f"no 2023 profile for {prof}"}
        price = [hourly[int(s0.ts[i0 + k][11:13])] for k in range(n)]
        imp, imp0 = [0.0] * n, [0.0] * n
        moved_from, moved_to = [0.0] * 24, [0.0] * 24
        sites = []
        for a in plan_sites:
            r = dc.price_plan(a.s, i0, n, a.w, price, min_gain=float(self.cfg.get("dc_today_min_gain", 5.0)))
            for k in range(n):
                imp[k] += r["import_kw"][k]
                imp0[k] += r["import_cf_kw"][k]
            for t, j, x in r["moves"]:
                moved_from[int(s0.ts[i0 + t][11:13])] += x * dc.STEP_H
                moved_to[int(s0.ts[i0 + j][11:13])] += x * dc.STEP_H
            sites.append({"name": a.name, "moved_mwh": round(r["moved_kwh"] / 1000, 2),
                          "batch_mwh": round(r["flex_kwh"] / 1000, 1), "cost_usd": r["cost_usd"],
                          "cost_cf_usd": r["cost_cf_usd"], "saved_usd": round(r["cost_cf_usd"] - r["cost_usd"], 2),
                          "longest_wait_h": r["max_wait_steps"] * dc.STEP_H, "can_wait_h": a.w * dc.STEP_H,
                          "over_capacity_kw": r["over_capacity_kw"]})
        pw = da["peak_window"]
        ks = [k for k in range(n) if pw["start"] <= s0.ts[i0 + k][11:16] < (pw["end"] if pw["end"] != "00:00" else "24:00")]
        cost, cost0 = sum(x["cost_usd"] for x in sites), sum(x["cost_cf_usd"] for x in sites)
        top = lambda arr: [{"hour": f"{hh:02d}:00", "mwh": round(v / 1000, 1)}            # noqa: E731
                           for hh, v in sorted(enumerate(arr), key=lambda x: -x[1])[:4] if v > 1]
        out = {"ok": True, "date": da["date"], "profile_day": prof, "node": da.get("node"),
               "prices": "MISO day-ahead LMP at the Illinois hub, $/MWh, by hour (Illinois time)",
               "hourly_usd_mwh": [round(x, 2) for x in hourly],
               "batch_mwh": round(sum(x["batch_mwh"] for x in sites), 1),
               "moved_mwh": round(sum(x["moved_mwh"] for x in sites), 1),
               "cost_usd": round(cost, 2), "cost_cf_usd": round(cost0, 2), "saved_usd": round(cost0 - cost, 2),
               "saved_pct": round(100 * (cost0 - cost) / cost0, 2) if cost0 > 0 else None,
               "peak_window": {"start": pw["start"], "end": pw["end"], "usd_mwh": pw["avg_usd_mwh"],
                               "mw_with": round(sum(imp[k] for k in ks) / len(ks) / 1000, 1) if ks else None,
                               "mw_without": round(sum(imp0[k] for k in ks) / len(ks) / 1000, 1) if ks else None},
               "moved_from": top(moved_from), "moved_to": top(moved_to),
               "series_mw": {"with": [round(x / 1000, 2) for x in imp], "without": [round(x / 1000, 2) for x in imp0],
                             "steps": [s0.ts[i0 + k][11:16] for k in range(n)]},
               "sites": sites, "late_kwh": 0.0,
               "over_capacity_kw": round(max(x["over_capacity_kw"] for x in sites), 3),
               "min_gain_usd_mwh": float(self.cfg.get("dc_today_min_gain", 5.0)),
               "hours_priced_from_the_hour_before": estimated,
               "how": "Each data center agent moves its deferrable batch work to the cheapest hours it can reach "
                      "inside its own deadline (4.5-10.5 h) and under its capacity (10 % kept free), within today, "
                      "and only where the hour is at least $5/MWh cheaper; critical load never moves. Today's batch "
                      "work is forecast from the same weekday of the same week in 2023 (the team dataset). Cost = "
                      "grid energy at today's MISO day-ahead price.",
               "source": snap.get("source"), "fetched_at_utc": snap.get("fetched_at_utc")}
        if note:
            out["note"] = note
        return out

    # ------------------------------------------------------------------ the record the Control Room reads
    def current_event(self, day: str | None = None) -> dict[str, Any] | None:
        """The event the record is about: that day's, when asked for a day (the Control Room asks for the homes' event
        day, tools/dc_timeline.py for the day it records); else the earliest one whose evening (8 h after its window)
        is not over yet, so on an event day the next day's event (asked for at 10:00) waits its turn; else the latest."""
        if self.desk is None:
            return None
        now = self.now()
        evs = list(self.desk.events.values())
        if day:
            mine = [e for e in evs if e["need"]["window_start"][:10] == day]
            if mine:
                return mine[-1]
        live = [e for e in evs if now <= beckn.add_minutes(e["need"]["window_end"], 8 * 60)]
        if live:
            return min(live, key=lambda e: e["need"]["window_start"])
        return evs[-1] if evs else None

    def _event_view(self, ev: dict[str, Any]) -> dict[str, Any]:
        n = ev["need"]
        cs = ev["commitments"]
        res = ev.get("result") or {}
        cap = ev.get("capacity_offered_final") or ev.get("capacity_offered") or {}
        split = (res.get("split") or {}).get("per_home", {})
        sites = {}
        for name in self.names:
            o = ev["offers"].get(name) or {}
            cm = cs.get(name) or {}
            cc = cm.get("contract") or {}
            rs = ev["readings"].get(name) or {}
            sites[name] = {
                "offer_kw": [x["kw_offered"] for x in o.get("slots", [])],
                "offer_pause_kw": [x.get("compute_pause_kw") for x in o.get("slots", [])],
                "offer_battery_kw": [x.get("dc_battery_kw") for x in o.get("slots", [])],
                "confidence": o.get("confidence"),
                "commit_kw": [x["kw"] for x in cc.get("commitments", [])], "status": cm.get("status"),
                "reason": cm.get("reason"),
                "resume_kwh": round(sum((cc.get("resume_plan") or {}).values()) * 0.25, 1) if cc else None,
                "readings": len(rs),
                "usage_kw": [r.get("usage_kw") for _, r in sorted(rs.items())],
                "true_kwh": round(sum((r.get("truth") or {}).get("delivered_true_kw", 0.0) * SLOT_H
                                      for r in rs.values()), 1) if rs else None,
                "baseline_kw": [v.get("BASELINE") for _, v in sorted(((ev.get("baselines") or {}).get(name) or {}).items())
                                if isinstance(v, dict)],
                "paid_usd": round(split.get(name, 0.0), 2) if split else None}
        truth = sum((r.get("truth") or {}).get("delivered_true_kw", 0.0) * SLOT_H
                    for rs in ev["readings"].values() for r in rs.values())
        return {"id": n["event_id"], "day": n["window_start"][:10], "window": [n["window_start"], n["window_end"]],
                "commit_by": n["commit_by"], "price_usd_per_kwh": n["slots"][0]["price"],
                "requested_kw": n["slots"][0]["kw_requested"],
                "slots": [{"id": s["id"], "start": s["start"], "kw_requested": s["kw_requested"]} for s in n["slots"]],
                "buyer": "grid operator (mock): large-load flexibility, paid at the day-ahead price",
                "phase": ev["phase"], "offers": len(ev["offers"]),
                "accepted": sum(1 for x in cs.values() if x["status"] == "ACCEPTED"),
                "declined": sum(1 for x in cs.values() if x["status"] == "DECLINED"),
                "capacity_kw": [cap.get(s["id"], cap.get(str(s["id"]))) for s in n["slots"]] if cap else None,
                "offered_kw_raw": (ev.get("clear_summary") or {}).get("offered_kw_raw"),
                "region_guard": ev.get("region_guard"),
                "readings": sum(len(v) for v in ev["readings"].values()),
                "promised_kwh": round(sum(cap.values()) * SLOT_H, 1) if cap else None,
                "delivered_kwh_true": round(truth, 1) if ev["readings"] else None,
                "delivered_kwh_measured": res.get("delivered_kwh_measured"),
                "seller_usd": res.get("seller_usd"), "sites": sites,
                "verified": self.verified.get(n["event_id"]), **self._operator_view(n["event_id"])}

    def _operator_view(self, eid: str) -> dict[str, Any]:
        """The grid operator's order for this event (its AI agent's, signed) and its sign-off of the settlement."""
        if self.op is None:
            return {}
        o = self.op.orders.get(eid)
        ct = self.op.contract_of_event(eid) or {}
        rv = ct.get("buyer_review")
        return {"order": {"by": o["buyer"], "placed_at": o["placed_at"], **{k: o["order"].get(k) for k in
                                                                              ("start", "end", "kw", "price_usd_per_kwh", "note")}}
                if o else None,
                "review": {k: rv.get(k) for k in ("decision", "reason", "by", "at", "settlement_usd")} if rv else None,
                "signing": self.signing.get(eid)}

    def _sites_record(self) -> dict[str, Any]:
        """The sites' part of the record (the sites that run here): every step since the clock was set (at most
        RECORD_STEPS), per site and in total, with the region against the sites' no-event twins. Under self.lock."""
        cur = self.clock.current()
        first = next(iter(self.agents.values())).site
        lo = max(self.rec_from if self.rec_from is not None else cur, cur - RECORD_STEPS + 1)
        idx = [i for i in sorted(first.recent) if lo <= i <= cur]
        sites = list(self.agents.values())
        agg = {k: [] for k in ("import_kw", "import_cf_kw", "queue_kwh", "battery_kw", "held_kw", "pv_kw",
                               "region_mw", "region_cf_mw", "price_usd_mwh")}
        per = {k: [[] for _ in sites] for k in ("mode",) + SITE_KEYS}
        for i in idx:
            tot = dict.fromkeys(("import_kw", "import_cf_kw", "queue_kwh", "battery_kw", "held_kw", "pv_kw"), 0.0)
            for k, a in enumerate(sites):
                t = a.site.recent.get(i) or {}
                for key in tot:
                    tot[key] += t.get(key) or 0.0
                per["mode"][k].append(MODES.index(t["mode"]) if t.get("mode") in MODES else 0)
                for key in SITE_KEYS:
                    per[key][k].append(round(t.get(key) or 0.0, 1) + 0.0)
            for key, v in tot.items():
                agg[key].append(round(v, 1) + 0.0)
            rm = self.reg["system_load_mw"][i]
            agg["region_cf_mw"].append(round(rm, 2))
            agg["region_mw"].append(round(rm - (tot["import_cf_kw"] - tot["import_kw"]) / 1000, 2))
            agg["price_usd_mwh"].append(round(self.reg["price_usd_mwh"][i], 2))
        return {
            "idx": idx, "steps": [first.s.ts[i] for i in idx], "series": agg, "site_steps": per,
            "sites": [{"name": a.name, "entity": a.site.id, "peak_mw": round(self.meta[a.site.id]["peak_kw"] / 1000, 1),
                       "avg_mw": round(self.meta[a.site.id]["annual_mwh"] / 8760, 1),
                       "battery_mw": round(a.site.P / 1000, 2), "battery_mwh": round(a.site.E / 1000, 2),
                       "backup_pct": round(a.site.backup * 100), "defer_h": a.site.w * dc.STEP_H,
                       "pv_mw": round(self.meta[a.site.id]["pv_ac_kw"] / 1000, 1),
                       "lat": round(self.meta[a.site.id]["lat"], 3), "lon": round(self.meta[a.site.id]["lon"], 3),
                       "reputation": a.reputation, "paid_usd": round(sum(float(p.get("payment") or 0) for p in a.payments), 2),
                       "late_kwh": round(a.site.late_kwh, 3)} for a in sites],
            "late_kwh": round(sum(a.site.late_kwh for a in sites), 3),
            "over_capacity_kw": round(max(a.site.over_cap_kw for a in sites), 3)}

    def _static_sites(self) -> list[dict[str, Any]]:
        """The sites as enrolled (no telemetry): what the market shows while the campus does not answer."""
        out = []
        for e, n in zip(self.ids, self.names):
            m = self.meta[e]
            out.append({"name": n, "entity": e, "peak_mw": round(m["peak_kw"] / 1000, 1),
                        "avg_mw": round(m["annual_mwh"] / 8760, 1), "battery_mw": round(m["bess_power_kw"] / 1000, 2),
                        "battery_mwh": round(m["bess_energy_kwh"] / 1000, 2), "backup_pct": m.get("backup_pct", 50),
                        "defer_h": m["max_defer_steps"] * dc.STEP_H, "pv_mw": round(m["pv_ac_kw"] / 1000, 1),
                        "lat": round(m["lat"], 3), "lon": round(m["lon"], 3), "reputation": None, "paid_usd": None,
                        "late_kwh": None})
        return out

    def state(self, day: str | None = None, raw: bool = False) -> dict[str, Any]:
        """The record: every step since the clock was set (at most RECORD_STEPS), and the event of `day` (see
        current_event). In the market role the sites' part comes from the campus (its telemetry: simulation truth for
        the Control Room, never read by the desk or the grid operator) and is merged with the desk's. raw: what the
        market merges from (step indices, log lines with their wall times, the Town delays one by one)."""
        camp = self._campus_record() if self.dc_role == "market" else None
        agents = self._agents_view()                  # outside the lock: an LLM status may need a quick HTTP check
        if camp and (camp.get("agents") or {}).get("sites"):
            agents["sites"] = {**camp["agents"]["sites"], **(agents.get("sites") or {})}
        with self.lock:
            ev = self.current_event(day)
            dw = dc.files().get("data_word") or "the 2023 dataset"
            src = ("flexgrid-dc: ten data center agents, the coordinator's large-load desk and the grid operator (mock), "
                   f"the same code as tools/sim_datacenters.py; sites simulated from {dw}")
            if self.dc_role == "sites":
                src = ("flexgrid-dc data center campus: the ten data center agents and their sites (simulated from "
                       f"{dw}); the desk and the grid operator run on the Mac")
            elif self.dc_role == "market":
                src = ("flexgrid-dc: the coordinator's large-load desk and the grid operator (mock) here; the ten data "
                       f"center agents on the data center campus ({self.campus.get('host') or self.campus_url}), "
                       f"talking only over NANDA Town; sites simulated from {dw}")
            out = {"schema": "flexgrid.dc.v1", "live": True, "now": self.now(), "clock": self.clock.clock(),
                   "source": src, "role": self.dc_role, "where": self.where(),
                   "transport": self.transport_label(), "transport_mode": self.transport, "modes": MODES,
                   "event": self._event_view(ev) if ev else None,
                   "events": self.desk.summary() if self.desk is not None else {},
                   "clean_allowance": dict(self.allowance), "errors": self.errors, "how": HOW, "agents": agents,
                   "conditions": json.loads(json.dumps(self.conditions))}
            if self.desk is not None:
                # the last day whose 10:00 question about tomorrow the grid operator has answered (the live run's
                # speed control holds 1x until it has: coordinator.dc_brake)
                out["discover_answered"] = self.desk.last_discover_day
            if self.op_agent is not None and self.op_agent.busy:
                out["operator_busy"] = self.op_agent.busy             # its AI agent orders or checks: hold 1x
            if self.local_sites:
                rec = self._sites_record()
                out.update({k: rec[k] for k in ("steps", "series", "site_steps", "sites", "late_kwh", "over_capacity_kw")})
                out["messages"] = [dict(self.msg_steps.get(i, {})) for i in rec["idx"]]
                if raw:
                    out["idx"] = rec["idx"]
            mine_log = self._log_for(ev, keep_at=True)
            town = self.town_status(raw=raw or self.dc_role == "market")
            out["traffic"] = dict(self.traffic)
            msg_mine = {i: dict(c) for i, c in self.msg_steps.items()}
        if self.dc_role == "market":
            out.update(self._merge_campus(camp, msg_mine, mine_log, town))
        else:
            out["town"] = self._town_public(town, raw)
            out["log"] = mine_log if raw else [{k: v for k, v in x.items() if k != "at"} for x in mine_log]
        return json.loads(json.dumps(out, default=str))

    def _merge_campus(self, camp: dict[str, Any] | None, msg_mine: dict[int, dict[str, int]],
                      mine_log: list[dict[str, Any]], town: dict[str, Any] | None) -> dict[str, Any]:
        """The market's record with the campus's: the sites' steps, the message counts of both ends of each step, the
        two logs in time order, Town's delays measured at every receiver."""
        status = self.campus_status()
        if not camp:
            return {"steps": [], "series": {}, "site_steps": {}, "sites": self._static_sites(), "messages": [],
                    "late_kwh": None, "over_capacity_kw": None, "log": [{k: v for k, v in x.items() if k != "at"}
                                                                       for x in mine_log],
                    "town": self._town_public(town, False), "campus": status}
        idx = camp.get("idx") or []
        cm = camp.get("messages") or []
        msgs = [_sum_counts(cm[k] if k < len(cm) else {}, msg_mine.get(i)) for k, i in enumerate(idx)]
        rows = sorted(mine_log + list(camp.get("log") or []), key=lambda x: x.get("at") or 0)
        return {"steps": camp.get("steps") or [], "series": camp.get("series") or {},
                "site_steps": camp.get("site_steps") or {}, "sites": camp.get("sites") or self._static_sites(),
                "messages": msgs, "late_kwh": camp.get("late_kwh"), "over_capacity_kw": camp.get("over_capacity_kw"),
                "traffic": _sum_counts(dict(self.traffic), camp.get("traffic")),
                "errors": self.errors + int(camp.get("errors") or 0),
                "log": [{k: v for k, v in x.items() if k != "at"} for x in rows],
                "town": self._town_public(self._merge_town(town, camp.get("town")), False), "campus": status}

    def _log_for(self, ev: dict[str, Any] | None, others: int = 150, keep_at: bool = False) -> list[dict[str, Any]]:
        """Every log line of the current event (the Control Room reads its key times from them), plus the latest
        other lines, in time order."""
        rows = list(self.events)
        eid = ev["need"]["event_id"] if ev else None
        mine = (lambda x: x.get("event_id") == eid or str(x.get("contract_id", "")).startswith(f"{eid}:")) if eid \
            else (lambda x: False)                                    # noqa: E731
        keep = {id(x) for x in rows if mine(x)} | {id(x) for x in [x for x in rows if not mine(x)][-others:]}
        return [{k: v for k, v in x.items() if keep_at or k != "at"} for x in rows if id(x) in keep]

    def town_status(self, raw: bool = False) -> dict[str, Any] | None:
        """What NANDA Town is doing for the parties here in this run, measured here: how long a message takes from being
        made to being handled by its receiver (delivery_s: sending, waiting in Town until the receiver's next check,
        claiming), each kind of Town call's own time, sends that needed another try, redeliveries, errors, and how many
        checks and claims the parties made (their load on Town)."""
        if self.transport != "town":
            return None
        with self._tlock:
            lat = list(self.lat)
            calls = {op: _stats(list(v), 0) for op, v in self.call_ms.items()}
            out = {"url": self.base.get("town_url"), "run_id": self.base.get("town_run_id"), "parties": sorted(self.towns),
                   "where": self.where(), "received": dict(self.town_in), "sent": dict(self.town_out),
                   "errors": dict(self.town_errors), "last_error": self.town_last_error,
                   "dropped_from_earlier_runs": self.stale_in, "retries": self.retries, "redelivered": self.redelivered,
                   "recovered_without_hint": sum(p.recovered for p in self.pollers.values()) - self._recovered0,
                   "polling": {"checks": sum(p.checks for p in self.pollers.values()),
                               "claims": sum(p.claims for p in self.pollers.values())},
                   "identity_conflicts": sum(t.conflicts for t in self.towns.values())}
        out["delivery_s"] = _stats([x for _, x in lat], 2)
        out["delivery_s_by_receiver"] = {r: _stats([x for k, x in lat if k == r], 2) for r in sorted({k for k, _ in lat})}
        out["calls_ms"] = calls
        if raw:
            out["lat_raw"] = [[k, round(x, 3)] for k, x in lat]
        return out

    @staticmethod
    def _town_public(t: dict[str, Any] | None, raw: bool) -> dict[str, Any] | None:
        if t is None or raw:
            return t
        return {k: v for k, v in t.items() if k != "lat_raw"}

    def _merge_town(self, mine: dict[str, Any] | None, camp: dict[str, Any] | None) -> dict[str, Any] | None:
        """Town as both machines measured it: counts added up, the delays recomputed from every receiver's own
        measurements, each machine's Town call times side by side."""
        if not camp:
            return mine
        if not mine:
            return camp
        lat = [tuple(x) for x in (mine.get("lat_raw") or []) + (camp.get("lat_raw") or [])]
        out = {k: mine.get(k) for k in ("url", "run_id")}
        out.update({"parties": sorted(set(mine.get("parties") or []) | set(camp.get("parties") or [])),
                    "where": {**(camp.get("where") or {}), **{k: v for k, v in (mine.get("where") or {}).items()
                                                               if k in (self.desk_name, self.op_name)}},
                    "received": _sum_counts(mine.get("received"), camp.get("received")),
                    "sent": _sum_counts(mine.get("sent"), camp.get("sent")),
                    "errors": _sum_counts(mine.get("errors"), {f"campus {k}": v for k, v in (camp.get("errors") or {}).items()}),
                    "last_error": mine.get("last_error") or camp.get("last_error"),
                    "polling": {"mac": mine.get("polling"), "campus": camp.get("polling")},
                    "calls_ms": mine.get("calls_ms"), "calls_ms_campus": camp.get("calls_ms"),
                    "identity_conflicts": int(mine.get("identity_conflicts") or 0) + int(camp.get("identity_conflicts") or 0)})
        for k in ("dropped_from_earlier_runs", "retries", "redelivered", "recovered_without_hint"):
            out[k] = int(mine.get(k) or 0) + int(camp.get(k) or 0)
        out["delivery_s"] = _stats([x for _, x in lat], 2)
        out["delivery_s_by_receiver"] = {r: _stats([x for k, x in lat if k == r], 2) for r in sorted({k for k, _ in lat})}
        out["lat_raw"] = [list(x) for x in lat]
        return out

    def health(self) -> dict[str, Any]:
        camp_town = None
        if self.dc_role == "market":
            camp = self._campus_record()
            camp_town = (camp or {}).get("town")
        with self.lock:
            out = {"ok": True, "agent": "flexgrid-dc", "role": self.dc_role, "now": self.now(), "clock": self.clock.clock(),
                   "datacenters": len(self.agents) if self.local_sites else len(self.site_pins),
                   "desk": self.desk_name, "desk_pubkey": self.cpub,
                   "grid_operator": self.op_name, "grid_operator_pubkey": self.op.pub if self.op else None,
                   "transport": self.transport_label(), "transport_mode": self.transport,
                   "town_pending": self.town_pending is not None,
                   "events": {k: v["phase"] for k, v in self.desk.events.items()} if self.desk is not None else {},
                   "errors": self.errors, "uptime_s": round(time.time() - self.started), "where": self.where()}
            town = self.town_status(raw=self.dc_role == "market")
        out["town"] = self._town_public(self._merge_town(town, camp_town) if self.dc_role == "market" else town, False)
        if self.dc_role == "market":
            out["campus"] = self.campus_status()
        if self.dc_role == "sites":
            t = self._trust()
            out["trust"] = {"desk": t.get("desk"), "desk_pubkey": t.get("desk_pub"), "since": t.get("at")} if t else None
            out["contracts"] = {n: {k: v.get("status") for k, v in a.contracts.items()} for n, a in self.agents.items()}
        return out

    # ------------------------------------------------------------------ the campus link (market role)
    def _start_campus_link(self) -> None:
        threading.Thread(target=self._campus_loop, args=(self._gen,), daemon=True).start()

    def _campus_loop(self, gen: int) -> None:
        """Hand the desk's and the grid operator's keys to the campus and pin its data centers' keys; again every 30 s
        (a campus that restarted, or was set up again, gets them again; a changed key is caught)."""
        while gen == self._gen:
            ok = self._campus_handshake()
            time.sleep(30.0 if ok else 5.0)

    def _campus_handshake(self) -> bool:
        body = {"desk": self.desk_name, "desk_pub": self.cpub, "grid_operator": self.op_name,
                "grid_operator_pub": self.op_pub, "town_run_id": self.base.get("town_run_id")}
        try:
            r = self.chttp.post(self.campus_url + "/dc/trust", json=body)
            if r.status_code == 409:
                self.campus.update(reached=True, at=time.time(), alert=str(r.json().get("detail"))[:300])
                return False
            r.raise_for_status()
            j = r.json()
        except Exception as exc:
            self.campus.update(reached=False, checked=time.time(), error=str(exc)[:160])
            return False
        self._accept_pins(j.get("pins") or {})
        self.campus.update(reached=True, at=time.time(), error=None, host=j.get("host"), parties=j.get("parties"),
                           transport=j.get("transport_mode"), trusted_since=j.get("trusted_since"))
        # a campus that restarted is on its own clock again: give it the demo's (the same wall-clock anchor)
        mine, theirs = self.clock.clock(), j.get("clock") or {}
        if any(mine.get(k) != theirs.get(k) for k in ("start", "step_seconds", "loop_days", "at_wall")):
            body = {"start": mine["start"], "step_seconds": mine["step_seconds"], "loop_days": mine["loop_days"]}
            if mine.get("at_wall") is not None:
                body.update(at=mine["start"], at_wall=mine["at_wall"])
            self.campus["clock_synced"] = self._campus_post("/demo/clock", body)
            self._logger("flexgrid-dc")("campus_clock", campus=self.campus["clock_synced"])
        return True

    def _accept_pins(self, pins: dict[str, str]) -> None:
        """The campus's data center keys, pinned on first sight (~/.config/flexgrid/dc/campus_pins.json); a key that
        differs from the pinned one is not accepted and says so (to accept new keys, remove that file)."""
        with self.lock:
            pf = self._pinned()
            stored = dict(pf.get("pins") or {})
            changed = sorted(n for n, k in pins.items() if n in stored and stored[n] != k)
            new = {n: k for n, k in pins.items() if n in self.names and n not in stored and _hex64(k)}
            if new:
                stored.update(new)
                p = config.CONFIG_DIR / "dc" / PINS_FILE
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({"pins": stored, "campus": self.campus_url,
                                         "first_pinned": pf.get("first_pinned") or time.time(),
                                         "updated": time.time()}, indent=1))
            self.site_pins = {n: stored[n] for n in self.names if n in stored}
            alerts = [f"the campus shows another key for {', '.join(changed)} than the one pinned here: not accepted "
                      f"(to accept new keys: remove ~/.config/flexgrid/dc/{PINS_FILE})"] if changed else []
            if self.desk is not None:                                  # names the desk and the grid operator have not
                for n, k in self.site_pins.items():                    # pinned yet (a first run with no campus answer)
                    if n not in self.desk.pins:
                        self.desk.pins[n] = k
                    elif self.desk.pins[n] != k:
                        alerts.append(f"{n} reached the desk over Town with another key than the campus shows")
                    self.op.pins.setdefault(n, k)
            self.campus["alert"] = "; ".join(alerts) or None

    def campus_status(self) -> dict[str, Any]:
        c = dict(self.campus)
        c["pins"] = len(self.site_pins)
        c["pinned_since"] = self._pinned().get("first_pinned")
        return c

    def _campus_record(self) -> dict[str, Any] | None:
        """The campus's own record (GET /dc/state?raw=1), at most every 2 s."""
        t, rec = self._camp_cache
        if time.time() - t < 2.0:
            return rec
        try:
            r = self.chttp.get(self.campus_url + "/dc/state", params={"raw": 1})
            r.raise_for_status()
            rec = r.json()
            self.campus.update(reached=True, record_at=time.time(), error=None)
        except Exception as exc:
            rec = None
            self.campus.update(reached=False, checked=time.time(), error=str(exc)[:160])
        self._camp_cache = (time.time(), rec)
        return rec

    def _campus_post(self, path: str, body: dict[str, Any]) -> str:
        try:
            r = self.chttp.post(self.campus_url + path, json=body)
            r.raise_for_status()
            j = r.json()
            return str(j.get("now") or j.get("ok") or "ok")
        except Exception as exc:
            return f"not reached: {str(exc)[:80]}"

    # ------------------------------------------------------------------ the campus side (sites role)
    def accept_trust(self, body: dict[str, Any]) -> dict[str, Any]:
        """The Mac's flexgrid-dc hands over the desk's and the grid operator's keys (through its SSH tunnel). The first
        desk key is pinned; another one is refused until campus_trust.json is removed. Then this campus joins NANDA
        Town (once). Returns its data centers' public keys."""
        if self.dc_role != "sites":
            raise ValueError("409: only a data center campus (dc_role sites) takes a desk's trust")
        desk, dpub = str(body.get("desk") or ""), str(body.get("desk_pub") or "")
        opn, opub = str(body.get("grid_operator") or ""), str(body.get("grid_operator_pub") or "")
        if not desk or not opn or not _hex64(dpub) or not _hex64(opub):
            raise ValueError("400: desk, desk_pub, grid_operator and grid_operator_pub (64 hex) are needed")
        if body.get("town_run_id") and self.base.get("town_run_id") and body["town_run_id"] != self.base["town_run_id"]:
            raise ValueError(f"409: this campus is on Town run {self.base['town_run_id']}, the desk on {body['town_run_id']}")
        with self.lock:
            t = self._trust()
            if t and (t.get("desk_pub") != dpub or t.get("desk") != desk):
                raise ValueError(f"409: this campus already trusts the desk {t.get('desk')} with another key (since "
                                 f"{time.strftime('%Y-%m-%d %H:%M', time.localtime(t.get('at') or 0))}); to change it, "
                                 f"remove ~/.config/flexgrid/dc/{TRUST_FILE} on the campus")
            if not t:
                t = {"desk": desk, "desk_pub": dpub, "grid_operator": opn, "grid_operator_pub": opub, "at": time.time()}
                p = config.CONFIG_DIR / "dc" / TRUST_FILE
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(t, indent=1))
                self._logger("campus")("trust_pinned", desk=desk)
            go = self.transport != "town" and not self._joining
            if go:
                self.cpub, self.desk_name, self.op_name = dpub, desk, opn
                self._joining = True
        if go:
            threading.Thread(target=self._go_live, daemon=True).start()
        return {"pins": self.pins(), "host": self.base.get("dc_host_label") or socket.gethostname().split(".")[0],
                "parties": sorted(self.towns), "transport_mode": self.transport, "trusted_since": t.get("at"),
                "clock": self.clock.clock()}

    def _go_live(self) -> None:
        try:
            with self.lock:
                self._gen += 1                                        # a join retry from before stops here
            self._choose_transport()
            with self.lock:
                self._build()
            self._logger("campus")("on_town" if self.transport == "town" else "not_on_town",
                                   note=self.transport_note)
        finally:
            self._joining = False

    def standby(self) -> dict[str, Any]:
        """Stop and forget the desk: the Mac takes the data centers back (POST /dc/role all). The pollers stop at once
        and nothing joins again until a desk hands over its keys again."""
        with self.lock:
            self._gen += 1
            self.towns, self.pollers, self._pollers = {}, {}, False
            self.transport, self.transport_note = "standby", "stood down: the Mac runs the data centers again"
            self.cpub = None
            p = config.CONFIG_DIR / "dc" / TRUST_FILE
            if p.exists():
                p.unlink()
            self._build()
            self._logger("campus")("standby")
        return {"ok": True, "transport_mode": self.transport}

    def close(self) -> None:
        """Stop this instance for good: its pollers and its background join retry end at once and it holds no Town
        session. A process that gives up on one DCFleet and makes another must close the first: NANDA Town 0.2 gives a
        second join under a name the same session, so a forgotten instance that joins later claims messages meant for
        the live one (seen in tests/test_dc_town_flaky.py: a contract answered too late, an offer never made)."""
        with self.lock:
            self._gen += 1
            self.towns, self.pollers, self._pollers = {}, {}, False
            self.transport, self.transport_note = "closed", "closed"

    # ------------------------------------------------------------------ the Mac's role, switched at run time
    def set_role(self, role: str, campus_url: str | None = None, force: bool = False) -> dict[str, Any]:
        """all <-> market on the Mac, between events: the parties that leave stop at once (a message one of them had
        just claimed goes back to Town and on to the party's new home after its lease), the ones that stay join again
        (harmless), and the record starts afresh. Saved in agent.json, so a restart keeps the role. Going back to all
        first asks the campus to stand down, so the two machines never take turns at the same Town identity."""
        role = str(role).lower()
        if self.dc_role == "sites":
            raise ValueError("409: this is a data center campus; its role is set in its own agent.json")
        if role not in ("all", "market"):
            raise ValueError("400: role must be all or market")
        url = (campus_url or self.campus_url or "").rstrip("/") or None
        if role == "market" and not url:
            raise ValueError("400: campus_url is needed for the market role")
        with self.lock:
            if self._busy() and not force:
                raise ValueError("409: an event is under way here; switch between events (or force=true)")
            if role == self.dc_role and url == self.campus_url:
                return self.health()
            leaving_campus = self.dc_role == "market" and role == "all"
        if leaving_campus:
            self.campus["standby"] = self._campus_post("/dc/standby", {})
        with self.lock:
            self._gen += 1
            self.towns, self.pollers, self._pollers = {}, {}, False
            self.dc_role, self.campus_url = role, url
            self.campus = {"url": url, "reached": None}
            self._camp_cache = (0.0, None)
            self.base = {**self.base, "dc_role": role, "dc_campus_url": url}
            try:
                config.save({**config.load(), "dc_role": role, "dc_campus_url": url})
            except Exception as exc:                                   # the switch still holds until a restart
                log.warning("could not save dc_role: %s", exc)
            self._load_keys()
        self._choose_transport()                                       # joins the parties that run here now
        with self.lock:
            self._build()
            self._logger("flexgrid-dc")("role", role=role, campus=url)
        if role == "market":
            self._start_campus_link()
            self._campus_handshake()
        return self.health()

    # ------------------------------------------------------------------ demo controls (the coordinator's /demo/start)
    def reset(self) -> dict[str, Any]:
        """Fresh agents, desk, grid operator and record. Ticks wait for the demo's new clock (up to 30 s), so the fresh
        market never runs a step of the old clock in between. The market role resets the campus too."""
        with self.lock:
            self.conditions = {}                                      # a new run starts from a normal day
            self._build()
            self.hold_until = time.time() + 30.0
            out = {"ok": True, "datacenters": len(self.agents)}
        if self.dc_role == "market":
            out["campus"] = self._campus_post("/demo/reset", {})
            self._camp_cache = (0.0, None)
        return out

    def configure_clock(self, **kw) -> dict[str, Any]:
        """Set the replay clock (the demo's). A clock that goes back is a new run: fresh agents, desk, grid operator
        and record, nothing from the old timeline carried over. A clock that goes on from here (tools/demo_event.py
        changing speed, or starting a held clock) keeps everything: each site still plays every step once. The market
        role passes the same clock (the same wall-clock start) on to the campus."""
        with self.lock:
            r = self.clock
            r.configure(kw.get("start", r.start), float(kw.get("step_seconds", r.step_seconds)),
                        int(kw.get("loop_days", r.loop_days)), kw.get("at"), kw.get("at_wall"))
            self.hold_until = 0.0
            if self.last_idx is not None and r.current() < self.last_idx:
                self._build()
            out = r.clock()
        if self.dc_role == "market":
            out = {**out, "campus": self._campus_post("/demo/clock", kw)}
        return out


def replay_total(day: str, agents: dict[str, Any] | None = None, order: dict[str, Any] | None = None) -> float:
    """What the grid operator pays for this day's event in the offline simulation (tools/sim_datacenters.py, a fresh
    process and a throwaway config directory). agents: what the AI agents decided live (verify.replay_agents: the AI
    sites' offer decisions, the desk's margin, the declines), played through the same code; order: the grid operator's
    signed order for the day, when its agent placed one (it replaces the catalog's request there too)."""
    out = tempfile.mkdtemp(prefix="flexgrid-dcverify-")
    env = {**os.environ, "FLEXGRID_CONFIG_DIR": tempfile.mkdtemp(prefix="flexgrid-dcverify-cfg-")}
    cmd = [sys.executable, str(config.REPO_DIR / "tools" / "sim_datacenters.py"), "event", "--event", day, "--out", out]
    if agents:
        cmd += ["--agents", json.dumps(agents)]
    if order:
        cmd += ["--order", json.dumps(order)]
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=600, env=env)
    with open(os.path.join(out, f"dc_event_{day}.json")) as f:
        return float(json.load(f)["utility_pays_usd"])


def build_app(f: DCFleet):
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.middleware.gzip import GZipMiddleware
    from fastapi.responses import JSONResponse

    app = FastAPI(title="flexgrid-dc", version="0.3.0", dependencies=[doors.door()])
    app.add_middleware(GZipMiddleware, minimum_size=2000)           # the record crosses an SSH tunnel in the split

    def fail(exc: ValueError):
        code, _, msg = str(exc).partition(": ")
        raise HTTPException(int(code) if code in ("400", "404", "409") else 400, msg or str(exc))

    def market_only() -> None:
        if f.desk is None:
            raise HTTPException(404, "the desk and the grid operator run on the Mac (this is the data center campus)")

    @app.get("/health")
    def health():
        return f.health()

    @app.get("/dc/state")
    def state(day: str | None = None, raw: bool = False):
        """day=YYYY-MM-DD: the record about that day's event (the Control Room passes the homes' event day)."""
        return JSONResponse(f.state(day, raw=raw), headers={"Cache-Control": "no-store"})

    @app.get("/dc/today")
    def today():
        """Each data center agent's plan for today on MISO's real day-ahead prices (advice: nothing moves by itself)."""
        return JSONResponse(f.today(), headers={"Cache-Control": "no-store"})

    @app.post("/dc/today")
    def today_on(body: dict = Body(...)):
        """The same plan on a grid snapshot the caller already holds (the coordinator's GET /grid/now passes its own,
        so MISO is read once and the answer comes back at once)."""
        if not isinstance(body, dict) or not body.get("day_ahead"):
            raise HTTPException(400, "a grid_signals snapshot with day_ahead prices is needed")
        return JSONResponse(f.today(body), headers={"Cache-Control": "no-store"})

    @app.get("/dc/pins")
    def pins():
        """The data centers' public keys (the campus, or the all role)."""
        if not f.local_sites:
            raise HTTPException(404, "the data centers run on the campus")
        return f.pins()

    @app.post("/dc/trust")
    def trust(body: dict = Body(...)):
        """Campus only: the desk's and the grid operator's keys from the Mac (through its SSH tunnel)."""
        try:
            return f.accept_trust(body)
        except ValueError as exc:
            fail(exc)

    @app.post("/dc/standby")
    def standby(body: dict = Body(default={})):
        """Campus only: stop, forget the desk (the Mac takes the data centers back)."""
        if f.dc_role != "sites":
            raise HTTPException(409, "only a data center campus stands by")
        return f.standby()

    @app.post("/dc/role")
    def role(body: dict = Body(...)):
        """The Mac: {"role": "market", "campus_url": "http://127.0.0.1:7884"} (the data centers run on the campus) or
        {"role": "all"} (back here), between events."""
        try:
            return f.set_role(body.get("role", ""), body.get("campus_url"), bool(body.get("force")))
        except ValueError as exc:
            fail(exc)

    def campus_get(path: str):
        try:
            r = f.chttp.get(f.campus_url + path)
        except Exception as exc:
            raise HTTPException(502, f"the data center campus did not answer: {str(exc)[:120]}")
        if r.status_code >= 400:
            raise HTTPException(r.status_code, r.text[:300])
        return r.json()

    @app.get("/datacenters")
    def datacenters():
        if f.dc_role == "market":
            return campus_get("/datacenters")
        with f.lock:
            return [a.status() for a in f.agents.values()]

    @app.get("/datacenters/{name}")
    def datacenter(name: str):
        if f.dc_role == "market":
            return campus_get(f"/datacenters/{name}")
        with f.lock:
            a = f.agents.get(name)
            if not a:
                raise HTTPException(404, "unknown data center")
            return json.loads(json.dumps({**a.status(), "offers": a.offers, "contracts": a.contracts,
                                          "payments": a.payments, "telemetry": a.site.tel}, default=str))

    @app.get("/v2/summary")
    def summary():
        market_only()
        with f.lock:
            return {"now": f.desk.now(), "roster": len(f.desk.roster), "events": f.desk.summary(),
                    "reputation": f.desk.reputation}

    @app.get("/v2/events/{event_id}")
    def event(event_id: str, signed: bool = False):
        """The desk's view of one event; signed=true: with every signed meter reading and the pinned keys."""
        market_only()
        with f.lock:
            ev = f.desk.events.get(event_id)
            if not ev:
                raise HTTPException(404, "unknown event")
            if signed:
                return json.loads(json.dumps({**ev, "pins": dict(f.desk.pins)}, default=str))
            return json.loads(json.dumps({k: v for k, v in ev.items() if k != "readings"}, default=str))

    @app.get("/evidence/{event_id}")
    def evidence(event_id: str):
        """What the grid operator holds about one event (the same fields as the utility's GET /evidence)."""
        market_only()
        with f.lock:
            try:
                return json.loads(json.dumps(f.op.evidence(event_id), default=str))
            except KeyError:
                raise HTTPException(404, "no contract for that event")

    @app.get("/verify/{event_id}")
    def verify_event(event_id: str, replay: bool = False):
        """flexgrid_core/verify.py on a settled data center event; replay=true also re-runs the day offline (~20 s)."""
        market_only()
        with f.lock:
            ev = f.desk.events.get(event_id)
            if not ev:
                raise HTTPException(404, "unknown event")
            if ev.get("phase") != "done":
                raise HTTPException(409, f"not settled yet (phase {ev.get('phase')})")
            b = f.bundle(event_id)
        from . import verify
        try:
            ag = verify.replay_agents(b) if replay else None
        except ValueError as exc:                       # an agent decided, its offer does not say how
            raise HTTPException(409, str(exc))
        rt = replay_total(b["utility"]["need"]["window_start"][:10], ag) if replay else None
        r = verify.check(b, rt)
        return {**r, "report": verify.report(r)}

    @app.post("/admin/restart")
    def admin_restart(request: Request):
        """The Mac's restart (mac/restart_mac.sh), as the coordinator's POST /admin/restart: only from this Mac, with
        the header X-FlexGrid-Admin: restart, and only on macOS. Here too, because flexgrid-dc stays up when the
        coordinator cannot start."""
        from .coordinator import admin_restart as _restart
        return _restart(f.cfg, request, f._logger("flexgrid-dc"), "flexgrid-dc POST /admin/restart")

    @app.post("/dc/conditions")
    def conditions(body: dict = Body(default={})):
        """What if: conditions of the day for the grid operator's agent and the data centers (set_conditions)."""
        try:
            return f.set_conditions(body or {})
        except (TypeError, ValueError) as exc:
            raise HTTPException(400, f"conditions: {str(exc)[:160]}")

    @app.get("/dc/conditions")
    def conditions_get():
        return {"conditions": f.conditions, "now": f.now()}

    @app.post("/dc/needs")
    def presenter_order(body: dict = Body(...)):
        """The Control Room's data center request for a run: {"day": "YYYY-MM-DD", "mw": 10-100} (the program's window
        and day-ahead price), placed before the desk's 10:00 call."""
        market_only()
        try:
            datetime.strptime(str(body.get("day", "")), "%Y-%m-%d")
            mw = float(body.get("mw"))
        except (TypeError, ValueError):
            raise HTTPException(400, "day must be YYYY-MM-DD and mw a number")
        try:
            return f.presenter_order(str(body["day"]), mw, body.get("start"), body.get("end"))
        except ValueError as exc:
            fail(exc)

    @app.post("/dc/operator/order")
    def operator_order(body: dict = Body(...)):
        """The grid operator's AI agent orders a day's flexibility (in the background): {"day": "YYYY-MM-DD",
        "budget_s": 40}. Its decisions: GET /dc/state -> agents.operator."""
        market_only()
        try:
            datetime.strptime(str(body.get("day", "")), "%Y-%m-%d")
        except ValueError:
            raise HTTPException(400, "day must be YYYY-MM-DD")
        try:
            return f.operator_order(str(body["day"]), body.get("budget_s"))
        except ValueError as exc:
            fail(exc)

    @app.get("/contracts")
    def contracts():
        market_only()
        with f.lock, f.op.lock:
            return json.loads(json.dumps([{"transaction": t, "status": c.get("status"), "event_id": c.get("need", {}).get("event_id"),
                                           "meters": c.get("meters"), "offered": c.get("offered"),
                                           "settlement_usd": (c.get("settlement") or {}).get("total")}
                                          for t, c in f.op.contracts.items()], default=str))

    @app.get("/events")
    def events(n: int = 100):
        return [{k: v for k, v in x.items()} for x in list(f.events)[-n:]]

    @app.post("/demo/reset")
    def demo_reset(body: dict = Body(default={})):
        return f.reset()

    @app.post("/demo/clock")
    def demo_clock(body: dict = Body(default={})):
        return f.configure_clock(**body)

    return app


def main() -> None:
    ap = argparse.ArgumentParser(prog="flexgrid-dc")
    ap.add_argument("cmd", choices=["run"])
    a = ap.parse_args()
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    f = DCFleet()
    logs = config.REPO_DIR / "logs"
    logs.mkdir(exist_ok=True)
    if f.dc_role == "sites":                          # a Jetson's user journal may not persist: a log file of its own
        from logging.handlers import RotatingFileHandler
        fh = RotatingFileHandler(logs / "dc_campus.log", maxBytes=5_000_000, backupCount=3)
        fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(fh)
    wire.open_log(str(logs / "dc_wire.jsonl"))        # every Beckn message the parties here handle
    f.start_town()
    threading.Thread(target=f.loop, daemon=True).start()
    port = int(f.cfg.get("dc_port", 7874))
    log.info("flexgrid-dc (%s role): %d data centers here, desk %s, grid operator %s, transport %s; API "
             "http://127.0.0.1:%d", f.dc_role, len(f.agents), f.desk_name, f.op_name, f.transport_label(), port)
    uvicorn.run(build_app(f), host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()
