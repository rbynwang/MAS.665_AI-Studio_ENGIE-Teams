"""flexgrid-fleet - the community's simulated nodes, run as real agents in one process: the homes, or the buildings.

Every home in the dataset's fleet_v2.json with kind "replay" gets its own replay of its dataset trace, its own simulated
battery and EV, its own Ed25519 key, and its own NANDA Town identity, and runs the same HomeV2 code as a Jetson home (offer,
accept or decline, deliver, report). Nothing is shared between homes except the read-only dataset rows.

The buildings daemon is the same process with the building nodes (BuildingFleet): one process keeps all three in about
285 MB, and jetson-b has about 2 GB free. Each building also answers under /b/<name>, the base URL the Control Room polls.

Thinking nodes (cfg "thinking_homes", and a building with an agent): a node's agent (flexgrid_core/node) wakes before a sale
and when something leaves its limits. It runs on the node's own thread (Thinker), so a slow answer holds up nobody else.
  * NodeLock: the tick, the inbox and the agent take turns on a node. The agent lets go while it waits for the model, so
    the node's 5-minute controller keeps running.
  * Mind: the door to the model, `llm_slots` requests at a time, an exception first. The wait at the door doesn't count
    against a request's time limit, so 20 homes asking at once don't time out. The model stays on the Mac, and cfg llm_url
    is the tunnel to the brain queue in front of it (node/queue.py). Each request says which node asks, because they all
    reach the queue through one tunnel from one address.
  * Each decision goes to that node's chained log (node/record.py), under the config folder's state/decisions.
  * A node's answer can wait on the model far past Town's 30 s lease, so Town hands the message out again. A copy of a message
    it's still working on is left alone and one it finished is only acknowledged (Fleet._repeat): repeating a message must change
    nothing, and Town hears a message was done only when it was.
  * If the model is down the agent falls back to code by itself (node/agent.py), and the rule-based homes never ask it.

Memory: cfg lean_pack (or $FLEXGRID_PACK_DIR) reads the year from a pack (lean.py), which takes the fleet from 6.8 GB to 0.5,
or 1.6 with the 2026 forecasters.

  flexgrid-fleet run [--limit N]        # the homes: status API on 127.0.0.1:7872
  flexgrid-fleet run --buildings        # the buildings: status API on 127.0.0.1:7875
"""
from __future__ import annotations

import argparse
import contextlib
import heapq
import inspect
import itertools
import json
import logging
import math
import os
import resource
import sys
import threading
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime
from typing import Any, Callable

import httpx

from . import config, doors, keep, replay as replay_mod, rt_control, signing
from .adapters.simulated import SimulatedHome
from .home_v2 import HomeV2
from .node import record
from .replay import Replay
from . import beckn
from .town import TownSession

log = logging.getLogger("flexgrid-fleet")
KEEP_DECISIONS = 300           # what the status API keeps of a thinking node's decisions (all of them go to the file)
KEEP_SEEN = 512                # message ids a node remembers, so a repeat from Town changes nothing
TEXT_MAX = 400                 # the longest text an override may carry: the coordinator's own limit on a note
UNREAD = "the node's alarm file can't be read"
RETRY_S = (5.0, 15.0, 45.0, 120.0, 300.0)   # a batch of readings goes once, after the window: these waits outlast a dropped tunnel


def node_class(cfg: dict[str, Any]) -> type:
    """Which node speaks on this path, by the settings tools/sim_event.py uses: cfg buyers, report_rule."""
    if not cfg.get("buyers") and cfg.get("report_rule") != "exception":
        return HomeV2
    from .report_rule import PoolNode, QuietNode
    return QuietNode if cfg.get("report_rule") == "exception" else PoolNode


def memory_mb() -> dict[str, float]:
    """This process now and at its most, for /health: the numbers the step-by-step start on a Jetson watches."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1048576 if sys.platform == "darwin" else 1024)
    now = 0.0
    try:
        with open("/proc/self/statm") as f:
            now = int(f.read().split()[1]) * resource.getpagesize() / 1048576
    except OSError:
        try:
            import psutil
            now = psutil.Process().memory_info().rss / 1048576
        except ImportError:
            pass
    return {"now_mb": round(now, 1), "peak_mb": round(peak, 1)}


class NodeLock:
    """Whoever holds it may touch this node's state. A plain lock, plus let_go for the time spent waiting on the model."""

    def __init__(self) -> None:
        self._l = threading.Lock()
        self._who: int | None = None

    def __enter__(self) -> "NodeLock":
        self._l.acquire()
        self._who = threading.get_ident()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.leave()

    def leave(self) -> None:
        self._who = None
        self._l.release()

    def try_enter(self) -> bool:
        ok = self._l.acquire(blocking=False)
        if ok:
            self._who = threading.get_ident()
        return ok

    @contextlib.contextmanager
    def let_go(self):
        mine = self._who == threading.get_ident()
        if mine:
            self._who = None
            self._l.release()
        try:
            yield
        finally:
            if mine:
                self._l.acquire()
                self._who = threading.get_ident()


class Tagged:
    """An httpx client that adds {"kind", "node"} to a model request. The brain queue reads it, and Ollama ignores it. A request the
    queue refuses, or a model server that fails, comes back as an error, so agent.Brain takes it as "no answer" and the node falls
    back to code. A 400 passes: llm.py reads it, to drop a setting the model server doesn't know."""

    def __init__(self, http: httpx.Client, mine: threading.local):
        self.http, self.mine = http, mine

    def __getattr__(self, name: str) -> Any:
        return getattr(self.http, name)

    def post(self, url: str, json: dict[str, Any] | None = None, **kw: Any) -> httpx.Response:
        tag = getattr(self.mine, "tag", None)
        r = self.http.post(url, json={**json, "flexgrid": tag} if tag and json is not None else json, **kw)
        if r.status_code in (404, 429) or r.status_code >= 500:
            r.raise_for_status()
        return r


class Mind:
    """The process's one door to the model. Waiting here isn't counted against a request's own time limit."""
    RANK = {"exception": 0, "sale": 1, "report": 2}

    def __init__(self, cfg: dict[str, Any]):
        from .brain import OllamaBackend
        from .llm import LLM
        from .node.agent import MODEL, URL, Brain
        url, model = cfg.get("llm_url") or URL, cfg.get("llm_model") or MODEL
        self.mine = threading.local()
        self.brain = Brain(url, model)
        self.brain.http = Tagged(self.brain.http, self.mine)
        self.backend = OllamaBackend(LLM(url, model, label="fleet"))
        self.backend.llm.http = Tagged(self.backend.llm.http, self.mine)
        self.free = max(1, int(cfg.get("llm_slots", 1)))
        self.line: list[tuple[int, int]] = []
        self.seq = itertools.count()
        self.cv = threading.Condition()
        self.day_of: Callable[[str], str] = lambda node: ""      # the node's own day: the fleet sets it, so the queue's budgets follow the replay's clock

    @contextlib.contextmanager
    def turn(self, kind: str, node: str):
        me = (self.RANK.get(kind, 1), next(self.seq))
        with self.cv:
            heapq.heappush(self.line, me)
            while self.free <= 0 or self.line[0] != me:
                self.cv.wait(1.0)
            heapq.heappop(self.line)
            self.free -= 1
            self.cv.notify_all()
        self.mine.tag = {"kind": kind, "node": node, "day": self.day_of(node)}
        try:
            yield
        finally:
            self.mine.tag = None
            with self.cv:
                self.free += 1
                self.cv.notify_all()

    def status(self) -> dict[str, Any]:
        b = self.brain
        return {"url": b.url, "model": b.model, "calls": b.calls, "seconds": round(b.ms / 1000, 1), "tokens_in": b.tokens_in, "tokens_out": b.tokens_out,
                "errors": b.errors, "waiting": len(self.line)}


class NodeBrain:
    """The brain a node's agent holds (agent.Brain's ask and model): the shared model, in its turn, with the node let go."""

    def __init__(self, mind: Mind, lock: NodeLock, node: str):
        self.mind, self.lock, self.node = mind, lock, node
        self.model = mind.brain.model
        self.agent: Any = None

    def kind(self) -> str:
        """The queue's kind, from the step the agent's workflow is in."""
        steps = getattr(self.agent, "steps", None)
        step = steps[-1][0] if steps else ""
        return "exception" if step.startswith("8") else "report" if step.startswith("9") else "sale"

    def ask(self, system: str, tools: list[dict[str, Any]], text: str) -> dict[str, Any]:
        with self.lock.let_go(), self.mind.turn(self.kind(), self.node):
            return self.mind.brain.ask(system, tools, text)


class NodeBackend:
    """The same for brain.HomeBrain's offer and answer (steps 5 and 6)."""

    def __init__(self, mind: Mind, lock: NodeLock, node: str):
        self.mind, self.lock, self.node = mind, lock, node

    @property
    def model_tag(self) -> str | None:
        return self.mind.backend.model_tag

    def status(self) -> dict[str, Any]:
        return self.mind.backend.status()

    def decide(self, role: str, system: str, facts: dict[str, Any], schema: dict[str, Any], *, budget_s: float = 40.0, session: str | None = None):
        with self.lock.let_go(), self.mind.turn("sale", self.node):
            return self.mind.backend.decide(role, system, facts, schema, budget_s=budget_s, session=session)

    def converse(self, role: str, system: str, messages: list[dict[str, Any]], tools: Any, *, budget_s: float = 120.0, session: str | None = None):
        with self.lock.let_go(), self.mind.turn("sale", self.node):
            return self.mind.backend.converse(role, system, messages, tools, budget_s=budget_s, session=session)


class RecordedBackend(record.RecordingBackend):
    """record.RecordingBackend keeps decide(): brain.HomeBrain also asks its backend for status() and converse()."""

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class Thinker(threading.Thread):
    """A thinking node's own thread. It takes its mail first, then the newest step: a step it missed while busy isn't owed."""

    def __init__(self, fleet: "Fleet", name: str):
        super().__init__(daemon=True, name=f"think-{name}")
        self.fleet, self.node = fleet, name
        self.mail: deque = deque()
        self.want: str | None = None
        self.done: str | None = None
        self.wake = threading.Event()
        self.skipped = 0
        self.running = False                  # taking its mail or stepping: the clock waits for it (cfg clock_waits)

    def post(self, msg: dict[str, Any]) -> None:
        self.mail.append(msg)
        self.wake.set()

    def offer(self, ts: str | None) -> None:
        if ts and ts != self.want and ts != self.done:
            if self.want and self.want != self.done:
                self.skipped += 1
            self.want = ts
            self.wake.set()

    def run(self) -> None:
        while True:
            self.wake.wait()
            self.wake.clear()
            self.running = True
            try:
                while self.mail:
                    self.fleet.take(self.node, self.mail.popleft())
                ts = self.want
                if ts and ts != self.done:
                    self.done = ts
                    self.fleet.agent_step(self.node, ts)
            finally:
                self.running = False

    def busy(self) -> bool:
        """Still at work on what it was given: mail, or a step it hasn't finished."""
        return self.running or bool(self.mail) or bool(self.want and self.want != self.done)


class Fleet:
    kind = "homes"
    port_key, default_port, log_file = "fleet_port", 7872, "fleet.log"
    keep: Any = None                                      # keep.for_fleet: live only
    alarm_unsaved: dict[str, dict[str, Any]] = {}         # node -> what's waiting for the disk beside its alarm. Its own dict in __init__

    def __init__(self, limit: int | None = None):
        self.cfg = c = config.load()
        self.lean = None
        pack = c.get("lean_pack") or os.environ.get("FLEXGRID_PACK_DIR")
        if pack and self.kind == "homes":
            from . import lean
            self.lean = lean.install(pack)
            log.info("data from the pack: %s", self.lean)
        roster = self.roster()[:limit]
        tokens = json.loads((config.CONFIG_DIR / "secrets" / "fleet_tokens.json").read_text())
        self.coordinator_pub = signing.public_hex(signing.load_or_create(config.CONFIG_DIR / "secrets" / "signing.key")) \
            if c.get("role") == "coordinator" else None
        self.events = deque(maxlen=500)
        self.town_in: deque = deque(maxlen=4000)        # s from the sender making a message to a fleet home claiming it
        self.checks = self.claims = 0                     # Town notify checks and claims made (polling load on Town)
        self.seen: dict[str, dict[str, str]] = {}         # node -> the message ids it has finished, oldest first, and the status each came to. keep.py saves this
        self.taking: dict[str, dict[str, str]] = {}       # node -> the message ids it's still working on, and the fence of the last hand-out. Never saved: a process that dies drops them, so Town hands them out again and they run
        self.mail_lock = threading.Lock()                 # the poll thread and a thinker both move ids between those two
        self.repeats = 0                                  # messages Town handed out again and nobody acted on
        self.dropped = 0                                  # messages Town refused because their id held another message: /health shows it
        self.alarm_unsaved = {}
        self.kept = set(c.get("kept_days") or ())         # days a node doesn't step on: cfg kept_days, the benchmark's
        self.kept_passes = 0
        self.started = time.time()
        self.ready = False                                # the first pass over the nodes is over. Until then a reset waits on each node in turn: on 7 Nov the first pass took 166.6 s at full size,
                                                          # and two starts lost /demo/start's 20 s call to the fleet. /health says it, and the coordinator's start and the deploy wait for it
        self.pool = ThreadPoolExecutor(max_workers=int(c.get("send_workers", 8)))
        self.homes: dict[str, HomeV2] = {}
        self.towns: dict[str, TownSession] = {}
        self.locks: dict[str, NodeLock] = {}
        self.entries: dict[str, dict[str, Any]] = {}
        self.tick_s: deque = deque(maxlen=120)           # how long each pass over the nodes took, s
        self.tick_worst = 0.0                             # a sale's steps solve every home's plan: this must stay under a step
        self.busy_passes = 0                              # a node skipped in a pass because it was busy: it catches up next pass
        self.ticked: str | None = None                    # the step every node ticked at in the last whole pass: what the conductor waits for (cfg clock_waits)
        self.keydir = config.CONFIG_DIR / "fleet" / "keys"
        self.keydir.mkdir(parents=True, exist_ok=True)
        # one client for every session: each client holds a TLS context of its own
        self.http = httpx.Client(base_url=c["town_url"].rstrip("/"), timeout=40.0, limits=httpx.Limits(max_connections=64, max_keepalive_connections=32))
        for h in roster:
            name = h["name"]
            town = TownSession(c["town_url"], c["town_run_id"], name, tokens[name], http=self.http)
            self.towns[name] = town
            self.locks[name] = NodeLock()
            self.entries[name] = h
            self.homes[name] = self.make_node(h, lambda to, k, body, mid, t=town: self._send(t, to, k, body, mid))
        # thinking nodes
        self.mind: Mind | None = None
        self.damaged: list[tuple[str, str]] = []          # records read here that couldn't be used: keep.for_fleet makes each one an alarm
        self.agents: dict[str, Any] = {}
        self.channels: dict[str, Any] = {}                # a person's channel to each node, made on first use
        self.thinkers: dict[str, Thinker] = {}
        self.decisions: dict[str, deque] = {}
        self.logs: dict[str, Any] = {}
        think = self.thinking() if c.get("llm_enabled") else []
        if think:
            self.mind = Mind(c)
            self.mind.day_of = lambda n: self.homes[n].now_ts()[:10]
            self.version = record.model_version(self.mind.brain.model, self.mind.brain.url)
            for name in think:
                try:
                    self.decisions[name] = deque(maxlen=KEEP_DECISIONS)
                    self.logs[name] = record.Log(name, config.state_dir() / "decisions" / f"{name}.jsonl")
                    self.attach_agent(name)
                    self.thinkers[name] = Thinker(self, name)
                except Exception as exc:                  # noqa: BLE001  one node's damaged record must not stop the other 199. It runs on code, as a node with no agent does, and that's an alarm
                    log.exception("%s agent", name)
                    self.damaged.append((f"decisions/{name}", f"{type(exc).__name__}: {str(exc)[:120]}"))
                    self.homes[name].agent = self.homes[name].brain = None
                    for held in (self.decisions, self.logs, self.agents):
                        held.pop(name, None)
            log.info("%d thinking nodes on %s: %s", len(self.thinkers), self.mind.brain.url, sorted(self.thinkers))

    # ------------------------------------------------------------------ what differs from the buildings
    def roster(self) -> list[dict[str, Any]]:
        return [x for x in json.loads(config.data_path("fleet_v2.json").read_text())["homes"] if x["kind"] == "replay"]

    def make_node(self, h: dict[str, Any], send: Callable) -> HomeV2:
        c, name = self.cfg, h["name"]
        rp = Replay(h["profile"], c["replay_start"], c["replay_step_seconds"], c["replay_loop_days"])
        b = h["battery"]
        acfg = {"battery_kwh": b["kwh"], "battery_kw": b["kw"], "reserve_pct": b["reserve_pct"],
                "soc_max_pct": b["soc_max_pct"], "soc0_pct": b.get("soc0_pct", 55.0), "eta": b["eta"],
                "ev_max_kw": rp.sessions[0]["max_kw"] if rp.sessions else 0.0, "has_ev": h["has_ev"],
                "export_limit_kw": h["export_limit_kw"], "control": "auto", "forecast": c.get("forecast"),
                "opt_out_days": c.get("fleet_opt_out", {}).get(name, []),
                "name": name, "control_mode": c.get("control_mode", "fixed"),
                "owner_limits": c.get("owner_limits") or {}}
        sim = SimulatedHome(acfg, rp)
        self.owner_setup(name, sim)
        key = signing.load_or_create(self.keydir / f"{name}.key")
        return node_class(c)(name, acfg, sim, rp, key, send=send, trusted=self.trusted, log=self._logger(name))

    def owner_setup(self, name: str, sim: Any) -> None:
        """Each owner's own limits, from the dataset's owners.json, as tools/node_day.py sets them. A dataset without the file keeps the defaults."""
        o = self.owners().get(name)
        if o is None or sim.ctl is None:
            return
        lim = sim.ctl.limits
        lim.max_rise_c, lim.max_rise_away_c, lim.precool_c = o["max_rise_c"], o["max_rise_away_c"], o["precool_c"]
        sim.P_ch = float(o["charge_kw"]) if o.get("charge_kw") else sim.P_ch
        if "v2h" in (sim.app or {}):
            sim.app = {**sim.app, "v2h": {**sim.app["v2h"], "event_kwh": float(o["v2h_kwh"])}}

    def owners(self) -> dict[str, Any]:
        if not hasattr(self, "_owners"):
            try:
                self._owners = json.loads(config.data_path("owners.json").read_text())["homes"]
            except (OSError, ValueError, KeyError):
                self._owners = {}
        return self._owners

    def thinking(self) -> list[str]:
        """cfg thinking_homes: a list of names, or how many. A number takes every 10th home first, the benchmark's thinking homes."""
        want = self.cfg.get("thinking_homes") or 0
        if isinstance(want, list):
            return [n for n in want if n in self.homes]
        names = list(self.homes)
        return (names[9::10] + [n for i, n in enumerate(names) if (i + 1) % 10])[:int(want)]

    def log_arg(self, cls: type, name: str) -> dict[str, Any]:
        """An agent that takes a log writes its device calls to the node's log too (the context server's change to agent.py)."""
        return {"log": self.logs[name]} if "log" in inspect.signature(cls.__init__).parameters else {}

    def make_agent(self, name: str, brain: Any, record_: Callable[[dict[str, Any]], None]) -> Any:
        from .node.agent import NodeAgent
        return NodeAgent(self.homes[name], int(self.entries[name].get("tier") or 1), brain, record_, **self.log_arg(NodeAgent, name))

    def make_mind(self, name: str, backend: Any) -> Any:
        """brain.HomeBrain: steps 5 and 6, the offer and the answer."""
        from .brain import HomeBrain, Wishes
        c = self.cfg
        return HomeBrain(self.homes[name], backend, Wishes(config.state_dir() / f"{name}_wishes.json"), label=name,
                         log=self._logger(name), offer_budget_s=float(c.get("llm_offer_budget_s", 40)), accept_budget_s=float(c.get("llm_accept_budget_s", 45)),
                         chat_budget_s=float(c.get("llm_chat_budget_s", 120)))

    # ------------------------------------------------------------------ thinking nodes
    def attach_agent(self, name: str) -> None:
        """A new run starts each agent afresh: what it planned last time means nothing now. Its log goes on."""
        from .node import loans
        from .node import tools as T
        loans.DESK.node_lock = self.locks.__getitem__      # a loan touches another node's state, so it takes that node's lock
        lock, log_ = self.locks[name], self.logs[name]
        brain = NodeBrain(self.mind, lock, name)
        brain.agent = self.agents[name] = self.make_agent(name, record.RecordingBrain(brain, log_, T.PROMPT_VERSION, self.version, self.mind.brain.options), self._recorder(name))
        self.homes[name].brain = self.make_mind(name, RecordedBackend(NodeBackend(self.mind, lock, name), log_))

    def _recorder(self, name: str) -> Callable[[dict[str, Any]], None]:
        def f(rec: dict[str, Any]) -> None:
            self.decisions[name].append(rec)
            self.logs[name].decision(rec)
        return f

    def agent_step(self, name: str, ts: str) -> None:
        agent = self.agents.get(name)
        if agent is None:
            return
        with self.locks[name]:
            try:
                agent.step(ts)
            except Exception as exc:                      # the controller carries on without the agent
                log.exception("%s agent step", name)
                self.events.append({"at": time.time(), "home": name, "kind": "agent_error", "ts": ts, "error": str(exc)[:200]})
            if self.keep is not None:                     # what a step did to the guard's day, a plant's one stop among it, is on disk before anything else happens
                self.keep.node_done(name)

    def _send(self, town: TownSession, to: str, kind: str, body: dict[str, Any], mid: str) -> Future:
        """Sends go through a small pool so 200 homes reporting the same slot do not queue behind each other. The
        future tells the home whether the send went through: a meter reading is marked reported only then, and is
        sent again on the next tick if it failed (HomeV2._mark_reported). With two buyers each owns a copy of the meter, so
        a reading goes to both, the same signed body, as tools/sim_event.py's bus does. Nobody sends anything else again by itself (live, 6 Nov 2026: about 2 % of
        sends failed in bursts, and 1 to 7 offers of 203 were lost), so it retries here, under the same id: Town answers the same envelope as a replay."""
        buyers = self.cfg.get("buyers") or []
        m = body.get("message") or {}

        def run(dest: str, message_id: str, tries: int = 0):
            try:
                out = town.send(dest, kind, body, message_id)
            except Exception as exc:
                self.events.append({"at": time.time(), "home": town.name, "kind": "send_error", "to": dest,
                                    "message_kind": kind, "error": str(exc)[:200]})
                log.warning("%s: %s to %s failed, try %d: %s", town.name, kind, dest, tries + 1, str(exc)[:160])
                if ("performance" not in m or dest != to) and tries < len(RETRY_S):
                    t = threading.Timer(RETRY_S[tries], lambda: self.pool.submit(run, dest, message_id, tries + 1))
                    t.daemon = True
                    t.start()
                raise
            if isinstance(out, dict) and out.get("duplicate"):     # a retry here is the same envelope, which Town answers as a replay: this one went nowhere
                self.dropped += 1
                self.events.append({"at": time.time(), "home": town.name, "kind": "send_dropped", "to": dest, "message_kind": kind, "message_id": message_id})
                log.error("Town refused %s from %s to %s: id %s holds another message", kind, town.name, dest, message_id)
            return out
        if to in buyers and ("performance" in m or "batch" in m):
            for b in buyers:
                if b != to:
                    self.pool.submit(run, b, f"{mid}:{b}")
        if "batch" in m:              # report_rule.Outbox gives its one envelope the same id for every receiver, and Town takes an id once
            mid = f"{mid}:{to}"
        return self.pool.submit(run, to, mid)

    def trusted(self) -> list[str]:
        keys = list(config.load().get("coordinator_pubkeys") or [])
        return keys + ([self.coordinator_pub] if self.coordinator_pub else [])

    def _logger(self, name: str):
        shown = name == self.cfg.get("fleet_terminal_home", "hh-010")   # this home's steps go to the terminal too

        def f(kind: str, **d):
            self.events.append({"at": time.time(), "home": name, "kind": kind, **d})
            if shown:
                log.info("%s %s %s", name, kind, json.dumps(d, default=str)[:400])
        return f

    def configure_clock(self, only: str | None = None, **kw) -> dict[str, Any]:
        out = None
        for name, h in self.homes.items():
            if only and name != only:
                continue
            r = h.replay
            r.configure(kw.get("start", r.start), float(kw.get("step_seconds", r.step_seconds)),
                        int(kw.get("loop_days", r.loop_days)), kw.get("at"), kw.get("at_wall"), hold=bool(kw.get("hold")))
            if not kw.get("keep"):        # a new run starts each home afresh; a change of speed (keep) goes on with it
                h.adapter.idx = None
            out = r.clock()
        return out or {}

    def reset(self, only: str | None = None) -> None:
        for name, h in self.homes.items():
            if only and name != only:
                continue
            with self.locks[name]:
                h.reset()                                 # its play too: nothing it sends in the new run names a message of the last one
                self.channels.pop(name, None)
                if hasattr(h, "outbox"):                              # QuietNode: last run's readings don't go out in this one
                    h.outbox.held.clear()
                    h.outbox.first = None
                if name in self.thinkers:
                    self.attach_agent(name)
                    self.thinkers[name].want = self.thinkers[name].done = None
                    ctx = getattr(self.agents[name], "ctx", None)
                    if ctx is not None:                       # the same days again: its log must not answer "done already"
                        ctx.new_run(keep_alarm=True)          # this door is unsigned, so an alarm stays on through it

    def person_word(self, name: str, msg: Any) -> dict[str, Any]:
        """A person's signed word: the one way an alarm ends on a live node, and the one way a limit goes up. cfg person_keys names each role's public key. With none on file nothing passes here."""
        keys = self.cfg.get("person_keys") or {}
        if not keys:
            return {"status": "blocked", "why": "No person's key is on file here."}
        with self.locks[name]:
            ch = self.channels.get(name)
            if ch is None:
                ctx = getattr(self.agents.get(name), "ctx", None)
                seq, save = doors.numbers(name, keys.values()), lambda n: doors.note(name, seq=n)
                ch = self.channels[name] = ctx.owner_channel(keys, seq, save) if ctx is not None else code_channel(self.homes[name], self.kind, keys, seq, save)
            out = ch.receive(msg)
            if out.get("status") == "done":
                self.alarm_kept(name, **doors.limit_said(name, msg))
            return out

    def alarm_kept(self, name: str, **more: Any) -> None:
        """What the node's overrides say of an alarm, on disk before anyone hears it was taken. keep.py's pass is up to 5 s behind and its files go with their run or their age, and
        an alarm must do neither. A disk that won't take it doesn't stop the alarm: it stands in memory, /health says so, and each pass of the tick tries again."""
        ov = self.homes[name].adapter.overrides
        more = {**self.alarm_unsaved.get(name, {}), **more}
        try:
            doors.note(name, hold=ov.get("alarm_hold"), signal=ov.get("alarm"), **more)
            self.alarm_unsaved.pop(name, None)
        except OSError as exc:
            if name not in self.alarm_unsaved:
                log.error("%s: an alarm isn't reaching the disk: %s", name, exc)
                self.events.append({"at": time.time(), "home": name, "kind": "alarm_not_saved", "error": str(exc)[:160]})
            self.alarm_unsaved[name] = more

    def person_back(self, keeper: Any) -> None:
        """After keep.py has put the nodes back, and before any tick: a state file can be older than an alarm or a person's limit, or gone, so it has no say in them. The node's own file
        has, and once that exists it's the whole truth, or a state file from just before a person's word would hold the node again. With no file yet the state still counts: it's how an
        alarm from older code comes across. A thinking node's log counts as well, because it's on disk at once. A file that can't be read may have held an alarm, so the node stays held
        until a person says. Whatever is wrong with one node, the others come back and the service starts."""
        for name, h in self.homes.items():
            if not hasattr(h.adapter, "overrides"):
                continue
            try:
                self._person_back(name, h.adapter, keeper)
            except Exception as exc:                      # noqa: BLE001
                log.exception("%s: what a person said didn't come back", name)
                keeper.bad.append(f"person/{name}")
                keeper.event("state_set_aside", file=f"person/{name}", why=f"{type(exc).__name__}: {str(exc)[:120]}")
                h.adapter.hold_alarm(UNREAD)

    def _person_back(self, name: str, a: Any, keeper: Any) -> None:
        ctx = getattr(self.agents.get(name), "ctx", None)
        b = doors.book(name)
        if b and "bad" in b:
            keeper.bad.append(f"person/{name}")
            keeper.event("state_set_aside", file=f"person/{name}", why=b["bad"])
            hold, signal = UNREAD, None
        elif b is None:
            hold, signal = a.overrides.get("alarm_hold"), a.overrides.get("alarm")
        else:
            hold, signal = b.get("hold"), b.get("signal")
        hold = hold or getattr(ctx, "stopped", None)
        if hold:
            a.overrides.pop("alarm", None)
            if ctx is not None:
                ctx.alarm(hold)
            a.hold_alarm(hold)
            if signal:
                a.overrides["alarm"] = signal
            self.alarm_kept(name)
        elif "alarm_hold" in a.overrides:                 # the state is older than a person's word
            for k in ("alarm", "alarm_hold", "stop"):
                a.overrides.pop(k, None)
        for group, limits in ((b or {}).get("limits") or {}).items():
            for k, v in limits.items():
                rt_control.write_limit(a, group, k, v)

    def held(self) -> bool:
        """cfg kept_days: the days the benchmark keeps unseen. The clock runs on after a debug day, and the nodes don't step into one.
        The benchmark's own config leaves the list out."""
        h = next(iter(self.homes.values()), None)
        return bool(self.kept) and h is not None and h.now_ts()[:10] in self.kept

    def tick_loop(self) -> None:
        while True:
            t0 = time.time()
            if self.held():
                self.kept_passes += 1
                self.ready = True                         # no node steps on a kept day, so there's no pass to wait for
                time.sleep(1.0)
                continue
            whole = True
            for name, h in list(self.homes.items()):
                lock = self.locks[name]
                if not lock.try_enter():
                    self.busy_passes += 1
                    whole = False
                    continue
                try:
                    h.tick()
                except Exception as exc:
                    self.events.append({"at": time.time(), "home": name, "kind": "tick_error", "error": str(exc)[:200]})
                finally:
                    lock.leave()
                th = self.thinkers.get(name)
                if th is not None:
                    th.offer((h.adapter.tel or {}).get("ts"))
            for name in list(self.alarm_unsaved):
                self.alarm_kept(name)
            seen = {(h.adapter.tel or {}).get("ts") for h in self.homes.values()}
            self.ticked = next(iter(seen)) if whole and len(seen) == 1 else None
            self.ready = True
            d = round(time.time() - t0, 3)
            self.tick_s.append(d)
            self.tick_worst = max(self.tick_worst, d)
            time.sleep(max(0.2, 1.0 - (time.time() - t0)))

    def take(self, name: str, msg: dict[str, Any]) -> None:
        """One claimed message: handled under the node's lock, then acknowledged."""
        town, h = self.towns[name], self.homes[name]
        with self.locks[name]:
            try:
                note = h.handle(msg.get("kind", ""), msg.get("from"), msg.get("body") or {})
            except Exception as exc:
                note = beckn.Refused(f"error: {exc}")
                log.exception("%s handle", name)
            status, said = beckn.ack_of(note)
            fence = self._done(name, msg, status)         # a repeat of it says the same: _repeat
            if self.keep is not None:                     # on disk before Town hears it: if this process dies here, Town hands the message out again
                self.keep.node_done(name)
        try:
            town.ack(msg["message_id"], fence, status, said)
        except Exception as exc:                          # Town hands it out again after its lease; the answer is kept
            self.events.append({"at": time.time(), "home": name, "kind": "town_error", "error": f"ack: {str(exc)[:160]}"})

    def _done(self, name: str, msg: dict[str, Any], status: str) -> str:
        """This message is finished: it's seen now, and not before. Returns the fence of its last hand-out, the one Town will take an ack on."""
        mid = msg["message_id"]
        with self.mail_lock:
            seen = self.seen.setdefault(name, {})
            seen[mid] = status
            if len(seen) > KEEP_SEEN:
                seen.pop(next(iter(seen)))
            return self.taking.setdefault(name, {}).pop(mid, None) or msg.get("fence")

    def _repeat(self, name: str, msg: dict[str, Any]) -> bool:
        """True for a message that is not to be acted on now. Town delivers at least once, and hands a message out again when its lease runs out. A thinking node finishes a message
        only after the model answers, so that is usual here: on real Qwen a commitment was answered twice, and an rt_dispatch handled twice would add its kW twice.
        Finished, the copy is acknowledged with its new fence, so Town stops. It says what the first one came to, so a message refused and handed out again is on Town's record as refused.
        Still in the mail, it's left alone. Town keeps it until the work is done, because a kill here would lose the mail, and only this new fence can acknowledge it then."""
        mid = msg.get("message_id")
        with self.mail_lock:
            seen, taking = self.seen.setdefault(name, {}), self.taking.setdefault(name, {})
            done, working, said = mid in seen, mid in taking, seen.get(mid)
            if not done:
                taking[mid] = msg.get("fence")
            if not done and not working:
                return False
            self.repeats += 1
        if done:
            try:
                self.towns[name].ack(mid, msg["fence"], said or "processed", {"note": "a repeat: already taken"})
            except Exception as exc:                      # it comes back once more, and is acknowledged then
                self.events.append({"at": time.time(), "home": name, "kind": "town_error", "error": f"ack: {str(exc)[:160]}"})
        return True

    def _drain(self, name: str) -> int:
        """Claims until the inbox is empty; returns the messages handled. A thinking home takes its mail on its own thread,
        because an offer may wait on the model and this thread serves other homes too."""
        town, th = self.towns[name], self.thinkers.get(name)
        n = 0
        while True:
            msg = town.claim()
            self.claims += 1
            if not msg:
                return n
            n += 1
            if self._repeat(name, msg):
                continue
            sent = ((msg.get("body") or {}).get("context") or {}).get("sentWall")
            if isinstance(sent, (int, float)):
                self.town_in.append(max(0.0, time.time() - sent))
            if th is not None:
                th.post(msg)
            else:
                self.take(name, msg)
                if msg["message_id"] in self.taking.get(name, ()):      # a Fleet with its own take, as Town's track_seller.py has: back here, it's finished
                    self._done(name, msg, "processed")

    def poll_loop(self, names: list[str]) -> None:
        """One worker polls a share of the homes' inboxes in turn (a quick, non-blocking check each), instead of 200
        long-polls held open at once: NANDA Town 0.2 re-checks each open long-poll every 50 ms inside a write
        transaction (a scan of its whole notifications table), and records every request, so polling load is Town
        load. A home is checked every `fleet_poll_seconds` (5 s) for a minute after it last got mail, every
        `fleet_poll_idle_seconds` (15 s) otherwise: a call for offers waits at most 15 s (1.25 steps at 12 s a step;
        offers are due 6 replay hours later), and an idle fleet of 200 costs Town 13 checks a second."""
        every = float(self.cfg.get("fleet_poll_seconds", 5.0))
        idle = float(self.cfg.get("fleet_poll_idle_seconds", 15.0))
        sweep = float(self.cfg.get("fleet_sweep_seconds", 60.0))   # a hint is only a wake-up call: a claim pass this often anyway, as InboxPoller's sweep
        last_mail = dict.fromkeys(names, 0.0)
        swept = dict.fromkeys(names, 0.0)                          # 0: a first pass at once, for what came while this process was away
        due = {n: time.time() + k * idle / max(1, len(names)) for k, n in enumerate(names)}   # spread the checks
        while True:
            t0 = time.time()
            for name in names:
                if t0 < due[name]:
                    continue
                try:
                    self.checks += 1
                    # Town keeps a message until it's acknowledged, but it wakes a receiver once. Waiting for a hint alone left a message whose hint was lost, or
                    # one handed out again after a claim that was never acknowledged (a node that died), in the inbox until the next message came
                    if self.towns[name].notify(wait=0) or t0 - swept[name] >= sweep:
                        swept[name] = t0
                        if self._drain(name):
                            last_mail[name] = time.time()
                except Exception as exc:
                    swept[name] = 0.0                              # the check may have used up a hint: a pass on the next round
                    self.events.append({"at": time.time(), "home": name, "kind": "town_error", "error": str(exc)[:200]})
                due[name] = time.time() + (every if time.time() - last_mail[name] < 60.0 else idle)
            time.sleep(max(0.2, min(due.values()) - time.time()))

    def join_all(self) -> None:
        """Joined one after another, in the background: the status API stays up while Town is away, and a session that
        didn't join here joins at its first call."""
        for name, t in self.towns.items():
            try:
                t.join()
            except Exception as exc:
                self.events.append({"at": time.time(), "home": name, "kind": "town_error", "error": f"join: {str(exc)[:160]}"})
            time.sleep(0.02)

    def warm(self) -> None:
        """Makes each home's trained forecaster now, not at the first call for offers, so the fleet's real memory shows at the start and
        200 homes aren't all building one while offers are due. It reads no more than the first offer would, and changes no result.
        It holds no node's lock: a reset waiting behind this took over 20 s at full size, so /demo/start lost the fleet. A step that gets
        there first builds its own, and one of the two stays."""
        if self.kind != "homes" or self.cfg.get("forecast") not in ("trained", "trained_2026"):
            return
        for name, h in self.homes.items():
            try:
                h._models()
            except Exception as exc:                          # the home's own forecast path reports it and falls back
                self.events.append({"at": time.time(), "home": name, "kind": "forecast_error", "error": str(exc)[:200]})

    def start(self) -> None:
        threading.Thread(target=self.warm, daemon=True, name="warm").start()
        threading.Thread(target=self.join_all, daemon=True, name="town-join").start()
        threading.Thread(target=self.tick_loop, daemon=True, name="tick").start()
        for th in self.thinkers.values():
            th.start()
        names = list(self.homes)
        workers = min(len(names), max(1, int(self.cfg.get("fleet_poll_workers", 8))))
        for k in range(workers):
            threading.Thread(target=self.poll_loop, args=(names[k::workers],), daemon=True, name=f"poll-{k}").start()
        log.info("%d %s running", len(self.homes), self.kind)


class BuildingFleet(Fleet):
    """The building nodes in one process: the same protocol code and threads, a building's levers (building.py). Their
    identities come from secrets/fleet_tokens.json too, so the two daemons are set up the same way."""
    kind = "buildings"
    port_key, default_port, log_file = "buildings_port", 7875, "buildings.log"

    def roster(self) -> list[dict[str, Any]]:
        from . import building
        return building.enabled({"buildings": self.cfg.get("buildings") or "all"})

    def make_node(self, b: dict[str, Any], send: Callable) -> HomeV2:
        from . import building
        c, name = self.cfg, b["name"]
        rp = Replay(b["profile"], c["replay_start"], c["replay_step_seconds"], c["replay_loop_days"])
        acfg = {"agent_name": name, "name": name, "building": name, "control": "auto", "forecast": c.get("forecast"), "control_mode": "rolling", "opt_out_days": [],
                "owner_limits": c.get("owner_limits") or {}}
        sim = building.SimulatedBuilding(acfg, rp)
        key = signing.load_or_create(self.keydir / f"{name}.key")
        return node_class(c)(name, acfg, sim, rp, key, send=send, trusted=self.trusted, log=self._logger(name))

    def thinking(self) -> list[str]:
        """A building thinks once its agent exists: cfg thinking_buildings, default every building."""
        want = self.cfg.get("thinking_buildings", True)
        return [n for n in self.homes if want is True or (isinstance(want, list) and n in want)] if self.agent_class() else []

    @staticmethod
    def agent_class() -> Any:
        """THE HOOK. The building's agent, a NodeAgent subclass, from the buildings work: BuildingAgent(node, brain, record).
        Until that file exists the buildings run on their code: controller, guard and rule-based offers, and /health says so."""
        try:
            from .node.building_agent import BuildingAgent
        except ImportError:
            return None
        return BuildingAgent

    def make_agent(self, name: str, brain: Any, record_: Callable[[dict[str, Any]], None]) -> Any:
        cls = self.agent_class()
        return cls(self.homes[name], brain, record_, **self.log_arg(cls, name))

    def make_mind(self, name: str, backend: Any) -> Any:
        """brain.BuildingBrain: steps 5 and 6, as a home's. It can only lower what the planner offered."""
        from .brain import BuildingBrain, FacilityPolicy
        c = self.cfg
        return BuildingBrain(self.homes[name], backend, FacilityPolicy(config.state_dir() / f"{name}_policy.json"), label=name,
                             log=self._logger(name), offer_budget_s=float(c.get("llm_offer_budget_s", 40)), accept_budget_s=float(c.get("llm_accept_budget_s", 45)),
                             chat_budget_s=float(c.get("llm_chat_budget_s", 120)))


def lean_stats() -> dict[str, Any]:
    from . import lean
    return lean.stats()


def code_channel(h: Any, kind: str, keys: dict[str, str], seq: dict[str, int] | None = None, save: Callable[[dict[str, int]], Any] | None = None) -> Any:
    """A person's channel on a node with no agent: an alarm's end, and a limit, which the controller holds by itself. Closing a group isn't here. Without levers the controller
    uses every device it has, and an offer would go on counting what a person closed, so a closed group needs an agent."""
    from .node import policy as P

    def apply(verb: str, group: str, name: str, value: float | None) -> dict[str, Any]:
        ov = h.adapter.overrides
        if verb == "set_limit":
            rt_control.write_limit(h.adapter, group, name, value)
            return {"status": "done"}
        if verb != "clear_alarm":
            return {"status": "blocked", "why": "This node has no agent, so it takes a limit or an alarm's end, and nothing else yet."}
        if ov.get("alarm"):
            return {"status": "blocked", "why": "The alarm is still on."}
        if "alarm_hold" not in ov:
            return {"status": "blocked", "why": "No alarm is holding this node."}
        ov.pop("alarm_hold")
        h.adapter.rt_override("stop", False)
        return {"status": "done"}
    return P.OwnerChannel(h.name, P.Policy("building" if kind == "buildings" else "home"), keys, apply, h.log, seq, save)


def comfort_ok(value: Any) -> bool:
    """A comfort limit from outside, inside the range policy.py gives a person. The controller takes these numbers as they come, and this
    door isn't the signed owner channel, so nothing here may go past that range or carry another name."""
    from .node.policy import LIMITS
    rng = LIMITS["hvac_zones"]
    return isinstance(value, dict) and bool(value) and all(k in rng and isinstance(v, (int, float)) and not isinstance(v, bool) and rng[k][0] <= v <= rng[k][1] for k, v in value.items())


def raises_a_limit(a: Any, kind: str, value: Any) -> bool:
    """Would this value take a person's limit up? A limit goes up only by that person's signed word, because the guard trusts this number. This door may only bring one down, token
    or no token: what's lower can't hurt anyone, and a clear goes back to what the file or a signed word says."""
    if kind == "comfort" and getattr(a, "ctl", None) is not None:
        return any(float(v) > rt_control.person_limit(a, k) for k, v in value.items())
    return kind == "lights" and float(value) > a.light_limit()


def plain(value: Any) -> bool:
    """Plain text someone might say: a signal's name, a note. Not empty, and short enough to go on the record."""
    return isinstance(value, str) and bool(value.strip()) and len(value) <= TEXT_MAX


def override_ok(kind: str, value: Any) -> bool:
    """What each override may carry through this door. A switch is true or plain text, and false or null to end it. Anything else is refused and changes nothing,
    so what reaches rt_override means what it says. What freed an alarm was a value that was neither: "", [] or {}."""
    if kind == "comfort":
        return comfort_ok(value)
    if kind == "lights":
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    if kind == "note":
        return value is None or value is False or plain(value)
    if kind == "depart_at":
        try:
            return plain(value) and bool(datetime.strptime(value, "%Y-%m-%d %H:%M"))
        except ValueError:
            return False
    return value is None or isinstance(value, bool) or plain(value)


def build_app(f: Fleet):
    from fastapi import Body, FastAPI, HTTPException

    app = FastAPI(title="flexgrid-fleet", version="0.3.0", dependencies=[doors.door()])

    @app.get("/health")
    def health():
        any_home = next(iter(f.homes.values()), None)
        v = sorted(f.town_in)
        q = (lambda p: round(v[min(len(v) - 1, int(round(p * (len(v) - 1))))], 2)) if v else None   # noqa: E731
        ticks = sorted(f.tick_s)
        out = {"ok": True, "ready": f.ready, "kind": f.kind, "homes": len(f.homes), "clock": any_home.replay.clock() if any_home else None,
               "town_delivery_s": {"n": len(v), "p50": q(0.5), "p95": q(0.95), "max": round(v[-1], 2)} if v else {"n": 0},
               "town_calls": {"notify": f.checks, "claim": f.claims,
                              "uptime_s": round(time.time() - f.started)},
               "memory": memory_mb(), "pack": f.lean, "rows": lean_stats() if f.lean else None, "tick_s": {"p50": ticks[len(ticks) // 2], "max": ticks[-1], "worst": f.tick_worst} if ticks else None, "busy_passes": f.busy_passes,
               "repeats": f.repeats, "dropped": f.dropped, "kept_passes": f.kept_passes, "thinking": sorted(f.thinkers), "model": f.mind.status() if f.mind else None, "doors": doors.state()}
        if f.kind == "buildings":
            out["agent"] = "ready" if BuildingFleet.agent_class() else "missing: flexgrid_core/node/building_agent.py, so the buildings run on code"
        out["held"] = keep.held(f)                                                                                                                  # who waits for a person's word
        out["person_keys"] = sorted(f.cfg.get("person_keys") or ())                                                                                 # whose word can end it
        kept = keep.health(f.keep)
        if f.alarm_unsaved:
            kept = {**kept, "ok": False, "alarms": [*kept.get("alarms", ()), {"what": "alarm_not_saved", "why": f"{len(f.alarm_unsaved)} nodes' alarms aren't on disk, so a restart would lose them", "since": None}]}
        return {**out, **kept}

    @app.get("/homes")
    def homes():
        return [h.status() for h in f.homes.values()]

    @app.get("/telemetry")
    def telemetry(since: str | None = None):
        """Every home's latest telemetry in one small answer, for the Control Room on the Mac (simulation truth).
        With `since` (a dataset time), also each home's readings of the steps after it, at most 12: the steps a poll
        every few seconds would miss while the live clock runs fast."""
        from .monitor import TEL_KEYS
        keys = list(TEL_KEYS) + ["E"]                   # compact: one list of values per home (~8 kB for 98 homes)
        out: dict[str, Any] = {}
        recent: dict[str, Any] = {}
        for n, h in f.homes.items():
            t = h.adapter.tel or {}
            out[n] = [t.get(k) for k in TEL_KEYS] + [h.adapter.E]
            if since:
                rr = sorted(list((getattr(h.adapter, "recent", None) or {}).items()))[-13:]
                got = [x for _, x in rr if (x.get("ts") or "") > since]
                if len(got) > 1:
                    recent[n] = [[x.get(k) for k in TEL_KEYS] + [h.adapter.E] for x in got[-12:]]
        any_home = next(iter(f.homes.values()), None)
        res = {"now": any_home.now_ts() if any_home else None, "keys": keys, "homes": out}
        if since:
            res["recent"] = recent
        return res

    @app.get("/homes/{name}")
    def home(name: str):
        h = f.homes.get(name)
        if not h:
            raise HTTPException(404, "unknown home")
        return {**h.status(), "offers": h.offers, "contracts": h.contracts, "payments": h.payments,
                "telemetry": h.adapter.tel}

    @app.get("/agents")
    def agents():
        """The thinking nodes at a glance: what each is running now, and its workflow so far."""
        out = {}
        for n, a in f.agents.items():
            th = f.thinkers.get(n)
            out[n] = {"steps": [list(s) for s in a.steps[-12:]], "calls_running": len(a.calls), "levers": a.sim.levers, "decisions": len(f.decisions.get(n) or ()),
                      "reports": len(a.reports), "steps_skipped": th.skipped if th else 0,
                      "fallbacks": dict(getattr(a, "fallbacks", {})) | dict(getattr(f.homes[n].brain, "fallbacks", None) or {}),
                      "flags": dict(getattr(a, "flags", {})), "receipts": len(getattr(a, "receipts", ())), "told": len(getattr(a, "told", ()))}
        any_home = next(iter(f.homes.values()), None)
        busy = sorted(n for n, th in f.thinkers.items() if th.busy()) + sorted(n for n, ids in f.taking.items() if ids and n not in f.thinkers)
        return {"model": f.mind.status() if f.mind else None, "nodes": out,
                "clock": {"now": any_home.replay.ts(any_home.replay.current()) if any_home else None, "held": bool(any_home and any_home.replay.held),
                          "ticked": f.ticked, "busy": busy}}

    @app.get("/agents/{name}/receipts")
    def agent_receipts(name: str):
        """Every node an agent: each decision's receipt, and what the owner or manager was told (node/agent.py)."""
        a = f.agents.get(name)
        if a is None:
            raise HTTPException(404, "this node has no agent")
        return {"receipts": list(getattr(a, "receipts", ())), "told": list(getattr(a, "told", ())), "flags": dict(getattr(a, "flags", {})),
                "fallbacks": dict(getattr(a, "fallbacks", {})) | dict(getattr(f.homes[name].brain, "fallbacks", None) or {})}

    @app.get("/agents/{name}/decisions")
    def agent_decisions(name: str, n: int = 50):
        if name not in f.decisions:
            raise HTTPException(404, "this node has no agent")
        return list(f.decisions[name])[-n:]

    @app.get("/events")
    def events(n: int = 100):
        return list(f.events)[-n:]

    @app.get("/logs/tail")
    def logs_tail(after: int = -1):
        """The fleet's terminal (logs/fleet.log) from byte `after` on, for the Control Room's three terminals."""
        from . import wire
        return wire.tail(config.REPO_DIR / "logs" / f.log_file, after)

    @app.post("/demo/reset")
    def demo_reset():
        f.reset()
        scenario_reset()
        return {"ok": True, "homes": len(f.homes)}

    def scenario_reset() -> None:
        replay_mod.set_weather(1.0, 1.0, dT=0.0)
        for h in f.homes.values():
            h.adapter.set_battery_fault(False)
            if hasattr(h.adapter, "overrides"):
                h.adapter.rt_override("clear")          # an alarm stays: this door has no signature

    @app.post("/person/{name}")
    def person_word(name: str, body: dict = Body(default={})):
        """A person's signed word for one node (policy.signed): the end of an alarm, a limit, and on a node with an agent a group closed or opened. This is the door Fleet.person_word's docstring speaks of."""
        if name not in f.homes:
            raise HTTPException(404, "no such node here")
        return f.person_word(name, body)

    @app.post("/demo/scenario")
    def demo_scenario(body: dict = Body(default={})):
        """What if (the Control Room), for the homes here: {"weather": {"pv": 0.3, "load": 1.2}} from now on,
        {"battery_fault": {"n": 10, "on": true}} (n homes with a battery, picked the same way every time) or
        {"battery_fault": {"homes": [...], "on": false}}, {"reset": true}. Physical truth only: each home's agent
        reacts with its own code."""
        out: dict[str, Any] = {"homes": len(f.homes)}
        any_home = next(iter(f.homes.values()), None)
        now = any_home.replay.ts(any_home.replay.current()) if any_home else None
        if body.get("reset"):
            scenario_reset()
            out["reset"] = True
        w = body.get("weather")
        if isinstance(w, dict):                           # from now on, or from a step a test runner names ("from")
            frm = str(w.get("from") or now)
            to = beckn.add_minutes(frm, 60 * float(w["hours"])) if w.get("hours") else None
            out["weather"] = replay_mod.set_weather(w.get("pv"), w.get("load"), frm, to=to, dT=w.get("dT"))
        ov = body.get("override")
        if isinstance(ov, dict):                          # an owner's wish (the Control Room, or the phone app)
            name, k = ov.get("home"), ov.get("kind")
            kinds = ("comfort", "lights", "stop", "clear", "alarm") if f.kind == "buildings" else ("ev_now", "comfort", "stop", "depart_at", "clear", "note", "alarm")
            if not isinstance(name, str) or name not in f.homes or k not in kinds:
                raise HTTPException(400, f"override: a node here and kind {'|'.join(kinds)}")
            value = ov.get("value", "alarm" if k == "alarm" else True)
            if k == "comfort" and not comfort_ok(value):
                raise HTTPException(400, "override: comfort takes max_rise_c, max_rise_away_c or precool_c, inside policy.py's range")
            if not override_ok(k, value):
                raise HTTPException(400, f"override: {k} doesn't take that value. A switch is true, plain text, false or null")
            a = f.homes[name].adapter
            if not hasattr(a, "rt_override"):
                raise HTTPException(409, f"{name} has no real-time control")
            if raises_a_limit(a, k, value):
                raise HTTPException(403, f"override: a limit goes up only by a person's signed word, through POST /person/{name}")
            # an alarm's input: {"kind": "alarm", "value": "smoke"}, and value false when the signal ends. Neither frees the node: only the signed door does that
            a.rt_override(k, value)
            if k == "alarm":
                f.alarm_kept(name)
            f.homes[name].log("owner_override", override=k, value=value, source=ov.get("source", "control room"))
            out["override"] = {"home": name, "kind": k, "overrides": dict(a.overrides), "mode": getattr(a, "mode", "fixed")}
            if "alarm_hold" in a.overrides:
                out["held"] = f"an alarm holds {name} until a person clears it through POST /person/{name}"
        for k in ("closed", "zone_event"):                # a building closes for the day, or holds an event in a zone it sold (node/surprises.py)
            sp = body.get(k)
            if isinstance(sp, dict):
                name = sp.get("home")
                if f.kind != "buildings" or name not in f.homes:
                    raise HTTPException(400, f"{k}: a building here")
                from .node import surprises
                h = f.homes[name]
                t = str(sp.get("at") or now)              # now, or the step a test runner names: it posts while every clock waits one step before
                with f.locks[name]:
                    out.update(surprises.close_building(h, t, sp.get("share")) if k == "closed" else
                               surprises.zone_event(h, t, float(sp.get("share", 0.25)), float(sp.get("hours", 3.5)), sp.get("space")))
                h.log("what_if", what=k, **{x: v for x, v in sp.items() if x != "home"})
        bf = body.get("battery_fault")
        if isinstance(bf, dict):
            on = bool(bf.get("on", True))
            names = [n for n in (bf.get("homes") or []) if n in f.homes]
            if not names and bf.get("n"):
                with_batt = [n for n, h in f.homes.items() if h.adapter.E > 0]
                step = max(1, len(with_batt) // max(1, int(bf["n"])))
                names = with_batt[::step][:int(bf["n"])]
            for n in names:
                f.homes[n].adapter.set_battery_fault(on)
            out["battery_fault"] = {"on": on, "homes": names}
            f.events.append({"t": time.time(), "kind": "what_if_battery_fault", "on": on, "homes": len(names)})
        out["faulted"] = [n for n, h in f.homes.items() if h.adapter.E > 0 and h.adapter.P == 0]
        out["now"] = now
        return out

    @app.get("/demo/clock")
    def demo_clock_get():
        any_home = next(iter(f.homes.values()), None)
        return any_home.replay.clock() if any_home else {}

    @app.post("/demo/clock")
    def demo_clock(body: dict = Body(default={})):
        return f.configure_clock(**body)

    # ---- one base URL per building, for the Control Room: building.monitor_url gives http://.../b/<name>
    if f.kind == "buildings":
        def node(name: str) -> HomeV2:
            if name not in f.homes:
                raise HTTPException(404, "no such building here")
            return f.homes[name]

        @app.get("/b/{name}/health")
        def b_health(name: str):
            h = node(name)
            return {"ok": True, "ready": f.ready, "agent": name, "adapter": "building", "town": True, "replay": {"now": h.replay.ts(h.replay.current())}, "memory": memory_mb(), **keep.health(f.keep)}

        @app.get("/b/{name}/telemetry")
        def b_telemetry(name: str):
            h = node(name)
            return {"telemetry": h.adapter.tel, "history": list(h.history)[-8:], "limits": h.adapter.limits()}

        @app.get("/b/{name}/v2/status")
        def b_status(name: str):
            h = node(name)
            return {**h.status(), "offers": h.offers, "contracts": h.contracts, "payments": h.payments}

        @app.get("/b/{name}/agent/state")
        def b_agent(name: str, since: float = 0.0):
            h = node(name)
            if h.brain is None:
                return {"enabled": False, "home": name, "why": "the building's agent isn't attached here"}
            st = h.brain.state()
            if since:
                st["decisions"] = [d for d in st["decisions"] if d.get("wall", 0) > since]
                st["chat"] = [m for m in st.get("chat", []) if m.get("wall", 0) > since]
            return st

        @app.post("/b/{name}/demo/reset")
        def b_reset(name: str):
            node(name)
            f.reset(only=name)
            replay_mod.set_weather(1.0, 1.0, dT=0.0)      # a what-if here is the process's: this is the only reset a run's start sends the buildings
            return {"ok": True, "home": name}

        @app.get("/b/{name}/demo/clock")
        def b_clock_get(name: str):
            return node(name).replay.clock()

        @app.post("/b/{name}/demo/clock")
        def b_clock(name: str, body: dict = Body(default={})):
            node(name)
            return f.configure_clock(only=name, **body)

        @app.post("/b/{name}/demo/scenario")
        def b_scenario(name: str, body: dict = Body(default={})):
            node(name)
            ov = body.get("override")
            if isinstance(ov, dict):
                body = {**body, "override": {**ov, "home": name}}
            for k in ("closed", "zone_event"):
                if isinstance(body.get(k), dict):
                    body = {**body, k: {**body[k], "home": name}}
            return demo_scenario(body)

        @app.get("/b/{name}/logs/tail")
        def b_logs(name: str, after: int = -1):
            return logs_tail(after)

    return app


def main() -> None:
    ap = argparse.ArgumentParser(prog="flexgrid-fleet")
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--buildings", action="store_true", help="run the building nodes instead of the homes")
    a = ap.parse_args()
    import uvicorn
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    f = (BuildingFleet if a.buildings else Fleet)(a.limit)
    if not f.cfg.get("person_keys"):
        log.warning("no person's key is on file here, so nobody can end an alarm on these nodes: python -m flexgrid_core.doors says how")
    keeper = keep.for_fleet(f)                            # each node comes back where it stopped, before its tick starts
    f.start()
    try:
        uvicorn.run(build_app(f), host="127.0.0.1", port=int(f.cfg.get(f.port_key, f.default_port)), log_level="warning")
    finally:
        keeper.close()


if __name__ == "__main__":
    main()
