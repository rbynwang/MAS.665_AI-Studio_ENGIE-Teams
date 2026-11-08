"""The brain's queue: one model, many nodes, one request at a time (MULTI_AGENT_DESIGN, section 10).

  exception   something changed on the day: goes first, always
  sale        today's market steps: the offer, the commitment, the plan before a sale
  report      after an event
  plan        tomorrow's reasoning: overnight only
  strategy    the week's review: overnight only, after the plans

Real-time control never waits for this. A job that can't start in time, or that a node's budget can't cover, comes back at once as
an error, and the node falls back to code, as it does when the model is down. A job past its time comes back then, even while the
model is busy, and a caller that hangs up takes its job with it, so the model answers nobody who has left.

BrainQueue is the queue itself. QueuedBrain has agent.Brain's ask(), so a node can swap one for the other. serve() puts the queue in
front of Ollama on the Mac mini, as an Ollama-shaped /api/chat, because each node is its own process on a Jetson: a request may carry
"flexgrid": {"kind", "node"}. A plain request counts as a sale step with no node, so it takes its turn but no budget covers it.

  python -m flexgrid_core.node.queue --port 11435        # on the Mac mini. Nothing starts this yet
"""
from __future__ import annotations

import argparse
import json
import select
import socket
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

import httpx

URL = "http://127.0.0.1:11434"
NUM_CTX = 32768
KINDS = ("exception", "sale", "report", "plan", "strategy")
NIGHT_ONLY = ("plan", "strategy")
NIGHT = ("22:00", "06:00")
WAIT_S = {"exception": 60.0, "sale": 300.0, "report": 3600.0, "plan": 8 * 3600.0, "strategy": 8 * 3600.0}
# Answers one node may use, per kind, per day. Our choice, sized from the design's table: about 2 for today's market, 5 to 10 for
# tomorrow's plan, and room for one six-call step and its retries before a sale.
BUDGET = {"exception": 12, "sale": 16, "report": 2, "plan": 10, "strategy": 4}


def _hm(x: str) -> int:
    return int(x[:2]) * 60 + int(x[3:5])


def is_night(t: datetime, night: tuple[str, str] = NIGHT) -> bool:
    m, a, b = t.hour * 60 + t.minute, _hm(night[0]), _hm(night[1])
    return a <= m or m < b if a > b else a <= m < b


@dataclass
class Job:
    kind: str
    node: str
    body: dict[str, Any]
    seq: int
    made: datetime
    due: datetime
    done: threading.Event = field(default_factory=threading.Event)
    out: dict[str, Any] | None = None
    error: str | None = None
    model_s: float = 0.0
    waited_s: float = 0.0
    day: str = ""                                      # the node's own day, where its clock isn't the wall's

    def wait(self, timeout: float | None = None) -> "Job":
        self.done.wait(timeout)
        return self


class BrainQueue:
    """Worker mode: start() runs one thread, for live use. Virtual-clock mode: no thread, and the caller runs run_one()."""

    def __init__(self, run: Callable[[dict[str, Any]], dict[str, Any]], clock: Callable[[], datetime] = datetime.now,
                 budget: dict[str, int] | None = None, night: tuple[str, str] = NIGHT):
        self.run, self.clock, self.night = run, clock, night
        self.budget = {**BUDGET, **(budget or {})}
        self.pending: list[Job] = []
        self.used: Counter = Counter()                 # (period, node, kind) -> answers used
        self.served: Counter = Counter()               # (period, node) -> answers given, so nodes take turns
        self.stats: dict[str, Any] = {"answers": Counter(), "late": Counter(), "refused": Counter(), "errors": Counter(), "gone": Counter(), "model_s": 0.0, "waits": defaultdict(list)}
        self._lock = threading.Lock()
        self._one = threading.Lock()                   # one request at a time, whoever asks
        self._wake = threading.Condition(self._lock)
        self._seq = 0
        self._stop = False
        self._worker: threading.Thread | None = None

    # ------------------------------------------------------------------ putting a job in
    def _period(self, kind: str, t: datetime, day: str = "") -> str:
        """A node on a fast clock counts its own days, or it would run out of answers on its third day. A night's work belongs to the evening it started."""
        return day or str((t - timedelta(hours=12)).date() if kind in NIGHT_ONLY else t.date())

    def _can_start(self, j: Job, t: datetime) -> bool:
        """A night job's time may run out before the next night begins: then waiting only holds its caller."""
        if j.kind not in NIGHT_ONLY or is_night(t, self.night):
            return True
        m = _hm(self.night[0])
        begins = t.replace(hour=m // 60, minute=m % 60, second=0, microsecond=0)
        return (begins if begins > t else begins + timedelta(days=1)) <= j.due

    def submit(self, kind: str, node: str, body: dict[str, Any], wait_s: float | None = None, day: str = "") -> Job:
        kind = kind if kind in KINDS else "sale"
        with self._wake:
            t = self.clock()
            self._seq += 1
            j = Job(kind, str(node), body, self._seq, t, t + timedelta(seconds=WAIT_S[kind] if wait_s is None else wait_s), day=day)
            if self._stop:
                self._finish(j, error="stopped: the queue is shutting down")
                return j
            if not self._can_start(j, t):
                self.stats["late"][kind] += 1
                self._finish(j, error=f"late: a {kind} step can't start before its time is up")
                return j
            queued = sum(1 for x in self.pending if x.node == j.node and x.kind == kind)
            if j.node and self.used[(self._period(kind, t, day), j.node, kind)] + queued >= self.budget[kind]:
                self.stats["refused"][kind] += 1
                self._finish(j, error=f"budget: {node} has used its {kind} answers for now")
                return j
            self.pending.append(j)
            self._wake.notify()
            return j

    @staticmethod
    def _finish(j: Job, out: dict[str, Any] | None = None, error: str | None = None) -> None:
        j.out, j.error = out, error
        j.done.set()

    # ------------------------------------------------------------------ taking the next one
    def _expire(self, t: datetime) -> None:
        """Whatever is past its time comes back as late."""
        for j in [x for x in self.pending if x.due < t or not self._can_start(x, t)]:
            self.pending.remove(j)
            self.stats["late"][j.kind] += 1
            self._finish(j, error=f"late: no answer within the time a {j.kind} step has")

    def _next(self, t: datetime) -> Job | None:
        self._expire(t)
        ready = [j for j in self.pending if j.kind not in NIGHT_ONLY or is_night(t, self.night)]
        if not ready:
            return None

        def rank(j: Job) -> tuple[int, int, int]:
            turn = 0 if j.kind == "exception" else self.served[(self._period(j.kind, t, j.day), j.node)]
            return KINDS.index(j.kind), turn, j.seq
        j = min(ready, key=rank)
        self.pending.remove(j)
        return j

    def run_one(self) -> Job | None:
        """Runs the next job, if one may run now."""
        with self._one:
            with self._lock:
                t = self.clock()
                j = self._next(t)
                if j is None:
                    return None
                p = self._period(j.kind, t, j.day)
                self.used[(p, j.node, j.kind)] += 1
                self.served[(p, j.node)] += 1
                j.waited_s = (t - j.made).total_seconds()
            t0 = time.perf_counter()
            try:
                out, err = self.run(j.body), None
            except Exception as exc:                   # the model is down: this job fails, and the queue keeps going
                out, err = None, f"{type(exc).__name__}: {str(exc)[:200]}"
            j.model_s = time.perf_counter() - t0
            with self._lock:
                self.stats["model_s"] += j.model_s
                self.stats["waits"][j.kind].append(j.waited_s)
                self.stats["errors" if err else "answers"][j.kind] += 1
            self._finish(j, out, err)
            return j

    def withdraw(self, j: Job) -> None:
        """The caller has stopped waiting."""
        with self._lock:
            if j in self.pending:
                self.pending.remove(j)
                self.stats["gone"][j.kind] += 1
                self._finish(j, error="withdrawn: it couldn't run now")

    def wait(self, j: Job, gone: Callable[[], bool] = lambda: False, tick_s: float = 0.25) -> Job:
        """The caller's side, with a worker running. It comes back when the job does, when its time is up though the model is still busy with another,
        or when the caller has left. Past the time limit the job is taken out, so it can't run for nobody later."""
        end = time.monotonic() + WAIT_S[j.kind] + 300.0
        while not j.done.wait(tick_s) and time.monotonic() < end:
            with self._lock:
                self._expire(self.clock())
            if gone():
                break
        self.withdraw(j)
        return j

    def drain(self, until: Job | None = None) -> None:
        """Virtual-clock mode: run what can run now, or until this job is done."""
        while (until is None or not until.done.is_set()) and self.run_one():
            pass
        with self._lock:
            self._expire(self.clock())

    # ------------------------------------------------------------------ the worker
    def start(self) -> "BrainQueue":
        self._worker = threading.Thread(target=self._loop, name="brain-queue", daemon=True)
        self._worker.start()
        return self

    def stop(self) -> None:
        """Whatever is waiting comes back at once, so every node falls back to code."""
        with self._wake:
            self._stop = True
            for j in self.pending:
                self._finish(j, error="stopped: the queue is shutting down")
            self.pending.clear()
            self._wake.notify_all()
        if self._worker:
            self._worker.join(timeout=5)
            self._worker = None

    def _loop(self) -> None:
        while True:
            with self._wake:
                if self._stop:
                    return
            if self.run_one() is None:
                with self._wake:
                    if not self._stop:
                        self._wake.wait(0.5)           # short, so a night job starts when the night does

    @property
    def threaded(self) -> bool:
        return self._worker is not None

    def report(self) -> dict[str, Any]:
        s = self.stats
        n = sum(s["answers"].values())
        return {"answers": dict(s["answers"]), "late": dict(s["late"]), "refused": dict(s["refused"]), "errors": dict(s["errors"]), "gone": dict(s["gone"]), "waiting": len(self.pending),
                "model_s": round(s["model_s"], 1), "answers_per_hour": round(n / s["model_s"] * 3600) if s["model_s"] > 0 else None,
                "worst_wait_s": {k: round(max(v), 1) for k, v in s["waits"].items() if v}}


def overnight(answers_per_hour: float, hours: float = 8.0, keep_free: float = 0.2, calls_per_node: tuple[int, int] = (5, 10)) -> dict[str, Any]:
    """How many nodes can think overnight. keep_free leaves room for exceptions and for answers longer than the sample's."""
    total = answers_per_hour * hours * (1.0 - keep_free)
    return {"answers_per_hour": round(answers_per_hour), "hours": hours, "keep_free": keep_free, "answers_a_night": int(total),
            "nodes_at_most_calls": int(total // calls_per_node[1]), "nodes_at_fewest_calls": int(total // calls_per_node[0]), "calls_per_node": list(calls_per_node)}


# --------------------------------------------------------------------------------------------------- the model
def chat_body(system: str, tools: list[dict[str, Any]], text: str, model: str, options: dict[str, Any]) -> dict[str, Any]:
    """The same request agent.Brain sends."""
    return {"model": model, "stream": False, "think": False, "keep_alive": "30m", "tools": tools, "options": options,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": text}]}


class Ollama:
    """Every request leaves with the same runner settings, llm.py's, whoever sent it. A caller with other settings would make Ollama
    reload the model each time the callers alternate, and lose the prompt cache: seen live on 31 Oct, and again with the node agents."""

    def __init__(self, url: str = URL, timeout: float = 180.0):     # alone on the model an answer takes seconds, and past this it has gone wrong
        self.url = url.rstrip("/")
        self.http = httpx.Client(timeout=timeout)

    def __call__(self, body: dict[str, Any]) -> dict[str, Any]:
        same = {"keep_alive": "24h", "truncate": False, "shift": False, "options": {**(body.get("options") or {}), "num_ctx": NUM_CTX}}
        r = self.http.post(self.url + "/api/chat", json={**body, **same})
        r.raise_for_status()
        return r.json()


class QueuedBrain:
    """agent.Brain's ask(), for one node, through the queue. Set .agent and the kind follows the step the agent's workflow is in."""

    def __init__(self, queue: BrainQueue, node: str, kind: str = "sale", model: str | None = None, options: dict[str, Any] | None = None):
        if model is None or options is None:
            from .agent import MODEL, OPTIONS          # here, not above, so agent.py can import this file
            from .arm import model_options
            model, options = model or MODEL, options or model_options(OPTIONS)
        self.queue, self.node, self.kind, self.model, self.options = queue, node, kind, model, options
        self.agent: Any = None
        self.calls = 0
        self.ms = 0.0
        self.tokens_in = 0
        self.tokens_out = 0
        self.errors = 0

    def kind_now(self) -> str:
        steps = getattr(self.agent, "steps", None)
        step = steps[-1][0] if steps else ""
        return "exception" if step.startswith("8") else "report" if step.startswith("9") else self.kind

    def ask(self, system: str, tools: list[dict[str, Any]], text: str, kind: str | None = None) -> dict[str, Any]:
        j = self.queue.submit(kind or self.kind_now(), self.node, chat_body(system, tools, text, self.model, self.options))
        if not self.queue.threaded:
            self.queue.drain(j)
            self.queue.withdraw(j)                     # on a virtual clock nothing can wait: it wasn't its time, so it has no answer
        self.queue.wait(j)
        if j.out is None:
            self.errors += 1
            return {"calls": [], "text": "", "error": j.error or "no answer", "ms": 0}
        m = j.out.get("message") or {}
        calls = [{"name": c["function"]["name"], "args": c["function"].get("arguments") or {}} for c in (m.get("tool_calls") or [])]
        ms = j.model_s * 1000
        self.calls += 1
        self.ms += ms
        self.tokens_in += int(j.out.get("prompt_eval_count") or 0)
        self.tokens_out += int(j.out.get("eval_count") or 0)
        return {"calls": calls, "text": (m.get("content") or "")[:300], "ms": round(ms), "error": None}


# --------------------------------------------------------------------------------------------------- the service on the Mac mini
def serve(queue: BrainQueue, port: int, upstream: str = URL, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """An Ollama-shaped front door. Returns the server: the caller runs serve_forever(), on a thread or not."""
    pass_through = httpx.Client(timeout=10)

    class Door(BaseHTTPRequestHandler):
        def log_message(self, *a: Any) -> None:
            return

        def _send(self, code: int, obj: Any) -> None:
            data = json.dumps(obj).encode()
            try:
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except OSError:                            # the caller left while the model answered
                self.close_connection = True

        def _gone(self) -> bool:
            """The caller hung up: the socket reads as finished. A node's own time limit is shorter than a sale job's, so this happens under load."""
            try:
                readable, _, _ = select.select([self.connection], [], [], 0)
                return bool(readable) and not self.connection.recv(1, socket.MSG_PEEK)
            except OSError:
                return True

        def do_GET(self) -> None:
            if self.path == "/queue/stats":
                return self._send(200, queue.report())
            if self.path in ("/api/tags", "/api/version", "/api/ps"):
                try:
                    return self._send(200, pass_through.get(upstream.rstrip("/") + self.path).json())
                except (httpx.HTTPError, ValueError) as exc:
                    return self._send(502, {"error": str(exc)[:200]})
            return self._send(404, {"error": "no such path"})

        def do_POST(self) -> None:
            if self.path != "/api/chat":
                return self._send(404, {"error": "only /api/chat goes through the queue"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
                meta = body.pop("flexgrid", None)
                meta = meta if isinstance(meta, dict) else {}
                body["stream"] = False                 # one answer, whole: the queue hands back one reply
            except (ValueError, AttributeError, TypeError):
                return self._send(400, {"error": "not JSON"})
            j = queue.submit(str(meta.get("kind") or "sale"), str(meta.get("node") or ""), body, day=str(meta.get("day") or ""))
            queue.wait(j, self._gone)
            if j.out is not None:
                return self._send(200, j.out)
            late = (j.error or "").startswith(("late", "budget"))
            return self._send(429 if late else 502, {"error": j.error or "no answer"})

    return ThreadingHTTPServer((host, port), Door)


def main() -> None:
    ap = argparse.ArgumentParser(prog="flexgrid-brain-queue")
    ap.add_argument("--port", type=int, default=11435)
    ap.add_argument("--upstream", default=URL)
    a = ap.parse_args()
    q = BrainQueue(Ollama(a.upstream)).start()
    print(f"brain queue on 127.0.0.1:{a.port}, in front of {a.upstream}")
    serve(q, a.port, a.upstream).serve_forever()


if __name__ == "__main__":
    main()
