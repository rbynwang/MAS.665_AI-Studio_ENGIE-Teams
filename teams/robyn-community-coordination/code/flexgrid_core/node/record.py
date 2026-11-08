"""The decision record: what a node decided and why, so any decision can be replayed without the model.

  Log             append only, one per node, each row chained to the last by its hash
  Steps           every workflow step is a recorded state, and repeating a message changes nothing, even after a restart
  RecordingBrain  saves each question and answer. ReplayBrain gives them back with no model, and notices if the question changed
  RecordingBackend, ReplayBackend   the same for steps 5 and 6, while those still go through brain.HomeBrain
  replay          every recorded device call again, with no model and no device, compared with what happened

A device call's row holds its inputs, the guard's state before it, the policy verdict, the guard's answer and the command, and a
model row holds the prompt version, the model and its digest, the tools shown and the answer: MULTI_AGENT_DESIGN, section 6.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
from pathlib import Path
from typing import Any, Callable, Iterable

import httpx

COMPARE = ("clean", "policy", "answer", "command")


def clean(x: Any) -> Any:
    """Plain JSON, the same every time: numpy numbers become numbers, sets are sorted, NaN and text that isn't real text can't break the file."""
    if isinstance(x, str):
        return x if x.isascii() else x.encode("utf-8", "replace").decode("utf-8")     # a lone surrogate from a JSON escape can't be written
    if isinstance(x, dict):
        return {clean(str(k)): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted(clean(v) for v in x)
    if hasattr(x, "item") and not isinstance(x, (str, bytes)):
        x = x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return str(x)


def dumps(x: Any) -> str:
    return json.dumps(clean(x), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha(*parts: Any) -> str:
    return hashlib.sha256(dumps(list(parts)).encode()).hexdigest()


class Log:
    """In memory, and in a file if it has a path. A file that exists is read back, so a restarted node keeps its record.

    A power cut can tear the last line. That line goes to a .torn file beside the log and the node comes back: a row that was never whole
    wrote nothing that once() relied on. A bad line in the middle isn't an accident, so it stays as an unreadable row and verify() points at it.
    """

    def __init__(self, node: str, path: str | Path | None = None):
        self.node = node
        self.path = Path(path) if path else None
        self.rows: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        if self.path and self.path.exists():
            self.rows = self._read(self.path)

    def _read(self, path: Path) -> list[dict[str, Any]]:
        raw = path.read_bytes()
        *lines, tail = raw.split(b"\n")
        rows: list[dict[str, Any]] = []
        for x in lines:
            if x.strip():
                try:
                    rows.append(json.loads(x))
                except ValueError:
                    rows.append({"n": len(rows) + 1, "kind": "unreadable", "node": self.node, "text": x[:200].decode("utf-8", "replace"), "hash": ""})
        if tail.strip():
            with path.with_name(path.name + ".torn").open("ab") as f:
                f.write(tail + b"\n")
            with path.open("r+b") as f:
                f.truncate(len(raw) - len(tail))
        return rows

    def add(self, kind: str, **fields: Any) -> dict[str, Any]:
        with self._lock:
            r = {"n": len(self.rows) + 1, "kind": kind, "node": self.node, **clean(fields), "prev": self.rows[-1]["hash"] if self.rows else ""}
            r["hash"] = sha(r)
            self.rows.append(r)
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(dumps(r) + "\n")
                    f.flush()
                    os.fsync(f.fileno())               # once() counts on this row being there after a power cut
            return r

    def all(self, kind: str | None = None) -> list[dict[str, Any]]:
        return [r for r in self.rows if kind is None or r["kind"] == kind]

    def decision(self, rec: dict[str, Any]) -> None:
        """agent.py's own summary of a step: pass this as its `record`."""
        self.add("decision", **rec)

    def verify(self) -> dict[str, Any]:
        """A changed row, or one missing from the middle, breaks the chain from there on. This finds accidents, not forgery: there is no key, and
        a cut at the very end leaves a good chain, because nothing after it says there should be more."""
        prev = ""
        for i, r in enumerate(self.rows):
            body = {k: v for k, v in r.items() if k != "hash"}
            if r.get("n") != i + 1 or r.get("prev") != prev or sha(body) != r.get("hash"):
                return {"ok": False, "row": i + 1}
            prev = r["hash"]
        return {"ok": True, "rows": len(self.rows)}


class Steps:
    """The workflow's states. Each once() writes before it runs, so a crash in the middle is never run again by mistake."""

    UNKNOWN = {"status": "unknown", "why": "This started before a restart and its end wasn't recorded, so it isn't run again.",
               "view": "This started before a restart and its end wasn't recorded, so it isn't run again."}

    def __init__(self, log: Log):
        self.log = log
        self.done: dict[str, Any] = {}
        for r in log.rows:
            if r["kind"] == "started":
                self.done.setdefault(r["key"], dict(self.UNKNOWN))
            elif r["kind"] == "finished":
                self.done[r["key"]] = r["result"]
            elif r["kind"] == "not_final":
                self.done.pop(r["key"], None)
            elif r["kind"] == "new_run":
                self.done.clear()

    def new_run(self) -> None:
        """A run over the same days again, on the same log. A restart keeps what was done. A new run must not: its calls have to happen."""
        self.log.add("new_run")
        self.done.clear()

    def once(self, key: Any, fn: Callable[[], Any], final: Callable[[Any], bool] | None = None) -> Any:
        """fn runs the first time only, and later the same answer comes back. final(answer) False means nothing happened, so it may be tried again."""
        k = key if isinstance(key, str) else sha(key)
        if k in self.done:
            self.log.add("repeat", key=k)
            return self.done[k]
        self.log.add("started", key=k)
        self.done[k] = dict(self.UNKNOWN)
        out = clean(fn())
        if final is not None and not final(out):
            del self.done[k]
            self.log.add("not_final", key=k)
            return out
        self.done[k] = out
        self.log.add("finished", key=k, result=out)
        return out

    def enter(self, step: str, ts: str, **info: Any) -> None:
        self.log.add("step", step=step, ts=ts, state="entered", **info)

    def leave(self, step: str, ts: str, outcome: str = "done") -> None:
        self.log.add("step", step=step, ts=ts, state=outcome)

    def state(self) -> dict[str, dict[str, Any]]:
        """Where each step stands now."""
        out: dict[str, dict[str, Any]] = {}
        for r in self.log.all("step"):
            out[r["step"]] = {"state": r["state"], "ts": r["ts"]}
        return out


def model_version(name: str, url: str = "http://127.0.0.1:11434") -> dict[str, str]:
    """The model's name and digest, so "the same model" means the same weights."""
    try:
        tags = httpx.get(url + "/api/tags", timeout=5).json().get("models", [])
        return {"name": name, "digest": next((m.get("digest", "") for m in tags if m.get("name") == name), "")}
    except (httpx.HTTPError, ValueError):
        return {"name": name, "digest": ""}


class RecordingBrain:
    """Has agent.Brain's ask(). Everything else, such as .calls or .model, comes from the brain inside."""

    def __init__(self, inner: Any, log: Log, prompt: str = "", model: dict[str, str] | None = None, options: dict[str, Any] | None = None):
        self.inner, self.log, self.prompt, self.options = inner, log, prompt, options
        self.version = model or {"name": getattr(inner, "model", ""), "digest": ""}

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def ask(self, system: str, tools: list[dict[str, Any]], text: str, **kw: Any) -> dict[str, Any]:
        a = self.inner.ask(system, tools, text, **kw)
        self.log.add("model", asked=sha(system, tools, text), prompt=self.prompt, model=self.version, options=self.options, system=system, text=text, tools=tools, answer=a)
        return a


class ReplayDrift(Exception):
    """The code asked something the record doesn't hold: it no longer does what the live run did."""


class ReplayBrain:
    def __init__(self, rows: Iterable[dict[str, Any]], model: str = "replay"):
        self.rows = [r for r in rows if r["kind"] == "model"]
        self.i = 0
        self.model = model
        self.calls = 0
        self.ms = 0.0
        self.tokens_in = 0
        self.tokens_out = 0
        self.errors = 0

    def ask(self, system: str, tools: list[dict[str, Any]], text: str, **kw: Any) -> dict[str, Any]:
        if self.i >= len(self.rows):
            raise ReplayDrift(f"asked once more than the live run, after {self.i} answers")
        r = self.rows[self.i]
        if r["asked"] != sha(system, tools, text):
            raise ReplayDrift(f"question {self.i + 1} changed. Recorded: {r['text'][:120]!r} Now: {text[:120]!r}")
        self.i += 1
        self.calls += 1
        return dict(r["answer"])


class RecordingBackend:
    """Has agent.DecideBackend's decide(), for the offer and the commitment."""

    def __init__(self, inner: Any, log: Log):
        self.inner, self.log = inner, log

    def decide(self, role: str, system: str, facts: dict[str, Any], schema: dict[str, Any], *, budget_s: float = 40.0, session: str | None = None):
        ans, meta = self.inner.decide(role, system, facts, schema, budget_s=budget_s, session=session)
        self.log.add("decide", asked=sha(system, facts, schema), role=role, session=session, facts=facts, answer=ans, meta=meta)
        return ans, meta


class ReplayBackend:
    def __init__(self, rows: Iterable[dict[str, Any]]):
        self.rows = [r for r in rows if r["kind"] == "decide"]
        self.i = 0

    def decide(self, role: str, system: str, facts: dict[str, Any], schema: dict[str, Any], *, budget_s: float = 40.0, session: str | None = None):
        if self.i >= len(self.rows):
            raise ReplayDrift(f"decided once more than the live run, after {self.i} answers")
        r = self.rows[self.i]
        if r["asked"] != sha(system, facts, schema):
            raise ReplayDrift(f"decision {self.i + 1} ({role}) was asked on different facts")
        self.i += 1
        return r["answer"], r["meta"]


def replay(log: Log, rerun: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
    """rerun(row) runs one device call again from its row: context.rerun does this. The result says what didn't match."""
    rows = log.all("act")
    diffs = []
    for r in rows:
        again = clean(rerun(r))
        d = {k: {"live": r.get(k), "replay": again.get(k)} for k in COMPARE if r.get(k) != again.get(k)}
        if d:
            diffs.append({"n": r["n"], "tool": r.get("tool"), "diff": d})
    return {"checked": len(rows), "same": len(rows) - len(diffs), "diffs": diffs[:20], "chain": log.verify()}
