"""NANDA Town client (nandatown 0.2.0 coordinator HTTP contract).

One process per participant name. NANDA Town 0.2 hands a second join under the same name the session it already
has: two processes would then share one inbox and split its messages without any error (other Town versions end the
earlier session instead, and the two would take it from each other: TownSession pauses rather than fight, below).
So each name runs in exactly one process: flexgrid-dc stops its data center sessions before the campus joins, and asks
the campus to stand down before it takes them back.

How an inbox is watched (InboxPoller below): a short notify check every second or so, never a held long-poll.
NANDA Town 0.2 answers a long-poll (notify?wait=N) by re-checking its notifications table every 50 ms inside a write
transaction, and that table has no index: on the team's Town (29 Oct 2026, a 256 MB database) each check is a scan of
every wake-up hint ever written, and four held long-polls (the coordinator, the utility and the two Jetson homes) cost
it about 80 scans a second, against about 20 short requests a second from all the other participants together. With
the Town at 77 % of a CPU core, a plain request on the Town host itself took 3 s and the fleet homes got their
messages a minute late.

How far behind an inbox is (LagMeter): the delay of the messages it claims (claimed minus the sender's sentWall).
Town hands an inbox its messages oldest first, so an urgent message waits behind every meter reading that came before
it (seen live 1 Nov 2026: during an event's readings a grid call reached the coordinator 174 s after the buyer sent
it, and the demo clock had run ten dataset hours on). The coordinator's clock waits while the inboxes are behind.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable

import httpx


class TownIdentityConflict(RuntimeError):
    """Another process keeps taking this participant's Town session (each join ends the other's session)."""


class TownSession:
    # a participant that has to join again this many times within CONFLICT_WINDOW s is sharing its name with another
    # process (a join ends the other session, which joins again, ...): stop for CONFLICT_PAUSE s instead of fighting
    CONFLICT_REJOINS, CONFLICT_WINDOW, CONFLICT_PAUSE = 3, 120.0, 60.0

    def __init__(self, base_url: str, run_id: str, name: str, token: str, http: httpx.Client | None = None):
        self.base_url, self.run_id, self.name, self.token = base_url.rstrip("/"), run_id, name, token
        # http: a client to share. Each client holds its own TLS context, about 0.75 MB: 200 sessions on one client saved 150 MB
        self.http = http or httpx.Client(base_url=self.base_url, timeout=40.0)
        self.session: str | None = None
        self.lock = threading.Lock()
        self.retried = 0                   # calls this session had to try again (a dropped connection, a 502/503/504)
        self.rejoins: deque = deque(maxlen=10)       # when this session had to join again after a 401
        self.conflict_until = 0.0
        self.conflicts = 0                 # times another process was found using this name
        self.times: deque = deque(maxlen=3000)      # (wall, op, seconds) of each Town call: timing()

    def timing(self, window: float = 120.0) -> dict[str, dict[str, float]]:
        """How long this session's Town calls took in the last `window` s, by kind: {op: {n, p50_ms, p95_ms}}. A
        slow Town (or its tunnel) shows here before it shows anywhere else."""
        now, by = time.time(), {}
        for t, op, d in list(self.times):
            if now - t <= window:
                by.setdefault(op, []).append(d)
        out = {}
        for op, ds in by.items():
            ds.sort()
            out[op] = {"n": len(ds), "p50_ms": round(1000 * ds[len(ds) // 2]), "p95_ms": round(1000 * ds[int(0.95 * (len(ds) - 1))])}
        return out

    @staticmethod
    def _op(path: str) -> str:
        for key, op in (("/inbox/notify", "notify"), ("/inbox/claim", "claim"), ("/inbox/ack", "ack"),
                        ("/messages", "send"), ("/participants", "participants")):
            if path.endswith(key):
                return op
        return "other"

    def _h(self) -> dict[str, str]:
        return {"X-Town-Session": self.session or ""}

    def join(self) -> dict[str, Any]:
        with self.lock:
            return self._join()

    def join_until_up(self, then: Any = None, log: Any = None, every: float = 15.0) -> threading.Thread:
        """Join in the background, again every `every` s while NANDA Town cannot be reached (its tunnel down: a 502 or
        530 from Cloudflare, a refused connection), then call `then()`: the agent's own API stays up meanwhile, so the
        Control Room can say what is wrong instead of a process that exited (seen 1 Nov 2026: the coordinator died on
        a 502 at its start while jetson-b was off the network)."""
        def run() -> None:
            tries = 0
            while True:
                try:
                    self.join()
                    break
                except Exception as exc:               # noqa: BLE001  any failure: wait and try again
                    tries += 1
                    if log and (tries == 1 or tries % 20 == 0):
                        log(f"NANDA Town not reachable ({str(exc)[:120]}); trying again every {every:g} s")
                    time.sleep(every)
            if log and tries:
                log(f"NANDA Town joined after {tries} failed tries")
            if then:
                then()
        t = threading.Thread(target=run, daemon=True, name=f"town-join-{self.name}")
        t.start()
        return t

    def _join(self) -> dict[str, Any]:
        r = self.http.post(f"/runs/{self.run_id}/join", json={"name": self.name, "token": self.token})
        r.raise_for_status()
        data = r.json()
        self.session = data["session"]
        return data

    def _rejoin(self, stale: str | None) -> None:
        """Join again only if nobody has done it since `stale` was read: a join ends the previous session, so threads
        that all saw the same 401 must not each join (they would end each other's new session). A name that keeps
        losing its session is held by another process too: pause instead of taking it back again and again."""
        with self.lock:
            if self.session == stale or not self.session:
                now = time.time()
                if stale is not None:
                    self.rejoins.append(now)
                    recent = [t for t in self.rejoins if now - t <= self.CONFLICT_WINDOW]
                    if len(recent) >= self.CONFLICT_REJOINS:
                        self.rejoins.clear()
                        self.conflicts += 1
                        self.conflict_until = now + self.CONFLICT_PAUSE
                        raise TownIdentityConflict(
                            f"{self.name}: another process keeps taking this Town session; pausing "
                            f"{self.CONFLICT_PAUSE:.0f} s (run only one process per participant name)")
                self._join()

    def _request(self, method: str, path: str, session: str | None, **kw) -> httpx.Response:
        """One HTTP call, tried once more if the connection drops on the way (a kept-alive connection that the
        server or the Cloudflare tunnel had already closed). Every Town call is safe to repeat: a send carries its
        message id (Town takes it once), a claim whose answer was lost comes back after its lease, an ack repeated
        gets a stale-fence answer."""
        try:
            return self.http.request(method, path, headers={"X-Town-Session": session or ""}, **kw)
        except httpx.TransportError:
            self.retried += 1
            time.sleep(0.3)
            return self.http.request(method, path, headers={"X-Town-Session": session or ""}, **kw)

    def _call(self, method: str, path: str, **kw) -> httpx.Response:
        if time.time() < self.conflict_until:
            raise TownIdentityConflict(f"{self.name}: paused, another process holds this Town session")
        if not self.session:
            self._rejoin(None)
        used = self.session
        t0 = time.time()
        r = self._request(method, path, used, **kw)
        if r.status_code == 401:                           # session replaced or expired: re-join once
            self._rejoin(used)
            r = self._request(method, path, self.session, **kw)
        if r.status_code in (502, 503, 504):               # Town's "lost ack" fault (503), or the tunnel in front of
            self.retried += 1                              # it (502/504, seen live through Cloudflare): retry once
            time.sleep(0.5)
            r = self._request(method, path, self.session, **kw)
        self.times.append((t0, self._op(path), time.time() - t0))
        return r

    def send(self, to: str, kind: str, body: dict[str, Any], message_id: str) -> dict[str, Any]:
        r = self._call("POST", f"/runs/{self.run_id}/messages",
                       json={"message_id": message_id, "to": to, "kind": kind, "body": body})
        if r.status_code == 409:
            return {"message_id": message_id, "duplicate": True}
        r.raise_for_status()
        return r.json()

    def notify(self, wait: float = 0.0) -> bool:
        """Pop one wake-up hint. Keep `wait` at 0 (see the module docstring: a held long-poll costs Town ~20 scans a
        second); InboxPoller does the waiting on this side."""
        r = self._call("GET", f"/runs/{self.run_id}/inbox/notify", params={"wait": wait})
        r.raise_for_status()
        return bool(r.json().get("hint"))

    def claim(self) -> dict[str, Any] | None:
        r = self._call("POST", f"/runs/{self.run_id}/inbox/claim")
        if r.status_code == 204:
            return None
        r.raise_for_status()
        return r.json()

    def ack(self, message_id: str, fence: str, status: str, note: dict[str, Any] | None = None) -> dict[str, Any]:
        r = self._call("POST", f"/runs/{self.run_id}/inbox/ack",
                       json={"message_id": message_id, "fence": fence, "status": status, "note": note or {}})
        if r.status_code == 409:
            return {"stale_fence": True}
        r.raise_for_status()
        return r.json()

    def participants(self) -> list[dict[str, Any]]:
        r = self._call("GET", f"/runs/{self.run_id}/participants")
        r.raise_for_status()
        return r.json()


def _once_more(f: Callable, *a, **k):
    """A Town call tried once more at once when the connection drops (a kept-alive connection the server had already
    closed). Safe for notify, claim and ack: a claim whose answer was lost comes back after its lease, an ack repeated
    gets a stale-fence answer."""
    try:
        return f(*a, **k)
    except httpx.TransportError:
        return f(*a, **k)


class LagMeter:
    """How far behind an inbox is: the median delay (claimed minus the sender's sentWall) of the messages it claimed in
    the last `window` s, counting first deliveries only (a message handed out again waited out a lease, which says
    nothing about a backlog) and only once at least `min_n` came (one late message is not a backlog). 0 when nothing
    came lately: an empty inbox is not behind."""

    def __init__(self, window: float = 20.0, min_n: int = 3):
        self.window, self.min_n = window, min_n
        self._seen: deque = deque(maxlen=600)
        self._lock = threading.Lock()

    def seen(self, msg: dict[str, Any], now: float | None = None) -> float | None:
        try:
            if int(msg.get("attempt") or 1) > 1 or msg.get("duplicate"):
                return None
            sent = float(((msg.get("body") or {}).get("context") or {}).get("sentWall") or 0.0)
            if sent <= 0:
                return None
            now = time.time() if now is None else now
            lag = max(0.0, now - sent)
            with self._lock:
                self._seen.append((now, lag))
            return lag
        except (TypeError, ValueError):
            return None

    def value(self, now: float | None = None) -> float:
        now = time.time() if now is None else now
        with self._lock:
            lags = sorted(lag for t, lag in self._seen if now - t <= self.window)
        return lags[len(lags) // 2] if len(lags) >= self.min_n else 0.0


class InboxPoller:
    """Watches one Town inbox with short checks and claims until it is empty; `handle(msg)` processes a claimed message
    and acknowledges it (a message it raises on is not acknowledged: Town hands it out again after its lease).

    Cadence: a check every `every` s while mail flows, every `idle_every` s once nothing has come for `idle_after` s
    (the first message of a quiet spell waits at most `idle_every`). Town keeps one wake-up hint per message (a hint
    is only a wake-up call, never the message itself):
      * a hint: claim until the inbox is empty. A pass that handled n messages leaves n - 1 hints for messages already
        handled: they are popped (the light call) without another claim pass. If the hints run out early, some were
        lost: one more pass, in case a message came meanwhile
      * a hint but nothing to claim, or a claim that failed on the way: the message may be held by a claim whose answer
        was lost, which Town hands out again after its lease: one more pass when the lease is over
      * a check that failed on the way may have used up a hint: one pass at once
      * while hints of messages already handled are popped, still a pass every `every` s: after a pass of 400
        messages, popping their hints one call at a time takes a minute, and new mail must not wait for it (seen live
        1 Nov 2026: the homes' answers to a grid call reached the coordinator 60 s after they were sent)
      * a pass every `sweep` s anyway
    `claimers` > 1: a pass that finds a message brings in that many claimers in parallel (bursts: 100 offers, 400 meter
    readings). An empty inbox costs Town one claim, not `claimers`."""

    def __init__(self, town: TownSession, handle: Callable[[dict[str, Any]], Any], *, every: float = 1.0,
                 idle_every: float = 4.0, idle_after: float = 60.0, sweep: float = 30.0, lease: float = 30.0,
                 claimers: int = 1, on_error: Callable[[str, Exception], None] | None = None,
                 on_empty: Callable[[], None] | None = None, alive: Callable[[], bool] | None = None,
                 timed: Callable[[str, float], None] | None = None):
        self.town, self.handle = town, handle
        self.every, self.idle_every, self.idle_after, self.sweep = every, idle_every, idle_after, sweep
        self.lease = lease * 1.1 + 0.2                  # Town hands a lost claim out again after its lease
        self.claimers = max(1, int(claimers))
        self.on_error = on_error or (lambda where, exc: None)
        self.on_empty = on_empty or (lambda: None)
        self.alive = alive or (lambda: True)
        self.timed = timed or (lambda op, t0: None)
        self.pool = ThreadPoolExecutor(max_workers=self.claimers) if self.claimers > 1 else None
        self.handled = self.recovered = self.checks = self.claims = 0
        self.last_mail = 0.0
        self.fresh = 0                                  # messages of the last pass handed out for the first time
        self._flock = threading.Lock()

    def start(self, name: str = "town-inbox") -> threading.Thread:
        t = threading.Thread(target=self.run, name=name, daemon=True)
        t.start()
        return t

    def run(self) -> None:
        credit, recheck, sweep = 0, float("inf"), time.time()      # a first pass at once: whatever came while away
        last_pass = 0.0
        while self.alive():
            t0 = time.time()
            if t0 < self.town.conflict_until:               # another process holds this name: wait it out
                time.sleep(min(5.0, self.town.conflict_until - t0))
                continue
            tried = self.town.retried
            try:
                hint = self.town.notify(wait=0)
                self.timed("notify", t0)
                self.checks += 1
            except Exception as exc:
                self.on_error("notify", exc)
                hint = None                                  # unknown: a hint may have been used up
            if hint is False and self.town.retried != tried:
                hint = None                                  # the check had to be tried again: the first try may
                                                             # have popped a hint whose answer was lost
            if hint and credit > 0:
                credit -= 1                                  # a message the last pass already handled: pop the next
                if credit == 0:                              # hint at once. The count can be one too many (a hint
                    n, trouble = self.drain()                # lost on the way): the last one gets a pass, so a new
                    last_pass = time.time()                  # message whose hint was popped as stale is not left
                    if n:                                    # for the sweep
                        self.recovered += n
                    credit = max(0, self.fresh - 1) if n else 0
                    if trouble:
                        recheck = min(recheck, t0 + self.lease)
                elif t0 - last_pass >= self.every:           # popping many stale hints: mail that came meanwhile
                    n, trouble = self.drain()                # gets a pass every `every` s; its hints wait behind the
                    last_pass = time.time()                  # stale ones, so they join the count
                    credit += self.fresh
                    if trouble:
                        recheck = min(recheck, t0 + self.lease)
                continue
            if hint:
                n, trouble = self.drain()
                last_pass = time.time()
                credit = max(0, self.fresh - 1)              # hints left behind: one per message seen the first time
                if n == 0 or trouble:
                    recheck = min(recheck, t0 + self.lease)
            elif hint is None or credit > 0 or t0 >= recheck or t0 >= sweep:
                n, trouble = self.drain()
                last_pass = time.time()
                if n:
                    self.recovered += n
                credit, sweep = 0, t0 + self.sweep
                recheck = t0 + self.lease if trouble else float("inf")
            self._pause(t0)

    def _pause(self, t0: float) -> None:
        quiet = time.time() - self.last_mail > self.idle_after
        time.sleep(max(0.05, (self.idle_every if quiet else self.every) - (time.time() - t0)))

    def drain(self) -> tuple[int, bool]:
        """Claims until the inbox is empty: (messages handled, whether a claim failed on the way). self.fresh: how many
        of them were handed out for the first time (attempt 1). Only those left a wake-up hint behind; a message Town
        hands out again (its lease ran out: a claim whose answer was lost, a machine that slept) brought none, and
        counting it would make the next new message's hint look stale (seen as a data center's contract answered after
        its deadline in tests/test_dc_town_flaky.py)."""
        with self._flock:
            self.fresh = 0
        got = self._claim_one()
        if got <= 0:
            if got == 0:
                self.on_empty()
            return 0, got < 0
        if self.pool is None:
            n, bad = self._claim_all()
        else:
            futs = [self.pool.submit(self._claim_all) for _ in range(self.claimers - 1)]
            n, bad = self._claim_all()
            for f in futs:
                m, b = f.result()
                n, bad = n + m, bad or b
        if not bad:
            self.on_empty()
        return 1 + n, bad

    def _claim_all(self) -> tuple[int, bool]:
        n = 0
        while self.alive():
            got = self._claim_one()
            if got <= 0:
                return n, got < 0
            n += 1
        return n, False

    def _claim_one(self) -> int:
        """1 if a message was claimed and handed over, 0 if the inbox was empty, -1 if the claim failed on the way."""
        t0, tried = time.time(), self.town.retried
        try:
            msg = _once_more(self.town.claim)
            self.timed("claim", t0)
            self.claims += 1
        except Exception as exc:
            self.on_error("claim", exc)
            return -1
        if not msg:                                   # empty; but if the call had to be tried again, the first try
            return -1 if self.town.retried != tried else 0   # may have claimed a message whose answer was lost
        self.last_mail = time.time()
        if int(msg.get("attempt") or 1) <= 1 and not msg.get("duplicate"):
            with self._flock:
                self.fresh += 1
        try:
            self.handle(msg)
            self.handled += 1
        except Exception as exc:                      # not acknowledged: Town hands it out again after its lease
            self.on_error("handle", exc)
        return 1

    def stats(self) -> dict[str, Any]:
        return {"checks": self.checks, "claims": self.claims, "handled": self.handled,
                "recovered_without_hint": self.recovered, "every_s": self.every, "idle_every_s": self.idle_every}
