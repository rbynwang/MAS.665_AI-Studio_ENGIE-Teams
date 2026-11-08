"""What a live sale needs to survive a restart, and the truth about the day for /health.

Live only. Each service's main() attaches this (coordinator.py, fleet.py, utility.py, deploy_v4/run_buyer.py). tools/sim_event.py builds the same objects and never does,
so an offline run can't reach this file: Market.owed stays None and the fleet, the only other caller, isn't in the simulator at all.

What's kept, in the config's state/live folder:
  coordinator       every sale as it stands (offers, commitments, readings, orders, what each buyer settled and what each node was paid), the desk, reputations, the clock
  utility, others   each buyer's contracts with their readings and settlements, its orders, pins, calls, the replies Town hasn't taken yet, its clock
  nodes/<name>      offers, contracts, payments, readings held for the batch, the meter rows a chase needs, the battery, car, house and plan as they stand
  homes, buildings  the fleet's clock and what-if weather
  sent/<name>       what each participant sent (town.Sent): a line written before Town hears it, not on the timer below
  lost/<service>    a mark for each thing this service lost, so the alarm outlasts a restart (Keeper.lost)
Each service keeps its play tag too (ids.py), so a restarted one names the same messages, and with sent/ says them in the same bytes: Town answers replay, and nothing goes twice.
A node comes back where it stopped and plays the steps it missed (SimulatedHome.advance), so its rows have no hole. A node's agent isn't kept: it wakes, finds the sale
in the plan and plans it again, with its own log (node/record.py) to say what people said. Two things of its are, because a plan can't make them again: what its guard counts over a
day (a plant's one stop, a box's stored writes), and that its levers stood open. They come back shut, so until the agent wakes the controller uses only what a call opened, and that
is nothing: no device goes round a wish.

When it's written: every EVERY_S a service pickles what it holds, under its own lock, and writes it only if it changed. A message nobody would send again is on disk before
Town hears it was taken, so if the process dies there, Town hands it out again: every message to a node, and at the coordinator and the buyers a contract step, a settlement
or a call (a buyer that lost a contract it had signed answers "unknown transaction" to everything after it, and the sale never pays). What's left is the nodes' readings and
chatter at the coordinator and the buyers, which are many and cost too much to write one by one. A hard kill can lose up to EVERY_S of those, and they come back: a reading by
the chase, an offer by ask_silent, an answer by the commitment sent again.

Why pickle: slot ids are ints all through these dicts (ev["alloc"], od["share"], ct["meter_readings"]) next to baselines keyed by str(slot), and a JSON round trip turns
the first into the second, silently. It's also the fastest way to take a consistent copy under a lock. _Plain loads plain data and nothing else: no name in a file reaches a class or
any of numpy's own pickling, so a file can't run code, reach into memory or change a class for the whole process. dumps writes numpy as plain bytes for that reason.

/health (health below): `ok` is false while an alarm stands, and `alarms` says which. Each alarm is logged as an `alarm` event when it starts and `alarm_over` when it ends.
  town_down          no call to Town has worked for TOWN_DOWN_S
  messages_failing   FAIL_SHARE of this process's sends failed in the last minute, at least FAIL_MIN of them
  day_at_risk        a sale is open and either of those holds, or a day has no contract, or a settlement or readings are missing past their time, or an order was refused at the door,
                     or a sale is stuck clearing, or a buyer's sums aren't what its meters and its price come to, or a part of the market's tick fails TICK_FAILS times running
  keep_stuck         the keeper has missed STUCK_PASSES passes: a lock or the disk is stuck. /health works this out itself, since the keeper can't
  held               the fleet: an alarm holds this node, one alarm each, until a person's signed word ends it
"""
from __future__ import annotations

import argparse
import collections
import copy
import datetime
import hashlib
import io
import logging
import os
import pickle
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from . import beckn, config, replay as replay_mod

log = logging.getLogger("flexgrid-keep")
VERSION = 3                       # a file written by another shape of this module is set aside, not guessed at. 3: numpy goes as plain bytes, so version 2's arrays can't be read safely
EVERY_S = 3.0                     # the coordinator and the buyers. Measured at 203 nodes on the Mac: a sale is 2.3 MB and takes 12 ms under the market's lock, and writing is off it
NODES_EVERY_S = 5.0               # the fleet: each message already wrote its node, so a pass only adds the steps played since. 203 nodes, 3 MB: 32 ms, a node's lock at a time
MAX_AGE_S = float(beckn.TTL_EVIDENCE_S)   # older than this is another day: no message of its run is still valid
TOWN_DOWN_S = 30.0                # today's tunnel drops lasted 5 to 26 s, and a retry covers those
FAIL_MIN, FAIL_SHARE = 5, 0.2     # the usual background is about 2 % of sends (live_path.md)
MARK = b"\x80\x04C\x20"           # a file with a checksum starts so: pickle's protocol 4, then 32 bytes
SETTLE_LATE_MIN = 180             # the market waits 120 dataset minutes for readings, then a buyer 30 more
CLEARING_LATE_S = 180.0           # a clearing takes seconds, and a Langflow run 120 s at most (coordinator.run_flow). Past this nothing is running it
TICK_FAILS = 3                    # a part of the market's tick that fails this many seconds running is an alarm. One that fails once and goes on is only logged
BAD_FOR_S = 3600.0                # something lost stays an alarm this long, however often the service restarts
STUCK_PASSES = 10                 # a keeper this many passes behind is stuck: 30 s on the coordinator and the buyers, 50 s on the fleet. A clearing that holds the market's lock a few seconds isn't
KEEP_SEEN = 64                    # message ids a node keeps across a restart: Town hands out again only what wasn't acked, so the last few are enough
OPEN = ("offers", "clearing", "cleared", "utility", "homes", "settling")     # a sale still in progress

SENT_MARKET, SENT_NODE = 8192, 128     # bodies kept per participant for a retry (town.Sent): a coordinator's day at full size is a few thousand, a node's a few dozen
SALE_KEEP = ("need", "phase", "offers", "commitments", "readings", "baselines", "utxn")     # what every sale has from its first moment, and every pass over the sales reads
NEED_KEEP = ("event_id", "window_start", "window_end", "commit_by", "slots")
MARKET_KEEP = ("events", "reputation", "run_wall", "last_discover_day", "other_discover", "discover_asked", "discover_asked_wall", "discover_rounds", "too_late", "rated", "traffic", "play")
BUYER_KEEP = ("contracts", "orders", "pins", "event_days", "run_wall", "traffic_homes", "play")
WATCH_KEEP = ("alerts", "calls", "price_calls")
NODE_KEEP = ("offers", "contracts", "payments", "reputation", "history", "_fc_now", "_rt_fc", "_rt_reported", "_rt_at", "_rt_why", "play")
# everything that changes while a home plays. tests/test_restart.py fails if SimulatedHome grows state that isn't here or in its list of things left out
ADAPTER_KEEP = ("soc", "P", "_P0", "idx", "prev", "cf", "cf_before", "tel", "ev_left", "ev_deadline", "ev_session", "ev_wait", "ev_skip", "prev_wait", "ev_log", "cf_ev_log",
                "plan", "ev_holds", "override", "overrides", "fault", "frozen_reading", "x", "prev_x", "rt_extra", "dev_off", "kept_out", "app_q", "app_log", "app_hold_end",
                "v2h_day", "v2h_used", "v2h_owed", "prev_owed", "v2h_kw", "e_part", "meter_room", "hold_cold", "x_recent", "clips", "clip_at", "car_first", "note_seen")


# ---------------------------------------------------------------------- the files
def _array(kind: str, shape: tuple, data: bytes) -> Any:
    """numpy as a file may give it: plain numbers of a plain type. numpy's own pickling takes a dtype and a buffer from the file, and an object array over the file's bytes crashed the
    interpreter on every start, out of reach of any except."""
    d = np.dtype(kind) if isinstance(kind, str) else None
    if d is None or d.kind not in "biuf":
        raise pickle.UnpicklingError(f"not plain numbers: {kind!r}"[:80])
    return np.frombuffer(data, d).reshape(shape).copy()


def _scalar(kind: str, data: bytes) -> Any:
    return _array(kind, (), data)[()]


class _Make:
    """A constructor, as a file sees it. Handing the file the class itself let BUILD set attributes on it: Counter stayed broken for the whole process, and the load still returned."""
    __slots__ = ("make",)

    def __init__(self, make: Callable) -> None:
        self.make = make

    def __call__(self, *a: Any) -> Any:
        return self.make(*a)

    def __setstate__(self, state: Any) -> None:
        raise pickle.UnpicklingError("a constructor takes no state")


class _Plain(pickle.Unpickler):
    """Loads plain data and nothing else: the only names a file can use are these, and they only build a value."""
    ALLOWED = {("collections", "deque"): collections.deque, ("collections", "Counter"): collections.Counter, ("collections", "OrderedDict"): collections.OrderedDict,
               ("datetime", "date"): datetime.date, ("datetime", "datetime"): datetime.datetime, ("datetime", "timedelta"): datetime.timedelta,
               ("flexgrid_core.keep", "_array"): _array, ("flexgrid_core.keep", "_scalar"): _scalar}

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in self.ALLOWED:
            return _Make(self.ALLOWED[module, name])
        raise pickle.UnpicklingError(f"not plain data: {module}.{name}")


class _Writer(pickle.Pickler):
    def reducer_override(self, x: Any) -> Any:
        if isinstance(x, (np.ndarray, np.generic)):
            if x.dtype.kind not in "biuf" or (isinstance(x, np.ndarray) and type(x) is not np.ndarray):
                raise pickle.PicklingError(f"state holds what _Plain won't read back: {type(x).__name__} of {x.dtype}")      # an alarm now, not a lost sale at the next start
            return (_array, (x.dtype.str, x.shape, x.tobytes())) if isinstance(x, np.ndarray) else (_scalar, (x.dtype.str, x.tobytes()))
        return NotImplemented


def dumps(x: Any) -> bytes:
    """A pickle with its own checksum in front: MARK and the sum push 32 bytes, and the POP that starts `body` drops them. A reader that knows nothing of the sum still reads this, so
    the code of a rollback finds the version and sets the file aside as it should."""
    f = io.BytesIO(b"0")
    f.seek(1)
    _Writer(f, protocol=4).dump(x)
    body = f.getvalue()
    return MARK + hashlib.sha256(body).digest() + body


def loads(b: bytes) -> Any:
    """A file that changed after it was written is refused before anything in it is read: of 3,000 single-bit flips with no checksum, 57 % loaded as good with something different in
    them, and a battery at 0.81 came back at 1.62. A file with no sum is read as it is: it's from before this, and the version says what happens to it."""
    if b[:len(MARK)] == MARK and hashlib.sha256(b[len(MARK) + 32:]).digest() != b[len(MARK):len(MARK) + 32]:
        raise pickle.UnpicklingError("its checksum doesn't match: it changed after it was written")
    return _Plain(io.BytesIO(b)).load()


def from_a_node(msg: dict[str, Any]) -> bool:
    """A meter reading, or a batch of them: what a buyer takes 600 of in a sale, and what the chase asks for again."""
    m = (msg.get("body") or {}).get("message") or {}
    return "performance" in m or "batch" in m


def code_stamp() -> str:
    """A short hash of every .py file under flexgrid_core as this process found them at its start: /health "code", which deploy_v4/deploy_fix.sh compares with the tree it copied."""
    h = hashlib.sha1()
    root = config.REPO_DIR / "flexgrid_core"
    for p in sorted(x for x in root.rglob("*.py") if "__pycache__" not in x.parts):
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def write_atomic(path: Path, data: bytes) -> None:
    """The file is the old one or the new one, whatever happens: the bytes go to disk under another name in the same folder, then that name takes over."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stem = f".{path.name}."
    tmp = path.with_name(f"{stem}{os.getpid()}.{threading.get_ident()}")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    try:                                                  # the rename itself, past a power cut
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        with os.scandir(path.parent) as found:            # a kill before the rename leaves its temp file, and nothing else would ever remove it
            for f in found:
                if f.name.startswith(stem):
                    os.unlink(f.path)
    except OSError:
        pass


# ---------------------------------------------------------------------- what each service holds
def clock_take(r: Any) -> dict[str, Any]:
    wall0, step, i0, loop, held = r._line
    return {"wall0": wall0, "step": step, "at": r.ts(max(0, i0)), "loop": loop, "at_wall": r.at_wall, **({"held": True} if held is not None else {})}


def clock_give(r: Any, s: dict[str, Any] | None) -> bool:
    """A run's clock comes back on its own line, so every agent still agrees on the step. An idle clock isn't kept: each process starts on it by itself."""
    if not s or s["loop"] or s["at_wall"] is None:
        return False
    r.configure(s["at"], s["step"], 0, s["at"], s["wall0"], hold=bool(s.get("held")))     # a held clock comes back held: the conductor sets the next step
    return True


def market_take(m: Any) -> dict[str, Any]:
    return {**{k: getattr(m, k) for k in MARKET_KEEP}, "desk": {k: v for k, v in vars(m.rt).items() if k != "m"}, "owed": m.owed, "orders_refused": m.orders_refused}


def market_give(m: Any, s: dict[str, Any]) -> list[str]:
    """A step that was running on a thread when the process stopped isn't running now, and no tick picks these phases up. A sale caught clearing goes back to offers, and
    the tick clears it. A sale caught settling is settled again: nobody was paid, because a split happens whole under the market's lock. Returns those sales."""
    for k in MARKET_KEEP:
        setattr(m, k, s[k])
    vars(m.rt).update(s["desk"])
    if m.owed is not None:
        m.owed.update(s.get("owed") or {})
    m.orders_refused = dict(s.get("orders_refused") or {})     # older state has none
    settle = []
    for eid, ev in m.events.items():
        if ev["phase"] == "clearing":
            ev["phase"] = "offers"
            ev.pop("flow_failed", None)
        elif ev["phase"] == "settling" and not ev.get("settle_waiting"):
            ev["phase"] = "homes"
            settle.append(eid)
    return settle


def market_resume(m: Any, settle: list[str]) -> None:
    """What the market owes the moment Town answers again: the splits that were cut short, and the payment notices that hadn't gone. Nobody else sends those. A node counts
    a notice once, by its event and round, so one that did go arrives as a repeat and changes nothing."""
    for eid in settle:
        try:
            (m.on_settle or m.settle)(eid)
        except Exception as exc:                          # noqa: BLE001  a sale that can't settle stays an alarm, and the next one and the notices still go
            m.log("v2_split_failed", event_id=eid, why=f"{type(exc).__name__}: {exc}"[:160], top_up=False)
    items = list((m.owed or {}).values())
    if items:
        m._broadcast(items)
        m.log("v2_notices_sent_again", notices=len(items))


def buyer_take(u: Any) -> dict[str, Any]:
    return {**{k: getattr(u, k) for k in BUYER_KEEP}, "unsent": list(u.unsent), "watch": {k: getattr(u.watch, k) for k in WATCH_KEEP}}


def buyer_give(u: Any, s: dict[str, Any]) -> None:
    for k in BUYER_KEEP:
        setattr(u, k, s[k])
    u.unsent.extend(s["unsent"])
    for k in WATCH_KEEP:
        setattr(u.watch, k, s["watch"][k])


def rows_needed(h: Any) -> set[int]:
    """The meter rows a reading can still be made from: each half hour a node owes, and the hour the same-day adjustment reads. `recent` holds 400 rows a home and these
    are about 16, so a restart keeps what a chase needs and not the Control Room's charts."""
    r, out = h.replay, set()
    for c in h.contracts.values():
        if c.get("status") != "ACCEPTED":
            continue
        for x in c.get("commitments") or ():
            i = r.idx_of(x["start"])
            if i is not None:
                out.update((i, i + 1))
        w0 = c.get("window_start")
        j = r.idx_of(beckn.add_minutes(w0, -120)) if w0 else None
        if j is not None:
            out.update(range(j, j + 4))
    return out


def node_take(h: Any, seen: Any = (), agent: Any = None) -> dict[str, Any]:
    """Under the node's lock. agent: a thinking node's. What its levers held isn't here on purpose: its agent opens them again when it wakes."""
    a = h.adapter
    out: dict[str, Any] = {"node": {k: getattr(h, k) for k in NODE_KEEP if hasattr(h, k)}, "adapter": {k: getattr(a, k) for k in ADAPTER_KEEP if hasattr(a, k)},
                           "rows": {i: a.recent[i] for i in rows_needed(h) if i in a.recent}, "seen": dict(list(dict(seen).items())[-KEEP_SEEN:])}
    if getattr(a, "ctl", None) is not None:
        last = a.ctl.last                                 # its solver's own objects aren't state: the next re-plan makes them again
        out["ctl"] = {"last": last and {k: v for k, v in last.items() if k not in ("problem", "plan")}, "replans": a.ctl.replans, "dims": getattr(a.ctl, "_dims", None)}
    if hasattr(a.house, "cut"):                          # a building's plant cap
        out["house"] = {"cut": a.house.cut, "cut_span": a.house.cut_span}
    box = getattr(h, "outbox", None)
    if box is not None:
        out["outbox"] = {"held": box.held, "first": box.first, "last": box.last}
    g = getattr(agent, "guard", None)
    if hasattr(g, "day_stops"):
        out["guard"] = {"day_stops": dict(g.day_stops), "mode_writes": dict(g.mode_writes), "mode_day": g.mode_day}
    if agent is not None and getattr(a, "levers", None) is not None:
        out["levers_open"] = True
    return out


def node_give(h: Any, s: dict[str, Any], agent: Any = None) -> None:
    a = h.adapter
    for k, v in s["node"].items():
        setattr(h, k, v)
    for k, v in s["adapter"].items():
        setattr(a, k, v)
    a.recent.update(s["rows"])
    for c in h.contracts.values():
        c.pop("in_flight", None)                          # those sends died with the process: the slots are due again
    ctl = s.get("ctl")
    if ctl:
        a.ctl.last, a.ctl.replans = ctl["last"], ctl["replans"]
        if ctl.get("dims") is not None:
            a.ctl._dims = ctl["dims"]
    if s.get("house"):
        a.house.cut, a.house.cut_span = s["house"]["cut"], s["house"]["cut_span"]
    if s.get("outbox") and hasattr(h, "outbox"):
        h.outbox.held, h.outbox.first, h.outbox.last = s["outbox"]["held"], s["outbox"]["first"], s["outbox"]["last"]
    if agent is not None:
        if s.get("levers_open"):
            agent.open()
        g = getattr(agent, "guard", None)
        if s.get("guard") and hasattr(g, "day_stops"):
            g.day_stops.update(s["guard"]["day_stops"])
            g.mode_writes.update(s["guard"]["mode_writes"])
            g.mode_day = s["guard"]["mode_day"]


def open_business(h: Any) -> bool:
    """This node holds something it hasn't been paid for, or readings not yet sent."""
    paid = {p.get("event_id") for p in list(h.payments)}
    box = getattr(h, "outbox", None)
    return bool(box and box.held) or any(c.get("status") == "ACCEPTED" and c.get("event_id") not in paid for c in list(h.contracts.values()))


# ---------------------------------------------------------------------- is the day at risk?
def market_risks(m: Any, town_bad: bool) -> list[tuple[str, str]]:
    """What the market owes a person now. A sale or a call that can't be read is an alarm itself: one that stopped this list would hide the others."""
    out = []
    with m.lock:
        now = m.now()
        for part, f in getattr(m, "failing", {}).items():
            if f["times"] >= TICK_FAILS:
                out.append((f"tick:{part}", f"{part} has failed {f['times']} times in a row, so it stands still: {f['why']}"))
        for eid, o in getattr(m, "orders_refused", {}).items():
            if now < o["until"]:
                out.append((f"order:{eid}", f"{o['buyer']}'s order {eid} was refused ({o['why']}), so it's in no sale"))
        for cid, rec in getattr(getattr(m, "rt", None), "calls", {}).items():
            out += _unread(cid, lambda: call_risks(cid, rec, now))
        for eid, ev in m.events.items():
            out += _unread(eid, lambda: sale_risks(m, eid, ev, now, town_bad))
    return out


def _unread(what: str, risks: Callable[[], list[tuple[str, str]]]) -> list[tuple[str, str]]:
    try:
        return risks()
    except Exception as exc:                              # noqa: BLE001
        return [(f"unread:{what}", f"{what} can't be read: {type(exc).__name__}: {str(exc)[:120]}")]


def call_risks(cid: str, rec: dict[str, Any], now: str) -> list[tuple[str, str]]:
    out = []
    if rec.get("settlement_refused"):
        out.append((f"call:{cid}", f"{cid}: its buyer's settlement was refused ({rec['settlement_refused']}), so what it would pay hasn't gone"))
    if rec.get("differs"):
        out.append((f"word:{cid}", f"{cid}: its buyer's sums don't follow from its kWh and its price, and the coordinator's own full price stood in ({rec['differs']})"))
    return out if out and now < beckn.add_minutes(rec["end"], 1440) else []


def sale_risks(m: Any, eid: str, ev: dict[str, Any], now: str, town_bad: bool) -> list[tuple[str, str]]:
    out = []
    n, ph = ev["need"], ev["phase"]
    if ph == "clearing" and (ev.get("flow_failed") or time.time() >= float(ev.get("clearing_wall") or 0.0) + CLEARING_LATE_S or now >= n["commit_by"]):
        out.append((f"clearing:{eid}", f"{eid} is stuck clearing ({ev.get('flow_failed') or 'nothing is working on it'}), and no buyer or node has heard"))
    if ph in OPEN and town_bad:
        out.append((f"open:{eid}", f"{eid} is open, in phase {ph}, and messages aren't getting through"))
    if ph == "failed" and now < n["window_end"]:
        out.append((f"failed:{eid}", f"{eid} failed: no contract, and nobody has been told"))
    refused = [f"{od['buyer']}'s settlement was refused ({od['refused']})" for od in m._orders(ev) if od.get("refused")]
    if refused and ph != "done":
        out.append((f"settlement:{eid}", f"{eid}: {'; '.join(refused)}, and the split waits for every buyer"))
    if ev.get("split_failed"):
        out.append((f"split:{eid}", f"{eid}: the split can't finish ({ev['split_failed']}), so what it would pay hasn't gone"))
    differs = [f"{od['buyer']}: {od['differs']}" for od in m._orders(ev) if od.get("differs")]
    if differs and now < beckn.add_minutes(n["window_end"], 1440):
        out.append((f"word:{eid}", f"{eid}: a buyer's sums don't follow from its meters and its price, and the coordinator's own full price stood in ({'; '.join(differs)})"))
    if ph in OPEN and ev.get("telemetry_sent") and now >= beckn.add_minutes(n["window_end"], SETTLE_LATE_MIN) and not refused:
        owing = [od["buyer"] for od in m._orders(ev) if not od.get("settlement")]
        if owing:
            out.append((f"settlement:{eid}", f"{eid}: no settlement from {', '.join(owing)}, and the split waits for every buyer"))
        elif not ev.get("split_failed"):
            out.append((f"split:{eid}", f"{eid}: every buyer settled, and nobody has been paid"))
    if ev.get("readings_missing") and now < beckn.add_minutes(n["window_end"], 1440):
        out.append((f"readings:{eid}", f"{eid}: readings missing from {len(ev['readings_missing'])} nodes, and they can't be paid until they come"))
    return out


def buyer_risks(u: Any, town_bad: bool) -> list[tuple[str, str]]:
    if not town_bad:
        return []
    with u.lock:
        n = sum(1 for c in u.contracts.values() if c.get("status") == "ACTIVE") + sum(1 for c in u.watch.calls.values() if c.get("status") == "OPEN")
    return [("open", f"{n} contracts or calls are open, and messages aren't getting through")] if n else []


def held(f: Any) -> dict[str, Any]:
    """The nodes an alarm holds, and what it said. No lock: each node's own dict is read once, so a word that ends one while this runs can't trip it."""
    out = {}
    for n, h in list(f.homes.items()):
        for _ in range(5):
            try:
                ov = dict(getattr(h.adapter, "overrides", None) or {})
                break
            except RuntimeError:                          # the dict changed while it was copied: once more
                ov = {}
        if "alarm_hold" in ov:
            out[n] = ov["alarm_hold"]
    return out


def fleet_risks(f: Any, town_bad: bool) -> list[tuple[str, str]]:
    if not town_bad:
        return []
    n = 0
    for h in list(f.homes.values()):
        try:
            n += open_business(h)
        except RuntimeError:                              # a dict changed under this read, with no lock held: the next check counts it
            pass
    return [("open", f"{n} nodes hold a contract not yet paid or readings not yet sent, and messages aren't getting through")] if n else []


# ---------------------------------------------------------------------- does it fit?
def _need(d: Any, what: str, keys: Iterable[str]) -> None:
    if not isinstance(d, dict):
        raise ValueError(f"{what} isn't a table")
    gone = [k for k in keys if k not in d]
    if gone:
        raise ValueError(f"{what} has no {gone[0]}")


def _tables(x: Any, what: str) -> None:
    if not isinstance(x, dict) or not all(isinstance(v, dict) for v in x.values()):
        raise ValueError(f"{what} isn't a table of tables")


def clock_fits(r: Any, c: Any) -> None:
    """All that clock_give and FleetKeeper.restore read."""
    if c is None:
        return
    _need(c, "the clock", ("wall0", "step", "at", "loop", "at_wall"))
    if not c["loop"] and c["at_wall"] is not None:
        float(c["step"]), float(c["wall0"])
        if c["at"] not in r.index:
            raise ValueError(f"the clock's {c['at']!r} isn't in this dataset")


def market_fits(m: Any, s: Any) -> None:
    _need(s, "the market", (*MARKET_KEEP, "desk"))
    _tables(s["events"], "its sales")
    for ev in s["events"].values():
        _need(ev, "a sale", SALE_KEEP)
        _need(ev["need"], "a sale's need", NEED_KEEP)
    probe = copy.copy(m)                                  # the market's own reads of a sale, on a copy, with nothing changed yet
    probe.events = s["events"]
    probe.summary()
    _need(s["desk"], "the desk", ("calls",))
    _tables(s["desk"]["calls"], "the desk's calls")
    if not isinstance(s.get("owed") or {}, dict):
        raise ValueError("what's owed isn't a table")


def coordinator_fits(c: Any) -> Callable[[Any], None]:
    def fits(s: Any) -> None:
        _need(s, "the state", ("market", "clock", "what_if", "signed", "pace"))
        market_fits(c.market, s["market"])
        clock_fits(c.clock, s["clock"])
        if s["pace"] is not None and not isinstance(s["pace"], dict):
            raise ValueError("the pace isn't a table")
        if not isinstance(s["what_if"], dict) or not isinstance(s["signed"], set):
            raise ValueError("what-if or the buyer's sign-offs can't be read")
    return fits


def buyer_fits(u: Any) -> Callable[[Any], None]:
    def fits(s: Any) -> None:
        _need(s, "the state", ("buyer", "clock", "weather"))
        _need(s["buyer"], "the buyer", (*BUYER_KEEP, "unsent", "watch"))
        _need(s["buyer"]["watch"], "its watch", WATCH_KEEP)
        _tables(s["buyer"]["contracts"], "its contracts")
        if not all(isinstance(c.get("need", {}), dict) for c in s["buyer"]["contracts"].values()):
            raise ValueError("a contract's need isn't a table")
        if not all(len(x) == 4 for x in s["buyer"]["unsent"]):
            raise ValueError("what it hadn't sent can't be read")
        _tables(s["buyer"]["watch"]["calls"], "its calls")
        clock_fits(u.clock, s["clock"])
        dict(s["weather"])
    return fits


def node_fits(s: Any) -> None:
    """All that node_give reads, so a node that fails here is left as it was built, not half restored."""
    _need(s, "the node's state", ("node", "adapter", "rows", "seen"))
    for k in ("node", "adapter", "rows", "seen"):
        _need(s[k], k, ())
    for k, need in (("ctl", ("last", "replans")), ("house", ("cut", "cut_span")), ("outbox", ("held", "first", "last")), ("guard", ("day_stops", "mode_writes", "mode_day"))):
        if s.get(k):
            _need(s[k], k, need)


def fleet_fits(f: Any) -> Callable[[Any], None]:
    def fits(s: Any) -> None:
        _need(s, "the fleet's state", ("clock", "weather"))
        h = next(iter(f.homes.values()), None)
        if h is not None:
            clock_fits(h.replay, s["clock"])
        dict(s["weather"])
    return fits


def coordinator_holds(s: dict[str, Any]) -> bool:
    return any(ev.get("phase") in OPEN for ev in s["market"]["events"].values())


def buyer_holds(s: dict[str, Any]) -> bool:
    b = s["buyer"]
    return any(c.get("status") == "ACTIVE" for c in b["contracts"].values()) or any(c.get("status") == "OPEN" for c in b["watch"]["calls"].values())


def node_holds(s: dict[str, Any]) -> bool:
    n, paid = s["node"], {p.get("event_id") for p in s["node"].get("payments") or ()}
    return bool((s.get("outbox") or {}).get("held")) or any(c.get("status") == "ACCEPTED" and c.get("event_id") not in paid for c in (n.get("contracts") or {}).values())


def set_aside(path: Path) -> None:
    """A file that can't be used, moved out of the way and kept for a person to look at."""
    try:
        path.replace(path.with_name(path.name + ".old"))
    except OSError as exc:
        log.error("%s couldn't be set aside: %s", path, exc)


def sent(cfg: dict[str, Any], name: str, size: int = SENT_MARKET) -> Any:
    """This participant's journal of what it sent, on disk beside its state (town.Sent). keep.py keeps the play tag and this keeps the bodies: together a restarted service names the
    same messages and sends the same bytes, so Town answers replay. A snapshot every EVERY_S can't do that, because the send that matters is the one just before the kill."""
    from .town import Sent
    return Sent(size, config.state_dir() / "live" / "sent" / f"{name}.jsonl", cfg.get("town_run_id"))


# ---------------------------------------------------------------------- the service's keeper
class Keeper:
    """One service's state folder and its alarms. Without a thread (the tests): save(), load() and check() by hand."""

    def __init__(self, name: str, cfg: dict[str, Any], event: Callable[..., None], watch: Any = None, risks: Callable[[bool], list[tuple[str, str]]] | None = None,
                 every: float = EVERY_S, folder: Path | None = None):
        self.name, self.cfg, self.event, self.watch, self.every = name, cfg, event, watch, every
        self.risks = risks or (lambda bad: [])
        self.dir = Path(folder) if folder else config.state_dir() / "live"
        self.parts: Callable[[], Iterable[tuple[str, Callable[[], bytes | None]]]] = lambda: ()
        self.sha: dict[str, bytes] = {}                   # what each file holds now: unchanged state isn't written again
        self.saved_at = 0.0
        self.save_error: str | None = None
        self.started, self.code = time.time(), code_stamp()
        self.passed_at = self.started                     # when the last pass ended
        self.stuck_told = False
        self.alarms: list[dict[str, Any]] = []
        self.restored: dict[str, Any] | None = None
        self.bad: list[str] = []                          # what this service has lost, and bad_since when
        self.bad_since = self.started
        self.resume: Callable[[], None] | None = None     # runs once, when Town next answers
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._order = threading.Lock()                    # what's taken first is written first, so an older copy can't land on a newer one
        self._lost_before()

    def path(self, key: str) -> Path:
        return self.dir / f"{key}.state"

    def marks(self) -> Path:
        return self.dir / "lost" / self.name

    def _lost_before(self) -> None:
        """What an earlier start of this service lost. The file was moved out of the way, so the alarm has to live somewhere else: a restart ended it, and a crash loop hid it for good.
        Each mark lasts BAD_FOR_S. To end one sooner, delete it."""
        try:
            for m in list(self.marks().iterdir()):
                at = m.stat().st_mtime
                if self.started - at < BAD_FOR_S:
                    self.bad.append(m.name.replace("%", "/"))
                    self.bad_since = min(self.bad_since, at)
                else:
                    m.unlink(missing_ok=True)
        except OSError as exc:                            # no folder yet, as a rule. Whatever it is, this can't stop a start
            log.debug("lost marks: %s", exc)

    def put(self, key: str, blob: bytes) -> None:
        """Writes this part if it changed. The envelope says which run it belongs to, so state from another run is never loaded into this one."""
        d = hashlib.sha1(blob).digest()
        if self.sha.get(key) == d:
            return
        try:
            write_atomic(self.path(key), dumps({"version": VERSION, "saved_wall": time.time(), "run": self.cfg.get("town_run_id"), "dataset": config.DATASET, "blob": blob}))
        except OSError as exc:
            if self.save_error is None:
                self.event("state_not_saved", file=key, error=str(exc)[:160])
            self.save_error = str(exc)[:160]
            return
        self.sha[key], self.saved_at, self.save_error = d, time.time(), None

    def save(self) -> None:
        for key, take in self.parts():
            self._save(key, take)

    def _save(self, key: str, take: Callable[[], bytes | None]) -> None:
        """Nothing here may reach the message or the tick it runs beside. State that can't be copied is an alarm, not a stopped service."""
        with self._order:
            try:
                blob = take()
            except RuntimeError:                          # a thread without the lock added to a dict being copied: the next pass has it
                return
            except Exception as exc:                      # noqa: BLE001
                if self.save_error is None:
                    self.event("state_not_saved", file=key, error=f"{type(exc).__name__}: {str(exc)[:160]}")
                self.save_error = f"{type(exc).__name__}: {str(exc)[:160]}"
                return
            if blob is not None:
                self.put(key, blob)

    def load(self, key: str, now: float | None = None, fits: Callable[[Any], None] | None = None, holds: Callable[[Any], bool] | None = None) -> Any:
        """The state of this part, or None. A file that doesn't belong to this run is set aside, and one that can't be read counts as an alarm too: it held a sale.
        fits: raises if what the file holds isn't what this code reads. A key gone is what a code change with no new VERSION would do, and it's no better than a torn file:
        given half of it, a service dies at its start, and under Restart=always it loops for ever with no /health to ask.
        holds: says if a file too old to load still held something open. Then it's an alarm too: a sale went with it."""
        p = self.path(key)
        if not p.exists():
            return None
        why = bad = None
        try:
            e = loads(p.read_bytes())
            age = (time.time() if now is None else now) - float(e.get("saved_wall") or 0.0)
            if e.get("version") != VERSION:
                why = f"written by version {e.get('version')}"
            elif e.get("run") != self.cfg.get("town_run_id") or e.get("dataset") != config.DATASET:
                why = "another run or dataset"
            elif age > MAX_AGE_S:
                why = f"{age / 3600:.0f} hours old"
                if holds is not None and holds(loads(e["blob"])):
                    why = bad = f"{why}, with business still open in it"
            else:
                out = loads(e["blob"])
                if fits is not None:
                    fits(out)
                self.sha[key] = hashlib.sha1(e["blob"]).digest()
                return out
        except Exception as exc:                          # noqa: BLE001  whatever is wrong with the file, the service must start
            why = bad = f"unreadable: {type(exc).__name__}: {str(exc)[:120]}"
        try:
            p.replace(p.with_suffix(".old"))
        except OSError:
            pass
        if bad:
            self.lost(key, why)
        else:
            self.event("state_set_aside", file=key, why=why)
        return None

    def lost(self, key: str, why: str) -> None:
        """Something this service held is gone: an alarm, it says so on the record, and it stays an alarm through a restart."""
        if key not in self.bad:
            self.bad.append(key)
        self.event("state_set_aside", file=key, why=why)
        try:
            write_atomic(self.marks() / key.replace("/", "%"), why.encode())
        except OSError as exc:
            log.error("%s: %s", key, exc)

    def journals(self, sents: Iterable[Any]) -> None:
        for s in sents:
            if getattr(s, "damaged", None):
                self.lost(f"sent/{s.path.stem}", s.damaged)

    def town_alarms(self, now: float) -> dict[str, str]:
        """What needs no lock, so it can be told even when the keeper is stuck."""
        new: dict[str, str] = {}
        if self.watch is not None:
            t = self.watch.state(now)
            if t["town_ok_s"] >= TOWN_DOWN_S:
                new["town_down"] = f"no call to NANDA Town has worked for {t['town_ok_s']:.0f} s"
            if t["failed"] >= FAIL_MIN and t["failed"] >= FAIL_SHARE * t["sends"]:
                new["messages_failing"] = f"{t['failed']} of {t['sends']} sends failed in the last minute"
        return new

    def stuck(self, now: float | None = None) -> list[dict[str, Any]]:
        """The alarms this keeper can't raise, because it isn't passing: it's blocked on a lock or a disk, so what it last said is old. With the market's lock stuck and Town gone
        ten minutes, /health said ok. Called from /health, which holds no lock."""
        now = time.time() if now is None else now
        late = now - self.passed_at
        if self._thread is None or self._stop.is_set() or late <= STUCK_PASSES * self.every:
            return []
        why = f"no pass of the keeper for {late:.0f} s, so a lock or the disk is stuck and nothing is being saved"
        if not self.stuck_told:
            self.stuck_told = True
            log.error("%s alarm: %s", self.name, why)
            self.event("alarm", what="keep_stuck", why=why)
        have = {a["key"] for a in self.alarms}
        return [{"what": k, "why": v, "since": round(self.passed_at, 1)} for k, v in {"keep_stuck": why, **self.town_alarms(now)}.items() if k not in have]

    def standing(self) -> dict[str, str]:
        """Alarms of the service's own kind, not a day's: FleetKeeper's held nodes."""
        return {}

    def check(self, now: float | None = None) -> list[dict[str, Any]]:
        """What's wrong now. An alarm is logged when it starts and when it ends, and /health reads the list."""
        now = time.time() if now is None else now
        new = self.town_alarms(now)
        new.update({f"day_at_risk:{k}": v for k, v in self.risks(bool(new))})
        new.update(self.standing())
        if self.bad and now - self.bad_since < BAD_FOR_S:
            new["day_at_risk:state"] = f"{len(self.bad)} state files were lost, and what they held is gone: {', '.join(self.bad[:4])}"
        if self.save_error:
            new["state_not_saved"] = f"state isn't reaching the disk: {self.save_error}"
        was = {a["key"]: a for a in self.alarms}
        for k in was.keys() - new.keys():
            self.event("alarm_over", what=was[k]["what"], why=was[k]["why"], lasted_s=round(now - was[k]["since"]))
        out = []
        for k, why in new.items():
            a = was.get(k) or {"key": k, "what": k.split(":")[0], "since": round(now, 1)}
            if k not in was:
                log.error("%s alarm: %s", self.name, why)
                self.event("alarm", what=a["what"], why=why)
            out.append({**a, "why": why})
        self.alarms = out
        return out

    def _pass(self) -> None:
        if self.resume is not None and (self.watch is None or self.watch.ok_at > 0):
            again, self.resume = self.resume, None
            again()
        self.save()
        self.check()
        self.passed_at = time.time()
        if self.stuck_told:
            self.stuck_told = False
            self.event("alarm_over", what="keep_stuck")

    def loop(self) -> None:
        while not self._stop.wait(self.every):
            try:
                self._pass()
            except Exception as exc:                      # noqa: BLE001  the keeper must not die of one bad pass
                log.exception("keep")
                self.event("keep_error", error=str(exc)[:200])

    def start(self) -> None:
        self._thread = threading.Thread(target=self.loop, daemon=True, name=f"keep-{self.name}")
        self._thread.start()

    def close(self) -> None:
        """A clean stop writes what the last pass didn't."""
        self._stop.set()
        self.save()


def health(k: Keeper | None) -> dict[str, Any]:
    """What goes into /health. It reads the keeper's last check, not the service's locks, so /health answers even when a lock is stuck, and says so."""
    if k is None:
        return {}
    t = k.watch.state() if k.watch is not None else {}
    alarms = [{x: a[x] for x in ("what", "why", "since")} for a in k.alarms] + k.stuck()
    return {"ok": not alarms, "alarms": alarms, "code": k.code,
            "keep": {**t, "saved_s_ago": round(time.time() - k.saved_at, 1) if k.saved_at else None, "restored": k.restored}}


# ---------------------------------------------------------------------- one call from each main()
def for_coordinator(c: Any) -> Keeper:
    """Before the inbox and the market's tick start. A sale that was running carries on, on its own clock. Town's waiting mail reaches it as it would have."""
    m = c.market
    m.owed = {}
    if c.town:
        c.town.sent = sent(c.cfg, c.town.name)
    k = Keeper("coordinator", c.cfg, c.event, watch=c.town.watch if c.town else None, risks=lambda bad: market_risks(m, bad))
    for key, why in getattr(c, "damaged", ()):
        k.lost(key, why)
    k.journals([c.town.sent] if c.town else [])

    def blob() -> bytes:
        with m.lock:
            return dumps({"market": market_take(m), "clock": clock_take(c.clock), "what_if": getattr(c, "what_if", {}), "signed": c.buyer_signed,
                          "pace": {x: c.pace.get(x) for x in ("x", "auto", "slow", "day", "start_wall")} if getattr(c, "pace", None) else None})
    k.parts = lambda: [("coordinator", blob)]
    s = k.load("coordinator", fits=coordinator_fits(c), holds=coordinator_holds)
    if s is not None:
        fresh = market_take(m)
        try:
            settle = market_give(m, s["market"])
            clocked = clock_give(c.clock, s["clock"])
            c.what_if, c.buyer_signed = s["what_if"], s["signed"]
            c.pace_saved = s["pace"] if clocked and (s["pace"] or {}).get("slow") else None
            k.resume = lambda: market_resume(m, settle)
            k.restored = {"at": round(time.time(), 1), "clock": clocked, "sales": {e: v["phase"] for e, v in m.events.items()}}
            c.event("state_restored", **k.restored)
        except Exception as exc:                          # noqa: BLE001  the service must start, on nothing rather than on half of it
            m.owed.clear()
            market_give(m, fresh)
            k.resume = None
            set_aside(k.path("coordinator"))
            k.lost("coordinator", f"{type(exc).__name__}: {str(exc)[:120]}")
    c.keeper = k
    k.start()
    return k


def for_buyer(u: Any) -> Keeper:
    u.town.sent = sent(u.cfg, u.name)
    k = Keeper(u.name, u.cfg, u.event, watch=u.town.watch, risks=lambda bad: buyer_risks(u, bad))
    k.journals([u.town.sent])

    def blob() -> bytes:
        with u.lock:
            return dumps({"buyer": buyer_take(u), "clock": clock_take(u.clock), "weather": dict(replay_mod.WEATHER)})
    k.parts = lambda: [(u.name, blob)]
    s = k.load(u.name, fits=buyer_fits(u), holds=buyer_holds)
    if s is not None:
        fresh = {**buyer_take(u), "unsent": []}
        try:
            buyer_give(u, s["buyer"])
            clocked = clock_give(u.clock, s["clock"])
            replay_mod.WEATHER.update(s["weather"])
            k.restored = {"at": round(time.time(), 1), "clock": clocked, "contracts": {c.get("need", {}).get("event_id"): c.get("status") for c in u.contracts.values()}}
            u.event("state_restored", **k.restored)
        except Exception as exc:                          # noqa: BLE001  the service must start, on nothing rather than on half of it
            u.unsent.clear()
            buyer_give(u, fresh)
            set_aside(k.path(u.name))
            k.lost(u.name, f"{type(exc).__name__}: {str(exc)[:120]}")
    u.keeper = k
    k.start()
    return k


class FleetKeeper(Keeper):
    """A file per node: they change at different times, and one bad file costs one node."""

    def __init__(self, f: Any):
        super().__init__(f.kind, f.cfg, self._event, watch=f.watch, risks=lambda bad: fleet_risks(f, bad), every=NODES_EVERY_S)
        self.f = f

    def _event(self, kind: str, **d: Any) -> None:
        self.f.events.append({"at": time.time(), "home": None, "kind": kind, **d})

    def standing(self) -> dict[str, str]:
        """A node an alarm holds is an alarm here too, one each: `held` on /health was read by nobody, and the watcher reads only these. It stands through restarts and new runs, as the hold does."""
        return {f"held:{n}": f"{n} is held by an alarm ({str(why)[:80]}): it sells nothing, and only a person's signed word ends it" for n, why in held(self.f).items()}

    def _blob(self, name: str) -> bytes:
        return dumps(node_take(self.f.homes[name], self.f.seen.get(name, ()), self.f.agents.get(name)))

    def node_done(self, name: str) -> None:
        """Called with the node's lock held, before Town hears the message was taken."""
        self._save(f"nodes/{name}", lambda: self._blob(name))

    def _take(self, name: str) -> bytes | None:
        lock = self.f.locks[name]
        if not lock.try_enter():                          # busy: the next pass. A pass must never make the tick wait
            return None
        try:
            return self._blob(name)
        finally:
            lock.leave()

    def _fleet(self) -> bytes:
        h = next(iter(self.f.homes.values()), None)
        return dumps({"clock": clock_take(h.replay) if h else None, "weather": dict(replay_mod.WEATHER)})

    def all_parts(self) -> Iterable[tuple[str, Callable[[], bytes | None]]]:
        yield self.f.kind, self._fleet
        for name in list(self.f.homes):
            yield f"nodes/{name}", lambda n=name: self._take(n)

    def restore(self) -> None:
        f = self.f
        s = self.load(f.kind, fits=fleet_fits(f)) or {}   # the nodes come back without it: the coordinator sends the clock again (_clock_again), and nothing would give back a contract
        c = s.get("clock")
        clocked = bool(c and not c["loop"] and c["at_wall"] is not None)
        if clocked:
            f.configure_clock(start=c["at"], at=c["at"], at_wall=c["wall0"], step_seconds=c["step"], loop_days=0, keep=True, hold=bool(c.get("held")))
        replay_mod.WEATHER.update(s.get("weather") or {})
        n = 0
        for name, h in f.homes.items():
            d = self.load(f"nodes/{name}", fits=node_fits, holds=node_holds)
            if d is None:
                continue
            try:
                node_give(h, d, f.agents.get(name))
                f.seen[name] = dict(d["seen"])
                n += 1
            except Exception as exc:                      # noqa: BLE001  one node's bad state must not stop the other 199
                self.bad.append(name)
                self._event("state_set_aside", file=name, why=f"{type(exc).__name__}: {str(exc)[:120]}")
        f.person_back(self)                               # whatever came back or didn't: an alarm or a person's limit isn't this folder's to lose
        if n or clocked or self.bad:
            self.restored = {"at": round(time.time(), 1), "clock": clocked, "nodes": n, "of": len(f.homes), "unread": len(self.bad)}
            self._event("state_restored", **self.restored)


def for_fleet(f: Any) -> FleetKeeper:
    """Before the fleet's threads start."""
    from .town import TownWatch
    f.watch = TownWatch()
    for n, t in f.towns.items():
        t.watch = f.watch
        t.sent = sent(f.cfg, n, SENT_NODE)
    k = FleetKeeper(f)
    for key, why in getattr(f, "damaged", ()):
        k.lost(key, why)
    k.journals(t.sent for t in f.towns.values())
    k.parts = k.all_parts
    k.restore()
    f.keep = k
    k.start()
    return k


def main() -> None:
    """A state file, for a person: what it says it holds. No service needs this."""
    ap = argparse.ArgumentParser(prog="python -m flexgrid_core.keep", description=main.__doc__)
    ap.add_argument("file")
    e = loads(Path(ap.parse_args().file).read_bytes())
    s = loads(e["blob"])
    print({k: v for k, v in e.items() if k != "blob"}, f"saved {time.time() - e['saved_wall']:.0f} s ago")
    for k, v in s.items():
        print(f"  {k}: {len(dumps(v)):,} bytes")
    for eid, ev in ((s.get("market") or {}).get("events") or {}).items():
        print(f"  sale {eid}: {ev['phase']}, {len(ev['offers'])} offers, {sum(len(x) for x in ev['readings'].values())} readings, paid {bool(ev.get('result'))}")


if __name__ == "__main__":
    sys.exit(main())
