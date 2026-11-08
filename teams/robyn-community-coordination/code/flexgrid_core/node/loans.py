"""A node with no agent of its own borrows the model for one decision.

Five nodes of 203 think, and a note written to any other home used to go unread: rules can't read free text, and the 9 Jul note case failed for that reason alone. Here a code-only
node that holds a new note waits at the desk, and the next thinking node to wake asks the model for it, through the same queue and in the owner's own short step. The wish then
sits on the node, and the node goes on under its controller, with what the wish keeps out left out, as a thinking node does when its model goes.

A number doesn't borrow. At a code-only node the controller already has every device open, so a shortfall or a tripped battery has no choice left in it: code reports it, and
that report is at the desk at once.

The load stays small, and no note misses its sale for want of budget. Each note is read once, the nearest sale first. A step reads PER_STEP, or more if what waits couldn't be read in
time at that pace: within the hour, or before its sale if that comes sooner, and MAX_STEP at most, because the lender's own step waits for these answers. Only an answer counts: a try
the model never took costs nothing, and after one such try the rest wait for the next step. PER_DAY is a ceiling for a day of floods, above what the whole community can write at once.
A note still unread when its sale starts goes on the record and in the node's log, so a person's need never goes quietly unkept.

A loan lasts as long as its wishes: NodeAgent.lapse says when each ends.
"""
from __future__ import annotations

import contextlib
import copy
import math
import threading
from typing import Any, Callable

from . import owner

PER_STEP = 2
MAX_STEP = 16          # a note takes about 3 s of the model, so at most about 48 s of the lender's step, and only while so many wait: a storm warning to every owner an hour before a sale
CLEAR_STEPS = 4        # what waits is read within this many 15-minute steps
PER_DAY = 240          # more than one note for each of the 203 nodes: about 11 minutes of the model
FAR = "9999"           # sorts after every time: a node with no sale yet


class Desk:
    def __init__(self) -> None:
        self.lender: Any = None                     # any agent in this process: it holds the way to the model
        self.tier = 1
        self.waiting: list[Any] = []
        self.lent: list[Any] = []                   # agents made for a loan, until their sale ends
        self.at: dict[str, int] = {}                # answers given, by step and by day
        self.failed: str | None = None              # the last step a try found no model
        self.since: str | None = None               # when notes began to wait
        self.queue = threading.Lock()               # a home's tick adds to the queue while a lender's step takes from it
        self.serving = threading.Lock()             # one lender at a time: the next to wake finds the step's answers spent
        self.node_lock: Callable[[str], Any] = lambda name: contextlib.nullcontext()      # whoever holds a node's lock may touch its state: fleet.py gives its own


DESK = Desk()


def open_desk(agent: Any) -> None:
    """Every agent says it is here. A new run's agents take over from an old run's."""
    if DESK.lender is None or DESK.lender.home is not agent.home:
        DESK.lender = agent
    if agent.word == "home":
        DESK.tier = agent.tier


def forget(home: Any) -> None:
    """A new run: this node's loan goes with the last run, its wish and its agent too, or the node's next offer leaves out what nobody asked this run to keep. So does what the desk spent: it counts by the
    date and the step, and the same date is about to come round again. No lock of the desk's here: a lender holds DESK.serving while it waits for this node's lock, and the fleet holds that lock now."""
    with DESK.queue:
        if home in DESK.waiting:
            DESK.waiting.remove(home)
    for a in [x for x in list(DESK.lent) if x.home is home]:
        DESK.lent.remove(a)
        a.finish()
    home.note_asked = home.note_late = None
    DESK.at.clear()
    DESK.failed = DESK.since = None


def ask(home: Any) -> None:
    """A node with no agent has a note. It goes once per note, and only where there is a model to ask."""
    ov = getattr(home.adapter, "overrides", None) or {}
    key = owner.note_key(ov)
    if not ov.get("note") or DESK.lender is None or getattr(home, "note_asked", None) == key:
        return
    home.note_asked = key
    with DESK.queue:
        if home not in DESK.waiting:                # a newer note on a node still waiting takes the older one's place
            DESK.waiting.append(home)


def next_sale(home: Any, ts: str) -> str:
    """The next step this node owes kW, from its contract: the sale its note has to reach. A node with no contract yet goes after those with one."""
    plan = getattr(getattr(home, "adapter", None), "plan", None) or {}
    return min((t for t, a in plan.items() if t >= ts and float(a.get("deliver_kw") or 0.0) > 0), default=FAR)


def pace(ts: str) -> int:
    """How many this step may read: enough to clear what waits in time, counted from when the desk last stood empty."""
    from ..replay import Replay
    if not DESK.waiting:
        DESK.since = None
        return PER_STEP
    DESK.since = DESK.since or ts
    steps = CLEAR_STEPS - int(Replay.minutes_between(DESK.since, ts) // 15)
    soon = min(next_sale(h, ts) for h in DESK.waiting)
    if soon != FAR:
        steps = min(steps, int(Replay.minutes_between(ts, soon) // 15))
    return max(PER_STEP, min(MAX_STEP, math.ceil(len(DESK.waiting) / max(1, steps))))


def spend(ts: str) -> None:
    for k in (ts, ts[:10]):
        DESK.at[k] = DESK.at.get(k, 0) + 1


def too_late(agent: Any, ts: str) -> None:
    """The sale has started and these notes are still unread: the record says so, once for each note."""
    for home in DESK.waiting:
        ov = getattr(getattr(home, "adapter", None), "overrides", None) or {}
        owed = ((getattr(home.adapter, "plan", None) or {}).get(ts) or {}).get("deliver_kw", 0.0) if ov.get("note") else 0.0
        if float(owed or 0.0) > 0 and getattr(home, "note_late", None) != owner.note_key(ov):
            home.note_late = owner.note_key(ov)
            agent.record({**agent.blank(ts, "owner's note", 0.0, text=f"{home.name}: a note was still unread when its sale started."), "node": home.name, "lent": True, "note_late": True})
            home.log("note_unread", ts=ts, waiting=len(DESK.waiting))


def serve(agent: Any, ts: str) -> None:
    """A thinking node, as it wakes, hands the desk's next notes to the model. Another node serving now has this step in hand."""
    if DESK.serving.acquire(blocking=False):
        try:
            serve_now(agent, ts)
        finally:
            DESK.serving.release()


def nearest(ts: str) -> Any:
    """The next to read, off the queue."""
    with DESK.queue:
        home = min(DESK.waiting, key=lambda h: next_sale(h, ts))
        DESK.waiting.remove(home)
    return home


def holding(name: str, agent: Any) -> Any:
    """That node's lock, unless the lender already holds it. A node lent to as code-only and made a thinker in a new run would wait on itself."""
    return contextlib.nullcontext() if name == getattr(agent, "name", None) else DESK.node_lock(name)


def serve_now(agent: Any, ts: str) -> None:
    for a in list(DESK.lent):
        with holding(a.name, agent):
            done = a.over(ts)
            if done:
                a.finish()
        if done and a in DESK.lent:                 # a new run may have taken it off already
            DESK.lent.remove(a)
    if DESK.failed == ts:
        return                                      # no model at this step: another node's try found that
    most = pace(ts)
    room = lambda: DESK.at.get(ts, 0) < most and DESK.at.get(ts[:10], 0) < PER_DAY      # noqa: E731
    for a in list(DESK.lent):                       # a second note on a node already on loan
        ov = getattr(a.sim, "overrides", None) or {}
        if room() and ov.get("note") and owner.note_key(ov) != a.noted:
            with holding(a.name, agent):
                a.read_note(ts)
            if a.noted != owner.note_key(ov):
                DESK.failed = ts
                return
            a.home.note_asked = a.noted             # read on its loan: when the loan ends it doesn't go to the desk again
            spend(ts)
    while DESK.waiting and room():
        home = nearest(ts)
        with holding(home.name, agent):             # the model's short wait goes with it: this node's tick skips a pass, and catches up
            said = lend(agent, home, ts)
        if said is False:
            DESK.failed = ts                        # it goes round again, and the rest wait for the next step
            return
        spend(ts)
    too_late(agent, ts)


def lent_brain(brain: Any, name: str) -> Any:
    """The lender's way to the model, as this node's: the queue counts the answer against the node it was for, so a flood of notes can't spend the lender's own answers. On a fleet that way is a
    fleet.NodeBrain inside a record.RecordingBrain, and the queue reads the name from the one inside. It keeps the lender's lock, so the lender's tick runs on while the model answers. It reads the
    lender's step to say what kind of job this is, and a note is none of those: with no agent it goes as a sale step."""
    out = core = copy.copy(brain)
    if hasattr(out, "inner"):
        out.inner = core = copy.copy(out.inner)
    if hasattr(core, "node"):
        core.node = name
    if hasattr(core, "agent"):
        core.agent = None
    return out


def lend(agent: Any, home: Any, ts: str) -> bool:
    """One decision: the agent is made for it, reads the note, and hands the node back to its controller with the wish in force. False: the model never read it, so it goes to the desk again."""
    from .agent import NodeAgent
    from .building_agent import BuildingAgent
    brain = lent_brain(agent.brain, home.name)
    record = lambda r: agent.record({**r, "lent": True})     # noqa: E731
    a = BuildingAgent(home, brain, record) if getattr(home, "is_building", False) else NodeAgent(home, DESK.tier, brain, record)
    a.lent_on = ts
    said = a.read_note(ts)
    if a.noted is None:
        home.agent = None
        home.note_asked = None
        return False
    if said and a.guard.wishes:
        a.ctx.hand_back()
        DESK.lent.append(a)
    else:
        home.agent = None                           # no wish: nothing changes, and the node stays code-only
    return True
