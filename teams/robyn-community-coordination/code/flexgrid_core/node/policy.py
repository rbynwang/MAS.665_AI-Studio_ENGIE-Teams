"""Who may change what: deny by default, kept as data, and run before the guard (MULTI_AGENT_DESIGN, section 6).

The guard asks "is this safe right now?". Policy asks first "may this one ask at all?".

  Who           who is asking. Code sets it, never the model, so a tool's arguments can't claim to be the owner
  Policy.check  one answer: allowed or not, with the reason and the rule that decided
  OwnerChannel  how a person's change reaches a node: a signed message, never text a model read

Device groups are the categories in tools.json. Three rules hold whatever else the data says: nobody touches life safety or
switches the guard off, the coordinator commands no device, and the agent raises no limit.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

from .. import signing
from . import tools as T

VERSION = "2026-11-06"
ROLES = ("owner", "facility_manager", "agent", "coordinator")
VERBS = ("read", "call", "trade", "set_limit", "need_now", "close_group", "open_group", "clear_alarm", "request", "move_gap")
SITE, MARKET, LIFE_SAFETY, GUARD = "site", "market", "life_safety", "guard"      # beside the device groups
PERSON_VERBS = ["read", "set_limit", "need_now", "close_group", "open_group", "clear_alarm"]

# The range a person may set a limit in. These sit inside what the device can do: a person sets limits, not physics.
LIMITS: dict[str, dict[str, tuple[float, float]]] = {
    "hvac_zones": {"max_rise_c": (0.0, 4.0), "max_rise_away_c": (0.0, 4.0), "precool_c": (0.0, 3.0)},
    "energy_storage": {"reserve_pct": (0.0, 100.0)},
}


@dataclass(frozen=True)
class Who:
    role: str
    via: str = "code"
    proven: bool = False           # only OwnerChannel and the market gate set this


AGENT = Who("agent", "model")


@lru_cache(maxsize=1)
def groups() -> dict[str, str]:
    """Every device tool's group, homes and buildings together."""
    b = T.bundle()
    return {t: c["category"] for side in ("home", "building") for t, c in b[side].items()}


def group_of(tool: str) -> str:
    return groups().get(tool, "")


def device_groups() -> list[str]:
    return sorted(set(groups().values()))


def default_rules() -> dict[str, Any]:
    """The table. `never` beats everything, then a person's closed groups, then `allow`. Anything else is denied."""
    dev = device_groups()
    return {"version": VERSION,
            "never": [
                {"who": "*", "groups": [LIFE_SAFETY, GUARD], "verbs": ["*"], "why": "Nobody touches life safety or switches the guard off."},
                {"who": "coordinator", "groups": dev, "verbs": ["*"], "why": "The coordinator may request and move a gap. It commands no device."},
                {"who": "agent", "groups": ["*"], "verbs": ["set_limit", "need_now", "close_group", "open_group", "clear_alarm"], "why": "The agent can't change a limit."}],
            "allow": [
                {"who": "owner", "side": "home", "groups": [SITE, *dev], "verbs": PERSON_VERBS, "proven": True},
                {"who": "facility_manager", "side": "building", "groups": [SITE, *dev], "verbs": PERSON_VERBS, "proven": True},
                {"who": "owner", "side": "building", "groups": [SITE, *dev], "verbs": ["read"], "proven": True},
                {"who": "agent", "side": "*", "groups": [SITE, MARKET, *dev], "verbs": ["read"]},
                {"who": "agent", "side": "*", "groups": dev, "verbs": ["call"]},
                {"who": "agent", "side": "*", "groups": [MARKET], "verbs": ["trade"]},
                {"who": "coordinator", "side": "*", "groups": [MARKET], "verbs": ["request", "move_gap"], "proven": True}]}


def _hit(rule: Any, who: Who, verb: str, group: str, side: str) -> bool:
    """A rule that isn't the right shape matches nothing, so it allows nothing: text in place of a list would match by substring."""
    if not isinstance(rule, dict) or not isinstance(rule.get("groups"), list) or not isinstance(rule.get("verbs"), list):
        return False
    return (rule.get("who") in ("*", who.role) and rule.get("side", "*") in ("*", side)
            and ("*" in rule["groups"] or group in rule["groups"])
            and ("*" in rule["verbs"] or verb in rule["verbs"]))


class Policy:
    """One per node. `closed` is a person's own "not this group": it goes through OwnerChannel and nothing else."""

    def __init__(self, side: str, rules: dict[str, Any] | None = None):
        self.side = side
        self.rules = rules if rules is not None else default_rules()
        self.closed: dict[str, str] = {}             # group -> who closed it

    @classmethod
    def from_file(cls, side: str, path: str | Path) -> "Policy":
        """A file that can't be read denies everything, rather than falling back to something looser."""
        try:
            r = json.loads(Path(path).read_text())
            if not isinstance(r.get("allow"), list) or not isinstance(r.get("never"), list):
                raise ValueError("needs allow and never lists")
            r["never"] = default_rules()["never"] + r["never"]      # a file can add to never, not take from it
            return cls(side, r)
        except (OSError, ValueError, AttributeError):
            return cls(side, {"version": "unreadable", "never": default_rules()["never"], "allow": []})

    def check(self, who: Who, verb: str, group: str) -> dict[str, Any]:
        def no(why: str, rule: str) -> dict[str, Any]:
            return {"allow": False, "why": why, "rule": rule}
        if not isinstance(who, Who) or who.role not in ROLES:
            return no("Nobody by that name may ask.", "default")
        if verb not in VERBS or not isinstance(group, str) or not group:
            return no("That isn't something anyone may ask.", "default")
        for i, r in enumerate(self.rules.get("never", [])):
            if _hit(r, who, verb, group, self.side):
                return no(r.get("why") or "Never allowed.", f"never.{i}")
        if who.role == "agent" and verb == "call" and group in self.closed:
            return no(f"The {self.closed[group].replace('_', ' ')} closed this group to the agent.", "closed")
        for i, r in enumerate(self.rules.get("allow", [])):
            if _hit(r, who, verb, group, self.side):
                if r.get("proven") and not who.proven:
                    return no("This needs the signed channel.", f"allow.{i}")
                return {"allow": True, "why": "", "rule": f"allow.{i}"}
        return no("Nothing allows this, so it's denied.", "default")


class OwnerChannel:
    """A person's change reaches a node only here. Each message is signed with that person's key, carries a rising number so
    an old one can't be sent again, and names this node, so text a model read has no way in.

    apply(verb, group, name, value) does the change: context.py passes it down to the driver.
    """

    def __init__(self, node: str, policy: Policy, keys: dict[str, str], apply: Callable[[str, str, str, float | None], dict[str, Any]],
                 record: Callable[..., Any] | None = None):
        self.node, self.policy, self.keys, self.apply = node, policy, dict(keys), apply
        self.record = record or (lambda *a, **k: None)
        self.seq: dict[str, int] = {}

    def receive(self, msg: dict[str, Any]) -> dict[str, Any]:
        m = msg if isinstance(msg, dict) else {}
        out = self._receive(m)
        self.record("owner_channel", **{k: m.get(k) for k in ("role", "verb", "group", "name", "value", "seq")}, result=out)
        return out

    def _receive(self, msg: dict[str, Any]) -> dict[str, Any]:
        def no(why: str) -> dict[str, Any]:
            return {"status": "blocked", "why": why}
        role, verb, group, name = msg.get("role"), msg.get("verb"), msg.get("group"), msg.get("name")
        key = self.keys.get(role) if isinstance(role, str) else None
        try:
            real = bool(key) and signing.verify(msg, key)
        except (TypeError, AttributeError):            # a signature that isn't even text
            real = False
        if not real:
            return no("The signature doesn't match a person on this node.")
        if msg.get("node") != self.node:
            return no("This message is for another node.")
        seq = msg.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq <= self.seq.get(role, 0):
            return no("This message was already used.")
        v = self.policy.check(Who(role, "signed", True), verb, group)
        if not v["allow"]:
            return no(v["why"])
        value = None
        if verb == "set_limit":
            rng = LIMITS.get(group, {}).get(name)
            value = msg.get("value")
            if rng is None:
                return no("That limit can't be set here.")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not rng[0] <= value <= rng[1]:
                return no(f"{name} must be a number from {rng[0]:g} to {rng[1]:g}.")
        self.seq[role] = seq
        if verb in ("close_group", "need_now"):
            self.policy.closed[group] = role
        elif verb == "open_group":
            self.policy.closed.pop(group, None)
        try:
            out = self.apply(verb, group, name or "", value)
        except Exception as exc:                       # the device couldn't take it: say so, and the record shows it
            return no(f"That didn't go through ({type(exc).__name__}).")
        return {"status": out.get("status", "done"), "why": out.get("why", ""), "verb": verb, "group": group}


def signed(key: Any, role: str, node: str, seq: int, verb: str, group: str, name: str = "", value: float | None = None) -> dict[str, Any]:
    """A person's message, signed on their own device. Here for the owner's app and for tests."""
    return signing.sign({"role": role, "node": node, "seq": seq, "verb": verb, "group": group, "name": name, "value": value}, key)
