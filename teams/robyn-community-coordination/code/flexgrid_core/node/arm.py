"""Every node an agent (agentic/BRIEF.md, 7 Nov 2026): the switches, read from the node's own config.

  all_agents       true: the model answers at every decision point. The offer and the commitment are asked even when code sees nothing
                   to weigh, a sale's opening is a plan the agent chooses, every exception is asked, and a model that fails leaves that
                   one point to code's plan, recorded as a fallback. An answer that gives less than the rule's amount needs a reason: with
                   none it is asked once more, then the rule's amount goes out ("no reason given"). A reason that cites no fact stands,
                   flagged. Each decision leaves a receipt in the node's log, the coordinator hears only the amount, and the person hears
                   why. False (the default): everything runs as before.
  llm_temperature  the sampling temperature of every node's answers (default 0.3, as before). 0 for runs that must repeat
  llm_seed         a fixed seed for every answer, or none
  llm_short_answers  true: every question ends asking for short reasons (SHORT), so answers come back sooner. A speed-up, off by default:
                   on the Homework 8 normal day the median answer was 116 tokens out, at 18 tokens a second

FG_ALL_AGENTS and FG_SHORT_ANSWERS, when set, win over the config: tests use them.
"""
from __future__ import annotations

import os
from typing import Any


def _cfg() -> dict[str, Any]:
    from .. import config
    try:
        return config.load()
    except Exception:                               # noqa: BLE001  a node with no readable config keeps the old behaviour
        return {}


def _flag(v: Any) -> bool:
    return str(v).strip().lower() not in ("", "0", "false", "no", "none", "off")


def all_agents() -> bool:
    env = os.environ.get("FG_ALL_AGENTS")
    return _flag(env) if env is not None else bool(_cfg().get("all_agents"))


def model_options(base: dict[str, Any]) -> dict[str, Any]:
    """base with the run's temperature and seed, if the config sets them."""
    c = _cfg()
    out = dict(base)
    if c.get("llm_temperature") is not None:
        out["temperature"] = float(c["llm_temperature"])
    if c.get("llm_seed") is not None:
        out["seed"] = int(c["llm_seed"])
    return out


SHORT = " Keep each reason to one plain sentence of at most 20 words, with only the numbers that mattered."


def short_answers() -> bool:
    env = os.environ.get("FG_SHORT_ANSWERS")
    return _flag(env) if env is not None else bool(_cfg().get("llm_short_answers"))


def shorten(system: str) -> str:
    """The question's system prompt, asking for short reasons when the run says so. The decision log keeps it as it was sent."""
    return system + SHORT if short_answers() and not system.endswith(SHORT) else system
