"""The device tools a node's agent can call. The full cards live in the device tools project: only what Qwen reads is here.

Each node sees only the tools for what it owns (shelves.json), and code fills start and end, so the model never has to.
`tools/build_bundle.py` in device_tools writes the three json files beside this module.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
PROMPT_VERSION = "2026-11-07"
SYSTEM = ("You are the energy agent for one node. You sell power cuts to the grid. Each step tells you the situation and gives "
          "you tools. First read each tool's Not when line. If a Not when line fits the situation, or nothing needs doing, call "
          "no_action. Otherwise call the one tool that fits, with the values the situation supports. Use only what the situation "
          "tells you.")
PLAN_SYSTEM = ("You are the energy agent for one node in a power sale, and something just changed. You see the whole situation: what is owed, what runs, what can "
               "open, what each is worth and what it costs. Decide the whole plan in one call to set_plan, by letter. Code sizes every number and holds every limit, so you "
               "weigh money against what people feel and nothing else. Keeping every running call is the normal answer. Leaving kW uncovered, or ending a call, needs a fact from the "
               "list, named in because: anything else is ignored. You may end one call in a plan, the one that fact is about. If you need a reading or a test first, call that tool, then "
               "decide. Use only what the situation tells you.")
TYPES = {"string": "string", "number": "number", "enum": "string", "time": "string"}


@lru_cache(maxsize=1)
def bundle() -> dict[str, Any]:
    return json.loads((HERE / "tools.json").read_text())


@lru_cache(maxsize=1)
def shelves() -> dict[str, Any]:
    return json.loads((HERE / "shelves.json").read_text())


@lru_cache(maxsize=1)
def home_setup() -> dict[str, Any]:
    return json.loads((HERE / "home_setup.json").read_text())


def home_id(name: str) -> str:
    """hh-010 in the fleet is HH010 in the records."""
    return name.replace("hh-", "HH").upper()


def shelf(name: str, tier: int) -> list[str]:
    """The tools this home holds at this tier."""
    return sorted((shelves()["homes"].get(home_id(name)) or {}).get(f"tier{tier}", []))


def building_shelf(key: str) -> list[str]:
    return list(shelves()["buildings"].get(key, []))


def spec(side: str, tool: str, targets: list[str] | None = None, only: dict[str, list[str]] | None = None) -> dict[str, Any]:
    """The tool as Qwen sees it: its description and example, with start and end left to code. only: the values that fit now."""
    c = bundle()[side].get(tool) or bundle()["common"][tool]
    mv = c["model_view"]
    props: dict[str, Any] = {}
    req: list[str] = []
    for p in mv["parameters"]:
        if p["name"] in ("start", "end"):
            continue
        d: dict[str, Any] = {"type": TYPES.get(p["type"], "string"), "description": p.get("says", "")}
        if p["type"] == "enum":
            vals = [x.strip() for x in str(p["allowed"]).split("|") if x.strip()]
            d["enum"] = list(targets) if (p["name"] == "target" and targets) else (only or {}).get(p["name"], vals)
        elif p["name"] == "target" and targets:
            d["enum"] = list(targets)
        props[p["name"]] = d
        req.append(p["name"])
    desc = mv["description"]
    if mv.get("example"):
        desc += " Example: " + re.sub(r', (start|end)="[^"]*"', "", mv["example"])
    return {"type": "function", "function": {"name": tool, "description": desc,
                                             "parameters": {"type": "object", "properties": props, "required": req}}}
