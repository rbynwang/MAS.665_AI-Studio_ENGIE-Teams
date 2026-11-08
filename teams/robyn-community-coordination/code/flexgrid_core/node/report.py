"""Step 9: after the event the agent explains what happened, and the node sends up a labelled answer.

Code gathers the facts and marks where each came from. The model only words them, so it can't invent a number. Each fact keeps
its label: measured (a meter or device reading), records (the contract, the calls, the guard's log) or human (the owner's note).
"""
from __future__ import annotations

from typing import Any

SYSTEM = ("You write a short report for a home's owner after a power sale. Use only the facts given, and copy each number as it is. "
          "Say what was owed, what the home did and whether anything went wrong. Two short sentences, in plain words.")
TOOL = {"type": "function", "function": {
    "name": "file_report", "description": "Files the report.",
    "parameters": {"type": "object", "required": ["summary"], "properties": {"summary": {"type": "string", "description": "two short sentences"}}}}}


def facts(agent: Any, sale: dict[str, Any], ts: str) -> list[dict[str, str]]:
    """What the node knows at the end of the sale, each fact with its source."""
    tel = agent.sim.tel or {}
    word = getattr(agent, "word", "home")
    out = [{"label": "records", "text": f'The sale ran {sale["start"][11:16]} to {sale["end"][11:16]} and the {word} owed {sale["owed_kw"]:.1f} kW.'}]
    for c in agent.log_calls:
        out.append({"label": "records", "text": f'{c["target"]}: {c["status"]}, {c["expected_kw"]:.2f} kW expected.' + (f' {c["why"]}' if c["why"] else "")})
    last = agent.sim.ctl.last or {}
    short = sum((last.get("short_by_ts") or {}).values()) * 0.25
    held = f'Battery at {tel.get("soc_pct", 0):.0f}% at {ts[11:16]}. ' if agent.sim.E > 0 and "set_battery" in agent.shelf else ""
    out.append({"label": "measured", "text": f"{held}The last plan fell {short:.1f} kWh short."})
    if hasattr(agent, "delivered"):                          # the last plan has nothing left to say about steps already gone
        out.append({"label": "measured", "text": agent.delivered(sale)})
    for w in agent.guard.wishes:
        out.append({"label": "human", "text": f'The owner asked: {w["wish"].replace("_", " ")}.'})
    return out


def write(agent: Any, sale: dict[str, Any], ts: str) -> dict[str, Any]:
    fs = facts(agent, sale, ts)
    system = SYSTEM.replace("a home's owner", "a building's manager").replace("the home", "the building") if getattr(agent, "word", "home") == "building" else SYSTEM
    a = agent.brain.ask(system, [TOOL], "Facts:\n" + "\n".join(f'- {x["text"]}' for x in fs))
    text = next((c["args"].get("summary") for c in a["calls"] if c["name"] == "file_report"), None) or a["text"]
    if not text:
        text = " ".join(x["text"] for x in fs[:2])           # no answer from the model: the facts themselves are the report
    return {"node": agent.name, "ts": ts, "summary": str(text)[:400], "facts": fs, "ms": a["ms"], "by": "model" if a["calls"] else "code"}
