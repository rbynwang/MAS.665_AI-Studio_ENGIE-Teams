"""Step 9: after the event the agent explains what happened, and the node sends up a labelled answer.

Code gathers the facts and marks where each came from. The model only words them, so it can't invent a number. Each fact keeps
its label: measured (a meter or device reading), records (the contract, the calls, the guard's log) or human (the owner's note).
write() makes the report, and send() puts it on the wire to the coordinator, signed: the facts with their labels, the summary and who wrote it.
"""
from __future__ import annotations

from typing import Any

SYSTEM = ("You write a short report for the owner or manager of a home or building after a power sale. Use only the facts given, and copy each number as it is. "
          "Say what was owed, what was done and whether anything went wrong. Two short sentences, in plain words.")
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
    since = add_day(sale["start"], -1)
    for m in getattr(agent, "told", ()):              # every node an agent: what the owner or manager was told, when the node gave less than the rule's amount
        if since <= m["ts"] <= ts:
            out.append({"label": "records", "text": f'Told the {m["to"]}: {m["text"]}'})
    return out


def add_day(ts: str, days: int) -> str:
    from datetime import datetime, timedelta
    return (datetime.strptime(ts[:16], "%Y-%m-%d %H:%M") + timedelta(days=days)).strftime("%Y-%m-%d %H:%M")


def write(agent: Any, sale: dict[str, Any], ts: str) -> dict[str, Any]:
    fs = facts(agent, sale, ts)
    a = agent.brain.ask(SYSTEM, [TOOL], "Facts:\n" + "\n".join(f'- {x["text"]}' for x in fs))      # one prompt for both, or a building's report queues as an exception
    said = next((c["args"].get("summary") for c in a["calls"] if c["name"] == "file_report" and isinstance(c.get("args"), dict)), None)
    text = (said if isinstance(said, str) else "") or a["text"]       # junk in the tool's place isn't a report, and the sale's report is written once
    by = "model" if text else "code"                         # the model's plain text counts as the model's: 21 of 172 reports came back with no tool call and the record called them code
    if not text:
        text = " ".join(x["text"] for x in fs[:2])           # no answer from the model: the facts themselves are the report
    return {"node": agent.name, "ts": ts, "summary": str(text)[:400], "facts": fs, "ms": a["ms"], "by": by}


def send(agent: Any, rep: dict[str, Any], sale: dict[str, Any]) -> dict[str, Any]:
    """The labelled answer goes up. A node with no market side, or a message that can't go, keeps its report: the sale is over and nothing waits on this."""
    desk = agent.ctx.desk
    up = {"sale": sale["start"], "ts": rep["ts"], "by": rep["by"], "summary": rep["summary"], "facts": rep["facts"]}
    try:
        out = desk.report(up) if hasattr(desk, "report") else {"sent": False}
    except Exception as exc:
        out = {"sent": False, "why": type(exc).__name__}
    rep["sent"] = out.get("sent") is not False      # a transport may answer with nothing: only a refusal means it didn't go
    agent.ctx.log.add("report_sent", ts=rep["ts"], sale=sale["start"], sent=rep["sent"], by=rep["by"], facts=len(rep["facts"]))
    return rep
