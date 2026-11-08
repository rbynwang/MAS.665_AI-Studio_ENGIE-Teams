"""Wire log: one JSON line per Beckn message an agent sends or receives over NANDA Town.

Off unless the agent's main() calls open_log(path) (the coordinator writes logs/town_wire.jsonl, the utility
logs/town_wire_utility.jsonl). tools/live_talk.py prints these lines live, so a run can be watched message by message.
Each line: wall time, direction, sender, receiver, Beckn action, whether the Ed25519 signature checked out, the
dataset time, the message id, the first characters of the signer key and of the signature, and the message itself.
Writing never raises into the agent.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Any

_path: str | None = None
_lock = threading.Lock()


def open_log(path: str) -> None:
    global _path
    _path = path


def note(direction: str, frm: str, to: str, action: str, body: dict[str, Any], ok: bool = True,
         sim: str | None = None) -> None:
    if not _path:
        return
    try:
        ctx = body.get("context") or {}
        rec = {"t": round(time.time(), 3), "dir": direction, "from": frm, "to": to, "action": action, "ok": ok,
               "sim": sim, "mid": ctx.get("messageId"), "signer": str(body.get("signer") or "")[:16],
               "sig": str(body.get("sig") or "")[:16], "sent": ctx.get("sentWall"), "message": body.get("message")}
        line = json.dumps(rec, separators=(",", ":"), default=str) + "\n"
        with _lock, open(_path, "a") as f:
            f.write(line)
    except Exception:
        pass


def tail(path, after: int = -1, max_bytes: int = 200_000, max_lines: int = 400) -> dict[str, Any]:
    """New lines of a log file from byte `after` (-1: the last ~64 kB), for the Control Room's terminals.
    Returns {"next": byte offset to ask from next time, "lines": [...]}; a file that was started again reads from 0."""
    import os
    try:
        size = os.path.getsize(path)
    except OSError:
        return {"next": 0, "lines": [], "missing": True}
    start = max(0, size - 64_000) if after < 0 else (0 if after > size else after)
    with open(path, "rb") as f:
        f.seek(start)
        data = f.read(min(max_bytes, size - start))
    cut = data.rfind(b"\n")
    if cut < 0:
        return {"next": start, "lines": []}
    raw = data[:cut].split(b"\n")
    if after < 0:
        if start > 0 and raw:
            raw = raw[1:]                                 # the first line may be cut
        return {"next": start + cut + 1, "lines": [b.decode("utf-8", "replace") for b in raw[-max_lines:]]}
    # reading on from an offset (a run read back from its start): the first max_lines, and on from the end of the
    # last one given, so no line is skipped however far behind the reader is
    raw = raw[:max_lines]
    return {"next": start + sum(len(b) + 1 for b in raw), "lines": [b.decode("utf-8", "replace") for b in raw]}
