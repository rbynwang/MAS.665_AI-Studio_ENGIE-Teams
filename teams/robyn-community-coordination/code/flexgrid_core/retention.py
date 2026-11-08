"""Keep Langflow's run history bounded.

Langflow stores every flow run: a trace with one span per component, the chat messages, the transactions and the
vertex builds. It caps transactions and vertex builds itself (LANGFLOW_MAX_TRANSACTIONS_TO_KEEP and
LANGFLOW_MAX_VERTEX_BUILDS_TO_KEEP, set to 1000 in deploy/), but traces and messages grow without limit: on
jetson-a one day of H1 runs every replay step left 37 MB of spans. This prunes, through Langflow's own API:
  * traces (and their spans) that started more than `keep_hours` ago
  * Playground sessions whose wall-clock hour (in the session name, "<flow> YYYY-MM-DD HHh ...") is older
Called hourly by the home daemon and the coordinator. Freed pages are reused by SQLite, so the file stops growing.
"""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone
from typing import Any

import httpx

SESSION_HOUR = re.compile(r" (\d{4}-\d{2}-\d{2}) (\d{2})h")


def prune_langflow(base_url: str, api_key: str, keep_hours: float = 48.0, max_deletes: int = 2000,
                   max_pages: int = 20, pause_s: float = 0.2) -> dict[str, Any]:
    """Bounded on every path: at most `max_pages` pages and `max_deletes` deletes per call, a short pause between pages,
    and it stops at the first page where nothing could be deleted (a trace Langflow refuses to delete would otherwise
    stay on page 1 and keep the loop spinning)."""
    if not api_key:
        return {"skipped": "no Langflow API key"}
    base = base_url.rstrip("/")
    h = {"x-api-key": api_key}
    cutoff = time.time() - keep_hours * 3600
    cutoff_iso = datetime.fromtimestamp(cutoff, timezone.utc).isoformat()
    out = {"traces_deleted": 0, "sessions_deleted": 0, "trace_delete_errors": 0}
    with httpx.Client(base_url=base, headers=h, timeout=60) as http:
        for _ in range(max_pages):                           # oldest first, one page at a time
            if out["traces_deleted"] >= max_deletes:
                break
            r = http.get("/api/v1/monitor/traces", params={"end_time": cutoff_iso, "size": 200, "page": 1})
            r.raise_for_status()
            traces = r.json().get("traces", [])
            if not traces:
                break
            deleted = 0
            for t in traces:
                if http.delete(f"/api/v1/monitor/traces/{t['id']}").status_code in (204, 404):
                    deleted += 1
                else:
                    out["trace_delete_errors"] += 1
            out["traces_deleted"] += deleted
            if deleted == 0:                                 # no progress: stop instead of spinning on the same page
                out["stopped"] = "traces on the first page could not be deleted"
                break
            time.sleep(pause_s)
        r = http.get("/api/v1/monitor/messages/sessions")
        r.raise_for_status()
        old = []
        for sid in r.json():
            m = SESSION_HOUR.search(sid or "")
            if m:
                t = time.mktime(time.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H"))
                if t + 3600 < cutoff:
                    old.append(sid)
        for i in range(0, len(old), 500):
            r = http.request("DELETE", "/api/v1/monitor/messages/sessions", json=old[i:i + 500])
            if r.status_code < 300:
                out["sessions_deleted"] += len(old[i:i + 500])
    return out
