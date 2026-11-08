"""FlexGrid's agent tools as an MCP server (stdio), for OpenClaw (or any MCP harness).

The tools themselves live in the agent's own process and run there with their arguments checked: a Jetson home's
daemon (GET /agent/tools, POST /agent/tool/{name} on 127.0.0.1:7870) or the coordinator on the Mac
(.../agent/coordinator/tools). This bridge only lists them and forwards each call, so OpenClaw's agent sees exactly
the tools the home's code allows (home_now, next_event, earnings, forecast, decisions, wishes, set_wishes), never a
device, a key or NANDA Town.

  python -m flexgrid_core.agent_mcp                                   # this Jetson's home agent (127.0.0.1:7870/agent)
  python -m flexgrid_core.agent_mcp --base http://127.0.0.1:7871/agent/coordinator --name flexgrid-coordinator

OpenClaw: openclaw mcp set flexgrid-home '{"command": "<repo>/.venv/bin/python", "args": ["-m",
"flexgrid_core.agent_mcp"], "cwd": "<repo>"}' (openclaw/setup_openclaw.sh writes it). Dependency-free on purpose
(newline-delimited JSON-RPC 2.0 on stdin/stdout, MCP's stdio transport): the Jetsons need no MCP SDK.

serve_context() is the same transport for a node's Context MCP (flexgrid_core/node/context.py), which runs in the node's own process.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Callable, Iterable, Iterator

import httpx

VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")


def _out(msg: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(msg, separators=(",", ":"), default=str) + "\n")
    sys.stdout.flush()


def _text(val: Any, is_error: bool) -> dict[str, Any]:
    return {"content": [{"type": "text", "text": val if isinstance(val, str) else json.dumps(val, default=str)}], "isError": is_error}


def reply(msg: dict[str, Any], tools: Callable[[], list[dict[str, Any]]], call: Callable[[str, Any], dict[str, Any]], name: str,
          instructions: str, changes: bool = False) -> dict[str, Any] | None:
    """The answer to one JSON-RPC message, or None for a notification. tools() lists, call(name, args) runs one tool."""
    mid, method, params = msg.get("id"), msg.get("method"), msg.get("params") or {}
    if mid is None:                                   # a notification (initialized, cancelled ...): no answer
        return None
    try:
        if method == "initialize":
            v = params.get("protocolVersion")
            res: Any = {"protocolVersion": v if v in VERSIONS else VERSIONS[1],
                        "capabilities": {"tools": {"listChanged": changes}},
                        "serverInfo": {"name": name, "version": "1.0"}, "instructions": instructions}
        elif method == "ping":
            res = {}
        elif method == "tools/list":
            res = {"tools": [{"name": t["name"], "description": t.get("description", ""),
                              "inputSchema": t.get("parameters") or {"type": "object", "properties": {}}} for t in tools()]}
        elif method == "tools/call":
            res = call(str(params.get("name")), params.get("arguments") or {})
        elif method in ("resources/list", "prompts/list"):
            res = {method.split("/")[0]: []}
        else:
            return {"jsonrpc": "2.0", "id": mid, "error": {"code": -32601, "message": f"no method {method}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": res}
    except Exception as exc:                          # the agent's process is down: say so, keep serving
        return {"jsonrpc": "2.0", "id": mid, "result": _text(f"the FlexGrid agent is not reachable: {type(exc).__name__}: {str(exc)[:200]}", True)}


def _lines(src: Iterable[str]) -> Iterator[dict[str, Any]]:
    for line in src:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            _out({"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}})
            continue
        if isinstance(msg, dict):
            yield msg
        else:
            _out({"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "not a request"}})


def serve(base: str, name: str, timeout: float = 30.0) -> None:
    http = httpx.Client(timeout=timeout)
    instructions = ("FlexGrid's tools for this agent. Every number you tell the person must come from a tool. "
                    "The tools read and change only what the agent's own code allows; safety limits always apply.")

    def tools() -> list[dict[str, Any]]:
        r = http.get(base + "/tools")
        r.raise_for_status()
        return r.json()

    def call(tool: str, args: Any) -> dict[str, Any]:
        r = http.post(f"{base}/tool/{tool}", json=args)
        if r.status_code >= 400:
            return _text(f"error {r.status_code}: {r.text[:300]}", True)
        val = r.json().get("result")
        res = _text(val, isinstance(val, dict) and "error" in val and len(val) == 1)
        if isinstance(val, dict):
            res["structuredContent"] = val
        return res

    for msg in _lines(sys.stdin):
        out = reply(msg, tools, call, name, instructions)
        if out is not None:
            _out(out)


def serve_context(ctx: Any, name: str | None = None, src: Iterable[str] | None = None) -> None:
    """A node's Context MCP (node/context.py), in the node's own process. tools/list shows only the step the workflow is in, and says so
    when the step moves on. A tool answers with its short view: the rest of what it knows stays on the node."""
    instructions = ("This node's tools for the step it is in. Every number you give must come from a tool. "
                    "Each call goes through the node's policy and its guard, and a blocked call says why.")

    def call(tool: str, args: Any) -> dict[str, Any]:
        r = ctx.mcp_call(tool, args)
        return _text(r["view"], r["status"] not in ("done", "adjusted"))

    seen = ctx.step
    for msg in _lines(sys.stdin if src is None else src):
        out = reply(msg, ctx.mcp_tools, call, name or f"flexgrid-{ctx.node.name}", instructions, changes=True)
        if out is not None:
            _out(out)
        if ctx.step != seen:
            seen = ctx.step
            _out({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})


def main() -> None:
    ap = argparse.ArgumentParser(prog="flexgrid-agent-mcp")
    ap.add_argument("--base", default=os.environ.get("FLEXGRID_AGENT_URL", "http://127.0.0.1:7870/agent"))
    ap.add_argument("--name", default="flexgrid-home")
    a = ap.parse_args()
    serve(a.base.rstrip("/"), a.name)


if __name__ == "__main__":
    main()
