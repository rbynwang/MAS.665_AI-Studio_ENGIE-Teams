"""The grid buyer's AI agent (the utility, mock ENGIE), on the Mac: it orders tomorrow's flexibility and, after the
event, checks the settlement and signs off. FlexGrid's rule holds here too: the LLM proposes, the code decides.

It uses the buyer's MCP tools (flexgrid_core/mcp_server.py: list_communities, forecast_community_demand,
request_flexibility, get_settlement, verify_settlement, sign_off) and nothing else:
  * through the Mac's OpenClaw gateway (agent "buyer": its workspace openclaw/agents/buyer, the flexgrid-buyer skill,
    the MCP server "flexgrid") when the gateway is up: a normal OpenClaw agent run, one session per event, so at the
    sign-off it remembers why it ordered what it did
  * else the same tools in this process, with the local model's own tool loop (Ollama straight)
  * else nothing: the utility's catalog request stands, and no sign-off is recorded
What the code decides, whatever the agent says: the utility refuses an order outside its limits (kW <= 500, price <=
the limit given here, a window of 30 min to 4 h, before the 10:00 call for offers) and sign_off refuses an approval
unless verify_settlement passed. If the agent never signs off, the rules do (approve only when every check passes).
"""
from __future__ import annotations

import inspect
import json
import threading
import time
from typing import Any, Callable

import httpx

from . import config, mcp_server
from .brain import Decisions, make_backend

ORDER_TOOLS = ("list_communities", "forecast_community_demand", "request_flexibility")
SIGN_TOOLS = ("get_settlement", "verify_settlement", "sign_off")
SYSTEM = (open(config.REPO_DIR / "agent" / "buyer_agent.md").read() if (config.REPO_DIR / "agent" / "buyer_agent.md").exists()
          else "You are the demand-response buyer agent for a utility.")


def _schema(fn: Callable[..., Any]) -> dict[str, Any]:
    """A JSON schema for a tool function from its signature (str, int, float, bool)."""
    kinds = {str: "string", int: "integer", float: "number", bool: "boolean", "str": "string", "int": "integer",
             "float": "number", "bool": "boolean"}
    props, req = {}, []
    for p in inspect.signature(fn).parameters.values():
        props[p.name] = {"type": kinds.get(p.annotation, "string")}
        if p.default is inspect.Parameter.empty:
            req.append(p.name)
    return {"type": "object", "properties": props, "required": req}


class BuyerAgent:
    def __init__(self, cfg: dict[str, Any], log: Callable[..., None], utility_url: str):
        self.cfg, self.log, self.u = cfg, log, utility_url.rstrip("/")
        self.backend = make_backend({**cfg, "openclaw_agent": "buyer"}, "buyer")
        table = mcp_server.ToolTable()
        mcp_server.build(mcp_server.Live(), table)
        self.fns = table.tools
        self.decisions = Decisions()
        self.busy: str | None = None                   # what it is doing now (the Control Room's brake and card)
        self.lock = threading.Lock()
        self.sessions: dict[str, str] = {}             # day -> this run's OpenClaw session (the order, then the sign-off)

    def _session(self, day: str, new: bool = False) -> str:
        """One OpenClaw session per run and event day: at the sign-off the agent remembers its order. Never the same
        key across runs: OpenClaw keeps a session's whole history, and a run of the same day reusing it carried the
        last run's order and sign-off (17K prompt tokens, a minute of reading, live 31 Oct 2026)."""
        if new or day not in self.sessions:
            self.sessions[day] = f"buyer-{day}-{time.strftime('%Y%m%d-%H%M%S')}"
        return self.sessions[day]

    def status(self) -> dict[str, Any]:
        return {"enabled": True, "llm": self.backend.status(), "busy": self.busy, "decisions": self.decisions.list(20)}

    def _tools(self, names: tuple[str, ...]) -> dict[str, tuple[dict[str, Any], Callable[..., Any]]]:
        out = {}
        for n in names:
            fn = self.fns.get(n)
            if fn:
                out[n] = ({"description": (fn.__doc__ or "").strip()[:900], "parameters": _schema(fn)}, fn)
        return out

    def _need(self, day: str) -> dict[str, Any] | None:
        try:
            nd = httpx.get(self.u + "/needs", timeout=10).json()
            return next((x for x in nd.get("needs") or [] if x["window"][0][:10] == day), None)
        except Exception:                                # noqa: BLE001
            return None

    # ------------------------------------------------------------------ before the market: the order
    def order(self, day: str, limit_usd_per_kwh: float, budget_s: float = 45.0, note: str | None = None) -> dict[str, Any]:
        """Ask the agent to order `day`'s flexibility (price at or below the limit). Returns what was ordered (the
        utility's /needs for that day) and the agent's own words; the catalog's request stands if it ordered nothing.
        note: what the weather service says about the day (the Control Room's scenario), told to the agent as such."""
        with self.lock:
            self.busy = f"ordering flexibility for {day}"
        import os
        os.environ["FLEXGRID_BUYER_MAX_PRICE"] = f"{limit_usd_per_kwh:g}"   # request_flexibility refuses a higher price
        t0 = time.time()
        before = self._need(day)
        # what list_communities and forecast_community_demand answer, read by this code beforehand (the same MCP tool
        # functions): the agent's own tool call is the order itself, so the order fits the minute before 10:00
        ctx = {}
        try:
            lc = self.fns["list_communities"]()
            ctx["list_communities"] = {"now": lc.get("now"), "homes": [c.get("homes") for c in lc.get("communities", [])],
                                       "program": lc.get("program"), "needs": [{k: n.get(k) for k in ("event_id", "window", "kw",
                                       "price_usd_per_kwh", "notice", "source")} for n in lc.get("needs", [])
                                       if str(n.get("event_id", "")).endswith(day)]}
            fc = self.fns["forecast_community_demand"](day)
            ctx["forecast_community_demand"] = {
                **{k: fc.get(k) for k in ("issued_at", "homes", "unit", "peak_window", "error", "available") if k in fc},
                "evening_by_hour": [h for h in fc.get("hours") or [] if 14 <= int(h.get("hour", 0)) <= 21]}
        except Exception as exc:                         # noqa: BLE001  the agent can still call the tools itself
            ctx["note"] = f"pre-read failed ({type(exc).__name__}); call list_communities and forecast_community_demand"
        task = (f"It is the morning before {day}. Order the flexibility you need for {day} now, before the 10:00 call "
                f"for offers. What list_communities and forecast_community_demand({day}) answer (JSON):\n"
                f"{json.dumps(ctx, default=str)[:3500]}\n"
                + (f"The weather service, this morning: {note} The forecast above does not include this yet.\n"
                   if note else "") +
                f"Call request_flexibility once: the evening window where demand peaks, kW the community can deliver even "
                f"on a low day, a price at or below ${limit_usd_per_kwh:.2f} per kWh, and your reason in `note`. Do not "
                f"follow the deal yet. Then answer in one or two sentences: what you ordered and why.")
        out = self.backend.converse("buyer", SYSTEM, [{"role": "user", "content": task}], self._tools(ORDER_TOOLS),
                                    budget_s=budget_s, session=self._session(day, new=True))
        after = self._need(day)
        placed = bool(after and after.get("source") == "buyer agent order" and
                      not (before and before.get("source") == "buyer agent order"))
        price_ok = placed and float(after.get("price_usd_per_kwh") or 0) <= limit_usd_per_kwh + 1e-9
        d = self.decisions.add(kind="order", day=day, placed=placed, sim=(after or before or {}).get("notice"),
                               kw=(after or before or {}).get("kw"), price=(after or before or {}).get("price_usd_per_kwh"),
                               window=(after or before or {}).get("window"), price_within_limit=price_ok,
                               reason=(out.get("reply") or "")[:500] if placed else
                               ("the catalog's request stands: " + (out.get("error") or out.get("reply") or "no order")[:300]),
                               note=(after or {}).get("note"), fallback=not placed,
                               llm={"model": out.get("model"), "ms": out.get("ms"), "error": out.get("error"),
                                    "runtime": out.get("runtime")},
                               tools=[t.get("tool") for t in out.get("trace") or []], seconds=round(time.time() - t0, 1))
        self.log("agent_decision", what="order", day=day, placed=placed, kw=d["kw"], price=d["price"],
                 window=d["window"], reason=d["reason"][:240], fallback=not placed,
                 runtime=out.get("runtime"), ms=out.get("ms"))
        with self.lock:
            self.busy = None
        return d

    # ------------------------------------------------------------------ after settlement: verify and sign off
    def sign_off(self, event_id: str, budget_s: float = 120.0) -> dict[str, Any]:
        with self.lock:
            self.busy = f"checking the settlement of {event_id}"
        t0 = time.time()
        day = event_id.replace("flex-need-", "")
        task = (f"The event {event_id} is settled. Call verify_settlement for it, then sign_off: approve only if every "
                f"check passes, otherwise dispute and give the failing checks as the reason. Then answer in one or two "
                f"sentences: what the utility pays, for how much delivered, and your decision.")
        out = self.backend.converse("buyer", SYSTEM, [{"role": "user", "content": task}], self._tools(SIGN_TOOLS),
                                    budget_s=budget_s, session=self._session(day))
        review = self._review(event_id)
        by_rules = review is None
        if by_rules:                                 # the agent did not sign off: the rules do (approve iff all pass)
            try:
                v = self.fns["verify_settlement"](event_id)
                ok = bool(v.get("all_pass"))
                fails = [c["check"] for c in v.get("checks", []) if c.get("result") == "FAIL"]
                self.fns["sign_off"](event_id, ok, "rules: every check passed" if ok else f"rules: failed {fails}")
            except Exception as exc:                 # noqa: BLE001
                self.log("agent_error", where="buyer sign-off", error=str(exc)[:200])
            review = self._review(event_id)
        d = self.decisions.add(kind="sign_off", event_id=event_id, decision=(review or {}).get("decision"),
                               usd=(review or {}).get("settlement_usd"), sim=(review or {}).get("at"),
                               reason=(out.get("reply") or "")[:500] if not by_rules else (review or {}).get("reason"),
                               fallback=by_rules, llm={"model": out.get("model"), "ms": out.get("ms"),
                                                       "error": out.get("error"), "runtime": out.get("runtime")},
                               tools=[t.get("tool") for t in out.get("trace") or []], seconds=round(time.time() - t0, 1))
        self.log("agent_decision", what="sign_off", event_id=event_id, decision=d["decision"], usd=d["usd"],
                 reason=str(d["reason"] or "")[:240], fallback=by_rules, runtime=out.get("runtime"), ms=out.get("ms"))
        with self.lock:
            self.busy = None
        return d

    def _review(self, event_id: str) -> dict[str, Any] | None:
        try:
            r = httpx.get(f"{self.u}/evidence/{event_id}", timeout=10)
            return (r.json() or {}).get("buyer_review") if r.status_code == 200 else None
        except Exception:                            # noqa: BLE001
            return None


def limit_price(cfg: dict[str, Any], catalog_price: float | None) -> float:
    """The most the buyer agent may pay per kWh: the program's price (the catalog's) unless the config says more."""
    lim = cfg.get("buyer_max_price_usd_per_kwh")
    return float(lim) if lim else float(catalog_price or 1.0)


def describe() -> str:
    return json.dumps({"order_tools": ORDER_TOOLS, "sign_tools": SIGN_TOOLS})
