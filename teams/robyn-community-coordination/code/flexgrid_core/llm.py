"""The agents' brain: a local LLM that proposes; the agent's own code decides.

FlexGrid's one rule for the LLM: it proposes and the deterministic code decides. Safety limits, signing and NANDA Town
messaging stay in the agents' own code (proved live); the LLM never sees a signing key, never drives a device, and
never reads another party's free text (the facts it gets are structured fields). If it does not answer in time, or its
answer cannot be used, the agent takes today's rule-based decision, so the market cannot stall.

Served by Ollama on the Mac (mac/setup_llm.sh): 127.0.0.1:11434 there, 127.0.0.1:11500 on each Jetson (a tunnel that
mac/start_mac.sh opens from the Mac). One model (Qwen3.8-27B) answers the three agents in parallel: the two Jetson homes
and the coordinator.

  decide(system, facts, schema)       one answer in a fixed JSON shape (the server constrains decoding to the schema),
                                      no tools: market decisions, which have deadlines. The caller checks every field
  converse(system, messages, tools)   a conversation that may call the agent's tools (the phone app's Agent tab): at
                                      most `max_steps` rounds of tool calls, each tool run by the agent's own code
"""
from __future__ import annotations

import json
import re
import threading
import time
from typing import Any, Callable

import httpx

# the 3-bit build under the short name mac/setup_llm.sh gives it (the team's choice), the same by its full name, 4-bit
PREFERRED = ("qwen3.8-27b-q3:latest", "qwen3.8-27b-q3", "hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q3_K_XL", "qwen3.8:27b")
NUM_CTX = 32768        # the same context as the server's default and OpenClaw's num_ctx: Ollama never reloads the model
_THINK = re.compile(r"<think>.*?</think>\s*", re.S)
_TOOLCALL = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


class LLM:
    def __init__(self, url: str, model: str | None = None, *, decide_options: dict[str, Any] | None = None,
                 chat_options: dict[str, Any] | None = None, retry_down_s: float = 20.0, label: str = "llm"):
        self.url = url.rstrip("/")
        self.want = model
        self.model: str | None = None
        self.label = label
        # Qwen3.8 without thinking: temperature 0.7, top_p 0.8, top_k 20 (the model card); decisions a little cooler
        from .node.arm import model_options          # the run's temperature and seed, if its config sets them
        self.decide_options = model_options({"temperature": 0.3, "top_p": 0.8, "top_k": 20, "num_predict": 400, "num_ctx": NUM_CTX,
                                             **(decide_options or {})})
        self.chat_options = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5, "num_predict": 600,
                             "num_ctx": NUM_CTX, **(chat_options or {})}
        self.retry_down_s = retry_down_s
        self._down_until = 0.0
        self._err: str | None = None
        self._lock = threading.Lock()
        self._think_ok = True                       # the server accepts "think": false for this model
        self.http = httpx.Client(timeout=10)
        self.calls = 0
        self.ms: list[float] = []

    # ------------------------------------------------------------------ the server
    def status(self, force: bool = False) -> dict[str, Any]:
        """Is the server up, and which model answers. Checked at most every `retry_down_s` while it is down, so an
        agent whose brain is away decides by its rules at once instead of waiting for a timeout."""
        with self._lock:
            if not force and self.model and time.time() >= self._down_until:
                return self._view(True)
            if not force and time.time() < self._down_until:
                return self._view(False)
        try:
            r = self.http.get(self.url + "/api/tags", timeout=3)
            r.raise_for_status()
            names = [m.get("name") or m.get("model") for m in r.json().get("models", [])]
            pick = next((m for m in ([self.want] if self.want else []) + list(PREFERRED) if m in names), None) or \
                (names[0] if names else None)
            with self._lock:
                self.model, self._err = pick, (None if pick else "no model pulled on the server")
                self._down_until = 0.0 if pick else time.time() + self.retry_down_s
            return self._view(bool(pick))
        except Exception as exc:
            with self._lock:
                self._err, self._down_until = f"{type(exc).__name__}: {str(exc)[:120]}", time.time() + self.retry_down_s
            return self._view(False)

    def _view(self, ok: bool) -> dict[str, Any]:
        p50 = sorted(self.ms)[len(self.ms) // 2] if self.ms else None
        return {"ok": ok, "url": self.url, "model": self.model if ok else None, "error": None if ok else self._err,
                "calls": self.calls, "median_ms": round(p50) if p50 is not None else None}

    def _mark_down(self, err: str) -> None:
        with self._lock:
            self._err, self._down_until = err, time.time() + self.retry_down_s

    def _chat(self, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        # truncate/shift false: what OpenClaw's Ollama provider sends. Ollama starts the model with or without context
        # shifting by these, so a different value from the OpenClaw agents on the same server reloads the model each
        # time the callers alternate (seen live, 31 Oct: 2 s each and the prompt cache lost). Our prompts are far
        # below the 32K context, so neither is ever needed.
        body = {"model": self.model, "stream": False, "keep_alive": "24h", "truncate": False, "shift": False, **payload}
        if self._think_ok:
            body["think"] = False                  # market decisions have deadlines: no long reasoning
        t0 = time.perf_counter()
        r = self.http.post(self.url + "/api/chat", json=body, timeout=max(1.0, timeout))
        if r.status_code == 400 and "think" in r.text.lower() and self._think_ok:
            self._think_ok = False                 # a build without the switch: ask again without it
            body.pop("think", None)
            r = self.http.post(self.url + "/api/chat", json=body, timeout=max(1.0, timeout - (time.perf_counter() - t0)))
        r.raise_for_status()
        ms = (time.perf_counter() - t0) * 1000
        self.calls += 1
        self.ms = (self.ms + [ms])[-200:]
        j = r.json()
        j["_ms"] = ms
        return j

    # ------------------------------------------------------------------ one decision in a fixed shape
    def decide(self, system: str, facts: dict[str, Any], schema: dict[str, Any], *, budget_s: float = 40.0,
               ask: str | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        """(answer, meta). answer is None on any failure (meta['error'] says why): the caller falls back to its rules.
        The answer is only parsed here; the caller checks every value before using it."""
        meta: dict[str, Any] = {"model": None, "ms": None, "error": None}
        if ask is None:
            from .brain import _answer_shape
            ask = "Decide now. " + _answer_shape(schema)
        st = self.status()
        if not st["ok"]:
            meta["error"] = f"no LLM: {st['error']}"
            return None, meta
        meta["model"] = self.model
        t_end = time.time() + budget_s
        msgs = [{"role": "system", "content": system},
                {"role": "user", "content": "Facts (JSON):\n" + json.dumps(facts, separators=(",", ":")) + "\n\n" + ask}]
        ms = 0.0
        for attempt in (1, 2):              # once more (if time is left) when the answer was cut off or missed a key
            try:
                j = self._chat({"messages": msgs, "format": schema, "options": self.decide_options},
                               t_end - time.time())
                ms += j["_ms"]
                meta["ms"] = round(ms)
                text = _THINK.sub("", (j.get("message") or {}).get("content") or "").strip()
                meta["raw"] = text[:600]
                cut = "{" in text and "}" not in text[text.find("{"):]
                ans = json.loads(text[text.find("{"): text.rfind("}") + 1]) if "{" in text and not cut else None
                if not isinstance(ans, dict):
                    meta["error"] = ("unusable answer: cut off before the JSON object closed" if cut
                                     else "the answer was not a JSON object")
                else:
                    missing = [k for k in schema.get("required", []) if k not in ans]
                    if not missing:
                        meta["error"] = None
                        return ans, meta
                    meta["error"] = f"missing {missing}"
            except httpx.TimeoutException:
                meta["error"] = f"no answer within {budget_s:.0f} s"
                return None, meta
            except (httpx.HTTPError, OSError) as exc:
                meta["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
                self._mark_down(meta["error"])
                return None, meta
            except (ValueError, KeyError) as exc:
                meta["error"] = f"unusable answer: {str(exc)[:160]}"
            if attempt == 1 and t_end - time.time() > 8:
                meta["retried"] = True
                msgs = msgs + [{"role": "user", "content": "Your last answer could not be used (" + meta["error"] +
                                "). Answer again with exactly that one JSON object, every key, and a short reason."}]
                continue
            break
        return None, meta

    # ------------------------------------------------------------------ a conversation with tools
    def converse(self, system: str, messages: list[dict[str, Any]],
                 tools: dict[str, tuple[dict[str, Any], Callable[..., Any]]], *, max_steps: int = 4,
                 budget_s: float = 120.0) -> dict[str, Any]:
        """{reply, trace, model, ms, error}. tools: name -> (JSON-schema parameters, function). Each tool call is run
        by the agent's code with the arguments checked there; the result goes back to the model as JSON."""
        out: dict[str, Any] = {"reply": None, "trace": [], "model": None, "ms": 0, "error": None}
        st = self.status()
        if not st["ok"]:
            out["error"] = f"no LLM: {st['error']}"
            return out
        out["model"] = self.model
        specs = [{"type": "function", "function": {"name": n, "description": spec.get("description", ""),
                                                     "parameters": spec.get("parameters", {"type": "object", "properties": {}})}}
                 for n, (spec, _) in tools.items()]
        msgs = [{"role": "system", "content": system}] + list(messages)
        t_end = time.time() + budget_s
        try:
            for step in range(max_steps + 1):
                left = t_end - time.time()
                if left <= 1:
                    out["error"] = f"no answer within {budget_s:.0f} s"
                    return out
                j = self._chat({"messages": msgs, "tools": specs if step < max_steps else [],
                                "options": self.chat_options}, left)
                out["ms"] += round(j["_ms"])
                m = j.get("message") or {}
                content = _THINK.sub("", m.get("content") or "").strip()
                calls = list(m.get("tool_calls") or [])
                if not calls and "<tool_call>" in content:          # a build whose template does not parse them
                    for raw in _TOOLCALL.findall(content):
                        try:
                            x = json.loads(raw)
                            calls.append({"function": {"name": x.get("name"), "arguments": x.get("arguments") or {}}})
                        except ValueError:
                            pass
                    content = _TOOLCALL.sub("", content).strip()
                if not calls:
                    out["reply"] = content or "(no answer)"
                    return out
                msgs.append({"role": "assistant", "content": content, "tool_calls": calls})
                for c in calls:
                    fn = (c.get("function") or {})
                    name, args = fn.get("name"), fn.get("arguments") or {}
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except ValueError:
                            args = {}
                    t0 = time.perf_counter()
                    if name in tools:
                        try:
                            res = tools[name][1](**(args if isinstance(args, dict) else {}))
                        except TypeError as exc:
                            res = {"error": f"bad arguments: {str(exc)[:120]}"}
                        except Exception as exc:                     # a tool's own failure goes back to the model
                            res = {"error": f"{type(exc).__name__}: {str(exc)[:160]}"}
                    else:
                        res = {"error": f"no tool called {name!r}"}
                    out["trace"].append({"tool": name, "args": args, "result": res,
                                         "ms": round((time.perf_counter() - t0) * 1000, 1)})
                    msgs.append({"role": "tool", "tool_name": name,
                                 "content": json.dumps(res, default=str, separators=(",", ":"))[:6000]})
            out["error"] = "too many tool calls"
        except httpx.TimeoutException:
            out["error"] = f"no answer within {budget_s:.0f} s"
        except (httpx.HTTPError, OSError) as exc:
            out["error"] = f"{type(exc).__name__}: {str(exc)[:160]}"
            self._mark_down(out["error"])
        return out
