"""flexgridd - the always-on home agent daemon (one per Jetson; a second instance with role "building" runs a building
node, e.g. the library on jetson-a: FLEXGRID_CONFIG_DIR=~/.config/flexgrid-building, port 7875, logs/building.log).

It owns everything that touches the outside world, so a Langflow flow can never skip a check:
  * the device adapter (dataset replay now; Tesla / Enphase later)
  * the signing key and the NANDA Town session
  * the executor: hard limits, expiry, read-back, hash-chained receipts, automatic revert
Langflow flows (H1 envelope loop, H2 dispatch handler) call it over 127.0.0.1 and show each step.
It also drives them: one H1 run per replay step, one H2 run per dispatch that arrives in Town.
If a flow is not configured or fails, the same steps run in-process, so safety never depends on the UI.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
import traceback
from collections import deque
from typing import Any

import httpx
import uvicorn
from fastapi import Body, FastAPI, HTTPException

from . import beckn, config, doors, envelope as envlib, forecast as fc, ids, signing, telemetry as tcheck
from .adapters import make_adapter
from .home_v2 import HomeV2
from . import replay as replay_mod
from .replay import Replay
from .town import InboxPoller, TownSession

log = logging.getLogger("flexgridd")


class Agent:
    def __init__(self):
        self.cfg = config.load()
        c = self.cfg
        self.replay = Replay(c["profile"], c["replay_start"], c["replay_step_seconds"], c["replay_loop_days"])
        self.adapter = make_adapter(c, self.replay)
        self.key = signing.load_or_create(config.CONFIG_DIR / "secrets" / "signing.key")
        self.pub = signing.public_hex(self.key)
        self.history: deque = deque(maxlen=96)
        self.last_hist_idx = None
        self.seq = 0
        self.last_envelope: dict[str, Any] | None = None
        self.events: deque = deque(maxlen=200)
        self.town_in: deque = deque(maxlen=2000)          # (Beckn action, s from the sender making it to this home)
        self.town_send_ms: deque = deque(maxlen=500)      # how long each send to Town took, ms
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(config.state_dir() / "receipts.db"), check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS receipts (n INTEGER PRIMARY KEY AUTOINCREMENT, "
                        "dispatch_id TEXT, hash TEXT, prev TEXT, body TEXT, at REAL)")
        self.db.commit()
        self.town: TownSession | None = None
        tok = config.secret("town_token")
        if c.get("town_run_id") and tok:
            self.town = TownSession(c["town_url"], c["town_run_id"], c["agent_name"], tok)
        self.last_dispatch_wall = 0.0
        # what-if "offline" (the Control Room's What if panel): until then this home neither reads its NANDA Town inbox
        # nor sends; what it would have sent waits here and goes out, in order, when it is back
        self.offline_until = 0.0
        self.outbox: deque = deque(maxlen=500)
        # v2: offers, contracts, delivery and signed readings (same code as the replay fleet on jetson-b)
        self.v2 = HomeV2(c["agent_name"], c, self.adapter, self.replay, self.key, send=self._town_send,
                         trusted=self.trusted_coordinators, log=self.event, coordinator=c["coordinator_name"])
        # the AI agent (brain.py, llm_enabled): thinks with the Mac's local LLM through this Jetson's OpenClaw gateway
        # (Ollama straight as the stand-in), within this home's rules: it can lower the safe offer, decline a commitment
        # the guard found safe, and talk with the homeowner in the phone app. Nothing else changes: the checks,
        # signing, NANDA Town messaging and device control stay in this daemon
        self.brain = None
        if c.get("llm_enabled") and c.get("role", "home") in ("home", "building"):
            from .brain import BuildingBrain, FacilityPolicy, HomeBrain, Wishes, make_backend
            # a building node (role "building", building.py): the same daemon, a facility manager instead of a
            # homeowner, its own OpenClaw agent ("building") on the same Mac model
            bld = c.get("role") == "building"
            cls, wish = (BuildingBrain, FacilityPolicy) if bld else (HomeBrain, Wishes)
            self.brain = cls(self.v2, make_backend(c, c.get("openclaw_agent") or ("building" if bld else "home")),
                             wish(config.state_dir() / "agent_wishes.json"), label=c.get("site_label", ""),
                             log=self.event, offer_budget_s=float(c.get("llm_offer_budget_s", 40)),
                             accept_budget_s=float(c.get("llm_accept_budget_s", 45)),
                             chat_budget_s=float(c.get("llm_chat_budget_s", 120)))
            self.v2.brain = self.brain

    def offline(self) -> bool:
        return time.time() < self.offline_until

    def _town_send(self, to: str, kind: str, body: dict[str, Any], message_id: str) -> dict[str, Any]:
        if not self.town:
            raise RuntimeError("no NANDA Town session")
        if self.offline():
            self.outbox.append((to, kind, body, message_id))
            log.info("offline (what-if): %s to %s held until the home is back", kind, to)
            return {"held_offline": True}
        t0 = time.time()
        out = self.town.send(to, kind, body, message_id)
        if isinstance(out, dict) and out.get("duplicate"):    # Town refused it: see town.send
            self.event("send_dropped", to=to, message_kind=kind, message_id=message_id)
        self.town_send_ms.append((time.time() - t0) * 1000)
        log.info("town_out %s to %s %.0f ms", kind, to, (time.time() - t0) * 1000)
        return out

    def town_stats(self) -> dict[str, Any]:
        """How late Town messages reach this home (from the sender making the message to this home claiming it:
        sending, waiting in Town, this home's next check) and how long its own sends take."""
        lat = list(self.town_in)
        by = {}
        for a in sorted({k for k, _ in lat}):
            by[a] = _stats([x for k, x in lat if k == a], 2)
        return {"delivery_s": _stats([x for _, x in lat], 2), "delivery_s_by_action": by,
                "send_ms": _stats(list(self.town_send_ms), 0)}

    # ------------------------------------------------------------------ readings
    def read(self) -> dict[str, Any]:
        with self.lock:
            tel = self.adapter.telemetry()
            if tel.get("idx") != self.last_hist_idx:
                self.history.append(tel)
                self.last_hist_idx = tel.get("idx")
            return tel

    def health(self) -> dict[str, Any]:
        return tcheck.check(list(self.history), self.adapter.limits())

    def trusted_coordinators(self) -> list[str]:
        """Public keys of coordinators this home accepts commands from (config: coordinator_pubkeys)."""
        self.cfg = {**self.cfg, "coordinator_pubkeys": config.load().get("coordinator_pubkeys", [])}
        return list(self.cfg.get("coordinator_pubkeys") or [])

    def event(self, kind: str, **detail) -> None:
        self.events.append({"at": time.time(), "kind": kind, **detail})
        log.info("%s %s", kind, json.dumps(detail, default=str)[:300])

    # ------------------------------------------------------------------ H1 steps
    def build_envelope(self, tel=None, health=None) -> dict[str, Any]:
        tel = tel or self.read()
        health = health or self.health()
        fcast = fc.seasonal_naive(self.replay, tel["idx"], n=8)
        with self.lock:
            self.seq += 1
            seq = self.seq
        return envlib.build(tel, self.adapter.limits(), health, fcast, self.cfg["agent_name"],
                            self.cfg["site_label"], seq, self.cfg["brand"])

    def publish(self, env: dict[str, Any]) -> dict[str, Any]:
        env = {**env, "agent": self.cfg["agent_name"], "published_wall": time.time()}
        if not env.get("seq"):
            with self.lock:
                self.seq += 1
                env["seq"] = self.seq
        signed = signing.sign(env, self.key)
        with self.lock:
            self.last_envelope = signed
        sent = None
        if self.town and self.cfg.get("v1_town_reports", False):   # v2 runs events: the v1 report every minute only
            mid = ids.sealed(self.key, self.v2.play, self.cfg["agent_name"], self.cfg["coordinator_name"], "flex.report", signed["seq"], signed.get("slot_start"))   # filled the coordinator's inbox
            try:
                sent = self.town.send(self.cfg["coordinator_name"], "flex.report", signed, mid)
            except Exception as exc:                                 # Town down: keep running locally
                sent = {"error": str(exc)}
        self.event("envelope_published", seq=signed["seq"], up=signed["up_kw"], down=signed["down_kw"],
                   health=signed["health"], town=bool(sent and "error" not in sent))
        return {"envelope": signed, "town": sent}

    # ------------------------------------------------------------------ H2 steps
    def verify(self, d: dict[str, Any]) -> dict[str, Any]:
        """All checks that decide accept / modify / reject. Runs again inside apply()."""
        c, lim = self.cfg, self.adapter.limits()
        env = self.last_envelope
        reasons = []
        if d.get("signer") not in self.trusted_coordinators() or not signing.verify(d, d.get("signer", "")):
            return {"status": "rejected", "reason": "bad_signature"}
        if d.get("to") != c["agent_name"]:
            return {"status": "rejected", "reason": "wrong_recipient"}
        if float(d.get("expires_wall", 0)) < time.time():
            return {"status": "rejected", "reason": "expired"}
        minutes = float(d.get("minutes", 15))
        if minutes <= 0 or minutes > c["max_dispatch_minutes"]:
            return {"status": "rejected", "reason": "duration_limit"}
        if self.health().get("status") in ("frozen", "jump"):
            return {"status": "rejected", "reason": "telemetry_unverified"}
        delta = float(d.get("delta_kw", 0.0))
        hard = lim["battery_kw"] + lim["ev_max_kw"]
        if abs(delta) > hard + 1e-6:
            return {"status": "rejected", "reason": "battery_power_limit",
                    "limit_kw": hard, "requested_kw": delta}
        if env is None:
            return {"status": "rejected", "reason": "no_envelope_published"}
        allowed = env["up_kw"] if delta >= 0 else -env["down_kw"]
        status, apply_kw = "accepted", delta
        if (delta > 0 and delta > env["up_kw"] + 1e-6) or (delta < 0 and -delta > env["down_kw"] + 1e-6):
            status, apply_kw = "modified", allowed
            reasons.append("clamped_to_envelope")
        if lim.get("control") == "advisory":
            status = "advisory"
            reasons.append("homeowner_approval_needed")
        return {"status": status, "reason": ",".join(reasons) or "within_envelope",
                "apply_kw": round(apply_kw, 3), "minutes": minutes, "envelope_seq": env["seq"]}

    def apply(self, d: dict[str, Any], verdict: dict[str, Any] | None = None) -> dict[str, Any]:
        v = self.verify(d)                          # defense in depth: never trust the caller's verdict
        if verdict and verdict.get("status") == "rejected" and v["status"] != "rejected":
            v = verdict                             # a flow may reject more, never accept more
        result = {}
        with self.lock:
            if v["status"] in ("accepted", "modified"):
                result = self.adapter.apply(v["apply_kw"], v["minutes"], d["dispatch_id"])
                self.last_dispatch_wall = time.time()
            tel = self.adapter.telemetry()
            if self.history and self.history[-1].get("idx") == tel.get("idx"):
                self.history[-1] = tel              # keep the health check's history consistent with the read-back
            prev = self.db.execute("SELECT hash FROM receipts ORDER BY n DESC LIMIT 1").fetchone()
            receipt = {"type": "flex.receipt", "dispatch_id": d.get("dispatch_id"), "agent": self.cfg["agent_name"],
                       "status": v["status"], "reason": v.get("reason"), "requested_kw": d.get("delta_kw"),
                       "applied_kw": result.get("applied_kw", 0.0), "battery_target_kw": result.get("battery_target_kw"),
                       "ev_action": result.get("ev_action"), "readback": result.get("readback"),
                       "sim_ts": tel["ts"], "wall_ts": time.time(), "prev_hash": prev[0] if prev else None}
            receipt = signing.sign(receipt, self.key)
            h = signing.digest(receipt)
            self.db.execute("INSERT INTO receipts (dispatch_id, hash, prev, body, at) VALUES (?,?,?,?,?)",
                            (receipt["dispatch_id"], h, receipt["prev_hash"], json.dumps(receipt), time.time()))
            self.db.commit()
        self.event("dispatch_" + v["status"], dispatch_id=d.get("dispatch_id"), reason=v.get("reason"),
                   requested=d.get("delta_kw"), applied=receipt["applied_kw"])
        return {**receipt, "receipt_hash": h}

    def ack(self, d: dict[str, Any], receipt: dict[str, Any], message_id: str | None, fence: str | None) -> dict[str, Any]:
        out = {}
        if self.town:
            try:
                out["sent"] = self.town.send(self.cfg["coordinator_name"], "flex.dispatch_ack", receipt,
                                             f"{self.cfg['agent_name']}:ack:{d.get('dispatch_id')}")
                if message_id and fence:
                    status = "processed" if receipt["status"] in ("accepted", "modified", "advisory") else "rejected"
                    out["inbox"] = self.town.ack(message_id, fence, status, {"receipt_hash": receipt.get("receipt_hash")})
            except Exception as exc:
                out["error"] = str(exc)
        return out

    # ------------------------------------------------------------------ Langflow bridge
    def run_flow(self, name: str, text: str) -> str | None:
        """Run flow H1/H2 in the local Langflow. One Playground session per wall-clock hour keeps runs easy to browse."""
        flow_id = (self.cfg.get("flows") or {}).get(name)
        key = config.secret("langflow_api_key")
        if not flow_id or not key:
            return None
        r = httpx.post(f"{self.cfg['langflow_url'].rstrip('/')}/api/v1/run/{flow_id}",
                       params={"stream": "false"}, headers={"x-api-key": key}, timeout=120,
                       json={"input_value": text, "input_type": "chat", "output_type": "chat",
                             "session_id": f"{name} {time.strftime('%Y-%m-%d %H')}h · replay "
                                           f"{self.replay.ts(self.replay.current())[:10]}"})
        r.raise_for_status()
        try:
            return r.json()["outputs"][0]["outputs"][0]["results"]["message"]["text"]
        except Exception:
            return json.dumps(r.json())[:500]

    def h1(self) -> None:
        try:
            out = self.run_flow("H1", json.dumps({"tick": self.replay.ts(self.replay.current())}))
            if out is None:
                self.publish(self.build_envelope())
                self.event("h1_inprocess")
            else:
                self.event("h1_langflow", output=out[:160])
        except Exception as exc:
            self.event("h1_error", error=str(exc))
            self.publish(self.build_envelope())             # never miss a report because the UI is down

    def h2(self, msg: dict[str, Any]) -> None:
        payload = {"message_id": msg["message_id"], "fence": msg["fence"], "from": msg.get("from"),
                   "dispatch": msg["body"]}
        try:
            out = self.run_flow("H2", json.dumps(payload))
            if out is None:
                rec = self.apply(msg["body"])
                self.ack(msg["body"], rec, msg["message_id"], msg["fence"])
                self.event("h2_inprocess", dispatch_id=msg["body"].get("dispatch_id"))
            else:
                self.event("h2_langflow", output=out[:160])
        except Exception as exc:
            self.event("h2_error", error=str(exc))
            rec = self.apply(msg["body"])
            self.ack(msg["body"], rec, msg["message_id"], msg["fence"])

    # ------------------------------------------------------------------ loops
    def clock_loop(self) -> None:
        last = None
        while True:
            try:
                tel = self.read()
                if tel["idx"] != last:
                    last = tel["idx"]
                    with self.lock:          # the v2 sensor check (offers, commitments) sees every step, as in the fleet
                        self.v2.observe()
                    if self.cfg["role"] == "home":
                        threading.Thread(target=self.h1, daemon=True).start()
                self.h4()
                if self.outbox and not self.offline():
                    self.event("back_online", sending=len(self.outbox))
                    while self.outbox and not self.offline():
                        to, kind, body, mid = self.outbox.popleft()
                        try:
                            self._town_send(to, kind, body, mid)
                        except Exception as exc:
                            self.event("town_error", error=f"after offline: {str(exc)[:160]}")
            except Exception:
                log.error("clock loop: %s", traceback.format_exc())
            time.sleep(1.0)

    # ------------------------------------------------------------------ v2 (Beckn contracts)
    def h_v2(self, msg: dict[str, Any]) -> str:
        """beckn.discover -> flow H3 Offer, beckn.confirm -> flow H4 Accept, the rest in-process. The in-process checks
        run first on every path: a message that is not from the trusted coordinator never reaches a flow, and a
        commitment this home has answered already gets the same answer again without running H4."""
        kind, sender, body = msg.get("kind", ""), msg.get("from"), msg.get("body") or {}
        rejected = self.v2.gate(kind, sender, body)
        if rejected:
            return rejected
        if kind == "beckn.confirm":
            again = self.v2.repeat_answer(body)
            if again is not None:
                return again
        flow = {"beckn.discover": "H3", "beckn.confirm": "H4"}.get(kind)
        if flow:
            try:
                out = self.run_flow(flow, json.dumps({"kind": kind, "from": sender, "body": body}))
                if out is not None:
                    self.event(f"{flow.lower()}_langflow", output=out[:160])
                    return out
            except Exception as exc:
                self.event("v2_flow_error", flow=flow, error=str(exc)[:200])
        return self.v2.handle(kind, sender, body)

    def h4(self) -> None:
        """Each finished event slot: flow H5 Report (or in-process) signs and sends the meter reading."""
        for cid, sid in self.v2.due_readings():
            try:
                out = self.run_flow("H5", json.dumps({"contract_id": cid, "slot_id": sid}))
                if out is not None:
                    self.event("h5_report_langflow", output=out[:160])
                    continue
            except Exception as exc:
                self.event("v2_flow_error", flow="H5", error=str(exc)[:200])
            c = self.v2.contracts[cid]
            self.v2.send_reading(c.get("transaction_id", ""), self.v2.build_reading(cid, sid))

    def inbox_loop(self) -> None:
        """The Town inbox, watched with short checks (town.InboxPoller: no held long-poll, which costs NANDA Town 0.2
        about 20 database scans a second): every `town_poll_seconds` (1 s) while messages flow, every
        `town_poll_idle_seconds` (4 s) after a quiet minute."""
        while not self.town:
            time.sleep(5)
        c = self.cfg
        self.poller = InboxPoller(self.town, self._handle_claimed, every=float(c.get("town_poll_seconds", 1.0)),
                                  idle_every=float(c.get("town_poll_idle_seconds", 4.0)),
                                  on_error=lambda where, exc: self.event("town_error", error=f"{where}: {str(exc)[:200]}"),
                                  alive=lambda: not self.offline())
        while True:                       # the poller stops while the home is offline (what-if) and starts again after
            if self.offline():
                time.sleep(0.5)
                continue
            self.poller.run()

    def _handle_claimed(self, msg: dict[str, Any]) -> None:
        if msg.get("kind") == "flex.dispatch_request":
            self.h2(msg)
        elif str(msg.get("kind", "")).startswith("beckn."):
            sent = ((msg.get("body") or {}).get("context") or {}).get("sentWall")
            if isinstance(sent, (int, float)):
                late = max(0.0, time.time() - sent)
                self.town_in.append((msg["kind"][6:], late))
                log.info("town_in %s from %s %.2f s after it was sent", msg["kind"], msg.get("from"), late)
            try:
                note = self.h_v2(msg)
            except Exception as exc:
                note = beckn.Refused(f"error: {exc}")
                log.error("v2: %s", traceback.format_exc())
            self.town.ack(msg["message_id"], msg["fence"], *beckn.ack_of(note))
        else:
            self.town.ack(msg["message_id"], msg["fence"], "rejected", {"reason": "unknown_kind"})


def retention_loop(cfg: dict[str, Any], event) -> None:
    """Hourly: keep Langflow's stored runs to the last `langflow_keep_hours` (see retention.py)."""
    from .retention import prune_langflow
    time.sleep(120)
    while True:
        try:
            out = prune_langflow(cfg["langflow_url"], config.secret("langflow_api_key"),
                                 float(cfg.get("langflow_keep_hours", 48)))
            event("langflow_pruned", **out)
        except Exception as exc:
            event("langflow_prune_error", error=str(exc)[:200])
        time.sleep(3600)


def build_app(agent: Agent) -> FastAPI:
    app = FastAPI(title="flexgridd", version="0.1.0", dependencies=[doors.door()])

    @app.get("/health")
    def health():
        return {"ok": True, "agent": agent.cfg["agent_name"], "site": agent.cfg["site_label"], "brand": agent.cfg["brand"],
                "adapter": agent.cfg["adapter"], "town": bool(agent.town), "pubkey": agent.pub,
                "town_stats": agent.town_stats(), "log": str(config.REPO_DIR / "logs" / "daemon.log"),
                "town_calls": agent.poller.stats() if getattr(agent, "poller", None) else None,
                "replay": {"start": agent.replay.start, "step_seconds": agent.replay.step_seconds,
                           "now": agent.replay.ts(agent.replay.current())},
                "what_if": {"weather": dict(replay_mod.WEATHER), "offline_s": max(0.0, round(agent.offline_until - time.time(), 1)),
                            "held": len(agent.outbox), "battery_fault": getattr(agent.adapter, "P", 1) == 0 and getattr(agent.adapter, "E", 0) > 0}}

    @app.get("/telemetry")
    def telemetry():
        if agent.offline():               # what-if: the Jetson is unreachable, as the coordinator would see it
            raise HTTPException(503, f"offline (what-if) for {agent.offline_until - time.time():.0f} s more")
        tel = agent.read()
        return {"telemetry": tel, "history": list(agent.history)[-8:], "limits": agent.adapter.limits()}

    @app.post("/telemetry/check")
    def telemetry_check(body: dict = Body(default={})):
        hist = body.get("history") or list(agent.history)
        return tcheck.check(hist, agent.adapter.limits())

    @app.get("/forecast")
    def forecast(n: int = 8):
        tel = agent.read()
        return {"slots": fc.seasonal_naive(agent.replay, tel["idx"], n=n)}

    @app.post("/envelope/build")
    def envelope_build(body: dict = Body(default={})):
        return agent.build_envelope(body.get("telemetry"), body.get("health"))

    @app.post("/envelope/publish")
    def envelope_publish(body: dict = Body(...)):
        return agent.publish(body)

    @app.get("/envelope/last")
    def envelope_last():
        return agent.last_envelope or {}

    @app.post("/dispatch/verify")
    def dispatch_verify(body: dict = Body(...)):
        return agent.verify(body)

    @app.post("/dispatch/apply")
    def dispatch_apply(body: dict = Body(...)):
        return agent.apply(body["dispatch"], body.get("verdict"))

    @app.post("/dispatch/ack")
    def dispatch_ack(body: dict = Body(...)):
        return agent.ack(body["dispatch"], body["receipt"], body.get("message_id"), body.get("fence"))

    @app.get("/receipts")
    def receipts(n: int = 20):
        rows = agent.db.execute("SELECT body, hash FROM receipts ORDER BY n DESC LIMIT ?", (n,)).fetchall()
        return [{**json.loads(b), "receipt_hash": h} for b, h in rows]

    @app.get("/events")
    def events(n: int = 50):
        return list(agent.events)[-n:]

    @app.get("/logs/tail")
    def logs_tail(after: int = -1):
        """This agent's terminal (logs/daemon.log; a building node's logs/<daemon_log>) from byte `after` on, for the
        Control Room's terminals."""
        from . import wire
        return wire.tail(config.REPO_DIR / "logs" / log_name(agent.cfg), after)

    @app.post("/demo/fault")
    def demo_fault(body: dict = Body(default={})):
        if not hasattr(agent.adapter, "inject_fault"):
            raise HTTPException(400, "adapter has no fault injection")
        return agent.adapter.inject_fault(body.get("kind", "freeze"), int(body.get("steps", 6)))

    @app.post("/demo/reset")
    def demo_reset():
        agent.v2.reset()
        scenario_reset()
        return {"ok": True}

    def scenario_reset() -> None:
        replay_mod.set_weather(1.0, 1.0, dT=0.0)
        agent.offline_until = 0.0
        if hasattr(agent.adapter, "set_battery_fault"):
            agent.adapter.set_battery_fault(False)
        if hasattr(agent.adapter, "overrides"):
            agent.adapter.overrides.clear()

    @app.post("/demo/scenario")
    def demo_scenario(body: dict = Body(default={})):
        """What if (the Control Room): {"weather": {"pv": 0.3, "load": 1.2}} from now on, {"offline_s": 120} (0 = back
        now), {"battery_fault": true|false}, {"reset": true}. Changes this home's physical truth or its link, never
        its decisions: the agent reacts with its own code."""
        out: dict[str, Any] = {"agent": agent.cfg["agent_name"]}
        if body.get("reset"):
            scenario_reset()
            out["reset"] = True
        w = body.get("weather")
        if isinstance(w, dict):
            now = agent.replay.ts(agent.replay.current())
            to = beckn.add_minutes(now, 60 * float(w["hours"])) if w.get("hours") else None
            out["weather"] = replay_mod.set_weather(w.get("pv"), w.get("load"), now, to=to, dT=w.get("dT"))
        ov = body.get("override")
        if isinstance(ov, dict):                          # the owner's wish: "I need the car now", a comfort limit
            k = ov.get("kind")                            # (a building's facility manager: comfort, lights, stop)
            kinds = ("comfort", "lights", "stop", "clear") if agent.cfg.get("role") == "building" else \
                ("ev_now", "comfort", "stop", "depart_at", "clear")
            if k not in kinds or not hasattr(agent.adapter, "rt_override"):
                raise HTTPException(400, f"override: kind {'|'.join(kinds)} on a simulated "
                                         f"{'building' if agent.cfg.get('role') == 'building' else 'home'}")
            agent.adapter.rt_override(k, ov.get("value", True))
            agent.event("owner_override", override=k, value=ov.get("value", True), source=ov.get("source", "control room"))
            out["override"] = {"kind": k, "overrides": dict(agent.adapter.overrides),
                               "mode": getattr(agent.adapter, "mode", "fixed")}
        if "offline_s" in body:
            sec = max(0.0, min(2700.0, float(body["offline_s"] or 0)))
            agent.offline_until = time.time() + sec if sec > 0 else 0.0
            out["offline_s"] = sec
            agent.event("what_if_offline", seconds=sec)
        if "battery_fault" in body and hasattr(agent.adapter, "set_battery_fault"):
            out.update(agent.adapter.set_battery_fault(bool(body["battery_fault"])))
            agent.event("what_if_battery_fault", on=bool(body["battery_fault"]))
        out["now"] = agent.replay.ts(agent.replay.current())
        return out

    @app.post("/demo/clock")
    def demo_clock(body: dict = Body(default={})):
        agent.replay.configure(body.get("start", agent.replay.start),
                               float(body.get("step_seconds", agent.replay.step_seconds)),
                               int(body.get("loop_days", agent.replay.loop_days)), body.get("at"), body.get("at_wall"))
        if not body.get("keep"):          # a new run starts the home afresh; a change of speed (keep) goes on with it
            agent.adapter.idx = None
        return agent.replay.clock()

    # ---- v2 steps, one endpoint per Langflow component (H3 Offer, H4 Accept, H5 Report)
    @app.get("/v2/status")
    def v2_status():
        return {**agent.v2.status(), "offers": agent.v2.offers, "contracts": agent.v2.contracts,
                "payments": agent.v2.payments}

    @app.post("/v2/message/check")
    def v2_check(body: dict = Body(...)):
        b = body["body"]
        ok = body.get("from") == agent.v2.coordinator and b.get("signer") in agent.trusted_coordinators() \
            and signing.verify(b, b.get("signer", ""))
        return {"ok": ok, "kind": body.get("kind"), "transaction_id": b.get("context", {}).get("transactionId")}

    # the Langflow flows call these as tools: each call is logged with its input and output (the three terminals)
    @app.post("/v2/offer/forecast")
    def v2_forecast(body: dict = Body(...)):
        n, t0 = body["need"], time.perf_counter()
        sl = n.get("slots") or []
        agent.v2.log("recv", what="call for offers", sender=body.get("from") or agent.cfg.get("coordinator_name", "coordinator"),
                     event=n.get("event_id"), window=f"{n['window_start'][5:]}-{n['window_end'][11:]}", half_hours=len(sl),
                     kw_wanted=sl[0].get("kw_requested") if sl else None, price=sl[0].get("price") if sl else None)
        fc = agent.v2.forecast_window(n)
        agent.v2._tool("forecast_window", f"[Langflow H3 Offer] event {n['window_start'][5:]}-{n['window_end'][11:]} "
                       f"({len(sl)} half hours), as of {agent.v2.now_ts()[5:]}", agent.v2._fc_words(fc), t0)
        return fc

    @app.post("/v2/offer/build")
    def v2_build(body: dict = Body(...)):
        t0 = time.perf_counter()
        offer = agent.v2.build_offer(body["need"], body["forecast"])
        lim = agent.adapter.limits()
        soc = (body.get("forecast") or {}).get("soc_now_pct")
        agent.v2._tool("plan_offer", f"[Langflow H3 Offer] battery {lim['battery_kwh']:g} kWh" + (f" at {soc:.0f} %" if soc is not None else "")
                       + f", reserve {lim['reserve_pct']:.0f} %" + (f"; EV charger {lim['ev_max_kw']:g} kW" if lim.get("ev_max_kw") else ""),
                       " · ".join(f"{x['kw_offered']:g}" for x in offer["slots"]) + f" kW per half hour = {offer['energy_kwh']} kWh, "
                       f"confidence {offer['confidence']}", t0)
        return offer

    @app.post("/v2/offer/send")
    def v2_send_offer(body: dict = Body(...)):
        return agent.v2.send_offer(body["transaction_id"], body["offer"])

    @app.post("/v2/contract/verify")
    def v2_verify(body: dict = Body(...)):
        t0 = time.perf_counter()
        v = agent.v2.verify_contract(body["from"], body["body"])
        c = ((body.get("body") or {}).get("message") or {}).get("contract") or {}
        agent.v2._tool("check_commitment", "[Langflow H4 Accept] share " + " · ".join(f"{x['kw']:g}" for x in c.get("commitments", []))
                       + f" kW, opt out by {str(c.get('opt_out_by', ''))[5:]}", f"{v['decision']} ({v['reason']})", t0)
        return v

    @app.post("/v2/contract/install")
    def v2_install(body: dict = Body(...)):
        return agent.v2.install_plan(body["contract"], body["verdict"])

    @app.post("/v2/contract/answer")
    def v2_answer(body: dict = Body(...)):
        return agent.v2.answer(body["transaction_id"], body["contract"], body["verdict"])

    @app.post("/v2/report/build")
    def v2_report_build(body: dict = Body(...)):
        return agent.v2.build_reading(body["contract_id"], int(body["slot_id"]))

    @app.post("/v2/report/send")
    def v2_report_send(body: dict = Body(...)):
        rd = body["reading"]
        txn = body.get("transaction_id") or agent.v2.contracts.get(rd["contract_id"], {}).get("transaction_id", "")
        return agent.v2.send_reading(txn, rd)

    # ---- the AI agent: the phone app's Agent tab (through the coordinator) and OpenClaw's FlexGrid tools (agent_mcp)
    def _brain():
        if agent.brain is None:
            raise HTTPException(404, "this home has no AI agent (llm_enabled is off on this Jetson)")
        return agent.brain

    @app.get("/agent/state")
    def agent_state(since: float = 0.0):
        """The agent's decisions and its chat with the owner; since = only what is newer (wall time) than that."""
        if agent.brain is None:
            return {"enabled": False, "home": agent.cfg["agent_name"],
                    "why": "the AI agent is off on this Jetson (llm_enabled in agent.json)"}
        st = agent.brain.state()
        if since:
            st["decisions"] = [d for d in st["decisions"] if d.get("wall", 0) > since]
            st["chat"] = [m for m in st["chat"] if m.get("wall", 0) > since]
        return st

    @app.post("/agent/chat")
    def agent_chat(body: dict = Body(...)):
        return _brain().chat(str(body.get("message") or ""))

    @app.get("/agent/wishes")
    def agent_wishes():
        return _brain().wishes.get()

    @app.post("/agent/wishes")
    def agent_set_wishes(body: dict = Body(...)):
        w = _brain().wishes.set(**{k: v for k, v in (body or {}).items()})
        agent.event("agent_wishes", by="owner", **{k: v for k, v in w.items() if k != "notes"})
        return w

    @app.get("/agent/tools")
    def agent_tools():
        """The agent's tools (name, description, JSON-schema parameters): agent_mcp serves exactly these to OpenClaw."""
        return [{"name": n, **spec} for n, (spec, _) in _brain().tools().items()]

    @app.post("/agent/tool/{name}")
    def agent_tool(name: str, body: dict = Body(default={})):
        """One tool, run by this daemon's own code (the arguments are checked there)."""
        return {"result": _brain().run_tool(name, body or {}, by="mcp")}

    return app


def _stats(v: list[float], nd: int) -> dict[str, Any]:
    if not v:
        return {"n": 0}
    v = sorted(v)
    q = lambda p: v[min(len(v) - 1, int(round(p * (len(v) - 1))))]          # noqa: E731
    return {"n": len(v), "p50": round(q(0.5), nd), "p95": round(q(0.95), nd), "max": round(v[-1], nd)}


def log_name(cfg: dict[str, Any]) -> str:
    """This daemon's log file under logs/: daemon.log for the home; a second daemon on the same Jetson (a building
    node, role "building") writes its own (cfg daemon_log, default building.log), so the two never share a file."""
    return str(cfg.get("daemon_log") or ("building.log" if cfg.get("role") == "building" else "daemon.log"))


def main() -> None:
    from logging.handlers import RotatingFileHandler
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)      # one line per Town poll is noise
    # a record that outlives the process (the Jetsons keep no journal): logs/daemon.log, 4 x 5 MB at most
    logs = config.REPO_DIR / "logs"
    logs.mkdir(exist_ok=True)
    fh = RotatingFileHandler(logs / log_name(config.load()), maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(fh)
    agent = Agent()
    threading.Thread(target=agent.clock_loop, daemon=True).start()
    threading.Thread(target=agent.inbox_loop, daemon=True).start()
    if agent.cfg.get("role", "home") == "home":       # Langflow's stored runs: the home's flows (a building has none)
        threading.Thread(target=retention_loop, args=(agent.cfg, agent.event), daemon=True).start()
    uvicorn.run(build_app(agent), host=agent.cfg["daemon_host"], port=int(agent.cfg["daemon_port"]), log_level="warning")


if __name__ == "__main__":
    main()
