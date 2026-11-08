"""The coordinator's own agent on the new path. The model proposes, and the market's code decides.

It keeps the two decisions the coordinator's agent has always had (brain.CoordinatorBrain):
  margin    how much more to ask of the nodes than is sold. The model reads a short sheet of ready numbers: the offers, what each margin
            would sell, the community forecast. Code holds the answer to 5-30 %, and the clearing still sells only what's there
  explain   after a call or a gap, what happened, in plain words. The numbers stay the code's

If Qwen is down or answers badly, the market takes its default margin and the template explanation, so a sale never waits on it. Market
decisions use the lowest sampling variation the model allows (MULTI_AGENT_DESIGN, section 6).
"""
from __future__ import annotations

from typing import Any, Callable

from .brain import CoordinatorBrain, OllamaBackend
from .llm import LLM

URL = "http://127.0.0.1:11434"
MODEL = "qwen3.8-27b-q3:latest"
PROMPT = """You are the agent of a FlexGrid coordinator. Homes and buildings have sent offers, and the coordinator sells them as one pool to the
buyers' fixed orders. The code will ask the nodes for more than it sells: that extra is the margin, and it covers nodes that deliver less than they
offered. Choose a margin from 0.05 to 0.30. The default is 0.10.
A bigger margin can sell less, but keeps the promise on an uncertain day. A smaller one risks a shortfall, which costs reputation and future income.
Read sell_by_margin first: for each margin it gives what would be sold, what that earns, and what the nodes would be asked for. If a bigger margin
sells the same, it's a safer promise at no cost.
community_forecast gives the homes' demand in the sale hours with its range: a wide range is a less certain day. Firm power is surer than tentative.
Use only these facts. One or two plain sentences, with the numbers that mattered."""


def margin_sheet(facts: dict[str, Any]) -> dict[str, Any]:
    """The facts as the agent reads them: only what bears on the margin, so the prompt stays short."""
    keep = ("now", "default_margin", "request", "community_forecast", "nodes", "pool_first_half_hour", "orders", "sell_by_margin", "this_run_finished_events")
    out = {k: facts[k] for k in keep if k in facts}
    o = facts.get("offers") or {}
    out["offers"] = {"nodes": o.get("homes"), "offering_nothing": o.get("homes_offering_nothing"), "confidence": o.get("confidence")}
    return out


class CoordinatorAgent(CoordinatorBrain):
    def decide_margin(self, snap: dict[str, Any]) -> dict[str, Any]:
        return super().decide_margin(margin_sheet(snap))


def make_agent(url: str = URL, model: str = MODEL, log: Callable[..., None] | None = None, budget_s: float = 60.0) -> CoordinatorAgent:
    llm = LLM(url, model, decide_options={"temperature": 0.0, "seed": 7}, label="coordinator")
    return CoordinatorAgent(OllamaBackend(llm), log=log, budget_s=budget_s, prompt=PROMPT)
