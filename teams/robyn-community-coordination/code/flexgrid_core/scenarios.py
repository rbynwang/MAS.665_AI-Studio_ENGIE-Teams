"""What if, for the audience: conditions of the day that change the need and the supply, set before a live run (the
Control Room's "Run the event live") or while it runs (What if). A preset is only a set of changes to the world the
agents live in; every agent reacts with its own code and its own AI agent, and nothing here decides for them:

  weather        the homes' sunshine (solar x) and household load (x), all 100 homes, from now on (their daemons)
  region         the grid operator's forecast of the region's load (x) and what the weather service says: its AI agent
                 reads both before it orders the data centers' megawatts (flexgrid-dc /dc/conditions "operator")
  datacenters    urgent batch work at the data centers (the share that cannot wait today: their safe offers shrink, and
                 so does what they can hold in the window) and data center batteries out of service ("sites")
  battery_fault  home batteries trip (the Tesla home, the Enphase home, or n of the fleet)
  note           what the utility's buyer agent is told about tomorrow before it orders the homes' kilowatts

The coordinator (POST /demo/scenario {"preset": ...}, or /demo/start {"scenario": ...}) sends each part to the agents it
concerns; tests/test_scenarios.py checks the presets.
"""
from __future__ import annotations

from typing import Any

PRESETS: dict[str, dict[str, Any]] = {
    "normal": {
        "label": "A normal day",
        "what": "The dataset's own weather and data centers: nothing changed.",
        "expect": "The program's requests and the agents' usual answers.",
        "changes": {}},
    "heat": {
        "label": "Heat wave",
        "what": "A heat advisory for tomorrow: air conditioning runs about 30 % above normal in every home, and the "
                "weather service expects the region's evening load about 12 % above the last days'.",
        "expect": "More need: the grid operator's agent sees its forecast peak above the planning level and may order "
                  "more megawatts; the utility's buyer agent hears the forecast before it orders the homes' kilowatts.",
        "changes": {"weather": {"pv": 1.0, "load": 1.3},
                    "region": {"load_x": 1.12, "note": "Heat advisory for tomorrow: the weather service expects the "
                                                       "region's evening load about 12 % above the last days'."},
                    "note": "Heat advisory for tomorrow: household air conditioning runs about 30 % above normal in "
                            "the evening, so the community's evening peak will be higher than its forecast shows."}},
    "cool": {
        "label": "Cool, cloudy day",
        "what": "Cool and overcast: a third of the usual sunshine on every roof, household load 10 % below normal, "
                "and the region's load about 6 % below the last days'.",
        "expect": "Less need: the grid operator's agent sees its forecast peak further under the planning level and "
                  "may order less. The homes' offers, made from the forecast as issued, hardly move: their batteries "
                  "charge before the window anyway, and with less of their own load to cover they deliver a little more.",
        "changes": {"weather": {"pv": 0.35, "load": 0.9},
                    "region": {"load_x": 0.94, "note": "Cool and cloudy tomorrow: the weather service expects the "
                                                       "region's load about 6 % below the last days'."},
                    "note": "Cool and cloudy tomorrow: little sun to charge the home batteries, and household demand "
                            "a little below normal in the evening."}},
    "dc_busy": {
        "label": "Data centers busy",
        "what": "A rush of urgent work at all ten data centers: 60 % of their batch work must run as it arrives today.",
        "expect": "Less supply: the sites' safe offers shrink (dc-09's agent sees why), the desk may sell less than the "
                  "grid operator ordered, and the money follows what is delivered.",
        "changes": {"datacenters": {"urgent_share": 0.6}}},
    "battery_trouble": {
        "label": "Battery trouble",
        "what": "Batteries out of service: 10 of the simulated homes' and two data centers' (dc-03 and dc-08, the "
                "largest).",
        "expect": "Less supply: offers made now leave those batteries out; during the window, homes and sites that "
                  "committed them fall short, and the signed readings and the settlement show it.",
        "changes": {"battery_fault": {"home": "fleet", "n": 10, "on": True},
                    "datacenters": {"battery_out": ["dc-03", "dc-08"]}}},
}


def expand(name: str) -> dict[str, Any]:
    """A preset as a /demo/scenario body: everything back to normal first, then its changes. ValueError if unknown."""
    p = PRESETS.get(str(name or ""))
    if p is None:
        raise ValueError(f"no scenario {name!r}: one of {', '.join(PRESETS)}")
    return {"reset": True, "preset": name, **p["changes"]}


def catalog() -> list[dict[str, Any]]:
    """The presets for the Control Room (GET /demo/scenario)."""
    return [{"id": k, **{x: v[x] for x in ("label", "what", "expect")}} for k, v in PRESETS.items()]
