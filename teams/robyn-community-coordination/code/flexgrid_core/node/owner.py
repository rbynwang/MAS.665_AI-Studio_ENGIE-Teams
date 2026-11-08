"""An owner's note: the model reads the free text, and code keeps the promise.

A note goes through its own short step, because a sale step gave the model a sale to fill and it spent the battery anyway. Here
the model only says what the owner needs. The wish then sits on the node, the guard enforces it at step 5, and nothing else has
to remember it.
"""
from __future__ import annotations

from typing import Any

SYSTEM = ("You read a note from a home's owner to the home's energy agent. Call set_owner_wish once for each thing the owner needs "
          "from the home's devices. If the note asks for nothing about them, call no_wish. Use only what the note says.")
WISH = {"type": "function", "function": {
    "name": "set_owner_wish",
    "description": ("Records what the owner needs. keep_battery_full: the home battery stays full and isn't spent. keep_car_ready: the car must be "
                    "charged and isn't held or used, however the note says it. keep_house_as_is: no easing of heating or cooling. keep_hot_water: hot water isn't held."),
    "parameters": {"type": "object", "required": ["wish"], "properties": {
        "wish": {"type": "string", "description": "what the owner needs", "enum": ["keep_battery_full", "keep_car_ready", "keep_house_as_is", "keep_hot_water"]},
        "by": {"type": "string", "description": "HH:MM if the note gives a time, otherwise empty"}}}}}
NONE = {"type": "function", "function": {
    "name": "no_wish", "description": "The note asks for nothing about the home's devices.",
    "parameters": {"type": "object", "properties": {}}}}
BLOCKS = {"keep_battery_full": {"set_battery"}, "keep_car_ready": {"limit_ev_charging", "discharge_car"},
          "keep_house_as_is": {"set_thermostat", "set_room_ac"}, "keep_hot_water": {"set_water_heater"}}
SAYS = {"keep_battery_full": "The owner needs the battery kept full.", "keep_car_ready": "The owner needs the car ready.",
        "keep_house_as_is": "The owner asked to leave the house as it is.", "keep_hot_water": "The owner needs hot water."}


def read_note(brain: Any, note: str) -> dict[str, Any]:
    """{wishes: [...], ms, said}. If the model gives nothing usable, no wish is recorded and the record says so."""
    a = brain.ask(SYSTEM, [WISH, NONE], f'The owner wrote: "{note}"')
    out = []
    for c in a["calls"]:
        if c["name"] == "set_owner_wish" and c["args"].get("wish") in BLOCKS:
            out.append({"wish": c["args"]["wish"], "by": str(c["args"].get("by") or "")})
    return {"wishes": out, "ms": a["ms"], "said": a["text"], "error": a.get("error")}


def blocked_tools(wishes: list[dict[str, Any]]) -> set[str]:
    return set().union(*[BLOCKS[w["wish"]] for w in wishes]) if wishes else set()


def why(tool: str, wishes: list[dict[str, Any]]) -> str:
    return next((SAYS[w["wish"]] for w in wishes if tool in BLOCKS[w["wish"]]), "The owner asked for this to stay as it is.")
