"""The smart systems FlexGrid's agents can call: the 2026 setting of the benchmark copy. Nothing here is deployed.

data_ma/devices/catalog.json lists them per node type (home, building). data_ma/devices/<node>_<device>.npy holds each
one: row = node, column = replay step, kW it draws on the node's own day, already inside the load the node's meter reads
(or, for a car's bidirectional charger, what it could give while plugged in).

  kind shift    it waits while something is owed, then catches up at its own power (share = the part that may wait)
  kind shed     it is turned down while something is owed and nothing comes back (share = the part turned down)
  kind source   it gives stored energy (a car); what it gave is charged back before it leaves

Switch: FG_DEVICES = "all", a comma list of devices, or unset (off: this code then does nothing).
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from .config import data_path

_ON = "" if os.environ.get("FLEXGRID_DATASET") == "il2023" else os.environ.get("FG_DEVICES", "")
_CAT: dict[str, Any] | None = None
_ARR: dict[tuple[str, str], Any] = {}


def catalog() -> dict[str, Any]:
    global _CAT
    if _CAT is None:
        _CAT = json.loads(data_path("devices/catalog.json").read_text())
    return _CAT


def enabled(node: str) -> dict[str, dict[str, Any]]:
    if not _ON:
        return {}
    cat = catalog()["devices"].get(node) or {}
    return dict(cat) if _ON == "all" else {k: v for k, v in cat.items() if k in set(_ON.split(","))}


def array(node: str, device: str):
    key = (node, device)
    if key not in _ARR:
        _ARR[key] = np.load(data_path(f"devices/{node}_{device}.npy"), mmap_mode="r")
    return _ARR[key]


def for_node(node: str, name: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any], int | None]:
    """(the devices this node owns and has switched on, their arrays, its row)."""
    on = enabled(node)
    if not on:
        return {}, {}, None
    row = (catalog()["rows"].get(node) or {}).get(name)
    if row is None:
        return {}, {}, None
    specs, arrs = {}, {}
    for k, v in on.items():
        a = array(node, k)
        if float(np.asarray(a[row]).max()) > 0.0:
            specs[k], arrs[k] = v, a
    return specs, arrs, row


def stagger_min() -> int:
    return int(catalog().get("stagger_max_min", 120))


def catch_up_after_min() -> int:
    return int(catalog().get("catch_up_after_min", 90))
