"""The two stand-in buyers of the new path: the utility and the grid operator. A separate buy-side project takes their place once the seam is agreed.

Each is the utility's code (baselines, signed readings, settlement), with three differences:
  * it publishes only its fixed orders, not the dataset's catalog, so every run gets the same ask
  * it makes no calls of its own on price, only on an alert such as a grid emergency
  * the grid operator has its own name, key and order ids

Its meters are the community's homes and buildings. No data center is on this path: grid_operator.py stays for the old one.
"""
from __future__ import annotations

from typing import Any

from . import beckn
from .utility import Utility


class StandInBuyer(Utility):
    CLASS_RATES = True                                  # the desk pays a pool's call by class, so a call is billed by class too

    def __init__(self, name: str = "utility"):
        super().__init__(f"{name}.key")
        self.name = name
        if name != "utility":
            self.ORDER_PREFIX = f"{name}-need-"
            self.PARTICIPANT = f"{name}-stand-in"
        # the program's days stay out of every baseline, whether or not this buyer orders on them
        self.event_days = {beckn.need_from_catalog(r, o)["window_start"][:10] for r, o in Utility._pairs(self)}
        self.watch.price_rule = False

    def _pairs(self) -> list[tuple[dict[str, Any], dict[str, Any]]]:
        return [(o["resource"], o["offer"]) for o in self.orders.values()]
