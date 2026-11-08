"""flexgrid-grid-operator - the stand-in buyer of large-load flexibility ("grid operator large-load program (mock)").

Same Beckn DEG demand-flex flow and the same code as the household buyer (utility.py: discover, select / init /
confirm, baselines, signed meter readings, settlement); only three things differ:
  * its needs: the dataset's large-load events (data/datacenters/large_load_events_2023.json from
    tools/make_dc_events.py; data_ma/datacenters/large_load_events_2026.json from tools/make_ma_datacenters.py): the
    household program's event windows, a megawatt request, paid at the window's day-ahead price (the energy the grid
    did not have to buy)
  * its meters: the data centers'. Meter history = each site in default operation (every job runs on arrival,
    battery idle), from data/datacenters/ (flexgrid_core/datacenter.py)
  * its name: `grid_operator_name` (default "grid-operator"), with its own signing key
Baseline: Ameren Rider PTR (3 highest of the 14 previous non-holiday, non-event days), as for the homes. A data
center's load hardly follows the weather, so the PTR's "highest 3 of 14" sits a little above a normal day.
"""
from __future__ import annotations

import json
from typing import Any

from . import beckn, config, datacenter as dc, settlement as st, signing
from .utility import Utility


class GridOperator(Utility):
    # its own AI agent's order replaces the day's catalog request (brain.GridOperatorBrain, through dc_fleet):
    # megawatts, under the same rules as the utility's buyer (a window of 30 min to 4 h, placed before the 10:00 call)
    ORDER_MAX_KW = 100000.0
    ORDER_PREFIX = "dc-flex-need-"

    def __init__(self):
        super().__init__()
        c = self.cfg
        # a program sized to its fleet: the dataset's order_max_kw lifts the ceiling (none: the 100 MW of the ten sites)
        self.ORDER_MAX_KW = float(c.get("dc_order_max_kw") or dc.files().get("order_max_kw") or type(self).ORDER_MAX_KW)
        self.name = c.get("grid_operator_name", "grid-operator")
        self.key = signing.load_or_create(config.CONFIG_DIR / "secrets" / "grid_operator.key")
        self.pub = signing.public_hex(self.key)
        self.catalog = dc.events_catalog()
        self.fleet = {}
        self.event_days = {beckn.need_from_catalog(r, o)["window_start"][:10] for r, o in self._pairs()}
        self.aggregators = set(c.get("grid_operator_aggregators") or ["coordinator"])

    def baselines(self, ct: dict[str, Any], meters: list[str]) -> dict[str, dict[str, dict[str, float]]]:
        """Each data center's Rider PTR baseline per slot, from its own meter history in default operation."""
        need = ct["need"]
        day = need["window_start"][:10]
        ev_days = {st.day_of(x) for x in self.event_days}
        days = st.similar_days(st.day_of(day), ev_days) if st.MARKET else st.ptr_baseline_days(st.day_of(day), ev_days)
        out = {}
        for m in meters:
            try:
                s = dc.Series(dc.entity_of(m))
            except (FileNotFoundError, StopIteration):
                continue
            usage_by_day = {}
            for d in days:
                vals = []
                for sl in need["slots"]:
                    i = s.idx_of(d.isoformat() + sl["start"][10:])
                    if i is not None:
                        vals.append((s.default_import(i) + s.default_import(i + 1)) / 2)
                if len(vals) == len(need["slots"]):
                    usage_by_day[d] = vals
            b = (st.ten_of_ten if st.MARKET else st.ameren_baseline)(usage_by_day, st.day_of(day), ev_days)
            out[m] = {str(sl["id"]): {"BASELINE": b["per_slot"][j] if b["per_slot"] else 0.0}
                      for j, sl in enumerate(need["slots"])}
            out[m]["_days"] = b["days"]
        return out

    def settlement_baselines(self, ct: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
        return ct.get("baselines", {})
