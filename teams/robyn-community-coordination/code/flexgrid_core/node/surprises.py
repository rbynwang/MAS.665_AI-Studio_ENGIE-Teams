"""What can happen to a building during the day that nobody planned for, on a real node as in the offline simulation.

  close_building(h, now, share=None)                   closed for the day: its true load falls to `share` of normal (the night's level by
                                                       default), the no-event twin's with it, and it leaves the sale
  zone_event(h, now, share=0.25, hours=3.5, space=None)  an event in a zone already sold: that share of the lights can't dim and that share
                                                       of the comfort room is gone, from now on, for `hours`

tools/sim_event.py's building_closed and zone_event call these, and so does a building service's /b/<name>/demo/scenario, so the
Homework 8 day and the offline day change a building the same way. Only the building's physical truth and its person's word change:
its agent sees the rest through its own sensors.
"""
from __future__ import annotations

from typing import Any

from .. import beckn


def zones(h: Any) -> Any:
    """The building's zone map, for what its sensors read. None for a node without one."""
    from . import buildings
    return buildings.load(h.name) if h.name in buildings.KEYS else None


def close_building(h: Any, now: str, share: float | None = None) -> dict[str, Any]:
    """Closed for the day: its true load falls, the no-event twin's with it, so nothing it does from here counts as relief. It leaves the
    sale. Its report and its agent's answer are theirs to make."""
    a, rp = h.adapter, h.replay
    i = rp.idx_of(now)
    night = sum(rp.rows[j]["base_kw"] for j in range(rp.idx_of(f"{now[:10]} 02:00"), rp.idx_of(f"{now[:10]} 04:00"))) / 8
    share = float(share) if share is not None else min(1.0, night / max(rp.rows[i]["base_kw"], 1.0))
    end = beckn.add_minutes(f"{now[:10]} 00:00", 1440)
    h.replay.weather = {**(h.replay.weather or {}), "from": now, "to": end, "pv": 1.0, "load": share}
    a.rt_override("closed", {"from": now, "share": share})
    a.rt_override("stop", True)
    zm = zones(h)
    if zm:
        zm.surprises.append({"from": now, "to": end, "space": None, "share": 0.0})
    return {"closed": {"from": now, "share": round(share, 3)}}


def zone_event(h: Any, now: str, share: float = 0.25, hours: float = 3.5, space: str | None = None) -> dict[str, Any]:
    """A zone already sold must run as normal from now on: that share of the lights can't dim and that share of the comfort room is gone.
    The controller finds what's left, and the node reports the rest."""
    a, z = h.adapter, float(share)
    end = beckn.add_minutes(now, 60 * float(hours))
    a.rt_override("zone_event", {"from": now, "to": end, "share": z, "space": space})
    zm = zones(h)
    s = None
    if zm:                                                     # the sensors read the room in use: the building's agent sees it itself
        s = space or max((k for k, v in zm.spaces.items() if not v.get("closed")), key=lambda k: zm.spaces[k].get("area", 0.0))
        zm.surprises.append({"from": now, "to": end, "space": s, "share": 1.0})
    if hasattr(a, "light_limit"):                              # the limit, not what the levers have open: with none open that would read as 0
        a.rt_override("lights", a.light_limit() * (1 - z))
    if hasattr(a, "sell"):                                     # its fans and plugs too: the sale's part of them shrinks with the free space
        a.sell = {**a.sell, **{k: float(a.sell.get(k, a.app[k].get("share", 1.0))) * (1 - z) for k in a.app if k != "ev"}}
        if a.levers is not None:                               # and what the agent's calls already opened
            a.levers["dev"] = {k: v * (1 - z) if k != "ev" else v for k, v in a.levers["dev"].items()}
    from ..rt_control import person_limit
    a.rt_override("comfort", {k: person_limit(a, k) * (1 - z) for k in ("max_rise_c", "max_rise_away_c")})
    return {"zone_event": {"from": now, "to": end, "share": z, "space": s}}
