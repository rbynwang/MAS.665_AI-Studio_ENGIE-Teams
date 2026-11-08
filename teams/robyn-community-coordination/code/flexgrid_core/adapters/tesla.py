"""Tesla Powerwall adapter (real hardware). Read: pypowerwall (local TEDAPI / Fleet API).
Write: Fleet API energy_cmds (backup reserve <= 80 %, operation mode, grid charging/export).
Not wired yet: the replay adapter stands in until the Powerwall credentials are set up."""
from __future__ import annotations


class TeslaAdapter:
    name = "tesla"

    def __init__(self, cfg):
        raise NotImplementedError("Tesla adapter: set up pypowerwall / Fleet API credentials first; "
                                  "use adapter=simulated meanwhile")
