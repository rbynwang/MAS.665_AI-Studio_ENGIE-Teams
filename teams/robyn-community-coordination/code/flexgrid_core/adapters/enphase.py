"""Enphase IQ Gateway adapter (real hardware). Read: pyenphase with an owner token.
Write: blocked locally on gateway firmware >= 8.2.4225, so this adapter runs in advisory
mode (recommend, homeowner approves in the Enphase app). Not wired yet."""
from __future__ import annotations


class EnphaseAdapter:
    name = "enphase"

    def __init__(self, cfg):
        raise NotImplementedError("Enphase adapter: set up the IQ Gateway token first; "
                                  "use adapter=simulated meanwhile")
