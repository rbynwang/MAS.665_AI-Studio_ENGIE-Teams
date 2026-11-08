"""Seasonal-naive probabilistic forecast from the replay's own past (no peeking ahead).

For each of the next n 15-minute slots: P50 = mean of the same slot over the previous 3 days,
P10/P90 = min/max of those days. A baseline the ML forecasters (Granite TTM, Chronos-Bolt)
must beat later.
"""
from __future__ import annotations

from typing import Any


def seasonal_naive(replay, idx: int, n: int = 8, days: int = 3) -> list[dict[str, Any]]:
    out = []
    for h in range(1, n + 1):
        j = idx + h
        pv = [replay.row(j - 96 * d)["pv_kw"] for d in range(1, days + 1) if j - 96 * d >= 0]
        ld = [replay.row(j - 96 * d)["base_kw"] for d in range(1, days + 1) if j - 96 * d >= 0]
        if not pv:
            pv, ld = [0.0], [0.0]
        out.append({"slot": replay.ts(j),
                    "pv_p50": round(sum(pv) / len(pv), 3), "pv_p10": round(min(pv), 3), "pv_p90": round(max(pv), 3),
                    "load_p50": round(sum(ld) / len(ld), 3), "load_p10": round(min(ld), 3), "load_p90": round(max(ld), 3)})
    return out
