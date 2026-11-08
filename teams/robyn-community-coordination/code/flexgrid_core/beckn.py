"""Beckn v2.0.0 messages for FlexGrid, in the DEG demand-flex vocabulary.

Shape: {"context": {...}, "message": {...}} as in beckn/protocol-specifications-v2 (api/v2.0.0/beckn.yaml) and the
DEG demand-flex implementation guide (beckn/DEG docs/implementation-guides/v2/Demand_Flexibility).
Transport in phase 1: NANDA Town messages whose `kind` is "beckn.<action>" (e.g. beckn.discover, beckn.on_confirm).
Trust in phase 1: our Ed25519 signature over the canonical JSON body (fields `signer`, `sig`); beckn-onix's HTTP
signature replaces it at the grid edge in phase 2. Same key type, same keys.

Community-side actions (coordinator <-> homes) reuse the demand-flex objects:
  discover     coordinator -> home   call for offers: the utility's need (event window, 30-min slots, price)
  on_discover  home -> coordinator   the home's offer: CAPACITY_OFFERED per slot, energy, confidence
  confirm      coordinator -> home   the home's commitment: kW per slot, price, opt-out deadline, EV restart time
  on_confirm   home -> coordinator   ACCEPTED or DECLINED (+ reason)
  update       coordinator -> home   event notice ("pre-charge now")
  on_status    home -> coordinator   signed meter reading per slot (USAGE kW)
  on_status    coordinator -> home   payment notice after settlement
Grid-side actions (coordinator <-> utility) follow the guide: discover/on_discover, select/on_select, init/on_init,
confirm/on_confirm, update (opt-in meters, seller telemetry), on_status (baselines, actuals, settled).
"""
from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from . import signing

TTL_S = 900          # how long a message stays valid after it was sent (Beckn context.ttl); an older one is dropped
# a statement of fact (a signed meter reading of a past slot, the "readings are in" notice, baselines, a settlement, a
# payment) stays valid for a day: live on 29 Oct 2026 the utility's Mac slept through an event's readings, and on waking
# it dropped every reading over 15 minutes old, so the event could never settle. Late evidence is still evidence; what
# it may change is decided by the receiver (a contract already settled stays settled)
TTL_EVIDENCE_S = 86400
VERSION = "2.0.0"
DOMAIN = "beckn.one:deg:demand-flex:2.0.0"
NETWORK = "flexgrid.local/demo-deg"
SLOT_MINUTES = 30


def iso_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def iso_of(wall: float) -> str:
    return datetime.fromtimestamp(wall, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make(action: str, *, sender: str, receiver: str, transaction_id: str, message: dict[str, Any], key,
         sender_role: str = "bap", message_id: str | None = None, ttl_s: float | None = None, wall: float | None = None) -> dict[str, Any]:
    """Build and sign one Beckn v2 message. `sender_role` says which context slot the sender fills. `ttl_s`: how long
    the message stays valid (context.ttl); TTL_S for a request or an answer that only matters while the market is
    deciding, TTL_EVIDENCE_S for a statement of fact (a signed meter reading, a settlement, a payment), which stays
    true however late it arrives.
    The message is put through JSON first, so what is signed is exactly what arrives over NANDA Town (an int dict key
    becomes a string there and sorts differently in the canonical form; signing the Python object would break the
    signature for 11+ slots) and so the in-memory simulation sees what the live agents see.
    An answer that has to come out the same byte for byte however often it's made, with no memory of the last time (ids.py), passes its `message_id` and `wall`, the time it
    carries: 0.0 is undated, and never expires."""
    message = json.loads(json.dumps(message))
    bap, bpp = (sender, receiver) if sender_role == "bap" else (receiver, sender)
    made = time.time() if wall is None else wall
    body = {"context": {"networkId": NETWORK, "version": VERSION, "domain": DOMAIN, "action": action,
                        "bapId": bap, "bppId": bpp, "transactionId": transaction_id,
                        "messageId": message_id or str(uuid.uuid4()), "timestamp": iso_now() if wall is None else iso_of(wall),
                        "ttl": f"PT{int(ttl_s if ttl_s is not None else TTL_S)}S", "sentWall": made},
            "message": message}
    return signing.sign(body, key)


def check(body: dict[str, Any], public: str) -> bool:
    return signing.verify(body, public)


def ttl_seconds(ttl: Any, default: float = TTL_S) -> float:
    """Beckn context.ttl (ISO 8601 duration such as PT900S, PT15M, PT1H) in seconds."""
    if not isinstance(ttl, str):
        return default
    import re
    m = re.fullmatch(r"P(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?", ttl)
    if not m or not any(m.groups()):
        return default
    d, h, mi, se = (float(x or 0) for x in m.groups())
    return d * 86400 + h * 3600 + mi * 60 + se


def expired(body: dict[str, Any], now_wall: float | None = None, default_ttl: float = TTL_S) -> float | None:
    """How many seconds past its validity a message is (None while valid, or with no send time). A message that
    waited in a Town inbox while its receiver was off is answered too late to matter: the receiver drops it."""
    ctx = body.get("context") or {}
    sent = ctx.get("sentWall")
    if not isinstance(sent, (int, float)) or sent <= 0:
        return None
    late = (now_wall if now_wall is not None else time.time()) - sent - ttl_seconds(ctx.get("ttl"), default_ttl)
    return late if late > 0 else None


class Refused(str):
    """What a receiver answers about a message it did not act on: a bad signature, a sender it doesn't know, past its time, or nothing it can do with it. It reads like any other
    answer, and ack_of acknowledges it as `rejected`.

    Why it isn't just text: Town's record is the evidence, and `processed` there says the work was done. Every handler used to return such a line and then acknowledge `processed`,
    so a forged message looked like work done. What Town does with each status (nandatown db.ack):
      processed, received, rejected, failed   the message is done: it's never handed out again, and the inbox goes on, so a true rejection can't come back for ever or stop a run
      retryable                               back in the inbox at once. Nothing here uses it: a message that can't be read would come back for ever
    Town's evaluator reads only the quote messages' acks, and counts `rejected` as settled custody, so a run with rejections in it still ends. A repeat of something already done is
    not a rejection: it stays `processed` (a settlement, a payment or an answer delivered twice)."""


class Partly(str):
    """An envelope of readings where some counted and some didn't. The envelope was processed, and the note says how many were refused."""
    refused = 0

    def __new__(cls, text: str, refused: int):
        self = super().__new__(cls, text)
        self.refused = refused
        return self


def of_batch(parts: list[str]) -> str:
    """One answer for one envelope: rejected only if nothing in it counted."""
    text = "; ".join(parts)[:300]
    bad = sum(isinstance(p, Refused) for p in parts)
    return Refused(text) if parts and bad == len(parts) else Partly(text, bad) if bad else text


def ack_of(note: Any, limit: int = 200, field: str = "note") -> tuple[str, dict[str, Any]]:
    """What goes to Town for this handler's answer: the status and the note."""
    out: dict[str, Any] = {field: str(note)[:limit]}
    if isinstance(note, Partly):
        out["refused"] = note.refused
    return ("rejected" if isinstance(note, Refused) else "processed"), out


def kind(action: str) -> str:
    return f"beckn.{action}"


def action_of(kind_str: str) -> str:
    return kind_str.split(".", 1)[1] if kind_str.startswith("beckn.") else kind_str


def dataset_utc_offset_h() -> int:
    """The dataset's local time: Illinois 2023 runs on CDT (-5), the Massachusetts summer on EDT (-4)."""
    from .config import dataset_meta
    return int(dataset_meta()["tz_offset_h"])


def local_from_utc(iso_utc: str, utc_offset_h: int | None = None) -> str:
    """2023-08-24T21:00:00Z -> 2023-08-24 16:00 (the dataset's summer local time)."""
    t = datetime.strptime(iso_utc, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    if utc_offset_h is None:
        from .config import dataset_meta
        name = dataset_meta().get("tz_name")          # a dataset that crosses the clock changes
        if name:
            from zoneinfo import ZoneInfo
            return t.astimezone(ZoneInfo(name)).strftime("%Y-%m-%d %H:%M")
    off = dataset_utc_offset_h() if utc_offset_h is None else utc_offset_h
    return (t + timedelta(hours=off)).strftime("%Y-%m-%d %H:%M")


def utc_from_local(ts_local: str) -> str:
    """2023-08-24 16:00 (dataset local) -> 2023-08-24T21:00:00Z."""
    from .config import dataset_meta
    name = dataset_meta().get("tz_name")              # a dataset that crosses the clock changes
    if name:
        from zoneinfo import ZoneInfo
        t = datetime.strptime(ts_local, "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo(name))
        return t.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:00Z")
    return add_minutes(ts_local, -60 * dataset_utc_offset_h()).replace(" ", "T") + ":00Z"


def add_minutes(ts_local: str, minutes: float) -> str:
    t = datetime.strptime(ts_local, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc).timestamp() + minutes * 60
    return datetime.fromtimestamp(t, timezone.utc).strftime("%Y-%m-%d %H:%M")


def need_from_catalog(resource: dict[str, Any], offer: dict[str, Any]) -> dict[str, Any]:
    """Flatten a DEG DemandFlexNeed resource + DemandFlexBuyOffer into what the community works with."""
    ra = resource["resourceAttributes"]
    start = local_from_utc(ra["intervalPeriod"]["start"])
    slots = []
    for iv in ra["intervals"]:
        vals = {p["type"]: p["values"][0] for p in iv["payloads"]}
        slots.append({"id": iv["id"], "start": add_minutes(start, SLOT_MINUTES * iv["id"]),
                      "kw_requested": vals.get("CAPACITY_REQUESTED", 0.0), "price": vals.get("PRICE", 0.0),
                      "penalty": vals.get("SHORTFALL_PENALTY", 0.0)})
    buyer = [i for i in offer["offerAttributes"]["inputs"] if i["role"] == "buyer"][0]
    return {"event_id": resource["id"], "offer_id": offer["id"], "name": resource["descriptor"]["name"],
            "window_start": slots[0]["start"], "window_end": add_minutes(slots[-1]["start"], SLOT_MINUTES),
            "slots": slots, "currency": buyer["inputs"].get("currency", "USD"),
            "baseline": buyer["inputs"].get("baselineMethodology", {}),
            "commit_by": local_from_utc(offer["validity"]["endDate"]),
            "buyer": buyer["participantId"], "policy": offer["offerAttributes"]["contractAttributes"]["policy"]}
