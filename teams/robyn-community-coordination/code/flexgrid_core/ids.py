"""Message ids: one logical message, one id. It's NANDA Town's rule, and what a retry rests on.

Town takes an id once. The same id with the same sender, receiver, kind and body is a retry: Town answers `replay` and delivers nothing new. The same id with anything else is
refused with a 409 and logged as identity_reuse_rejected. It delivers at least once, under leases, so a sender that dies, or never hears the answer, has to be able to say the
same thing again and have it count once.

On 6 Nov 2026 a replayed day reused batch ids with new bodies, Town refused them, and the sellers went unpaid. The first fix gave every send a fresh uuid. That's the opposite of
Town's rule: a node that restarts answers again under a new id and Town takes two answers, which is its own negative control, quote-amnesia-fresh-ids. So an id says which message
this is, and never when it happened to be sent:

    scope:sender:receiver:kind:key...[:seal]

  scope   what keeps one play of a day apart from the next.
            a message nobody asked for (a batch of readings, a contract, a payment): the sender's play tag, new at each /demo/start or /demo/reset (new_play), kept through
            a restart by keep.py. A sender with nothing kept starts on a new one: what it says now can't be mistaken for what it said before it lost its memory.
            an answer: what it answers (asked). The ask already carries its sender's play, so an answer needs nothing of its own, and a node that remembers nothing at all
            still names the same answer. An ask sent again is a new message with a new messageId, so the answer goes again too. An ask Town hands out again is the same one.
  key     what the message is: the event, contract, slot set, call id, round.
  seal    on everything but an answer: a tag only the sender can make (sealed). Town takes an id once, whoever sends it first, so an id somebody else can work out is one they can
          take first, and the real message gets a 409. A node sees the coordinator's play in each id it's sent, and every other part of the id is public, so on 7 Nov one home could
          block another's payment. An answer needs no seal: its scope comes from the messageId inside the ask, a uuid only the asker and this receiver ever see, and it has to come
          out the same from a node that remembers nothing.

The same id goes only with the same body, and a body made again differs in its time and signature. So:
  a retry                    the sender isn't sure it went. Same id, and town.Sent puts the first body back, so Town answers replay. Sent writes before Town hears it, so this
                             holds across a restart, and an answer lost in a crash still comes to replay
  a resend round             the receiver may have lost it, or was never told. A new logical message: its round is in the key, and Town delivers it (a chase, a commitment sent
                             again, a call for offers sent again)
  a statement of the moment  what it says is what it is, so a digest of it is in the key: the same words again are a retry, and new words are a new message (a shortfall report,
                             "readings are in")
A body made again that says something else under the same id is neither: it goes as it is, Town refuses it, and that stays loud (town.TownSession.send).

Offline runs carry no ids: the simulator's bus drops them.
"""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

from . import signing


def new_play() -> str:
    """A new tag: a sender's first, and at each /demo/start or /demo/reset. Nobody else needs to know it: every id names its sender, so each sender's tag only has to change."""
    return uuid.uuid4().hex[:8]


def digest(*parts: Any) -> str:
    """A short stable name for what a message says. Nothing here is secret: it keeps ids short and tells two different things apart."""
    return hashlib.sha1(json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[:10]


def message_id(scope: str, sender: str, to: str, kind: str, *key: Any) -> str:
    return ":".join(str(x) for x in (scope, sender, to, kind.removeprefix("beckn."), *key))


def sealed(key: Any, scope: str, sender: str, to: str, kind: str, *more: Any) -> str:
    """The id of a message nobody asked for. It's a pure function of the sender's key file, its play and what the message is, so a retry and a restart name it again (keep.py keeps the
    play) and nobody else can name it first. key None: a message that needs no seal, as an answer doesn't."""
    plain = message_id(scope, sender, to, kind, *more)
    return f"{plain}:{signing.seal(key, plain)}" if key is not None else plain


def asked(body: Any) -> str:
    """The scope of an answer: the message it answers. Its messageId stays the same however often Town hands it out, and is new when the asker asks again."""
    ctx = body.get("context") if isinstance(body, dict) else None
    made = ctx.get("messageId") if isinstance(ctx, dict) else None
    return "a" + digest(made if made else body)


def readings(bodies: list[Any]) -> str:
    """The name of a batch: the readings in it, whatever order they were held in. The same slots sent again after a restart are the same batch."""
    def one(b: Any) -> tuple:
        r = ((b.get("message") or {}).get("performance") or {}) if isinstance(b, dict) else {}
        return str(r.get("contract_id")), str(r.get("slot_id"))
    return digest(sorted(one(b) for b in bodies))
