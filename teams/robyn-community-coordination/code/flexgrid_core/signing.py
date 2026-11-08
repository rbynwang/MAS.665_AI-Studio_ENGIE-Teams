"""Ed25519 signing over canonical JSON. Every envelope, dispatch and receipt is signed."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(obj: Any) -> str:
    return "sha256:" + hashlib.sha256(canonical(obj).encode()).hexdigest()


def load_or_create(path: Path) -> Ed25519PrivateKey:
    """The key in this file, made if there's none. There's only ever one, however many processes get here at once: on a fresh config both buyers did, each made a key, and the last
    to write won, so one ran on a key that wasn't on disk and its first restart changed it. A new key takes the name with a hard link, which fails if the name is taken, and
    whoever loses reads the winner's. It's whole and mode 600 before anyone can read it under that name."""
    path = Path(path)
    if not path.exists():
        _make_key(path)
    return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(path.read_text().strip()))


def _make_key(path: Path) -> None:
    raw = Ed25519PrivateKey.generate().private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")      # mode 600 from the start
    try:
        with os.fdopen(fd, "w") as f:
            f.write(raw.hex() + "\n")
            f.flush()
            os.fsync(f.fileno())
        try:
            os.link(tmp, path)
        except FileExistsError:
            pass
    finally:
        os.unlink(tmp)


def public_hex(key: Ed25519PrivateKey) -> str:
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()


def seal(key: Ed25519PrivateKey, text: str, size: int = 16) -> str:
    """A short tag on this text: the same every time from this key, and nobody without the key can work it out. A hash of the key stands in for the key, so a tag can't say anything
    about it. It comes back with the key after a restart, and there's nothing new to keep or to deploy."""
    secret = hashlib.sha256(b"flexgrid ids:" + key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption())).digest()
    return hmac.new(secret, text.encode(), hashlib.sha256).hexdigest()[:size]


def sign(payload: dict[str, Any], key: Ed25519PrivateKey) -> dict[str, Any]:
    body = {k: v for k, v in payload.items() if k != "sig"}
    body["signer"] = public_hex(key)
    return {**body, "sig": key.sign(canonical(body).encode()).hex()}


def verify(payload: dict[str, Any], public: str) -> bool:
    if not payload or "sig" not in payload or not public:
        return False
    body = {k: v for k, v in payload.items() if k != "sig"}
    if body.get("signer") != public:
        return False
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(public)).verify(
            bytes.fromhex(payload["sig"]), canonical(body).encode())
        return True
    except (InvalidSignature, ValueError):
        return False
