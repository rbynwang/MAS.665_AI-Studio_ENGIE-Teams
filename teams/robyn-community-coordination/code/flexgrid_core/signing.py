"""Ed25519 signing over canonical JSON. Every envelope, dispatch and receipt is signed."""
from __future__ import annotations

import hashlib
import json
import os
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
    path = Path(path)
    if path.exists():
        return Ed25519PrivateKey.from_private_bytes(bytes.fromhex(path.read_text().strip()))
    key = Ed25519PrivateKey.generate()
    raw = key.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                            serialization.NoEncryption())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(raw.hex() + "\n")
    os.chmod(path, 0o600)
    return key


def public_hex(key: Ed25519PrivateKey) -> str:
    return key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw).hex()


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
