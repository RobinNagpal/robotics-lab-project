"""The secret key that signs the safety log and the model registry.

A hash chain shows that an entry was changed, but someone who knows the method can change an entry and
recompute every hash after it. Keying the hashes with a secret (HMAC-SHA256) closes that: without the key, no
valid entry can be written. The same goes for the model registry, which says which model files may run and
with which settings.

The key is 32 random bytes in ``keys/safety.key`` next to this file, made the first time it is needed and
never committed (``keys/`` is in ``.gitignore``). A real system would keep it in the robot controller's secure
storage, where the software that writes the log can use it but cannot read it out.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path

KEY_FILE = Path(__file__).resolve().parent / "keys" / "safety.key"


def load(path: str | Path = KEY_FILE) -> bytes:
    """The key, made on first use (readable by this user only)."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(os.urandom(32))
    key = path.read_bytes()
    if len(key) < 32:
        raise ValueError(f"the key in {path} is too short")
    return key


def sign(key: bytes, text: str) -> str:
    """HMAC-SHA256 of ``text`` under ``key``, as hex."""
    return hmac.new(key, text.encode(), hashlib.sha256).hexdigest()


def same(a: str, b: str) -> bool:
    """Compare two signatures in constant time."""
    return hmac.compare_digest(a, b)
