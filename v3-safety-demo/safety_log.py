"""The safety log: every safety decision, each entry chained to the one before by a hash.

Each entry is one line of JSON. Besides its own fields (time, frame number, model version, score, zone,
action,
reason) it carries

- ``seq``: its place in the log (0, 1, 2, ...),
- ``prev``: the hash of the entry before it (64 zeros for the first),
- ``hash``: the SHA-256 of the entry itself, ``prev`` included.

So changing any entry changes its hash and breaks the next entry's ``prev``; deleting or inserting one breaks
``seq`` and ``prev``. What the chain alone cannot show is the end being cut off, or every hash after an edit
being recomputed by someone who knows the method. For that the last hash (the "head") is kept somewhere else
too: ``verify(entries, head=...)`` checks it. (A real system would also sign the head, or key the hashes
with a secret; not done here.)

    log = SafetyLog("run.jsonl")
    log.append(kind="detector", t=1.2, frame=12, model="perfect-1", score=1.0, zone="clear", action="run")
    problems = verify(read("run.jsonl"), head=log.head)    # [] when nothing was changed
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

GENESIS = "0" * 64


def _plain(v: Any) -> Any:
    """Plain JSON values: numpy numbers and arrays become Python ones; NaN and infinity become None."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [_plain(x) for x in v]
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, np.integer)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return v if math.isfinite(v) else None
    if v is None or isinstance(v, str):
        return v
    return str(v)


def digest(entry: dict) -> str:
    """The SHA-256 of an entry, over every field but its own ``hash``, in a fixed key order."""
    body = {k: v for k, v in entry.items() if k != "hash"}
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


class SafetyLog:
    """An append-only log, kept in memory and, if ``path`` is given, written line by line as it grows."""

    def __init__(self, path: str | Path | None = None):
        self.entries: list[dict] = []
        self.head = GENESIS
        self.path = Path(path) if path else None
        self._file = None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self.path.open("w")

    def append(self, **fields: Any) -> dict:
        entry = {"seq": len(self.entries), **_plain(fields), "prev": self.head}
        entry["hash"] = digest(entry)
        self.head = entry["hash"]
        self.entries.append(entry)
        if self._file:
            self._file.write(json.dumps(entry, sort_keys=True, separators=(",", ":")) + "\n")
            self._file.flush()
        return entry

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def __enter__(self) -> SafetyLog:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def read(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def verify(entries: list[dict], head: str | None = None) -> list[str]:
    """What is wrong with the chain, in plain words; an empty list when every entry is as it was written."""
    problems = []
    prev = GENESIS
    for i, e in enumerate(entries):
        if e.get("seq") != i:
            problems.append(f"entry {i}: numbered {e.get('seq')} (an entry before it was deleted or added)")
        if e.get("prev") != prev:
            problems.append(
                f"entry {i}: does not follow the entry before it (one was changed, deleted or added)"
            )
        if digest(e) != e.get("hash"):
            problems.append(f"entry {i}: its contents were changed after it was written")
        prev = e.get("hash")
    if head is not None and prev != head:
        problems.append(
            "the log's last entry is not the one recorded at the end of the run (cut short or replaced)"
        )
    return problems
