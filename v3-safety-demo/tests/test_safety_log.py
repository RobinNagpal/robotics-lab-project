import json
import math

import numpy as np

from safety_log import GENESIS, SafetyLog, read, verify


def make(path=None, n=5):
    log = SafetyLog(path)
    for i in range(n):
        log.append(
            kind="detector", t=0.1 * i, frame=i + 1, model="perfect-1", score=1.0, zone="clear", action="run"
        )
    return log


def test_untouched_log_verifies(tmp_path):
    log = make(tmp_path / "a.jsonl")
    log.close()
    entries = read(tmp_path / "a.jsonl")
    assert entries == log.entries
    assert entries[0]["prev"] == GENESIS
    assert verify(entries, head=log.head) == []


def test_edited_entry_is_found():
    log = make()
    entries = json.loads(json.dumps(log.entries))
    entries[2]["action"] = "slow"
    problems = verify(entries, head=log.head)
    assert any("entry 2" in p and "changed" in p for p in problems)


def test_edit_with_recomputed_hashes_is_found_by_the_head():
    from safety_log import digest

    log = make()
    entries = json.loads(json.dumps(log.entries))
    entries[1]["zone"] = "stop"
    prev = entries[0]["hash"]
    for e in entries[1:]:  # someone who knows the method rebuilds the chain
        e["prev"] = prev
        e["hash"] = digest(e)
        prev = e["hash"]
    assert verify(entries) == []  # the chain alone cannot see it
    assert verify(entries, head=log.head)  # the head kept elsewhere does


def test_deleted_entry_is_found():
    log = make()
    entries = [e for e in log.entries if e["seq"] != 2]
    problems = verify(entries, head=log.head)
    assert any("entry 2" in p for p in problems)


def test_inserted_entry_is_found():
    log = make()
    entries = list(log.entries)
    fake = dict(entries[1], action="run", seq=2)
    entries.insert(2, fake)
    assert verify(entries, head=log.head)


def test_cut_short_is_found_by_the_head():
    log = make()
    assert verify(log.entries[:-1]) == []
    assert verify(log.entries[:-1], head=log.head)


def test_numpy_and_nan_values_are_plain():
    log = SafetyLog()
    e = log.append(
        t=np.float64(1.5), frame=np.int64(3), score=float("nan"), pts=np.zeros(2), ok=np.bool_(True)
    )
    assert (
        e["t"] == 1.5
        and e["frame"] == 3
        and e["score"] is None
        and e["pts"] == [0.0, 0.0]
        and e["ok"] is True
    )
    assert verify(log.entries) == []
    assert not any(isinstance(v, float) and math.isnan(v) for v in e.values())


def test_keyed_log_needs_the_key(tmp_path):
    """A log written with a key: verifies with it; recomputing the hashes without the key is caught."""
    import hashlib
    import json as _json

    key, other = b"k" * 32, b"o" * 32
    log = SafetyLog(tmp_path / "keyed.jsonl", key=key)
    for i in range(5):
        log.append(kind="detector", t=i * 0.1, zone="clear", action="run")
    log.close()
    entries = read(tmp_path / "keyed.jsonl")
    assert verify(entries, head=log.head, key=key) == []
    assert verify(entries, head=log.head, key=other)  # the wrong key: nothing verifies
    # Someone without the key edits entry 2 and rebuilds every plain SHA-256 after it.
    entries[2]["action"] = "stop"
    prev = entries[1]["hash"]
    for e in entries[2:]:
        e["prev"] = prev
        body = {k: v for k, v in e.items() if k != "hash"}
        e["hash"] = hashlib.sha256(
            _json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        prev = e["hash"]
    assert verify(entries, key=key)
