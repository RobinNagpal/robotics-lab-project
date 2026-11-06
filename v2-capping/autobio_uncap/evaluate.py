"""AutoBio's uncapping expert on many seeds, for each variant (one change at a time), into runs/autobio/.

    python -m autobio_uncap.evaluate --seeds 0-9 --variants base end135 tight lift lift135 --workers 2

Variants (``expert.Options``):
- ``base``: AutoBio's expert as it is; the thread ends where AutoBio's thread geometry says (1.68 turns).
- ``end135``: the same, but the thread lets go after 1.35 turns, as AutoBio's fixed 1.4 turns assume.
- ``tight``: end135 with our tight cap (0.6 N m to start turning).
- ``lift``: tight, with our lift test deciding when the cap is off (thread to 1.68 turns).
- ``lift135``: tight, lift test, thread to 1.35 turns.
- ``lift1N``: tight, lift test with its threshold at 1 N (set for these softer arms), thread to 1.68.
- ``loose_lift1N``: AutoBio's loose cap, lift test at 1 N, thread to 1.68 turns.
- ``tight_strong``: tight, thread to 1.35, with grips strong enough on paper for 0.6 N m: friction 2 where
  the fingers touch, both grippers told to close fully (0.002).
- ``tight_strong_lift1N``: the same with the lift test at 1 N and the thread to 1.68 turns.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

from .expert import Expert, Options

OUT = Path(__file__).resolve().parents[1] / "runs" / "autobio"
VARIANTS = {
    "base": Options(),
    "end135": Options(turns=1.35),
    "tight": Options(turns=1.35, tight=True),
    "lift": Options(tight=True, lift_test=True),
    "lift135": Options(turns=1.35, tight=True, lift_test=True),
    "lift1N": Options(tight=True, lift_test=True, free_force=1.0),
    "loose_lift1N": Options(lift_test=True, free_force=1.0),
    "tight_strong": Options(turns=1.35, tight=True, friction=2.0, hold_grip=0.002, cap_grip=0.002),
    "tight_strong_lift1N": Options(
        tight=True, friction=2.0, hold_grip=0.002, cap_grip=0.002, lift_test=True, free_force=1.0
    ),
}
_EXPERTS: dict[str, Expert] = {}


def _one(job: tuple[str, int]) -> dict:
    name, seed = job
    if name not in _EXPERTS:
        _EXPERTS[name] = Expert(VARIANTS[name])
    t0 = time.perf_counter()
    r = asdict(_EXPERTS[name].run(seed))
    r["variant"], r["wall_s"] = name, round(time.perf_counter() - t0, 1)
    return r


def seeds(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def summary(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "runs": n,
        "their_check": sum(r["ok_their_check"] for r in rows),
        "cap_off": sum(r["cap_off"] for r in rows),
        "nothing_dropped": sum(r["nothing_dropped"] for r in rows),
        "off_and_nothing_dropped": sum(r["cap_off"] and r["nothing_dropped"] for r in rows),
        "ik_errors": sum(r["error"] is not None for r in rows),
        "turned": [r["turned"] for r in rows],
        "tube_slip_deg_max": max(r["tube_slip_deg"] for r in rows),
        "hold_squeeze_N": [r["hold_squeeze_N"] for r in rows],
        "rounds": [len(r["rounds"]) for r in rows],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="AutoBio's uncapping expert on many seeds.")
    p.add_argument("--seeds", default="0-9")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS))
    p.add_argument("--workers", type=int, default=2)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(v, s) for v in a.variants for s in seeds(a.seeds)]
    with get_context("spawn").Pool(a.workers) as pool:
        rows = pool.map(_one, jobs, chunksize=1)
    for v in a.variants:
        mine = [r for r in rows if r["variant"] == v]
        (OUT / f"expert_{v}.json").write_text(json.dumps({"summary": summary(mine), "runs": mine}, indent=1))
        s = summary(mine)
        print(
            f"{v:8s} check {s['their_check']}/{s['runs']}  off {s['cap_off']}  nothing dropped "
            f"{s['nothing_dropped']}  off+held {s['off_and_nothing_dropped']}  IK errors {s['ik_errors']}  "
            f"slip max {s['tube_slip_deg_max']} deg  rounds {s['rounds']}  turned {s['turned']}"
        )


if __name__ == "__main__":
    main()
