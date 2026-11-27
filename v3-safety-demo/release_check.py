"""May a new detector version replace the current one? The update test, on the same frozen test pictures.

The rule is fixed before the new version is evaluated (this file was committed before v2's results existed): a
new version is released only if, on the 3,000 test pictures,

1. it misses no more hands in the stop zone than the current version;
2. it misses no more hands in any zone than the current version;
3. its false alarms on clear pictures are at most 1 percentage point above the current version's;
4. under every condition (lighting, hand colour, sleeve, hand size, hand height, how much of the hand shows),
   it misses at most 1 more hand than the current version (one picture of slack per condition, so chance alone
   does not block a better model; a real step backwards in one condition still does).

5. with nobody there, over the recorded test jobs (``nobody_scan.py``, the same pictures for both versions),
   it raises no more false alarms than the current version. Added after v2's closed-loop run (a false alarm
   on the robot's own gripper held the arm for good: docs/integration.md), and before v3 was evaluated.

    python release_check.py results/evaluation_v1.json results/evaluation_v2.json
    python release_check.py results/evaluation_v2.json results/evaluation_v3.json \
        --nobody results/nobody_test_v2.json results/nobody_test_v3.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

MISSED = "missed (hand in a zone, less serious zone chosen)"
MISSED_STOP = "missed stop (hand in stop zone, arm not stopped)"
FALSE_CLEAR = "false alarm when clear"
FALSE_SLACK = 0.01
CONDITION_SLACK = 1


def check(current: dict, new: dict, nobody: tuple[dict, dict] | None = None) -> list[tuple[str, bool, str]]:
    """Each rule: (what, passed, the numbers)."""
    a, b = current["test"], new["test"]
    out = [
        (
            "missed stop-zone hands: no more than now",
            b[MISSED_STOP]["k"] <= a[MISSED_STOP]["k"],
            f"{a[MISSED_STOP]['k']} -> {b[MISSED_STOP]['k']} of {b[MISSED_STOP]['n']}",
        ),
        (
            "missed hands in any zone: no more than now",
            b[MISSED]["k"] <= a[MISSED]["k"],
            f"{a[MISSED]['k']} -> {b[MISSED]['k']} of {b[MISSED]['n']}",
        ),
        (
            "false alarms when clear: at most 1 point more",
            b[FALSE_CLEAR]["rate"] <= a[FALSE_CLEAR]["rate"] + FALSE_SLACK,
            f"{a[FALSE_CLEAR]['rate']:.2%} -> {b[FALSE_CLEAR]['rate']:.2%}",
        ),
    ]
    for group, values in current["test by condition"].items():
        for value, numbers in values.items():
            before = numbers["missed"]["k"]
            after = new["test by condition"].get(group, {}).get(value, {}).get("missed", {}).get("k")
            if after is None:
                out.append((f"{group} = {value}: present", False, "missing from the new evaluation"))
                continue
            if numbers["missed"]["n"] == 0:
                continue
            out.append(
                (
                    f"{group} = {value}: at most {CONDITION_SLACK} more missed",
                    after <= before + CONDITION_SLACK,
                    f"{before} -> {after} of {numbers['missed']['n']}",
                )
            )
    if nobody is not None:
        before, after = (n["false alarms (zone not clear)"] for n in nobody)
        out.append(
            (
                "false alarms with nobody there (recorded test jobs): no more than now",
                after["k"] <= before["k"] and after["n"] == before["n"],
                f"{before['k']} -> {after['k']} of {after['n']}",
            )
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("current", type=Path)
    p.add_argument("new", type=Path)
    p.add_argument("--out", type=Path, default=None, help="write the verdict as JSON")
    p.add_argument("--nobody", type=Path, nargs=2, default=None, help="nobody_scan results: current, new")
    args = p.parse_args()
    nobody = tuple(json.loads(f.read_text()) for f in args.nobody) if args.nobody else None
    rules = check(json.loads(args.current.read_text()), json.loads(args.new.read_text()), nobody)
    for what, passed, numbers in rules:
        print(f"{'pass' if passed else 'FAIL'}  {what}: {numbers}")
    release = all(passed for _, passed, _ in rules)
    print(f"\nrelease: {'yes' if release else 'no'}")
    if args.out:
        verdict = {
            "current": str(args.current),
            "new": str(args.new),
            "release": release,
            "rules": [{"rule": w, "passed": p, "numbers": n} for w, p, n in rules],
        }
        args.out.write_text(json.dumps(verdict, indent=1) + "\n")
    sys.exit(0 if release else 1)


if __name__ == "__main__":
    main()
