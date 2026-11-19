"""Print the tables in results.md from results/part_*.json."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

R = Path(__file__).parent / "results"


def load(part):
    p = R / f"part_{part}.json"
    return json.loads(p.read_text()) if p.exists() else None


def windows(turns, step):
    """[0, 5, 10, 30] -> '0-10, 30'."""
    if not turns:
        return "none"
    out, start, prev = [], turns[0], turns[0]
    for t in turns[1:] + [None]:
        if t is None or t != prev + step:
            out.append(f"{start}" if start == prev else f"{start}-{prev}")
            if t is not None:
                start = t
        if t is not None:
            prev = t
    return ", ".join(out)


a = load("a")
if a:
    print("## A. Rack cells (axial grip on cap / rim), 8 seeds")
    n_ok = sum(1 for r in a if r["turns_ok"])
    print(f"tubes with at least one clear turn: {n_ok} of {len(a)}")
    counts = Counter(len(r["turns_ok"]) for r in a)
    print("clear turns (of 12) per tube:", dict(sorted(counts.items())))
    print("worst best-gap: ", min(r["best"]["gap_mm"] for r in a), "mm")
    for r in a:
        if not r["turns_ok"]:
            print("  FAIL", r)

b = load("b")
if b:
    print("\n## B. Balance holder")
    for r in b:
        print(f"  {r['tube']}: turns ok {windows(r['turns_ok'], 15)}; best gap {r['best']['gap_mm']} mm")

c = load("c")
if c:
    print("\n## C. Rotor holes (other five holes full), turn 0 = fingers close along the rim")
    for r in sorted(c, key=lambda r: (r["capped"], r["depth_mm"], r["hole"])):
        kind = (
            ("capped, pinch {:.0f} mm under top".format(r["depth_mm"]))
            if r["capped"]
            else "open balance tube"
        )
        print(
            f"  hole {r['hole']} {kind:32s}: clear turns {windows(r['turns_ok'], 5):20s} "
            f"best {r['best']['gap_mm']:5.1f} mm  closest: {r['best']['what']} {r['best']['note']}"
        )
    # The turns that work in every hole
    groups = defaultdict(list)
    for r in c:
        groups[(r["capped"], r["depth_mm"])].append(set(r["turns_ok"]))
    for key, sets in groups.items():
        common = sorted(set.intersection(*sets))
        print(f"  turns clear in all six holes {key}: {windows(common, 5)}")

d = load("d")
if d:
    print("\n## D. Pour (water tube held from the side, tipped 40-130 deg)")
    ok = [r for r in d if r["ok"]]
    print(f"clear set-ups: {len(ok)} of {len(d)}")
    by = defaultdict(list)
    for r in ok:
        by[(r["below_rim_mm"], r["approach"])].append(r["pour_yaw"])
    for k, v in sorted(by.items()):
        best = max((r for r in ok if (r["below_rim_mm"], r["approach"]) == k), key=lambda r: r["gap_mm"])
        print(
            f"  grip {k[0]:.0f} mm under rim, fingers from {k[1]:4d} deg: "
            f"pour directions {v}; best gap {best['gap_mm']} mm"
        )
    fails = Counter((r["note"] or r["what"]) for r in d if not r["ok"])
    print("  why the others fail:", fails.most_common(6))

e = load("e")
if e:
    print("\n## E. Water tube picked from its rack from the side")
    for h in sorted({r["below_rim_mm"] for r in e}):
        rows = [r for r in e if r["below_rim_mm"] == h]
        n = sum(1 for r in rows if r["approaches_ok"])
        print(
            f"  {h:.0f} mm under rim: seeds with a clear approach {n} of {len(rows)}; "
            f"approaches per seed {[len(r['approaches_ok']) for r in rows]}"
        )

f = load("f")
if f:
    print("\n## F. Lid")
    for r in f["rows"]:
        print("  ", r)

g = load("g")
if g:
    print("\n## G. Carries")
    for r in g:
        print(
            f"  seed {r['seed']} {r['tube']:8s} {r['move']:18s} ok={r['ok']} via={r.get('via_cm', '-')} "
            f"gap={r.get('gap_mm', '-')} {r.get('what', '')} {r.get('note', '')}"
        )

h = load("h")
if h:
    print("\n## H. Rotor holes by neighbours (opposite hole full too); release opening = hold + 4 mm")
    for r in sorted(h, key=lambda r: (r["lid_deg"], r["held"], r["neighbours"], r["hole"])):
        print(
            f"  lid {r['lid_deg']:5.1f}  {r['held']:6s} {r['neighbours']:11s} hole {r['hole']}: "
            f"clear turns {windows(r['turns_ok'], 5):22s} best {r['best']['gap_mm']:5.1f} mm "
            f"{r['best']['what']} {r['best']['note']}"
        )

lid = load("l")
if lid:
    print("\n## L. Lid pinched at its front edge, swung shut from its resting angle")
    for r in lid:
        print("  ", r)
