"""Step 2A: plan the load. From the samples' weights: which hole each tube goes in, how heavy each balance
tube must be, and the order to load them.

Pure arithmetic: no simulator, no camera. Its input is what Step 1 read off the balance.

The balance number. Spinning, each tube pulls outwards with a force proportional to its mass, and all six
holes are at the same radius. Drawn as arrows, one per loaded hole, each as long as its tube's weight and
pointing out through its hole, the rotor is balanced when the arrows add up to nothing. The length of their
sum is the imbalance, in grams (see the simulation's grading.imbalance, which the examiner uses).

The loading rule (grasp-plan/docs/results.md): the gripper's finger links need an empty hole beside the hole
a tube goes into. So every tube needs at least one empty neighbouring hole when it goes in, and an open
balance tube (whose grip has less room) should go in while both neighbours are empty.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field

SLOTS = 6
# The rotor counts as balanced when the arrows add up to at most this (settings.BALANCE_TOLERANCE).
TOLERANCE_G = 1.0
# Two empty tubes in the supply rack can become balance tubes.
MAX_BALANCE_TUBES = 2
# A full rotor cannot be loaded: the last tube would have both neighbours full.
MAX_TUBES = SLOTS - 1
# An empty 50 ml tube without a cap weighs 11.5 g (each one up to 0.15 g more or less). Step 3 weighs the real
# one before filling it; the plan only needs to know the target is possible.
EMPTY_TUBE_G = 11.5
TUBE_SPREAD_G = 0.15
# The most water an open balance tube may hold: 106 balls (34.6 g) stay in it at 35 degrees in the rotor
# (the simulation's test of exactly that).
MAX_WATER_G = 34.6
# Step 3 fills a balance tube to within this of its target.
FILL_ERROR_G = 0.5

DIRECTIONS = [(math.cos(2 * math.pi * k / SLOTS), math.sin(2 * math.pi * k / SLOTS)) for k in range(SLOTS)]


def imbalance(masses: dict[int, float]) -> float:
    """The length of the sum of the tubes' outward pulls, from {hole: grams}."""
    x = sum(m * DIRECTIONS[k][0] for k, m in masses.items())
    y = sum(m * DIRECTIONS[k][1] for k, m in masses.items())
    return math.hypot(x, y)


def neighbours(hole: int) -> tuple[int, int]:
    return (hole - 1) % SLOTS, (hole + 1) % SLOTS


@dataclass(frozen=True)
class Tube:
    hole: int
    name: str  # a sample's name, or "balance A" / "balance B"
    grams: float  # a sample's weight, or the weight a balance tube must be filled to
    balance: bool


@dataclass
class Plan:
    tubes: list[Tube]  # in loading order
    imbalance_g: float  # if every balance tube hits its target exactly
    worst_g: float  # if every balance tube is off by the most Step 3 allows, the worst way
    crowded: list[str] = field(default_factory=list)  # tubes that go in with one neighbour already full

    @property
    def balance_tubes(self) -> list[Tube]:
        return [t for t in self.tubes if t.balance]

    def holes(self) -> dict[int, str]:
        return {t.hole: t.name for t in self.tubes}

    def describe(self) -> str:
        """The plan in plain words, in loading order."""
        lines = []
        for i, t in enumerate(self.tubes, 1):
            what = f"fill an empty tube to {t.grams:.2f} g" if t.balance else f"{t.grams:.2f} g"
            lines.append(f"{i}. {t.name} ({what}) into hole {t.hole}")
        lines.append(
            f"Balance: {self.imbalance_g:.2f} g out (limit {TOLERANCE_G:.1f} g); "
            f"at most {self.worst_g:.2f} g if each balance tube is {FILL_ERROR_G} g off."
        )
        return "\n".join(lines)


def plan_load(weights: dict[str, float]) -> Plan | None:
    """The best plan for samples weighing ``weights`` ({name: grams}), or None if there is none.

    Fewest balance tubes first. Among plans with that many: balance tubes that go in with both neighbours
    empty, then the lowest imbalance, then the fewest tubes going in beside a full hole, then the lowest hole
    numbers.
    """
    names = sorted(weights)
    lo = EMPTY_TUBE_G + TUBE_SPREAD_G
    hi = EMPTY_TUBE_G - TUBE_SPREAD_G + MAX_WATER_G
    for n_balance in range(MAX_BALANCE_TUBES + 1):
        if len(names) + n_balance > MAX_TUBES:
            break
        best, best_key = None, None
        for sample_holes in itertools.permutations(range(SLOTS), len(names)):
            masses = {k: weights[n] for k, n in zip(sample_holes, names, strict=True)}
            free = [k for k in range(SLOTS) if k not in masses]
            for balance_holes in itertools.combinations(free, n_balance):
                fill = _balance_weights(masses, balance_holes, lo, hi)
                if fill is None:
                    continue
                full = masses | dict(zip(balance_holes, fill, strict=True))
                off = imbalance(full)
                if off > TOLERANCE_G:
                    continue
                order = load_order(set(full), set(balance_holes))
                if order is None:
                    continue
                sequence, balance_crowded, crowded = order
                key = (balance_crowded, round(off, 3), crowded, tuple(sample_holes), balance_holes)
                if best_key is None or key < best_key:
                    labels = dict(zip(balance_holes, ("balance A", "balance B"), strict=False))
                    by_hole = {k: n for k, n in zip(sample_holes, names, strict=True)} | labels
                    tubes = [Tube(k, by_hole[k], full[k], k in labels) for k in sequence]
                    best_key = key
                    near_full = crowded_set(sequence)
                    best = Plan(
                        tubes,
                        off,
                        _worst(full, balance_holes),
                        [t.name for t in tubes if t.hole in near_full],
                    )
        if best is not None:
            return best
    return None


def _balance_weights(masses, holes, lo, hi) -> list[float] | None:
    """Weights in [lo, hi] for balance tubes in ``holes`` that cancel the samples' pull as best they can."""
    rx = sum(m * DIRECTIONS[k][0] for k, m in masses.items())
    ry = sum(m * DIRECTIONS[k][1] for k, m in masses.items())
    if not holes:
        return []
    if len(holes) == 1:
        ex, ey = DIRECTIONS[holes[0]]
        return [min(hi, max(lo, -(rx * ex + ry * ey)))]
    (ax, ay), (bx, by) = DIRECTIONS[holes[0]], DIRECTIONS[holes[1]]

    def residual(w1, w2):
        return math.hypot(rx + w1 * ax + w2 * bx, ry + w1 * ay + w2 * by)

    def best_other(w_fixed, fixed_first):
        # With one weight fixed, the other that leaves the least, kept within [lo, hi].
        if fixed_first:
            px, py, ex, ey = rx + w_fixed * ax, ry + w_fixed * ay, bx, by
        else:
            px, py, ex, ey = rx + w_fixed * bx, ry + w_fixed * by, ax, ay
        return min(hi, max(lo, -(px * ex + py * ey)))

    candidates = []
    det = ax * by - ay * bx
    if abs(det) > 1e-9:  # not opposite each other: one exact answer
        w1 = (-rx * by + ry * bx) / det
        w2 = (-ay * -rx + ax * -ry) / det
        if lo <= w1 <= hi and lo <= w2 <= hi:
            candidates.append((w1, w2))
    for w in (lo, hi):  # otherwise the best is on an edge of the allowed range
        candidates.append((w, best_other(w, True)))
        candidates.append((best_other(w, False), w))
    w1, w2 = min(candidates, key=lambda c: residual(*c))
    return [w1, w2]


def _worst(full: dict[int, float], balance_holes) -> float:
    """The imbalance if each balance tube ends up FILL_ERROR_G over or under its target, the worst way."""
    worst = 0.0
    for signs in itertools.product((-1, 1), repeat=len(balance_holes)):
        changed = dict(full)
        for k, s in zip(balance_holes, signs, strict=True):
            changed[k] += s * FILL_ERROR_G
        worst = max(worst, imbalance(changed))
    return worst


def load_order(holes: set[int], balance: set[int]) -> tuple[list[int], int, int] | None:
    """The order to load ``holes`` so every tube has an empty neighbour when it goes in, or None.

    Returns (order, balance tubes that go in beside a full hole, all tubes that do). Lower is better for both.
    """
    best = None
    for order in itertools.permutations(sorted(holes)):
        loaded: set[int] = set()
        balance_crowded = crowded = 0
        ok = True
        for k in order:
            full_neighbours = sum(n in loaded for n in neighbours(k))
            if full_neighbours == 2:
                ok = False
                break
            if full_neighbours == 1:
                crowded += 1
                balance_crowded += k in balance
            loaded.add(k)
        if not ok:
            continue
        # Balance tubes first among equals: Step 3 fills one in the holder, and it goes straight to the rotor.
        first_sample = next((i for i, k in enumerate(order) if k not in balance), len(order))
        late_balance = sum(i > first_sample for i, k in enumerate(order) if k in balance)
        key = (balance_crowded, crowded, late_balance, order)
        if best is None or key < best[0]:
            best = (key, list(order), balance_crowded, crowded)
    return None if best is None else (best[1], best[2], best[3])


def crowded_set(order: list[int]) -> set[int]:
    """The holes in ``order`` whose tube goes in with one neighbour already full."""
    loaded: set[int] = set()
    out = set()
    for k in order:
        if any(n in loaded for n in neighbours(k)):
            out.add(k)
        loaded.add(k)
    return out
