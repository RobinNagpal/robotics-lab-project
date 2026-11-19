"""Print what is in the scene: the job the seed chose, every tube's weight, sizes, and how fast it runs.

python -m centrifuge_scene.info --seed 0
"""

from __future__ import annotations

import argparse
import itertools
import time

import mujoco

from . import grading, liquid, scene
from .settings import BALANCE_TOLERANCE, LIQUID, ROTOR, TUBE
from .tubes import inner_volume


def best_load(masses: dict[str, float]) -> tuple[float, dict[int, str]]:
    """The lowest imbalance any placement of these tubes reaches, with no balance tubes. For the printout."""
    best = (float("inf"), {})
    for holes in itertools.permutations(range(ROTOR.slots), len(masses)):
        load = dict(zip(holes, masses, strict=True))
        value = grading.imbalance({k: masses[n] for k, n in load.items()})
        best = min(best, (value, load), key=lambda t: t[0])
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    start = time.time()
    model, truth = scene.build(args.seed)
    print(f"built in {time.time() - start:.1f} s: {model.nbody} bodies, {model.ngeom} geoms, {model.nv} dofs")
    data = mujoco.MjData(model)
    scene.reset(model, data)
    start = time.time()
    scene.settle(model, data, 1.0)
    print(f"1 s of simulation took {time.time() - start:.1f} s; {data.nbody_awake} bodies awake\n")

    print(f"job: {truth.scenario}")
    print(f"{'tube':8} {'rack':12} {'cell':>4} {'cap':>4} {'balls':>6} {'ml':>6} {'weighs g':>9}")
    for name in truth.tubes:
        rack, cell = truth.cells[name]
        balls = len(liquid.balls_in(model, data, TUBE, name))
        grams = grading.tube_mass(model, data, truth, name) * 1000
        ml = balls * LIQUID.ball_mass * 1000
        print(
            f"{name:8} {rack:12} {cell:4d} {'yes' if truth.capped[name] else 'no':>4} "
            f"{balls:6d} {ml:6.1f} {grams:9.2f}"
        )
    masses = {n: s.mass for n, s in truth.samples.items()}
    value, load = best_load(masses)
    verdict = "no balance tube needed" if value <= BALANCE_TOLERANCE else "needs balance tube(s)"
    print(f"\nsamples alone, best placement: {value * 1000:.2f} g out of balance -> {verdict}")
    print(f"  holes: {dict(sorted(load.items()))}")
    print(
        f"\n50 ml tube: {inner_volume(TUBE, TUBE.height) * 1e6:.1f} ml inside, "
        f"holds {liquid.capacity()} balls;"
        f" one ball weighs {LIQUID.ball_mass * 1000:.3f} g"
    )


if __name__ == "__main__":
    main()
