# Step 1 Results: Weigh Each Sample (programmed)

How it works: [approach.md](approach.md). Measured 2026-11-18. Part A (the programmed arm) is built. Part B
(π0.5) waits for a GPU; part A is its teacher.

**2026-11-19:** run again after a second water tube was added to the supply rack (for Step 3): the same
**60 of 60**, weight error median 0.002 g, max 0.006 g; back in its cell within 1.0 mm and 1.92°. The old
scene's run is `results/evaluation_programmed_one_water_tube.json`.

## In short

**60 of 60 sample tubes weighed and put back** on 20 test seeds (5 per job), with nothing else on the bench
moved.

| Check | Result |
|---|---|
| Weighings that worked (steady reading, within 0.05 g, back in its cell) | **60 of 60** |
| Balance reading against the tube's true weight | median **0.002 g** off, worst 0.006 g |
| Back in its cell | 60 of 60: at most 1.0 mm off the cell's centre (median 0.3), at most 1.9° from upright |
| Other tubes and the racks | none moved more than 0.6 mm |
| Moves refused by the collision check | 0 |
| Smallest gap the arm planned to anything | 3.0 mm (median 7.7 mm per weighing) |
| Time per tube (simulated) | 18 to 23 s (median 20) |
| Wall-clock time per seed on this Mac | about 36 s |

By job: pairs 20 of 20, odd one 15 of 15, triangle 15 of 15, mismatch 10 of 10.

The cells came from the simulator, standing in for Step 0's answer (Step 0 found all 120 of 120 tubes with the
high rack camera). The arm never read a tube's weight or liquid from the simulator: only the balance's display.

## What went wrong first, and the fixes

The first 20-seed run gave **40 of 60**: on some seeds tubes were flung across the bench. Seeds 0 and 1, tried
by hand first, had both worked, so luck hid the problem. Three causes:

1. **Squeezing the round cap too hard.** The fingers were told to close 4 mm past the cap's 35 mm (v1 did this
   on its 29 mm tubes). On the round cap, MuJoCo's contact between the flat pads and the cylinder slowly spun
   the tube and drove it down into the rack's tip seat until the simulation jumped. Closing 2 mm past the cap,
   for 0.5 s instead of 0.8, fixed it. Compared on all 20 seeds: 2 mm flung no tubes; 4 mm still knocked the
   rack on 2 seeds. The 2 mm grip still holds every tube firmly (all 60 back within 1 mm). Because the failure
   is chaotic (a tiny change flips one seed), only the full 20 seeds could tell a fix from luck.
2. **The scene's start.** The scene's tubes start "asleep" (not simulated until touched) about 1 mm into the
   rack's floor and tip seats. The first touch wakes the whole rack, and every tube in it jumps out of the
   overlap at once. Now the scene starts awake and settles for 1.5 s first, as a real bench is at rest before
   the arm starts. (The shared scene code is unchanged: `scene.build(seed, awake=True)` already existed.)
3. **The fingers came down fully open.** The gripper starts fully open (85 mm), but the fingers' turn at each
   cell is chosen for an opening of 45 mm. On 2 seeds the collision check refused the move down (a pad would
   have come 1.5 mm from a neighbouring cap), so nothing was hit. Now the fingers open to 45 mm above the cell
   first.

Also: the balance's raw reading never stops flickering (by about 0.02 g), because the liquid's balls never
lie perfectly still, so "steady" never came. Like a real balance's filter, readings are now averaged over
0.1 s, and steady means those averages agree within 0.01 g for 0.5 s.

## When the bench is not where the layout says (test B)

On the same 20 seeds with the sample rack moved 1.6 to 3.0 cm and the balance 1.0 to 1.9 cm
(`test_scenes.py`), the programmed arm weighs **0 of 60**: it finds no tube where the layout puts the cell
and refuses to move, so nothing is knocked over. It is blind to such changes by design (it uses the layout's
numbers, not its cameras). This is the test where a learned model that steers by its cameras has something
to show ([dataset.md](dataset.md)).

## Which way the fingers close

At each rack cell, 12 turns (every 15°) are checked for the way down with the fingers open and for the fingers
closed on the cap. The one with most room is used, and the same turn puts the tube back. The turns used:
0° (18 tubes), 90° (18), 75° (6), 105° (5), 15° (4), and 45°, 120°, 135°, 150°, 165°.

## Reproduce

```bash
cd v3-centrifuge-loading/step-1-weigh
P="pixi run --manifest-path ../simulation-environment/pixi.toml"
PYTHONPATH=$PWD:$PWD/../simulation-environment $P python -m pytest -q tests        # about 20 s
PYTHONPATH=$PWD:$PWD/../simulation-environment $P python evaluate.py --seeds 20 --workers 4   # about 4 min
```

The run saves every tube's numbers in `results/evaluation_programmed.json`.

## Next

- Demonstrations are recorded ([dataset.md](dataset.md)): 120 training episodes (clean and nudged) for
  SmolVLA, then π0.5.
- The same grip and moves load the rotor in Step 4: the next physics question is whether a tube held only by
  its cap stays lined up at 35° in a hole with 0.9 mm to spare.
