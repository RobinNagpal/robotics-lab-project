# Step 2 Results: Plan the Load

How it works: [approach.md](approach.md). Measured 2026-11-17. Part A (the solver) is built; part B (Gemini
Robotics-ER 2 with the solver as a tool) is not yet.

## The solver on 100 seeds

25 seeds of each job. The solver gets each sample's weight as Step 1 would read it: the true weight plus up to
0.05 g of balance error. Its plan is then graded with the **true** weights (the examiner's side). "After
filling" adds Step 3's allowed error: each balance tube 0.5 g over or under its target at random (200 tries per
seed); "worst case" is both errors the worst way.

| Job | Seeds | Plan found | Right number of balance tubes | Within 1.0 g | Imbalance, median / max | Within 1.0 g after filling | Worst case | Tubes going in beside a full hole |
|---|---|---|---|---|---|---|---|---|
| pairs | 25 | 25 | 25 | 25 | 0.28 / 0.55 g | 100 % | 0.54 g | 1.4 per job (balance tubes: none) |
| odd one | 25 | 25 | 25 | 25 | 0.24 / 0.51 g | 100 % | 0.75 g | 2.0 (balance tubes: none) |
| triangle | 25 | 25 | 25 | 25 | 0.32 / 0.67 g | 100 % | 0.64 g | 0 |
| mismatch | 25 | 25 | 25 | 25 | 0.04 / 0.07 g | 100 % | 0.87 g | 2.0 (balance tubes: none) |

- **Every plan is right**: the right number of balance tubes for its job, within 1.0 g with the true weights, and
  a loading order where every tube has an empty hole beside it.
- **Balance tubes always go in first**, with both neighbours empty (their grip has the least room).
- The balance number agrees with the simulation's own grading code (`grading.imbalance`) on every seed, so both
  count the holes the same way.
- A plan takes 1.5 ms (at most 5 ms).

## What the numbers say

- **Pairs and triangle** are only as balanced as the samples themselves: nearly-equal tubes differ by up to
  about 0.6 g, and nothing can be added to a sample. The solver lays them out so the differences cancel where
  they can (two pairs side by side can add up to more than 1.0 g; turned the right way they do not).
- **One balance tube** cancels only the pull along its own line. In the odd-one job the pair's small
  difference points 60° away from it, so a little is left (median 0.24 g).
- **Two balance tubes** cancel anything exactly (mismatch: 0.04 g, just the reading error). But two fill errors
  can add up: **0.87 g worst case**. Plans with the two balance tubes side by side would halve that, but then
  the second one goes in beside a full hole, where its grip has only 3.5 mm to spare. The solver keeps the grip
  room. So **Step 3 should aim at 0.3 g**, not 0.5 g, when there are two balance tubes.

## Examples (the solver's own words)

Seed 1, odd one:
```
1. balance A (fill an empty tube to 45.09 g) into hole 4
2. sample0 (40.23 g) into hole 0
3. sample2 (45.24 g) into hole 1
4. sample1 (40.53 g) into hole 3
Balance: 0.26 g out (limit 1.0 g); at most 0.56 g if each balance tube is 0.5 g off.
```

Seed 3, mismatch:
```
1. balance A (fill an empty tube to 25.31 g) into hole 3
2. balance B (fill an empty tube to 41.91 g) into hole 5
3. sample0 (25.31 g) into hole 0
4. sample1 (41.91 g) into hole 2
Balance: 0.00 g out (limit 1.0 g); at most 0.87 g if each balance tube is 0.5 g off.
```

## Run it

```bash
cd step-2-plan
PYTHONPATH=$PWD pixi run --manifest-path ../simulation-environment/pixi.toml python -m pytest -q tests
PYTHONPATH=$PWD:$PWD/../simulation-environment \
  pixi run --manifest-path ../simulation-environment/pixi.toml python evaluate.py --workers 4   # about 15 s
```
