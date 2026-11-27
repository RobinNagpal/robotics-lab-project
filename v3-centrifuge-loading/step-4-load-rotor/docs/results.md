# Step 4 Results: Load the Rotor (programmed)

How it works: [approach.md](approach.md). The first physics test (one tube, one hole) and the gripper fix it
led to: [insert-test.md](insert-test.md). Measured 2026-11-23, and again after the wedge fix below (the same 20 of 20). Raw
results: `results/evaluation_plain.json` (every tube of every seed).

## Where we are (read this first)

- **20 of 20 jobs loaded completely**: all 75 tubes (60 samples, 15 balance tubes with water) seated in
  their planned holes, in Step 2's order, on seeds 0 to 19 (5 of each job).
- **Every rotor balanced**: 0.10 to 0.75 g out (limit 1.0 g), graded with the simulator's true weights and
  where the tubes really are.
- **No ball spilled, nothing knocked over, no simulation blow-ups.** About 1.5 minutes per job on this Mac.
- Programmed, like Steps 1 to 3. Nothing here is learned, and Gemini is not used.

## Words used here

- **Seated**: the grading code's check: the tube's tip within 6 mm of the hole's bottom, its axis within 3° of
  the hole's.
- **Balance tube**: an empty tube that Step 3 filled with water so the rotor balances. It is open (no cap)
  and is held by its top.
- **Loading order**: Step 2 plans which hole each tube goes in and in what order. Balance tubes go first,
  and every tube goes in beside at least one empty hole, because the fingers need room.
- **Start state**: the scene just after Step 3 has filled the balance tubes (`filled_states.py` saves it
  once per seed; for jobs without balance tubes, the scene as built).

## What the arm does, for each tube in the plan's order

```
pick it from its rack cell
    samples: on the cap (Step 1's grip, 2 mm squeeze)
    balance tubes: on the open top (Step 3's grip, 6 mm under the rim, 1 mm squeeze)
──► up to the travel height
──► to the start of the slide, the hand tilted to the hole's 35°, 20 cm back along the hole's axis
    (10 cm for holes 4 and 5)
──► slide in along the axis (3 cm/s, then 1 cm/s near the hole), watching the wrist force sensor:
    pushed back more than 4 N? wait; near the bottom, that means the tip is down: stop
──► hold the arm where it is ──► open ──► back out along the axis (10 cm, or 8 or 6 if that is not clear)
──► back to the start pose, fingers down
```

The fingers' turn about the tube is chosen at each hole: the clearest of every 5° from −90° to 90°. With a
full neighbouring hole, only a narrow window is clear, on the empty side (the arm used ±50°).

## Results, 20 seeds

| Job | Seeds | Tubes | Seated as planned | Rotor balanced | Imbalance, median / worst |
|---|---|---|---|---|---|
| pairs (4 or 2 samples) | 5 | 20 | **20** | 5 of 5 | 0.38 / 0.44 g |
| odd one (3 samples + 1 balance tube) | 5 | 20 | **20** | 5 of 5 | 0.33 / 0.59 g |
| triangle (3 samples) | 5 | 15 | **15** | 5 of 5 | 0.18 / 0.67 g |
| mismatch (2 samples + 2 balance tubes) | 5 | 20 | **20** | 5 of 5 | 0.39 / 0.75 g |
| **All** | **20** | **75** | **75** | **20 of 20** | **0.33 / 0.75 g** |

Every tube:

| | Smallest | Middle | Largest |
|---|---|---|---|
| Tip's distance from the hole's bottom | 0.35 mm | 0.56 mm | 0.73 mm |
| Tube's axis against the hole's | 0.04° | 0.61° | 1.02° |
| Largest sideways push felt while sliding | 1.3 N | 2.1 N | 3.9 N |

- **Balls spilled: 0.** The balance tubes held 42 to 103 balls each.
- 60 slides reached their end; 15 were stopped early by the force sensor (the tube was a little lower in the
  fingers than planned, so it reached the bottom sooner).
- Smallest planned gap between the arm (or the held tube) and anything else: 2.3 mm.
- Time: median 130 s simulated, 93 s on this Mac per job.
- The imbalance comes from the tubes' weights, not from the loading. Step 2's plan with Step 3's fills gives
  it (Step 3: 18 of 20 rotors balanced on its own seeds; on these 20, every one is).

## Problems met on the way, and what fixed them

| What happened | Why | Fix |
|---|---|---|
| A cap-held tube turned up to 23° in the fingers | The gripper model's pads had no grip against twisting | Twist friction on the pads, now in the shared scene ([insert-test.md](insert-test.md)) |
| A balance tube was seated but **lost all 103 balls** on the way | The only clear joint move to hole 4 swung the open tube to 92° | Open tubes first go to an upright pose above the start, then tilt evenly to 35° along a straight line. They never lean more than 40° (`rotor.MAX_TILT`) |
| Seeds 0, 8 and 12: "no clear way to the start" for the last tube | Hole 5 leans towards the arm. With hole 0 already full, no straight joint move to its start was clear | The same upright-then-tilt route, for capped tubes too |
| Backing out of holes 4 and 5 refused: the wrist 1.2 mm from the open lid | 10 cm back along the axis, the open fingers reach the lid | Back out 10, 8 or 6 cm, the first that is clear (6 cm clears the tube) |
| Simulation blew up just after a good slide | The arm trails its targets, and kept pressing the seated tube | On stopping, hold the arm where it **is** |
| Holes 4 and 5 refused at first | The first arm pose found reached over the arm's head | Find every arm pose for the start and use the one whose joint move has the most room |
| In the whole run (Step 6, seed 0) a tube wedged 12 mm short of hole 4's bottom; letting go blew the simulation up | 3° crooked in the fingers; the hole forgives about 1.4°, and the firm grip kept it from lining itself up | A slide that keeps pausing inside the hole (more than 25 pauses or 1.5 s) backs out 15 mm, loosens the grip to 0.5 mm and slides again, at most twice; then it stops without letting go. Before opening, the fingers' opening is checked for room (45, 41 or 39 mm) |

## Not covered yet

- **Errors in where the holes are.** The arm uses the bench layout. In [insert-test.md](insert-test.md), a
  hole up to 5 mm from where the arm thinks still worked. A real robot would get the rotor's place from a
  camera (Step 0's point cloud gives about 1 mm).
- Checking each tube is seated is done in Step 5 (the rotor camera's depth picture).
- **The lid** (Step 5): out of reach as the centrifuge stands now
  ([lid-reach.md](../../step-5-close-and-report/docs/lid-reach.md)).

## Files

| What | Where |
|---|---|
| One tube into one hole (both grips) | `rotor.py` |
| A whole job, and the 20-seed measurement | `load.py` |
| The start states (after Step 3) | `filled_states.py`, `results/filled/` |
| Single tests | `insert_test.py` (a sample), `balance_test.py` (a balance tube) |
| Geometry tests | `tests/test_rotor.py` |

## Reproduce

```bash
cd v3-centrifuge-loading/step-4-load-rotor
export PYTHONPATH=$PWD:$PWD/../step-1-weigh:$PWD/../step-3-fill-balance-tube:$PWD/../step-2-plan:$PWD/../simulation-environment
P="pixi run --manifest-path ../simulation-environment/pixi.toml"
$P python filled_states.py --seeds $(seq 0 19) --workers 4   # Step 3's fills, once: about 12 min
$P python load.py --workers 4                                # 20 seeds: about 10 min
$P pytest -q tests
```
