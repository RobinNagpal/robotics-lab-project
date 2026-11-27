# Step 3 Results: Fill a Balance Tube

How it works: [approach.md](approach.md). Measured 2026-11-19 on the programmed arm. Raw results:
`results/evaluation_programmed.json` (every pour round of every tube), the earlier run in
`results/evaluation_run1.json`, the bench tests in `results/tilt_study_*.json`.

## Update 2026-11-23: rerun with the pads' twist friction

Step 4 found that the gripper model's pads had no grip against twisting, so the scene's pads now have it
(`settings.PAD_TORSION`, see [step-4-load-rotor/docs/insert-test.md](../../step-4-load-rotor/docs/insert-test.md)).
Step 3 was run again on the same 20 seeds.

The first rerun broke one seed. On seed 5 the water tube slid 12 mm up the fingers while pouring. It was let
go at the planned height, fell 12 mm crooked into the rack's small seat, and the simulation blew up (396 balls
spilled). The pads now resist twisting, which uses up part of their grip, so the tube slid more along the
fingers. Without twist friction, the loose grip had let the tube swing straight as it went in. Fix
(`moves.lower_until_touch`): put the water tube back by lowering it until the wrist force sensor feels it
touch the seat (at most 2 cm lower, as far as clear of the rack's plate). Let go, rise back to the planned
height, then back out.

| | Before (no twist friction) | Twist friction, first rerun | **Twist friction + feeling for the seat** |
|---|---|---|---|
| Filled with no refused move | 30 of 30 | 29 of 30 (seed 5 blew up) | **30 of 30** |
| Within 0.5 g / 0.3 g of target | 27 / 20 | 22 / 17 | 23 / 19 |
| Off target, median / worst | 0.24 / 1.56 g | 0.23 / 1.28 g | **0.13 / 1.28 g** |
| Rotor balanced (true weights) | 18 of 20 | 17 of 20 | **18 of 20** (median 0.31 g, worst 1.12 g) |
| Balls spilled | 73 (3.7 a seed) | 539 | 154 (median 7 a seed, at most 19) |
| Tubes back in their cells (3 mm, 3°) | 14 of 20 seeds | 17 of 20 | **18 of 20** (2 water tubes lean 3.8 and 5.1°) |
| Something else on the bench moved | 0 | 1 | 0 |

The step works as before. Rotors are balanced as often, more tubes are put back straight, and nothing is
knocked. The spill is about twice as large, still a few balls a seed: big bursts bouncing out of the balance
tube. Run-to-run changes are large here: the same seed often ends a ball or two different. Logs:
`results/evaluation_pad_torsion.log` (with the fix), `results/evaluation_pad_torsion_before_fix.log`. The
tables below are the 2026-11-19 run without twist friction.

## Words used here

| Word | Meaning |
|---|---|
| **Off target** | the balance tube's true weight (from the simulator) minus the weight it was meant to reach |
| **Rotor balanced** | with every tube's true weight in its planned hole, the pull adds up to 1.0 g or less (the job's limit) |
| **Spilled** | balls that ended up in no tube at all |
| **Round** | one go of the pour: creep until a ball lands, let it run or tip back, read the settled weight |

## In short

20 seeds that need balance tubes (the odd seeds 1 to 39): 10 "odd one" jobs (one balance tube each) and 10
"mismatch" jobs (two each), so 30 balance tubes.

| | Result |
|---|---|
| Balance tubes filled with no refused move | **30 of 30** |
| Within 0.5 g of target | **27 of 30** |
| Within 0.3 g of target | 20 of 30 |
| Off target | median 0.24 g; from 0.28 g under to 1.56 g over |
| **Rotor balanced** (true weights) | **18 of 20** seeds |
| Balls spilled | 73 in all (3.7 a seed, at most 11); none on 3 seeds |
| Filled balance tubes back in their cells | 30 of 30: at most 0.5 mm off the cell's centre, 1.95° from upright |
| Water tubes back in their cells | 24 of 30 within 3 mm and 3°; 6 lean 4 to 5° (3 to 3.7 mm off), all `water0` |
| Nothing else on the bench moved over 2 mm | 20 of 20 |
| Smallest gap planned between the arm (or a held tube) and anything | 5.3 mm |
| Time | median 159 s simulated per balance tube; about 2.5 to 4.5 minutes on this Mac per seed |

By job:

| Job | Balance tubes | Within 0.5 g | Within 0.3 g | Rotor balanced | Imbalance, median / worst |
|---|---|---|---|---|---|
| Odd one (one balance tube) | 10 | 9 | 4 | 9 of 10 | 0.44 / 1.57 g |
| Mismatch (two balance tubes) | 20 | 18 | 16 | 9 of 10 | 0.29 / 1.26 g |

## What the numbers say

- **The pour gets within one or two balls almost every time.** One ball is 0.33 g, and the pour can only stop
  between balls. Most tubes end one ball over or exactly on, because the pour aims to arrive from below and a
  round often brings two balls where one was wanted.
- **The second balance tube makes up for the first.** In the mismatch jobs the second tube's target is worked
  out again from what the first weighs. The second tubes all ended within 0.3 g of their new target (from
  0.28 g under to 0.29 g over).
- **Both misses come from one oversized round.** Rounds that tip straight back usually bring 1 to 3 balls (105
  of 118), but 13 brought 4 to 9. On seed 13 (odd one) a round brought 6 balls when 3 were wanted: 1.56 g over.
  On seed 35 (mismatch) the first tube's last round did the same (1.44 g over). The second tube could not make
  up for it, because one tube only cancels the part of the error along its own line in the rotor (1.26 g out).
- **The balance reads true.** Every final reading agrees with the true weight to 0.01 g, except one tube that
  lost a ball after it was weighed (seed 19, 0.33 g; graded with the true weight).
- **Spills are small.** 3.7 balls a seed, mostly balls that bounce out of the balance tube during big bursts.
  The bench test showed that a balance tube leaning 4.6° in the holder spilled 47 balls; with the grip below
  it leans at most 1.7°.

## How it got here

| Run | What changed | Within 0.5 g | Rotor balanced | Spilled | Main failures |
|---|---|---|---|---|---|
| 1 | first full run | 12 of 30 | 8 of 20 | 268 | 5 seeds stopped by a joint limit; simulation blew up on 1 seed; bursts not stopped (up to 3.9 g over); a tube leaning 4.6° in the holder |
| 2 | the fixes below | **27 of 30** | **18 of 20** | **73** | two oversized rounds |

| Fix | Why |
|---|---|
| Big turns of the hand at the travel height by sliding the joints, checked for collisions (`moves.reach`) | straight lines for the hand passed the wrist's singular pose or a joint limit; once the simulation blew up (tubes thrown metres) putting `water1` back |
| Back to the start pose between the side grip and fingers-down moves, the sweep checked first | turning the hand down where it was passed a joint limit |
| Tip back by jumping the target 10° at once (was a 40°/s ramp) | the arm trails its targets by about 0.2 s, enough for a whole burst |
| Stop a burst 3 g short of the target (was 2); creep slowly for the last 2 g (was 1) | bursts carry on up to 1.5 g past the moment they are stopped |
| Balance tube held 6 mm under its rim (a 25 mm body grip had been tried) | tested on 8 seeds: in the holder it leans at most 1.7° (25 mm: 4.8°), put back full at most 2.0° (25 mm: 4.3°) |

## The bench test (before the pour was built)

`tilt_study.py`, seed 1: the water tube tipped at 0.7° a second over the balance tube in the holder.

| Grip and pour direction | What happened |
|---|---|
| Fingers from behind (the grip plan's), pouring away from the arm | the tube turned between the pads: it hung nearly upright while the hand tipped to 113°; all 126 balls missed |
| From the side, 15° off, pouring sideways | the tube turned 4° in the fingers, the lip ended 6 mm to the side: 18 balls landed, 108 missed |
| **Squarely from the side, pouring sideways** | the tube follows the hand within 0.5°, the lip within 3 mm: **122 of 126 landed** |

With the grip that works: the first ball at 73°, then nothing until about 82°, then bursts of 10 to 20 balls, the
tube empty by 92°. While balls land the reading jumps by up to 5 g; it is exact again 0.2 s after a burst.

## Known problems

- **Oversized rounds** (2 misses in 20 seeds). The longer the creep goes without anything flowing, the bigger the
  slide when it comes. A next try: creep a shorter way, or tip back further between rounds, so the pile at the
  mouth stays small; or stop one ball early when only a ball or two is left.
- **`water0` stands back leaning 4 to 5°** in 6 of 20 seeds (never `water1`). It is not used again in the job
  and knocks nothing over, so it was left for now.
- **Each step moves the water tube from the side at 40 mm under its rim**, between the rack plate (deeper) and
  the balance tube's rim (shallower), with only about 5 mm of room either way. A real gripper's fingers are
  bigger than this model's in places; this would need checking on the real arm.

## Run it

    cd step-3-fill-balance-tube
    P="pixi run --manifest-path ../simulation-environment/pixi.toml"
    PYTHONPATH=$PWD:$PWD/../step-1-weigh:$PWD/../step-2-plan:$PWD/../simulation-environment \
        $P python evaluate.py --seeds 20 --workers 4          # about 25 minutes on this Mac
    PYTHONPATH=$PWD:$PWD/../step-1-weigh:$PWD/../simulation-environment $P python -m pytest tests
