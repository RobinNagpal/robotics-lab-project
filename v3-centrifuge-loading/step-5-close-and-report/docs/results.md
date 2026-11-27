# Step 5 Results: Report (the lid left open)

How it was planned: [approach.md](approach.md). Measured 2026-11-23. Decided the same day: **the lid stays
open**. As the centrifuge stands, the arm cannot reach to push it shut, and the one layout where it can puts
the open lid over the balance ([lid-reach.md](lid-reach.md)). So Step 5 is the report only.

## The report

After Step 4, the robot works out how far out of balance the rotor is, from what it knows itself:

- the weight of each sample, as it read it on the balance in Step 1;
- the weight of each balance tube, as it read it after filling in Step 3;
- which hole it put each tube in (Step 4).

The examiner works out the same from the simulator: each tube's true weight and the hole it is really seated
in (`grading.grade`). Both use the same rule: each tube pulls outwards through its hole, and the imbalance is
the length of all the pulls added together (limit 1.0 g).

What the robot says, for example:

> Balanced: 0.33 g out (limit 1.0 g), 4 tubes in holes 0, 1, 3, 4; examiner 0.33 g; lid left open (out of
> the arm's reach)

## The camera check

Before it reports, the robot checks every tube is seated with the rotor camera's depth picture (`camera_check`
in `report.py`). From the depth picture, it finds the top of whatever stands on each hole's axis and compares
it with where a seated tube's top would be. Within 3 mm: seen seated. Higher: stuck short of the bottom.
Nothing there: missing. A tube not seen seated is left out of the number and named: "NOT done: tube not
seated (hole 4 12 mm high)".

| Test (tubes put in the holes by hand) | Measured |
|---|---|
| seated (capped and open tubes) | 0.4 to 0.8 mm |
| held 4 / 8 / 12 mm high | 4.4 to 4.8 / 8.4 to 8.8 / 12.4 to 12.8 mm |
| empty hole | −31 and −52 mm (no tube top there) |

It was added after the whole run found a tube wedged 12 mm short that the robot had reported as fine (Step 6,
seed 0). In the whole run it saw all 75 tubes seated (0.6 to 1.5 mm).

## Results in the whole run (Step 6, 20 seeds, the robot's own readings)

| | Result |
|---|---|
| Robot's number against the examiner's | median 0.003 g apart, at most 0.006 g |
| Same verdict | 20 of 20 |
| Rotors balanced | 20 of 20 (0.08 to 0.67 g) |

## Results on Step 4's 20 seeds

`report.py` reads Step 4's saved results (no new simulation).

| | Result |
|---|---|
| Robot's number against the examiner's | median **0.004 g** apart, at most 0.008 g |
| Same verdict (balanced or not) | **20 of 20** |
| Rotors balanced | 20 of 20 (0.10 to 0.75 g) |

On these seeds, the sample weights are Step 3's stand-in for Step 1's readings (the true weights to the
balance's 0.01 g). The balance tubes' readings are real. Step 6 runs the whole job with Step 1's own
readings: [step-6-whole-run](../../step-6-whole-run/docs/results.md).

## The lid

Not closed. `close.py` holds the closing planner: closed fingertips on the lid's inside face, following its
swing, the wrist force sensor watching for the latch. It is ready for a layout where the lid can be reached.
The study and the options: [lid-reach.md](lid-reach.md).

## Files

| What | Where |
|---|---|
| The report | `report.py` |
| The lid study, and the closing planner | `lid_reach.py`, `close.py` |
| Results | `results/report_step4_seeds.json`, `results/lid_reach.json` (the first, wrong study: `results/lid_reach_v1_wrong_fingertips.json`) |
