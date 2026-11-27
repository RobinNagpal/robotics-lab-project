# Step 6 Results: The Whole Job (programmed)

How it was planned: [approach.md](approach.md) (with Gemini as the coordinator; Gemini is blocked, so this is
the programmed version). Measured 2026-11-23. Script: `run.py`; raw results: `results/whole_run.json`.

## Where we are (read this first)

- **20 of 20 scenes done end to end**: every tube found, every sample weighed, the load planned, every
  balance tube filled, all 75 tubes seated in their planned holes, every rotor balanced (0.08 to 0.67 g, limit
  1.0 g). Each step used only what the earlier steps found.
- **The robot's own report agrees with the examiner on all 20**: within 0.006 g, and its camera check saw every
  tube seated.
- **11 of 20 with no ball spilled at all** (the simulator's strict rule). The other 9 are all jobs with balance
  tubes: Step 3's pour spills 2 to 14 balls (73 in all), as it does on its own.
- **The lid stays open**: as the centrifuge stands, the arm cannot reach to push it shut
  ([lid-reach.md](../../step-5-close-and-report/docs/lid-reach.md)).
- About 3.5 minutes per scene on this Mac (6 to 7 for jobs with two balance tubes).

**Since these results (2026-11-24): balance tubes are capped.** Step 3 now puts a spare cap on each balance
tube once it is filled, and Step 4 loads it by the cap like a sample
([Step 3's notes](../../step-3-fill-balance-tube/docs/approach.md#the-cap-the-whole-run)). The planner adds the
cap's 2 g to a balance tube's limits, and the pour stops once within 0.4 g. Tested on scene 13 only; the 20
scenes below were run before the change.

## What runs, in order

| Step | What | Uses |
|---|---|---|
| 0 Find | SAM 3 outlines the tubes in the high rack camera's picture; our rules say what each is (sample and its colour, water, empty) and which cell it stands in | the picture, the bench layout |
| 1 Weigh | every sample Step 0 found, on the balance, then back to its cell | Step 0's cells |
| 2 Plan | holes, balance tube weights, loading order | Step 1's readings |
| 3 Fill | each balance tube in the plan: an empty tube Step 0 found, water from a water tube it found; the second tube's target worked out again from the first's reading; then a spare cap on it | Steps 0 and 2, the balance |
| 4 Load | every tube into its hole in the plan's order (force-guided slide; a wedged tube is backed out and slid again with a looser grip) | Steps 0 and 2 |
| 5 Report | the rotor camera's depth picture checks every tube is seated; the robot's balance number from its own readings; the examiner's from the simulator | Steps 1 to 4, the rotor camera |

Gemini Robotics-ER was planned for Step 0 (finding), Step 2 (planning) and Step 6 (coordinating). Without it,
Step 0 uses SAM 3 (as good on these scenes: [Step 0 results](../../step-0-find/docs/results.md)), and the
plan and the order of the steps are fixed code.

## Results, seeds 0 to 19 (5 of each job)

| | Result |
|---|---|
| Step 0: every tube found, kind and colour right | **20 of 20** scenes (140 tubes) |
| Step 1: weighing error | median 0.002 g, at most 0.006 g (60 samples) |
| Step 4: tubes seated in their planned holes | **75 of 75** |
| Rotor balanced (examiner) | **20 of 20**; median 0.31 g, worst 0.67 g |
| Robot's number against the examiner's | median 0.003 g apart, at most 0.006 g |
| Robot's camera check: how far each tube's top is from where a seated one would be | 0.6 to 1.5 mm (75 tubes): all seen seated |
| No ball spilled | 11 of 20 (all 10 jobs without balance tubes, and 1 with) |
| Time per scene | median 206 s on this Mac (267 s simulated) |

By job:

| Job | Scenes | Loaded and balanced | No spill |
|---|---|---|---|
| pairs | 5 | 5 | 5 |
| odd one (1 balance tube) | 5 | 5 | 1 |
| triangle | 5 | 5 | 5 |
| mismatch (2 balance tubes) | 5 | 5 | 0 |

## Problems the whole run found, and their fixes

Running the steps together found two things the steps alone had not.

1. **A tube wedged short of the bottom, and the robot did not notice (seed 0, first run).** Step 1's real
   readings gave a slightly different plan from Step 4's test (holes 0, 1, 3, 4). The last tube, 3° crooked in
   the fingers, wedged 12 mm short of the bottom of hole 4. The hole forgives about 1.4°, and the firm grip
   (the pads now resist twisting) kept it from lining itself up. It crept in by pausing 76 times, the "bottom"
   rule fired falsely, and letting go of the wedged tube blew the simulation up. The robot still reported
   "Balanced: 0.38 g"; the examiner measured 35 g out.
   - **Fix 1 (Step 4):** a slide that keeps pausing inside the hole (more than 25 pauses, or 1.5 s in all) is
     wedged. Back out 15 mm, loosen the grip to a light 0.5 mm squeeze, slide again; at most twice, then stop
     without letting go. Seed 0 now seats on the first retry.
   - **Fix 2 (Step 5):** the robot checks with the rotor camera before it reports. From the depth picture, the
     top of whatever stands on each hole's axis must be within 3 mm of where a seated tube's top would be. On
     test scenes it measures a tube held 4, 8 and 12 mm high as 4.4, 8.4 and 12.4 mm (seated: 0.4 to 0.8 mm),
     and finds nothing in an empty hole. A tube not seen seated is left out of the number and named in the
     report.
2. **Step 3's spills** count against the strict rule: 2 to 14 balls on 9 of the 10 jobs with balance tubes
   (bouncing out of the balance tube during big bursts; [Step 3 results](../../step-3-fill-balance-tube/docs/results.md)).
   Not fixed.

## What is not done

- **The lid.** Out of reach as the centrifuge stands; the options are in
  [lid-reach.md](../../step-5-close-and-report/docs/lid-reach.md).
- **Gemini** as the coordinator, and in Steps 0 and 2: waits for the quota.
- **Learned models** in the arm's steps: SmolVLA had one short run for Step 1 (0 of 6,
  [smolvla-first-run.md](../../step-1-weigh/docs/smolvla-first-run.md)). Everything here is programmed.
- **Bench changes the robot must cope with** (racks or balance moved): Step 1's test B showed the programmed arm
  refuses safely there. This run uses the layout as drawn.

## Reproduce

```bash
cd v3-centrifuge-loading/step-6-whole-run
export PYTHONPATH=$PWD/../step-0-find:$PWD/../step-1-weigh:$PWD/../step-2-plan:$PWD/../step-3-fill-balance-tube:$PWD/../step-4-load-rotor:$PWD/../step-5-close-and-report:$PWD/../simulation-environment
pixi run -e sam --manifest-path ../simulation-environment/pixi.toml python run.py --workers 4   # about 25 min
```

The `sam` environment has SAM 3 (PyTorch). Step 0 runs first for every scene in one process (one copy of the
3.4 GB model); Steps 1 to 5 then run in up to 4 processes.
