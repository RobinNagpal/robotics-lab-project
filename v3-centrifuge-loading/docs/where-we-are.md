# Where We Are (23 Nov 2026, night)

**The whole job runs end to end, programmed: 20 of 20 scenes found, weighed, planned, filled, loaded and
balanced** ([Step 6 results](../step-6-whole-run/docs/results.md)). The robot's own report agrees with the
examiner on all 20. Not done: the lid (out of the arm's reach as the centrifuge stands; left open, decided
today), Gemini (quota still blocked), learned models (SmolVLA had one short run, 0 of 6).

## Where each step stands

| Step | What it does | Status | Headline result |
|---|---|---|---|
| 0 Find | which cells hold tubes, what each is | built (SAM 3) | 20 of 20 scenes all right in the whole run (140 tubes) |
| 1 Weigh | each sample: rack to balance, read, back | programmed; SmolVLA first run | 60 of 60; error median 0.002 g. SmolVLA 0 of 6 |
| 2 Plan | holes, balance tube weights, loading order | solver (no model needed) | 100 of 100 seeds planned right |
| 3 Fill balance tube | pour water until the tube weighs the target | programmed | 30 of 30 filled, 18 of 20 rotors balanced; spills a few balls a seed |
| 4 Load rotor | each tube into its 35° hole, force-guided | **programmed** | **20 of 20 jobs, 75 of 75 tubes seated, 0 spilled** ([results](../step-4-load-rotor/docs/results.md)) |
| 5 Report | camera check, robot's balance number vs examiner's | **built; lid left open** | 20 of 20 agree within 0.006 g ([results](../step-5-close-and-report/docs/results.md), [lid](../step-5-close-and-report/docs/lid-reach.md)) |
| 6 Whole run | Steps 0 to 5 in a row | **programmed** | **20 of 20 loaded and balanced; 11 of 20 with no ball spilled** (the misses: Step 3's pour) |

## What changed today

- **The gripper's pads got twist friction** (`settings.PAD_TORSION`): a tube held by its cap and tilted no
  longer turns in the fingers. Steps 1 and 3 were rerun: Step 1 the same; Step 3 needed one fix (put the
  water tube back by feeling for the seat), then works as before.
- **Step 4 built**: `rotor.py` (one tube into one hole, both grips), `load.py` (a whole job). Lessons: open
  tubes must go upright first and then tilt (one straight joint move emptied a tube); hole 5 beside a full
  hole 0 needs the same route; back out less where the lid is close; after a force stop, hold the arm where
  it is.
- **Lid study** (Step 5): a first version recommended turning the centrifuge −90°. It was wrong (fingertips
  placed 1 cm too close; loading not checked; turned, Step 4 failed 20 of 20), and the centrifuge was turned
  back. Redone with the real planners: only "turned +90°, 8 cm closer" both loads all holes and closes the lid,
  and there the open lid hangs over the balance unless it rests at 85°.

- **Whole run built** (`step-6-whole-run/run.py`). It found one real problem: a tube wedged short of the bottom
  (3° crooked in the firm grip) that the robot reported as fine. Fixed by a back-out-and-retry with a looser
  grip (Step 4) and a rotor-camera depth check before the report (Step 5).
- **Lid**: decided to leave it open (Step 5 = report only).

## Decisions waiting

1. The lid, later: a layout where the arm can close it (turn +90°, 8 cm closer, an 85° lid stop; rerun Steps 1, 3, 4). Options in [lid-reach.md](../step-5-close-and-report/docs/lid-reach.md#options).
2. SmolVLA: train longer (cloud GPU, or about 70 min per pass on this Mac), or leave it for now.

## Next checklist

- [ ] Step 3's spills (2 to 14 balls on balance-tube jobs): the one thing between 11 and 20 of 20 on the strict rule
- [ ] The lid (a new layout), if wanted
- [ ] When the Gemini quota is approved: Step 2's Gemini planner, the seated-tube check, Step 6 with Gemini
- [x] Commit the uncommitted work (Step 1 SmolVLA files, Step 3 fix, Step 4, Step 5 study, the scene change)

## How to run things

    cd v3-centrifuge-loading
    P="pixi run --manifest-path simulation-environment/pixi.toml"

    # Step 4, 20 seeds (about 10 min on 4 workers; the start states are in results/filled/)
    cd step-4-load-rotor
    PYTHONPATH=$PWD:$PWD/../step-1-weigh:$PWD/../step-3-fill-balance-tube:$PWD/../step-2-plan:$PWD/../simulation-environment \
        pixi run --manifest-path ../simulation-environment/pixi.toml python load.py --workers 4

    # Step 3 evaluation, 20 seeds (about 25 min on 4 workers)
    cd step-3-fill-balance-tube
    PYTHONPATH=$PWD:$PWD/../step-1-weigh:$PWD/../step-2-plan:$PWD/../simulation-environment \
        pixi run --manifest-path ../simulation-environment/pixi.toml python evaluate.py --seeds 20 --workers 4

    # Step 1 programmed evaluation (about 4 min)
    cd step-1-weigh
    PYTHONPATH=$PWD:$PWD/../simulation-environment \
        pixi run --manifest-path ../simulation-environment/pixi.toml python evaluate.py --seeds 20 --workers 4
