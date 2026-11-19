# Where We Are (19 Nov 2026)

Steps 0 to 3 are built and measured. The only thing blocking SmolVLA training for Step 1 is one 906 MB
weights file that the slow internet could not download. First thing tomorrow: get that file, then run a short
training speed trial and decide on the full run. If the internet stays slow, Step 4 (load the rotor) can be
built offline instead.

## Where each step stands

| Step | What it does | Status | Headline result |
|---|---|---|---|
| 0 Find | which cells hold tubes, what each is | built | SAM 3 + high rack camera: 140 of 140 tubes, kind and colour all right (new scene). Gemini: 4 seeds only (quota hit), old scene |
| 1 Weigh | each sample: rack to balance, read, back | programmed built; SmolVLA not trained | programmed 60 of 60 (new scene); dataset recorded and converted |
| 2 Plan | holes, balance tube weights, loading order | solver built; the Gemini part waits for the key | 100 of 100 seeds planned right |
| 3 Fill balance tube | pour water until the tube weighs the target | programmed built today | 27 of 30 tubes within 0.5 g; 18 of 20 rotors balanced |
| 4 Load rotor | each tube into its 35° hole, force-guided | not built | |
| 5 Close lid, report | push the lid shut, report the balance | not built; the lid's reach is still an open problem | |
| 6 Whole run | Gemini runs Steps 0 to 5 as tools | not built; needs the Gemini key | |

## What changed today: a second water tube

One water tube (126 balls, 41 g) could not supply the "mismatch" job: its two balance tubes need up to 198
balls together (24 of 25 test seeds needed more than one tube holds). The supply rack now has `water0` and
`water1` in the corner cells 0 and 3, and the two empty tubes in cells 5 and 6. Sample weights are unchanged.

Redone after the change (all local):

- Step 1 programmed arm: again 60 of 60.
- Step 1 dataset re-recorded (66 of 66 weighings, 10 min) and converted to LeRobot format: 120 training
  episodes (21,085 frames), 12 validation. The old recording is in `step-1-weigh/data_one_water_tube/`.
- SAM 3 Step 0 check: 140 of 140 with the high rack camera.
- Gemini was not run again (no calls without asking); its Step 0 numbers are from the old scene.

## Step 3: fill a balance tube (built today)

Details: `step-3-fill-balance-tube/docs/approach.md` and `results.md`.

Results on 20 seeds (30 balance tubes): 30 of 30 filled, 27 within 0.5 g, 20 within 0.3 g, 18 of 20 rotors
balanced with the true weights, 3.7 balls spilled per seed.

What we learned:

- The grip plan's grip from behind fails: the pads sit on the line the tube is tipped about, so it swings like
  a door on a hinge. The water tube must be held squarely from the side, 40 mm under its rim.
- Water is poured sideways (from the racks' side of the balance tube): pouring away from the arm with the hand
  at the side lines up the wrist's joints (a singular pose).
- The balls don't pour like water: nothing leaves until about 82° of tilt, then they come in bursts of 10 to 20.
  So the pour goes in rounds: creep until a ball lands, let it run or tip straight back, read the settled weight.
- With two balance tubes, the second one's target is recomputed from what the first really weighs.

Known problems:

- 2 of 20 seeds missed: one round brought 6 balls when about 3 were wanted (up to 1.56 g over).
- `water0` stands back in its cell leaning 4 to 5° on 6 seeds (not used again, knocks nothing over).

## SmolVLA for Step 1: the one missing file

Ready: the dataset (LeRobot format, `step-1-weigh/learn/outputs/datasets/`), LeRobot 0.7.0 installed
(`step-1-weigh/learn/`), and the SmolVLM2-500M weights (2.03 GB, in the Hugging Face cache).

Missing: `model.safetensors` (about 906 MB) from https://huggingface.co/lerobot/smolvla_base. The small config
files of that model are already downloaded. Today's internet ran at 0.03 to 0.3 MB/s, so the download was
stopped (it had reached about 690 MB but kept almost nothing; a new try starts from scratch).

Two ways to get it:

1. Restart the download here if the internet is fast (check the speed first).
2. Download that one file on a faster connection, copy it to this Mac, and point the training at it.

Then: write the training script (cameras renamed overhead → camera1, racks_high → camera2,
wrist → camera3 to match smolvla_base), run a few-minute trial on the Mac's GPU with a modest batch size,
report how long the full training takes, and start it only after a go-ahead.

## Tomorrow's checklist

- [ ] Check the internet speed; get `model.safetensors` for `lerobot/smolvla_base` (either way above)
- [ ] Write the SmolVLA training script and run a short speed trial; report the full training time
- [ ] Full training (after a go-ahead), then test on seeds 0 to 19 (test A) and the moved scenes (test B)
- [ ] If the internet is still slow: start Step 4 (load the rotor), which needs no internet
- [ ] When the Gemini credits arrive: Step 2's Gemini planner, and later Step 6
- [ ] Commit `v3-centrifuge-loading/` (the whole folder is still untracked in git)

## How to run things

    cd v3-centrifuge-loading
    P="pixi run --manifest-path simulation-environment/pixi.toml"

    # Step 3 evaluation, 20 seeds (about 25 min on 4 workers)
    cd step-3-fill-balance-tube
    PYTHONPATH=$PWD:$PWD/../step-1-weigh:$PWD/../step-2-plan:$PWD/../simulation-environment \
        pixi run --manifest-path ../simulation-environment/pixi.toml python evaluate.py --seeds 20 --workers 4

    # Step 1 programmed evaluation (about 3 min)
    cd step-1-weigh
    PYTHONPATH=$PWD:$PWD/../simulation-environment \
        pixi run --manifest-path ../simulation-environment/pixi.toml python evaluate.py --seeds 20 --workers 4

    # Step 1 dataset to LeRobot format (about 6 min)
    cd step-1-weigh/learn && pixi run python convert.py
