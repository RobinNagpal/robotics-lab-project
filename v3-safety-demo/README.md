# v3 safety demo

A sample ML safety evaluation on the v3 centrifuge simulation: a camera model spots a person's hand reaching
into the robot's area and slows or stops the arm, and we test it the way a notified body would ask us to.

- [docs/where-we-are.md](docs/where-we-are.md): **start here**: what is done, what is waiting, what to run next
- [docs/plan.md](docs/plan.md): the plan (what, why, what we test, what we deliver)
- [docs/steps.md](docs/steps.md): the work broken into steps and two parallel tracks, and where each stands
- [docs/camera-check.md](docs/camera-check.md): which camera the safety feature uses, with the go / no-go pictures
- [docs/detector.md](docs/detector.md): the hand detector (YOLO + depth + a fixed rule), its data, and the known
  limitations found so far; results in [docs/detector-results.md](docs/detector-results.md)
- [docs/safety-layer.md](docs/safety-layer.md): the safety layer (Track B)
- [docs/integration.md](docs/integration.md): the two joined, in real v3 jobs with a person reaching in

v3 is not changed: everything here imports v3's code and adds to its scene from outside.

## What is here

The person, the camera and the detector (Track A):

| File | What it does |
|---|---|
| `scene_plus.py` | v3's scene with the person, the safety camera and a floor added (v3's physics and cameras unchanged); `patched()` makes v3's own step code use it |
| `hand.py` | the person's arm and hand (simple shapes), their look (size, colours), and random reaches |
| `zones.py` | the stop and warning zones, the truth for each frame, and the gap between the person and the robot |
| `cameras.py` | the RGB-D safety camera, and turning world points into pixels and back |
| `render.py` | pictures with their truth; the go / no-go contact sheet |
| `record_jobs.py` | records real v3 jobs (the scene every 0.25 s) to draw the person into |
| `dataset.py` | the detector's pictures, YOLO labels, depth pictures and truth |
| `train_yolo.py` | trains the YOLO11n hand detector on the Mac's GPU |
| `box_zone.py` | the fixed rule: hand box + depth -> zone |
| `evaluate.py` | missed hands and false alarms on the test pictures, per condition |
| `nobody_scan.py` | false alarms with nobody there over recorded jobs; mining them as training pictures |
| `release_check.py` | the update test: may a new detector version replace the current one |

The safety layer (Track B):

| File | What it does |
|---|---|
| `guard.py` | the wrapper around v3's `scene.step`: slow, hold, the behaviour rules, the watchdog, the normal layer |
| `detector.py` | the detector interface, the perfect detector, and `load` (through the registry only) |
| `safety_log.py` | the hash-chained, keyed safety log |
| `registry.py` | approved models with their settings, signed; activate, revoke, rollback |
| `keys.py` | the secret key that signs the log and the registry (made on first use, never committed) |
| `standin.py` | phase 1 stand-ins, now used only by the unit tests |

Joined (phase 3):

| File | What it does |
|---|---|
| `sensing.py` | the person during a job, the RGB-D camera with sensor noise, the YOLO detector, contact measuring |
| `run_guarded.py` | v3's whole job with the guard on, and a person reaching in if asked |
| `models/registry.json` | the signed registry: v1 and v2 (the model files themselves are not committed) |
| `tests/` | 82 tests: v3's scene, physics and cameras unchanged; zones, hand, reaches; guard, watchdog, normal layer; log, registry, signing; the joined parts |

## How to run

The demo has its own environment (`pixi.toml`: v3's simulation packages plus YOLO); it sets `PYTHONPATH` to this
folder and v3's simulation code.

```bash
cd v3-safety-demo
pixi run python -m pytest tests                                              # about 20 s
pixi run python render.py --sheet docs/figures/go-no-go-sheet.jpg --n 16 --rng 1

# Phase 2 (2 workers, as the Mac's limit allows while another job runs)
pixi run python record_jobs.py --seeds 0 1 2 3 4 5 6 7 --workers 2          # about 18 min
pixi run python dataset.py --split train --jobs 0 1 2 3 --n 1200 --workers 2
pixi run python dataset.py --split val --jobs 0 1 2 3 --n 300 --workers 2 --mode both
pixi run python dataset.py --split test --jobs 4 5 6 7 --n 3000 --workers 2 --mode both
pixi run python train_yolo.py --name v1 --epochs 20                          # about 25 min, GPU
pixi run python evaluate.py --weights runs/v1/weights/best.pt --name v1

# Phase 3: the joined system in v3's jobs (2 workers each)
pixi run python run_guarded.py --hand none --detector perfect --name nobody_perfect_20        # about 45 min
pixi run python run_guarded.py --hand person --detector perfect --seeds 4 5 6 7 --name person_perfect
pixi run python run_guarded.py --hand person --detector yolo --seeds 4 5 6 7                 # the active model
pixi run python nobody_scan.py --version v2 --jobs 4 5 6 7 --name test_v2
```

The recorded jobs, pictures and trained models (`data/`, `runs/`) are not committed; the commands above remake
them.
