# v3 safety demo

A sample ML safety evaluation on the v3 centrifuge simulation: a camera model spots a person's hand reaching
into the robot's area and slows or stops the arm, and we test it the way a notified body would ask us to.

- [docs/plan.md](docs/plan.md): the plan (what, why, what we test, what we deliver)
- [docs/steps.md](docs/steps.md): the work broken into steps and two parallel tracks, and where each stands
- [docs/camera-check.md](docs/camera-check.md): which camera the safety feature uses, with the go / no-go pictures
- [docs/detector.md](docs/detector.md): the hand detector (YOLO + depth + a fixed rule), its data, and the known
  limitations found so far

v3 is not changed: everything here imports v3's code and adds to its scene from outside.

## What is here (Track A)

| File | What it does |
|---|---|
| `scene_plus.py` | v3's scene with the person, the safety camera and a floor added; `patched()` makes v3's own step code use it |
| `hand.py` | the person's arm and hand (simple shapes), their look (size, colours), and random reaches |
| `zones.py` | the stop and warning zones, the truth for each frame, and the gap between the person and the robot |
| `cameras.py` | the safety camera, and projecting world points into its picture |
| `render.py` | pictures with their truth; the go / no-go contact sheet |
| `record_jobs.py` | records real v3 jobs (the scene every 0.25 s) to draw the person into |
| `dataset.py` | the detector's pictures, YOLO labels, depth pictures and truth |
| `train_yolo.py` | trains the YOLO11n hand detector on the Mac's GPU |
| `box_zone.py` | the fixed rule: hand box + depth -> zone |
| `evaluate.py` | missed hands and false alarms on the test pictures, per condition |
| `tests/` | the demo leaves v3's scene and physics unchanged; zones, hand and reaches behave |

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
```

The recorded jobs, pictures and trained models (`data/`, `runs/`) are not committed; the commands above remake
them.
