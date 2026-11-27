# v3 safety demo

A sample ML safety evaluation on the v3 centrifuge simulation: a camera model spots a person's hand reaching
into the robot's area and slows or stops the arm, and we test it the way a notified body would ask us to.

- [docs/plan.md](docs/plan.md): the plan (what, why, what we test, what we deliver)
- [docs/steps.md](docs/steps.md): the work broken into steps and two parallel tracks, and where each stands
- [docs/camera-check.md](docs/camera-check.md): which camera the safety feature uses, with the go / no-go pictures

v3 is not changed: everything here imports v3's code and adds to its scene from outside.

## What is here (Track A, phase 1)

| File | What it does |
|---|---|
| `scene_plus.py` | v3's scene with the person, the safety camera and a floor added; `patched()` makes v3's own step code use it |
| `hand.py` | the person's arm and hand (simple shapes), their look (size, colours), and random reaches |
| `zones.py` | the stop and warning zones, the truth for each frame, and the gap between the person and the robot |
| `cameras.py` | the safety camera, and projecting world points into its picture |
| `render.py` | pictures with their truth; the go / no-go contact sheet |
| `tests/` | the demo leaves v3's scene and physics unchanged; zones, hand and reaches behave |

## How to run

The demo uses v3's simulation environment (`v3-centrifuge-loading/simulation-environment/pixi.toml`).

```bash
cd v3-safety-demo
export PYTHONPATH=$PWD:$PWD/../v3-centrifuge-loading/simulation-environment
P="pixi run --manifest-path ../v3-centrifuge-loading/simulation-environment/pixi.toml"

$P python -m pytest tests                                         # about 10 s
$P python render.py --sheet docs/figures/go-no-go-sheet.jpg --n 16 --rng 1   # about 5 s
```
