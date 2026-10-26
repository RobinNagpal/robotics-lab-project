# Step 0 Results: Detect and Classify with YOLO11-seg

The plan for this step is in [step-0-detect-and-classify.md](../../docs/solutions/step-0-detect-and-classify.md).
This file says what we built, how we trained it, and how well it works.

---

## Summary

On 400 test scenes it has never seen, the detector:

- **finds 98.1%** of the objects (4,445 of 4,529),
- **gives the right class to 98.2%** of the objects it finds,
- **places 97.6%** of the correctly classified objects **within 1 cm** of where they really stand.

Racks, the tray, the stand, beakers and flasks are close to perfect. The weak spot is
**test tube vs centrifuge tube**: from above both are rings about 25 to 29 mm across.

---

## Model

| | |
|---|---|
| Model | `yolo11s-seg` (YOLO11, small, with masks), Ultralytics 8.4.123 |
| Start point | weights trained on COCO (everyday photos), then trained on our pictures |
| Why YOLO11 | newer than YOLOv8, more accurate for its size, same code |
| Why small (`s`) | a good middle: nano is less accurate, medium is slower to train |
| Licence | AGPL-3.0 (fine for research; matters if this ships in a product) |

## Data

All pictures come from the overhead camera, 1280 x 960, in random scenes built by
`detect_classify/random_scene.py`. The labels come from MuJoCo's segmentation picture, so
nothing is labelled by hand.

What changes from scene to scene:

- how many tubes stand in the source rack (1 to 4) and the empty rack (1 to 5), and which kinds,
- how many beakers and flasks there are (0 to 2), and whether there is a tray (80% of scenes),
- where the 3 racks, the tray and the stand stand, and which way they face,
- colours of the table, plastic, glass and liquid; the lights; the camera mount (up to 2 cm, 2 degrees);
  the arm's pose.

| Split | Pictures | Seeds | Objects |
|---|---|---|---|
| Train | 2,000 | 0 to 1,999 | 22,517 |
| Validation | 400 | 1,000,000 to 1,000,399 | 4,503 |
| Test | 400 | 2,000,000 to 2,000,399 | 4,529 |

Objects in the training set, per class:

| test_tube | centrifuge_tube | beaker | flask | rack | tray | pouring_stand |
|---|---|---|---|---|---|---|
| 9,970 | 955 | 1,018 | 981 | 6,000 | 1,593 | 2,000 |

Rendering all 2,800 pictures takes about 9 minutes on 8 cores.

## Training settings

| Setting | Value | Why |
|---|---|---|
| `imgsz` | 640 | 1280 needed more memory than the Mac has (see below) |
| `batch` | 8 | fits in 5.4 GB |
| `epochs` | 30, `patience=10` | stop early if the score stops improving |
| `device` | `mps` (Apple M5 GPU) | |
| `hsv_h`, `hsv_s`, `hsv_v` | 0.5, 0.7, 0.4 | strong colour changes, so it learns shapes, not colours |
| `fliplr`, `flipud` | 0.5, 0.5 | a straight-down camera has no up or down |
| everything else | Ultralytics defaults | (mosaic, scale, optimizer `auto`) |
| `seed` | 0, `deterministic=True` | so a run can be repeated |

### How long it took

- **About 6.3 minutes per epoch** (4.5 training, about 1 checking the validation set), on an Apple M5 with 16 GB.
- The first try, at `imgsz=1280` and `batch=8`, needed 17.3 GB. The Mac ran out of memory, swapped
  to disk, and took 5 to 10 minutes per batch. Measured speeds of other settings, for one epoch of
  2,000 pictures (training only):

| Model | `imgsz` | `batch` | Memory | Time per epoch |
|---|---|---|---|---|
| yolo11s-seg | 640 | 8 | 5.4 GB | 4.5 min (used) |
| yolo11n-seg | 640 | 8 | 3.5 GB | 3.3 min |
| yolo11s-seg | 1280 | 2 | 8.5 GB | 15 min |
| yolo11n-seg | 1280 | 4 | 9.3 GB | 9.4 min |

- **The run stopped in epoch 13 of 30**, after 78 minutes, with an error inside Ultralytics'
  training code (`RuntimeError: shape mismatch` in the box assigner). This looks like a bug in
  PyTorch on the Apple GPU, not in our data. The best weights, from **epoch 10**, were saved and
  are the ones graded here.

Validation score per epoch (masks; 1.0 is perfect):

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | **10** | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mAP50 | .890 | .904 | .942 | .952 | .967 | .943 | .968 | .967 | .977 | **.978** | .977 | .978 |
| mAP50-95 | .657 | .671 | .698 | .701 | .725 | .725 | .734 | .731 | .732 | **.746** | .743 | .737 |

The score had mostly levelled off by epoch 10, so the missing epochs would likely add little.

---

## Results on the test set

### YOLO's own scores

| | mAP50 | mAP50-95 |
|---|---|---|
| Boxes | 0.985 | 0.894 |
| Masks | 0.982 | 0.745 |

Masks, per class:

| Class | mAP50 | mAP50-95 |
|---|---|---|
| test_tube | 0.978 | 0.529 |
| centrifuge_tube | 0.925 | 0.482 |
| beaker | 0.995 | 0.683 |
| flask | 0.995 | 0.696 |
| rack | 0.995 | 0.927 |
| tray | 0.995 | 0.956 |
| pouring_stand | 0.995 | 0.941 |

mAP50-95 is low for tubes because a tube is only about 11 pixels wide at `imgsz=640`: being one
pixel off at the edge is already a big share of the outline. This matters less than it looks,
because the position comes from depth, not from the outline's exact edge (next table).

### Step 0 grading: count, class, position

From `python -m detect_classify.evaluate` (confidence 0.5). A detection counts as finding an
object when its mask overlaps the true outline by at least half (intersection over union).
Positions are only graded for objects given the right class.

| Class | True | Found | Right class | x-y error, median | x-y error, 95% under | Within 1 cm | Top height error, median |
|---|---|---|---|---|---|---|---|
| test_tube | 1,976 | 1,906 (96.5%) | 1,849 (97.0%) | 0.9 mm | 9.2 mm | 95.5% | 0.1 mm |
| centrifuge_tube | 213 | 203 (95.3%) | 190 (93.6%) | 0.5 mm | 21.1 mm | 93.7% | 0.1 mm |
| beaker | 214 | 214 (100%) | 214 (100%) | 1.2 mm | 3.0 mm | 99.5% | 0.8 mm |
| flask | 198 | 195 (98.5%) | 183 (93.8%) | 0.6 mm | 1.2 mm | 100% | 0.1 mm |
| rack | 1,200 | 1,199 (99.9%) | 1,199 (100%) | 0.0 mm | 0.4 mm | 99.3% | 0.1 mm |
| tray | 328 | 328 (100%) | 328 (100%) | 0.0 mm | 0.3 mm | 99.7% | 0.1 mm |
| pouring_stand | 400 | 400 (100%) | 400 (100%) | 0.0 mm | 0.3 mm | 99.8% | 0.1 mm |

Also: **71 detections** matched no real object (about 1 in every 6 pictures).

Mistaken classes:

| True class | Called | Times |
|---|---|---|
| test_tube | centrifuge_tube | 57 |
| centrifuge_tube | test_tube | 13 |
| flask | beaker | 12 |

### Compared with perfect outlines

To tell detector mistakes from locating mistakes, the same grading was run with the true
outlines in place of the detector (`python -m detect_classify.evaluate --oracle`):

| Class | x-y error, median | Within 1 cm |
|---|---|---|
| test_tube | 0.7 mm | 98.9% |
| centrifuge_tube | 0.3 mm | 99.1% |
| beaker | 1.2 mm | 100% |
| flask | 0.6 mm | 100% |
| rack, tray, stand | 0.0 mm | 100% |

So the locating maths is sound. Almost all the loss comes from the detector: missed tubes,
wrong tube classes, and a few tube masks that spill onto a neighbour (the 9 to 21 mm tail).

---

## What to improve next

In order of how much they should help:

1. **Test tube vs centrifuge tube.** From above they look almost the same. Options: give the
   model the depth picture as well (the centrifuge tube is shorter and plastic), train at a larger
   picture size on a bigger GPU, or leave this to a close-up with the wrist camera.
2. **Tubes standing close together.** Most missed tubes and the larger position errors are tubes
   right next to each other in a rack. A larger `imgsz` should help here too.
3. **Finish the run.** Resume from `last.pt` for the remaining epochs. The score had mostly
   levelled off, so expect a small gain.
4. **Roles** (source rack, empty rack, source, receiver) are not done yet. They need "does this
   container hold liquid?", which belongs with Step 0.2 (liquid level).

---

## How to repeat this

From `step-0-detect-and-classify/`:

```
make data                      # render the 2,000 / 400 / 400 pictures with labels
make preview                   # draw labels over 12 pictures, to check them
make train NAME=yolo11s_640_2000
make evaluate NAME=yolo11s_640_2000
make oracle                    # grade the locating maths alone
```

Files: weights in `runs/yolo11s_640_2000/weights/best.pt`, training curves in
`runs/yolo11s_640_2000/results.csv` and `results.png`, grading in
`runs/yolo11s_640_2000/step0_results.json`.
