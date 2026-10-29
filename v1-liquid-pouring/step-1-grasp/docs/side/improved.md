# Step 1 (Side Grasps), Version 2: the Improved Model

The first side-grasp model is described in [approach.md](approach.md) and graded in
[results.md](results.md). Those pages stay as they were. This page is the second version: what was
changed, why, with pictures, and how it compares.

**In one line:** more tubes can now be taken from the side (62% instead of 40%), the network makes
**46% more good grasps** (531 instead of 364 on the same 1,151 test tubes), and in full physics
**18 of 20** reachable grasps of pickable tubes succeeded (version 1: 6 of 9). The price: it is less
sure which tubes are boxed in (it refuses 80% of them, version 1 refused 98%).

---

## What the words mean

| Word | Meaning |
|---|---|
| **Pickable** | the fast check finds at least one clear direction for the tube (it can be taken from the side) |
| **Boxed in** | no direction is clear: the right answer is to refuse and take another tube |
| **Good grasp** | the network's choice is clear, and the pinch point within 5 mm of the tube's axis |
| **Wrong direction** | the network picked a direction that is blocked, while the tube had clear ones |
| **Aim off** | the network picked a clear direction, but put the fingers 1 or 2 mm off, and at that spot they touch a neighbour |
| **Wrongly refused** | the network said "no clear direction" for a pickable tube |
| **Row neighbour** | the tube in the next cell along the same row, 36 mm away centre to centre: 11 mm of air between the two |
| **Pad** | the rubber tip of each finger |
| **Shove** | in the fast check, the closing fingers are also tried 3 mm to either side, for a first finger that pushes the arm aside |

---

## The three changes

### 1. The fingers come in opened just wider than the tube

Version 1 came in with the fingers fully open, 85 mm between the pads. The tubes in a row are 36 mm
apart, so the pads landed on the row neighbours even when there was empty space behind the tube. We
know the tube is 25 mm wide, so version 2 opens the fingers to **35 mm** before going in: 5 mm of room
on each side of the tube.

![Fully open vs opened 35 mm](images/v2/fingers.png)

*Seen from above, the gripper coming straight in from outside the rack. Left: fully open (version 1);
the pads sit on the tubes beside the target. Right: opened 35 mm (version 2); the pads fit in the gaps.*

**Tried in physics first.** On 26 tubes the arm could reach, the fingers opened 35 mm and the tube slid
between the pads and lifted cleanly every time. The same runs showed that the fast check's 3 mm shove
(a lesson from top-down grasps) is too cautious here: **12 of 12** grasps that only the shove had called
blocked lifted cleanly. With the pads already 5 mm from the tube, the first pad to touch moves the tube
in its cell's play rather than pushing the arm. So version 2 checks the closing fingers without the shove.

### 2. The directions are counted from the rack, not the table

Racks stand at random angles on the table. Version 1's 24 directions (15 degrees apart) were counted
from the table, so the most useful direction, straight out of the rack, often fell between two of them,
and 7 degrees off is enough to touch a neighbour in an 11 mm gap.

Version 2 still has **24 directions** (no extra cost), but turns the close-up camera to line up with the
rack first. Then four of the 24 point exactly straight out of the rack's four sides.

![The close-up before and after turning it to the rack](images/v2/rack_frame.png)

*The grey lines are the 24 directions, the yellow ones the rack's sides. Left (version 1): the yellow
lines fall between grey ones. Right (version 2): the picture is turned to the rack, and four of the 24
are the rack's sides.*

**How the rack's angle is found** (no training, worked out from the picture): the camera first takes
the close-up as before. The rack's top plate is a flat rectangle at a known height (6 cm), so the pixels
at that height near the middle are the target's rack. Its holes are filled in, and the narrowest
rectangle that fits round it gives the angle. Then the picture is taken again, turned by that angle.
Measured against the truth on 236 tubes: off by 0.2 degrees half the time, under 1.4 degrees for 95 out
of 100. About 1 in 20 is several degrees out (little of the rack in view); those tubes lose the
benefit of directions lined up with the rack, but their labels are still right.

### 3. Aim at the middle of a run of clear directions

In version 1, most wrong directions (74 of 92) were **one step (15 degrees) next to a clear one**: the
network picked the edge of a run of clear directions, just past it. Version 2 mixes each direction's
score with its two neighbours' (half its own, a quarter each neighbour's) before choosing. A direction
in the middle of a run then beats one at its edge. No training is needed for this part.

![Choosing with and without mixing in the neighbours](images/v2/smoothing.png)

*Left: the truth for one tube. Middle: version 1's choice (yellow), the single best score, at the edge of
the clear run: blocked. Right: the same scores mixed with their neighbours; the choice moves into the run:
a good grasp.*

On its own (the version 1 network, nothing retrained), this change gave 370 good grasps instead of 364,
and 87 wrong directions instead of 92. A small gain.

---

## The new data and training

Everything else is as for version 1: the same 400 / 50 / 100 crowded scenes (4,660 / 558 / 1,151
pictures), the same network (24 outputs, 1.82 million weights), 16 epochs.

| | Version 1 | Version 2 |
|---|---|---|
| Fingers on the way in | fully open (85 mm) | **opened 35 mm** |
| Shove when closing | 3 mm either way | none (physics showed it is not needed) |
| Directions | 24, from the table | 24, **from the rack** (picture turned to it) |
| Choosing | best single score | **score mixed with the two neighbours** |
| Pickable, train / val / test | 38% / 42% / 40% | **63% / 63% / 62%** |
| Making the data | about 5 min | about 20 min (each picture is labelled in its own directions) |
| Training | 25 min (other jobs running) | 16 min |
| Cut-off for "no clear direction" | 0.10 | 0.15 (chosen on validation) |

Both kinds of tube, pickable and boxed in, are in every split.

---

## The result

All on the test tubes: 100 scenes neither network saw, 1,151 tubes. Version 1 is in
[results.md](results.md).

| | Version 1 | Version 1 + mixing only | **Version 2** |
|---|---|---|---|
| Pickable tubes | 457 (40%) | 457 (40%) | **712 (62%)** |
| **Good grasps (number of tubes)** | 364 | 370 | **531** |
| Good grasps, share of pickable | 79.6% | 81.0% | 74.6% |
| Wrong direction | 92 (20.1%) | 87 (19.0%) | 93 (**13.1%**) |
| Aim off | 1 | 0 | 85 (11.9%) |
| Wrongly refused | 0 | 0 | 3 |
| Boxed-in tubes refused (right answer) | **98.4%** (683 of 694) | 97.8% | 79.7% (350 of 439) |
| Position error, median / 95% under | 1.7 / 2.9 mm | 1.3 / 2.9 mm | 1.4 / 2.6 mm |
| **Full physics: pickable, lifted cleanly, of reachable** | 6 of 9 | not run | **18 of 20** |
| Full physics: boxed-in tubes tried | 0 | not run | 3 the arm could reach: 2 lifted cleanly, 1 touched a neighbour |
| Fast check vs physics (random directions) | agree 90% | | agree 93% |

By tube size, version 2 (version 1 in brackets):

| Tube | Pickable | Good grasp, of pickable |
|---|---|---|
| short (100 mm) | 38% (20%) | 75.9% (77.8%) |
| middle (150 mm) | 67% (37%) | 74.2% (84.7%) |
| tall (200 mm) | 78% (61%) | 74.3% (77.2%) |

![What version 2 says, next to the truth](images/v2/outputs.png)

*Top: the truth (long green = clear, short red = blocked), in the picture turned to the rack. Bottom:
version 2's mixed score per direction and its choice (yellow). Left: a pickable tube. Right: a boxed-in
tube, refused.*

![A version 2 grasp in full physics](images/v2/trial.png)

*A version 2 grasp with the real arm: the fingers already opened 35 mm, down beside the tube, in,
closed, lifted.*

### What got better

- **Many more tubes can be taken from the side**, and short ones nearly twice as often (38% vs 20%).
- **531 good grasps instead of 364** on the same tubes. Wrong directions dropped from 20% to 13% of
  pickable tubes.
- **In physics, 18 of 20** reachable grasps lifted cleanly. The fast check never called a direction clear
  that then failed in physics, in either version.

### What got worse, and why

- **Aim off: 85 tubes.** With the fingers opened only 35 mm, each pad's outer face passes within 1 to 2 mm
  of the row neighbour. The network places the fingers within 1.4 mm (median) of the tube's axis, and on
  these tubes that is just enough to clip the neighbour. Version 1 had the opposite trouble (pads far
  too wide) and so never got this close.
- **Boxed-in tubes: refused 80% instead of 98%.** The same cause from the other side. Many tubes are now
  clear or blocked by a millimetre, so the labels flip on differences smaller than a pixel (2.3 mm), and
  the network is less sure of every answer. No cut-off fixes it: at 0.40 it refuses 99.3% of boxed-in
  tubes but also refuses 232 pickable ones. In physics, though, 2 of the 3 boxed-in tubes it tried lifted
  cleanly: the fast check is cautious, so part of this loss is on paper only.

**A thing seen in both versions:** after a lift, a neighbour often settles 5 to 8 degrees in its cell
(it was standing about 1 degree off before). Pulling the tube out jostles the rack, and a 25 mm tube in
a 30 mm cell has room to lean. The arm never touches it, and it stays in its cell. This happened in 3 of
version 1's 6 physics successes too.

---

## What is next

The remaining problems both come from labels that are right or wrong by a millimetre. The next step is
to **label a direction clear only if it is still clear with the fingers 2 mm to either side** (the
network's aim error). Then the network learns to pick only directions with room to spare. That should
cut the "aim off" grasps and make boxed-in tubes easier to tell apart, at the cost of fewer pickable
tubes. Narrowing the opening a little (32 mm instead of 35) could balance the room between the tube and
the neighbours.

**Step 1b still uses version 1.** Moving it to version 2 needs its arm to open the fingers before going in
and its camera to turn to the rack, as `side.run` does here.

## How to repeat this

From `step-1-grasp/`:

```
make data-side2       # about 20 minutes: 400 + 50 + 100 scenes, every picture turned to its rack
make train-side2      # about 16 minutes on the Mac
make evaluate-side2   # version 1 with mixing only (fast check), then version 2 (fast check + 60 physics)
make figures-side2    # the pictures on this page
make test             # 48 tests, 4 of them for version 2
```

## Log of runs

| Run | Date | What | Validation: right overall | Test: good grasps / pickable | Physics: pickable, reachable |
|---|---|---|---|---|---|
| `side_grconv` | 2026-10-28 | version 1 | 92.3% | 364 / 457 | 6 of 9 |
| `side_grconv_smoothed` | 2026-10-29 | version 1 network, choosing with mixed scores | 92.1% | 370 / 457 | not run |
| `side_v2` | 2026-10-29 | version 2: fingers opened 35 mm, no shove, directions from the rack, mixed scores, 16 epochs | 85.5% | **531 / 712** | **18 of 20** |

## Files changed

| File | What changed |
|---|---|
| `grasp/side.py` | `PRESHAPE_GAP`, `SideChecker(preshape=True)` (fingers opened 35 mm, no shove), `labels(frame_yaw=...)`, `rack_yaw` (the rack's angle from the close-up), `smoothed` (mixing the scores), `run(preshape=True)` |
| `grasp/scenes.py` | the close-up camera can be turned (`look(yaw=...)`) |
| `grasp/generate_data.py` | `--kind side2`: each picture turned to its rack, labelled in its own directions, into `data_side2/` |
| `grasp/train.py` | `--kind side2` |
| `grasp/evaluate_side.py` | `--version 2`, `--smooth`, `--out`; the pickable tubes split into good / wrong direction / aim off / wrongly refused |
| `grasp/figures_side.py` | `--version 2`: the pictures on this page, into `docs/side/images/v2/` |
| `tests/test_side.py` | the fingers open 35 mm; the rack's angle is read right; a turned camera turns the picture; mixing prefers the middle of a run |
