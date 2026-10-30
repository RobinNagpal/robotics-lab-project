# Step 1b Results: Both Arms Pick and Meet at the Pouring Point

How it works is in [approach.md](approach.md). This page is only about how well it works.

**30 runs:** 10 task scenes (seeds 0 to 9), each run three times, asking first for small, medium or
large tubes. Every run starts from the overhead picture, with the arms knowing nothing else about the
table. Both tubes are held **from the side** (Step 1's side-grasp network), so their openings are free
for the pour.

---

## What the words mean

| Word | Meaning |
|---|---|
| **Run** | one full go: perceive, choose a pair, Arm B picks the source and parks it, Arm A picks the receiver and holds it at the pouring point, Arm B comes in beside it |
| **Success** | both tubes in place, same size, from the right racks, each rim within 5 mm of where it should be, both within 5 degrees of upright, no liquid lost, and neither arm touched anything but its own tube |
| **Fell back** | the size asked for could not be taken safely, so the coordinator chose another size before anything moved |
| **Rim error** | the distance between where a tube's rim ended up and where the coordinator wanted it |

---

## The result in one table

| | |
|---|---|
| Successes | **14 of 30** |
| Successes, medium tubes | **13 of 17** |
| Successes, large tubes | **0 of 12** |
| Successes, small tubes | 1 of 1 (9 of 10 runs asking for small fell back to medium) |
| Runs where liquid was lost | **none** |
| Runs where an arm touched anything but its own tube | 4 of 30 |
| In the successful runs: receiver's rim from the pouring point | median **0.8 mm**, at most 1.4 mm |
| In the successful runs: source's rim from its meeting place | median 3.6 mm, at most 4.1 mm |
| In the successful runs: lean | at most 3.5 degrees |
| Distance between the two tubes' axes at the meeting | 85 mm |
| Time until Arm B is in place | about 67 s (simulated); about 100 s of computer time per run |

**In short: the medium pair works, the large pair does not yet.** When a run completes, both tubes are
placed within a few millimetres, upright, with no liquid lost. Every large-tube run stopped because one
of the arms could not find a route it could follow, not because of the grasp.

---

## 1. Why the large tubes fail

| What stopped the run | Runs |
|---|---|
| Arm 2, carrying the tall receiver, would have to swing a joint by about 2.7 rad on the way (near an awkward pose of the arm) | 6 |
| Arm 2 cannot reach the pouring point with the fingers level from the direction it needs | 2 |
| Arm 1, carrying the tall source, would hit its own forearm with the tube | 2 |
| Arm 1 would have to swing a joint while lifting the tall source | 2 |

A tall tube has to be held high to clear the rims of the others (its bottom 4 cm above the tallest rim,
so the fingers about 42 cm up), and held from the side, 17 cm of tube hangs below the fingers. At those
heights, and with the fingers level, the arms are near the edge of what they can do smoothly. The
planner refuses such moves before the arm makes them, so none of these runs spilled; in 4 of them an
arm had brushed a rack or a tube earlier on the way.

## 2. The 4 medium runs that were not a success

- 2 runs (scene 2): the route check refused the lift, because a finger pad would brush the tall
  neighbour in the next row.
- 2 runs (scene 1): the source ended tilted 8.7 degrees, far from its place.

## 3. Small tubes

A short tube is gripped only 2 cm above the rack's top plate, with tall neighbours beside it. In 9 of 10
runs asking for small tubes, no side direction passed the arm's check, and the coordinator took the
medium pair instead, before anything moved.

---

## How it got here

| Version | Successes | What changed |
|---|---|---|
| Top-down grasps (Step 1's first network) | 24 of 30 | the first version; **but a top-down grasp leaves the tube's opening under the gripper's base, so it cannot pour**: replaced |
| Side grasps, both arms moving at once | 9 of 30 | the arms actually touched each other: each move was checked at its start, but the other arm kept moving into its way |
| One arm at a time, Arm A first | 4 of 30 | Arm B could not get past Arm A in the middle; and the tubes sat about 4 mm further back in the fingers than aimed for |
| Arm B first and parked on its own side, arms swapped to their own sides, aim corrected | **14 of 30** | each arm takes the tube on its own side; the held tube is now also checked against its own arm |

Along the way, the side-grasp arm needed: the closing fingers checked with 2 cm of room (the passive
finger linkage swings out under a full tube's load), the pick planned once and followed exactly, the
fingers aimed 15 mm past where the tube's axis should be (where the pads hold it), the pouring point
raised from 25 to 35 cm (the wrists reach out over the racks), and the source held from +y and the
receiver from -y.

## What is next

- **Large tubes:** carry them lower where there is room (only the tubes they pass over need clearing),
  and give the arms more starting poses for the joint search at high, level-finger poses.
- **Then Step 2, the pour.**

## How to repeat this

From `step-1b-pick-and-carry/` (needs Step 0's detector and Step 1's side-grasp network next door):

```
make run SEED=0 SIZE=0.15   # one run, about 1.5 minutes
make evaluate               # the 30 runs, about 7 minutes on 8 cores
make test                   # the tests
```
