# Side Grasps: Results of All Versions

[Version 1](version1.md) · [Version 2](version2.md) · [Version 3](version3.md) · [All results](results.md) ·
[Overview](../../../docs/solutions/step-1-grasp.md)

How each version works is on its own page. This page is only about how well they work, and how each one was
improved. All versions were graded on the **same 1,151 test tubes**, in 100 crowded scenes they never saw.

## What the words mean

| Word | Meaning |
|---|---|
| **Pickable** | the fast check finds at least one clear direction. It tests the gripper and wrist only, **not whether the arm can get there** |
| **Boxed in** | no direction is clear: the right answer is to refuse |
| **Good grasp** | the network's choice is clear, and the pinch point is within 5 mm of the tube's axis |
| **Wrong direction** | the network picked a blocked direction while the tube had clear ones |
| **Aim off** | the direction was right, but the fingers were 1 or 2 mm off and touch a neighbour there |
| **Wrongly refused** | the network said "no clear direction" for a pickable tube |
| **Reachable** | the nearer arm could get to the network's first choice on its fixed route |
| **Full physics** | the simulated arm really moves, closes and lifts |

---

## The result in one table

| | Version 1 | Version 2 | **Version 3** |
|---|---|---|---|
| Pickable tubes | 457 (40%) | 712 (62%) | 712 (62%) |
| **Good grasps** | 364 | 531 | **558** |
| Good grasps, share of pickable | 79.6% | 74.6% | 78.4% |
| Wrong direction | 92 | 93 | 88 |
| Aim off | 1 | 85 | 49 |
| Wrongly refused | 0 | 3 | 17 |
| Boxed-in tubes refused | 98.4% (683 of 694) | 79.7% (350 of 439) | 83.1% (365 of 439) |
| Position error, median / 95% under | 1.7 / 2.9 mm | 1.4 / 2.6 mm | 1.4 / 2.5 mm |
| Physics: pickable tubes lifted cleanly, of reachable | 6 of 9 | 18 of 20 | 15 of 17 |
| Fast check agrees with physics | 90% | 93% | 100% (34 of 34) |
| Cut-off | 0.10 | 0.15 | 0.25 |
| Training | 25 min | 16 min | 20 min |

Version 2 gave the big jump, and version 3 added a little on top of it. Its main value shows up in the whole task.

## By tube size

| Tube | Pickable: V1 / V2 / V3 | Good grasp, of pickable: V1 / V2 / V3 |
|---|---|---|
| small (100 mm) | 20% / 38% / 38% | 77.8% / 75.9% / 75.9% |
| medium (150 mm) | 37% / 67% / 67% | 84.7% / 74.2% / 80.7% |
| large (200 mm) | 61% / 78% / 78% | 77.2% / 74.3% / 77.5% |

**Which grip height version 3 chose**, with the network alone: nearly always 3 cm.

| Tube | 3 cm | 5 cm | 7 cm |
|---|---|---|---|
| small | 142 | 16 | 11 |
| medium | 274 | 3 | 2 |
| large | 296 | 4 | 21 |

## In the whole task (Step 1b, 30 runs)

| | Using version 1 | Using version 3, carry planned ahead |
|---|---|---|
| All successes | 14 of 30 | **21 of 30** |
| Large tubes | 0 of 12 | **7 of 12** |
| Medium tubes | 13 of 17 | 13 of 17 |
| Small tubes | 1 of 1 (9 of 10 fell back to medium) | 1 of 1 (9 of 10 fell back to medium) |

These are the same 10 scenes that were used for tuning, so they may flatter the result. Failures are covered in
[Step 1b's results](../../../step-1b-pick-and-carry/docs/results.md).

---

## Why most tubes were blocked in version 1

Only 40% of tubes had a clear way in, and this analysis led to version 2.

![Why a side grasp is blocked](images/why_blocked.png)

*A tube on the outer row, the gripper come straight in from outside, seen from above. Left: fully open. Right:
nearly closed.*

**The rack is tight.** Its cells are 36 mm apart and the tubes 25 mm wide, so there is only **11 mm of air** between
a tube and its row neighbour.

**What touches first**, measured on 110 tubes, with the gripper coming straight in from outside the rack:

| What touches first | Fully open | Nearly closed |
|---|---|---|
| nothing: **clear** | 36 | **55** |
| a pad on a row neighbour, moving in | 35 | 8 |
| the inner finger link on a row neighbour, moving in | 7 | 27 |
| the wrist on a rack, tube or arm, on the way down | 19 | 19 |
| another part of the gripper on a tube, on the way down | 1 | 1 |
| the closing fingers on a neighbour | 12 | 0 |

**The same check over every direction**, on 169 tubes:

| Setting | Tubes with a clear direction |
|---|---|
| Fully open, 24 directions | 46% |
| Nearly closed, 24 directions | 54% |
| **Nearly closed, directions lined up with the rack too** | **64%** |

That prediction held: version 2's test tubes came out at 62%.

## What each change gave

| Change | Effect |
|---|---|
| Mixing scores only, on the version 1 network | 370 good grasps instead of 364 |
| Fingers opened 35 mm, directions from the rack | pickable tubes 40% to 62%, good grasps 370 to 531 |
| 72 outputs, same data | good grasps 531 to 558, aim off 85 to 49, but wrongly refused 3 to 17 |

**Why boxed-in tubes got harder (version 2).** Many tubes are now clear or blocked by a millimetre, so labels flip on
differences smaller than a pixel. No cut-off fixes it: at 0.40, version 2 refuses 99.3% of boxed-in tubes but also 232
pickable ones.

---

## Full physics

60 random test tubes per version. Each try is the network's **first choice, with the nearer arm**.

| | Version 1 | Version 2 | Version 3 |
|---|---|---|---|
| Pickable tubes in the sample | 19 | 37 | 37 |
| Tried by the network, all tubes | 19 | 41 | 40 |
| Arm could not reach it | 10 | 18 | 21 |
| Pickable tubes lifted cleanly, of reachable | 6 of 9 | 18 of 20 | 15 of 17 |
| Boxed-in tubes refused | 41 of 41 | 19 of 23 | 20 of 23 |

**About half of the network's choices could not be reached by the nearer arm.** This shows the network's first choice
and one arm only, so it does not say whether another direction or the other arm would have worked.

**The fast check is careful.** It has never called a direction clear that then failed in physics, in any version. When
it disagrees, it calls a good grasp blocked: 3 times in version 1 and 2 times in version 2.

## What we know and do not know yet

**Not yet measured: the score on tubes the arm can really take.** "Pickable" ignores the arm. A fair score would count
only tubes with a clear direction that an arm can reach and follow, for each tube size. A measurement was started
and stopped, so this number does not exist yet.

**Two later quick checks (2026-10-30):**

- **Small tubes.** 36 small-tube grasps version 3 rated good, tried in physics: the arm could not reach 23, 12 lifted
  cleanly, 1 did not lift, and none touched a neighbour. So the arm's reach looks like the main limit here. These use
  Step 1's route; Step 1b's checks are stricter.
- **Fully open fingers.** The 20 grasps version 2 lifted cleanly were run again with the fingers fully open, and all 20
  still lifted cleanly. So the 35 mm opening raises the number of tubes the fast check calls clear, but physics did
  not show that it was needed on these 20. The check's caution and the direction changes may matter as much.

---

## Log of runs

| Run | Date | What | Validation right | Test: good / pickable | Physics: pickable, reachable |
|---|---|---|---|---|---|
| `side_grconv` | 2026-10-28 | version 1 | 92.3% | 364 / 457 | 6 of 9 |
| `side_grconv_smoothed` | 2026-10-29 | version 1 network, mixed scores | 92.1% | 370 / 457 | not run |
| `side_v2` | 2026-10-29 | version 2 | 85.5% | 531 / 712 | 18 of 20 |
| `side_v3` | 2026-10-29 | version 3 | 86.2% | 558 / 712 | 15 of 17 |
