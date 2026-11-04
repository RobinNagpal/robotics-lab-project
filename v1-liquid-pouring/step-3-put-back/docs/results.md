# Steps 3 to 5 Results: Put Both Tubes Back

How well the whole run works, from the end of Step 1b to both tubes in racks, with the custom CNN reading the
pour. The overview is in [step-3-put-back.md](../../docs/solutions/step-3-put-back.md); the settings are in
[approach.md](approach.md).

---

## The test

| | |
|---|---|
| Scenes | 6 saved Step 1b runs of test scenes (seeds 150 to 159, never used to train or choose the CNN): 3 medium pairs (151, 152, 154), 3 large (150, 153, 156) |
| Each | Step 0 again on the scene before anything moved; the pour to 50% read by the custom CNN; Step 3; Step 4; the check |
| Run | 4 at a time on the Mac, about 2 minutes each (one 7.5 minutes: see below) |

---

## Scene by scene

| Scene | Size | Level (CNN read) | Source: cell, off centre, lean | Receiver: cell, off centre, lean | Touched | Passed |
|---|---|---|---|---|---|---|
| 150 | large | **50.5%** (51.3) | own cell 8, 2.9 mm, 6.0° | cell 9, 2.4 mm, 5.1° | nothing | **yes** |
| 151 | medium | **49.5%** (48.7) | own cell 1, 2.5 mm, 2.2° | cell 9, 2.6 mm, 0.2° | nothing | **yes** |
| 152 | medium | 47.5% (49.5) | own cell 8, 4.5 mm, 6.3° | cell 9, 2.5 mm, 1.4° | nothing | level |
| 153 | large | **49.0%** (48.4) | own cell 1, 3.4 mm, 6.1° | cell 9, 2.2 mm, 0.6° | nothing | **yes** |
| 154 | medium | **51.5%** (49.1) | own cell 6, 1.7 mm, 5.0° | cell 8, 2.3 mm, 0.7° | nothing | **yes** |
| 156 | large | 52.5% (50.4) | own cell 8, 2.1 mm, 5.0° | cell 9, 2.9 mm, 5.2° | nothing | level |

- **Both tubes went back in all 6**: the source into its own cell, the receiver into the destination rack,
  each standing on its cell's floor (within 1 mm) and leaning no more than the cell lets it (a 25 mm tube in a
  30 mm cell leans up to about 7 degrees against a wall).
- **No ball was lost** putting the tubes back. One ball fell outside both tubes during the pour in scene 156.
- **The two failures are the pour's level**, about half a point outside the window each: the CNN read 2.0 points
  high (scene 152) and 2.1 points low (scene 156) at the end. See the pour's results in
  [step-2-pour/docs/results.md](../../step-2-pour/docs/results.md).
- **Time**, in simulated seconds: Step 1b about 65, the pour about 22, Steps 3 and 4 about 55 to 62.
- **Planning** took about 1 s per tube, except one receiver (scene 154), where the nearest destination cell had
  no clear way and 768 ways were tried before the next cell passed (340 s).

### Touching at the end of the pour

With all 3 medium pairs, the two grippers were already touching when the pour ended (Arm B's left coupler on
Arm A's left finger, less than 0.1 mm deep). With large pairs they were not. The pour's checks cover its way in,
not the tipping, so it was not seen before. The put-back's first move (straight up, the gripper's angle kept) is
allowed to start so, if no contact gets deeper and it ends clear; in all 3 it did, and nothing was touched after.

---

## How it got here

The 6 scenes were run four times; after each, the failures were traced to their cause (on single scenes,
saved where the pour ends), fixed, and all 6 run again.

| Run | Both tubes back | What failed | Fix |
|---|---|---|---|
| 1 | 2 of 6 | tipping back, Arm B touched Arm A in the 3 medium scenes, and in one knocked the source out; the source could not be planned in 2 more; in one the receiver dragged its rack 19 mm after letting go | lift first and tip back along the pour's own way in; let that first move start in the pour's contact; back away the way the fingers point (tracing contacts showed the palm, not the fingers, pushing the tube); try every turn of the gripper |
| 2 | 5 of 6 | scene 152: straight up rested the wrist harder on the forearm; two medium receivers took 8 to 11 minutes to plan | smaller lifts, and lifts aside |
| 3 | 6 of 6 | the same slow planning: the destination rack's far cells are out of reach for a medium tube | cells nearest Arm A first |
| 4 | **6 of 6** | | |

Earlier, before the first run, one scene (150) was used to find the approach:

| Tried | Result |
|---|---|
| Carry with the fingers level, as Step 1b | the source leans 13 degrees (it has turned in the fingers): cannot go into a cell |
| Carry the tube fully upright | near the pouring point, Arm B's wrist rests on its forearm (up to 23 mm deep) or a joint must swing too far, at every turn of the gripper tried |
| Turn the gripper round the tube first, then carry | the wrist folds, or the tube hits Arm B's own forearm |
| **Carry half upright, stand it up just above the cell** | clear: just above the cell the arm is bent less, and fully upright never folds there |

---

## What is left

- **The pour's level** (4 of 6 here; 12 of 17 in Step 2's own test of the CNN pour). The CNN's final reading is
  off by about 1 point on average and up to 2.4.
- **The grippers touching in medium pours**: Step 2 should keep them apart and check the tipping.
- **Step 1b in the same go**, on new scenes, instead of from saved runs.
