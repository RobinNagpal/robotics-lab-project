# Step 3b: The Whole Task with the Learned Pours (ACT and CNN + MLP)

The same whole task as [Step 3](../../step-3-put-back/docs/results.md), but the pour is done by one of
the two **learned policies** from [Step 2b](../../step-2b-pour-act/docs/results.md) instead of the
programmed pour with the custom CNN reading the level:

- **ACT** (Action Chunking with Transformers), 10.9 million weights;
- **CNN + MLP**, the simpler policy it is compared with, 5.0 million weights.

Both were trained on the same demonstrations for the same 30 minutes. Everything but the pour is the
same code as Step 3:

```
 STEP 0   the overhead camera (YOLO) finds the racks and tubes
 STEP 1b  both tubes picked up and brought together (from a saved run, as in Step 3)
 STEP 2b  the policy pours: it sees the cameras and moves Arm B's joints itself, until it says "finished"
 STEP 3   Arm B puts the source back in its own cell        (Step 3's code, unchanged)
 STEP 4   Arm A puts the receiver in the destination rack   (Step 3's code, unchanged)
 STEP 5   the check: level, spill, each tube in its cell, anything touched
```

**One change to make them fit.** Step 3 takes the source out of the pour the way the programmed pour
brought it in, backwards. A policy moves the arm in by itself, so there is no planned way in to reverse.
Here the programmed pour's way in is planned for the same start (no physics, nothing moves) and the
source leaves by that. The policies copy that way in closely, and Step 3 checks every move out for
collisions before making it.

If the pour fails (the policy never says "finished" within 60 s), the tubes are not put back, as in
Step 3.

```
make run              # ACT, the 6 test scenes, 4 at a time (about 3 minutes)
make run POLICY=mlp   # the CNN + MLP policy instead (about 3 minutes)
```

The results go to `runs/workflow/act/` and `runs/workflow/mlp/`, one file per scene.

## Results on the 6 test scenes

Target 50 %, window 48 to 52 %. The same 6 scenes as Step 3, never used for training.

### In short

| | ACT | CNN + MLP | Programmed pour + CNN (Step 3) |
|---|---|---|---|
| Pour finished on its own | 4 of 6 | 3 of 6 | 6 of 6 |
| Level in 48 to 52 % | 1 of 6 | 1 of 6 | 4 of 6 |
| Both tubes back in racks, when the pour finished | **4 of 4** | **3 of 3** | 6 of 6 |
| Balls lost or anything bumped while putting back | none | none | none |
| **Whole run passed** | **1 of 6** | **1 of 6** | **4 of 6** |

### Scene by scene

| Scene | Tubes | ACT | CNN + MLP | Step 3 (CNN) |
|---|---|---|---|---|
| 150 | large | 10.5 %, never said finished | 47.5 %, finished, put back | 50.5 %, **pass** |
| 151 | medium | 57.5 %, never said finished | 54.0 %, never said finished | 49.5 %, **pass** |
| 152 | medium | 56.0 %, finished, put back | 60.0 %, never said finished, **emptied the source** | 47.5 %, fail |
| 153 | large | 49.0 %, finished, put back, **pass** | 49.5 %, finished, put back, **pass** | 49.0 %, **pass** |
| 154 | medium | 45.5 %, finished, put back | 55.5 %, never said finished | 51.5 %, **pass** |
| 156 | large | 46.5 %, finished, put back | 47.0 %, finished, put back | 52.5 %, fail |

"Put back" means both tubes ended standing in the right cells: the source in its own cell of the source
rack, the receiver in the destination rack.

## What this shows

**1. The whole chain works with both learned pours.** Every time a policy finished its pour (7 runs
in all), Step 3's code put both tubes back in the right cells, with nothing bumped and no balls lost.
The way out planned from the programmed pour was safe after every learned pour.

**2. The learned pours are the weak part.** Both end near the target, but rarely inside the narrow
window (only 3 balls each way on a medium tube): 1 of 6 each, against 4 of 6 for the programmed pour
with the CNN. Both pass the same scene, 153.

**3. ACT and the CNN + MLP fail in different ways.**

- **The CNN + MLP does not stop on medium tubes.** On all 3 medium scenes it never said "finished", and
  it overfilled (54 to 60 %). On scene 152 it kept tipping, to 110 degrees, until the source was empty.
  On all 3 large scenes it stopped on its own, a little early (47 to 49.5 %). This matches the offline
  grading in Step 2b, where it said "finished" in only 49 % of the ticks after the pour had ended
  (ACT: 72 %).
- **ACT stops more often (4 of 6), but its two failures are odd.** On scene 150 it tipped to 85
  degrees, where balls only just start to flow, and stayed around there: after 60 s only 19 of the 103
  balls had gone in. On scene 151 it overfilled to 57.5 % and kept going.
- **ACT leans much more on the safety check.** The check refuses a move that would put Arm B or the
  source into Arm A or the receiver. It refused 336 of ACT's moves over the 6 scenes (252 on scene 154
  alone, about half that pour), and 25 of the CNN + MLP's.

**4. Copying well is not the same as pouring well.** Offline, both policies copied the expert's moves
closely (ACT a little better). Driving the arm themselves, small errors build up, and in this
narrow window that is enough to miss. This is the drift the
[Step 2b results](../../step-2b-pour-act/docs/results.md) warned about.

## What could be tried next

- **ACT's temporal ensembling** (averaging its overlapping 1 s predictions), the standard way to make
  ACT smoother when it drives the arm.
- **A safer stop**: end the pour when the level network reads the target, whatever the policy says.
  Most of the overfilled runs passed through 50 % on the way, so a programmed stop on top might bring
  many of them into the window. (It would not help scene 150, where ACT hardly poured.)
- **DAgger**: let the policy pour, ask the programmed pour what it should have done at each moment, and
  train on that. The expert can answer for any moment, because it is a program.
