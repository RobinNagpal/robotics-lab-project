# Grip Plan: Approach

**Why this exists.** In v1 we built a top-down grasp, trained a model for it, and only at the pour found that
the gripper covered the tube's opening. That cost three more versions of the grasp. So before any v3 step is
built, every grip is checked against **every place the job puts a tube**, all at once.

Results: [results.md](results.md).

## What is checked

Nothing moves under physics here. For each pose the arm is put there by inverse kinematics (the joint
angles that put the fingers exactly where we want, turned exactly the way we want), and the smallest gap
between the arm, the gripper or the held tube and everything else is measured. A gap under **3 mm** counts
as a touch: the real arm, Step 0's positions and the tube's angle in the fingers all have errors of about
that size.

| Part | The move | Where it is used |
|---|---|---|
| A | Lower a tube into its rack cell, let go, come back up (read backwards: pick it up) | Steps 1, 3, 4 |
| B | The same into the balance's holder | Steps 1, 3 |
| H | Slide a tube 6 cm along a rotor hole's 35° axis, let go, back out; with each neighbour layout | Step 4 |
| D | The water tube held from the side, tipped from 50° to 130° over a balance tube in the holder | Step 3 |
| E | The water tube picked from its rack cell from the side | Step 3 |
| L | The lid pushed or pinched and swung shut | Step 5 |

The rotor's top is checked as the solid metal it really is (its collision model in the simulation has only
thin webs between the holes, so a finger could pass through it there without the simulation noticing).

## The grips tried

The tube is round, so the arm can turn a tube in its fingers freely between two places. Each place picks
its own best **turn** (which way the fingers close, round the tube). What one grip fixes for the whole job is
only **where on the tube** the fingers are.

| Grip | Fingers | Pads on |
|---|---|---|
| **Under the cap** (the old plan in Step 1's approach.md) | along the tube, from its top | the body, just under the cap: grip point 35 mm under the top |
| **On the cap** | along the tube, from its top | the 35 mm wide cap (open tubes: the top 12 mm of the tube) |
| **Side** | across the tube | the body, 25 to 45 mm under the rim: the opening stays free |

Two facts about the Robotiq 2F-85 drive all of this:

- Its pads are **38 mm long**, and the grip point is in their middle. To keep the pads under the cap, the grip
  point must be at least 33 mm below the top, so the fingertips reach 50 mm down. A seated tube sticks out
  only 45 mm.
- Beside each pad is a **follower** link that sticks out further than the pad. In the rotor, the followers are
  what hit neighbouring tubes.

## Files

| File | What |
|---|---|
| `grasp_check.py` | the grips, inverse kinematics with a fixed turn, the gap measurement, the move "slide in, let go, back out" |
| `run_check.py` | every part above; saves `results/part_*.json` |
| `summarize.py` | prints the tables in results.md |

```bash
cd v3-centrifuge-loading
PYTHONPATH=$PWD/simulation-environment:$PWD/grasp-plan \
  pixi run --manifest-path simulation-environment/pixi.toml python grasp-plan/run_check.py --parts abhd --workers 4
pixi run --manifest-path simulation-environment/pixi.toml python grasp-plan/summarize.py
```

Parts A + C + D + E + G took about 10 minutes on 4 workers; H alone about 3.
