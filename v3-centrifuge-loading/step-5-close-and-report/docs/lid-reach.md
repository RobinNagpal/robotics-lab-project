# Step 5: Where Can the Centrifuge Stand So the Arm Can Close Its Lid? (2026-11-23)

The grip plan found the centrifuge's lid mostly out of reach
([grasp-plan results, section 5](../../grasp-plan/docs/results.md)). This study looks for a layout where the
arm can both **load every rotor hole** (Step 4) and **push the lid shut** (Step 5). Script: `lid_reach.py`;
raw results: `results/lid_reach.json`. The closing planner it uses: `close.py`.

## In short

- **As the centrifuge stands now, the arm cannot close the lid.** The lid opens away from the arm, to 99°,
  past upright, and its hinge is 0.83 m from the arm's base (the arm reaches about 0.85 m). Nothing about
  where or how it pushes changes that.
- **One layout does both jobs:** the centrifuge turned a quarter turn (+90°, the lid opens sideways, towards
  the balance) and moved 8 cm towards the arm. All six holes stay loadable, and the lid can be pushed shut
  from 99°.
- **But there the open lid hangs over the balance**, 33 mm from the balance holder's centre line, where Steps
  1 and 3 put tubes down and pour. With a stop so the lid rests at 85° (or 75°) instead of 99°, it is about
  65 mm away, and the lid can still be closed from there. Whether Steps 1 and 3 still work then is not tested.
- The scene is **unchanged** (the centrifuge as before). Step 4's 20 of 20 is for this layout. A decision is
  needed (below).

## A first study that was wrong

The first version of this study (earlier the same day) recommended turning the centrifuge −90°. It was wrong
on two counts:

1. It put the pinch point 8 mm from the lid's face. The closed fingertips reach 17.7 mm past the pinch point,
   so the arm really has to stand about 1 cm farther back. In the turned layout the closing planner then found
   no way to push the lid.
2. It did not check loading. With the centrifuge turned −90°, Step 4 could not load holes 0, 4 and 5: **20 of
   20 jobs failed** (`step-4-load-rotor/results/evaluation_plain_centrifuge_turned.json`).

The centrifuge was turned back. This version uses the real planners for both jobs.

## What was tried

| | Options |
|---|---|
| Which way it faces | as now (0°: lid opens away from the arm); turned ±30°, ±60°, ±90°, 180° |
| Where it stands | as now; 8 cm towards the arm |
| How far the lid rests open | 99° (now); 85° or 75° (a stop would be added) |
| Where the fingertips push | halfway along the lid; 70% of the way from the hinge |

- **The lid**: the closed fingertips on its inside face must follow the whole swing from its resting angle to
  shut, keeping 3 mm from everything but the lid. The fingers' turn about the push may change along the way.
  The lid turns the hand through up to 99°, and no single turn is reachable the whole way.
- **The holes**: for each of the six holes, a clear finger turn for the slide from its start to the bottom
  (Step 4's own check; empty rotor, a capped sample).

## Results

| Turned | Moved closer | Holes loadable | Lid can be closed from |
|---|---|---|---|
| 0° (now) | 0 or 8 cm | 6 of 6 | never |
| +30° or −30° | 0 or 8 cm | 6 of 6 | never |
| +60° | 0 / 8 cm | 5 / 6 of 6 | never |
| −60° | 0 / 8 cm | 4 / 5 of 6 | never |
| +90° | 0 cm | 5 of 6 | 75° only |
| **+90°** | **8 cm** | **6 of 6** | **99°, 85° or 75° (either push point)** |
| −90° | 0 / 8 cm | 3 / 4 of 6 | 75° only / any |
| 180° | 0 / 8 cm | 0 of 6 | 85° or 75° / never |

## The catch: the open lid beside the balance

Turned +90° and 8 cm closer, the lid opens towards the balance. How close the open lid comes to the balance
holder's centre line, at the heights where the hand works over the balance (10 to 45 cm):

| Lid rests at | Nearest point of the lid |
|---|---|
| 99° (now) | 33 mm: the gripper coming down to the holder would hit it |
| 85° | 66 mm |
| 75° | 65 mm |

## Options

1. **Turn +90°, 8 cm closer, lid stop at 85°.** Then rerun Steps 1 and 3 (do they still weigh and pour with
   the lid 66 mm from the holder?), make Step 4's start states again and rerun it, and build Step 5. About 1
   hour of runs at 4 workers. If Steps 1 or 3 are blocked, also move the balance (option 2).
2. **As option 1, and move the balance a few cm away from the centrifuge.** More room, but the SmolVLA
   training data (recorded with the balance where it is) would no longer match the scene.
3. **Leave the lid open.** Step 5 does only the report (the robot's balance number against the examiner's),
   and the lid is listed as not done. Then on to Step 6, the whole run. Cheapest; the scene stays as it is.
