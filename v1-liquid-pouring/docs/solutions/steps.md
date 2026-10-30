# Steps to Perform the Task

This file lists the steps the two arms follow to fill one receiving tube to 50%.
It follows the [problem statement](../problem-statement/problem-statement.md).

---

## Simplifications for v1

- **Matching pairs.** For every receiving tube in the empty rack there is a source tube of the
  same height in the source rack (for example one small and one medium in each).
  Each pour goes from a source to the receiver of the same size.
- **Always enough liquid.** Every source starts more than half full. One pour always fills its
  receiver to 50%, so the arms never need a second source.
- **Target.** "50%" means 50% of the receiving tube's inside height. The source keeps whatever is left.

Because of this, the arms do not need to know how full a source is before pouring.
They only need to see the receiver's level while it fills.

---

## Step 0: Detect and Classify

Find every rack and tube on the table, say what kind it is, where it stands, and how high its top is.
From that follow the roles: source rack, empty rack, destination rack, and which tube is a source or a receiver.
Details: [step-0-detect-and-classify.md](step-0-detect-and-classify.md).

### Step 0.1: Tube Size

Say whether each tube is the small (100 mm), medium (150 mm) or large (200 mm) one.
A tube's top height (from Step 0) minus the height of the rack floor it stands on gives its height.
This is what pairs each receiver with its source. Built in Step 1b, with which tubes hold liquid
(the wrist camera looks straight down into each) and which rack is which.

---

## Step 1: Both Arms Pick and Meet at the Pouring Point

Built in two parts:

- **The grasp** (where the fingers close and at which turn): [step-1-grasp.md](step-1-grasp.md).
- **Both arms pick and meet at the pouring point** (above the middle of the table, 35 cm up), with the
  coordinator below: [step-1b-pick-and-carry.md](step-1b-pick-and-carry.md). Both arms hold their tube
  from the side (Step 1's side-grasp network), each with its own wrist camera, so the openings are free
  for the pour. Works for the medium pair; large tubes not yet.

### Who decides: one coordinator

The two arms do not talk to each other. **One coordinator** (a single program) runs both arms. It
holds the shared list of what is on the table (from Step 0), gives each arm its job, and waits for
each arm to report back: "picked", "at the waiting spot", "holding at the pouring point". This is
the usual way to run two arms; in a real lab it would be a planner sending commands and each arm
sending back its status (as in ROS).

### Choosing the pair, before anything moves

1. From Step 0 and Step 0.1, list the receivers and the sources, with their sizes.
   For example: receivers medium and large, sources medium and large.
2. Choose a size that is in both, for example medium. **Arm A** is given the medium receiver,
   **Arm B** the medium source.
3. Each arm looks at its tube up close and runs the grasp model. If either tube is boxed in (no
   clear turn), the coordinator chooses another pair, for example the large ones, before either
   arm has moved a tube.

### Who moves when

| # | Arm A (receiver, arm 2) | Arm B (source, arm 1) |
|---|---|---|
| 1 | waits at home | grasps the source from the side, lifts it straight up, parks it behind the source rack, reports **"waiting"** |
| 2 | grasps the receiver from the side, carries it to the pouring point, holds it upright and still, reports **"ready"** | waits |
| 3 | keeps holding, all through the pour | carries the source to a waiting spot beside the middle, then moves in beside the receiver: **"in place"** |

**One arm moves at a time.** Held from the side, both arms reach far across the table, and with both
moving at once they touched in testing. Each arm takes the tube on its own side of the table.

If something goes wrong (a pick fails, a tube slips), that arm reports it and the coordinator
decides: try again, or choose another pair. (Choosing another pair before moving is built; retrying
after a failed pick is not yet.)

---

## Step 2: Tilt and Pour Until the Receiver Reaches 50%

Planned, not built yet: [step-2-pour.md](step-2-pour.md) (the plan, with the decisions still open).
The weight from Arm A's wrist (programmed) is compared with a camera CNN (learned), both feeding the
same controller.

1. **Tilt slowly at first.** The angle at which the liquid starts to flow depends on how full the source is,
   and that is not known.
2. **Watch for the first flow.** When the first balls leave the source, remember this angle.
3. **Keep the flow steady.** Keep tilting slowly. Each tube size pours differently at the same angle.
4. **Keep checking the receiver's level** and compare it with its 50% mark.
5. **Stop early.** Some balls are still falling through the air. Start tilting back *before* the receiver
   reaches 50%, so the falling balls fill the last part.
6. **Tilt back to stop the flow.** Turn the source upright until nothing more comes out.
7. **Check the final level.** Let the balls settle, then check the receiver.
   - If it is a little low, pour a small extra amount.
   - If it is at 50%, the pour is done.

---

## Step 3: Return the Source to Its Rack

1. Make sure the source is fully upright so nothing falls out.
2. Arm B moves it back to its cell in the source rack, puts it down gently and lets go.

---

## Step 4: Put the Filled Tube in the Destination Rack

1. Arm A lifts the filled receiver slowly and keeps it upright so nothing spills.
2. It moves it to a free cell in the destination rack, puts it down gently and lets go.

---

## Step 5: Check the Result

These are measured by counting balls in the simulation. The arms never see them; they are only for grading.

- **Final level:** how close the receiver is to 50%.
- **Spill:** how many balls fell outside both tubes.
- **Time:** how long the whole task took.

Then go back to Step 1 for the next pair.
