# Steps to Perform the Task

This file lists the steps the robot arm follows to fill one tube to 50%.
It follows the [problem statement](../problem-statement/problem-statement.md).

---

## Before Starting: What the Arm Has to Find Out

The arm starts knowing nothing about the scene (see the problem statement).
Before it can pour, it has to find out three things, each a separate step:

| Step | Question | Needed for |
|---|---|---|
| **0. Detect and classify** | What is on the table, and where? | everything after |
| 0.1 Measure size | How tall and wide is each container? | the 50% target, the grip height |
| 0.2 Estimate liquid level | How full is each source? | picking a source, the first tilt angle |

Steps 0.1 and 0.2 come later. They need Step 0's output first, because the arm
has to know where a container is before it can look at it up close.

The simulation also needs a way to count particles, for grading only:
how much liquid is in the receiving tube, how much is left in the source, and how much has spilled.

---

## Step 0: Detect and Classify

Find every object on the table, say what kind it is, and where it stands.
Details: [step-0-detect-and-classify.md](step-0-detect-and-classify.md).

---

## Step 1: Pick an Empty Tube and Place It on the Stand

1. Choose an empty tube from the empty rack.
2. Check its size (height). Work out the target amount for 50% of this tube.
3. Grip the tube and lift it out of the rack.
4. Place it upright on the fixed pouring stand.
5. Let go and check that it is standing straight and stable.

---

## Step 2: Pick a Source

1. Choose a source from the source rack.
2. Check how much liquid it has. The level is unknown (somewhere between 20% and 90%), so we need to measure or estimate it.
3. Grip the source firmly and lift it straight up so no liquid spills.
4. Move it over the receiving tube, with its lip (spout) just above the tube opening.

---

## Step 3: Tilt and Pour Until the Tube Reaches 50%

1. **Tilt slowly at first.** We do not know at which angle the liquid will start to flow, because the starting level is different every time.
2. **Watch for the first flow.** When the first particles start to leave the source, remember this angle.
3. **Control the flow.** Keep tilting slowly so the flow stays steady. Each shape (tube, beaker, flask) pours differently at the same angle, so adjust the speed for each shape.
4. **Keep checking the receiving tube level.** Compare it with the 50% target for this tube size.
5. **Stop early.** Some liquid is already falling through the air. Start tilting back *before* the tube reaches 50%, so the falling liquid fills the last part.
6. **Tilt back to stop the flow.** Turn the source back up until no more liquid comes out.
7. **Check the final level.** Wait for the liquid to settle, then check if the tube is at 50%.
   - If it is a little low, pour a small extra amount.
   - If it is at 50%, the pour is done.

---

## Step 4: If the Source Runs Out, Use a Second Source

1. **Notice the source is empty.** Signs: the source is tilted a lot but no liquid comes out, or the particle count in the source is near zero.
2. Turn the empty source back upright.
3. Return it to its place in the source rack.
4. Work out how much more liquid the tube still needs.
5. Pick a second source (go back to Step 2).
6. Pour only the missing amount (repeat Step 3).

---

## Step 5: Return the Source to Its Rack

1. Make sure the source is fully upright so nothing drips out.
2. Move it back to its place in the source rack.
3. Put it down gently and let go.

---

## Step 6: Move the Filled Tube to the Filled Rack

1. Grip the filled tube on the stand.
2. Lift it slowly and keep it upright so nothing spills.
3. Move it to a free spot in the filled rack.
4. Put it down gently and let go.

---

## Step 7: Check the Result

- **Final level:** how close the tube is to 50%.
- **Spill:** how many particles fell outside the tube.
- **Time:** how long the whole task took.

Then go back to Step 1 for the next empty tube.
