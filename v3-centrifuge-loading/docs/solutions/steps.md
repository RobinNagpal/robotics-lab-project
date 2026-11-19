# Steps to Perform the Task

This file lists the steps the arm follows to load the centrifuge safely, and how each will be built. It
follows the [problem statement](../problem-statement/problem-statement.md). Each step has its own folder
with the code, an `approach.md` (how it is done, in detail) and, once built, a `results.md`.

**Status: the simulation is built and tested; the grips are checked against the whole job ([grip plan](../../grasp-plan/docs/results.md)); Step 2's solver is built (100 of 100 seeds planned right); Step 1's programmed arm weighs and puts back 60 of 60 tubes on 20 seeds; Step 3's programmed pour fills 27 of 30 balance tubes within 0.5 g, and 18 of 20 rotors balance. The other steps are planned, not built yet.**

---

## The whole job at a glance

```
 FIND (Step 0)        3 pictures ──► Gemini Robotics-ER 2 (points) ──► what each tube is
                      (overhead,      or SAM 3 (outlines)              (sample / water / empty, colour)
                       racks high,
                       racks low)  ── pixel + depth picture ──► 3D point ──► nearest known rack cell

 WEIGH (Step 1)       for each sample: rack ──► balance holder ──► read grams ──► back to its cell
                      (π0.5 moves the arm; a programmed version to compare)

 PLAN (Step 2)        weights ──► balance solver ──► which hole each tube goes in,
                                                     and the weight each balance tube must reach
                      Gemini Robotics-ER 2 decides with the solver, and explains the plan in words

 FILL (Step 3)        for each balance tube: empty tube ──► balance holder
                      pour water from the water tube, watching the balance, until it matches its partner

 LOAD (Step 4)        for each tube, in the planned order: pick ──► line up with its hole's 35° axis
                      ──► slide in, feeling with the wrist force sensor ──► let go ──► check it is seated

 CLOSE (Step 5)       push the lid down at its front edge until the latch catches
                      report: "Balanced: 0.08 g out ✓" (from the arm's own weights) ── examiner checks

 WHOLE RUN (Step 6)   Gemini Robotics-ER 2 runs Steps 0 to 5 as functions, in its own order, and recovers
                      from failures
```

Which AI model is used at each step, and why: [models.md](../libraries/models.md).

---

## Simplifications for v3

- **One arm.** Everything is in its reach; it does one thing at a time.
- **The lid starts open** and the rotor starts empty.
- **Balance tubes are left open** (not capped). The tests check they do not spill at 35° when as full as
  any job needs. A real lab would cap them.
- **The rotor never spins.** The run ends when the lid is latched; balance is worked out from the weights.
- **Water is poured from a tube**, not pipetted.
- **Every job has an answer** with at most **4 tubes** in the rotor (at most two balance tubes). The gripper's
  fingers need an empty hole beside the hole a tube goes into, so a full rotor (6 tubes) could not be loaded.
- **Two water tubes**, one for each balance tube (two balance tubes can need more water than one tube holds),
  stand in fixed corners of the supply rack with the cells beside them empty: they are held from the side to
  pour, and a side grip needs the room.

**The grips were checked against the whole job before any step was built** (v1 found at the pour that its
grip covered the tube's opening): [grip plan](../../grasp-plan/docs/results.md). Tubes that go into the rotor
are held by the cap; the water tubes are held from the side.

---

## Step 0: Find Everything

From three pictures (overhead, and a high and a low camera on the racks), find which rack cells hold a tube
and what each is (sample: capped, coloured liquid; empty tube; water tube). Where the racks, balance and
centrifuge stand is known, like a real work cell; the job is not. Each point Gemini gives becomes a cell using
the depth picture and the camera's known position, and the pictures vote cell by cell.

Built with **Gemini Robotics-ER 2** (no training: it is asked in words and answers with points and boxes),
compared with **SAM 3**. v1 trained its own detector (YOLO) for this; here nothing is trained.
Details: [step-0-find/docs/approach.md](../../step-0-find/docs/approach.md).

Where the cameras stand, what each sees and why three are used: [the cameras](../problem-statement/problem-statement.md#the-cameras).
What the arm knows before it looks, and how a pixel becomes a place in the room, is explained in the
[problem statement](../problem-statement/problem-statement.md#what-the-arm-knows-at-the-start). For objects
that could be anywhere (a tube knocked over), the same outlines and depth give a **point cloud**, and a fitted
cylinder gives the tube's place, direction and size with no layout
([results, section 6](../../step-0-find/docs/results.md#6-a-tubes-place-direction-and-size-from-its-outline-point-cloud)).

## Step 1: Weigh Each Sample

For each sample: pick it from its cell, stand it in the balance's holder, let go, wait for a steady reading,
pick it up again and put it back in its cell. The weights go to Step 2.

Built twice: a **programmed** version (inverse kinematics and a grip on the cap), and
**π0.5** fine-tuned on demonstrations recorded from the programmed version, told what to do in a sentence.
Details: [step-1-weigh/docs/approach.md](../../step-1-weigh/docs/approach.md).

## Step 2: Plan the Load

From the weights: which hole each tube goes in, and whether balance tubes are needed and how heavy each must
be, and **the order to load them** (each tube needs an empty hole beside it when it goes in). A small exact
**solver** tries every placement (six holes, at most five tubes: at most 720 options) and adds balance tubes
only if no placement of the samples alone is within 1.0 g. Samples are never changed,
only balance tubes. **Gemini Robotics-ER 2** gets the weights and the solver as a tool, decides, and explains
the plan in plain words. Also measured: Gemini without the solver.
Details: [step-2-plan/docs/approach.md](../../step-2-plan/docs/approach.md).

## Step 3: Fill a Balance Tube

For each balance tube in the plan: stand an empty tube in the balance's holder, weigh it, then pick up a
water tube from the side and pour into it, watching the balance, until the tube weighs what the plan says
(within 0.5 g). Like v1's pour, but with a much better sensor: the balance instead of a camera reading the
level. The "water" is balls, which come in bursts, so the pour goes in rounds: creep until a ball lands, let it
run or tip back, read the settled weight. With two balance tubes, the second one's target is worked out again
from what the first weighs. **Built (programmed):** 27 of 30 balance tubes within 0.5 g, 18 of 20 rotors
balanced, 3.7 balls spilled a seed.
Details: [approach.md](../../step-3-fill-balance-tube/docs/approach.md),
[results.md](../../step-3-fill-balance-tube/docs/results.md).

## Step 4: Load the Rotor

For each tube, in the planned order: pick it, bring it above its hole lined up with the hole's 35° axis,
slide it in along the axis while the wrist force sensor watches for sideways push (and corrects it), let go
when it reaches the bottom, and back out. The rotor camera and Gemini check each tube is seated.
Built programmed first; then π0.5 for the carry, with the force-guided last few centimetres kept.
Details: [step-4-load-rotor/docs/approach.md](../../step-4-load-rotor/docs/approach.md).

## Step 5: Close the Lid and Report

Push the lid down at its front edge until the latch catches. Report the balance from the arm's own weights,
for example "Balanced: 0.08 g out ✓". The examiner checks the same with the simulator's true weights.
Details: [step-5-close-and-report/docs/approach.md](../../step-5-close-and-report/docs/approach.md).

## Step 6: The Whole Run

**Gemini Robotics-ER 2** as the coordinator: Steps 0 to 5 become functions it can call ("find_tubes",
"weigh", "plan_load", "fill_balance_tube", "load", "close_lid"). It decides the order, retries when
something fails, and writes the final report. Compared with a fixed Python coordinator on the same seeds.
Details: [step-6-whole-run/docs/approach.md](../../step-6-whole-run/docs/approach.md).

---

## Side experiment: a world model (V-JEPA 2-AC)

If time allows: Meta's V-JEPA 2-AC for one move (put a tube in the balance holder), given only a picture of
the goal. It plans by imagining where each candidate move leads, with no demonstrations of the task. It was
trained on real robot videos, so it may not work on simulated pictures; finding that out is part of the
lesson.

## Order of building

1. Steps 0, 2 and 5 first: they need no GPU and give the demo its "it decides" part.
2. Step 1, 3 and 4 programmed versions: with them the whole job runs end to end (the demo is safe).
3. Step 6 with Gemini as coordinator.
4. π0.5 for Steps 1 and 4, once a GPU is available.
