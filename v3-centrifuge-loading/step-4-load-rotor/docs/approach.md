# Step 4 Approach: Load the Rotor

**Status: programmed version built**: 20 of 20 jobs loaded, 75 of 75 tubes seated ([results.md](results.md)). First physics test (one sample tube into one hole: 23 of 23 seated, all six holes;
[insert-test.md](insert-test.md)). The overview is in
[steps.md](../../docs/solutions/steps.md#step-4-load-the-rotor).

## Goal

Put each tube in its planned hole, fully seated: its tip at the hole's bottom, its axis along the hole's.

## The holes

Six holes, leaning 35° (the top leans in), 0.9 mm wider than the tube all round, with a 6 mm lead-in at the
top. A seated tube sticks out 45 mm (with its cap) to be held. Neighbouring caps are 13 mm apart.
Details: [objects.md](../../simulation-environment/docs/objects.md#the-centrifuge).

## The moves, for one tube

```
 pick (from its rack cell, or from the balance holder for a balance tube)
   grip on the cap (open balance tubes: on their top 12 mm)
 carry to a point back along the hole's axis (20 cm for holes 0 to 3, 10 cm for holes 4 and 5),
   tube turned to the hole's 35° slant, fingers turned into the clear window (below)
 slide in along the axis, slowly, watching the wrist force sensor
   pushed sideways (the tube touching the hole's wall)? ── move a little the other way
   pushed back along the axis (the tip at the bottom)? ── stop
 open, back out along the axis, then up
 check: rotor camera + Gemini Robotics-ER 2: "is the tube in hole 3 fully in?"
```

The arm reaches every hole along its axis (the tests check this, seated and 6 cm back).

**The grip and the loading order** (checked before building: [grip plan](../../grasp-plan/docs/results.md)). The Robotiq's pads are 38 mm
long and a seated tube sticks out only 45 mm, so the pads must be on the cap: a grip under the cap puts the
fingertips 12 mm into the rotor's top. The finger links beside the pads are what limit the rest:

| Neighbouring holes | Sample (capped) | Balance tube (open) |
|---|---|---|
| both empty | fingers within 45° of along-the-rim, 10 mm to spare | a 40° window, 7.5 mm |
| one full | a 25° window (towards the empty side), 8 mm | a 5 to 10° window, 3.5 mm |
| both full | **cannot go in** | **cannot go in** |

So **every tube needs an empty neighbouring hole when it goes in**, and balance tubes go in first, into holes
that are not next to each other. Step 2 plans the order; with up to 5 tubes it always exists (no job has more than 4).

## Why force sensing

The tube goes in 75 mm through a gap of 0.9 mm. Small errors in where Step 0 thinks the hole is (a few
millimetres) would jam it. The wrist force sensor tells which way the tube is touching, as a person feels
it, and the arm moves away from the touch (v2 used the same sensor for its threads). The lead-in at the top
of each hole catches a tube up to 3 mm off.

## A. Programmed

Inverse kinematics along the axis, with the force-guided slide above. Order: the planned holes in turn,
nearest first.

## B. π0.5 for the carry

π0.5, fine-tuned like Step 1 (demonstrations from A), takes the tube from the rack to the start of the slide,
told *"put the red sample tube in hole 3"*. The last few centimetres stay force-guided: tight insertions are
the weakest skill of today's open models (even Google reports 74 to 90 % on insertion with its best,
partners-only model).

## How it will be measured (`results.md`)

On 20 seeds: tubes seated (examiner: tip within 6 mm of the hole's bottom, axis within 3°); jams; tubes
dropped; neighbouring tubes disturbed; balls spilled from open balance tubes; time per tube.

## Open questions

- ~~Does a tube held only by its 12 mm cap stay lined up with the fingers when tilted to 35° and slid into
  a hole with 0.9 mm to spare?~~ **Answered 2026-11-23** ([insert-test.md](insert-test.md)): not with the
  gripper model as built. The tube turns about the line between the pads, about 1.4° a second, whatever the
  squeeze. With twist (torsional) friction on the pads it stays within 1.5° on arrival, and 23 of 23 tries
  seated, even with the hole 5 mm from where the arm thinks it is.
- Open balance tubes: held by their top with a 1 mm squeeze and water inside. Do they stay lined up, and
  does water spill at 35°?
- A hole with one full neighbour: only a 25° window of finger turns, 8 mm to spare.
