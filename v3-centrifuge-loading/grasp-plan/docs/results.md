# Grip Plan: Results

How it is checked: [approach.md](approach.md). Checked 2026-11-17, before any step was built.
These checks ran on the scene as it was then (jobs with up to 5 samples, the water tube in a random cell). The
scene changes they led to (sections 4 and 6) only remove tubes from the racks and the rotor, so every
result here still holds or gets easier.

## In short

| Question | Answer |
|---|---|
| Grip for every tube that goes into the rotor | **On the cap** (open balance tubes: the top of the tube). The planned grip under the cap cannot load the rotor at all |
| Grip for pouring | **Side grip from behind** the water tube, 25 to 45 mm under its rim. The opening stays free |
| What limits the rotor | The gripper's finger links. **A tube can go in only if at least one of its two neighbouring holes is still empty** |
| What that means for the jobs | A **full rotor (6 tubes) cannot be loaded**: the last tube would have both neighbours full. Up to 5 can, in a planned order. The 5-sample "odd one" job (6 tubes) is dropped; every job now uses at most 4 |
| Not settled | The lid: from its resting angle it is mostly out of the arm's reach |

## 1. Rack cells and the balance holder: no problem

| | Result |
|---|---|
| Rack cells, every tube on 8 seeds (50 tubes), grip on the cap | 50 of 50 have clear turns (at least 6 of the 12 tried; 31 tubes have all 12). Smallest best gap 5 mm |
| Balance holder | all 12 turns clear, about 10 mm |

## 2. The rotor: the grip under the cap fails, the grip on the cap works with a rule

**Grip under the cap** (pads just under the cap, as planned): the fingertips go **12 mm into the rotor's top**,
in every hole, at every turn. The tube sticks out 45 mm; the pads reach 50 mm down.

**Grip on the cap**, by which neighbouring holes are already full (the opposite hole full too; the same result
in all six holes, so reach is not a problem):

| Held tube | Both neighbours empty | One neighbour full | Both neighbours full |
|---|---|---|---|
| Sample (capped) | turns within 45° of "fingers along the rim", **10 mm** | a 25° window, **8 mm** | **none** (followers 3 mm into the neighbour) |
| Balance tube (open, gripped 6 mm under the rim) | a 40° window, 7.5 mm | a 5 to 10° window, 3.5 mm (tight) | **none** |

Fingers closing towards the rotor's centre (radially) hit the rotor's centre knob; fingers along the rim hit the
neighbours. The clear turns are in between.

**The loading rule this gives:** every tube needs an empty neighbour when it goes in, and balance tubes should go
in while both neighbours are empty. With up to 5 tubes an order always exists (the last tube goes next to an
empty hole, and so on backwards). For example "mismatch" (2 samples, 2 balance tubes): balance tubes into holes
1 and 3 first, then the samples into 4 and 0. With 6 tubes the last
one always has both neighbours full.

**Getting there:** coming straight down the hole's axis is clear from 20 cm back for holes 0 to 3. Holes 4 and 5
(nearest the arm; their axes lean towards the lid) are clear only for the last 10 cm: further back the wrist
meets the open lid (3.8 mm at 10 cm).

## 3. The pour: side grip from behind works

The balance tube stands in the holder; the water tube's rim is held 15 mm above it and tipped about its lip.

| | Result |
|---|---|
| Fingers from behind (the side away from the balance tube), 25, 35 or 45 mm under the rim | clear from 50° to 130° for **11 of 12** pour directions, 10 mm |
| Fingers from the side (along the tipping axis) | the pads hit the balance tube's rim in most directions |
| Starting at 40° | the water tube comes within 0.5 mm of the balance tube: start at 50° (an 85 % full tube starts to flow at about 50°) |

A top-down grip would cover the opening again (v1's lesson), so the water tube is the one tube held from the side.

## 4. Picking the water tube from the side: the rack is too tight

| Grip height under the rim | Seeds where some approach is clear |
|---|---|
| 25 mm | 5 of 8 |
| 35 mm | 5 of 8 |
| 45 mm | 0 of 8 |

The gripper's body (76 mm across) hits the rack's top plate or a blank tube in the next cell.

**Fixed in the scene:** the water tube now always stands in a corner cell with the cells beside it empty
(`settings.WATER_CELL`). Each corner was checked on 8 seeds, gripping 25 and 35 mm under the rim:

| Corner cell | Seeds with a clear approach | Clear approach directions (of 24) |
|---|---|---|
| **0** (nearest the arm, chosen) | 8 of 8 | 7 |
| 3 | 8 of 8 | 5 |
| 7 | 8 of 8 | 3 |
| 4 | 8 of 8 | 2 |

## 5. The lid: not settled

The lid rests 99° open, past upright, so closing it starts with a pull towards the arm at its top edge. Its
hinge is 0.83 m from the arm's base; the UR5e reaches about 0.85 m.

| | Result |
|---|---|
| Pinch the front edge and swing it | out of reach from every resting angle 80° to 99° |
| Push it (any finger direction) from 99° | 5 of 8 angles along the swing reachable; the first part, near fully open, is not |
| Same with the centrifuge 14 cm closer | 6 of 8 |

## 6. What this changes in the plan

1. **Steps 1, 3, 4:** grip on the cap (open tubes: the top 12 mm), not under it. Step 1's approach.md said
   15 to 25 mm under the cap.
2. **Step 2:** the planner chooses holes **and the loading order**: each tube with an empty neighbour, balance
   tubes into non-adjacent holes first. At most 5 tubes (the jobs use at most 4).
3. **The scene:** the "odd one" job uses 3 samples only (a pair plus one), so no job needs 6 tubes. Done.
4. **Step 3:** the water tube needs room for a side grip. Done: it stands in the supply rack's cell 0, with the
   cells beside it empty (simpler than a new stand, and every seed has 7 clear approaches).
5. **Step 4:** the approach is straight down the hole's axis (20 cm for holes 0 to 3, 10 cm for 4 and 5), the
   fingers turned into the clear window for that hole's neighbours.
6. **Step 5:** the lid needs its own small study (turn the centrifuge so the lid opens sideways, move it
   closer, or start the lid less open).

## 7. Physics check: does the tube stay straight in the fingers? (2026-11-17)

Full physics: pick the tube from its rack cell (fingers 45 mm open, then closed), lift 15 cm, tilt to 35° (the
rotor's slant), hold 2 s. Measured: the angle between the tube and the fingers' axis, and how far the tube's top
slid along the fingers.

| Tube, grip | Squeeze | Angle in the fingers at 35°, after 2 s | Slid |
|---|---|---|---|
| sample (seed 0, two tubes), on the cap | 4 mm | 0.15° and 1.07° | 0 mm |
| water tube (85 % full, the heaviest open tube), top 12 mm | 0 / **1** / 2 mm | 0.76° / **0.61°** / 1.47° | 0 mm |
| water tube, top 12 mm | 3 or 4 mm | **thrown out of the fingers** | |
| empty tube, top 12 mm | 0 / **1** / 2 / 3 mm | 0.99° / **1.19°** / 1.37° / 0.09° | 0 mm |
| empty tube, top 12 mm | 4 mm | **thrown out of the fingers** | |

- **The cap grip holds.** No tube slides, and none turns as it is tilted: the angle is set when the fingers
  close (up to about 1.4°) and then stays.
- **Open tubes need a 1 mm squeeze, not 4.** The simulated tube wall is 2 mm thick; at the narrow threaded top a
  4 mm squeeze pushes the pads through it and the solver throws the tube. This is the simulation, not real
  plastic, so the fix is in how hard we squeeze.
- **About 1° in the fingers is more than the hole forgives without help** (0.9 mm over 75 mm is about 1.4°, and
  the lead-in takes 3 mm at the top). Step 4's force-guided slide must correct it, as planned.
