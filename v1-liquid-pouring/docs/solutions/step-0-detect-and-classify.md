# Step 0: Detect and Classify

The first step from the [list of steps](steps.md).
It follows the [problem statement](../problem-statement/problem-statement.md).

---

## Goal

From camera images, make a list of every object on the table with:

- **its class:** what kind of object it is,
- **its position:** the 3D point in the robot's base frame where it stands,
  accurate enough for the gripper to reach it (target: within 1 cm).

Example output:

```
beaker        at (0.52, -0.31, 0.05)
flask         at (0.52, -0.22, 0.05)
test_tube     at (0.48,  0.30, 0.08)   in rack_A
rack          at (0.48,  0.30, 0.00)   -> role: empty rack
pouring_stand at (0.60,  0.00, 0.00)
```

## What the arm is allowed to know

- **Allowed:** camera images (colour and depth), each camera's fixed position and lens settings
  (a real lab calibrates its cameras once), and the arm's own joint readings.
- **Not allowed:** anything from the scene's `Truth` record or MuJoCo body positions.
  These are used only to make training labels and to grade the result.

## Classes

| Class | Looks like |
|---|---|
| `test_tube` | narrow glass tube (sources and receivers look the same) |
| `centrifuge_tube` | plastic tube with a conical tip |
| `beaker` | wide, short glass cylinder |
| `flask` | wide base, narrow neck |
| `rack` | blue plastic rack with holes |
| `tray` | flat tray the beaker and flask stand on |
| `pouring_stand` | the fixed stand |

**Role is not a class.** A source tube and a receiving tube look the same, and so do
the three racks. We work out the role from what is around each object, after detection:

- a rack holding tubes with liquid is the **source rack**,
- a rack holding empty tubes is the **empty rack**,
- a rack with nothing in it is the **filled rack**,
- a container is a **source** if it holds liquid and a **receiver** if it is in the empty rack.

"Holds liquid" can be a yes/no answer here. How much liquid is Step 0.2.

## Cameras

| Camera | Where | Used for |
|---|---|---|
| **Overhead** | 1.2 m above the table, looking straight down | **finding and locating** (main camera) |
| **Front** | in front of the table, looking down at an angle | **checking the class** (side view shows shape) |
| Wrist | on the gripper | not used in Step 0; used for close-ups in 0.1 and 0.2 |

Why this split:

- From straight above, nothing hides anything else, and a pixel maps simply onto the table,
  so positions are most accurate there.
- From above, every container is a circle. A beaker is a bigger circle than a tube, but a flask
  and a beaker, or a test tube and a centrifuge tube, can be hard to tell apart. The front
  camera sees their side outlines.

**Decision:** start with the overhead camera alone. Add the front camera only if
classification from above is not accurate enough.

## How to locate an object (both methods use this)

Detection gives a 2D region in the image. To get a 3D position:

1. Take the object's pixels (a mask is better than a box: the centre of an open tube's box
   looks down into the tube, not at its rim).
2. Read the depth at those pixels from the overhead depth image.
3. Use the camera's known lens settings to turn each pixel plus depth into a 3D point
   in the camera frame.
4. Use the camera's known position to move these points into the robot's base frame.
5. The object's position is the centre of its points in x and y. The highest point gives its top.

## Method A: Programmed (written down, not built)

1. Render the overhead depth image.
2. Remove the table: every point at table height is background.
3. Group the remaining points into separate objects (points that touch belong to one object).
4. A rack and its tubes touch, so they come out as one group. Split them by height:
   the rack's top is one flat level, and anything sticking up above it is a tube.
   Each tube that sticks up is a separate small group.
5. For each group, measure its shape from depth only (no colour): the footprint size, the height,
   and whether it is a ring (an open container seen from above) or solid.
6. Classify with hand-written rules, for example:
   - large flat footprint with a grid of holes -> `rack`,
   - ring about 5 cm across and short -> `beaker`,
   - ring about 3 cm across, tall -> tube.

Weak points: objects touching each other merge into one group and need special rules
(like the rack split above), look-alike shapes are hard to tell apart with rules,
and every new object type needs new rules. Glass is see-through, so a real depth camera
often misses it. MuJoCo's depth image treats glass as solid, so this problem does not show
in the simulation.

Questions and answers about this step: [step-0-questions.md](step-0-questions.md).

## Method B: Learned (the one we build)

### Which learned model

All of these find the objects in the image. The position still comes from depth (see above),
except in option 5.

| # | Approach | Example models | Good | Bad |
|---|---|---|---|---|
| 1 | **One-stage segmentation** | YOLOv8-seg, YOLO11-seg | fast, easy to train, good tools | masks a bit rough on thin objects |
| 2 | **Two-stage segmentation** | Mask R-CNN (Detectron2) | often finer masks | slower, more setup |
| 3 | **Zero-shot, no training** | Grounding DINO + SAM ("find the beaker") | works with no training data | weak at telling lab glassware apart |
| 4 | **Depth-only network** | a small network on the height map, or PointNet++ on the 3D points | uses shape only, no colour at all | fewer ready-made tools |
| 5 | **Direct 3D prediction** | a network that outputs class + 3D position per object (CenterNet-style) | no separate depth maths | harder to train, usually less accurate |

A small variation of 1 or 2: give the model the depth image as an extra input next to colour.

**Decision:** build option 1 (YOLO-seg) first. Then compare it against option 3
(no training, quick to try) and the programmed method. Options 2, 4 and 5 are kept in case
YOLO is not good enough.

### How we build it

1. **Detector:** YOLO with segmentation (for example YOLOv8-seg or YOLO11-seg),
   so it gives a mask for each object as well as the class.
2. **Training data from the simulation, labelled for free:**
   - For each seed, render colour, depth and a segmentation image from the overhead camera.
     MuJoCo's segmentation image says which object each pixel belongs to.
   - Map each MuJoCo body to its class (`source_beaker` -> `beaker`, and so on).
   - Turn each object's pixels into a YOLO mask label.
3. **Variety in the scenes**, so the detector learns what objects look like and not where they
   usually are. The scene builder has to change, from seed to seed:
   - how many sources and receivers there are, and which kinds,
   - where the racks, tray and stand stand,
   - lighting, and the camera position a little.

   Right now only the liquid levels and the receivers' rack cells change, so this is
   the first piece of work before training.
4. **Locate** each detected mask with the steps above.
5. **Assign roles** with the rules above.

## Learned vs Programmed

Can the learned method beat the programmed one? **Partly.** It depends on the sub-task:

| Sub-task | Better method | Why |
|---|---|---|
| Finding the position (where) | **same** | both use the same depth maths |
| Separating touching objects (tube in rack) | **learned** | no special rules needed |
| Telling look-alike shapes apart | **learned** | rules get long and fragile |
| New objects or layouts | **learned** | more training data instead of new rules |
| Clean simulation, first try | **programmed** | works with no training, easy to debug |
| Real glass with a real camera | **learned** | real depth fails on glass |

In our clean simulation, the programmed method will probably be just as good at the start.
The learned method wins once the scenes get varied and crowded. It also wins when we move to
real cameras.

So the plan is a mix, as in published work: **a learned model for what and which pixels,
depth maths for where.** The programmed method stays as a baseline to compare against.

## Grading

Compare the arm's list with the `Truth` record and MuJoCo body positions, over many unseen seeds:

- **Count:** did it find every object, with nothing extra?
- **Class:** fraction of objects with the right class.
- **Role:** fraction of racks and containers with the right role.
- **Position:** distance between the found position and the true one (target: under 1 cm).
