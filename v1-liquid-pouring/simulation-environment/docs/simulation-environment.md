# Simulation Environment: How It Is Built and Why

The code is in [`simulation-environment/`](../README.md). This page explains the main choices,
so nobody undoes them by accident.

## Where the objects come from

We did not make the objects from nothing. We took free 3D models (shapes made of small triangles,
called **meshes**) for how things look:

| From | What we take |
|---|---|
| [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) | the UR5e arm and the Robotiq gripper |
| [AutoBio](https://github.com/autobio-bench/AutoBio) | the tube rack |
| [Lab Assets](https://opengameart.org/content/lab-assets) (free for any use) | the test tube (all six tubes use it, stretched to their heights) |

`make assets` downloads them into `assets/`, always the same version.

## Two arms

The two arms face each other, 0.9 m apart, and everything they work on stands in a row
halfway between them. So either arm can do any job. In the task one arm holds the source
and the other the receiver, and they meet at the **pouring point**: the empty middle of
the row, 0.25 m up. Nothing stands there, so the pour happens in the air.

Moving an arm needs some care:

- **Each arm has its own names**: `arm1/...` and `arm2/...`, for its joints, motors, gripper and camera.
- **Move in small steps, through a high point.** Jumping straight to a far target swings the arm
  the shortest way for its joints, and that can sweep it low through the racks.
  `kinematics.move_to` slides the arm there gradually.
- **Point the fingers the right way.** `kinematics.move_to` takes the direction the fingers
  should point (straight down unless told otherwise). Holding a tube from above puts the
  gripper right over its rim, where the source has to pour in, so at the pouring point the
  receiver arm should hold its tube from the side. Both arms reach the pouring point either way.
- **Keep the joint angles real.** When working out the joint angles for a gripper position
  (**inverse kinematics**), the answer can spin a joint through several full turns, which a real arm cannot do.
  The solver keeps each joint within its limits and as close as possible to where it already is.

## Gripping a tube

Two things make the difference between lifting a tube and throwing it across the room:

- **Do not close the gripper all the way.** At 255 the fingers try to close to 0 mm. On a
  25 mm tube that means 25 mm of squeeze; the finger links wrap round the tube and the
  solver flings it away. `scene.grip_command(width)` gives the command that closes the
  fingers 4 mm narrower than the tube, which holds it firmly. The table behind it
  (`GRIPPER_GAP`) was measured with nothing between the fingers: 85 mm open, 25 mm at 180.
- **Keep tubes apart.** Open, the fingers reach about 60 mm to either side. With tubes one
  cell (72 mm) apart in a row, a finger hit the next tube on the way down. So tubes in one
  row stand at least three cells apart. Also keep the finger pads above the rack's top plate:
  for the short tube, grip higher than 3 cm below its rim.

Even so, gripping is not fully reliable yet: in a run of 48 picks (every tube, both arms,
4 seeds) 2 still blew up. Those happen as the tube wakes from sleep under the fingers (see
**Speed tricks** below). With sleeping turned off the same grips were clean. Making the
grasp reliable is part of the pick-and-place step.

## Containers: one shape to look at, another to touch

MuJoCo fills in any mesh when it checks what touches what (**collision**). A tube mesh would
act like a solid stick, and the balls would sit on top of it instead of going in.

So every container has two shapes:

- **To look at:** the downloaded mesh, stretched to the real size.
- **To touch:** a ring of thin flat boxes standing around the edge, plus a disc at the bottom.
  Together they make a hollow cup. AutoBio builds its tubes the same way.

The walls are 2 mm thick (real glass is about 1 mm), because balls can slip through thinner walls.
Each container weighs what the real one does (the 150 mm test tube weighs 28 g).

## The liquid

The liquid is made of small balls.

| Setting | Value | Why |
|---|---|---|
| Ball size | 7 mm across | With 5 mm balls we needed about 2,000, and the simulation ran 25 times slower than real life. |
| Friction (grip between surfaces) | none | Liquid does not grip, so the balls slide freely. |
| Contact stiffness (how hard a ball is when pressed) | made 10 times harder than MuJoCo's default | With the default, balls sank into each other under their own weight, and the level dropped by half. |
| Weight | 1 ml of liquid weighs 1 g, like water | So the arm would feel the real weight when it lifts a source. |

**How a source gets filled:**

1. Balls pack less tightly in a narrow tube than in a wider container. So, once per container,
   we fill it, let the balls settle, and measure how many fit. The result is saved and reused.
2. Each source is filled and settled on its own, standing straight, **before** the scene is
   built. When we dropped balls into the full scene instead, tubes leaning slightly in the
   rack caught them badly, and some balls fell out.

## Speed tricks

| Setting | Why |
|---|---|
| **Time step** (how far the simulation moves the clock each step): 1 ms | A falling ball moves less than half its size per step, so it cannot jump through a wall. |
| **Solver** (the part that works out the contact forces): CG with 100 tries per step | MuJoCo's default solver managed only 2–3 steps a second. With fewer than 100 tries, a tube full of balls burst apart. |
| **Sleeping** (MuJoCo stops working on things that are not moving) | Everything except the arm starts asleep. Something wakes up only when it is touched. |

Sleeping matters most:

| Situation | Speed |
|---|---|
| Nothing touched (only the arms moving) | about 10 times faster than real life |
| Gripper holding a full source tube (the tube, its rack, the tubes in it and their balls awake) | slower; how much depends on how many balls wake |

Things can only be woken by touching them. To move an object, the arm has to grip it.

We also kept two of the arms' own settings, because changing them made things unstable
(shake and fly apart): the **integrator** (the method that turns forces into movement) and the
**friction cone** (how sideways grip is worked out, which stops held objects from slipping).

## Limits

- **The balls are not water.** They pour more like small marbles: in lumps, not a smooth stream.
- **Gripping can still fail now and then** (see **Gripping a tube**).
- **Only the liquid levels and the tubes' places in the racks change between scenes.** The number
  of objects and where the racks stand stay the same. Step 0 (finding objects with the camera) uses
  its own, more varied scenes.
