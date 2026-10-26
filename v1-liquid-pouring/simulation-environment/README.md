# Simulation Environment

The MuJoCo lab scene for the [liquid pouring task](../docs/problem-statement/problem-statement.md):
two UR5e arms with Robotiq 2F-85 grippers, three tube racks, a tray, a pouring stand,
four liquid sources and three receiving tubes. The liquid is small balls.

![The scene from the front camera](images/front.png)

## Run it

Everything installs into this folder (`.pixi/` and `assets/`), nothing outside it.

```bash
make setup    # install the environment, download the models (once)
make view     # open the scene in MuJoCo's window
make render   # save a picture from each camera into figures/
make info     # print sizes, weights, ball counts, levels and speed
make test     # run the tests (about 25 s)
```

`SEED=3` picks a different scene: other liquid levels, and the receivers in other rack cells.
For example `make view SEED=3`. In the window, press Tab and turn on geom group 3 to see
the collision walls.

## What is in the scene

Two arms face each other across the table, 0.9 m apart. The racks, the tray and the stand
stand in one row halfway between them, so both arms can reach everything.

| Group | Objects |
|---|---|
| Arms | 2 x UR5e, each with a Robotiq 2F-85 gripper and a wrist camera |
| Sources | glass test tube, centrifuge tube, beaker, Erlenmeyer flask |
| Receivers | 3 glass test tubes: 100, 150 and 200 mm tall |
| Racks | source rack, empty rack, filled rack (AutoBio, 10 cells each) |
| Other | tray for the beaker and flask, pouring stand fixed to the table |
| Cameras | `overhead`, `front`, `arm1/wrist`, `arm2/wrist` |

Each source starts filled to a random level between 20 % and 90 %, chosen by the seed.
Sizes, weights, pictures and where each model comes from: [`docs/objects.md`](docs/objects.md).

## How it is built

| File | What it does |
|---|---|
| `lab_scene/settings.py` | every number: sizes, weights, friction, ball size, layout, cameras |
| `lab_scene/glassware.py` | a container: the mesh for looks, a ring of thin boxes for collision |
| `lab_scene/racks.py` | the AutoBio rack, the tray, the pouring stand |
| `lab_scene/liquid.py` | the balls: filling a container, and counting balls for grading |
| `lab_scene/scene.py` | puts it all together for one seed; finds each arm's joints, actuators and gripper |
| `lab_scene/kinematics.py` | moves one arm's gripper to a point (for tests and demos) |
| `lab_scene/view.py`, `render.py`, `info.py` | the `make` commands |

Why each part is built the way it is: [`docs/simulation-environment.md`](docs/simulation-environment.md).

## The rule that matters

The arms must not read the simulator's answers. `scene.Truth`, body positions and
`liquid.balls_in` / `liquid.level` / `liquid.spilled` are for tests and grading only.
The arms get camera images and their own joint readings.
