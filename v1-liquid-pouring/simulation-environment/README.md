# Simulation Environment

The MuJoCo lab scene for the [liquid pouring task](../docs/problem-statement/problem-statement.md):
a UR5e arm with a Robotiq 2F-85 gripper, three tube racks, a tray, a pouring stand,
four liquid sources and three receiving tubes. The liquid is small balls.

![The scene from the front camera](images/front.png)

## Run it

Everything installs into this folder (`.pixi/` and `assets/`), nothing outside it.

```bash
make setup    # install the environment, download the models (once)
make view     # open the scene in MuJoCo's window
make render   # save a picture from each camera into figures/
make info     # print sizes, weights, ball counts, levels and speed
make test     # run the tests (about 10 s)
```

`SEED=3` picks a different scene: other liquid levels, and the receivers in other rack cells.
For example `make view SEED=3`. In the window, press Tab and turn on geom group 3 to see
the collision walls.

## What is in the scene

| Object | Where it comes from | Size | Weight |
|---|---|---|---|
| UR5e arm, Robotiq 2F-85 gripper | MuJoCo Menagerie | | |
| 3 tube racks (source, empty, filled) | AutoBio, 10 cells of 30 mm | 205 x 96 x 60 mm | 150 g |
| Source: glass test tube | CC0 Lab Assets pack | 25 x 150 mm | 28 g |
| Source: centrifuge tube | AutoBio, 50 ml | 29 x 115 mm | 13 g |
| Source: beaker | CC0 Lab Assets pack, 50 ml Griffin | 42 x 60 mm | 35 g |
| Source: Erlenmeyer flask | CC0 Lab Assets pack, 50 ml | 51 x 80 mm, 22 mm neck | 45 g |
| Receivers: 3 glass test tubes | CC0 Lab Assets pack | 25 x 100, 150, 200 mm | 19, 28, 38 g |
| Tray for the beaker and flask | built here | 200 x 110 mm | |
| Pouring stand, fixed to the table | built here, one rack cell on a base | | |

Each source starts filled to a random level between 20 % and 90 %, chosen by the seed.

Cameras: `overhead` (1.2 m above the table, looking down), `front` (for people to watch)
and `wrist` (on the gripper, looking along the fingers).

## How it is built

| File | What it does |
|---|---|
| `lab_scene/settings.py` | every number: sizes, weights, friction, ball size, layout, cameras |
| `lab_scene/glassware.py` | a container: the mesh for looks, a ring of thin boxes for collision |
| `lab_scene/racks.py` | the AutoBio rack, the tray, the pouring stand |
| `lab_scene/liquid.py` | the balls: filling a container, and counting balls for grading |
| `lab_scene/scene.py` | puts it all together for one seed |
| `lab_scene/kinematics.py` | moves the gripper to a point (for tests and demos) |
| `lab_scene/view.py`, `render.py`, `info.py` | the `make` commands |

Why each part is built the way it is: [`docs/simulation-environment.md`](docs/simulation-environment.md).

## The rule that matters

The arm must not read the simulator's answers. `scene.Truth`, body positions and
`liquid.balls_in` / `liquid.level` / `liquid.spilled` are for tests and grading only.
The arm gets camera images and its own joint readings.
