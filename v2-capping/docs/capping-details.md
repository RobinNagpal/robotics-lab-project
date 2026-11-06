# Screw caps: the details

The exact settings, code names and history behind [capping.md](capping.md). Read that page first.

## 1. The files and the code

```
v2-capping/
  capping/        the cap and the capped scene
    caps.py         the cap: shape, thread, friction, the swap at the end of the thread, step(), reset()
    lab_build.py    v1's scene builder, copied line for line, with a way in before compiling (v1 unchanged)
    scene.py        v1's scene with a cap on every receiver: build(seed), reset(model, data)
    cap_trial.py    the trial with the real arm and gripper (capping.md, section 4)
    diagrams.py     the diagrams in capping.md
    render.py, view.py, autobio_tubes.py
  uncap/          taking the cap off in the whole task, programmed (uncapping-programmed.md)
  learn/          taking it off with ACT (uncapping-learning.md)
  autobio_uncap/  AutoBio's own uncapping, rebuilt for comparison (autobio-uncapping.md)
  tests/          test_caps.py (7 tests: the thread, the swap, the scene), test_lab_build.py
```

v1 compiles its scene inside `lab_scene.scene.build`, so v2 has its own copy (`capping/lab_build.py`)
with an `edit(spec)` hook before compiling, used to add the caps, and a choice of the capped tubes' cells.
Without either it builds exactly v1's scene (`tests/test_lab_build.py` checks this).

```python
import mujoco
from capping import caps, scene

model, truth = scene.build(150)        # v1's scene for seed 150, with caps on the receivers
data = mujoco.MjData(model)
scene.reset(model, data)               # arms home, grippers open, every cap on its thread
caps.step(model, data, 1000)           # ALWAYS step a capped scene with caps.step, not mujoco.mj_step
caps.on_thread(model, data, "receiver_150")   # True until it has been unscrewed
caps.turned(model, data, "receiver_150")      # how far it has been turned off, in turns (0 to 1.5)
caps.cap_body(model, data, "receiver_150")    # the body that is the cap right now (section 3)
```

`on_thread`, `turned` and `cap_body` are **simulator knowledge**. Arm code must never use them to decide
anything (a real arm can't know them): they are for checking and grading only.

## 2. The cap's numbers in code

| | Value | Constant in `caps.py` |
|---|---|---|
| Thread | 24-410: 3.175 mm a turn, 1.5 turns | `PITCH`, `TURNS` |
| Outside, between the ribs | 28.0 mm across | `OUTER_R` = 14.0 mm |
| Grip ribs | 48, standing out 0.4 mm | `RIBS`, `RIB` |
| Inside | 26.0 mm across | `INNER_R` |
| Height | 14 mm: 11.5 mm side, 2.5 mm top | `HEIGHT`, `SKIRT`, `TOP` |
| Weight | 4 g | `MASS` |
| Friction | 0.8 (phenolic) | `PHENOLIC` |
| Seat | the inside of the top 0.5 mm above the tube's collision walls (which reach about 1 mm above the drawn rim) | `GAP` |
| Seal | 0.6 N m for the first 20° (default; `uncap` sets 0.45 to 1.25 per run) | `BREAKAWAY_TORQUE`, `SEAL_ANGLE` |
| Turning | 0.1 N m (default; `uncap` sets a sixth of the seal) | `RUNNING_TORQUE` |
| End of the thread | within 0.02 turns of 1.5 | `END_MARGIN` |

The cap's shape is made in code (`caps.visual_mesh_path`, cached in `assets/converted/cap_24_410.obj`).
AutoBio's caps are made for its own centrifuge tubes (16.8 and 29.1 mm), not a 25 mm test tube.

## 3. The two bodies per cap

| Body | What it is | When it is the cap |
|---|---|---|
| `<tube>_thread` (e.g. `receiver_150_thread`) | the cap on the thread: a child of the tube with two joints | from the start to the end of the thread |
| `<tube>_cap` (e.g. `receiver_150_cap`) | the cap off the thread: a free body | after it comes off |

- `<tube>_thread_turn` (hinge, 0 to 1.5 turns) and `<tube>_thread_rise` (slide, 0 to 4.76 mm), tied by the
  equality `<tube>_thread_screw`: rise = 3.175 mm x turns. The turn joint's `frictionloss` is set by
  `caps.step` each step (the seal, then the running twist). Being a child of the tube, it never collides
  with it (MuJoCo never collides a body with its parent): the joints carry that contact.
- `<tube>_cap` is hidden (invisible, touches nothing) and held in place by a weld (`<tube>_cap_kept`) until
  the swap. It never touches its own tube either (an exclude), so a cap held just above the tube doesn't
  catch on it.
- **The swap**, at the end of the thread: the weld is switched off, the free cap is shown and made
  touchable (geom and body collision flags), and the threaded one is hidden, made untouchable and nearly
  weightless (1 mg, `HIDDEN`). The free cap starts exactly where the threaded one was, with its speed.
- **Reset** with `capping.scene.reset` (or `caps.reset`) after `mujoco.mj_resetData`: the swap changes the
  *model*, and `mj_resetData` only resets the *data*.
- MuJoCo's per-step hook (`mjcb_control`) would make `caps.step` unnecessary, but in the Python bindings any
  model built while it is set fails to compile.
- Because friction and the swap are written into the model, step only one `MjData` per model (copies used
  only for kinematics or collision checks are fine).

## 4. Settings that had to be chosen with care

MuJoCo's constraints give way in proportion to the force over the mass they hold, and a 4 g cap squeezed
by fingers takes forces thousands of times its weight. Everything below was measured.

| Setting | Value | Why |
|---|---|---|
| Thread constraints (tie, stops, friction) | `solref` (0.002, 1), `solimp` (0.99, 0.999, 0.001) (`SOLREF`, `SOLIMP`) | the shortest time constant the 1 ms step allows, almost hard. With the defaults a 2 N pull lifted the cap 12 mm |
| Friction made stiff too | `solref_friction`, `solimp_friction` the same | with the default the friction crept, and pulling straight up turned the cap 18° |
| Armature on the turn joint | 1e-4 kg m² (`TURN_ARMATURE`) | MuJoCo sets how far the tie gives way from the inertia of the joints it ties; with only the cap's own (6e-7) it gave 4.7 mm under 2 N. Armature adds inertia, not weight |
| Armature on the rise joint | 4 g (`RISE_ARMATURE`) | the same reason |
| The cap's touch | one solid cylinder | a ring of boxes (as the tubes are built) gave the flat finger pads corners to catch on: the grip was anything from 0 to 2,300 N |
| Sleeping | `capping.scene`: capped tubes, caps and their rack never sleep | asleep, the thread froze half-way while being turned |
| | `uncap.episode.build`: they start asleep and wake when Arm A touches them | awake, a tube standing in its cell slowly leans (5.6° in 15 s), and Arm A's planned way to it stops fitting. Held in Arm A's fingers it stays awake |
| The tube while the cap is turned | must be held | turning the cap of a free tube spins the whole 28 g tube, and spun hard the simulation fails |

## 5. The trial in detail (`capping/cap_trial.py`)

- **Holding the tube:** it stands in its rack, three of its points tied to where they are (two on its
  axis, one 10 cm out), a little springy as a gripper is (1 N m turns it 0.4°). A single weld was unstable
  with the stiff thread. In the full task Arm A holds it instead (`uncap/grip.py`).
- **The arm:** in the trial Arm A (`arm2`) opens the cap; in the full task Arm B (`arm1`) does.
- **Direction:** with the fingers pointing down, the wrist's last joint (`wrist_3`) points down, so
  anticlockwise seen from above is a negative turn of `wrist_3`.
- **Why quarter turns:** the UR5e's last joint turns ±360°, the cap needs 540°; AutoBio's unscrewing does
  the same (turn, let go, turn back, grip again).
- **The grip:** the pads (37.5 mm tall, much taller than the cap) 4 mm above the middle of the cap's side,
  told to close 10 mm past it (`GRIP_ABOVE_MIDDLE`, `GRIP_SQUEEZE`): about 90 N, inside the real 2F-85's 20
  to 235 N, carrying about 1.2 N m. Told to close only 4 mm past, 20 N: too weak for the seal. (The full
  task grips 18 mm above the middle: lower, the fingertips touched the tube as the cap came off.)
- **What it measures, only what a real UR5e senses:** the wrist's motor torque, less the same move with
  empty fingers (a "tare", 4 cm above the cap); and the force between fingers and cap, which is what the
  wrist force sensor reads less the gripper's weight. Simulator values are recorded beside them, to check.
- **The lift:** 6 mm, held 0.3 s, judged while holding (`LIFT_TEST`, `LIFT_TIME`, `LIFT_HOLD`), "off"
  under 5 N (`FREE_FORCE`).

## 6. Problems found on the way, and their fixes

All of these were real failures in earlier versions. Keep them in mind when changing anything.

| What went wrong | Why | Fix |
|---|---|---|
| The thread gave 12 mm under a 2 N pull | MuJoCo's soft constraints on a 4 g body | near-hard constraints; armature on the joints |
| A joint range of 1.5 turns came out as 9° | the scene's spec takes ranges in degrees | convert with `math.degrees` when `spec.compiler.degree` |
| The cap rose 0.5 mm at rest | it sat on the drawn rim, inside the collision walls | seat it 0.5 mm above the walls |
| The scene blew up when a free tube's cap was twisted | the whole tube spun up to hundreds of turns a second | hold the tube |
| The thread froze half-way | the tube fell asleep | capped tubes never sleep (or are held, awake) |
| A free cap welded to the thread kept making the scene unstable with real fingers | a 4 g free body squeezed between a stiff weld and stiff contacts | the two-body design (section 3) |
| Grip anything from 0 to 2,300 N | box corners of the cap's collision ring | one solid cylinder |
| After the swap the free cap passed through the fingers | MuJoCo skips whole bodies by collision flags worked out at build time | change the body's flags too (`body_contype`) |
| Pulling straight up turned the cap 18° | soft friction crept | stiff friction |
| The grip drifted 3 mm sideways when turning the wrist | the point between the fingers is not exactly on the last joint's axis | set the wrist angle first, then solve the arm |
| The lift test said "off" too early | a 3 mm lift only took up the arm's 1.5 to 2.4 mm sag | lift 6 mm and hold, judge while holding |
| The tare made the scene unstable on the short tube | open fingers, turning, hit the taller tube next to it | tare 4 cm higher, fingers only 40 mm open |

## 7. Other tubes for later: AutoBio's

Kept as an option; the capped scene doesn't use them. `make assets` downloads (pinned, into
`assets/autobio/`, not committed: AutoBio has no licence file) its four screw-cap tubes with their caps;
`make autobio-tubes` unscrews each 1.5 turns on our thread (pictures in `figures/`).

| Tube | Tube size | Cap size | Cap top above the tube's base |
|---|---|---|---|
| 1.5 ml screw tube | 11.4 mm across, 42.6 mm tall | 12.4 mm across, 8.6 mm tall | 44.6 mm |
| 15 ml centrifuge tube | 16.8 mm across, 118.6 mm tall | 22.6 mm across, 10.2 mm tall | 122.6 mm |
| 50 ml centrifuge tube | 29.1 mm across, 114.7 mm tall | 35.1 mm across, 12.3 mm tall | 119.7 mm |
| 1.8 ml cryovial | 10.8 mm across, 34.9 mm tall | 13.0 mm across, 11.5 mm tall | 36.4 mm |

- The meshes are for looks only (no helix drawn); the touch would be simple shapes, and the 15 ml mesh
  isn't closed.
- AutoBio uses a 3 mm pitch for all four, a stand-in: look up each tube's real thread first.
- The cap meshes are upside down in their files; AutoBio flips the cap body (`euler="pi 0 0"`).
- They needed the same two settings: armature on the turn joint, and joint ranges in radians.
- To use one, `caps.py` would take the cap's size, mesh and pitch as inputs, and the trial would be run on
  it first.

## Sources

- [Kimble 25 x 150 mm culture tube with screw cap (Capitol Scientific)](https://www.capitolscientific.com/Kimble-45066A-25150-KIMAX-25-x-150mm-Glass-Culture-Tube-with-PTFE-Rubber-Lined-Phenolic-Screw-Cap)
- [Fisherbrand Kimax test tubes with screw caps, thread 24-410 (Fisher Scientific)](https://www.fishersci.com/shop/products/fisherbrand-kimax-reusable-test-tubes-rubber-lined-screw-caps/FB70125150)
- [What 400, 410 and 415 neck finishes mean (Regent Plast)](https://regentplast.com/?p=8189)
- [Torque guide: 24 mm application torque (The Cary Company)](https://www.thecarycompany.com/torque-guide)
- [Application and removal torque: removal at 40 to 60 % (Oberk)](https://www.oberk.com/packaging-crash-course/quick-question-monday-how-to-make-sure-your-cap-is-tight-enough)
- [Automated container cap remover: stops when the torque drops to nearly zero (US patent 5271296)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/5271296)
- [Micronic screw cap decapper with torque monitoring (SelectScience)](https://www.selectscience.net/article/micronic-launches-automatic-screw-cap-decapper-tm-system)
- [UR5e datasheet: wrist 3 turns ±360° (Crossco)](https://www.crossco.com/wp-content/uploads/2024/04/UR5e-Datasheet.pdf)
- [UR5e force/torque sensor accuracy (Elmark)](https://cdn.elmark.com.pl/34833)
- [A screw joint in MuJoCo from a hinge and a slide (RL bottle-cap report)](https://chrisliang33.github.io/files/twist-rl-report.pdf)
- [AutoBio on GitHub](https://github.com/autobio-bench/AutoBio)
- [MuJoCo's bolt add-on source](https://github.com/google-deepmind/mujoco/blob/main/plugin/sdf/bolt.cc)
