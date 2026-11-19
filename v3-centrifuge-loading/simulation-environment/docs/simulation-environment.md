# How the Simulation Is Built, and Why

This page explains the choices behind the scene. What is in it is in [objects.md](objects.md); how to run
it is in the [README](../README.md).

---

## 1. The simulator: MuJoCo, as in v1 and v2

Same simulator, same arm, same gripper, same liquid as the earlier projects, so their code and lessons carry
over (v1's grasping and pouring, v2's wrist force sensing). MuJoCo is pinned below 4.0: version 4.0 (Nov
2026) changed parts of the API this code uses.

## 2. The arm and gripper: UR5e + Robotiq 2F-85

We considered a small hobby arm (the SO-101, which many AI model tutorials use) and kept the UR5e:

| | UR5e + 2F-85 | SO-101 |
|---|---|---|
| Reach | 850 mm: reaches into a 24 cm tall centrifuge from the front | about 300 mm: too short for the centrifuge |
| In real labs | common: UR arms are a usual choice for lab automation | a teaching arm |
| Our code | v1 and v2 already use it (grasping, pouring, wrist force) | nothing |
| AI models | π0's openpi has an official UR5 example; AutoBio's own centrifuge task uses a UR5e + 2F-85 | most tutorials (SmolVLA, GR00T) |
| Grip | 85 mm opening, 20 N or more: holds a 50 ml tube (30 mm) easily | small |

In simulation every learned model has to be fine-tuned on our own demonstrations anyway (simulated pictures
look different from the real-world pictures the models were trained on), so "the model has seen this arm
before" matters less than reach and the code we already have.

## 3. The tubes: 50 ml, not 1.5 ml

AutoBio's centrifuge comes with a rotor for 30 small 1.5 ml tubes. We use 50 ml tubes instead:

- **The liquid.** 1.5 ml tubes are 8.6 mm wide inside. v1's 7 mm balls would not fit, and balls small
  enough would need thousands per tube. A 50 ml tube is 25 mm wide inside, like v1's test tubes, so v1's
  balls and pouring work as they are.
- **Pouring.** Nobody pours into a 1.5 ml tube; people pipette. Pouring water into a 50 ml tube is normal.
- **Seeing and grasping.** A 30 mm tube with a 35 mm cap is easy to see in a 640 x 480 picture and easy
  to grip; an 11 mm one is not.
- **The balance story is clearer** with six big holes than with thirty small ones.

## 4. The centrifuge: AutoBio's housing, our rotor

AutoBio (a 2025 benchmark of biology lab tasks in MuJoCo) has a detailed model of the **Eppendorf 5430**, a
common benchtop centrifuge: housing, hinged lid with a gas spring, and its 30 x 1.5 ml rotor. The real 5430
also takes a rotor for six 15/50 ml tubes at 35° (Eppendorf's F-35-6-30), so we kept AutoBio's housing and
lid and built that rotor ourselves.

**The rotor's size** was chosen to fit AutoBio's housing:

- The shut lid's underside is 203 mm above the table; capped tubes at 35° must stay below it. That sets
  the rotor's top at 148 mm and how deep a tube goes in (75 mm along its axis). The tests check the tallest
  tube stays 18 mm below the lid.
- The chamber is 141 mm in radius; the bottoms of the holes reach 125 mm out.
- The holes' tops are 70 mm from the axis, so neighbouring caps (35 mm wide) are 13 mm apart: room for the
  fingers.

**Collision and looks are separate**, as for the tubes:

- Each hole is a ring of thin boxes shaped like the tube's outline, 0.9 mm wider all round, with a small
  lead-in at the top. A tube slid in along the hole's axis seats with its tip at the bottom (the tests put
  a capped tube in every hole: each seats within 1 mm, at the hole's angle).
- The rotor's top between the holes is a few boxes (a knob, six webs, a rim).
- What you see is a solid rotor with six holes cut out (a mesh made once with a boolean cut, cached in
  `assets/converted/`).
- AutoBio's floor slab was 125 mm up, inside our deeper rotor; it is replaced with one under it.

**The rotor does not spin.** The task ends when the lid is latched. Spinning would test nothing the arm
does, and a spinning rotor with free tubes in MuJoCo is a different problem.

**The lid** keeps AutoBio's hinge and gas spring. Two changes:

- **A latch.** AutoBio's model has a lock (an equality constraint), off by default. Here it switches on when
  the lid is pushed within 1.7° of shut, while closing, like a real latch. `centrifuge.release_latch` lets
  it go, and the spring lifts the lid a little.
- **Less damping.** AutoBio's damping made a firmly pushed lid take 3 to 5 s to shut; with ours it takes
  about 1.3 s under a 20 N push, as a real lid does.

## 5. The balance: a pan on a spring

A real balance's pan rests on a load cell, which bends very slightly under the weight. Here the pan rides on
a very stiff spring (20 kN/m: a 50 g tube moves it 25 micrometres), and a force sensor under the pan feels
everything resting on it, including the holder. The reading is zeroed with the empty holder (tared) and
rounded to 0.01 g.

**Why the pan never sleeps.** MuJoCo saves time by putting bodies that have stopped moving to sleep: their
contacts are no longer computed. A tube asleep on the pan would weigh nothing. The pan is set to never
sleep, and anything touching an awake body stays awake, so the tube and the balls in it keep pressing on the
pan. The lid never sleeps either: asleep, it would not notice its latch letting go.

**Why a holder.** A conical tube cannot stand on a flat pan. People weigh them in a beaker or a holder; ours
is a square sleeve with a seat for the tip.

The arm **may** read the balance, as a person reads its display: it is a sensor, like the cameras.

## 6. The liquid: v1's balls

Same 7 mm frictionless balls, 1 ml = 1 g. One ball is 0.33 g: that is how finely water can be poured into
a balance tube. Samples hold 30 to 100 balls; each water tube 126.

Liquid is settled in each tube on its own before the scene is built (as in v1), and everything but the arm
starts asleep, so a scene with 400 balls runs faster than real time until the arm touches something.

**Samples are capped, balance tubes are not.** Sample tubes have screw caps, so nothing falls out when they
are tilted into the 35° holes. The balance tubes are left open: they are filled by pouring, and capping
them would add a whole screwing task (v2 did that). The tests check that an open tube as full as any job
needs (106 balls) does not spill at 35° in the rotor. Real labs would cap it; this is a simplification.

## 7. The balance rule

Spinning, each tube pulls outwards with a force proportional to its mass. Drawn as arrows, one per loaded
hole, each as long as its tube's mass and pointing out through its hole, the rotor is balanced when the
arrows add up to nothing. The length of what they add up to is the **imbalance, in grams**:

- two equal tubes in opposite holes: 0 g;
- three equal tubes 120° apart: 0 g (the "triangle" job: no balance tube needed);
- one tube on its own: its whole weight.

The limit is **1.0 g** (`BALANCE_TOLERANCE`). Real limits are in each rotor's manual and depend on the rotor
and speed; this one is chosen so pouring to within one or two balls is enough.

## 8. The jobs

The seed picks one of four jobs in turn (pairs, odd one, triangle, mismatch) so every kind of decision gets
tested: see the [README](../README.md#the-jobs). Within a job, weights that should pair are within one ball,
and weights that should not are at least 9 balls (3 g) apart. A scene with an ambiguous answer would test
luck, not the planner.

No job needs more than **4 tubes** in the rotor. The gripper's fingers need an empty hole beside the hole a
tube goes into, so a full rotor of 6 could not be loaded ([grip plan](../../grasp-plan/docs/results.md)).
That is why the odd-one job has 3 samples (4 tubes with its balance tube), not 5 (6 tubes).

## 9. What the arm may and may not read

| May read (a real robot has these) | Must not read (the simulator's answers) |
|---|---|
| Camera pictures: colour, and depth as from a depth camera | `scene.Truth`: the job, the weights, the cells |
| Its own joint angles, and its gripper's | Body positions (`data.xpos`), ball positions |
| The wrist force/torque sensor (`arm/wrist_force`, `arm/wrist_torque`) | `liquid.balls_in`, `grading.*` |
| The balance's display (`scale.Balance.grams()`) | |
| The lid's state as a person sees it (from the cameras) | |
