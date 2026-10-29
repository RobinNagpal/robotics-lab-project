# Step 2: Pour Until the Receiver Is 50% Full (plan, for review)

Step 2 from the [list of steps](steps.md). It starts where [Step 1b](step-1b-pick-and-carry.md) ends:
Arm A (arm 2) holds the receiver upright and still at the pouring point, and Arm B (arm 1) holds the full
source beside it, both from the side. It follows the [problem statement](../problem-statement/problem-statement.md).

**Nothing of this is built yet.** This page is the plan, to be agreed before any code is written. The
decisions still open are at the end.

---

## Goal

Tip the source over the receiver until the receiver is **50% full**, then stop the flow. Because sources and
receivers come in matching pairs and every source starts at least 60% full, one source always holds enough.

| Receiver | Inside | 50% mark |
|---|---|---|
| short, 100 mm | 33 ml | 44 balls |
| middle, 150 mm | 50 ml | 74 balls |
| tall, 200 mm | 68 ml | 103 balls |

## Where we start

| | |
|---|---|
| Receiver | rim 35 cm above the middle of the table, held by Arm A from -y |
| Source | held by Arm B from +y, its rim 2 cm higher, its axis 85 mm from the receiver's along the line between the arms (+x) |
| Why 85 mm | the two grippers' finger links reach about 7 cm along that line; closer, they touch |
| Sizes | **the medium pair first**: Step 1b does not yet bring large tubes to the pouring point, and small ones are rarely reachable |

## What makes it hard

- **The starting level is unknown** (60 to 90%), so the tilt at which the balls start to flow differs every time.
- **Liquid in the air keeps falling** after the tilt stops, so the arm has to start tipping back early:
  stop when level + flow x reaction time reaches 50%.
- **The balls pour in lumps**, not a smooth stream.
- **The lip has to reach over a 21 mm opening** from 85 mm away, without the two grippers meeting.

---

## The pour motion (the same for both methods)

Arm B holds the source from +y, so **tipping it towards the receiver is a roll of its wrist** about the line
its fingers point along. The source turns about its **lip** (the point of its rim nearest the receiver):

1. Raise the source until its fingers are above the receiver's fingers.
2. Tip it and move it together, so the lip ends just above the receiver's opening, a little to the source's
   side: the lip stays fixed while the tube turns about it.
3. From there, the tilt angle is the only thing the controller changes.

Every pose on the way is checked with the planner from Step 1b (both arms, both tubes, the closing fingers)
before the arm moves. The tube's real place in the fingers (Step 1b measured 14 to 16 mm behind the pinch
point) is used to work out where the lip is.

---

## Two ways to know how full the receiver is

As in Steps 0 and 1: **one programmed method and one learned method, graded the same way.**

### Programmed: weight

**Arm A's wrist force sensor** feels the receiver getting heavier. The real UR5e has one built in; in the
simulation it is a force sensor at the wrist, and 1 ml of balls weighs 1 g.

- Zero it before the pour (the empty receiver and the gripper).
- Fill (%) = grams in / (the receiver's volume at 50% x 2).
- Flow = how fast the grams rise, averaged over 0.2 s.
- Known problem: balls landing in the tube push on it, so while pouring the reading is a little high; it
  settles when the flow stops.

### Learned: a camera CNN

A network looks at the receiver from the side and says **how full it is (%)** and **how fast it is filling
(% per second)**, as agreed.

| | Plan |
|---|---|
| Camera | the fixed front camera (it already looks at the pouring point from the side), if a test picture shows the receiver clearly below Arm A's fingers; otherwise a camera placed for the clearest view |
| Input | the receiver cut out of the picture (known place: the pouring point); **3 frames, 0.1 s apart**, stacked, so the network can see the flow, not only the level |
| Output | level (%) and flow (% per second) |
| Network | a small CNN, trained from scratch |
| Labels | free: the simulation counts the balls in the receiver; the flow is the change in that count |
| Data | mostly still pictures of settled balls at random levels (fast: no pour needed), plus a smaller set recorded during real pours (the falling stream, splashes, the source in front) |
| Training budget | 15 to 20 minutes on the Mac, as for the other models |

The same CNN later lets the camera check the final level without touching the tube.

---

## The controller (the same for both methods)

A state machine, fed either by the weight or by the CNN:

| State | What the arm does | Moves on when |
|---|---|---|
| **Find the flow** | tip slowly (about 5 degrees a second) | the flow passes a small threshold: remember this angle |
| **Pour** | hold the flow steady: tip a little more if it slows, a little back if it speeds up | level + flow x reaction time reaches 50% |
| **Stop** | tip back quickly to upright | upright |
| **Settle** | wait about 1 s, read the level again | the reading is still |
| **Top up** | if more than a few balls short, tip again briefly | at 50% |

The reaction time (how long the balls already falling keep arriving) is measured from a few trial pours, not
guessed.

---

## How it will be graded

On the medium pair from Step 1b's scenes (10 scenes), each method pours once per scene from the same starting
state (the simulation is saved after Step 1b, so both methods start exactly alike):

| Measure | How |
|---|---|
| Final level | balls in the receiver after it settles, against the 50% mark (74 balls); target within 5 balls |
| Spill | balls that ended outside both tubes |
| Time | from the first tip to the source upright again |
| Sensing error | the method's reading against the true count, all through the pour |

---

## Later: imitation learning (ACT, Diffusion Policy)

As discussed: once the controller pours well, its runs become **demonstrations**, and a policy learns from them
to go from camera pictures straight to tilt commands.

- The demonstrations must cover **all sizes and the whole range of starting levels** (60 to 90%). Otherwise
  the policy can learn one fixed sequence, because the start level, and so the tilt where flow begins and how
  long to pour, changes every time, and only the camera tells it when to stop.
- The size is given to the policy, or seen.
- **Edge case to test:** a source that runs dry (not possible in our setup, but a learned policy has never
  seen it and would keep tipping); the controller's rule "tipped far, nothing comes out: stop" is the
  baseline for it.

---

## Decisions for the review

1. **Medium pair only for now**, and large tubes after Step 1b's routes are fixed?
2. **The camera:** the front camera if its view of the receiver is clear, otherwise a new fixed side camera?
3. **The CNN's input:** 3 stacked frames for level and flow (my pick), or a single frame for the level only and
   the flow from consecutive readings?
4. **The weight method:** the force sensor at Arm A's wrist, zeroed before the pour?
5. **Success limit:** within 5 balls of the 50% mark (about 7% of it), with no more than 5 balls spilled?
6. **After the pour** (Steps 3 and 4: the source back to its rack, the filled tube to the destination rack):
   part of Step 2, or a step of its own?
