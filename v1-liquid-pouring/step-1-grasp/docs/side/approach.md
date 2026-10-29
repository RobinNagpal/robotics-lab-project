# Step 1 Approach (Side Grasps): Taking a Tube from the Side

This page has the exact settings for the second grasp network, and why each was chosen. The overview is
in [step-1-grasp.md](../../../docs/solutions/step-1-grasp.md); how well it works is in
[results.md](results.md). The first network, for grasps from above, is in
[top-down/approach.md](../top-down/approach.md). Everything not listed here is the same as there.

> This page is version 1. Version 2 (fingers opened just wider than the tube, directions counted from
> the rack, scores mixed with their neighbours) is in [improved.md](improved.md).

---

## Why a second network

A top-down grasp is fine for carrying, but no good for pouring. The tube's rim ends up 2.5 cm under the
gripper's base, with only a 14 mm slot between the finger links above it, too narrow for the 7 mm
balls that stand in for the liquid. From the side, the gripper's body is beside the tube and the
opening is free.

```
        TOP-DOWN (first network)                     SIDE (second network)

             wrist                                 wrist   gripper body
               │                                  ════════[█████]═╗
          ┌────┴────┐                                             ║ fingers
          │ gripper │                                           ┌─╨─┐  <- opening free, can pour
          └┬───────┬┘                                           │   │
           │ ┌───┐ │  <- fingers either side,                   │   │
           │ │   │ │     opening under the gripper              │   │
           └─┤   ├─┘                                            │   │
             │   │                                              │   │
             └───┘                                              └───┘

   Way in:  over the tube, straight DOWN, close,    Way in: down BESIDE the tube (6 cm out),
            straight up                                     straight IN sideways, close, up
```

## The whole pick, step by step

```
 1. LOOK      the close-up camera 35 cm above the rim, looking straight down (128 x 128 depth picture)
 2. CHOOSE    the network gives a score for each of 24 directions at every pixel;
              best pixel near the middle + best direction  ->  where, and from which side
              (best score under 0.10  ->  "no clear direction": take another tube)
 3. MOVE      up and across with the fingers down; turn them level above the way in;
              down beside the tube, 6 cm out from it; straight in; close; lift 10 cm
```

The pictures from the real arm (full physics), one of the network's grasps:

![A side grasp in full physics](images/trial.png)

*Beside: the gripper has come down 6 cm out from the tube, fingers level. In: it has moved straight
in. Closed. Lifted: the tube's opening is free.*

---

## Settings

| | Value | Why |
|---|---|---|
| Directions | 24, 15 degrees apart, all the way round | the gripper's body is on one side of the tube, so half a turn is a different grasp (top-down: 18 turns over half a circle) |
| Grip height | the pinch point 3 cm under the rim, never under rack top + 2 cm | the finger links reach 2 cm above and below the pinch point, so the opening stays clear |
| Way in | down beside the tube 6 cm out from its axis (from 12 cm above the grip height), then straight in | |
| Fingers on the way in | **fully open** (pads 93 mm apart) | as for top-down grasps; see "Why most tubes are blocked" in [results.md](results.md): this is the main thing to change |
| Depth | the pinch point goes 11.5 mm past the tube's axis | closed on a 25 mm tube, the pads sit that far behind the pinch point (the linkage moves in an arc); without this the tube ended up a radius off-centre |
| Close-up | 35 cm above the rim, 128 x 128 (2.3 mm a pixel) | covers the gripper's whole way in (top-down: 25 cm) |
| Labels | the open gripper at every point of the way in, then closing (25 to 100%, shifted 3 mm either way along the finger line) | as for top-down grasps |
| The wrist | a rod 20 cm long, 4.5 cm thick, behind the spare gripper, on only for side checks | coming in sideways at the grip height, the arm's wrist is right behind the gripper at the same height, over whatever stands there |
| Turning and mirroring the pictures | turned by 90 degrees = 6 directions; mirrored, a direction d becomes 180 - d | more training pictures for free |
| Network | GR-ConvNet style, as for top-down, 24 outputs, 1.82 million weights | |
| Training | 16 epochs, batch 32, learning rate 0.001, about 25 minutes | |
| Cut-off | 0.10, chosen on the validation pictures (92.3% of targets right) | |
| Data | 400 / 50 / 100 crowded scenes: 4,660 / 558 / 1,151 pictures | |

### Why the picture is still taken from above

Everything on the table is an upright tube or a flat rack, so a height map from above shows where every
object stands and how tall it is. That is all that decides whether a sideways approach at a given height
hits something. A camera from the side would see only the nearest tube, with everything behind it hidden.

### The labels: try every direction

For each tube, the fast check (a spare gripper moved by hand, no physics) tries all 24 directions: the
whole way in, then closing. Green: nothing but the target touched. Red: something else touched.

![How a picture is labelled](images/labels.png)

*Each line is one direction the gripper can come from (long green = clear, short red = blocked). Left:
a tube at the edge of its rack, clear only from outside. Middle: boxed in, though it stands on the outer
row with nothing behind it (why: [results.md](results.md), "Why most tubes are blocked"). Right: a tube
with no neighbours, clear from every side.*

---

## The arm for side grasps (physics runs)

The physics runs move the nearer arm: up and across with the fingers down, turn them level above the
tube, down and in, in straight lines 2 cm at a time. Getting a side grasp to work on the real arm needed:

- **Elbow up:** with the fingers level, the arm can also reach by folding its forearm low across the
  table. The search for the joint angles starts from several poses and keeps the one whose elbow is highest.
- **No folding onto itself:** poses where a link touches the arm's own gripper, or the table, are
  refused (otherwise the wrist stalls against the forearm).
- **The level roll, nearest to now:** two rolls half a turn apart are both level; taking either one at
  random flipped the wrist at every step.
- **Joints within range, nearest to now:** a joint past its limit stops there, and the arm drifted 11 cm
  off its path.
- **No jumps:** near an awkward pose, 2 cm of the fingers can need half a turn of a wrist; such a move is
  refused.
- **Arrive before closing:** the arm is given time to catch up with its targets.

## Files

| File | What it does |
|---|---|
| `grasp/side.py` | the 24 directions, the way in, the fast check (`SideChecker`), the real arm's side grasp |
| `grasp/generate_data.py --kind side` | pictures and labels, into `data_side/` |
| `grasp/train.py --kind side` | training (same network, 24 outputs) |
| `grasp/evaluate_side.py` | the grading |
| `grasp/figures_side.py` | the pictures in these docs, into `docs/side/images/` |
