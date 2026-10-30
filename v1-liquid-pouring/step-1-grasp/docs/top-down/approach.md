# Step 1 Approach (Top-Down Grasps): Taking a Tube from Above

This page has the exact settings, and why each was chosen. The overview is in
[step-1-grasp.md](../../../docs/solutions/step-1-grasp.md); how well it works is in [results.md](results.md).
Side grasps, the second model, are in [side/approach.md](../side/approach.md).

---

## The whole pick, step by step

```
 Step 0 (overhead camera)               Step 1
 ─────────────────────────              ──────────────────────────────────────────────────────────
 target tube at (x, y), rim at top  ─►  1. LOOK    arm moves the camera 25 cm above the rim
 + every other tube Step 0 saw              │      close-up depth picture, 128 x 128
                                            ▼
                                        2. CHOOSE  where the fingers close, and at which turn
                                            │      (programmed rule  OR  grasp network)
                                            │      -> (x, y) on the table, turn, rim height
                                            │      -> or "no clear turn": pick another tube
                                            ▼
                                        3. WORK OUT  grip height  = 3 cm below the rim,
                                            │                      never under rack top + 3 cm
                                            │        squeeze      = close 4 mm under 25 mm
                                            ▼
                                        4. MOVE    over the tube at 35 cm, turn the wrist,
                                                   down to 5 cm above the rim, down to the
                                                   grip height, close, lift 10 cm straight up
```

Only step 2 is learned. Steps 1, 3 and 4 are the same for both methods.

---

## 1. The close-up picture

| Setting | Value | Why |
|---|---|---|
| Camera | straight down, on the gripper's axis | the grasp is straight down too, so the picture and the grasp share one view |
| Height | 25 cm above the rim Step 0 reported | the open fingers (85 mm) and every tube they could touch fit in the picture |
| Lens | 45 degrees | |
| Size | 128 x 128 pixels | about 1.6 mm a pixel at the rim; a tube is about 15 pixels across |
| What it holds | height above the table, per pixel, in metres | the camera's height is known, so depth turns into height by one subtraction |

**A simplification.** The real wrist camera is clamped to the side of the wrist and tilted
towards the fingers. Here the camera is placed on the gripper's axis instead. A tilted camera
would only need its picture turned into this view with its known position, which a real
setup does once.

Looking from this close shows things the overhead camera cannot: how tall each neighbour is,
and exactly where the target's rim is, even when Step 0 was a centimetre off.

---

## 2. The training data

### Crowded racks

The task's own scene keeps tubes three cells apart, so every turn fits and there is nothing to
learn. The training scenes are Step 0's random scenes (racks anywhere, turned any way, tubes of
random heights), with fuller racks:

| Rack | Tubes |
|---|---|
| source rack | 2 to 8 of its 10 cells |
| empty rack | 2 to 8 |
| destination rack | 0 to 4 |

| Split | Scenes | Pictures (one per tube) | Targets with a clear turn | Turns that are clear |
|---|---|---|---|---|
| Train | 400 | 4,660 | 74% | 43% |
| Validation | 50 | 558 | 75% | 45% |
| Test | 100 | 1,151 | 76% | 44% |

Making all of it takes a few minutes on 8 processor cores.

### Step 0's errors, made up

The camera aims where Step 0 says the tube is, so the pictures should be off-centre the way
Step 0 really is. Step 0's own results (median about 1 mm, 95% under about 10 mm, about 3% of
tubes missed) are copied with simple random numbers:

| What | How |
|---|---|
| Most tubes (88%) | off by a random amount, 0.7 mm spread in each direction |
| A few tubes (12%) | off by 3 to 12 mm in a random direction |
| Rim height | off by a random amount, 2 mm spread |
| Missed tubes (3%) | left out of Step 0's list; never a target, and the programmed rule does not know they are there |

### One settled scene for everything

Each seed gives one scene with both arms at home, a spare gripper on its own (put in place by
hand, for the labels) and the close-up camera. Before anything else it is **let settle for 0.3 s**:
a 25 mm tube in a 30 mm cell leans a little, and a 200 mm tube leaning one degree moves its rim by
3.5 mm. The labels, the pictures and the physics runs all start from that same settled state, and
a tube's true place is its **rim's centre after settling**.

**The tubes are empty in Step 1.** Full of the simulation's stiff balls, every tube rattles and the
racks never come to rest: tubes kept leaning up to 8 degrees for as long as we watched. Empty, they
settle within 0.2 degrees. Liquid only changes a tube's weight, and that matters only for the
squeeze, so the squeeze is tested separately with full tubes (section 6). The cost: the close-up
pictures never show liquid inside a tube.

### The labels: try every turn

For every tube in a scene, at its true place, the spare gripper is tried at each of the 18 turns,
and MuJoCo is asked whether it touches anything (**collision test**: a check of shapes only, no
movement, under a millisecond each):

1. **On the way down:** the open gripper at every point 5 mm apart, from 5 cm above the rim to the
   grip height. Any touch blocks the turn.
2. **While closing:** at the grip height, the fingers at 25%, 50%, 75% and 100% of the way to
   closed, each also shifted 3 mm either way along the finger line. Touching the target is
   allowed; anything else blocks the turn.

If nothing blocks it, the turn is **clear**. Step 2 was added after the first physics runs (see
[results.md](results.md), "The first try"): on the Robotiq gripper the finger linkage moves as the
fingers close, and the first finger to reach the tube shoves the arm a couple of millimetres
sideways (the tube has about 2.5 mm of play in its cell). That made the inner finger link graze a
neighbour's rim that the open fingers had missed.

![How a picture is labelled](images/labels.png)

*Left: the open fingers at a clear turn (green) and a blocked one (red), which would land on a
neighbour. The other three: all 18 turns of a tube. Long green lines are clear turns, short red
ones blocked. A tube can be boxed in (no clear turn) or completely open.*

### From labels to what the network should output

The network outputs 18 maps, one per turn. For each tube in the picture, each clear turn's map
gets a small bump (1.5 pixels wide) at the tube's rim centre. Blocked turns, and everywhere
away from tube centres, stay at 0. Every tube in the picture is labelled, not only the target,
so one picture teaches about several tubes.

---

## 3. The grasp network

### Which model, and why

| Model | What it does | Why not, or why |
|---|---|---|
| **GR-ConvNet** (Kumra et al., 2020) / **GG-CNN** (Morrison et al., 2018) | one picture from above in; for every pixel out: how good a grasp centred there is, its angle, its width | **Chosen.** Our grasps are from above, tubes stand upright. Small, trains from scratch in minutes |
| Dex-Net 2.0 (GQ-CNN) | suggests many grasps, then a CNN scores each one | the scoring idea is the same as ours, but it needs a separate sampler and is slower |
| Contact-GraspNet, AnyGrasp | full 3D grasps from any side, from a 3D point cloud | built for messy real-world piles; heavy, usually pretrained; our tubes only need one direction |
| Reinforcement learning (QT-Opt) | learns by trial and error over many full grasps | needs very many slow physics trials; little to gain for a known shape |

### Our change: a score for every turn

GR-ConvNet gives **one** angle per pixel. A tube often has several clear turns (for example
0 to 40 degrees and 120 to 170 degrees). Trained with one angle, the network would average the
right answers into a wrong one in between. So ours outputs **18 scores per pixel**, one per turn,
each from 0 (blocked) to 1 (clear). The width output is dropped: every tube is 25 mm.

### The network

```
height picture        1 x 128 x 128   (heights divided by 0.25 m)
  9x9 conv              32 x 128 x 128
  4x4 conv, stride 2    64 x  64 x  64
  4x4 conv, stride 2   128 x  32 x  32
  5 residual blocks    128 x  32 x  32    (each: two 3x3 convs and a shortcut)
  4x4 up-conv, x2       64 x  64 x  64
  4x4 up-conv, x2       32 x 128 x 128
  3x3 conv              18 x 128 x 128    one score per turn, per pixel
```

Each conv is followed by batch norm and ReLU. **1.81 million weights.** A score at one pixel
"sees" a patch about 100 pixels wide around it (16 cm at the rim), wider than the open fingers
plus a neighbour on each side.

### Training settings

| Setting | Value | Why |
|---|---|---|
| Loss | binary cross-entropy on every pixel and turn | each turn is its own yes/no question |
| Loss weight | pixels within 4 of a tube's centre count 20 times more | that is where the answer is read; otherwise the empty table outweighs everything |
| Turning and mirroring | each picture turned by 0, 90, 180 or 270 degrees, and mirrored half the time; **the 18 turn labels are shifted to match** (90 degrees = 9 turns; a mirror maps turn t to 180 - t) | four times more views for free. Getting the label shift wrong would teach nonsense, so it has its own tests |
| Depth noise | 1 mm, random | real depth cameras are noisy |
| Optimiser | AdamW, learning rate 0.001 rising then falling (one-cycle), weight decay 0.0001 | standard |
| Batch | 32 pictures | |
| Epochs | 16 | about 1 minute each on the Mac (Apple M5, MPS), so about 16 minutes, inside the 15 to 20 minute budget |
| Best epoch | the one with the most validation targets answered right | it was the last one, epoch 16: still improving slowly |

### At test time

1. Score the picture.
2. Look only within 12 pixels (about 2 cm) of the middle, where the camera aimed: that is the
   target. Neighbours' centres are at least 22 pixels away.
3. Take the pixel and turn with the highest score. **If it is under 0.25, say "no clear turn".**
   The cut-off was chosen on the validation pictures. The labels are soft bumps whose peak is
   often under 1 (the tube's centre falls between pixels), so a good turn can score as little as
   0.5, while boxed-in tubes mostly stay under 0.05. Anywhere from 0.2 to 0.3 does about equally
   well on validation (96% of targets right); 0.5 refuses a few more tubes that do have a clear
   turn (94.8%).
4. The rim's height is the highest point within 10 pixels of that pixel. (At the very centre the
   camera looks into the tube and sees the liquid or the bottom.)
5. Turn the pixel and that height into a place on the table, with the camera's known position.

---

## 4. The programmed rule (the baseline)

The rule uses only Step 0's list: where each tube stands and how high its rim is. It never sees
the close-up picture.

1. Seen from above, the open fingers are a **rectangle** around the pinch point: `reach` along
   the opening direction, `half_width` across it.
2. Neighbours shorter than 2 cm under the grip height are ignored: they cannot reach the fingers.
3. For each turn, find the neighbour whose wall comes closest to the rectangle.
4. Take the turn where that closest neighbour is furthest away. If at every turn some neighbour
   is inside the rectangle, say "no clear turn".
5. The pinch point goes where Step 0 said the tube is.

The two rectangle sizes were **fitted on the training pictures** (a grid search for the sizes
whose answers agree most with the labels), the same data the network learns from. Fitting on the
test pictures would be unfair. The best: `reach` 72 mm, `half_width` 16 mm, which agree with the
labels on 83% of training targets.

---

## 5. Grip height and squeeze (worked out, not learned)

| What | Rule | Why |
|---|---|---|
| Grip height | the pinch point 3 cm below the rim, but never below rack top + 3 cm (9 cm) | keeps the opening clear and the pads on the glass; lower, the pads catch on the rack's top plate. For the short tube this means only 1.6 cm under the rim |
| Squeeze | the fingers are told to close to 4 mm under the tube's 25 mm | the simulation's tested setting: fully closed (255) makes the finger links wrap round the tube and fling it |
| Lift | 10 cm straight up, over 1 second | slow enough that the tube's own swing adds little force |

### How much squeeze is enough

Two fingers hold a tube up by friction:

```
2 x friction x squeeze force  >=  weight x safety factor
```

| | |
|---|---|
| Heaviest tube | a full 200 mm source: 38 g of glass and up to about 60 g of balls, about 1 N |
| Friction, pad on glass | 0.7 in the simulation (the pad's value; MuJoCo takes the larger of the two surfaces) |
| Safety factor | 2 to 3, for the arm speeding up and slowing down |
| Needed | about 1.5 to 2 N |

The gripper is not told a force; it is told a finger position, and the squeeze is whatever the
fingers press with when the tube stops them. [results.md](results.md) measures it.

---

## 6. How it is graded

| Part | On what | How |
|---|---|---|
| Fast | every test target (1,151) | each method's grasp is fast-checked **where it chose to grasp** (not the true place). Good = clear all the way down **and** the pinch point within 5 mm of the tube's axis. Targets with no clear turn should be refused |
| Physics | 60 random test targets | each method's grasp is run with the real arm and full physics. Success = the tube came up at least 8 cm, within about 18 degrees of upright, and **the arm touched nothing else** |
| Fast check vs physics | the same 60 targets | one random turn each, at the true place: does the fast check's label match what physics shows? |
| Squeeze | 3 of the heaviest tubes, **full of liquid** | lifted with the fingers told to close 0.5, 1, 2 and 4 mm under the tube's width: force, slip, still lifted? |

A physics grasp takes about 5 to 10 seconds; on 8 processor cores the whole grading takes about
5 minutes.

---

## Files

| File | What it does |
|---|---|
| `grasp/rules.py` | the 18 turns, grip height, label shifts for turned pictures, Step 0's made-up errors |
| `grasp/scenes.py` | the settled crowded scene, the close-up camera, the fast check |
| `grasp/generate_data.py` | pictures and labels |
| `grasp/net.py` | the network, its labels and loss |
| `grasp/train.py` | training |
| `grasp/pick.py` | the programmed rule and reading the network's answer |
| `grasp/trial.py` | a grasp with the real arm and full physics |
| `grasp/evaluate.py` | the grading |
| `grasp/figures.py` | the pictures in these docs, into `docs/top-down/images/` |
| `tests/test_grasp.py` | label shifts, the fast check (way down and closing), the picture has no stray gripper, a physics grasp |
