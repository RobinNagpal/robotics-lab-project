# Step 0: Detect and Classify

The first step from the [list of steps](steps.md).
It follows the [problem statement](../problem-statement/problem-statement.md).

The code is in [`step-0-detect-and-classify/`](../../step-0-detect-and-classify/). Two more pages go deeper:

- [approach.md](../../step-0-detect-and-classify/docs/approach.md): the two models, their exact settings, and why.
- [results.md](../../step-0-detect-and-classify/docs/results.md): how well each works, explained in plain words.
- [improvements.md](../../step-0-detect-and-classify/docs/improvements.md): how to make the custom CNN better, and a log of every change we try.

---

## Goal

Before the arms can do anything, they have to know what is on the table and where.
From the overhead camera, Step 0 makes a list of every object with:

- **its class:** what kind of object it is, `rack` or `test_tube`,
- **its position:** where it stands, in the robot's frame, within 1 cm (close enough for the gripper),
- **its top:** how high its highest point is.

Example output:

```
rack_1  rack       at (0.45,  0.22)  top 0.06 m   -> role: source rack
rack_2  rack       at (0.45, -0.22)  top 0.06 m   -> role: empty rack
rack_3  rack       at (0.45, -0.36)  top 0.06 m   -> role: destination rack
tube_1  test_tube  at (0.52,  0.20)  top 0.156 m  in rack_1  -> role: source
tube_2  test_tube  at (0.38, -0.24)  top 0.206 m  in rack_2  -> role: receiver
```

Which of the three tube sizes (100, 150, 200 mm) a tube is, is Step 0.1 (see [steps.md](steps.md)).

## What the arms may use

- **Allowed:** camera pictures (colour and depth), where each camera is and its lens settings
  (a real lab measures these once), and the arms' own joint readings.
- **Not allowed:** the simulator's own answers (where it put each object). Those are only used
  to make training labels and to grade the result.

## Classes and roles

| Class | Looks like |
|---|---|
| `test_tube` | a narrow glass tube, 25 mm across, 100, 150 or 200 mm tall |
| `rack` | a plastic rack with 10 holes |

**A role is not a class.** All tubes are the same kind of tube, and the three racks are the
same rack. What a thing is *for* comes from what is around it, after detection:

- the rack with tubes of liquid at many levels is the **source rack**,
- the rack with only empty tubes is the **empty rack**,
- the third rack (empty at the start, later holding half-filled tubes) is the **destination rack**,
- a tube is a **source** or a **receiver** depending on its rack and whether it holds liquid.

How much liquid a source holds is not needed: every source starts more than half full, so one pour
always fills its receiver (see [steps.md](steps.md)).

## How it works

Step 0 is split into two jobs:

1. **What and which pixels: a trained model.** The model looks at the overhead colour picture and
   draws an outline (a mask) around every rack and tube, with its class.
2. **Where: plain geometry.** For each outline, the depth picture gives how far each pixel is from
   the camera. With the camera's known position and lens, every pixel becomes a 3D point. The
   object's position is the middle of its highest points (a tube's rim, a rack's top plate).

So the model never has to guess a position. It only has to get the outline right; the depth
does the rest. With perfect outlines this geometry puts 99% of tubes and all racks within 1 cm.

```
overhead colour picture --> model --> outlines + classes --+
                                                            +--> positions, tops --> roles
overhead depth picture  -----------------------------------+
```

## The training data

The pictures come from the simulation, so labels are free: the simulator knows which object
every pixel belongs to. Each picture is a whole random scene from the overhead camera: both arms,
the three racks, and several tubes in them. From scene to scene, everything the model must not
memorise changes: where the racks stand and how they are turned, how many tubes there are and in
which cells, their heights, the colours, the lights and a small shift of the camera.

| Split | Pictures | Used for |
|---|---|---|
| Train | 800 | learning |
| Validation | 150 | checking progress during training, choosing the best epoch |
| Test | 300 | the final grade; never seen in training |

**Both models train on exactly the same pictures and are graded on exactly the same test
pictures, by the same script.** That is what makes the comparison fair. The size was chosen so
that YOLO trains in about 20 minutes on the Mac.

## The two models

### YOLO11-seg (a ready-made model)

A well-known, general detector that also draws outlines. It comes already trained on a large set
of everyday photos (COCO), and we teach it our two classes. Quick to set up and quick to train.

### Custom CNN (our own model)

A network we designed for this setup and trained from scratch. It is built to fix YOLO's weak
spot, **tubes standing close together**: instead of drawing whole outlines at once, it first
finds the **centre** of each object, then gives every pixel to the centre it belongs to. Two
neighbouring tubes have two centres, so they stay apart. It also uses what we know about the
setup (fixed camera, so objects are always the same size) and looks at the picture in finer
detail.

## Result in short

| | YOLO11-seg | Custom CNN |
|---|---|---|
| Tubes found | 97.1% | **97.6%** |
| Racks found | 100% | 100% |
| Wrong detections | 48 | **31** |
| Tubes placed within 1 cm | **95.1%** | 94.7% |
| Training time | **18 min** | 58 min |

The two are about level. The custom CNN is better at keeping touching tubes apart and makes
fewer wrong detections; YOLO places tubes a little more exactly and trains three times faster.
Both get every class right. The details, with pictures, are in
[results.md](../../step-0-detect-and-classify/docs/results.md).

## Other ways we considered

- **Hand-written rules on the depth picture** (no training): remove the table, group what is
  left into objects, and classify each group by its size and shape. Fine in a clean simulation,
  but touching objects merge into one group, and real depth cameras often miss glass.
- **Other learned models:** a two-stage model such as Mask R-CNN (finer outlines, slower), or a
  model that needs no training at all such as Grounding DINO + SAM (weak at telling lab
  glassware apart). Kept in reserve.

## How it is graded

On the 300 test pictures, each model's list is compared with the simulator's answers:

- **Count:** how many real objects it found, and how many of its detections match nothing real.
- **Class:** how many found objects got the right class.
- **Position:** how far each found position is from the true one (target: under 1 cm).
- **Role:** which rack and tube plays which part (not graded yet).
