# Step 0: How to Improve the Custom CNN

What the custom CNN is: [approach.md](approach.md). How well it does now: [results.md](results.md).
This page lists the ways to make it better, and keeps a log of every change we try.

---

## Where it stands, and what is left to win

On the 300 test pictures, the custom CNN (run `custom_rgb`):

| | Now | Best possible with this geometry | What is lost |
|---|---|---|---|
| Tubes found | 97.6% (46 missed) | 100% | 46 tubes, 27 of them short (100 mm) tubes next to taller ones |
| Wrong detections | 31 | 0 | 16 small fragments, 13 badly outlined tubes, 2 merged pairs |
| Tubes within 1 cm | 94.7% | 99.2% (with the true outlines) | 101 tubes more than 1 cm off, almost all with neighbours, outlines rough at the edges |
| Racks | all found, 97.6% within 1 cm | 100% within 1 cm | rack outlines slightly rough |

So there are three things to fix, in this order of size:

1. **Rough outlines**, which cost position accuracy (the biggest gap: 94.7% against 99.2%).
2. **Short tubes hidden by tall neighbours**, which get missed.
3. **Small fragments**, which become wrong detections.

The ideas below are of two kinds:

- **A. Outside the model:** more or better pictures. The model stays as it is.
- **B. Inside the model or its training:** the network, its settings, or what happens after it.

Each idea says which of the three problems it targets, what it costs, and whether it keeps
the comparison with YOLO fair.

---

## A. Outside the model: more or better pictures

### A1. More training pictures

**What:** render more scenes, for example 2,000 or 4,000 instead of 800 (rendering is cheap:
800 pictures take about 2 minutes).

**Why it should help:** the custom CNN starts from nothing, so it learns everything from our
pictures. It was still improving at epoch 40. Hard cases are rare: only 117 test tubes have 3 or
more neighbours. More pictures means more hard cases to learn from.

**Variant:** instead of only more pictures, render **more of the hard cases**: racks fuller
of tubes, more short tubes next to tall ones.

**Cost:** training time grows with the number of pictures (about 1.4 minutes per epoch per 800
pictures).

**Fair?** Only if YOLO gets the same bigger set. The test pictures must stay the same 300.

### A2. A second, close-up picture of each rack (wrist camera)

**What:** take the overhead picture first, as now, to find the racks. Then move an arm so its
wrist camera is straight above each rack, about 0.25 m over the tallest tube rim, and take a
second picture. Find the tubes in the close-up.

**Why it should help:**

- **More pixels per tube.** In the overhead picture, a 25 mm tube is about 21 pixels wide. From
  0.25 m above the rims, the wrist camera (75 degree lens) sees it about 31 pixels wide in a
  640 x 480 picture, and about 62 in a 1280 x 960 one.
- **Less hiding.** A tube far from the middle of the overhead picture is seen from the side, so
  a tall tube leans over its short neighbour and hides it. That is the cause of most misses.
  With the camera right above the rack, the tubes lean less, since none is far from the middle.
- **One rack at a time:** a simpler picture.

**Things to watch:**

- The custom CNN assumes **objects are always the same size in the picture**. That holds only
  if the wrist camera is always at the same height above the rack. So: fixed height, and a
  separate model (or a separate training set) for close-ups.
- It needs training pictures from the wrist camera, rendered the same way (cheap).
- It needs the arms to move before Step 0 is done: a few seconds per rack.
- The rack's position from the overhead picture is used to place the camera, so racks must
  still be found from above. That already works: every rack is found.

**Cost:** a new set of training pictures and a model trained on it, plus an arm move per rack.

**Fair?** This is a different setup, not the same test. Compare both models on close-ups to be fair.

### A3. A second camera at the same time (overhead + front)

**What:** add the front camera's picture, which sees the tubes from the side.

**Why:** from the side, tubes do not hide each other the same way, and their heights show.
It would also give each tube's size (100, 150 or 200 mm), which we need later.

**Things to watch:** each tube must be matched between the two pictures (which tube in the front
view is which in the overhead view). That needs the camera geometry, and is more work than A2.

### A4. The depth picture as an extra input

**What:** give the network the depth picture next to the colour picture (4 channels instead of 3).

**Why:** the depth picture already exists; we use it only for positions. Touching tubes of
different heights are easy to tell apart in depth: their rims are at different heights. It
should help with all three problems: hidden tubes, fragments (an empty rack hole is not as high
as a tube's rim) and rough outlines.

**Things to watch:** a real depth camera often misses clear glass. This works in the simulation,
and may need care on a real camera.

**Fair?** YOLO only takes colour pictures, so this is a step beyond the fair comparison. Report
it separately, as "custom CNN with depth".

### A5. A higher camera or a narrower lens

**What:** put the overhead camera higher, or give it a narrower lens.

**Why:** the higher the camera, the more straight-down it sees every tube, so tall tubes lean
over their neighbours less.

**Cost:** fewer pixels per tube unless the resolution goes up too. Probably weaker than A2.

---

## B. Inside the model or its training

### B1. Sharper outlines: predict at full size

**What:** make the network's output as big as the picture (1280 x 960) instead of half that.
Add one more decoder level, or a small refinement head that looks at the full-size picture near
the edges.

**Why:** the outlines are now drawn at half size and scaled up, which makes their edges rough.
That is the main reason it places tubes slightly worse than YOLO (94.7% against 95.1% within
1 cm), even though it keeps tubes apart better. **This targets the biggest gap.**

**Cost:** about twice the work in the last layer; training a little slower.

### B2. Train longer

**What:** 60 to 80 epochs instead of 40.

**Why:** the score was still rising at epoch 40, and the network starts from nothing.

**Cost:** time only (about 1.4 minutes per epoch).

### B3. Clean up after the network

**What:** simple rules on the network's output, with no retraining:

- **Merge small fragments** into the nearest larger tube, or drop them if they are far from one.
- **Tune the cut-offs** (centre peak 0.3, smallest tube 15 output pixels, how far a pixel's
  arrow may land from a centre) on the **validation pictures**, never on the test pictures.

**Why:** 16 of the 31 wrong detections are small fragments.

**Cost:** minutes. The cheapest idea on this page.

### B4. Pay more attention to the hard cases

**What:** make mistakes on short tubes and on tubes with neighbours count more during
training (a bigger weight in the loss), or sample training crops around crowded racks more often.

**Why:** the misses are concentrated there; the easy tubes are already right.

**Cost:** small code change.

### B5. Start from a pretrained encoder

**What:** replace our encoder with a small network already trained on everyday photos (for
example ResNet-18 or EfficientNet-B0), and keep our decoder and heads.

**Why:** YOLO's main advantage is that it starts already knowing edges, shapes and textures. With
the same head start, the custom CNN should learn faster and possibly better. It would also help
if we move to real cameras.

**Fair?** It makes the comparison closer to "our design against YOLO's design", both pretrained.

### B6. Look twice: test-time flips

**What:** run the network on the picture and on its mirror images, and average the results.

**Why:** smoother, more reliable masks and centres. It usually helps a little.

**Cost:** 2 to 4 times slower at test time (still under 0.1 seconds per picture).

### B7. A better position from depth (helps both models)

**What:** instead of "the middle of the highest points inside the outline", find the tube's
round rim in the depth picture near the outline, and take its centre.

**Why:** most tubes that are more than 1 cm off are found, but the outline is a little off, so
the highest points it finds are not the rim. Fitting the rim's circle would not need a perfect
outline.

**Note:** this is in the geometry after the model, so it helps YOLO as well. It does not change
the comparison between them.

---

## Suggested order

| Order | Idea | Targets | Cost | Changes the fair comparison? |
|---|---|---|---|---|
| 1 | **B3** clean up fragments, tune cut-offs on validation | wrong detections | minutes | no |
| 2 | **B1** sharper outlines | position | small code change, one training run | no |
| 3 | **B2** train longer | everything a little | one training run | no |
| 4 | **A1** more training pictures (YOLO too) | misses | rendering + retraining both | no, if YOLO gets the same |
| 5 | **A4** depth as an extra input | all three | one training run | yes: report separately |
| 6 | **A2** close-up of each rack | misses of short tubes | new pictures and model, arm moves | yes: a different setup |

B4, B5, B6 and B7 can be added at any point; B7 helps both models.

**Rules for every try**, so results stay comparable:

- The **300 test pictures never change**, and no setting is chosen by looking at them.
  Settings are chosen on the 150 validation pictures.
- One change at a time, so we know what helped.
- Every try gets its own run name, and is written in the log below.

---

## Log of changes

Every change to the model, its training or its pictures goes here, with its result on the
same 300 test pictures.

| Run | Date | What changed | Tubes found | Wrong detections | Tubes within 1 cm | Racks within 1 cm | Training time |
|---|---|---|---|---|---|---|---|
| `yolo11s_640_800` | 2026-10-27 | YOLO11s-seg baseline (for reference) | 97.1% | 48 | 95.1% | 98.4% | 18 min |
| `custom_rgb` | 2026-10-27 | Custom CNN, first version: colour only, output at half size, 40 epochs (see [approach.md](approach.md)) | 97.6% | 31 | 94.7% | 97.6% | 58 min |
