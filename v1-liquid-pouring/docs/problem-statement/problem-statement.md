# Problem Statement

## Setup

- **Source rack:** a mix of test tubes, a beaker and a flask (different sources), each holding liquid at an unknown level between 20% and 90%.
- **Empty rack:** receiving test tubes of 2–3 different sizes (height).
- **Filled rack:** where each finished tube goes.
- The liquid is simulated as small particles/balls, so pouring, spilling and level can be measured.

## What the Arm Knows at the Start

Nothing about the scene. The arm only gets camera images (overhead, front, wrist) and its own joint readings. It does not know:

- how many sources and receiving tubes there are,
- what kind each source is (test tube, beaker or flask),
- the size (height, width) of each container,
- where each container, rack and the pouring stand is,
- how much liquid each source holds.

It has to find all of this out from what it sees before it can start pouring.

## Task Sequence

1. Pick an empty tube and place it at a fixed pouring stand.
2. Pick a source and tilt it over the tube until the tube reaches 50%.
3. If that source runs out first, put it back and pour the rest from a second source.
4. Return the source upright to its rack.
5. Move the filled tube to the filled rack.

## What Makes It Hard

- **Unknown starting level.** The angle where the liquid starts to flow is different for every pour.
- **Different shapes.** In a flask, a beaker and a tube, the same tilt gives a different flow rate.
- **Different receivers.** "50%" means a different volume for each tube size.
- **Stopping on time.** Liquid that is already in the air keeps falling after the tilt stops, so the arm has to start tilting back early.
- **Two-source pours.** The arm has to notice that a source is empty and switch to another one.
