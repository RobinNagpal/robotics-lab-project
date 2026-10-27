# Problem Statement

## Setup

- **Source rack:** a mix of test tubes (different sources), each holding liquid at an unknown level between 20% and 90%.
- **Empty rack:** receiving test tubes of 2–3 different sizes (height).
- **Destination rack:** where each finished tube goes.
- **Two arms:** one handles the source tube, the other handles the receiving tube.
- The liquid is simulated as small particles/balls, so pouring, spilling and level can be measured.

## What the Arms Know at the Start

Nothing about the scene. The arms only get camera images (overhead, front, wrist) and their own joint readings. They do not know:

- how many source and receiving tubes there are,
- the size (height, width) of each tube,
- where each tube and rack is,
- how much liquid each source holds.

They have to find all of this out from what they see before they can start pouring.

## Task Sequence

1. One arm picks a filled source tube from the source rack; the other arm picks an empty tube from the empty rack.
2. Both arms bring their tubes to the center.
3. The source tube is tilted over the receiving tube to transfer its contents, so that 50% stays in the source and 50% is transferred.
4. The source tube is placed back upright in the source rack.
5. The filled tube is placed in the destination rack.

## What Makes It Hard

- **Unknown starting level.** The angle where the liquid starts to flow is different for every pour.
- **Different receivers.** "50%" means a different volume for each tube size.
- **Stopping on time.** Liquid that is already in the air keeps falling after the tilt stops, so the arm has to start tilting back early.
