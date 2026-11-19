# The AI Models Used, and Where

This project is for learning how the robotics models from big companies are used. Each step names the model
it tries, and most steps also have a plain programmed version, to compare against and to fall back on.

The background on every model (who makes it, what it does, access, accuracy) is in the robotics models
report: `reports/robotics-models/` (start with `00-overview.md`; Google is `01-google.md`, Physical
Intelligence is in `06-other-models.md`).

---

## The models

| Model | Company | What it does here | Helps with | Access |
|---|---|---|---|---|
| **Gemini Robotics-ER 2** (`gemini-robotics-er-2-preview`) | Google DeepMind | Finds and labels the tubes, holes and balance in a picture; plans; checks whether a step worked; runs the whole job by calling our functions | Perception, Planning | Public API (Gemini API / AI Studio). Free tier; paid $1 / $5 per million tokens in and out until 31 Dec 2026, then $2 / $10 |
| **π0.5** (openpi) | Physical Intelligence | Moves the arm: picks a tube and puts it where a sentence says ("put the sample tube in the balance holder"). Fine-tuned on our own simulated demonstrations | Movement | Open weights, Apache-2.0. Needs a GPU (below) |
| **SAM 3** (comparison only) | Meta | Outlines every tube from a word ("tube"), to compare with Gemini in Step 0 | Perception | Open weights |
| **V-JEPA 2-AC** (side experiment) | Meta | A world model: given a picture of the goal (tube in the holder), it imagines what each move would lead to and picks the best. Needs no demonstrations of the task | World model, Planning | Open weights, MIT |

Not used: NVIDIA's models for now (GR00T itself would not need Isaac Sim, but it needs a GPU just like
π0.5, and we are starting with the more widely used π0.5). SmolVLA and ACT were used in earlier projects.

## Which step uses which

| Step | Learned model | Programmed version |
|---|---|---|
| 0. Find | Gemini Robotics-ER 2 (pointing, boxes); SAM 3 to compare | none needed (the comparison is the two models) |
| 1. Weigh | π0.5: pick a sample, put it in the holder, put it back | inverse kinematics, top-down grasp (v1-style) |
| 2. Plan | Gemini Robotics-ER 2 decides and explains, **calling our balance solver** | the solver alone |
| 3. Fill a balance tube | (none: pouring to a weight is a sensor loop) | pour, watching the balance (v1's pour, a better sensor) |
| 4. Load | π0.5 for the carry; the last few centimetres guided by the wrist force sensor | inverse kinematics + force-guided insertion |
| 5. Close and report | Gemini Robotics-ER 2 checks the final picture | push the lid; the balance number from the weights |
| 6. Whole run | Gemini Robotics-ER 2 as coordinator, calling Steps 0 to 5 as functions | a fixed Python coordinator |

## Why language models do not do the maths

Gemini decides *which* plan, but the balance number comes from code (Step 2's solver), because language
models are unreliable at exact arithmetic. This is also how Google describes using Gemini Robotics-ER:
the model plans and calls tools; tools do the exact work. Step 2 measures the difference: Gemini with and
without the solver.

## Open decisions

- **GPU for π0.5.** Fine-tuning π0.5 needs a GPU with about 24 to 46 GB of memory (for example an L40S or
  an A100 on AWS); running it needs about 8 GB or more. Waiting on AWS credits. Without a GPU, Steps 1 and 4
  use their programmed versions, and everything else is unchanged.
- **Gemini credits.** Gemini Robotics-ER 2 is a normal Gemini API model, billed like Gemini 2.5 Pro or 3.
  The credits we already use for Gemini most likely cover it if they are on the same Google project; to be
  confirmed on the AI Studio billing page. Either way a whole run costs cents.
