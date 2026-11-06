"""Pictures of one uncapping run for the docs: eight moments, an animation of the whole run, a close-up
animation of the unscrewing, and Arm A's grip before and after the fix.

    python -m uncap.figures --seed 150 --size 0.2 --out docs/images/uncapping
    python -m uncap.figures --act runs/learn/act_A --seed 200 --no-grip   the same, ACT unscrewing
        (learn/rollout.py; a stress scene, as ACT was tested on; files named learned-*)
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import mujoco
from PIL import Image, ImageDraw, ImageFont

from capping import caps

from . import episode, grip, grip_test

FONT = ImageFont.load_default(size=16)
SMALL = ImageFont.load_default(size=12)
EVERY = 0.4  # s between pictures of the whole run
CLOSE_EVERY = 0.2  # s between close-ups
# Above the front-left corner of the table, looking at the middle (as v1's Step 3 pictures).
VIEW = {"lookat": (0.42, -0.02, 0.2), "azimuth": -135.0, "elevation": -28.0, "distance": 1.3}
# Close to where Arm A holds the tube up, level with the cap.
CLOSE = {"lookat": (0.45, 0.0, 0.355), "azimuth": -120.0, "elevation": -8.0, "distance": 0.36}
# (caption, the log message the moment comes after, how far into that stretch)
MOMENTS = [
    ("1. Arm A reaches in beside a capped tube", "waiting", 0.55),
    ("2. Arm A lifts it out, holds it up in the middle", "picked", 0.85),
    ("3. Arm A grips tighter; Arm B looks, comes down", "found the cap", 0.6),
    ("4. Arm B grips the cap", "gripped the cap", 0.0),
    ("5. Quarter turns, a lift test after each", "gripped the cap", 0.55),
    ("6. Cap off: Arm B lifts it away", "cap off", 0.05),
    ("7. Cap on the holder; Arm B goes home", "cap put down", 0.4),
    ("8. Arm A puts the open tube in the rack", "put back: let go", 0.6),
]


def _camera(view: dict) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = view["lookat"]
    cam.azimuth, cam.elevation, cam.distance = view["azimuth"], view["elevation"], view["distance"]
    return cam


def _caption(img: Image.Image, text: str, font=FONT, bar: int = 26) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + bar), (20, 20, 20))
    out.paste(img, (0, bar))
    ImageDraw.Draw(out).text((8, (bar - font.size) // 2), text, fill=(255, 255, 255), font=font)
    return out


def _stage(log: list, t: float) -> str:
    """The last message any arm sent before ``t``, shortened."""
    msg = "starting"
    for when, _arm, m in log:
        if when <= t:
            msg = m
    return "put back: " + msg.split(": ", 1)[1].split(",")[0] if msg.startswith("put back:") else msg


LABELS = {  # what is happening after each message, for the animations' captions
    "starting": "Arm A goes to a capped tube",
    "waiting": "Arm A picks the capped tube (Arm B waits)",
    "picked": "Arm A carries it to the middle",
    "ready": "Arm A grips tighter and locks its fingers",
    "holding tight": "Arm B looks down at the cap",
    "found the cap": "Arm B comes down over the cap",
    "gripped the cap": "Arm B: quarter turn, lift test, grip again ...",
    "cap off": "Cap off: Arm B carries it to the holder",
    "cap put down": "Arm B goes home",
    "home": "Arm A carries the open tube to the rack",
    "put back: let go": "Arm A lets go and goes home",
    "put back: home": "Done",
}


def _label(stage: str) -> str:
    if stage.startswith("put back:") and stage not in LABELS:
        return "Arm A carries the open tube to the rack"
    return LABELS.get(stage, stage)


def record(seed: int, size: float, **run) -> tuple[dict, list, list]:
    """One run, with pictures: (result, wide frames, close-up frames), each frame (time, image).
    ``run``: more for episode.run_episode (a learned unscrew: learn/rollout.py)."""
    wide, close = [], []
    state: dict = {"last": -math.inf, "last_close": -math.inf}
    orig = episode.Watch.step

    def step(self) -> None:
        orig(self)
        d = self.data
        if "r" not in state:
            state["r"] = mujoco.Renderer(self.model, 480, 640)
            state["rc"] = mujoco.Renderer(self.model, 360, 480)
            state["cam"], state["camc"] = _camera(VIEW), _camera(CLOSE)
        if d.time - state["last"] >= EVERY - 1e-6:
            state["last"] = d.time
            r = state["r"]
            r._gl_context.make_current()  # the wrist camera's renderer is open too
            r.update_scene(d, camera=state["cam"])
            wide.append((d.time, Image.fromarray(r.render().copy())))
        if d.time - state["last_close"] >= CLOSE_EVERY - 1e-6:
            state["last_close"] = d.time
            r = state["rc"]
            r._gl_context.make_current()
            r.update_scene(d, camera=state["camc"])
            close.append((d.time, Image.fromarray(r.render().copy())))

    episode.Watch.step = step
    try:
        result = episode.run_episode(seed, [size], **run)
    finally:
        episode.Watch.step = orig
    return result, wide, close


def moments_sheet(result: dict, wide: list, out: Path, prefix: str = "programmed") -> None:
    log = result["log"]
    times = {m: t for t, _a, m in reversed(log)}  # first time each message was sent
    order = sorted(times.items(), key=lambda kv: kv[1])
    picks = []
    for caption, after, where in MOMENTS:
        start = 0.0 if after == "starting" else times.get(after)
        if start is None:
            continue
        later = [t for _m, t in order if t > start + 1e-6]
        end = later[0] if later else wide[-1][0]
        chosen = [img for t, img in wide if start <= t <= end]
        if chosen:
            picks.append(_caption(chosen[min(len(chosen) - 1, int(where * (len(chosen) - 1)))], caption))
    w, h = picks[0].size
    sheet = Image.new("RGB", (4 * w + 3 * 8, 2 * h + 8), (255, 255, 255))
    for i, img in enumerate(picks):
        sheet.paste(img, ((i % 4) * (w + 8), (i // 4) * (h + 8)))
    sheet.resize((sheet.width // 2, sheet.height // 2), Image.LANCZOS).save(out / f"{prefix}-moments.png")


def gif(frames: list, log: list, path: Path, size: tuple[int, int], every: int, ms: int) -> None:
    """An animation, captioned with the time and what is happening, in 64 colours to keep it small."""
    small = [
        _caption(img.resize(size, Image.LANCZOS), f"{t:5.1f} s  {_label(_stage(log, t))}", SMALL, 20)
        for t, img in frames[::every]
    ]
    small = [f.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE) for f in small]
    small[0].save(path, save_all=True, append_images=small[1:], duration=ms, loop=0, optimize=True)


def grip_picture(seed: int, size: float, out: Path) -> dict:
    """Arm A's fingers on the tube seen from above: Step 1b's grip, and the grip now (grip.py)."""
    shots, held = [], {}
    for depth, title in (
        (grip.STEP1B_DEPTH, "Before: Step 1b's grip, tube deep in the hand"),
        (grip.HOLD_DEPTH, "Now: tube between the pads' middles, fingers locked"),
    ):
        grip.use_hold_depth(depth)
        model, data, tube, _ = grip_test.hold(seed, size)
        m, d = copy.copy(model), copy.copy(data)
        if depth == grip.HOLD_DEPTH:
            grip.squeeze(m, d, episode.HOLD_ARM)
            caps.step(m, d, round(grip.SETTLE / m.opt.timestep))
            grip.hold_tight(m, d, episode.HOLD_ARM, True)
            caps.step(m, d, 200)
        state = grip_test.grip_state(m, d, tube)
        held[title] = grip_test.twist(m, d, tube)["holds_Nm"]
        # From above, a little to the side, at the pads' height.
        pads = [m.geom(f"{episode.HOLD_ARM}/gripper/{p}").id for p in ("right_pad1", "left_pad1")]
        mid = (d.geom_xpos[pads[0]] + d.geom_xpos[pads[1]]) / 2
        # Straight down on the fingers, with the cap hidden so the open tube shows where it sits.
        for g in range(m.ngeom):
            if m.body(m.geom_bodyid[g]).name in (caps.follower_name(tube), caps.cap_name(tube)):
                m.geom_rgba[g, 3] = 0.0
        cam = _camera({"lookat": tuple(mid), "azimuth": 180.0, "elevation": -89.0, "distance": 0.17})
        r = mujoco.Renderer(m, 360, 360)
        r.update_scene(d, camera=cam)
        img = Image.fromarray(r.render().copy())
        r.close()
        text = f"{title}\nsqueeze {state['squeeze_N']:.0f} N, holds {held[title]} N m of twist"
        bar = Image.new("RGB", (img.width, 44), (20, 20, 20))
        ImageDraw.Draw(bar).multiline_text((8, 4), text, fill=(255, 255, 255), font=SMALL, spacing=4)
        both = Image.new("RGB", (img.width, img.height + 44))
        both.paste(bar, (0, 0))
        both.paste(img, (0, 44))
        shots.append(both)
    sheet = Image.new("RGB", (2 * shots[0].width + 8, shots[0].height), (255, 255, 255))
    for i, s in enumerate(shots):
        sheet.paste(s, (i * (s.width + 8), 0))
    sheet.save(out / "arm-a-grip.png")
    grip.use_hold_depth(grip.HOLD_DEPTH)
    return held


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--seed", type=int, default=150)
    parser.add_argument("--size", type=float, default=0.2)
    parser.add_argument("--out", type=Path, default=Path("docs/images/uncapping"))
    parser.add_argument("--no-grip", action="store_true", help="skip the grip picture")
    parser.add_argument("--act", type=Path, default=None, help="a trained ACT run: it unscrews the cap")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    run, prefix = {}, "programmed"
    if args.act is not None:
        from learn import rollout  # after the simulation (PyTorch second: evaluate.py)

        prefix = "learned"
        policy = rollout.Policy(args.act)
        run = {
            "vary": args.seed,
            "record_to": args.out / "learned-recording",
            "uncapper": rollout.learned_uncapper(policy, noise_seed=args.seed),
        }
        LABELS["gripped the cap"] = "ACT unscrews: quarter turn, lift test, grip again ..."
        LABELS["cap off"] = "ACT: done, cap lifted. Program carries it to the holder"
        MOMENTS[4] = ("5. ACT turns the cap, a lift test after each", "gripped the cap", 0.55)
        MOMENTS[5] = ("6. ACT lifts the cap away; the program takes over", "cap off", 0.05)
    result, wide, close = record(args.seed, args.size, **run)
    log = result["log"]
    moments_sheet(result, wide, args.out, prefix)
    gif(wide, log, args.out / f"{prefix}-run.gif", (320, 240), 2, 120)
    # The close-up: from Arm A holding tight until the cap is on its way to the holder.
    t0 = next(t for t, _a, m in log if m == "holding tight")
    t1 = next(t for t, _a, m in log if m == "cap off") + 3.0
    gif(
        [f for f in close if t0 - 1 <= f[0] <= t1],
        log,
        args.out / f"{prefix}-unscrewing-closeup.gif" if args.act else args.out / "unscrewing-closeup.gif",
        (320, 240),
        2,
        110,
    )
    held = {} if args.no_grip else grip_picture(args.seed, args.size, args.out)
    keys = ("done", "failure", "cap_off", "cap_on_holder", "tube_in_rack", "touched", "tube_held")
    summary = {
        k: result.get(k)
        for k in (*keys, "quarters", "seal_torque_Nm", "tube_turned_deg", "unscrew_seconds", "sim_seconds")
    }
    summary["grip_holds_Nm"] = held
    summary["messages"] = [(round(t, 1), a, m) for t, a, m in log]
    (args.out / f"{prefix}-run.json").write_text(json.dumps(summary, indent=1, default=str))
    print(json.dumps(summary, indent=1, default=str), f"\n{len(wide)} wide, {len(close)} close-up frames")


if __name__ == "__main__":
    main()
