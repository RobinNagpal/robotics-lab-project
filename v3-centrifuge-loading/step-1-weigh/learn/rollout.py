"""Let a trained SmolVLA checkpoint drive the arm in the simulator, and score it on Step 1's two tasks.

Each test uses one of the test scenes (seeds 0 to 19, never recorded for training) and its first sample tube:

- **A, to the holder**: the arm starts at home, as in the recordings; the model is told "put the red sample
  tube in the balance holder".
- **B, back to its cell**: the programmed teacher first stands the tube in the holder (as in the recordings,
  where episode B starts after the weighing), then the model is told "put the red sample tube back in row 1,
  column 2 of the sample rack".

So B is tested even when A fails. Every 0.1 s the model gets what it got in training (the three pictures at
256 x 192, the 6 joint angles + gripper opening, the sentence) and returns 6 joint targets + a gripper
command, which are carried out exactly as the teacher's were (joint targets slid over the 0.1 s). It plans
50 steps (5 s) at a time; only the first ``--replan`` of them are used before it looks again.

Scored, with the simulator's answers:
- **placed**: the tube stands within 4 mm of its goal and within 3 degrees of upright (the recordings' check);
- **lifted**: the tube rose more than 3 cm at some point (it was picked up);
- **final_mm**: how far the tube ended from its goal;
- **disturbed**: another tube or a rack moved more than 2 mm.

    cd step-1-weigh/learn
    PYTHONPATH=$PWD/..:$PWD/../../simulation-environment pixi run python rollout.py \
        --checkpoints outputs/train/smolvla_2pass/checkpoints/005272 --seeds 1 --video
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
from centrifuge_scene import scene
from centrifuge_scene.racks import cell_centres
from centrifuge_scene.settings import LAYOUT
from PIL import Image, ImageDraw

import teacher as T
import test_scenes
from motion import Arm, Blocked
from record import CAMERAS, COLOUR_NAMES, SETTLE, SIZE, holder_turn_near, placed, sentence_back, sentence_to_holder
from weigh import Weigher, tube_at

HERE = Path(__file__).parent
OUT = HERE / "outputs" / "rollouts"
KEYS = ("observation.images.overhead", "observation.images.racks_high", "observation.images.wrist")
MAX_FRAMES = 400  # 40 s; the teacher's episodes take 17 to 21 s
DONE_FRAMES = 15  # stop early once the tube has stood placed, fingers open, for 1.5 s
LIFTED = 0.03
# The recorded gripper command is 0.486 open (45 mm) and 0.618 closed on a cap; halfway is the line.
GRIP_OPEN_BELOW = 0.55
DISTURBED = 0.002


class Policy:
    """A checkpoint with its input and output processing, on the Mac's GPU."""

    def __init__(self, folder: Path, replan: int):
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        path = folder / "pretrained_model"
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.policy = SmolVLAPolicy.from_pretrained(path)
        self.policy.config.n_action_steps = replan
        self.policy.to(self.device).eval()
        self.pre, self.post = make_pre_post_processors(
            self.policy.config,
            pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": self.device}},
        )

    def reset(self) -> None:
        self.policy.reset()

    def act(self, pictures: list[np.ndarray], state: np.ndarray, task: str) -> np.ndarray:
        obs = {"observation.state": torch.from_numpy(state.astype(np.float32))[None], "task": [task]}
        for key, pic in zip(KEYS, pictures, strict=True):
            obs[key] = torch.from_numpy(pic).permute(2, 0, 1)[None].float() / 255
        with torch.inference_mode():
            action = self.policy.select_action(self.pre(obs))
        return self.post(action)[0].float().cpu().numpy()


class Sim:
    """One test scene, set up exactly as the recordings were."""

    def __init__(self, seed: int):
        self.model, self.data, self.truth = test_scenes.build(seed, False)
        scene.reset(self.model, self.data)
        scene.settle(self.model, self.data, SETTLE)
        self.arm = Arm(self.model, self.data)
        self.w = Weigher(self.arm)
        self.teach = T.Teacher(self.arm)
        self.arm.wait(0.3)
        self.w.balance.tare()
        self.renderer = mujoco.Renderer(self.model, height=SIZE[1], width=SIZE[0])
        self.name = next(iter(self.truth.samples))
        self.colour = COLOUR_NAMES[int(self.name.removeprefix("sample"))]
        rack, self.cell = self.truth.cells[self.name]
        self.cell_xy = np.array(cell_centres(getattr(LAYOUT, rack))[self.cell])
        self.tube = tube_at(self.arm, self.cell_xy)
        self.others = [self.model.body(n).id for n in self.truth.tubes if n != self.name]
        self.others += [self.model.body(r).id for r in scene.RACKS]

    def pictures(self) -> list[np.ndarray]:
        out = []
        for cam in CAMERAS:
            self.renderer.update_scene(self.data, camera=cam)
            out.append(self.renderer.render().copy())
        return out

    def teacher_to_holder(self) -> None:
        t_rack, _ = self.w.rack_turn(self.cell_xy, self.tube)
        t_hold = holder_turn_near(self.w, t_rack)
        ep = self.teach.run(T.to_holder(self.w, self.cell_xy, self.tube, t_rack, t_hold))
        if ep.error or not placed(self.data, self.tube, self.w.holder_xy):
            raise Blocked(f"teacher could not set up task B: {ep.error or 'tube not placed'}")
        self.w.read_steady()

    def apply(self, action: np.ndarray) -> None:
        """Carry out one action as the teacher did: slide the joint targets over 0.1 s, set the gripper."""
        a = self.arm
        q = np.clip(action[:6], a.limits[:, 0], a.limits[:, 1])
        a.data.ctrl[a.gripper] = float(np.clip(action[6], 0.0, 1.0)) * 255.0
        self.teach.drive(q)


def caption(pictures: list[np.ndarray], text: str) -> np.ndarray:
    strip = np.concatenate(pictures, axis=1)
    img = Image.new("RGB", (strip.shape[1], strip.shape[0] + 22), "white")
    img.paste(Image.fromarray(strip), (0, 22))
    ImageDraw.Draw(img).text((6, 5), text, fill="black")
    return np.asarray(img)


def save_video(frames: list[np.ndarray], path: Path) -> None:
    ffmpeg = Path(sys.executable).parent / "ffmpeg"
    h, w = frames[0].shape[:2]
    cmd = [str(ffmpeg), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", "10", "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", str(path)]
    subprocess.run(cmd, input=b"".join(f.tobytes() for f in frames), check=True)


def episode(policy: Policy, seed: int, part: str, video: Path | None) -> dict:
    sim = Sim(seed)
    row = {"seed": seed, "part": part, "tube": sim.name, "cell": sim.cell, "placed": False, "lifted": False,
           "error": ""}
    try:
        if part == "B":
            sim.teacher_to_holder()
            goal, task = sim.cell_xy, sentence_back(sim.colour, sim.cell)
        else:
            goal, task = sim.w.holder_xy, sentence_to_holder(sim.colour)
    except Blocked as e:
        row["error"] = str(e)
        return row
    row["task"] = task
    d = sim.data
    start_z = float(d.xpos[sim.tube][2])
    start_others = d.xpos[sim.others].copy()
    policy.reset()
    frames, done, t0, n = [], 0, time.time(), 0
    for n in range(1, MAX_FRAMES + 1):
        pics = sim.pictures()
        state = sim.teach.state()
        action = policy.act(pics, state, task)
        if video:
            frames.append(caption(pics, f"{task}   t={n / 10:.1f}s   grip={action[6]:.2f}"))
        sim.apply(action)
        if not np.all(np.isfinite(d.qpos)):
            row["error"] = "simulation blew up"
            break
        row["lifted"] = row["lifted"] or bool(d.xpos[sim.tube][2] - start_z > LIFTED)
        ok = placed(d, sim.tube, goal) and action[6] < GRIP_OPEN_BELOW
        done = done + 1 if ok else 0
        if done >= DONE_FRAMES:
            break
    row["placed"] = bool(placed(d, sim.tube, goal))
    row["final_mm"] = round(float(np.linalg.norm(d.xpos[sim.tube][:2] - goal[:2])) * 1000, 1)
    row["disturbed"] = bool(np.linalg.norm(d.xpos[sim.others] - start_others, axis=1).max() > DISTURBED)
    row["frames"] = n
    row["wall_s"] = round(time.time() - t0, 1)
    sim.renderer.close()
    if video and frames:
        save_video(frames, video)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoints", nargs="+", type=Path, required=True)
    parser.add_argument("--seeds", type=int, default=10, help="test seeds 0..N-1")
    parser.add_argument("--parts", nargs="+", default=["A", "B"])
    parser.add_argument("--replan", type=int, default=10, help="actions used from each 50-step plan")
    parser.add_argument("--video", action="store_true", help="save a video of every episode")
    parser.add_argument("--name", default="smolvla_2pass")
    args = parser.parse_args()
    out = OUT / args.name
    out.mkdir(parents=True, exist_ok=True)
    for ckpt in args.checkpoints:
        step = int(ckpt.resolve().name)
        policy = Policy(ckpt, args.replan)
        rows = []
        for seed in range(args.seeds):
            for part in args.parts:
                vid = out / f"step{step:06d}_seed{seed}_{part}.mp4" if args.video else None
                row = episode(policy, seed, part, vid)
                rows.append(row)
                print(step, json.dumps(row), flush=True)
        (out / f"step{step:06d}.json").write_text(json.dumps(rows, indent=1))
        for part in args.parts:
            rs = [r for r in rows if r["part"] == part]
            print(f"step {step} part {part}: placed {sum(r['placed'] for r in rs)}/{len(rs)}, "
                  f"lifted {sum(r['lifted'] for r in rs)}/{len(rs)}", flush=True)
        del policy
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
