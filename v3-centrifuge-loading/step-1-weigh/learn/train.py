"""Train SmolVLA on Step 1's demonstrations (``outputs/datasets/step1_train``), starting from ``lerobot/smolvla_base``.

A thin wrapper around LeRobot's ``lerobot-train`` that fixes this project's settings:

- Our three cameras are given the base model's camera names (overhead -> camera1, racks_high -> camera2,
  wrist -> camera3).
- The base model was pretrained on a 6-number arm; ours has 7 numbers (6 joints + gripper) in both the state
  and the action. A local copy of the base model's folder is made with ``config.json`` saying 7 (the weights are
  linked, not copied). SmolVLA pads every state and action to 32 numbers inside, so the weights fit either way;
  this only keeps the saved settings truthful.
- Training length is given in passes over the data ("epochs"). The learning rate warms up for the first 5% and
  then decays to its floor by the last step (the base settings assume a 30,000-step run).
- A copy of the model (a "checkpoint") is saved every ``--save-minutes`` minutes, worked out from the measured
  speed on this Mac (1.5 s per step at batch 8, speed trial 2026-11-20).

    cd step-1-weigh/learn
    pixi run python train.py --epochs 2 --save-minutes 20      # ~2 h 20 min on the Mac's GPU
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
DATASET = HERE / "outputs" / "datasets" / "step1_train"
BASE_REPO = "lerobot/smolvla_base"
RENAME = {
    "observation.images.overhead": "observation.images.camera1",
    "observation.images.racks_high": "observation.images.camera2",
    "observation.images.wrist": "observation.images.camera3",
}
SECONDS_PER_STEP_B8 = 1.5  # measured on this Mac (MPS), batch 8


def local_base(state_dim: int, action_dim: int) -> Path:
    """The base model's folder with config.json saying our state/action sizes; other files are links."""
    from huggingface_hub import snapshot_download

    src = Path(snapshot_download(BASE_REPO))
    dst = HERE / "outputs" / "smolvla_base_ur5e"
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.name == "config.json" or f.suffix in (".gif", ".ipynb"):
            continue
        link = dst / f.name
        if not link.exists():
            link.symlink_to(f.resolve())
    cfg = json.loads((src / "config.json").read_text())
    cfg["input_features"]["observation.state"]["shape"] = [state_dim]
    cfg["output_features"]["action"]["shape"] = [action_dim]
    (dst / "config.json").write_text(json.dumps(cfg, indent=4))
    return dst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--epochs", type=float, default=2.0, help="passes over the training data")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--save-minutes", type=float, default=20.0, help="save a checkpoint about this often")
    parser.add_argument("--name", default="smolvla_2pass")
    parser.add_argument("--workers", type=int, default=4, help="data-loading processes (at most 4 on this Mac)")
    args = parser.parse_args()

    info = json.loads((DATASET / "meta" / "info.json").read_text())
    frames = info["total_frames"]
    state_dim = info["features"]["observation.state"]["shape"][0]
    action_dim = info["features"]["action"]["shape"][0]
    steps = math.ceil(args.epochs * frames / args.batch_size)
    seconds_per_step = SECONDS_PER_STEP_B8 * args.batch_size / 8
    save_freq = max(1, round(args.save_minutes * 60 / seconds_per_step))
    out = HERE / "outputs" / "train" / args.name
    if out.exists():
        sys.exit(f"{out} already exists; pick another --name")

    print(
        f"{info['total_episodes']} episodes, {frames} frames, {args.epochs} passes, batch {args.batch_size} "
        f"-> {steps} steps, checkpoint every {save_freq} steps, about "
        f"{steps * seconds_per_step / 3600:.1f} h",
        flush=True,
    )
    cmd = [
        str(Path(sys.executable).parent / "lerobot-train"),
        f"--policy.path={local_base(state_dim, action_dim)}",
        "--policy.device=mps",
        "--policy.push_to_hub=false",
        f"--policy.scheduler_warmup_steps={max(1, steps // 20)}",
        f"--policy.scheduler_decay_steps={steps}",
        "--dataset.repo_id=local/step1_train",
        f"--dataset.root={DATASET}",
        f"--rename_map={json.dumps(RENAME)}",
        f"--batch_size={args.batch_size}",
        f"--steps={steps}",
        f"--num_workers={args.workers}",
        "--log_freq=50",
        "--save_checkpoint=true",
        f"--save_freq={save_freq}",
        f"--output_dir={out}",
        f"--job_name={args.name}",
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
