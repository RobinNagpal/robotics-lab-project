"""A learned policy (ACT, learn/model.py) unscrews the cap in the full uncapping run.

Everything else stays programmed (uncap/episode.py): Arm A picks and holds the tube, Arm B finds the cap,
comes down round it and grips it. From there the policy drives Arm B, RATE times a second, until it says
"done" (it has lifted the cap away); then the program carries the cap to the holder, and Arm A puts the
tube back.

The policy reads what the arm senses from the same recorder the demonstrations came from (uncap/record.py),
with the same sensor noise added (learn/data.py), so it sees what it was trained on.

How its predictions become motor commands (ACT's "temporal ensembling"): at every sample it predicts the
next CHUNK (0.5 s); the command for a sample is the weighted mean of every prediction made for it, the
older ones weighted a little more (exp(-0.01 x age)). Between samples the joint targets slide from this
sample's command to the next's, as the teacher's did.

    python -m learn.rollout --run runs/learn/act_A --seed 200
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

from uncap import episode, record  # the simulation before PyTorch (evaluate.py: "OMP: Error #15")
from uncap.episode import CAP_ARM, Uncapper, _yaw

from .data import NOISE_VEC, drift_vec, history_index, observations

# s: the teacher takes 26 to 31 s from gripping to lifted; give up after this (env POLICY_TIME to change)
POLICY_TIME = float(os.environ.get("POLICY_TIME", 60.0))
DONE_FOR = 3  # samples in a row "done" before handing back
ENSEMBLE_M = 0.01
TARE_TIME = record.TARE_TIME
_ROW_KEYS = ("q", "dq", "grip_opening", "wrist_force", "wrist_torque", "joint_torque")


class Policy:
    """A trained run (best.pt and stats.json): inputs in, the next CHUNK commands and "done" out."""

    def __init__(self, run: Path, which: str = "best.pt"):
        import torch

        from .data import Stats
        from .model import ACT

        self.torch = torch
        self.stats = Stats.load(run / "stats.json")
        self.net = ACT()
        self.net.load_state_dict(torch.load(run / which, map_location="cpu"))
        self.net.eval()

    def __call__(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        s = self.stats
        x = self.torch.from_numpy(((obs - s.obs_mean) / s.obs_std).astype(np.float32))[None]
        with self.torch.no_grad():
            out = self.net(x)
        actions = out["actions"][0].numpy() * s.act_std + s.act_mean
        done = self.torch.sigmoid(out["done"][0]).numpy()
        return actions, done


class Ensemble:
    """ACT's temporal ensembling: every prediction made for a sample, oldest weighted most."""

    def __init__(self, m: float = ENSEMBLE_M):
        self.m = m
        self.preds: dict[int, list[tuple[np.ndarray, float]]] = {}

    def add(self, tick: int, actions: np.ndarray, done: np.ndarray) -> None:
        for i in range(len(actions)):
            self.preds.setdefault(tick + i, []).append((actions[i], float(done[i])))

    def mean(self, tick: int) -> tuple[np.ndarray, float]:
        items = self.preds[tick]
        w = np.exp(-self.m * np.arange(len(items)))
        w /= w.sum()
        return sum(wi * a for wi, (a, _) in zip(w, items, strict=True)), float(
            sum(wi * p for wi, (_, p) in zip(w, items, strict=True))
        )

    def drop(self, tick: int) -> None:
        self.preds.pop(tick, None)


def learned_uncapper(policy: Policy, noise_seed: int, noise: bool = True) -> type:
    """An Uncapper whose ``unscrew`` is ``policy``'s (a class, as episode.run_episode takes)."""

    class LearnedUncapper(Uncapper):
        lifts_cap = True

        def unscrew(self, centre, grip_z, w0, tare, squeeze):
            d, rec = self.data, self.rec
            if rec is None:
                raise RuntimeError("the learned unscrew reads the recorder: run with recording on")
            self._phase("turn", 1)  # labels, for measuring only (learn/measure.py)
            rows = rec.rows
            times = np.asarray(rows["time"])
            first = times < times[0] + TARE_TIME
            zero = {
                "force": np.asarray(rows["wrist_force"])[first].mean(0),
                "torque": np.asarray(rows["wrist_torque"])[first].mean(0),
            }
            rng = np.random.default_rng(noise_seed)
            drift = drift_vec(rng) if noise else 0.0
            ens = Ensemble()
            start, tick, done_run = d.time, 0, 0
            log = []
            while True:
                n = len(rows["time"])
                idx = history_index(np.array([n - 1]), np.array([0]))[0]
                obs = observations({k: [rows[k][i] for i in idx] for k in _ROW_KEYS}, zero)
                if noise:
                    obs = obs + rng.normal(0, 1, obs.shape).astype(np.float32) * NOISE_VEC + drift
                actions, done = policy(obs)
                ens.add(tick, actions, done)
                now, p = ens.mean(tick)
                nxt, _ = ens.mean(tick + 1)
                ens.drop(tick)
                done_run = done_run + 1 if p > 0.5 else 0
                log.append(round(p, 3))
                if done_run >= DONE_FOR:
                    break
                if d.time - start > POLICY_TIME:
                    self.report["policy_done_prob"] = log[-50:]
                    raise episode.motion.Blocked(f"the policy did not finish in {POLICY_TIME:.0f} s")
                for i in range(1, rec.every + 1):
                    c = now + (nxt - now) * i / rec.every
                    d.ctrl[self.arm_b_act] = c[:6]
                    d.ctrl[self.grip_b] = c[6]
                    yield
                tick += 1
            self.report["policy_seconds"] = round(d.time - start, 2)
            self.report["tube_turned_deg"] = round(
                math.degrees(episode.motion._round(self.yaw_gripped, _yaw(d, self.tube))), 1
            )
            self.say(CAP_ARM, "cap off")

    return LearnedUncapper


def run(policy: Policy, seed: int, record_to: Path, vary: bool = True, noise: bool = True) -> dict:
    """One full uncapping run with the learned unscrew, on stress scene ``seed`` (as the teacher's)."""
    result = episode.run_episode(
        seed,
        [0.2],
        vary=seed if vary else None,
        record_to=record_to,
        uncapper=learned_uncapper(policy, noise_seed=seed, noise=noise),
    )
    result["learned"] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=200)
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("runs/learn/one"))
    args = parser.parse_args()
    import torch

    torch.set_num_threads(1)
    r = run(Policy(args.run), args.seed, args.out, noise=not args.no_noise)
    r.pop("log", None)
    print(json.dumps(r, indent=1, default=str))


if __name__ == "__main__":
    main()
