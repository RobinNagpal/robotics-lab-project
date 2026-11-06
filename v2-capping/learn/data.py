"""The recorded unscrewing runs (uncap/record.py), as examples for a learned policy.

Each run is cut to the part the policy learns: from the first quarter turn (the cap already gripped by
the programmed approach) to the end of the recording, the cap lifted 6 cm off the tube.

One example, at one sample t of a run:

- **inputs**: what Arm B senses, at HISTORY moments over the last second (every STRIDE samples, the
  newest at t): its 6 joint angles, 6 joint speeds, the gripper's opening, the wrist sensor's force (3)
  and torque (3) with the free-air reading taken off ("tare"), and its 6 joint motors' torques. 25 values
  per moment. Moments before the run started repeat its first sample.
- **targets**: what the teacher commanded for the next CHUNK samples (0.5 s): 6 joint targets and the
  gripper command; and "done" for each of them (the run has ended: hand back to the program).

Realistic sensor noise is added to the inputs when an example is drawn (``noisy``), never stored: per
sample (``NOISE``), plus a small offset per run on the wrist sensor (a real sensor drifts after taring).

Split by scene: training 300-469, validation 470-499. The stress scenes 200-229 are the test, run in
closed loop only (learn/evaluate.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

RATE = 50  # samples per second (record.RATE)
HISTORY = 10  # moments of input
STRIDE = 5  # samples between them: 0.1 s, so the inputs reach 0.9 s back
CHUNK = 25  # future samples predicted: 0.5 s
DONE_LAST = 5  # the last samples of a run (0.1 s) count as "done", as does everything after
TURN = 2  # record.PHASES.index("turn")
VAL_FROM = 470  # scenes from here on are validation

# Inputs: name, size. Noise per sample (1 sigma) and per run (the wrist sensor's drift after taring).
# A UR5e's wrist sensor is good to about 4 N and 0.3 N m overall (its datasheet's accuracy), most of it a
# slow drift that taring removes; the noise per sample is set well below that, the drift per run below too.
INPUTS = (("q", 6), ("dq", 6), ("grip_opening", 1), ("force", 3), ("torque", 3), ("joint_torque", 6))
NOISE = {"q": 1e-4, "dq": 5e-3, "grip_opening": 2e-3, "force": 1.5, "torque": 0.1, "joint_torque": 0.2}
DRIFT = {"force": 1.0, "torque": 0.05}
OBS = sum(n for _, n in INPUTS)
ACTIONS = 7  # 6 joint targets and the gripper command


def _slices() -> dict[str, slice]:
    out, i = {}, 0
    for name, n in INPUTS:
        out[name] = slice(i, i + n)
        i += n
    return out


SLICES = _slices()
NOISE_VEC = np.concatenate([np.full(n, NOISE[name]) for name, n in INPUTS]).astype(np.float32)


def drift_vec(rng: np.random.Generator, size: int | tuple = ()) -> np.ndarray:
    """A random offset per run on the wrist force and torque (zeros elsewhere)."""
    shape = (size,) if isinstance(size, int) else size
    out = np.zeros((*shape, OBS), np.float32)
    for name, sigma in DRIFT.items():
        s = SLICES[name]
        out[..., s] = rng.normal(0, sigma, (*shape, s.stop - s.start))
    return out


def observations(rows: dict, tare: dict) -> np.ndarray:
    """The 25 input values per sample, from a recording's arrays (or the live recorder's rows)."""
    n = len(rows["q"])
    return np.concatenate(
        [
            np.asarray(rows["q"], np.float32).reshape(n, 6),
            np.asarray(rows["dq"], np.float32).reshape(n, 6),
            np.asarray(rows["grip_opening"], np.float32).reshape(n, 1),
            np.asarray(rows["wrist_force"], np.float32).reshape(n, 3) - np.float32(tare["force"]),
            np.asarray(rows["wrist_torque"], np.float32).reshape(n, 3) - np.float32(tare["torque"]),
            np.asarray(rows["joint_torque"], np.float32).reshape(n, 6),
        ],
        axis=1,
    )


def history_index(t: np.ndarray, start: np.ndarray) -> np.ndarray:
    """For samples ``t`` (of runs beginning at ``start``): the HISTORY input moments, oldest first."""
    back = STRIDE * np.arange(HISTORY - 1, -1, -1)
    return np.maximum(t[:, None] - back[None, :], start[:, None])


@dataclass
class Stats:
    """Means and spreads, from the training runs, to scale inputs and targets to about 1."""

    obs_mean: np.ndarray
    obs_std: np.ndarray
    act_mean: np.ndarray
    act_std: np.ndarray

    @staticmethod
    def of(d: Demos) -> Stats:
        obs, act = d.obs[d.samples], d.act[d.samples]
        # The noise sets a floor on each spread, so a quiet input is not blown up.
        return Stats(
            obs.mean(0),
            np.maximum(obs.std(0), np.maximum(NOISE_VEC, 1e-3)),
            act.mean(0),
            np.maximum(act.std(0), 1e-3),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({k: v.tolist() for k, v in self.__dict__.items()}))

    @staticmethod
    def load(path: Path) -> Stats:
        raw = json.loads(path.read_text())
        return Stats(**{k: np.asarray(v, np.float32) for k, v in raw.items()})


class Demos:
    """The runs of a folder, scenes in ``seeds``, joined end to end (only the runs that went right)."""

    def __init__(self, folder: Path, seeds: range):
        obs, act, run_start, samples, ends, self.names = [], [], [], [], [], []
        offset = 0
        for path in sorted(folder.glob("seed*_size*.npz")):
            seed = int(path.stem.split("_")[0][4:])
            if seed not in seeds:
                continue
            f = np.load(path)
            meta = json.loads(str(f["meta"]))
            if not meta.get("done"):
                continue
            first = int(np.argmax(f["phase"] == TURN))
            n = len(f["time"])
            obs.append(observations({k: f[k] for k in f.files}, meta["tare"]))
            act.append(np.concatenate([f["action_q"], f["action_grip"][:, None]], 1).astype(np.float32))
            run_start.append(np.full(n, offset))
            ends.append(np.full(n, offset + n - 1))
            samples.append(offset + np.arange(first, n))
            self.names.append(path.stem)
            offset += n
        self.obs, self.act = np.concatenate(obs), np.concatenate(act)
        self.run_start, self.run_end = np.concatenate(run_start), np.concatenate(ends)
        self.samples = np.concatenate(samples)  # the samples examples are drawn at
        # A run number per sample, for its drift.
        self.run_of = np.searchsorted(np.unique(self.run_start), self.run_start)

    def __len__(self) -> int:
        return len(self.samples)

    def batch(
        self, which: np.ndarray, stats: Stats, rng: np.random.Generator | None = None
    ) -> dict[str, np.ndarray]:
        """Examples at ``which`` (positions in ``samples``), scaled. ``rng``: add sensor noise."""
        t = self.samples[which]
        start, end = self.run_start[t], self.run_end[t]
        obs = self.obs[history_index(t, start)]  # B, HISTORY, OBS
        if rng is not None:
            obs = obs + rng.normal(0, 1, obs.shape).astype(np.float32) * NOISE_VEC
            obs = obs + drift_vec(rng, len(t))[:, None, :]
        future = t[:, None] + np.arange(CHUNK)[None, :]
        act = self.act[np.minimum(future, end[:, None])]
        done = (future >= (end[:, None] - DONE_LAST + 1)).astype(np.float32)
        return {
            "obs": ((obs - stats.obs_mean) / stats.obs_std).astype(np.float32),
            "actions": ((act - stats.act_mean) / stats.act_std).astype(np.float32),
            "done": done,
        }


def split(folder: Path) -> tuple[Demos, Demos]:
    return Demos(folder, range(0, VAL_FROM)), Demos(folder, range(VAL_FROM, 10_000))
