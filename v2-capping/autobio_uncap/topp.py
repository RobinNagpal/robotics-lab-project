"""Timing a path of arm poses as fast as joint speed and acceleration limits allow (AutoBio's ``topp.py``).

AutoBio solves each pose's joint angles, runs a cubic spline through them, and times it with the TOPP-RA
library (time-optimal path parameterisation: the fastest way along the given path with every joint under
3 rad/s and 2.5 rad/s^2, starting and ending at rest). TOPP-RA and SciPy are not in this environment, so
both pieces are written here in numpy: the same spline (SciPy's default "not-a-knot" end condition) and
the same kind of timing (the classic two-pass method on a fine grid: from the end backwards, the fastest
speed at each point from which the arm can still stop in time; then from the start forwards, speeding up
as hard as the limits and that ceiling allow). The result is the same fastest timing, to grid accuracy.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

GRID = 400  # points along the path for the timing


class Spline:
    """Cubic spline through (s_k, q_k), each joint separately, with SciPy's not-a-knot ends."""

    def __init__(self, s: np.ndarray, q: np.ndarray):
        self.s = np.asarray(s, float)
        q = np.asarray(q, float)
        n = len(self.s)
        h = np.diff(self.s)
        if n == 2:
            m = np.zeros_like(q)  # a straight line
        elif n == 3:
            # not-a-knot with three points is one parabola
            m = np.tile(2 * ((q[2] - q[1]) / h[1] - (q[1] - q[0]) / h[0]) / (h[0] + h[1]), (3, 1))
        else:
            # Unknown second derivatives m_k. Rows: smooth slope at interior knots; ends: third derivative
            # continuous across the second and second-to-last knots.
            A = np.zeros((n, n))
            b = np.zeros((n, q.shape[1]))
            for k in range(1, n - 1):
                A[k, k - 1], A[k, k], A[k, k + 1] = h[k - 1], 2 * (h[k - 1] + h[k]), h[k]
                b[k] = 6 * ((q[k + 1] - q[k]) / h[k] - (q[k] - q[k - 1]) / h[k - 1])
            A[0, 0], A[0, 1], A[0, 2] = h[1], -(h[0] + h[1]), h[0]
            A[-1, -3], A[-1, -2], A[-1, -1] = h[-1], -(h[-2] + h[-1]), h[-2]
            m = np.linalg.solve(A, b)
        self.q, self.m, self.h = q, m, h

    def _seg(self, s):
        k = np.clip(np.searchsorted(self.s, s, side="right") - 1, 0, len(self.h) - 1)
        return k, s - self.s[k]

    def __call__(self, s, order: int = 0):
        s = np.atleast_1d(s)
        k, x = self._seg(s)
        h, q0, q1, m0, m1 = self.h[k][:, None], self.q[k], self.q[k + 1], self.m[k], self.m[k + 1]
        x = x[:, None]
        a = (m1 - m0) / (6 * h)
        b = m0 / 2
        c = (q1 - q0) / h - h * (2 * m0 + m1) / 6
        if order == 0:
            return q0 + x * (c + x * (b + x * a))
        if order == 1:
            return c + x * (2 * b + 3 * a * x)
        return 2 * b + 6 * a * x


class Trajectory:
    def __init__(self, spline: Spline, t: np.ndarray, s: np.ndarray):
        self.spline, self.t, self.s = spline, t, s
        self.duration = float(t[-1])

    def eval(self, t: float) -> np.ndarray:
        return self.spline(np.interp(t, self.t, self.s))[0]


def _u_range(dq, ddq, x, acc):
    """Range of path acceleration u allowed at squared path speed x: |dq u + ddq x| <= acc per joint."""
    lo, hi = -np.inf, np.inf
    for d, dd in zip(dq, ddq, strict=True):
        if abs(d) < 1e-9:
            if abs(dd * x) > acc:
                return 1.0, -1.0  # infeasible
            continue
        a, b = (-acc - dd * x) / d, (acc - dd * x) / d
        lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
    return lo, hi


def time_path(spline: Spline, vel: float, acc: float, grid: int = GRID) -> Trajectory:
    s = np.linspace(0, 1, grid + 1)
    ds = s[1] - s[0]
    dq, ddq = spline(s, 1), spline(s, 2)
    # Speed ceiling from the joint speed limit: |dq| sdot <= vel.
    speed = np.max(np.abs(dq), axis=1)
    xvel = np.where(speed > 1e-12, (vel / np.maximum(speed, 1e-12)) ** 2, 1e6)
    # Backward pass: the fastest squared speed at each point from which the end can still be reached at rest.
    kmax = np.zeros(grid + 1)
    for k in range(grid - 1, -1, -1):

        def ok(x, k=k):
            lo, hi = _u_range(dq[k], ddq[k], x, acc)
            return lo <= hi and x + 2 * ds * lo <= kmax[k + 1] + 1e-12

        lo_x, hi_x = 0.0, xvel[k]
        if ok(hi_x):
            kmax[k] = hi_x
            continue
        for _ in range(40):
            mid = (lo_x + hi_x) / 2
            lo_x, hi_x = (mid, hi_x) if ok(mid) else (lo_x, mid)
        kmax[k] = lo_x
    # Forward pass: from rest, as fast as allowed.
    x = np.zeros(grid + 1)
    for k in range(grid):
        lo, hi = _u_range(dq[k], ddq[k], x[k], acc)
        u = min(hi, (kmax[k + 1] - x[k]) / (2 * ds))
        x[k + 1] = min(max(0.0, x[k] + 2 * ds * u), kmax[k + 1])
    x[-1] = 0.0
    rt = np.sqrt(x)
    dt = 2 * ds / np.maximum(rt[:-1] + rt[1:], 1e-9)
    t = np.concatenate([[0.0], np.cumsum(dt)])
    return Trajectory(spline, t, s)


class Topp:
    """AutoBio's Topp: solve each pose with ``ik``, spline the angles, time them under the limits."""

    def __init__(self, dof: int, qc_vel: float, qc_acc: float, ik: Callable):
        self.dof, self.vel, self.acc, self.ik = dof, qc_vel, qc_acc, ik

    def jnt_traj(self, pose_path) -> Trajectory:
        ss = np.linspace(0, 1, len(pose_path))
        jnts = np.array([self.ik(p.pos, p.quat) for p in pose_path])
        return time_path(Spline(ss, jnts), self.vel, self.acc)

    @staticmethod
    def query(traj: Trajectory, t: float) -> np.ndarray:
        return traj.eval(float(np.clip(t, 0, traj.duration)))
