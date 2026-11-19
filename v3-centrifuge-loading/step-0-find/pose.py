"""From an outline to a 3D pose: where a tube is, which way it points, and how big it is.

Step 0 itself does not need this: the racks are known, so a tube can only stand upright in a cell, and the
cell gives its place. This is for an object that could be anywhere and lie any way (a tube knocked over on
the bench), where the arm knows nothing about it in advance.

1. **Outline to point cloud.** Every pixel inside the outline, with the depth picture's distance at that
   pixel, is one 3D point on the object's surface (``cloud``). The outline's edge is shaved off first: there
   an outline often spills onto the background.
2. **First guess.** The cloud's long direction (PCA) is the axis; a circle fitted to the cloud seen along it
   gives the axis's place and the radius.
3. **Cylinder fit.** The axis is tilted and slid and the radius changed until the points sit as close to the
   cylinder's surface as they can (least squares, ``fit_cylinder``). Points far off it (the wider cap, the
   cap's top, background) are dropped and the fit repeated. A camera sees only the near half of a tube, but
   half a cylinder still fixes the cylinder.
4. **The ends** are where the points near the axis stop. Only what a camera sees counts: a rack hides the
   bottom of a tube standing in it.
5. **End-on views.** A cloud that is short or small (an upright tube from straight above: mostly its cap's
   top) cannot fix a cylinder. Its axis is the way its flattest patch faces (``flat_patch``, RANSAC).

Several cameras' fits of the same object are joined and fitted together (``merge``): each sees a different
side, so together they cover more of the cylinder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from bench import View

# Shave this many pixels off an outline's edge before using it.
ERODE = 2
# A cloud is "long" (a tube seen from the side) if its main spread is this many times its second, and it has
# at least this many points: fewer are too few to fit a cylinder to. Set on seeds 100-105 (never tested on):
# from overhead an upright tube shows at most 153 points (mostly its cap's top), a lying one at least 232.
LONG = 2.5
MIN_SIDE_POINTS = 200
# After each cylinder fit, keep points this close to its surface (m), or closer if most points are closer.
NEAR_CIRCLE = 0.003
FIT_ROUNDS = 4
# Two cameras see the same object if each one's centre is this close to the other's axis (m), the two centres
# are this close together, and (both seen from the side) their directions agree this well.
SAME_AXIS = 0.020
SAME_PLACE = 0.080
SAME_ANGLE = 20.0  # degrees


@dataclass
class Pose:
    centre: np.ndarray  # the middle of the seen length, on the axis (m, world)
    axis: np.ndarray  # unit vector; points up when the object stands
    radius: float  # m
    ends: tuple[np.ndarray, np.ndarray]  # the two ends of the seen length, on the axis
    points: np.ndarray  # the points the fit kept
    cameras: tuple[str, ...]
    long: bool  # seen from the side (True), or only end-on (False)
    whole: np.ndarray  # every point of the outline, edge included (for the ends)

    @property
    def tilt_deg(self) -> float:
        """Angle from upright: 0 standing, 90 lying flat."""
        return math.degrees(math.acos(min(1.0, abs(float(self.axis[2])))))

    @property
    def length(self) -> float:
        return float(np.linalg.norm(self.ends[1] - self.ends[0]))

    @property
    def top(self) -> np.ndarray:
        return max(self.ends, key=lambda p: p[2])


def erode(mask: np.ndarray, n: int = ERODE) -> np.ndarray:
    """``mask`` with ``n`` pixels taken off all round."""
    m = mask.copy()
    for _ in range(n):
        inner = m.copy()
        inner[1:, :] &= m[:-1, :]
        inner[:-1, :] &= m[1:, :]
        inner[:, 1:] &= m[:, :-1]
        inner[:, :-1] &= m[:, 1:]
        m = inner
    return m


def cloud(view: View, mask: np.ndarray, shave: int = ERODE) -> np.ndarray:
    """The 3D points (N x 3, world, m) of the pixels inside ``mask``, from the depth picture."""
    cam = view.camera
    v, u = np.nonzero(erode(mask, shave))
    d = view.depth[v, u]
    local = np.stack(
        [(u + 0.5 - cam.width / 2) / cam.focal * d, -(v + 0.5 - cam.height / 2) / cam.focal * d, -d], axis=1
    )
    return cam.pos + local @ cam.rot.T


def main_axis(points: np.ndarray) -> tuple[np.ndarray, bool]:
    """The cloud's long direction (or, for a flat patch, the way it faces), and whether it is long."""
    values, vectors = np.linalg.eigh(np.cov((points - points.mean(0)).T))  # smallest first
    long = values[2] >= LONG * values[1] and len(points) >= MIN_SIDE_POINTS
    axis = vectors[:, 2] if long else vectors[:, 0]
    if abs(axis[2]) > 0.5:
        axis = axis if axis[2] > 0 else -axis  # standing: point up
    elif axis[0] < 0:
        axis = -axis  # lying: either end; pick one so answers are repeatable
    return axis, bool(long)


def fit_circle(xy: np.ndarray) -> tuple[np.ndarray, float]:
    """The circle through 2D points, by least squares (Kasa's method): centre and radius."""
    a = np.column_stack([2 * xy, np.ones(len(xy))])
    b = (xy**2).sum(1)
    (cx, cy, c), *_ = np.linalg.lstsq(a, b, rcond=None)
    return np.array([cx, cy]), float(math.sqrt(max(c + cx * cx + cy * cy, 0.0)))


def basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors at right angles to ``axis`` and to each other."""
    e1 = np.cross(axis, [1.0, 0, 0] if abs(axis[0]) < 0.9 else [0, 1.0, 0])
    e1 /= np.linalg.norm(e1)
    return e1, np.cross(axis, e1)


def off_cylinder(points: np.ndarray, base: np.ndarray, axis: np.ndarray, radius: float) -> np.ndarray:
    """How far each point is from the cylinder's surface (m; + outside, - inside)."""
    rel = points - base
    return np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1) - radius


def fit_cylinder(
    points: np.ndarray, base: np.ndarray, axis: np.ndarray, radius: float, steps: int = 15
) -> tuple[np.ndarray, np.ndarray, float]:
    """Nudge a cylinder (a point on its axis, its direction, its radius) to sit best on ``points``.

    Least squares, Gauss-Newton with damping: each step tilts the axis a little two ways, slides it two ways
    and changes the radius, whichever way lowers the summed squared distances of the points from the surface.
    """
    damping = 1e-3
    for _ in range(steps):
        e1, e2 = basis(axis)

        def surface(x, base=base, axis=axis, radius=radius, e1=e1, e2=e2):
            a = axis + x[0] * e1 + x[1] * e2
            return off_cylinder(points, base + x[2] * e1 + x[3] * e2, a / np.linalg.norm(a), radius + x[4])

        r0 = surface(np.zeros(5))
        jac = np.empty((len(points), 5))
        for k in range(5):
            dx = np.zeros(5)
            dx[k] = 1e-5
            jac[:, k] = (surface(dx) - r0) / 1e-5
        jtj = jac.T @ jac
        step = -np.linalg.solve(jtj + damping * np.diag(np.diag(jtj) + 1e-12), jac.T @ r0)
        if (surface(step) ** 2).sum() < (r0**2).sum():
            a = axis + step[0] * e1 + step[1] * e2
            axis, base, radius = a / np.linalg.norm(a), base + step[2] * e1 + step[3] * e2, radius + step[4]
            damping = max(damping / 3, 1e-6)
        else:
            damping *= 10
    return base, axis, float(radius)


def flat_patch(points: np.ndarray, near: float = 0.0015, tries: int = 200) -> tuple[np.ndarray, np.ndarray]:
    """The plane most points lie on (RANSAC: try planes through 3 random points, keep the best): its points
    and the way it faces. For a tube seen end-on, that plane is its cap's top or its rim."""
    rng = np.random.default_rng(0)
    best, normal = np.zeros(len(points), bool), np.array([0, 0, 1.0])
    for _ in range(tries):
        a, b, c = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        on = np.abs((points - a) @ n) <= near
        if on.sum() > best.sum():
            best, normal = on, n
    return points[best], normal


def fit(points: np.ndarray, cameras: tuple[str, ...] = (), whole: np.ndarray | None = None) -> Pose | None:
    """A tube's pose from points on its surface, or None if there are too few.

    ``whole``: the points of the whole outline, edge included, used only to find where the tube ends (the
    shaved outline in ``points`` stops a few pixels short of each end).
    """
    if len(points) < 20:
        return None
    whole = points if whole is None else whole
    axis, long = main_axis(points)
    if not long:
        # Seen end-on: the axis is the way the cap's top (or the rim) faces, and its centre is that patch's.
        patch, axis = flat_patch(points)
        if len(patch) >= 20:
            axis = axis if axis[2] >= 0 else -axis
            centre = patch.mean(0)
            radius = float(np.percentile(np.linalg.norm(patch - centre, axis=1), 95))
            along = (whole - centre) @ axis
            near = whole[np.linalg.norm((whole - centre) - np.outer(along, axis), axis=1) <= 1.2 * radius]
            lo = float(np.percentile((near - centre) @ axis, 0.5))
            ends = (centre + lo * axis, centre)
            return Pose((ends[0] + ends[1]) / 2, axis, radius, ends, patch, cameras, False, whole)
    e1, e2 = basis(axis)
    flat = points @ np.column_stack([e1, e2])
    centre2d, radius = fit_circle(flat)
    base = centre2d[0] * e1 + centre2d[1] * e2
    keep = points
    if long:
        # The first guess (long direction, then a circle) is tipped by points that are not on the body: the
        # cap's top, the wider cap. A cylinder fit, dropping points far off its surface each round, fixes it.
        for _ in range(FIT_ROUNDS):
            base, axis, radius = fit_cylinder(keep, base, axis, radius)
            off = np.abs(off_cylinder(keep, base, axis, radius))
            inside = off <= max(NEAR_CIRCLE, 2.5 * float(np.median(off)))
            if inside.sum() < 20 or inside.all():
                break
            keep = keep[inside]
        if abs(axis[2]) > 0.5:
            axis = axis if axis[2] > 0 else -axis
        elif axis[0] < 0:
            axis = -axis
    # The ends: from every point near the axis, not only those on the body's surface, so the cap counts.
    near = whole[off_cylinder(whole, base, axis, radius) <= 0.4 * radius]
    lo, hi = np.percentile((near - base) @ axis, [0.5, 99.5])
    ends = (base + lo * axis, base + hi * axis)
    return Pose((ends[0] + ends[1]) / 2, axis, radius, ends, keep, cameras, long, whole)


def distance_to_line(p: np.ndarray, origin: np.ndarray, direction: np.ndarray) -> float:
    rel = p - origin
    return float(np.linalg.norm(rel - (rel @ direction) * direction))


def same_object(a: Pose, b: Pose) -> bool:
    """Do two cameras' poses look like the same object? Each centre near the other's axis, the centres near
    each other, and, when both saw the object from the side, the two directions agreeing."""
    if float(np.linalg.norm(a.centre - b.centre)) > SAME_PLACE:
        return False
    if a.long and b.long:
        angle = math.degrees(math.acos(min(1.0, abs(float(a.axis @ b.axis)))))
        near = max(distance_to_line(a.centre, b.centre, b.axis), distance_to_line(b.centre, a.centre, a.axis))
        return angle <= SAME_ANGLE and near <= SAME_AXIS
    # An end-on view's direction is not to be trusted: only how near its centre is to the other's axis.
    side, end = (a, b) if a.long else (b, a)
    return distance_to_line(end.centre, side.centre, side.axis) <= SAME_AXIS


def merge(per_camera: dict[str, list[Pose]]) -> list[Pose]:
    """Join the poses that different cameras found for the same object, and fit each group again."""
    groups: list[list[Pose]] = []
    for poses in per_camera.values():
        for p in poses:
            # A group holds at most one pose per camera: one camera never sees one object twice.
            home = next(
                (
                    g
                    for g in groups
                    if all(q.cameras != p.cameras for q in g) and any(same_object(p, q) for q in g)
                ),
                None,
            )
            if home is None:
                groups.append([p])
            else:
                home.append(p)
    out = []
    for g in groups:
        if len(g) == 1:
            out.append(g[0])
            continue
        everything = np.vstack([p.points for p in g])
        joined = fit(everything, tuple(c for p in g for c in p.cameras), np.vstack([p.whole for p in g]))
        # One bad view can spoil the joined fit. Keep whichever fit, the joined one or one camera's own, the
        # most points of the whole group lie on (within NEAR_CIRCLE of its surface).
        candidates = [p for p in [joined, *g] if p is not None and p.long] or g
        out.append(max(candidates, key=lambda p: agree(p, everything)))
    return out


def agree(p: Pose, points: np.ndarray) -> int:
    """How many of ``points`` lie on ``p``'s surface."""
    return int((np.abs(off_cylinder(points, p.centre, p.axis, p.radius)) <= NEAR_CIRCLE).sum())


def poses(view: View, masks: list[np.ndarray], camera: str) -> list[Pose]:
    """A pose for each outline in one picture."""
    out = []
    for m in masks:
        p = fit(cloud(view, m), (camera,), cloud(view, m, shave=0))
        if p is not None:
            out.append(p)
    return out
