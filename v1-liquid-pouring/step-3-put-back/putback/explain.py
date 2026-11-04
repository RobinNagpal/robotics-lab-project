"""Close-up pictures that explain each problem met while putting the tubes back, and each fix, for the docs.

    python -m putback.explain --out ../docs/solutions/images/step-3-put-back

Two whole runs are filmed close up, as they work now: test scene 150 (large tubes) and test scene 154 (medium
tubes). The poses are kept during the run and the pictures drawn afterwards. Some pictures are poses set by
hand (no physics) from where a run actually was, and two are drawings.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import mujoco
import numpy as np
from carry.episode import SOURCE_ARM
from grasp import side
from lab_scene import scene as lab
from PIL import Image, ImageDraw, ImageFont

from . import place, workflow
from .upright import Mover, exp_rot

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FONT = ImageFont.load_default(size=17)
SMALL = ImageFont.load_default(size=14)
RED = (230, 40, 40)
EVERY = 0.2  # s between pictures
W, H = 480, 360
CAMS = {
    # (looked at, turned round, looking down by, from how far)
    "pour": ((0.48, 0.0, 0.33), 120, -10, 0.6),
    "tube": (None, 90, 0, 0.5),  # looks at a given point, from the front
    "source_rack": ((0.45, 0.22, 0.12), 150, -25, 0.6),
    "destination_rack": ((0.45, -0.36, 0.12), -90, -25, 0.6),
}


class Camera:
    def __init__(self, model: mujoco.MjModel):
        self.r = mujoco.Renderer(model, H, W)

    def shot(self, data: mujoco.MjData, name: str, lookat=None) -> Image.Image:
        default, az, el, dist = CAMS[name]
        lookat = default if lookat is None else lookat
        c = mujoco.MjvCamera()
        c.type = mujoco.mjtCamera.mjCAMERA_FREE
        c.lookat[:] = lookat
        c.azimuth, c.elevation, c.distance = az, el, dist
        self.r._gl_context.make_current()  # other renderers are open: draw with this one's own context
        self.r.update_scene(data, camera=c)
        # Copied: the renderer writes every picture into the same memory.
        return Image.fromarray(self.r.render().copy())

    def project(self, p: np.ndarray) -> tuple[float, float]:
        """Where a point of the last picture's scene lands in it, in pixels."""
        cam = self.r.scene.camera[0]
        fwd, up = np.array(cam.forward), np.array(cam.up)
        right = np.cross(fwd, up)
        v = np.asarray(p) - np.array(cam.pos)
        z = v @ fwd
        x, y = v @ right * cam.frustum_near / z, v @ up * cam.frustum_near / z
        # A free camera leaves the frustum's width to the picture's shape.
        half = cam.frustum_width or cam.frustum_top * W / H
        left, width = cam.frustum_center - half, 2 * half
        return (x - left) / width * W, (cam.frustum_top - y) / (cam.frustum_top - cam.frustum_bottom) * H


def caption(img: Image.Image, text: str, colour=(20, 20, 20)) -> Image.Image:
    out = Image.new("RGB", (img.width, img.height + 30), colour)
    out.paste(img, (0, 30))
    ImageDraw.Draw(out).text((8, 6), text, fill=(255, 255, 255), font=FONT)
    return out


def row(images: list[Image.Image], gap: int = 8) -> Image.Image:
    w, h = images[0].size
    out = Image.new("RGB", (len(images) * w + (len(images) - 1) * gap, h), (255, 255, 255))
    for i, img in enumerate(images):
        out.paste(img, (i * (w + gap), 0))
    return out


def column(images: list[Image.Image], gap: int = 8) -> Image.Image:
    w = max(i.width for i in images)
    out = Image.new("RGB", (w, sum(i.height for i in images) + gap * (len(images) - 1)), (255, 255, 255))
    y = 0
    for img in images:
        out.paste(img, (0, y))
        y += img.height + gap
    return out


def circle(img: Image.Image, uv, r: int = 22, text: str = "") -> Image.Image:
    d = ImageDraw.Draw(img)
    u, v = uv
    d.ellipse([u - r, v - r, u + r, v + r], outline=RED, width=4)
    if text:
        d.text((u + r + 4, v - 10), text, fill=RED, font=FONT, stroke_width=3, stroke_fill=(255, 255, 255))
    return img


def arrow(img: Image.Image, a, b, text: str = "") -> Image.Image:
    d = ImageDraw.Draw(img)
    d.line([a, b], fill=RED, width=4)
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    for s in (2.6, -2.6):
        d.line([b, (b[0] + 14 * math.cos(ang + s), b[1] + 14 * math.sin(ang + s))], fill=RED, width=4)
    if text:
        d.text((b[0] + 6, b[1] + 4), text, fill=RED, font=FONT, stroke_width=3, stroke_fill=(255, 255, 255))
    return img


# ------------------------------------------------------------------------------------------- the runs


class Film:
    """The simulation's pose every EVERY s, and at the start of each stage. The pictures are drawn
    afterwards, from these poses: drawn during the run, with the pour camera's renderer also open, some
    came out upside down or from the wrong camera."""

    def __init__(self):
        self.frames: list[tuple[float, str, np.ndarray]] = []
        self.states: dict[str, np.ndarray] = {}
        self.height = 0.0  # the source's height
        self.last = -math.inf
        self.model = None
        self._cam: Camera | None = None
        self._data = None

    def __call__(self, data: mujoco.MjData, stage: str) -> None:
        if stage not in self.states:
            self.states[stage] = data.qpos.copy()
        if data.time - self.last < EVERY - 1e-6:
            return
        self.last = data.time
        self.frames.append((data.time, stage, data.qpos.copy()))

    def image(self, i: int, cam: str) -> Image.Image:
        if self._cam is None:
            self._cam = Camera(self.model)
            self._data = mujoco.MjData(self.model)
        self._data.qpos[:] = self.frames[i][2]
        mujoco.mj_forward(self.model, self._data)
        return self._cam.shot(self._data, cam)

    def pick(self, wanted, cam: str, where: float = 0.5) -> Image.Image:
        chosen = [i for i, f in enumerate(self.frames) if wanted(f[1])]
        return self.image(chosen[min(len(chosen) - 1, int(where * (len(chosen) - 1)))], cam)


def film(start: str, cache: Path | None = None) -> tuple[Film, dict]:
    if cache is not None:
        path = cache / f"{start}.pkl"
        if path.exists():
            import pickle

            f, result = pickle.loads(path.read_bytes())
            f.model = lab.build(result["seed"])[0]
            return f, result
    f = Film()
    result = workflow.run_one(workflow.STARTS / f"{start}.npy", on_frame=f)
    f.model = lab.build(result["seed"])[0]
    if cache is not None:
        import pickle

        cache.mkdir(parents=True, exist_ok=True)
        model = f.model
        f.model = None
        path.write_bytes(pickle.dumps((f, result)))
        f.model = model
    return f, result


# --------------------------------------------------------------------------------- posed pictures


def posed(model, qpos, arm: str, tube: str, site_R: np.ndarray | None = None, tube_R=None):
    """A copy of the scene with the arm moved (no physics) so that its pinch point stays where it is and
    either the gripper is turned ``site_R`` or the tube is turned ``tube_R``; the tube moves with the hand."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    mover = Mover(model, arm)
    pos, rot = mover.grip(data, tube)  # the pinch point in the tube's frame
    held_pos, held_mat = -rot.T @ pos, rot.T  # the tube in the pinch point's frame
    b = data.body(tube).xpos.copy()
    if tube_R is not None:
        target_p, target_R = b + tube_R @ pos, tube_R @ rot
    else:
        target_p, target_R = data.site_xpos[mover.site].copy(), site_R
    q, miss = mover.solve(data, target_p, target_R, iterations=400)
    data.qpos[mover.joints] = q
    mujoco.mj_kinematics(model, data)
    site = data.site_xpos[mover.site]
    mat = data.site_xmat[mover.site].reshape(3, 3)
    adr = model.jnt_qposadr[model.body(tube).jntadr[0]]
    # The tube's body sits at its bottom; place it from the pinch point.
    data.qpos[adr : adr + 3] = site + mat @ held_pos
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, (mat @ held_mat).ravel())
    data.qpos[adr + 3 : adr + 7] = quat
    # The balls stay where they were; hide the source's balls by moving them with the tube is not needed
    # for these close-ups of the arm.
    mujoco.mj_forward(model, data)
    return data, miss


def level_fingers(data: mujoco.MjData, arm: str) -> np.ndarray:
    """The gripper turned the least way so that its fingers are level (as Step 1b holds a tube)."""
    site = data.site_xmat[data.model.site(lab.pinch_site(arm)).id].reshape(3, 3)
    x = site[:, 0]
    axis = np.cross(x, [0.0, 0.0, -1.0])
    s = np.linalg.norm(axis)
    return (exp_rot(axis / s * math.atan2(s, -x[2])) if s > 1e-9 else np.eye(3)) @ site


def fold_points(model, data, arm: str) -> list[np.ndarray]:
    mujoco.mj_collision(model, data)
    out = []
    for c in data.contact[: data.ncon]:
        n1 = model.body(model.geom_bodyid[c.geom1]).name
        n2 = model.body(model.geom_bodyid[c.geom2]).name
        if n1.startswith(f"{arm}/") and n2.startswith(f"{arm}/") and ("gripper/" in n1) != ("gripper/" in n2):
            out.append(np.array(c.pos))
    return out


def lean(data, tube) -> float:
    return math.degrees(math.acos(np.clip(data.body(tube).xmat.reshape(3, 3)[2, 2], -1, 1)))


def turned(f: Film, out: Path, source: str) -> None:
    """The source before the pour, and after: turned a little in the fingers, so with the fingers level it
    leans; tilting the gripper stands it straight again."""
    model = f.model
    cam = Camera(model)

    def at(qpos):
        data = mujoco.MjData(model)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        return data

    before = at(f.states["move in"])  # where Step 1b left it, before the pour
    after = at(f.states["arm1 upright"])  # tipped back after the pour
    level, _ = posed(model, after.qpos, SOURCE_ARM, source, site_R=level_fingers(after, SOURCE_ARM))
    straight, _ = posed(
        model, after.qpos, SOURCE_ARM, source, tube_R=place.upright(after.body(source).xmat.reshape(3, 3))
    )
    height = f.height
    pics = []
    for data, text in (
        (before, "Before the pour: straight in the fingers"),
        (level, f"After the pour, fingers level: leans {lean(level, source):.0f} degrees"),
        (straight, "Gripper tilted a little: straight again"),
    ):
        b = data.body(source).xpos.copy()
        axis = data.body(source).xmat.reshape(3, 3)[:, 2]
        cam_target = b + axis * height / 2
        img = cam.shot(data, "tube", lookat=cam_target)
        bottom, top = cam.project(b), cam.project(b + axis * height)
        straight_up = cam.project(b + [0, 0, height])
        d = ImageDraw.Draw(img)
        d.line([bottom, straight_up], fill=(255, 255, 255), width=3)
        d.line([bottom, top], fill=RED, width=3)
        pics.append(caption(img, text))
    row(pics).save(out / "turned-in-fingers.png")


def stand_up_strip(f: Film, out: Path) -> None:
    """How a tube goes into its cell: carried half upright, stood up just above the cell, lowered, let go."""
    s = "source_rack"
    pics = [
        caption(f.pick(lambda st: st.startswith("arm1 tube"), s, 0.25), "1. Carried half upright"),
        caption(f.pick(lambda st: st.startswith("arm1 tube"), s, 0.55), "2. Stood up just above the cell"),
        caption(f.pick(lambda st: st.startswith("arm1 tube"), s, 0.95), "3. Lowered into its own cell"),
        caption(f.pick(lambda st: st == "arm1 let go", s, 0.35), "4. Fingers open, hand backs away"),
    ]
    row(pics).save(out / "into-the-cell.png")


def tip_back_strip(f: Film, out: Path) -> None:
    p = "pour"
    pics = [
        caption(f.pick(lambda st: st == "done", p, 0.9), "1. Pour done: tipped back to 70 degrees"),
        caption(f.pick(lambda st: st == "tipping back", p, 0.12), "2. Straight up, still tipped"),
        caption(f.pick(lambda st: st == "tipping back", p, 0.55), "3. Back the way it came in"),
        caption(f.pick(lambda st: st == "arm1 upright", p, 0.0), "4. Where Step 1b left it"),
    ]
    row(pics).save(out / "tip-back.png")


def grippers_touch(f: Film, out: Path) -> None:
    """With medium tubes the two grippers end the pour touching."""
    model = f.model
    data = mujoco.MjData(model)
    data.qpos[:] = f.states["tipping back"]
    mujoco.mj_forward(model, data)
    cam = Camera(model)
    img = cam.shot(data, "pour")
    mujoco.mj_collision(model, data)
    touches = []
    for c in data.contact[: data.ncon]:
        n1 = model.body(model.geom_bodyid[c.geom1]).name
        n2 = model.body(model.geom_bodyid[c.geom2]).name
        if (n1.startswith("arm1/") and n2.startswith("arm2/")) or (
            n1.startswith("arm2/") and n2.startswith("arm1/")
        ):
            touches.append(np.array(c.pos))
    if touches:
        circle(img, cam.project(np.mean(touches, axis=0)), 20, "grippers touch")
    caption(img, "Medium tubes: the pour ends with the two grippers touching").save(
        out / "grippers-touch.png"
    )


def palm_diagram(out: Path) -> None:
    """Seen from above: why backing away "from the tube's centre through the pinch point" pushed the palm
    into the tube, and the way that works."""
    r = 12.5  # mm, the tube
    depth = side.DEPTH * 1000  # the pinch point is this far past the tube's centre, towards the fingertips
    gap = side.PRESHAPE_GAP * 1000 / 2  # the fingers, opened to 35 mm
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, wrong in zip(axes, (True, False), strict=True):
        ax.add_patch(plt.Circle((0, 0), r, fc="#dfe9f5", ec="k"))
        ax.text(0, -8, "tube", ha="center", fontsize=9)
        ax.add_patch(plt.Rectangle((-62, -30), 34, 60, fc="0.35", ec="k"))  # the palm
        ax.text(-45, 0, "palm", ha="center", va="center", color="w", fontsize=10)
        for y in (gap, -gap - 6):
            ax.add_patch(plt.Rectangle((-28, y), 52, 6, fc="0.55", ec="k"))  # a finger
        ax.plot([0], [0], "k+", ms=10)
        ax.plot([depth], [0], "x", color="#c03030" if wrong else "0.3", ms=11, mew=3)
        ax.annotate(
            "point between\nthe fingertips",
            (depth, 0),
            (14, -32),
            fontsize=9,
            ha="left",
            arrowprops={"arrowstyle": "->", "color": "0.3"},
        )
        ax.annotate(
            "tube's centre",
            (0, 0),
            (-20, 36),
            fontsize=9,
            ha="center",
            arrowprops={"arrowstyle": "->", "color": "0.3"},
        )
        if wrong:
            ax.annotate(
                "", (-10, -44), (-58, -44), arrowprops={"arrowstyle": "-|>", "color": "#c03030", "lw": 3}
            )
            ax.text(
                -34,
                -55,
                "the hand moved this way:\nthe palm hits the tube",
                ha="center",
                color="#c03030",
                fontsize=10,
            )
            ax.set_title(
                "First way: from the tube's centre\nthrough the point between the fingertips", fontsize=11
            )
        else:
            ax.annotate(
                "", (-80, -44), (-32, -44), arrowprops={"arrowstyle": "-|>", "color": "#208030", "lw": 3}
            )
            ax.text(
                -56,
                -55,
                "the hand moves this way:\nthe fingers slide off",
                ha="center",
                color="#208030",
                fontsize=10,
            )
            ax.set_title("Now: backwards, the way the fingers point", fontsize=11)
        ax.set_xlim(-90, 40)
        ax.set_ylim(-68, 46)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.suptitle("Seen from above, the moment the fingers have opened", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "palm-push.png", dpi=110)
    plt.close(fig)


def reach_map(out: Path) -> None:
    """The destination rack from above, drawn large: each cell's number and distance from Arm A's base."""
    cells = place.cells(np.array([0.45, -0.36]))
    base = np.array([0.9, 0.0])
    fig, ax = plt.subplots(figsize=(8.4, 4.3))
    for i, c in enumerate(cells):
        col, row = i % 5, i // 5
        dist = np.linalg.norm(c[:2] - base)
        tried_before = i in (0, 1, 2)
        ax.add_patch(
            plt.Rectangle((col, row), 0.92, 0.92, fc="#f3c4c4" if tried_before else "#d4ebd4", ec="k", lw=1.2)
        )
        ax.text(col + 0.46, row + 0.6, f"cell {i}", ha="center", fontsize=11, weight="bold")
        ax.text(col + 0.46, row + 0.28, f"{dist * 100:.0f} cm", ha="center", fontsize=11)
    ax.annotate(
        "to Arm A's base\n(nearest: cell 9, 51 cm)",
        (6.3, 2.7),
        (5.0, 1.6),
        fontsize=11,
        ha="left",
        arrowprops={"arrowstyle": "-|>", "lw": 2, "color": "0.3"},
    )
    ax.text(
        0.0,
        -0.55,
        "Red: tried first at the start (in number order). They failed for every medium tube.\n"
        "For cell 0, every try failed because the arm could not reach that far out and that low.\n"
        "Now the cells are tried nearest first: cell 9, then 4, 8, 3, ...",
        fontsize=10,
        va="top",
    )
    ax.set_xlim(-0.2, 7.6)
    ax.set_ylim(-1.3, 2.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("The destination rack seen from above, each cell's distance from Arm A's base", fontsize=12)
    fig.tight_layout()
    fig.savefig(out / "reach.png", dpi=110)
    plt.close(fig)


def clip(f: Film, out: Path) -> None:
    """Steps 3 and 4 close up: the source into its own cell, then the receiver into the destination rack."""
    frames = []
    for i, (_, stage, _) in enumerate(f.frames):
        if i % 2:
            continue
        if stage.startswith("arm1") and stage != "arm1 upright":
            text, cam = "Step 3: Arm B puts the source back in its own cell", "source_rack"
        elif stage.startswith("arm2"):
            text, cam = "Step 4: Arm A puts the receiver in the destination rack", "destination_rack"
        else:
            continue
        frames.append(caption(f.image(i, cam), text))
    small = [im.resize((360, 292), Image.LANCZOS) for im in frames]
    small[0].save(
        out / "put-back.gif", save_all=True, append_images=small[1:], duration=110, loop=0, optimize=True
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("runs/explain"), help="filmed runs kept here")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    reach_map(args.out)
    large, _ = film("seed150_size200", cache=args.cache)
    stand_up_strip(large, args.out)
    tip_back_strip(large, args.out)
    clip(large, args.out)
    right, _ = film("seed154_size150", cache=args.cache)
    right.height = 0.15
    turned(right, args.out, json_source("seed154_size150"))
    grippers_touch(right, args.out)
    palm_diagram(args.out)
    print("done")


def json_source(start: str) -> str:
    import json

    return json.loads((workflow.STARTS / f"{start}.json").read_text())["source"]


if __name__ == "__main__":
    main()
