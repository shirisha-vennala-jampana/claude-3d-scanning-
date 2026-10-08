"""Understand layer: flatten the 3D world into pictures an image model can read.

* ``floor_plan``    top-down occupancy map with camera path and tags
* ``context_sheet`` the floor plan beside numbered keyframes, plus a text
                    prompt explaining how they relate. Hand both to any
                    vision-language model and ask questions about the space
                    ("which way to the sofa from photo 3?").

This borrows the idea behind recent "put the whole space on one canvas"
work: image models are good at reading pictures, so give them a single
picture of the whole space instead of a raw 3D file.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .scene import Scene

PALETTE = [  # BGR, distinguishable on light and dark backgrounds
    (40, 120, 230), (60, 170, 60), (200, 90, 40), (170, 60, 170),
    (30, 170, 200), (120, 120, 30), (90, 60, 200), (150, 150, 150),
]


@dataclass
class PlanFrame:
    """Maps world coordinates to floor-plan pixels."""

    origin: np.ndarray  # world point at plan pixel (0, 0) before margin
    e1: np.ndarray  # world direction of +x on the plan
    e2: np.ndarray  # world direction of -y on the plan (north/up on the image)
    up: np.ndarray
    px_per_unit: float
    margin: int
    size: tuple[int, int]  # (width, height)
    floor_height: float
    ceiling_height: float

    def to_px(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X) - self.origin
        a, b = X @ self.e1, X @ self.e2
        return np.stack([self.margin + a * self.px_per_unit, self.size[1] - self.margin - b * self.px_per_unit], 1)

    def to_world(self, px: np.ndarray, height: float | None = None) -> np.ndarray:
        px = np.atleast_2d(px).astype(float)
        a = (px[:, 0] - self.margin) / self.px_per_unit
        b = (self.size[1] - self.margin - px[:, 1]) / self.px_per_unit
        h = self.floor_height if height is None else height
        return self.origin + a[:, None] * self.e1 + b[:, None] * self.e2 + (h - self.origin @ self.up) * self.up

    def to_dict(self) -> dict:
        return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.__dict__.items()}


def estimate_up(scene: Scene, iters: int = 400, seed: int = 0) -> np.ndarray:
    """Refine the up direction by finding the floor plane.

    Starts from the average camera 'up' (people hold phones roughly upright)
    and looks for a large plane among the lowest points whose normal is
    within 30 degrees of it. Falls back to the camera prior if none is found.
    """
    prior = scene.up_vector() if scene.up is None else np.asarray(scene.up, float)
    X = scene.points
    if len(X) < 100:
        return prior
    h = X @ prior
    low = X[h < np.percentile(h, 30)]
    ext = np.ptp(X, axis=0).max()
    thresh = 0.01 * ext
    rng = np.random.default_rng(seed)
    best_n, best_inl = None, 0
    for _ in range(iters):
        a, b, c = low[rng.choice(len(low), 3, replace=False)]
        n = np.cross(b - a, c - a)
        if np.linalg.norm(n) < 1e-12:
            continue
        n /= np.linalg.norm(n)
        if abs(n @ prior) < np.cos(np.radians(30)):
            continue
        inl = int(np.sum(np.abs((low - a) @ n) < thresh))
        if inl > best_inl:
            best_n, best_inl, best_a = n, inl, a
    if best_n is None or best_inl < max(50, 0.2 * len(low)):
        return prior
    P = low[np.abs((low - best_a) @ best_n) < thresh]
    _, _, Vt = np.linalg.svd(P - P.mean(0), full_matrices=False)
    n = Vt[-1]
    return n if n @ prior > 0 else -n


def _basis(up: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal axes aligned with the walls.

    Tries every rotation in 1 degree steps and keeps the one where the
    points pile up into the sharpest 1D histograms along both axes, which
    is what axis-aligned walls do. Rooms are rarely exactly rectangular, but
    nearly always have one dominant pair of wall directions.
    """
    helper = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    a = np.cross(up, helper)
    a /= np.linalg.norm(a)
    b = np.cross(up, a)
    if len(points) < 10:
        return a, b
    P = np.stack([points @ a, points @ b], 1)
    P -= np.median(P, 0)
    if len(P) > 20000:
        P = P[np.random.default_rng(0).choice(len(P), 20000, replace=False)]
    lim = np.percentile(np.abs(P), 99) + 1e-9
    best, best_score = 0.0, np.inf
    for th in np.radians(np.arange(0, 90, 1.0)):
        c, s_ = np.cos(th), np.sin(th)
        score = 0.0
        for proj in (P[:, 0] * c + P[:, 1] * s_, -P[:, 0] * s_ + P[:, 1] * c):
            hist, _ = np.histogram(proj, bins=200, range=(-lim, lim))
            q = hist[hist > 0] / hist.sum()
            score += -np.sum(q * np.log(q))
        if score < best_score:
            best, best_score = th, score
    e1 = np.cos(best) * a + np.sin(best) * b
    return e1, np.cross(up, e1)


def floor_plan(scene: Scene, max_px: int = 900, margin: int = 40, draw_cameras: bool = True,
               draw_tags: bool = True) -> tuple[np.ndarray, PlanFrame]:
    up = scene.up_vector()
    X = scene.points
    h = X @ up
    lo, hi = np.percentile(h, [2, 98]) if len(h) else (0.0, 1.0)
    span = max(hi - lo, 1e-6)
    # points in the middle height band are walls and furniture, not floor/ceiling
    band = (h > lo + 0.08 * span) & (h < hi - 0.08 * span)
    sel = X[band] if band.sum() > 50 else X
    e1, e2 = _basis(up, sel)
    a, b = X @ e1, X @ e2
    cams = np.array([c.center for c in scene.cameras]).reshape(-1, 3)
    ca, cb = (cams @ e1, cams @ e2) if len(cams) else (np.zeros(0), np.zeros(0))
    amin, amax = np.percentile(np.r_[a, ca], [1, 99]) if len(a) else (0, 1)
    bmin, bmax = np.percentile(np.r_[b, cb], [1, 99]) if len(b) else (0, 1)
    ext = max(amax - amin, bmax - bmin, 1e-6)
    ppu = (max_px - 2 * margin) / ext
    W = int((amax - amin) * ppu) + 2 * margin
    H = int((bmax - bmin) * ppu) + 2 * margin
    origin = amin * e1 + bmin * e2
    pf = PlanFrame(origin, e1, e2, up, ppu, margin, (W, H), float(lo), float(hi))

    # occupancy: how many points stack up over each floor cell
    px = pf.to_px(sel).astype(int)
    ok = (px[:, 0] >= 0) & (px[:, 0] < W) & (px[:, 1] >= 0) & (px[:, 1] < H)
    occ = np.zeros((H, W), np.float32)
    np.add.at(occ, (px[ok, 1], px[ok, 0]), 1)
    occ = cv2.GaussianBlur(occ, (0, 0), max(1.0, ppu * 0.01))
    if occ.max() > 0:
        occ = np.log1p(occ) / np.log1p(occ).max()
    # floor footprint: where floor points exist, shade lightly
    floor = X[h <= lo + 0.08 * span]
    fp = np.zeros((H, W), np.float32)
    if len(floor):
        q = pf.to_px(floor).astype(int)
        okf = (q[:, 0] >= 0) & (q[:, 0] < W) & (q[:, 1] >= 0) & (q[:, 1] < H)
        fp[q[okf, 1], q[okf, 0]] = 1
        fp = cv2.dilate(fp, np.ones((3, 3), np.uint8), iterations=max(1, int(ppu * 0.03)))
    img = np.full((H, W, 3), 255, np.uint8)
    img[fp > 0] = (235, 240, 245)
    dark = (occ[..., None] * np.array([215, 200, 180])).astype(np.uint8)
    img = np.clip(img.astype(int) - dark, 0, 255).astype(np.uint8)

    if draw_cameras and len(cams):
        path = pf.to_px(cams).astype(int)
        cv2.polylines(img, [path.reshape(-1, 1, 2)], False, (180, 180, 180), 1, cv2.LINE_AA)
    _scale_bar(img, pf, scene.units)
    if draw_tags and scene.annotations:
        # short numbered labels on the map, full text in a legend underneath
        legend_h = 14 + 22 * len(scene.annotations)
        img = np.vstack([img, np.full((legend_h, W, 3), 255, np.uint8)])
        cv2.line(img, (10, H + 4), (W - 10, H + 4), (220, 220, 220), 1)
        for i, ann in enumerate(scene.annotations):
            col = PALETTE[i % len(PALETTE)]
            p = tuple(pf.to_px(np.asarray(ann.position))[0].astype(int))
            r = max(4, int(ann.radius * ppu))
            cv2.circle(img[:H], p, r, col, 2, cv2.LINE_AA)
            cv2.circle(img[:H], p, 3, col, -1, cv2.LINE_AA)
            short = f"{tag_letter(i)} {ann.label}"
            short = short if len(short) <= 18 else short[:16] + ".."
            (tw, _), _ = cv2.getTextSize(short, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            x = p[0] + r + 4 if p[0] + r + 4 + tw < W - 4 else p[0] - r - 4 - tw
            _label(img, short, (max(2, x), p[1] + 5), col)
            y = H + 24 + 22 * i
            cv2.circle(img, (20, y - 5), 6, col, -1, cv2.LINE_AA)
            text = f"{tag_letter(i)}  {ann.label}"
            props = ", ".join(f"{k}={v}" for k, v in ann.properties.items() if not isinstance(v, (dict, list)))
            if props:
                text += f"  ({props})"
            max_chars = max(10, (W - 40) // 8)
            text = text if len(text) <= max_chars else text[: max_chars - 2] + ".."
            cv2.putText(img, text, (34, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)
    return img, pf


def draw_camera_markers(img: np.ndarray, pf: PlanFrame, scene: Scene, indices: list[int]) -> None:
    for n, ci in enumerate(indices, start=1):
        cam = scene.cameras[ci]
        p = pf.to_px(cam.center)[0]
        f = cam.forward - (cam.forward @ pf.up) * pf.up
        if np.linalg.norm(f) > 1e-6:
            f /= np.linalg.norm(f)
            tip = pf.to_px(cam.center + f * (30 / pf.px_per_unit))[0]
            cv2.arrowedLine(img, tuple(p.astype(int)), tuple(tip.astype(int)), (30, 30, 200), 2, cv2.LINE_AA, tipLength=0.35)
        cv2.circle(img, tuple(p.astype(int)), 11, (30, 30, 200), -1, cv2.LINE_AA)
        txt = str(n)
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(img, txt, (int(p[0] - tw / 2), int(p[1] + th / 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)


def tag_letter(i: int) -> str:
    """A, B, ..., Z, AA, AB, ... (photos use numbers, so tags use letters)."""
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def _label(img, text, org, col):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)


def _nice(x: float) -> float:
    e = 10 ** np.floor(np.log10(x))
    for m in (1, 2, 5, 10):
        if m * e >= x:
            return m * e
    return 10 * e


def _scale_bar(img, pf: PlanFrame, units: str) -> None:
    length = _nice(img.shape[1] / pf.px_per_unit / 6)
    L = int(length * pf.px_per_unit)
    x0, y0 = 15, img.shape[0] - 15
    cv2.line(img, (x0, y0), (x0 + L, y0), (40, 40, 40), 2)
    unit = "m" if units == "meters" else "units"
    cv2.putText(img, f"{length:g} {unit}", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 40, 40), 1, cv2.LINE_AA)


def pick_keyframes(scene: Scene, n: int = 8) -> list[int]:
    if not scene.cameras:
        return []
    return sorted(set(np.linspace(0, len(scene.cameras) - 1, min(n, len(scene.cameras))).round().astype(int).tolist()))


def context_sheet(scene: Scene, frames_dir: str | Path, n_views: int = 8, thumb_w: int = 320) -> tuple[np.ndarray, str]:
    """Floor plan + numbered keyframes on one canvas, and a prompt describing it."""
    plan, pf = floor_plan(scene, max_px=700)
    idx = pick_keyframes(scene, n_views)
    draw_camera_markers(plan, pf, scene, idx)

    thumbs = []
    for n, ci in enumerate(idx, start=1):
        cam = scene.cameras[ci]
        im = cv2.imread(str(Path(frames_dir) / cam.name))
        if im is None:
            im = np.full((cam.height, cam.width, 3), 200, np.uint8)
        th = cv2.resize(im, (thumb_w, int(im.shape[0] * thumb_w / im.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.rectangle(th, (0, 0), (34, 28), (30, 30, 200), -1)
        cv2.putText(th, str(n), (7, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        thumbs.append(th)
    cols = 2
    if thumbs:
        tw, thh = thumbs[0].shape[1], max(t.shape[0] for t in thumbs)
        rows = (len(thumbs) + cols - 1) // cols
        grid = np.full((rows * (thh + 6), cols * (tw + 6), 3), 255, np.uint8)
        for k, t in enumerate(thumbs):
            r, c = divmod(k, cols)
            grid[r * (thh + 6): r * (thh + 6) + t.shape[0], c * (tw + 6): c * (tw + 6) + tw] = t
    else:
        grid = np.full((plan.shape[0], 10, 3), 255, np.uint8)
    H = max(plan.shape[0], grid.shape[0])
    canvas = np.full((H + 40, plan.shape[1] + grid.shape[1] + 20, 3), 255, np.uint8)
    canvas[40:40 + plan.shape[0], : plan.shape[1]] = plan
    canvas[40:40 + grid.shape[0], plan.shape[1] + 20:] = grid
    cv2.putText(canvas, "Top-down map (numbers = where each photo was taken, arrow = facing)", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)

    unit = "metres" if scene.units == "meters" else "scene units (scale unknown)"
    lines = [
        "You are looking at a single image that describes one indoor space.",
        "LEFT: a top-down map. Dark marks are walls and furniture seen from above; the light area is floor.",
        "Red numbered circles show where each photo on the RIGHT was taken; the arrow shows which way the camera faced.",
        f"Distances on the map are in {unit}; see the scale bar in the bottom-left corner.",
    ]
    if scene.annotations:
        lines.append("Coloured circles marked with letters are tagged objects (listed under the map):")
        for i, a in enumerate(scene.annotations):
            props = ", ".join(f"{k}={v}" for k, v in a.properties.items() if not isinstance(v, (dict, list)))
            lines.append(f"  {tag_letter(i)}. {a.label}" + (f" ({props})" if props else ""))
    lines += ["", "Question: <ask about the space here, e.g. 'Standing where photo 2 was taken, which way do I turn to reach the table?'>"]
    return canvas, "\n".join(lines)


def save_plan(scene: Scene, out_dir: str | Path) -> None:
    out = Path(out_dir)
    img, pf = floor_plan(scene)
    cv2.imwrite(str(out / "floorplan.png"), img)
    (out / "floorplan.json").write_text(json.dumps(pf.to_dict(), indent=2))
    sheet, prompt = context_sheet(scene, out / "frames")
    cv2.imwrite(str(out / "context_sheet.png"), sheet)
    (out / "context_prompt.txt").write_text(prompt + "\n")
