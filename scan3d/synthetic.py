"""A tiny ray-caster for a textured room, with exact ground truth.

Used by the tests and by ``scan3d demo`` so the whole pipeline can be tried
without recording anything. It renders colour, depth and camera poses, i.e.
exactly what a phone with a depth sensor + tracking (ARKit/ARCore-style)
would hand you.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .scene import Camera


@dataclass
class Quad:
    """Axis-aligned rectangle: ``axis`` is its normal axis, at coordinate ``c``."""

    axis: int
    c: float
    lo: tuple[float, float]  # bounds on the other two axes (in increasing axis order)
    hi: tuple[float, float]
    texture: np.ndarray  # BGR uint8
    px_per_m: float = 200.0
    name: str = ""


def _texture(seed: int, size: int = 1024, tint=(1.0, 1.0, 1.0)) -> np.ndarray:
    """Multi-scale blobby noise: lots of corners/blobs for SIFT to latch on to."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size, 3), np.float32)
    for scale, weight in ((8, 0.5), (32, 0.3), (128, 0.2)):
        n = rng.random((size // scale + 2, size // scale + 2, 3)).astype(np.float32)
        img += weight * cv2.resize(n, (size, size), interpolation=cv2.INTER_CUBIC)
    # sprinkle some sharp-edged shapes
    for _ in range(250):
        c = tuple(int(x) for x in rng.integers(0, 255, 3))
        p = tuple(int(x) for x in rng.integers(0, size, 2))
        r = int(rng.integers(4, 22))
        if rng.random() < 0.5:
            cv2.circle(img, p, r, tuple(x / 255 for x in c), -1)
        else:
            cv2.rectangle(img, p, (p[0] + r, p[1] + r), tuple(x / 255 for x in c), -1)
    img = np.clip(img * np.array(tint, np.float32), 0, 1)
    return (img * 255).astype(np.uint8)


def box_room(width=5.0, depth=4.0, height=2.6, with_table=True, with_cabinet=True) -> list[Quad]:
    """A room spanning x in [-w/2, w/2], y in [-d/2, d/2], z (up) in [0, h]."""
    W, D, H = width / 2, depth / 2, height
    q = [
        Quad(2, 0.0, (-W, -D), (W, D), _texture(1, tint=(0.8, 0.9, 1.0)), name="floor"),
        Quad(2, H, (-W, -D), (W, D), _texture(2, tint=(1.0, 1.0, 1.0)), name="ceiling"),
        Quad(0, -W, (-D, 0), (D, H), _texture(3, tint=(1.0, 0.85, 0.85)), name="wall_west"),
        Quad(0, W, (-D, 0), (D, H), _texture(4, tint=(0.85, 1.0, 0.85)), name="wall_east"),
        Quad(1, -D, (-W, 0), (W, H), _texture(5, tint=(0.85, 0.85, 1.0)), name="wall_south"),
        Quad(1, D, (-W, 0), (W, H), _texture(6, tint=(1.0, 1.0, 0.8)), name="wall_north"),
    ]
    if with_table:
        q += box((-0.5, -0.4, 0.0), (0.5, 0.4, 0.75), seed=10, name="table")
    if with_cabinet:
        q += box((-W + 0.05, 0.4, 0.0), (-W + 0.6, 1.4, 1.8), seed=20, name="cabinet")
    return q


def box(lo, hi, seed=0, name="box") -> list[Quad]:
    (x0, y0, z0), (x1, y1, z1) = lo, hi
    t = _texture(seed, 512, tint=(0.9, 0.75, 0.6))
    return [
        Quad(2, z1, (x0, y0), (x1, y1), t, name=f"{name}_top"),
        Quad(0, x0, (y0, z0), (y1, z1), t, name=f"{name}_side"),
        Quad(0, x1, (y0, z0), (y1, z1), t, name=f"{name}_side"),
        Quad(1, y0, (x0, z0), (x1, z1), t, name=f"{name}_side"),
        Quad(1, y1, (x0, z0), (x1, z1), t, name=f"{name}_side"),
    ]


def look_at(eye, target, world_up=(0, 0, 1)) -> tuple[np.ndarray, np.ndarray]:
    """World->camera (R, t) for an OpenCV camera at ``eye`` looking at ``target``."""
    eye, target = np.asarray(eye, float), np.asarray(target, float)
    z = target - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, world_up)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z])
    return R, -R @ eye


def render(quads: list[Quad], cam: Camera) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (BGR image, z-depth in metres, quad index per pixel)."""
    h, w = cam.height, cam.width
    u, v = np.meshgrid(np.arange(w) + 0.0, np.arange(h) + 0.0)
    rays_c = np.stack([u, v, np.ones_like(u)], -1) @ np.linalg.inv(cam.K).T  # z = 1
    d = rays_c @ cam.R  # world directions, rows: R^T @ r
    o = cam.center
    best = np.full((h, w), np.inf)
    idx = np.full((h, w), -1, int)
    for qi, q in enumerate(quads):
        a, (b, c) = q.axis, [k for k in range(3) if k != q.axis]
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (q.c - o[a]) / d[..., a]
        pb, pc = o[b] + t * d[..., b], o[c] + t * d[..., c]
        hit = (t > 1e-6) & (t < best) & (pb >= q.lo[0]) & (pb <= q.hi[0]) & (pc >= q.lo[1]) & (pc <= q.hi[1])
        best[hit] = t[hit]
        idx[hit] = qi
    img = np.zeros((h, w, 3), np.uint8)
    P = o + best[..., None] * d
    for qi, q in enumerate(quads):
        m = idx == qi
        if not m.any():
            continue
        b, c = [k for k in range(3) if k != q.axis]
        th, tw = q.texture.shape[:2]
        mx = ((P[..., b] - q.lo[0]) * q.px_per_m) % tw
        my = ((P[..., c] - q.lo[1]) * q.px_per_m) % th
        samp = cv2.remap(q.texture, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        img[m] = samp[m]
    depth = np.where(np.isfinite(best), best, 0.0)  # t along a z=1 ray == z-depth
    return img, depth, idx


def orbit_cameras(n: int = 30, width: int = 640, height: int = 480, focal: float = 500.0,
                  radius: float = 1.1, eye_height: float = 1.5, arc_deg: float = 200.0) -> list[Camera]:
    """Walk around the room centre looking across it, like a handheld scan."""
    K = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], float)
    cams = []
    for i, a in enumerate(np.radians(np.linspace(-arc_deg / 2, arc_deg / 2, n))):
        eye = np.array([radius * np.cos(a), radius * np.sin(a), eye_height])
        target = np.array([-1.8 * np.cos(a + 0.35), -1.8 * np.sin(a + 0.35), 0.9])
        R, t = look_at(eye, target)
        cams.append(Camera(f"frame_{i:04d}.png", K.copy(), R, t, width, height, timestamp=i / 10.0))
    return cams


def write_video(images: list[np.ndarray], path: str, fps: float = 10.0) -> None:
    h, w = images[0].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for im in images:
        vw.write(im)
    vw.release()
