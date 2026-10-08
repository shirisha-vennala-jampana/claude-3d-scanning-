"""The hardware path: fuse colour + depth + camera pose into a metric 3D world.

When the capture device knows its own depth and position (a phone LiDAR
with ARKit/ARCore tracking, an Intel RealSense / Azure Kinect / OAK-D on a
tracked rig, or a learned depth+pose model run on ordinary video) we don't
need Structure-from-Motion at all: every pixel can be lifted straight into
the world, and the result is in real metres.

Input is a folder with a ``capture.json``::

    {
      "intrinsics": {"fx": 500, "fy": 500, "cx": 320, "cy": 240,
                     "width": 640, "height": 480},     # of the RGB images
      "depth_scale": 0.001,          # depth PNG value * scale = metres
      "pose_convention": "opencv",   # or "opengl" (ARKit / Unity style)
      "frames": [
        {"rgb": "rgb/0000.jpg", "depth": "depth/0000.png",
         "pose": [[...4x4 camera-to-world...]], "timestamp": 0.0},
        ...
      ]
    }

Depth images may be lower resolution than the RGB (ARKit gives 256x192);
the colour image is resized down to match. ``depth`` may also be a ``.npy``
float array in metres.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .scene import Camera, Scene

GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass
class RGBDFrame:
    name: str
    rgb: np.ndarray  # BGR uint8, same size as depth
    depth: np.ndarray  # float32 metres, 0 = invalid
    K: np.ndarray
    cam_to_world: np.ndarray  # 4x4, OpenCV convention
    timestamp: float | None = None

    def camera(self) -> Camera:
        w2c = np.linalg.inv(self.cam_to_world)
        h, w = self.depth.shape
        return Camera(self.name, self.K.copy(), w2c[:3, :3].copy(), w2c[:3, 3].copy(), w, h, self.timestamp)


def _read_depth(path: Path, scale: float) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    d = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if d is None:
        raise ValueError(f"cannot read depth image {path}")
    if d.ndim == 3:
        d = d[..., 0]
    return d.astype(np.float32) * scale


def load_rgbd_capture(folder: str | Path) -> list[RGBDFrame]:
    folder = Path(folder)
    meta = json.loads((folder / "capture.json").read_text())
    intr = meta["intrinsics"]
    if "K" in intr:
        K0 = np.asarray(intr["K"], float)
    else:
        K0 = np.array([[intr["fx"], 0, intr["cx"]], [0, intr["fy"], intr["cy"]], [0, 0, 1]], float)
    scale = float(meta.get("depth_scale", 0.001))
    gl = meta.get("pose_convention", "opencv").lower() == "opengl"
    frames = []
    for i, f in enumerate(meta["frames"]):
        rgb = cv2.imread(str(folder / f["rgb"]), cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError(f"cannot read {f['rgb']}")
        depth = _read_depth(folder / f["depth"], scale)
        h, w = depth.shape
        sx, sy = w / intr.get("width", rgb.shape[1]), h / intr.get("height", rgb.shape[0])
        K = K0.copy()
        K[0] *= sx
        K[1] *= sy
        if rgb.shape[:2] != depth.shape:
            rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        pose = np.asarray(f["pose"], float)
        if gl:
            pose = pose @ GL_TO_CV
        frames.append(RGBDFrame(Path(f["rgb"]).name or f"frame_{i:04d}", rgb, depth, K, pose, f.get("timestamp")))
    return frames


def write_rgbd_capture(folder: str | Path, frames: list[RGBDFrame], depth_scale: float = 0.001) -> None:
    """Write frames in the format ``load_rgbd_capture`` reads (used by the demo)."""
    folder = Path(folder)
    (folder / "rgb").mkdir(parents=True, exist_ok=True)
    (folder / "depth").mkdir(parents=True, exist_ok=True)
    K = frames[0].K
    h, w = frames[0].depth.shape
    entries = []
    for i, f in enumerate(frames):
        rgb_name, d_name = f"rgb/{i:04d}.jpg", f"depth/{i:04d}.png"
        cv2.imwrite(str(folder / rgb_name), f.rgb, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(str(folder / d_name), np.clip(f.depth / depth_scale, 0, 65535).astype(np.uint16))
        entries.append({"rgb": rgb_name, "depth": d_name, "pose": f.cam_to_world.tolist(), "timestamp": f.timestamp})
    meta = {
        "intrinsics": {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2], "width": w, "height": h},
        "depth_scale": depth_scale,
        "pose_convention": "opencv",
        "frames": entries,
    }
    (folder / "capture.json").write_text(json.dumps(meta, indent=2))


def backproject(frame: RGBDFrame, stride: int = 1, max_depth: float = 8.0) -> tuple[np.ndarray, np.ndarray]:
    d = frame.depth[::stride, ::stride]
    h, w = d.shape
    u, v = np.meshgrid(np.arange(w) * stride, np.arange(h) * stride)
    ok = (d > 0) & (d < max_depth) & np.isfinite(d)
    z = d[ok]
    K = frame.K
    x = (u[ok] - K[0, 2]) * z / K[0, 0]
    y = (v[ok] - K[1, 2]) * z / K[1, 1]
    Xc = np.stack([x, y, z], 1)
    T = frame.cam_to_world
    Xw = Xc @ T[:3, :3].T + T[:3, 3]
    rgb = frame.rgb[::stride, ::stride][ok][:, ::-1]
    return Xw, rgb


def voxel_downsample(xyz: np.ndarray, rgb: np.ndarray, voxel: float, min_count: int = 1):
    keys = np.floor(xyz / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    n = len(counts)
    sx = np.zeros((n, 3))
    sc = np.zeros((n, 3))
    np.add.at(sx, inv, xyz)
    np.add.at(sc, inv, rgb.astype(float))
    keep = counts >= min_count
    return sx[keep] / counts[keep, None], (sc[keep] / counts[keep, None]).astype(np.uint8)


def fuse(frames: list[RGBDFrame], voxel: float = 0.02, max_depth: float = 8.0, stride: int = 2,
         min_count: int = 1) -> Scene:
    xs, cs = [], []
    for f in frames:
        x, c = backproject(f, stride, max_depth)
        xs.append(x)
        cs.append(c)
    xyz, rgb = voxel_downsample(np.concatenate(xs), np.concatenate(cs), voxel, min_count)
    return Scene(
        points=xyz,
        colors=rgb,
        cameras=[f.camera() for f in frames],
        units="meters",
        info={"method": "rgbd-fusion", "frames_in": len(frames), "voxel": voxel, "max_depth": max_depth},
    )
