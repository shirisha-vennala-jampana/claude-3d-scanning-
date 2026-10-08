"""The Scene: geometry (points + cameras) plus the metadata embedded in it.

A scene lives in a directory:

    scene.json   cameras, annotations, units, up vector (human-readable)
    points.npz   xyz (N,3) float32, rgb (N,3) uint8
    frames/      the images the cameras refer to (optional)

Everything else in the directory (PLY, viewer, floor plan) is derived from
these files by ``scan3d.export`` and can be regenerated at any time.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Camera:
    """A pinhole camera. ``R``/``t`` map world -> camera: x_cam = R @ x_world + t."""

    name: str
    K: np.ndarray
    R: np.ndarray
    t: np.ndarray
    width: int
    height: int
    timestamp: float | None = None  # seconds since start of capture, if known

    @property
    def center(self) -> np.ndarray:
        return -self.R.T @ self.t

    @property
    def up(self) -> np.ndarray:
        """World-space 'up' of the image (OpenCV cameras have +y pointing down)."""
        return -self.R[1]

    @property
    def forward(self) -> np.ndarray:
        return self.R[2]

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project world points (N,3). Returns pixel coords (N,2) and depths (N,)."""
        Xc = X @ self.R.T + self.t
        z = Xc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = (Xc @ self.K.T)[:, :2] / z[:, None]
        return uv, z

    def ray(self, u: float, v: float) -> tuple[np.ndarray, np.ndarray]:
        """World-space ray (origin, unit direction) through pixel (u, v)."""
        d = self.R.T @ (np.linalg.inv(self.K) @ np.array([u, v, 1.0]))
        return self.center, d / np.linalg.norm(d)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "K": self.K.tolist(),
            "R": self.R.tolist(),
            "t": self.t.tolist(),
            "width": self.width,
            "height": self.height,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Camera":
        return cls(
            name=d["name"],
            K=np.asarray(d["K"], float),
            R=np.asarray(d["R"], float),
            t=np.asarray(d["t"], float),
            width=int(d["width"]),
            height=int(d["height"]),
            timestamp=d.get("timestamp"),
        )


@dataclass
class Annotation:
    """A piece of metadata anchored to a place in the 3D world.

    The anchor is a sphere (``position`` + ``radius``); every point inside it
    is considered to belong to the annotation. ``properties`` is free-form
    (strings, numbers, nested dicts) so it can hold anything: a label from a
    detector, a note, a sensor reading, a link, a price, a maintenance date.
    """

    label: str
    position: list[float]
    radius: float = 0.25
    properties: dict = field(default_factory=dict)
    source: str = "manual"  # manual | pixel | sensor | viewer | <model name>
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Annotation":
        known = {k: d[k] for k in ("label", "position", "radius", "properties", "source", "id", "created") if k in d}
        known["position"] = [float(x) for x in known["position"]]
        return cls(**known)


@dataclass
class Scene:
    points: np.ndarray  # (N,3) float
    colors: np.ndarray  # (N,3) uint8
    cameras: list[Camera] = field(default_factory=list)
    annotations: list[Annotation] = field(default_factory=list)
    units: str = "arbitrary"  # "meters" once the scale is known
    up: np.ndarray | None = None
    info: dict = field(default_factory=dict)  # provenance: method, input, settings

    # ------------------------------------------------------------------ geometry
    def up_vector(self) -> np.ndarray:
        if self.up is not None:
            u = np.asarray(self.up, float)
            return u / np.linalg.norm(u)
        if self.cameras:
            u = np.mean([c.up for c in self.cameras], axis=0)
            if np.linalg.norm(u) > 1e-6:
                return u / np.linalg.norm(u)
        return np.array([0.0, 0.0, 1.0])

    def rescale(self, factor: float, units: str | None = None) -> None:
        """Scale the whole world (points, cameras, annotations) by ``factor``."""
        self.points = self.points * factor
        for c in self.cameras:
            c.t = c.t * factor
        for a in self.annotations:
            a.position = [p * factor for p in a.position]
            a.radius *= factor
        if units:
            self.units = units

    # --------------------------------------------------------------- metadata
    def find_annotation(self, key: str) -> Annotation:
        for a in self.annotations:
            if a.id == key or a.label == key:
                return a
        raise KeyError(f"no annotation with id or label {key!r}")

    def add_annotation(self, ann: Annotation) -> Annotation:
        self.annotations.append(ann)
        return ann

    def remove_annotation(self, key: str) -> Annotation:
        a = self.find_annotation(key)
        self.annotations.remove(a)
        return a

    def point_labels(self) -> np.ndarray:
        """Per-point index into ``annotations`` (-1 = untagged).

        When anchors overlap, the closest anchor centre wins.
        """
        labels = np.full(len(self.points), -1, dtype=np.int32)
        best = np.full(len(self.points), np.inf)
        for i, a in enumerate(self.annotations):
            d = np.linalg.norm(self.points - np.asarray(a.position), axis=1)
            hit = (d <= a.radius) & (d < best)
            labels[hit] = i
            best[hit] = d[hit]
        return labels

    def points_in(self, key: str) -> np.ndarray:
        a = self.find_annotation(key)
        d = np.linalg.norm(self.points - np.asarray(a.position), axis=1)
        return np.nonzero(d <= a.radius)[0]

    def locate_pixel(self, camera: str | int, u: float, v: float, tol_px: float = 6.0) -> np.ndarray:
        """Turn a click on one of the input frames into a 3D position.

        Uses the reconstructed points that project near (u, v) in that frame
        and takes the nearest surface (low-percentile depth) along the ray.
        """
        cam = self.camera(camera)
        uv, z = cam.project(self.points)
        ok = z > 0
        dist = np.linalg.norm(uv - np.array([u, v]), axis=1)
        near = ok & (dist < tol_px)
        grow = tol_px
        while near.sum() < 3 and grow < max(cam.width, cam.height) / 8:
            grow *= 2
            near = ok & (dist < grow)
        if not near.any():
            raise ValueError(f"no reconstructed points near pixel ({u}, {v}) in {cam.name}")
        depth = float(np.percentile(z[near], 20))
        o, d = cam.ray(u, v)
        # convert z-depth to distance along the unit ray
        return o + d * depth / float(d @ cam.forward)

    def camera(self, key: str | int) -> Camera:
        if isinstance(key, int) or (isinstance(key, str) and key.isdigit()):
            return self.cameras[int(key)]
        for c in self.cameras:
            if c.name == key or Path(c.name).stem == key:
                return c
        raise KeyError(f"no camera named {key!r}")

    # -------------------------------------------------------------------- I/O
    def save(self, out_dir: str | Path) -> Path:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out / "points.npz", xyz=self.points.astype(np.float32), rgb=self.colors.astype(np.uint8))
        meta = {
            "format": "scan3d/1",
            "units": self.units,
            "up": None if self.up is None else np.asarray(self.up, float).tolist(),
            "info": self.info,
            "num_points": int(len(self.points)),
            "cameras": [c.to_dict() for c in self.cameras],
            "annotations": [a.to_dict() for a in self.annotations],
        }
        (out / "scene.json").write_text(json.dumps(meta, indent=2))
        return out

    @classmethod
    def load(cls, out_dir: str | Path) -> "Scene":
        out = Path(out_dir)
        meta = json.loads((out / "scene.json").read_text())
        pts = np.load(out / "points.npz")
        return cls(
            points=pts["xyz"].astype(np.float64),
            colors=pts["rgb"],
            cameras=[Camera.from_dict(c) for c in meta.get("cameras", [])],
            annotations=[Annotation.from_dict(a) for a in meta.get("annotations", [])],
            units=meta.get("units", "arbitrary"),
            up=None if meta.get("up") is None else np.asarray(meta["up"], float),
            info=meta.get("info", {}),
        )
