"""``scan3d demo``: the whole pipeline on a synthetic room, no camera needed.

Renders a textured room (walls, a table, a cabinet) along a hand-held style
path, then runs both input paths on it:

* video  -> Structure-from-Motion -> tag by clicking a pixel -> fix scale
* RGB-D  -> depth fusion (metric) -> tags + a fake sensor log

and reports how close the results are to the known ground truth.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from . import capture, export, synthetic
from .depth import RGBDFrame, fuse, write_rgbd_capture
from .floorplan import estimate_up
from .scene import Annotation
from .sensors import attach_sensor_log
from .sfm import SfMOptions, reconstruct

# ground-truth landmarks in the synthetic room (metres, z up)
TABLE_TOP = np.array([0.0, 0.0, 0.75])
CABINET_FACE = np.array([-1.9, 0.9, 1.0])  # the cabinet's east face, facing the room


def _visible_pixel(quads, cams, renders, X):
    """Frame index + pixel where world point X is visible nearest the image centre."""
    best = None
    for i, (cam, (_, depth, _)) in enumerate(zip(cams, renders)):
        uv, z = cam.project(X[None])
        u, v = uv[0]
        if z[0] <= 0 or not (0 <= u < cam.width and 0 <= v < cam.height):
            continue
        if abs(depth[int(v), int(u)] - z[0]) > 0.02:  # occluded
            continue
        score = np.hypot(u - cam.width / 2, v - cam.height / 2)
        if best is None or score < best[0]:
            best = (score, i, (float(u), float(v)))
    if best is None:
        raise RuntimeError("landmark not visible in any frame")
    return best[1], best[2]


def _align(est: np.ndarray, gt: np.ndarray):
    """Similarity transform (s, R, t) with gt ~ s R est + t (Umeyama)."""
    me, mg = est.mean(0), gt.mean(0)
    X, Y = est - me, gt - mg
    U, D, Vt = np.linalg.svd(Y.T @ X / len(est))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / X.var(0).sum()
    return s, R, mg - s * R @ me


def _write_sensor_log(path: Path, cams) -> None:
    """A pretend temperature + Wi-Fi logger carried along the path."""
    rng = np.random.default_rng(7)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "key", "value", "unit"])
        for cam in cams[:: max(1, len(cams) // 4)]:
            x, y, _ = cam.center
            w.writerow([f"{cam.timestamp:.2f}", "temperature", f"{21 + 0.8 * x + rng.normal(0, 0.1):.1f}", "C"])
            w.writerow([f"{cam.timestamp:.2f}", "wifi_rssi", f"{int(-55 - 6 * abs(y) + rng.normal(0, 1))}", "dBm"])


def run_demo(out: Path, n_frames: int = 24, mode: str = "both") -> None:
    out.mkdir(parents=True, exist_ok=True)
    quads = synthetic.box_room()
    cams = synthetic.orbit_cameras(n_frames)
    print(f"rendering {n_frames} frames of a synthetic 5 x 4 m room ...")
    renders = [synthetic.render(quads, c) for c in cams]
    inp = out / "input"
    inp.mkdir(exist_ok=True)
    synthetic.write_video([r[0] for r in renders], str(inp / "room.mp4"), fps=10)
    write_rgbd_capture(inp / "rgbd", [
        RGBDFrame(c.name, r[0], r[1].astype(np.float32), c.K, np.linalg.inv(np.r_[np.c_[c.R, c.t], [[0, 0, 0, 1]]]), c.timestamp)
        for c, r in zip(cams, renders)
    ])
    _write_sensor_log(inp / "sensors.csv", cams)
    print(f"  wrote {inp / 'room.mp4'}, {inp / 'rgbd'}/, {inp / 'sensors.csv'}")

    if mode in ("both", "video"):
        print("\n[1] video -> Structure-from-Motion")
        frames = capture.load_input(inp / "room.mp4", max_frames=n_frames)
        sfm_dir = out / "sfm"
        capture.save_frames(frames, sfm_dir / "frames")
        scene = reconstruct(frames, SfMOptions(focal=float(cams[0].K[0, 0])))
        scene.up = estimate_up(scene)
        # frame k of the video is rendered camera k
        gt_by_name = {f.name: cams[i] for i, f in enumerate(frames)}
        names = [c.name for c in scene.cameras]
        print(f"  registered {len(names)}/{len(frames)} frames, {len(scene.points):,} points")

        # tag two objects by "clicking" on them in a frame
        for label, X, props in (("table", TABLE_TOP, {"material": "oak", "seats": 4}),
                                ("cabinet", CABINET_FACE, {"contents": "router, spare cables"})):
            i, (u, v) = _visible_pixel(quads, cams, renders, X)
            frame_name = frames[i].name
            if frame_name not in names:
                continue
            p = scene.locate_pixel(frame_name, u, v)
            scene.add_annotation(Annotation(label, p.tolist(), 0.0, dict(props, picked_in={"frame": frame_name, "pixel": [round(u, 1), round(v, 1)]}), "pixel"))
        # give it real scale from one known distance, like measuring with a tape
        if len(scene.annotations) == 2:
            a, b = (np.asarray(x.position) for x in scene.annotations)
            f = float(np.linalg.norm(TABLE_TOP - CABINET_FACE) / np.linalg.norm(a - b))
            scene.rescale(f, "meters")
            print(f"  scaled with the table->cabinet distance ({np.linalg.norm(TABLE_TOP - CABINET_FACE):.2f} m)")
        for x in scene.annotations:
            x.radius = 0.4 if scene.units == "meters" else 0.1 * np.ptp(scene.points, axis=0).max()
        # accuracy vs ground truth
        est = np.array([c.center for c in scene.cameras])
        gt = np.array([gt_by_name[n].center for n in names])
        s, R, t = _align(est, gt)
        err = np.linalg.norm((s * est @ R.T + t) - gt, axis=1)
        print(f"  camera position error after alignment: mean {100 * err.mean():.1f} cm, max {100 * err.max():.1f} cm")
        print(f"  scale from one tape measurement is off by {100 * abs(s - 1):.1f}%")
        attach_sensor_log(scene, inp / "sensors.csv", merge_radius=0.3)
        export.export_all(scene, sfm_dir)
        print(f"  -> {sfm_dir / 'viewer.html'}")

    if mode in ("both", "rgbd"):
        print("\n[2] RGB-D capture -> depth fusion (metric)")
        from .depth import load_rgbd_capture

        frames_d = load_rgbd_capture(inp / "rgbd")
        rgbd_dir = out / "rgbd"
        capture.save_frames([capture.Frame(f.name, f.rgb) for f in frames_d], rgbd_dir / "frames")
        scene = fuse(frames_d, voxel=0.02)
        scene.add_annotation(Annotation("table", TABLE_TOP.tolist(), 0.6, {"material": "oak", "seats": 4}))
        scene.add_annotation(Annotation("cabinet", CABINET_FACE.tolist(), 0.5, {"contents": "router, spare cables"}))
        attach_sensor_log(scene, inp / "sensors.csv", merge_radius=0.3)
        print(f"  {len(scene.points):,} points at 2 cm spacing, {len(scene.annotations)} tags")
        export.export_all(scene, rgbd_dir)
        print(f"  -> {rgbd_dir / 'viewer.html'}")

    print("\nopen the viewer.html files in a browser; floor plans and VLM context sheets are next to them.")

