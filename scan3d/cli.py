"""Command line: ``scan3d <command> ...`` (run ``scan3d -h`` for the list)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

from . import capture, export
from .scene import Annotation, Scene


def _floats(s: str, n: int) -> list[float]:
    v = [float(x) for x in s.replace(" ", "").split(",")]
    if len(v) != n:
        raise argparse.ArgumentTypeError(f"expected {n} comma-separated numbers, got {s!r}")
    return v


def _props(items: list[str] | None) -> dict:
    out = {}
    for it in items or []:
        if "=" not in it:
            raise SystemExit(f"--prop expects key=value, got {it!r}")
        k, v = it.split("=", 1)
        try:
            out[k.strip()] = json.loads(v)
        except json.JSONDecodeError:
            out[k.strip()] = v
    return out


def _finish(scene: Scene, out: Path, msg: str) -> None:
    export.export_all(scene, out)
    print(msg)
    print(f"  scene:      {out / 'scene.json'}")
    print(f"  points:     {out / 'points.ply'}  ({len(scene.points):,} points, {len(scene.annotations)} tags)")
    print(f"  viewer:     {out / 'viewer.html'}")
    print(f"  floor plan: {out / 'floorplan.png'}")
    print(f"  VLM sheet:  {out / 'context_sheet.png'} + context_prompt.txt")


# ------------------------------------------------------------------- commands
def cmd_reconstruct(a) -> None:
    from .floorplan import estimate_up
    from .sfm import SfMOptions, reconstruct

    frames = capture.load_input(a.input, a.max_frames, a.max_size)
    print(f"loaded {len(frames)} frames from {a.input}")
    out = Path(a.output)
    capture.save_frames(frames, out / "frames")
    scene = reconstruct(frames, SfMOptions(focal=a.focal, max_features=a.features))
    scene.info["input"] = str(a.input)
    scene.up = estimate_up(scene)
    _finish(scene, out, f"registered {len(scene.cameras)}/{len(frames)} frames")
    if len(scene.cameras) < len(frames):
        print("  note: unregistered frames lacked overlap/texture; try more overlap or --focal")


def cmd_fuse(a) -> None:
    from .depth import fuse, load_rgbd_capture

    frames = load_rgbd_capture(a.input)
    out = Path(a.output)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    capture.save_frames([capture.Frame(f.name, f.rgb) for f in frames], out / "frames")
    scene = fuse(frames, voxel=a.voxel, max_depth=a.max_depth, stride=a.stride)
    scene.info["input"] = str(a.input)
    if a.up:
        scene.up = np.asarray(_floats(a.up, 3))
    _finish(scene, out, f"fused {len(frames)} RGB-D frames")


def cmd_tag(a) -> None:
    scene = Scene.load(a.scene)
    if a.at:
        pos, source = _floats(a.at, 3), "manual"
    elif a.frame is not None and a.pixel:
        u, v = _floats(a.pixel, 2)
        pos, source = scene.locate_pixel(a.frame, u, v).tolist(), "pixel"
    else:
        raise SystemExit("give either --at x,y,z or --frame NAME --pixel u,v")
    props = _props(a.prop)
    if a.frame is not None and a.pixel:
        props.setdefault("picked_in", {"frame": scene.camera(a.frame).name, "pixel": _floats(a.pixel, 2)})
    if a.radius is None:
        ext = np.ptp(scene.points, axis=0).max() if len(scene.points) else 1.0
        a.radius = round(float(0.04 * ext), 4)
    ann = scene.add_annotation(Annotation(a.label, pos, a.radius, props, a.source or source))
    _finish(scene, Path(a.scene), f"added tag {ann.id} '{ann.label}' at {np.round(pos, 3).tolist()} "
                                  f"({len(scene.points_in(ann.id))} points inside)")


def cmd_untag(a) -> None:
    scene = Scene.load(a.scene)
    ann = scene.remove_annotation(a.key)
    _finish(scene, Path(a.scene), f"removed tag {ann.id} '{ann.label}'")


def cmd_tags(a) -> None:
    scene = Scene.load(a.scene)
    if a.json:
        print(json.dumps([x.to_dict() for x in scene.annotations], indent=2))
        return
    if not scene.annotations:
        print("no tags")
    for x in scene.annotations:
        n = len(scene.points_in(x.id))
        props = ", ".join(f"{k}={v}" for k, v in x.properties.items() if not isinstance(v, dict))
        print(f"{x.id}  {x.label:<28} at {np.round(x.position, 3).tolist()}  r={x.radius:g}  {n} pts  [{x.source}]"
              + (f"  {props}" if props else ""))


def cmd_import_tags(a) -> None:
    scene = Scene.load(a.scene)
    added = export.import_viewer_tags(scene, a.file)
    _finish(scene, Path(a.scene), f"imported {len(added)} tags")


def cmd_sensor_log(a) -> None:
    from .sensors import attach_sensor_log

    scene = Scene.load(a.scene)
    added = attach_sensor_log(scene, a.csv, offset=a.offset, merge_radius=a.merge_radius)
    _finish(scene, Path(a.scene), f"placed {len(added)} sensor tags")


def cmd_scale(a) -> None:
    scene = Scene.load(a.scene)
    if a.factor:
        f = a.factor
    elif a.between and a.distance:
        p, q = (np.asarray(scene.find_annotation(k).position) for k in a.between)
        f = a.distance / float(np.linalg.norm(p - q))
    else:
        raise SystemExit("give --factor F, or --between TAG_A TAG_B --distance D")
    scene.rescale(f, a.units)
    _finish(scene, Path(a.scene), f"scaled scene by {f:.6g}; units are now '{scene.units}'")


def cmd_export(a) -> None:
    scene = Scene.load(a.scene)
    _finish(scene, Path(a.scene), "re-exported")


def cmd_demo(a) -> None:
    from .demo import run_demo

    run_demo(Path(a.output), n_frames=a.frames, mode=a.mode)


# ----------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="scan3d", description=__doc__)
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("reconstruct", help="video or photo folder -> 3D scene (Structure-from-Motion)")
    r.add_argument("input", help="video file or folder of images")
    r.add_argument("-o", "--output", required=True, help="scene directory to create")
    r.add_argument("--focal", type=float, help="focal length in pixels (after resizing); default ~65 deg FOV guess")
    r.add_argument("--max-frames", type=int, default=60)
    r.add_argument("--max-size", type=int, default=1280, help="resize frames so the long side is at most this")
    r.add_argument("--features", type=int, default=6000, help="SIFT features per frame")
    r.set_defaults(fn=cmd_reconstruct)

    f = sub.add_parser("fuse", help="RGB-D capture (colour + depth + poses) -> metric 3D scene")
    f.add_argument("input", help="folder containing capture.json")
    f.add_argument("-o", "--output", required=True)
    f.add_argument("--voxel", type=float, default=0.02, help="point spacing in metres")
    f.add_argument("--max-depth", type=float, default=8.0)
    f.add_argument("--stride", type=int, default=2, help="use every n-th depth pixel")
    f.add_argument("--up", help="world up vector x,y,z (default: from camera orientation)")
    f.set_defaults(fn=cmd_fuse)

    t = sub.add_parser("tag", help="attach metadata to a place in the scene")
    t.add_argument("scene")
    t.add_argument("--label", required=True)
    t.add_argument("--at", help="world position x,y,z")
    t.add_argument("--frame", help="frame name or index to click in (with --pixel)")
    t.add_argument("--pixel", help="u,v pixel in that frame; resolved to the 3D surface point")
    t.add_argument("--radius", type=float, help="tag sphere radius (default 4%% of scene size)")
    t.add_argument("--prop", action="append", metavar="KEY=VALUE", help="metadata; repeatable; VALUE may be JSON")
    t.add_argument("--source", help="where this tag came from (e.g. a detector name)")
    t.set_defaults(fn=cmd_tag)

    u = sub.add_parser("untag", help="remove a tag by id or label")
    u.add_argument("scene")
    u.add_argument("key")
    u.set_defaults(fn=cmd_untag)

    ls = sub.add_parser("tags", help="list tags")
    ls.add_argument("scene")
    ls.add_argument("--json", action="store_true")
    ls.set_defaults(fn=cmd_tags)

    it = sub.add_parser("import-tags", help="merge tags.json downloaded from the viewer")
    it.add_argument("scene")
    it.add_argument("file")
    it.set_defaults(fn=cmd_import_tags)

    sl = sub.add_parser("sensor-log", help="place timestamped sensor readings (CSV) where they were taken")
    sl.add_argument("scene")
    sl.add_argument("csv")
    sl.add_argument("--offset", type=float, default=0.0, help="seconds to add to log timestamps")
    sl.add_argument("--merge-radius", type=float, help="merge same-sensor readings closer than this")
    sl.set_defaults(fn=cmd_sensor_log)

    s = sub.add_parser("scale", help="give the scene real-world scale")
    s.add_argument("scene")
    s.add_argument("--between", nargs=2, metavar=("TAG_A", "TAG_B"))
    s.add_argument("--distance", type=float, help="true distance between the two tags")
    s.add_argument("--factor", type=float)
    s.add_argument("--units", default="meters")
    s.set_defaults(fn=cmd_scale)

    e = sub.add_parser("export", help="regenerate PLY, viewer, floor plan and context sheet")
    e.add_argument("scene")
    e.set_defaults(fn=cmd_export)

    d = sub.add_parser("demo", help="render a synthetic room and run the whole pipeline on it")
    d.add_argument("-o", "--output", default="demo_out")
    d.add_argument("--frames", type=int, default=24)
    d.add_argument("--mode", choices=["both", "video", "rgbd"], default="both")
    d.set_defaults(fn=cmd_demo)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if a.verbose else logging.WARNING, format="%(message)s")
    try:
        a.fn(a)
    except (ValueError, KeyError, RuntimeError, FileNotFoundError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
