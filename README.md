# scan3d: a 3D world from video, with metadata embedded

Point a phone at a room, walk around, and get:

- a **3D point cloud** of the space, with the camera path,
- **tags** attached to places in it: labels, notes, sensor readings, any JSON,
  stored *inside* the exported files,
- a **top-down floor plan**,
- a **context sheet**: one image (map plus numbered photos) and a prompt that you
  can give to any vision-language model to ask questions about the space,
- a **browser viewer** where you can orbit the scan, read tags, and add new ones.

It runs on a laptop CPU with no GPU or learned models. The only dependencies are
`numpy`, `opencv` and `scipy`.

## How it fits together

| Layer | What it does | In this repo | Where the field is going |
|---|---|---|---|
| **Capture** | photos or video, optionally with depth, position and other sensors | `capture.py`, `depth.py`, `sensors.py`, `examples/sensor_logger/` | depth sensor + tracking (the "electronics" extension) |
| **Reconstruct** | turn images into a 3D world | `sfm.py` (photos → 3D), `depth.py` (RGB-D → metric 3D) | learned whole-house / city-block reconstruction |
| **Understand / tag** | attach meaning to places, and let an AI read the space | `scene.py` (tags), `floorplan.py` (plan + context sheet), `viewer_template.html` | "one canvas" panoramas for VLM spatial Q&A |

There are two ways in:

1. **Ordinary video or photos** → `scan3d reconstruct`. This uses classic
   Structure-from-Motion: SIFT features, essential-matrix verification, multi-view
   tracks, PnP registration and bundle adjustment. It works with any camera,
   but **the scale is unknown** until you give it one real measurement
   (`scan3d scale`).
2. **Colour + depth + pose** (a LiDAR phone, a RealSense on a tracked rig, or a
   learned depth/pose model run on video) → `scan3d fuse`. Every pixel is lifted
   straight into the world in **real metres**. This is the path to use if you
   add hardware.

## Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"

scan3d demo -o demo_out          # renders a synthetic room and runs everything (~1 min)
```

The demo prints how close the results are to ground truth. On the synthetic room,
cameras land within about 1 cm and one tape measurement fixes the scale to within about 1%. Open
`demo_out/sfm/viewer.html` and `demo_out/rgbd/viewer.html` in a browser. The
floor plans and context sheets are saved next to them.

### Your own video

```bash
scan3d reconstruct room.mp4 -o my_room            # or a folder of photos
scan3d tag my_room --label thermostat --frame frame_000120.jpg --pixel 640,310 \
       --prop model=T6 --prop 'schedule={"weekday": 21, "night": 17}'
scan3d tag my_room --label door --frame 0 --pixel 100,400
scan3d scale my_room --between thermostat door --distance 2.35   # one tape measurement → metres
scan3d tags my_room
```

To tag something, click it in one of the frames (`my_room/frames/`). The
`--pixel` option turns that click into a 3D point on the reconstructed surface.

Tips for a capture that reconstructs well:

- Move sideways rather than only turning on the spot. Rotation alone gives no depth.
- Keep a lot of overlap between frames and avoid motion blur. The frame extractor
  already picks the sharpest frame in each time slot.
- Plain white walls have nothing to track. Textured surfaces and furniture help.
- If frames fail to register, pass `--focal` (in pixels, after resizing to `--max-size`).
  The default assumes a typical phone field of view of about 65°.

### Depth + pose captures (the hardware path)

Convert your device's export into a folder with `capture.json`. The format is
documented in [`scan3d/depth.py`](scan3d/depth.py): intrinsics, a depth scale,
and one 4×4 camera-to-world pose per frame. It accepts both OpenCV and OpenGL/ARKit axis
conventions, and depth maps at a lower resolution than the colour images. Then:

```bash
scan3d fuse my_capture/ -o my_room_metric --voxel 0.02
```

### Sensor readings → tags

You can log any sensor with timestamps while you film. The readings are then
placed where the camera was at that moment:

```bash
scan3d sensor-log my_room sensors.csv --offset -12.5
```

`examples/sensor_logger/sensor_logger.ino` is a short Arduino sketch for a
temperature and light sensor. `examples/serial_to_csv.py` records its output.

### Adding tags in the browser

In `viewer.html`, **Shift+click** (or long-press on a phone) a point, type a label
and `key = value` properties, then click **Download tags.json**. Embed the tags
with:

```bash
scan3d import-tags my_room tags.json
```

## What gets written

```
my_room/
  scene.json           cameras (intrinsics, pose, timestamp), tags, units, up vector
  points.npz           xyz + rgb (the source of truth for geometry)
  points.ply           binary PLY; per-vertex `tag` index + tags/units/up as JSON header comments
  viewer.html          self-contained viewer (loads three.js from a CDN)
  floorplan.png/.json  top-down plan, with the pixel↔world mapping
  context_sheet.png    plan + numbered keyframes for a vision-language model
  context_prompt.txt   matching prompt text
  frames/              the frames the cameras refer to
```

Every command rewrites the derived files, so the PLY, viewer and plan always match
`scene.json`. `points.ply` opens in MeshLab, CloudCompare, Blender or Open3D. The tags
travel with the geometry: each point knows which tag it belongs to, and the full tag
list is in the header.

**Tags** are spheres (`position` + `radius`) with free-form `properties`. A tag
can hold anything you want to attach to a place: a detector label, a
maintenance note, a price, a link, or a sensor time series.

## Asking an AI about the space

`context_sheet.png` puts the whole space in one picture: the floor plan with
numbered camera positions and facing arrows, lettered tags, and the matching
photos. Give it to a vision-language model together with `context_prompt.txt`
and replace the question at the end, for example: *"Standing where photo 3 was
taken, which way do I turn to reach the table?"* This mirrors the idea in the
recent "one canvas" research, but it is a simple hand-built version, not a
re-implementation of any paper.

## Tests

```bash
pytest -q
```

The tests render a synthetic textured room with exact ground truth
(`scan3d/synthetic.py`) and check the following:

- SfM camera positions are within 2 cm after alignment.
- Reconstructed points lie on the true surfaces.
- Fused RGB-D points are metric, including with ARKit-style OpenGL poses and low-resolution depth.
- Pixel → 3D tagging hits the right surface.
- Tags survive the PLY and viewer round trips.
- Sensor readings are placed and merged correctly.
- The floor plan aligns with the walls.
- The CLI works end to end, including from an `.mp4`.

## Limits, and where to take it next

- **Sparse geometry from photos.** SfM gives thousands of points, not a dense surface.
  For dense results, feed `scan3d fuse` with depth from a sensor or from a
  monocular depth/pose network. Anything that outputs per-frame depth and poses
  plugs in through `capture.json`.
- **Rooms, not buildings.** Incremental SfM drifts on long walks, and nothing here
  solves several connected rooms jointly yet. Recent work (e.g. whole-floor-plan
  reconstruction from photos) does this, and it would replace the Reconstruct layer.
- **Tags are placed by you or by sensors.** Running an object detector on the
  frames and tagging through `Scene.locate_pixel` is a small, natural next step.
