import numpy as np

from scan3d.depth import fuse
from scan3d.floorplan import context_sheet, estimate_up, floor_plan, tag_letter
from scan3d.scene import Annotation


def test_plan_is_aligned_with_walls_and_scaled(rgbd_frames):
    scene = fuse(rgbd_frames, voxel=0.04)
    scene.up = np.array([0, 0, 1.0])
    img, pf = floor_plan(scene)
    # e1 should be parallel to the x or y axis of the room
    ang = np.degrees(np.arccos(np.max(np.abs(pf.e1[:2]))))
    assert ang < 2.0
    # world -> pixel -> world roundtrip on the floor
    p = np.array([0.3, -0.7, 0.0])
    np.testing.assert_allclose(pf.to_world(pf.to_px(p), height=0.0)[0], p, atol=1e-9)
    assert img.shape[0] > 100 and img.shape[1] > 100


def test_estimate_up_finds_floor(rgbd_frames):
    scene = fuse(rgbd_frames, voxel=0.04)
    up = estimate_up(scene)
    assert np.degrees(np.arccos(up @ [0, 0, 1])) < 2.0


def test_context_sheet_lists_tags(rgbd_frames, tmp_path):
    scene = fuse(rgbd_frames, voxel=0.05)
    scene.add_annotation(Annotation("table", [0, 0, 0.75], 0.5, {"material": "oak"}))
    sheet, prompt = context_sheet(scene, tmp_path)  # no frames on disk: grey placeholders
    assert sheet.ndim == 3
    assert "A. table (material=oak)" in prompt
    assert "metres" in prompt


def test_tag_letters():
    assert [tag_letter(i) for i in (0, 25, 26, 27)] == ["A", "Z", "AA", "AB"]
