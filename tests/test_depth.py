import json

import numpy as np

from scan3d.depth import GL_TO_CV, fuse, load_rgbd_capture, write_rgbd_capture

from .conftest import distance_to_room


def test_fusion_is_metric_and_on_surfaces(rgbd_frames, room):
    scene = fuse(rgbd_frames, voxel=0.03)
    assert scene.units == "meters"
    d = distance_to_room(room, scene.points)
    assert np.percentile(d, 99) < 0.03
    assert len(scene.cameras) == len(rgbd_frames)


def test_capture_roundtrip(tmp_path, rgbd_frames):
    write_rgbd_capture(tmp_path, rgbd_frames[:3])
    back = load_rgbd_capture(tmp_path)
    assert len(back) == 3
    np.testing.assert_allclose(back[1].cam_to_world, rgbd_frames[1].cam_to_world, atol=1e-9)
    np.testing.assert_allclose(back[1].depth, rgbd_frames[1].depth, atol=1e-3)


def test_opengl_poses_and_low_res_depth(tmp_path, rgbd_frames, room):
    """ARKit-style capture: OpenGL camera axes and depth at half resolution."""
    write_rgbd_capture(tmp_path, rgbd_frames[:4])
    meta = json.loads((tmp_path / "capture.json").read_text())
    for f in meta["frames"]:
        f["pose"] = (np.asarray(f["pose"]) @ np.linalg.inv(GL_TO_CV)).tolist()
    meta["pose_convention"] = "opengl"
    import cv2

    for f in meta["frames"]:
        p = tmp_path / f["depth"]
        d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        cv2.imwrite(str(p), cv2.resize(d, (d.shape[1] // 2, d.shape[0] // 2), interpolation=cv2.INTER_NEAREST))
    (tmp_path / "capture.json").write_text(json.dumps(meta))
    frames = load_rgbd_capture(tmp_path)
    assert frames[0].depth.shape == (180, 240)
    scene = fuse(frames, voxel=0.03, stride=1)
    assert np.percentile(distance_to_room(room, scene.points), 95) < 0.04
