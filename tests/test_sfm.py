import numpy as np
import pytest

from scan3d.sfm import SfMOptions, build_tracks, reconstruct

from .conftest import distance_to_room, umeyama


@pytest.fixture(scope="module")
def sfm_scene(frames):
    return reconstruct(frames, SfMOptions(focal=375.0))


def test_registers_every_frame(sfm_scene, frames):
    assert len(sfm_scene.cameras) == len(frames)
    assert len(sfm_scene.points) > 1000


def test_camera_path_matches_ground_truth(sfm_scene, rendered):
    cams, _ = rendered
    gt = {c.name: c.center for c in cams}
    est = np.array([c.center for c in sfm_scene.cameras])
    ref = np.array([gt[c.name] for c in sfm_scene.cameras])
    s, R, t = umeyama(est, ref)
    err = np.linalg.norm(s * est @ R.T + t - ref, axis=1)
    assert err.max() < 0.02  # 2 cm on a ~2 m path


def test_points_lie_on_surfaces(sfm_scene, rendered, room):
    cams, _ = rendered
    gt = {c.name: c.center for c in cams}
    est = np.array([c.center for c in sfm_scene.cameras])
    s, R, t = umeyama(est, np.array([gt[c.name] for c in sfm_scene.cameras]))
    X = s * sfm_scene.points @ R.T + t
    d = distance_to_room(room, X)
    assert np.median(d) < 0.01
    assert np.mean(d < 0.05) > 0.95


def test_build_tracks_drops_inconsistent_chains():
    class F:
        def __init__(self, n):
            self.xy = np.zeros((n, 2))

    feats = [F(3), F(3), F(3)]
    # 0:0-1:0, 1:0-2:0 -> one 3-view track; 0:1-1:1 and 0:1-1:2 conflict in image 1
    matches = {(0, 1): np.array([[0, 0], [1, 1], [1, 2]]), (1, 2): np.array([[0, 0]])}
    tracks, kp_track = build_tracks(3, feats, matches)
    assert [(0, 0), (1, 0), (2, 0)] in tracks
    assert all(len({i for i, _ in t}) == len(t) for t in tracks)
    assert kp_track[0][1] == -1
