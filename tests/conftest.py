import numpy as np
import pytest

from scan3d import synthetic
from scan3d.capture import Frame
from scan3d.depth import RGBDFrame


def cam_to_world(cam):
    return np.linalg.inv(np.r_[np.c_[cam.R, cam.t], [[0, 0, 0, 1]]])


@pytest.fixture(scope="session")
def room():
    return synthetic.box_room()


@pytest.fixture(scope="session")
def rendered(room):
    """12 views of the synthetic room: (cameras, [(image, depth, quad_idx)])."""
    cams = synthetic.orbit_cameras(12, width=480, height=360, focal=375.0, arc_deg=150)
    return cams, [synthetic.render(room, c) for c in cams]


@pytest.fixture(scope="session")
def frames(rendered):
    cams, renders = rendered
    return [Frame(c.name, r[0], c.timestamp) for c, r in zip(cams, renders)]


@pytest.fixture(scope="session")
def rgbd_frames(rendered):
    cams, renders = rendered
    return [RGBDFrame(c.name, r[0], r[1].astype(np.float32), c.K, cam_to_world(c), c.timestamp)
            for c, r in zip(cams, renders)]


def distance_to_room(room, X):
    """Distance from each point to the nearest synthetic surface."""
    best = np.full(len(X), np.inf)
    for q in room:
        b, c = [k for k in range(3) if k != q.axis]
        pb = np.clip(X[:, b], q.lo[0], q.hi[0])
        pc = np.clip(X[:, c], q.lo[1], q.hi[1])
        d = np.sqrt((X[:, q.axis] - q.c) ** 2 + (X[:, b] - pb) ** 2 + (X[:, c] - pc) ** 2)
        best = np.minimum(best, d)
    return best


def umeyama(est, gt):
    me, mg = est.mean(0), gt.mean(0)
    X, Y = est - me, gt - mg
    U, D, Vt = np.linalg.svd(Y.T @ X / len(est))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / X.var(0).sum()
    return s, R, mg - s * R @ me
