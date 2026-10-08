"""Reconstruct layer: incremental Structure-from-Motion from ordinary photos.

Pipeline (classic, no learned models, runs on a CPU):

1. SIFT features in every frame
2. pairwise matching (ratio test + mutual check) and geometric verification
   with the essential matrix
3. matches are chained into multi-view *tracks*
4. pick the initial pair with many matches *and* enough parallax,
   recover its relative pose, triangulate
5. repeatedly register the frame that sees the most known 3D points
   (PnP + RANSAC), triangulate new tracks, run bundle adjustment
6. a final global bundle adjustment

Monocular reconstruction has no absolute scale; ``Scene.rescale`` (or the
``scan3d scale`` command) fixes that from one known distance.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix

from .capture import Frame
from .scene import Camera, Scene

log = logging.getLogger(__name__)


@dataclass
class SfMOptions:
    focal: float | None = None  # pixels; default guessed from a typical phone FOV
    max_features: int = 6000
    ratio: float = 0.8
    min_pair_inliers: int = 30
    exhaustive_below: int = 40  # match all pairs when there are fewer frames
    window: int = 10  # otherwise match each frame with the next `window`
    reproj_thresh: float = 4.0
    min_tri_angle_deg: float = 1.5
    ba_every: int = 4
    verbose: bool = False


@dataclass
class Features:
    xy: np.ndarray  # (N,2)
    desc: np.ndarray  # (N,128)
    rgb: np.ndarray  # (N,3) uint8


@dataclass
class _State:
    n_images: int
    kp_track: list[np.ndarray]
    track_obs: list[list[tuple[int, int]]]
    xyz: np.ndarray  # (T,3), nan when not triangulated
    poses: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)  # img -> (R, t)


def guess_K(width: int, height: int, focal: float | None = None) -> np.ndarray:
    # ~65 degree horizontal field of view, typical for phone main cameras
    f = focal or 0.78 * max(width, height)
    return np.array([[f, 0, width / 2.0], [0, f, height / 2.0], [0, 0, 1]], float)


# --------------------------------------------------------------------- features
def detect(frames: list[Frame], max_features: int) -> list[Features]:
    sift = cv2.SIFT_create(nfeatures=max_features)
    out = []
    for f in frames:
        gray = cv2.cvtColor(f.image, cv2.COLOR_BGR2GRAY)
        kps, desc = sift.detectAndCompute(gray, None)
        if desc is None:
            kps, desc = [], np.zeros((0, 128), np.float32)
        xy = np.array([k.pt for k in kps], float).reshape(-1, 2)
        ij = np.clip(np.round(xy).astype(int), 0, [f.image.shape[1] - 1, f.image.shape[0] - 1])
        rgb = f.image[ij[:, 1], ij[:, 0], ::-1].astype(np.uint8) if len(xy) else np.zeros((0, 3), np.uint8)
        out.append(Features(xy, desc.astype(np.float32), rgb))
    return out


def match_pair(a: Features, b: Features, ratio: float) -> np.ndarray:
    if len(a.desc) < 2 or len(b.desc) < 2:
        return np.zeros((0, 2), int)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    fwd = bf.knnMatch(a.desc, b.desc, k=2)
    bwd = bf.knnMatch(b.desc, a.desc, k=1)
    back = {m[0].queryIdx: m[0].trainIdx for m in bwd if m}
    pairs = [
        (m[0].queryIdx, m[0].trainIdx)
        for m in fwd
        if len(m) == 2 and m[0].distance < ratio * m[1].distance and back.get(m[0].trainIdx) == m[0].queryIdx
    ]
    return np.array(pairs, int).reshape(-1, 2)


def verify_pair(a: Features, b: Features, m: np.ndarray, K: np.ndarray, thresh: float) -> np.ndarray:
    if len(m) < 8:
        return m[:0]
    p1, p2 = a.xy[m[:, 0]], b.xy[m[:, 1]]
    E, mask = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.999, threshold=thresh / 2)
    if E is None or mask is None:
        return m[:0]
    return m[mask.ravel().astype(bool)]


def build_tracks(n_images: int, feats: list[Features], matches: dict[tuple[int, int], np.ndarray]):
    offsets = np.cumsum([0] + [len(f.xy) for f in feats])
    parent = np.arange(offsets[-1])

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for (i, j), m in matches.items():
        for a, b in m:
            ra, rb = find(offsets[i] + a), find(offsets[j] + b)
            if ra != rb:
                parent[rb] = ra

    groups: dict[int, list[tuple[int, int]]] = {}
    touched = set()
    for (i, j), m in matches.items():
        touched.update((offsets[i] + m[:, 0]).tolist())
        touched.update((offsets[j] + m[:, 1]).tolist())
    for node in touched:
        img = int(np.searchsorted(offsets, node, side="right") - 1)
        groups.setdefault(find(node), []).append((img, int(node - offsets[img])))

    track_obs = []
    for obs in groups.values():
        imgs = [o[0] for o in obs]
        if len(obs) >= 2 and len(set(imgs)) == len(imgs):  # drop inconsistent tracks
            track_obs.append(sorted(obs))
    kp_track = [np.full(len(f.xy), -1, int) for f in feats]
    for t, obs in enumerate(track_obs):
        for img, kp in obs:
            kp_track[img][kp] = t
    return track_obs, kp_track


# ------------------------------------------------------------------- geometry
def _triangulate(Ps: list[np.ndarray], xs: list[np.ndarray]) -> np.ndarray:
    A = []
    for P, (u, v) in zip(Ps, xs):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    _, _, Vt = np.linalg.svd(np.asarray(A))
    X = Vt[-1]
    return X[:3] / X[3]


def _rodrigues(rvecs: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(rvecs, axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(theta > 1e-12, rvecs / theta, 0.0)
    K = np.zeros((len(rvecs), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = np.sin(theta)[:, :, None], np.cos(theta)[:, :, None]
    return np.eye(3)[None] + s * K + (1 - c) * (K @ K)


def _left_jacobian(w: np.ndarray) -> np.ndarray:
    """SO(3) left Jacobian for each row of ``w`` (N,3)."""
    th = np.linalg.norm(w, axis=1)[:, None, None]
    W = np.zeros((len(w), 3, 3))
    W[:, 0, 1], W[:, 0, 2] = -w[:, 2], w[:, 1]
    W[:, 1, 0], W[:, 1, 2] = w[:, 2], -w[:, 0]
    W[:, 2, 0], W[:, 2, 1] = -w[:, 1], w[:, 0]
    small = th < 1e-8
    th = np.where(small, 1.0, th)
    a = np.where(small, 0.5, (1 - np.cos(th)) / th**2)
    b = np.where(small, 1.0 / 6, (th - np.sin(th)) / th**3)
    return np.eye(3)[None] + a * W + b * (W @ W)


class IncrementalSfM:
    def __init__(self, frames: list[Frame], options: SfMOptions | None = None):
        self.frames = frames
        self.opt = options or SfMOptions()
        h, w = frames[0].image.shape[:2]
        self.K = guess_K(w, h, self.opt.focal)
        self.feats: list[Features] = []
        self.matches: dict[tuple[int, int], np.ndarray] = {}

    # ---------------------------------------------------------------- stages
    def run(self) -> Scene:
        n = len(self.frames)
        log.info("detecting features in %d frames", n)
        self.feats = detect(self.frames, self.opt.max_features)
        if n <= self.opt.exhaustive_below:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        else:
            pairs = [(i, j) for i in range(n) for j in range(i + 1, min(n, i + 1 + self.opt.window))]
        log.info("matching %d image pairs", len(pairs))
        for i, j in pairs:
            m = match_pair(self.feats[i], self.feats[j], self.opt.ratio)
            m = verify_pair(self.feats[i], self.feats[j], m, self.K, self.opt.reproj_thresh)
            if len(m) >= self.opt.min_pair_inliers:
                self.matches[(i, j)] = m
        if not self.matches:
            raise RuntimeError("no image pairs could be matched; is there enough texture and overlap?")

        track_obs, kp_track = build_tracks(n, self.feats, self.matches)
        log.info("%d verified pairs, %d tracks", len(self.matches), len(track_obs))
        st = _State(n, kp_track, track_obs, np.full((len(track_obs), 3), np.nan))

        self._initialize(st)
        failed: set[int] = set()
        since_ba = 0
        while True:
            nxt = self._next_image(st, failed)
            if nxt is None:
                break
            if not self._register(st, nxt):
                failed.add(nxt)
                continue
            failed.clear()  # new points may make earlier failures succeed
            self._triangulate_new(st, nxt)
            since_ba += 1
            if since_ba >= self.opt.ba_every:
                self._bundle_adjust(st, max_nfev=30)
                since_ba = 0
            log.info("registered %d/%d frames, %d points", len(st.poses), n, int(np.isfinite(st.xyz[:, 0]).sum()))
        self._bundle_adjust(st, max_nfev=100)
        self._filter(st)
        return self._to_scene(st)

    def _initialize(self, st: _State) -> None:
        best, best_score = None, -1.0
        for (i, j), m in sorted(self.matches.items(), key=lambda kv: -len(kv[1]))[:30]:
            p1, p2 = self.feats[i].xy[m[:, 0]], self.feats[j].xy[m[:, 1]]
            E, _ = cv2.findEssentialMat(p1, p2, self.K, cv2.RANSAC, 0.999, self.opt.reproj_thresh / 2)
            if E is None or E.shape != (3, 3):
                continue
            ninl, R, t, mask = cv2.recoverPose(E, p1, p2, self.K)
            t = t.ravel()
            # parallax: angle between the viewing rays once the rotation is
            # removed, so pure camera rotations don't look like a good baseline
            Kinv = np.linalg.inv(self.K)
            r1 = (np.c_[p1, np.ones(len(p1))] @ Kinv.T) @ R.T
            r2 = np.c_[p2, np.ones(len(p2))] @ Kinv.T
            cos_par = np.sum(r1 * r2, 1) / np.linalg.norm(r1, axis=1) / np.linalg.norm(r2, axis=1)
            parallax = np.degrees(np.median(np.arccos(np.clip(cos_par, -1, 1))))
            score = ninl * min(parallax / 4.0, 1.0)
            log.debug("init candidate %s: %d inliers, parallax %.2f deg", (i, j), ninl, parallax)
            if score > best_score:
                best, best_score = (i, j, R, t), score
        if best is None:
            raise RuntimeError("could not find an initial image pair")
        i, j, R, t = best
        st.poses[i] = (np.eye(3), np.zeros(3))
        st.poses[j] = (R, t)
        self._triangulate_new(st, j)
        if np.isfinite(st.xyz[:, 0]).sum() < 20:
            raise RuntimeError("initial pair produced too few 3D points (not enough parallax?)")
        self._bundle_adjust(st, max_nfev=30)

    def _next_image(self, st: _State, failed: set[int]) -> int | None:
        known = np.isfinite(st.xyz[:, 0])
        best, best_n = None, 11
        for img in range(st.n_images):
            if img in st.poses or img in failed:
                continue
            tr = st.kp_track[img]
            nvis = int(np.sum(known[tr[tr >= 0]]))
            if nvis > best_n:
                best, best_n = img, nvis
        return best

    def _register(self, st: _State, img: int) -> bool:
        tr = st.kp_track[img]
        kps = np.nonzero(tr >= 0)[0]
        kps = kps[np.isfinite(st.xyz[tr[kps], 0])]
        obj = st.xyz[tr[kps]].astype(np.float64)
        pts = self.feats[img].xy[kps].astype(np.float64)
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj, pts, self.K, None, reprojectionError=self.opt.reproj_thresh, iterationsCount=2000, confidence=0.999
        )
        if not ok or inl is None or len(inl) < max(12, 0.15 * len(kps)):
            log.debug("PnP failed for %s", self.frames[img].name)
            return False
        inl = inl.ravel()
        rvec, tvec = cv2.solvePnPRefineLM(obj[inl], pts[inl], self.K, None, rvec, tvec)
        st.poses[img] = (cv2.Rodrigues(rvec)[0], tvec.ravel())
        return True

    def _triangulate_new(self, st: _State, img: int) -> None:
        tr = st.kp_track[img]
        min_cos = np.cos(np.radians(self.opt.min_tri_angle_deg))
        for t in np.unique(tr[tr >= 0]):
            if np.isfinite(st.xyz[t, 0]):
                continue
            obs = [(i, k) for i, k in st.track_obs[t] if i in st.poses]
            if len(obs) < 2:
                continue
            Ps = [self.K @ np.c_[st.poses[i][0], st.poses[i][1]] for i, _ in obs]
            xs = [self.feats[i].xy[k] for i, k in obs]
            X = _triangulate(Ps, xs)
            # cheirality, reprojection and parallax checks
            good, centers = True, []
            for (i, _), x in zip(obs, xs):
                R, tt = st.poses[i]
                Xc = R @ X + tt
                if Xc[2] <= 0:
                    good = False
                    break
                uv = (self.K @ Xc)[:2] / Xc[2]
                if np.linalg.norm(uv - x) > self.opt.reproj_thresh:
                    good = False
                    break
                centers.append(-R.T @ tt)
            if not good:
                continue
            rays = [(X - c) / np.linalg.norm(X - c) for c in centers]
            cos_min = min(float(a @ b) for ai, a in enumerate(rays) for b in rays[ai + 1 :])
            if cos_min > min_cos:
                continue
            st.xyz[t] = X

    def _observations(self, st: _State):
        cams = sorted(st.poses)
        cam_idx = {c: n for n, c in enumerate(cams)}
        pts = np.nonzero(np.isfinite(st.xyz[:, 0]))[0]
        pt_idx = {p: n for n, p in enumerate(pts)}
        ci, pi, uv = [], [], []
        for p in pts:
            for img, k in st.track_obs[p]:
                if img in cam_idx:
                    ci.append(cam_idx[img])
                    pi.append(pt_idx[p])
                    uv.append(self.feats[img].xy[k])
        return cams, pts, np.array(ci, int), np.array(pi, int), np.array(uv, float).reshape(-1, 2)

    def _bundle_adjust(self, st: _State, max_nfev: int = 50) -> None:
        """Jointly refine all poses and points (analytic sparse Jacobian).

        Rotations are parameterised as a small update ``w`` on top of the
        current rotation (R = exp(w) R0), so every call starts at w = 0.
        """
        cams, pts, ci, pi, uv = self._observations(st)
        if len(pts) < 10 or len(cams) < 2:
            return
        nc, npt, nobs = len(cams), len(pts), len(ci)
        R0 = np.stack([st.poses[c][0] for c in cams])
        x0 = np.concatenate(
            [np.concatenate([np.zeros(3), st.poses[c][1]]) for c in cams] + [st.xyz[pts].ravel()]
        )
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]

        def unpack(x):
            cp = x[: nc * 6].reshape(nc, 6)
            R = _rodrigues(cp[:, :3]) @ R0
            X = x[nc * 6 :].reshape(npt, 3)
            Y = np.einsum("nij,nj->ni", R[ci], X[pi])  # rotated point, before translation
            return cp, R, Y, Y + cp[ci, 3:]

        def residuals(x):
            _, _, _, Xc = unpack(x)
            uvp = np.c_[fx * Xc[:, 0] / Xc[:, 2] + cx, fy * Xc[:, 1] / Xc[:, 2] + cy]
            return (uvp - uv).ravel()

        rows = np.repeat(np.arange(2 * nobs), 9)
        cols_obs = np.concatenate(
            [ci[:, None] * 6 + np.arange(6), nc * 6 + pi[:, None] * 3 + np.arange(3)], axis=1
        )  # (nobs, 9)
        cols = np.repeat(cols_obs, 2, axis=0).ravel()

        def jacobian(x):
            cp, R, Y, Xc = unpack(x)
            z = Xc[:, 2]
            Jp = np.zeros((nobs, 2, 3))
            Jp[:, 0, 0] = fx / z
            Jp[:, 0, 2] = -fx * Xc[:, 0] / z**2
            Jp[:, 1, 1] = fy / z
            Jp[:, 1, 2] = -fy * Xc[:, 1] / z**2
            # d(exp(w) Y0)/dw = -[Y]_x J_l(w)
            Yx = np.zeros((nobs, 3, 3))
            Yx[:, 0, 1], Yx[:, 0, 2] = -Y[:, 2], Y[:, 1]
            Yx[:, 1, 0], Yx[:, 1, 2] = Y[:, 2], -Y[:, 0]
            Yx[:, 2, 0], Yx[:, 2, 1] = -Y[:, 1], Y[:, 0]
            Jw = -Jp @ Yx @ _left_jacobian(cp[:, :3])[ci]
            Jx = Jp @ R[ci]
            blocks = np.concatenate([Jw, Jp, Jx], axis=2)  # (nobs, 2, 9)
            return csr_matrix((blocks.ravel(), (rows, cols)), shape=(2 * nobs, x0.size))

        res = least_squares(
            residuals, x0, jac=jacobian, x_scale="jac", loss="huber", f_scale=2.0,
            method="trf", tr_solver="lsmr", max_nfev=max_nfev, verbose=0,
        )
        cp = res.x[: nc * 6].reshape(nc, 6)
        Rn = _rodrigues(cp[:, :3]) @ R0
        for n, c in enumerate(cams):
            st.poses[c] = (Rn[n], cp[n, 3:].copy())
        st.xyz[pts] = res.x[nc * 6 :].reshape(npt, 3)
        # drop points that still don't fit
        err = np.linalg.norm(res.fun.reshape(-1, 2), axis=1)
        bad = np.zeros(npt, bool)
        np.logical_or.at(bad, pi, err > self.opt.reproj_thresh * 2)
        st.xyz[pts[bad]] = np.nan
        log.debug("BA: %d cams, %d pts, median err %.2f px, dropped %d", nc, npt, np.median(err), bad.sum())

    def _filter(self, st: _State) -> None:
        pts = np.nonzero(np.isfinite(st.xyz[:, 0]))[0]
        if len(pts) < 20:
            return
        X = st.xyz[pts]
        med = np.median(X, axis=0)
        d = np.linalg.norm(X - med, axis=1)
        far = d > 10 * np.median(d)  # stray points near infinity
        st.xyz[pts[far]] = np.nan

    def _to_scene(self, st: _State) -> Scene:
        pts = np.nonzero(np.isfinite(st.xyz[:, 0]))[0]
        colors = []
        for p in pts:
            colors.append(np.mean([self.feats[i].rgb[k] for i, k in st.track_obs[p]], axis=0))
        h, w = self.frames[0].image.shape[:2]
        cams = [
            Camera(self.frames[i].name, self.K.copy(), *st.poses[i], w, h, self.frames[i].timestamp)
            for i in sorted(st.poses)
        ]
        return Scene(
            points=st.xyz[pts].copy(),
            colors=np.asarray(colors, np.uint8).reshape(-1, 3),
            cameras=cams,
            info={
                "method": "incremental-sfm",
                "frames_in": len(self.frames),
                "frames_registered": len(cams),
                "focal_px": float(self.K[0, 0]),
                "verified_pairs": len(self.matches),
            },
        )


def reconstruct(frames: list[Frame], options: SfMOptions | None = None) -> Scene:
    return IncrementalSfM(frames, options).run()
