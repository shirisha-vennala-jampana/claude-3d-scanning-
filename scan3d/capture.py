"""Capture layer: turn a video or a folder of photos into a clean set of frames."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


@dataclass
class Frame:
    name: str
    image: np.ndarray  # BGR uint8
    timestamp: float | None = None


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian: low values mean motion blur / out of focus."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def resize_max(img: np.ndarray, max_size: int | None) -> np.ndarray:
    if not max_size:
        return img
    h, w = img.shape[:2]
    s = max_size / max(h, w)
    if s >= 1:
        return img
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def load_images(folder: str | Path, max_size: int | None = 1280) -> list[Frame]:
    paths = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMAGE_EXTS)
    frames = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            continue
        frames.append(Frame(p.name, resize_max(img, max_size)))
    if not frames:
        raise ValueError(f"no readable images in {folder}")
    return frames


def extract_keyframes(
    video: str | Path,
    max_frames: int = 60,
    max_size: int | None = 1280,
    candidates_per_slot: int = 5,
) -> list[Frame]:
    """Pick ``max_frames`` evenly spread, sharp frames from a video.

    The video is split into ``max_frames`` time slots; within each slot we
    look at a few candidate frames and keep the sharpest one. That gives
    even coverage of the walk-through while avoiding motion-blurred frames,
    which ruin feature matching.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if n <= 0:  # some containers don't report a count; read everything
        imgs = []
        while True:
            ok, img = cap.read()
            if not ok:
                break
            imgs.append(img)
        n = len(imgs)
    else:
        imgs = None

    slots = np.linspace(0, n, min(max_frames, n) + 1).astype(int)
    frames: list[Frame] = []
    for s0, s1 in zip(slots[:-1], slots[1:]):
        if s1 <= s0:
            continue
        idxs = np.unique(np.linspace(s0, s1 - 1, min(candidates_per_slot, s1 - s0)).astype(int))
        best, best_score, best_idx = None, -1.0, None
        for i in idxs:
            if imgs is not None:
                img = imgs[i]
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, img = cap.read()
                if not ok:
                    continue
            img = resize_max(img, max_size)
            sc = sharpness(img)
            if sc > best_score:
                best, best_score, best_idx = img, sc, i
        if best is not None:
            frames.append(Frame(f"frame_{best_idx:06d}.jpg", best, best_idx / fps))
    cap.release()
    if len(frames) < 2:
        raise ValueError(f"could not read at least two frames from {video}")
    return frames


def load_input(path: str | Path, max_frames: int = 60, max_size: int | None = 1280) -> list[Frame]:
    p = Path(path)
    if p.is_dir():
        frames = load_images(p, max_size)
        if len(frames) > max_frames:
            keep = np.linspace(0, len(frames) - 1, max_frames).astype(int)
            frames = [frames[i] for i in keep]
        return frames
    if p.suffix.lower() in VIDEO_EXTS:
        return extract_keyframes(p, max_frames, max_size)
    raise ValueError(f"{path} is neither a folder of images nor a video ({', '.join(sorted(VIDEO_EXTS))})")


def save_frames(frames: list[Frame], folder: str | Path) -> None:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for f in frames:
        cv2.imwrite(str(folder / f.name), f.image)
