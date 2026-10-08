"""Attach readings from extra electronics to the place they were taken.

Carry any sensor (an Arduino/ESP32 with a temperature, CO2, Wi-Fi RSSI or
light sensor, a GPS, a BLE beacon scanner, ...) while you film, log its
readings with a timestamp, and this module places each reading where the
camera was at that moment.

CSV format (header required; ``unit`` optional)::

    timestamp,key,value,unit
    0.0,temperature,21.4,C
    0.5,wifi_rssi,-61,dBm

``timestamp`` is seconds since the start of the video (use ``--offset`` if
the logger clock started earlier or later than the recording).
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .scene import Annotation, Scene


def camera_track(scene: Scene) -> tuple[np.ndarray, np.ndarray]:
    cams = [c for c in scene.cameras if c.timestamp is not None]
    if len(cams) < 1:
        raise ValueError("no camera timestamps in this scene; was it built from a video or a timed capture?")
    cams.sort(key=lambda c: c.timestamp)
    return np.array([c.timestamp for c in cams]), np.array([c.center for c in cams])


def position_at(times: np.ndarray, centres: np.ndarray, t: float) -> np.ndarray:
    return np.array([np.interp(t, times, centres[:, k]) for k in range(3)])


def _num(v: str):
    try:
        f = float(v)
        return int(f) if f.is_integer() and "." not in v else f
    except ValueError:
        return v


def read_log(path: str | Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if not {"timestamp", "key", "value"} <= set(r):
            raise ValueError("sensor log needs columns: timestamp,key,value[,unit]")
    return rows


def attach_sensor_log(scene: Scene, path: str | Path, offset: float = 0.0, merge_radius: float | None = None) -> list[Annotation]:
    """Turn log rows into annotations.

    Readings taken close together (within ``merge_radius``) become one tag
    with a block per sensor key: its unit, latest value, mean and every
    timestamped reading.
    """
    times, centres = camera_track(scene)
    if merge_radius is None:
        ext = np.ptp(scene.points, axis=0).max() if len(scene.points) else 1.0
        merge_radius = 0.05 * ext
    added: list[Annotation] = []
    for row in read_log(path):
        t = float(row["timestamp"]) + offset
        if t < times[0] - 1 or t > times[-1] + 1:
            continue  # outside the recording
        p = position_at(times, centres, t)
        key, val, unit = row["key"].strip(), _num(row["value"].strip()), (row.get("unit") or "").strip()
        tag = next((a for a in added if np.linalg.norm(np.asarray(a.position) - p) < merge_radius), None)
        if tag is None:
            tag = Annotation(label="sensors", position=p.tolist(), radius=merge_radius, source="sensor", properties={})
            added.append(tag)
            scene.add_annotation(tag)
        block = tag.properties.setdefault(key, {"unit": unit, "readings": []})
        block["readings"].append({"t": round(t, 3), "value": val})
    for tag in added:
        parts = []
        for key, block in tag.properties.items():
            vals = [r["value"] for r in block["readings"]]
            block["latest"] = vals[-1]
            if all(isinstance(v, (int, float)) for v in vals):
                block["mean"] = round(float(np.mean(vals)), 4)
            parts.append(f"{key} {vals[-1]}{(' ' + block['unit']) if block['unit'] else ''}")
        tag.label = ", ".join(parts)
    return added
