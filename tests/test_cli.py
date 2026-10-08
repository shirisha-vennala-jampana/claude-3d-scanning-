import json

import cv2
import numpy as np
import pytest

from scan3d.cli import main
from scan3d.depth import write_rgbd_capture
from scan3d.scene import Scene


@pytest.fixture()
def fused(tmp_path, rgbd_frames):
    write_rgbd_capture(tmp_path / "cap", rgbd_frames)
    out = tmp_path / "scene"
    assert main(["fuse", str(tmp_path / "cap"), "-o", str(out), "--voxel", "0.04"]) == 0
    return out


def test_fuse_writes_all_outputs(fused):
    for f in ("scene.json", "points.npz", "points.ply", "viewer.html", "floorplan.png", "context_sheet.png",
              "context_prompt.txt", "floorplan.json"):
        assert (fused / f).exists(), f
    assert len(list((fused / "frames").iterdir())) == 12


def test_tag_by_coordinate_and_pixel_then_untag(fused, capsys):
    assert main(["tag", str(fused), "--label", "table", "--at", "0,0,0.75", "--radius", "0.5",
                 "--prop", "seats=4", "--prop", 'meta={"color": "oak"}']) == 0
    assert main(["tag", str(fused), "--label", "spot", "--frame", "3", "--pixel", "240,180"]) == 0
    scene = Scene.load(fused)
    t = scene.find_annotation("table")
    assert t.properties == {"seats": 4, "meta": {"color": "oak"}}
    spot = scene.find_annotation("spot")
    assert spot.source == "pixel" and spot.properties["picked_in"]["pixel"] == [240, 180]
    capsys.readouterr()
    assert main(["tags", str(fused), "--json"]) == 0
    assert len(json.loads(capsys.readouterr().out)) == 2
    assert main(["untag", str(fused), "spot"]) == 0
    assert [a.label for a in Scene.load(fused).annotations] == ["table"]


def test_scale_between_tags(fused):
    main(["tag", str(fused), "--label", "a", "--at", "0,0,0"])
    main(["tag", str(fused), "--label", "b", "--at", "1,0,0"])
    assert main(["scale", str(fused), "--between", "a", "b", "--distance", "3.28", "--units", "feet"]) == 0
    s = Scene.load(fused)
    assert s.units == "feet"
    assert s.find_annotation("b").position[0] == pytest.approx(3.28)


def test_reconstruct_from_image_folder(tmp_path, frames):
    imgs = tmp_path / "imgs"
    imgs.mkdir()
    for f in frames[:8]:
        cv2.imwrite(str(imgs / f.name), f.image)
    out = tmp_path / "sfm"
    assert main(["reconstruct", str(imgs), "-o", str(out), "--focal", "375"]) == 0
    s = Scene.load(out)
    assert len(s.cameras) == 8 and len(s.points) > 500
    assert s.units == "arbitrary"


def test_errors_are_reported_not_raised(tmp_path, capsys):
    assert main(["tags", str(tmp_path / "nope")]) == 1
    assert "error" in capsys.readouterr().err


def test_reconstruct_from_video(tmp_path, frames):
    from scan3d.synthetic import write_video

    write_video([f.image for f in frames], str(tmp_path / "v.mp4"), fps=5)
    out = tmp_path / "vid"
    assert main(["reconstruct", str(tmp_path / "v.mp4"), "-o", str(out), "--focal", "375", "--max-frames", "12"]) == 0
    s = Scene.load(out)
    assert len(s.cameras) >= 10
    ts = [c.timestamp for c in s.cameras]
    assert all(t is not None for t in ts) and np.all(np.diff(sorted(ts)) > 0)
