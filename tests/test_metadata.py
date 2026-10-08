import numpy as np
import pytest

from scan3d.depth import fuse
from scan3d.export import import_viewer_tags, read_ply, write_ply, write_viewer
from scan3d.scene import Annotation, Scene
from scan3d.sensors import attach_sensor_log


@pytest.fixture()
def scene(rgbd_frames):
    return fuse(rgbd_frames, voxel=0.04)


def test_save_load_roundtrip(tmp_path, scene):
    scene.add_annotation(Annotation("table", [0, 0, 0.75], 0.5, {"seats": 4, "nested": {"a": [1, 2]}}))
    scene.save(tmp_path)
    back = Scene.load(tmp_path)
    assert back.annotations[0].properties == {"seats": 4, "nested": {"a": [1, 2]}}
    assert back.units == "meters"
    np.testing.assert_allclose(back.points, scene.points, atol=1e-5)
    np.testing.assert_allclose(back.cameras[3].R, scene.cameras[3].R)


def test_point_labels_pick_nearest_anchor(scene):
    a = scene.add_annotation(Annotation("a", [0, 0, 0.75], 0.6))
    b = scene.add_annotation(Annotation("b", [0.5, 0, 0.75], 0.6))
    labels = scene.point_labels()
    pa = scene.points[labels == 0]
    assert len(pa) and len(scene.points[labels == 1])
    assert np.all(np.linalg.norm(pa - a.position, axis=1) <= np.linalg.norm(pa - b.position, axis=1) + 1e-9)


def test_locate_pixel_hits_the_surface(scene, rendered):
    cams, renders = rendered
    cam, (_, depth, _) = cams[5], renders[5]
    u, v = 200.0, 150.0
    p = scene.locate_pixel(cam.name, u, v)
    o, d = cam.ray(u, v)
    truth = o + d * depth[int(v), int(u)] / (d @ cam.forward)
    assert np.linalg.norm(p - truth) < 0.06


def test_rescale_moves_everything(scene):
    scene.add_annotation(Annotation("x", [1, 0, 0], 0.1))
    c0 = scene.cameras[0].center.copy()
    scene.rescale(2.0, "feet")
    np.testing.assert_allclose(scene.cameras[0].center, 2 * c0)
    assert scene.annotations[0].position == [2, 0, 0]
    assert scene.annotations[0].radius == pytest.approx(0.2)
    assert scene.units == "feet"


def test_ply_embeds_tags_and_metadata(tmp_path, scene):
    scene.add_annotation(Annotation("table", [0, 0, 0.75], 0.5, {"material": "oak"}))
    write_ply(scene, tmp_path / "p.ply")
    xyz, rgb, tag, meta = read_ply(tmp_path / "p.ply")
    assert len(xyz) == len(scene.points)
    np.testing.assert_array_equal(rgb, scene.colors)
    np.testing.assert_array_equal(tag, scene.point_labels())
    assert (tag == 0).sum() > 10
    assert meta["units"] == "meters"
    assert meta["annotations"][0]["properties"] == {"material": "oak"}


def test_viewer_and_tag_import(tmp_path, scene):
    write_viewer(scene, tmp_path / "v.html")
    html = (tmp_path / "v.html").read_text()
    assert "/*__SCENE_DATA__*/" not in html and '"positions":"' in html
    (tmp_path / "tags.json").write_text(
        '{"annotations":[{"id":"abc12345","label":"switch","position":[1,2,1.2],"radius":0.1,'
        '"properties":{"circuit":"B4"},"source":"viewer"}]}'
    )
    import_viewer_tags(scene, tmp_path / "tags.json")
    import_viewer_tags(scene, tmp_path / "tags.json")  # re-import replaces, no duplicates
    assert [a.label for a in scene.annotations] == ["switch"]


def test_sensor_log_placed_along_path(tmp_path, scene, rendered):
    cams, _ = rendered
    log = tmp_path / "log.csv"
    t_mid = (cams[2].timestamp + cams[3].timestamp) / 2
    log.write_text(
        "timestamp,key,value,unit\n"
        f"{cams[2].timestamp},temperature,21.0,C\n"
        f"{cams[2].timestamp},co2,600,ppm\n"
        f"{t_mid},temperature,21.4,C\n"
        f"{cams[10].timestamp},temperature,19.5,C\n"
        "999,temperature,0,C\n"  # outside the recording: ignored
    )
    tags = attach_sensor_log(scene, log, merge_radius=0.2)
    assert len(tags) == 2
    first = tags[0]
    np.testing.assert_allclose(first.position, cams[2].center, atol=1e-9)
    assert first.properties["temperature"]["latest"] == 21.4
    assert first.properties["temperature"]["mean"] == pytest.approx(21.2)
    assert first.properties["co2"]["latest"] == 600
    assert "co2 600 ppm" in first.label
    np.testing.assert_allclose(tags[1].position, cams[10].center, atol=1e-9)
