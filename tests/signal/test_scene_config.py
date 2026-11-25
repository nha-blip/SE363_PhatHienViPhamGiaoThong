import pytest

from src.scene.config import activation_problems, load, new_config, save


@pytest.fixture
def image(tmp_path):
    path = tmp_path / "ref.jpg"
    path.write_bytes(b"not really a jpeg, only hashed")
    return path


def config(image, **signal):
    return new_config("cam", 1, image, (1280, 720), [{"signal_id": "cam:S1", "roi_xyxy": [100, 50, 120, 100], **signal}], "tester")


def test_signal_only_draft_cannot_activate(image):
    problems = activation_problems(config(image))
    assert any("no approach" in p for p in problems)


def test_degenerate_stop_line_and_direction_are_rejected(image):
    c = config(image, controls_approach_ids=["A1"], mapping_status="CONFIRMED")
    c["approaches"] = [{"approach_id": "A1", "approach_roi": [[0, 400], [600, 400], [600, 720]], "observation_roi": [[0, 0], [1280, 0], [1280, 720]],
                        "stop_line": [[616.8, 327.2], [616.8, 327.2]], "direction": [0.0, 0.0]}]
    problems = activation_problems(c)
    assert any("stop_line shorter" in p for p in problems) and any("direction" in p for p in problems)


def test_extra_signal_for_another_direction_is_allowed(image):
    c = config(image, controls_approach_ids=["A1"], mapping_status="CONFIRMED")
    c["signals"].append({"signal_id": "cam:S2", "roi_xyxy": [200, 50, 220, 100], "controls_approach_ids": [], "mapping_status": "UNCONFIRMED"})
    c["approaches"] = [{"approach_id": "A1", "approach_roi": [[0, 400], [600, 400], [600, 720]], "observation_roi": [[0, 0], [1280, 0], [1280, 720]],
                        "stop_line": [[0, 420], [600, 420]], "direction": [0, -1]}]
    assert activation_problems(c) == []


def test_complete_config_can_activate(image):
    c = config(image, controls_approach_ids=["A1"], mapping_status="CONFIRMED")
    c["approaches"] = [{"approach_id": "A1", "approach_roi": [[0, 400], [600, 400], [600, 720], [0, 720]],
                        "observation_roi": [[0, 200], [1280, 200], [1280, 720], [0, 720]], "stop_line": [[0, 420], [600, 420]], "direction": [0, -1]}]
    assert activation_problems(c) == []


def test_roi_outside_image_and_unknown_approach_are_reported(image):
    c = config(image, roi_xyxy=[1270, 50, 1300, 100], controls_approach_ids=["X"], mapping_status="CONFIRMED")
    problems = activation_problems(c)
    assert any("outside" in p for p in problems)
    assert any("unknown approach X" in p for p in problems)


def test_saving_never_overwrites_a_version(image, tmp_path):
    c = config(image)
    path = save(c, tmp_path)
    assert load(path)["scene_config_id"] == "cam:v1"
    with pytest.raises(FileExistsError):
        save(c, tmp_path)


def test_approach_warnings_gap_and_thin_roi():
    from src.scene.config import approach_warnings
    good = {"approaches": [{"approach_id": "A1", "approach_roi": [[0, 300], [640, 300], [640, 360], [0, 360]],
                            "stop_line": [[0, 300], [640, 300]], "direction": [0, -1]}]}
    assert approach_warnings(good) == []
    gap_thin = {"approaches": [{"approach_id": "A1", "approach_roi": [[0, 337], [640, 337], [640, 359], [0, 359]],
                                "stop_line": [[0, 325], [640, 325]], "direction": [0, -1]}]}
    warnings = approach_warnings(gap_thin)
    assert len(warnings) == 2 and "12 px" in warnings[0] and "22 px" in warnings[1]
