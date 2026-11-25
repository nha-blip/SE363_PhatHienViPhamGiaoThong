"""Spec §19.3 test cases TC-01..06 (and the TC-12 config guard) on synthetic tracks.

Scene: stop line y=400 across the image, vehicles drive up (direction (0,-1)); the bottom-centre of the
bbox is the reference point. Frames at 10 fps.
"""
import pytest

from src.rules.red_light import TrackPoint, evaluate_approach, evaluate_scene

APPROACH = {"approach_id": "A1", "approach_roi": [[0, 400], [1280, 400], [1280, 720], [0, 720]],
            "observation_roi": [[0, 100], [1280, 100], [1280, 720], [0, 720]],
            "stop_line": [[0, 400], [1280, 400]], "direction": [0, -1], "signal_ids": ["S1"]}


def track(bottoms, t0=0.0, fps=10, cls="car", track_id=1, x=600):
    return [TrackPoint(t0 + k / fps, k, track_id, (x, y - 80, x + 60, y), cls) for k, y in enumerate(bottoms) if y is not None]


def moving(start, stop, step, **kw):
    return track(list(range(start, stop, -step)), **kw)


def signal(*parts):
    """parts: (state, t_start, t_end) observed intervals; consecutive different colours get a transition."""
    out = []
    for k, (state, a, b) in enumerate(parts):
        if k and parts[k - 1][0] != state:
            out.append({"kind": "transition", "state": "unknown", "t_start": parts[k - 1][2], "t_end": a})
        out.append({"kind": "observed", "state": state, "t_start": a, "t_end": b})
    return {"S1": out}


def test_tc01_stops_before_line_at_red():
    points = track([600, 560, 520, 480, 440, 420, 415] + [415] * 30)
    r = evaluate_approach(points, APPROACH, signal(("red", 0, 10)))
    assert r.assessment == "no_red_light_crossing" and not r.crossings


def test_tc02_crosses_clearly_after_red():
    r = evaluate_approach(moving(600, 200, 20), APPROACH, signal(("green", 0, 0.2), ("red", 0.4, 10)))
    assert r.assessment == "red_light_crossing"
    c = r.crossings[0]
    assert c["t_min"] < c["t_max"] and c["signal"] == "red"


def test_tc03_crossed_before_red_still_in_junction_when_red():
    # crosses the line at ~1.0 s while green, red starts at 3 s while the car is past the line
    r = evaluate_approach(moving(600, 100, 20), APPROACH, signal(("green", 0, 2.0), ("red", 3.0, 10)))
    assert r.assessment == "no_red_light_crossing"


def test_tc04_crossing_overlaps_colour_transition():
    # line crossed between 0.9 s and 1.0 s; transition from 0.8 s to 1.2 s
    r = evaluate_approach(moving(580, 200, 20), APPROACH, signal(("yellow", 0, 0.8), ("red", 1.2, 10)))
    assert r.assessment == "insufficient_evidence"
    assert "transition" in r.crossings[0]["reason"]


def test_tc05_red_light_of_another_direction_is_not_used():
    signals = {"S1": [{"kind": "observed", "state": "green", "t_start": 0, "t_end": 10}],
               "S2": [{"kind": "observed", "state": "red", "t_start": 0, "t_end": 10}]}
    scene = {"signals": [{"signal_id": "S1", "controls_approach_ids": ["A1"]}, {"signal_id": "S2", "controls_approach_ids": ["B1"]}],
             "approaches": [{k: v for k, v in APPROACH.items() if k != "signal_ids"}]}
    results = evaluate_scene(moving(600, 200, 20), scene, signals)
    assert [r.assessment for r in results] == ["no_red_light_crossing"]


def test_tc06_track_lost_at_the_line():
    points = track([600, 560, 520, 480, 440, 420])  # still moving, vanishes 20 px before the line
    r = evaluate_approach(points, APPROACH, signal(("red", 0, 10)))
    assert r.assessment == "insufficient_evidence"
    assert "lost near the stop line" in r.uncertainty_reasons[0]


def test_tc06_gap_across_the_line_is_not_interpolated():
    bottoms = [600, 560, 520, 480, 440] + [None] * 15 + [300, 260, 220]
    r = evaluate_approach(track(bottoms), APPROACH, signal(("green", 0, 1.0), ("red", 1.2, 10)))
    c = r.crossings[0]
    assert (c["t_min"], c["t_max"]) == (0.4, 2.0)  # bounded by real observations only
    assert r.assessment == "insufficient_evidence"  # interval spans green, transition and red


def test_truncated_approach_starting_after_the_line():
    r = evaluate_approach(moving(380, 100, 20), APPROACH | {"approach_roi": [[0, 300], [1280, 300], [1280, 720], [0, 720]]},
                          signal(("red", 0, 10)))
    assert r.assessment == "insufficient_evidence" and "truncated" in r.uncertainty_reasons[0]


def test_turning_vehicle_is_out_of_scope():
    points = [TrackPoint(k / 10, k, 1, (600 + 40 * k, 520 - 80, 660 + 40 * k, 520 - 2 * k), "car") for k in range(15)]
    r = evaluate_approach(points, APPROACH, signal(("red", 0, 10)))
    assert r.scope_status == "out_of_scope" and r.assessment is None


def test_unsupported_class_is_out_of_scope():
    r = evaluate_approach(moving(600, 200, 20, cls="bicycle"), APPROACH, signal(("red", 0, 10)))
    assert r.scope_status == "out_of_scope" and r.assessment is None


def test_stop_then_go_is_one_approach_with_one_crossing():
    bottoms = [600, 560, 520, 480, 440, 420] + [420] * 20 + [380, 340, 300, 260]
    r = evaluate_approach(track(bottoms), APPROACH, signal(("red", 0, 1.0), ("green", 1.5, 10)))
    assert len(r.crossings) == 1 and r.assessment == "no_red_light_crossing"


def test_jitter_over_the_line_is_not_a_crossing():
    bottoms = [600, 520, 440, 405, 398, 405, 410, 405, 410]  # one noisy frame just past the line
    r = evaluate_approach(track(bottoms), APPROACH, signal(("red", 0, 10)))
    assert not [c for c in r.crossings if c["status"] == "red_light_crossing"]


def test_off_signal_does_not_prove_right_of_way():
    r = evaluate_approach(moving(600, 200, 20), APPROACH, {"S1": [{"kind": "observed", "state": "off", "t_start": 0, "t_end": 10}]})
    assert r.assessment == "insufficient_evidence"


def test_tc12_scene_config_without_stop_line_cannot_activate(tmp_path):
    from src.scene.config import activation_problems, new_config
    image = tmp_path / "ref.jpg"
    image.write_bytes(b"x")
    config = new_config("cam", 1, image, (1280, 720), [{"signal_id": "S1", "roi_xyxy": [10, 10, 20, 40],
                                                         "controls_approach_ids": ["A1"], "mapping_status": "CONFIRMED"}], "t")
    config["approaches"] = [{"approach_id": "A1", "approach_roi": APPROACH["approach_roi"], "observation_roi": APPROACH["observation_roi"],
                             "stop_line": None, "direction": [0, -1]}]
    assert any("stop_line" in p for p in activation_problems(config))


def test_brief_track_is_not_evidence_of_compliance():
    r = evaluate_approach(track([560, 555, 550]), APPROACH, signal(("red", 0, 10)))
    assert r.assessment == "insufficient_evidence" and "too briefly" in r.uncertainty_reasons[0]


def test_track_ending_far_from_line_while_moving_is_not_compliance():
    r = evaluate_approach(track([700, 680, 660, 640, 620, 600, 580, 560]), APPROACH, signal(("red", 0, 10)))
    assert r.assessment == "insufficient_evidence" and "still moving" in r.uncertainty_reasons[0]


def test_vehicle_creeping_a_few_pixels_over_the_line_and_stopping_is_not_a_crossing():
    bottoms = [440, 430, 420, 410, 404, 401, 399, 397, 396, 396] + [396] * 15 + [402, 401]  # like C040 track 61
    r = evaluate_approach(track(bottoms), APPROACH, signal(("red", 0, 10)))
    assert r.assessment == "no_red_light_crossing" and not r.crossings
    assert "stopped at the line" in r.uncertainty_reasons[0]


def test_vehicle_nosing_over_the_line_then_lost_is_insufficient_not_red():
    bottoms = [440, 430, 420, 410, 404, 400, 396, 392, 388]  # like C040 track 70: 12 px past, then the track ends
    pts = [TrackPoint(k / 10, k, 1, (600, y - 40, 616, y), "motorcycle") for k, y in enumerate(bottoms)]
    r = evaluate_approach(pts, APPROACH, signal(("red", 0, 10)))
    assert r.assessment == "insufficient_evidence"


def test_large_vehicle_clearly_past_the_line_is_a_crossing():
    bottoms = [470, 450, 430, 410, 395, 380, 365, 350, 335]  # 65 px past, bbox 250 px tall
    pts = [TrackPoint(k / 10, k, 1, (500, y - 250, 700, y), "bus") for k, y in enumerate(bottoms)]
    r = evaluate_approach(pts, APPROACH, signal(("red", 0, 10)))
    assert r.assessment == "red_light_crossing"


def test_crossing_right_after_green_to_red_without_yellow_is_insufficient():
    signals = {"S1": [{"kind": "observed", "state": "green", "t_start": 0, "t_end": 0.5},
                      {"kind": "transition", "state": "unknown", "t_start": 0.5, "t_end": 0.6},
                      {"kind": "uncertain_colour", "state": "unknown", "t_start": 0.6, "t_end": 4.1},
                      {"kind": "observed", "state": "red", "t_start": 4.1, "t_end": 10}]}
    r = evaluate_approach(moving(600, 200, 20), APPROACH, signals)  # crosses at ~1 s
    assert r.assessment == "insufficient_evidence" and "misread yellow" in r.crossings[0]["reason"]


def test_redundant_head_without_observation_does_not_block_the_observed_head():
    signals = {"S1": [{"kind": "observed", "state": "red", "t_start": 0, "t_end": 10}], "S2": []}
    r = evaluate_approach(moving(600, 200, 20), APPROACH | {"signal_ids": ["S1", "S2"]}, signals)
    assert r.assessment == "red_light_crossing"


def test_heads_that_disagree_still_block():
    signals = {"S1": [{"kind": "observed", "state": "red", "t_start": 0, "t_end": 10}],
               "S2": [{"kind": "observed", "state": "green", "t_start": 0, "t_end": 10}]}
    r = evaluate_approach(moving(600, 200, 20), APPROACH | {"signal_ids": ["S1", "S2"]}, signals)
    assert r.assessment == "insufficient_evidence" and "disagree" in r.crossings[0]["reason"]


# ----- news pilot fixes (16/11/2026) -----
def test_crosses_straight_then_curves_away_is_in_scope():
    # crosses the line on red going up, then turns right far past the line (news_seg00 track 17)
    straight = [(600, y) for y in range(600, 300, -20)]
    curve = [(600 + 25 * k, 300 - 5 * k) for k in range(1, 20)]
    points = [TrackPoint(k / 10, k, 1, (x, y - 80, x + 60, y), "car") for k, (x, y) in enumerate(straight + curve)]
    r = evaluate_approach(points, APPROACH, signal(("red", 0, 10)))
    assert r.scope_status == "in_scope" and r.assessment == "red_light_crossing"


def test_turning_before_the_line_is_still_out_of_scope():
    points = [TrackPoint(k / 10, k, 1, (600 + 30 * k, 600 - 5 * k - 80, 660 + 30 * k, 600 - 5 * k), "car") for k in range(30)]
    r = evaluate_approach(points, APPROACH, signal(("red", 0, 10)))
    assert r.scope_status == "out_of_scope"


def test_turn_arrow_head_green_does_not_prove_right_of_way():
    approach = dict(APPROACH, arrow_signal_ids={"S1"})
    r = evaluate_approach(moving(600, 200, 20), approach, signal(("green", 0, 10)))
    assert r.assessment == "insufficient_evidence" and "arrow" in r.uncertainty_reasons[0]
    r = evaluate_approach(moving(600, 200, 20), approach, signal(("red", 0, 10)))
    assert r.assessment == "red_light_crossing"


def test_arrow_flag_is_read_from_the_scene_config():
    config = {"signals": [{"signal_id": "S1", "controls_approach_ids": ["A1"], "has_turn_arrow": True}],
              "approaches": [{k: v for k, v in APPROACH.items() if k != "signal_ids"}]}
    results = evaluate_scene(moving(600, 200, 20), config, signal(("green", 0, 10)))
    assert results[0].assessment == "insufficient_evidence"
