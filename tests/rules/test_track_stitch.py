"""Track stitching at the stop line (src/rules/track_stitch.py)."""
from src.rules.red_light import TrackPoint, evaluate_scene
from src.rules.track_stitch import stitch_tracks

LINE = [[0, 400], [1280, 400]]


def piece(track_id, t0, bottoms, cls="car", x=600, h=80, fps=10):
    return [TrackPoint(round(t0 + k / fps, 3), int(round((t0 + k / fps) * fps)), track_id, (x, y - h, x + 60, y), cls) for k, y in enumerate(bottoms)]


def test_id_switch_at_the_line_is_joined():
    a = piece(1, 0.0, [560, 520, 480, 440, 420])          # lost 20 px before the line
    b = piece(2, 0.6, [380, 340, 300, 260, 220])          # continues near the predicted place (mean speed 350 px/s -> 30 px off)
    points, joins = stitch_tracks(a + b, [LINE])
    assert joins == [{"kept_id": 1, "from_id": 1, "to_id": 2, "gap_s": 0.2, "distance_px": 30.0}]
    assert {p.track_id for p in points} == {1}


def test_id_switch_makes_the_crossing_visible_to_the_rules():
    a = piece(1, 0.0, [560, 520, 480, 440, 420])
    b = piece(2, 0.6, [380, 340, 300, 260, 220])
    config = {"signals": [{"signal_id": "S1", "controls_approach_ids": ["A1"]}],
              "approaches": [{"approach_id": "A1", "approach_roi": [[0, 400], [1280, 400], [1280, 720], [0, 720]],
                              "observation_roi": [[0, 0], [1280, 0], [1280, 720], [0, 720]], "stop_line": LINE, "direction": [0, -1]}]}
    intervals = {"S1": [{"kind": "observed", "state": "red", "t_start": 0, "t_end": 10}]}
    stitched, _ = stitch_tracks(a + b, [LINE])
    assert [r.assessment for r in evaluate_scene(stitched, config, intervals)] == ["red_light_crossing"]


def test_not_joined_far_from_the_line_or_after_a_long_gap_or_other_class_or_quality_gap():
    a = piece(1, 0.0, [700, 690, 680])                     # ends 280 px before the line
    b = piece(2, 0.3, [670, 660])
    assert stitch_tracks(a + b, [LINE])[1] == []
    a = piece(1, 0.0, [500, 460, 420])
    assert stitch_tracks(a + piece(2, 1.0, [380, 340]), [LINE])[1] == []          # gap 0.8 s
    assert stitch_tracks(a + piece(2, 0.3, [380, 340], cls="motorcycle"), [LINE])[1] == []
    assert stitch_tracks(a + piece(2, 0.3, [380, 340]), [LINE], gap_times=[0.25])[1] == []
    assert stitch_tracks(a + piece(2, 0.3, [380, 340], x=900), [LINE])[1] == []   # 300 px away from the prediction


def test_car_and_truck_labels_are_one_family_and_each_piece_joins_once():
    a = piece(1, 0.0, [500, 460, 420])
    b = piece(2, 0.3, [380, 340], cls="truck")
    c = piece(3, 0.3, [385, 345], x=605)
    _, joins = stitch_tracks(a + b + c, [LINE])
    assert len(joins) == 1 and joins[0]["to_id"] == 2


def test_pickup_and_minibus_join_the_four_wheel_family():
    a = piece(1, 0.0, [500, 460, 420], cls="pickup")
    b = piece(2, 0.3, [380, 340], cls="minibus")
    _, joins = stitch_tracks(a + b, [LINE])
    assert len(joins) == 1 and joins[0]["to_id"] == 2
