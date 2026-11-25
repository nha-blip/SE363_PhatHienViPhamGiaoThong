import pytest

from src.signal.phase_voting import PhaseVoter, RoiHeadAssociator, run, signal_intervals

ROI = [{"id": 1, "xyxy": [100, 100, 120, 150]}]


def frames(colours, fps=5):
    """One frame per entry; a colour adds one detection inside the ROI, None adds nothing."""
    return [{"t": round(k / fps, 3), "scene": True,
             "detections": [] if c is None else [{"xyxy": [102, 105, 118, 145], "cls": c, "conf": 0.8}]}
            for k, c in enumerate(colours)]


def timeline(colours, **voter):
    out, _ = run(frames(colours), RoiHeadAssociator(ROI), PhaseVoter(**voter))
    return out


def test_single_flicker_does_not_change_state():
    out = timeline(["red"] * 10 + ["green"] + ["red"] * 10)
    assert {f["voted"][1] for f in out} == {"red"}


def test_real_change_is_followed_with_a_delay():
    out = timeline(["green"] * 15 + ["red"] * 15)
    states = [f["voted"][1] for f in out]
    assert states[0] == "green" and states[-1] == "red"
    switch = states.index("red")
    # 2 s window at 5 fps: red must outweigh green 1.5:1, i.e. 6 of 10 frames -> about 1.2 s latency
    assert 15 < switch <= 22


def test_unknown_after_long_gap():
    out = timeline(["red"] * 5 + [None] * 40, unknown_after_s=6.0)
    assert out[-1]["voted"][1] == "unknown"
    assert out[10]["voted"][1] == "red"


def test_intervals_assert_only_observed_span_and_mark_transition():
    out = timeline(["green"] * 15 + [None] * 3 + ["red"] * 30)
    iv = signal_intervals(out, 1)
    kinds = [(i["kind"], i["state"]) for i in iv]
    assert kinds == [("observed", "green"), ("transition", "unknown"), ("uncertain_colour", "unknown"), ("observed", "red")]
    green, transition, guard, red = iv
    assert guard["t_end"] - guard["t_start"] == pytest.approx(3.5) and red["t_start"] == guard["t_end"]
    iv = signal_intervals(out, 1, missing_yellow_guard_s=0)
    green, transition, red = iv
    assert green["t_end"] == 14 / 5 and red["t_start"] == 18 / 5
    assert transition["t_start"] == green["t_end"] and transition["t_end"] == red["t_start"]
    assert transition["from"] == "green" and transition["to"] == "red"
    assert red["decision_latency_s"] > 0


def test_short_gap_in_same_colour_is_one_interval():
    out = timeline(["red"] * 10 + [None] * 5 + ["red"] * 10)
    iv = signal_intervals(out, 1)
    assert [(i["kind"], i["state"]) for i in iv] == [("observed", "red")]


def test_long_gap_in_same_colour_is_not_back_filled():
    out = timeline(["red"] * 10 + [None] * 50 + ["red"] * 10)
    iv = signal_intervals(out, 1)
    assert [i["kind"] for i in iv] == ["observed", "no_observation", "observed"]


def test_detection_outside_roi_is_ignored():
    f = frames(["red"] * 5)
    for item in f:
        item["detections"][0]["xyxy"] = [300, 300, 320, 350]
    out, _ = run(f, RoiHeadAssociator(ROI), PhaseVoter())
    assert all(1 not in o["raw"] for o in out)
    assert signal_intervals(out, 1) == []


def test_green_yellow_red_cycle_keeps_red_asserted():
    out = timeline(["green"] * 15 + ["yellow"] * 15 + ["red"] * 15)
    kinds = [i["kind"] for i in signal_intervals(out, 1)]
    assert "uncertain_colour" not in kinds
