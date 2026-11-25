import pytest

from scripts.normalize_tracking_human_review import normalize_row


def test_appended_motor_approval_is_repaired():
    row = normalize_row(["8", "motorcycle", "motorcycle", "53", "1.0", "False", "", "REVIEW_REQUIRED", "motor", " approve"])
    assert row["approved_class"] == "motorcycle"
    assert row["decision"] == "APPROVED"


def test_notes_with_comma_are_preserved():
    row = normalize_row(["68", "", "", "0", "0.0", "False", "", "REVIEW_REQUIRED", "car", "approve", " bbox is large", " a part of bbox is on another car"])
    assert row["notes"] == "bbox is large, a part of bbox is on another car"


def test_invalid_class_is_rejected():
    with pytest.raises(ValueError):
        normalize_row(["8", "", "", "0", "0", "False", "vehicle", "APPROVED", ""])
