import csv
import json
from pathlib import Path

import pytest

from src.tracking.io import load_frame_map, sha256, verify_snapshot
from src.tracking.schemas import FrameInfo


ROOT = Path(__file__).resolve().parents[2]
SEQUENCE = ROOT / "data/Tracking363_working/sequences/vntraffic_001"


def test_real_pilot_frame_mapping():
    metadata = json.loads((SEQUENCE / "metadata/sequence.json").read_text(encoding="utf-8"))
    frames = load_frame_map(SEQUENCE / "metadata/frame_map.csv", metadata)
    assert len(frames) == 501
    assert (frames[0].source_frame_id, frames[0].mot_frame_id) == (0, 1)
    assert (frames[2].source_frame_id, frames[2].mot_frame_id) == (2, 3)
    assert (frames[208].source_frame_id, frames[208].mot_frame_id) == (208, 209)
    assert (frames[-1].source_frame_id, frames[-1].mot_frame_id) == (500, 501)


def test_reversed_map_rejected(tmp_path):
    metadata = json.loads((SEQUENCE / "metadata/sequence.json").read_text(encoding="utf-8"))
    with (SEQUENCE / "metadata/frame_map.csv").open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows, fields = list(reader), reader.fieldnames
    path = tmp_path / "bad_map.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader()
        writer.writerows(reversed(rows))
    with pytest.raises(ValueError):
        load_frame_map(path, metadata)


def test_invalid_frame_offset_rejected():
    with pytest.raises(ValueError):
        FrameInfo("vntraffic_001", 2, 2, 1920, 1080, 30, 501)


def test_pilot_and_original_inputs_match_immutable_provenance():
    manifest = json.loads((SEQUENCE / "metadata/provenance_manifest.json").read_text(encoding="utf-8"))
    for collection in ("input_sha256", "derived_artifacts_sha256"):
        for path, expected in manifest[collection].items():
            assert sha256(ROOT / path) == expected


def test_approved_integration_snapshot():
    path = ROOT / "runs/project363/tracking/work_5_3_integration_snapshot.json"
    snapshot = json.loads(path.read_text(encoding="utf-8"))
    assert snapshot["payload"]["human_qa_status"] == "PASS"
    assert snapshot["payload"]["pilot_frames"] == 501
    assert verify_snapshot(snapshot, ROOT)
