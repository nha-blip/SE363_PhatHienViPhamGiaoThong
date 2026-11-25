import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DRAFT = ROOT / "data/Tracking363_working/annotation_drafts/vntraffic_001_mot201_300"
QA = ROOT / "data/Tracking363_working/annotation_drafts/vntraffic_001_mot201_230_track_qa"
OUTPUT = ROOT / "data/Tracking363_working/annotation_drafts/vntraffic_001_mot201_300_after_track_qa"
FIRST_FRAME, LAST_FRAME = 201, 230


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def find(parent, item):
    while parent.setdefault(item, item) != item:
        parent[item] = parent[parent[item]]
        item = parent[item]
    return item


def main():
    parser = argparse.ArgumentParser(description="Apply the track-level QA answers to a new copy of the MOT 201-300 draft")
    parser.add_argument("--execute", action="store_true", help="Write the new draft; without it only the plan is printed")
    arguments = parser.parse_args()
    source = DRAFT / "annotations_draft.json"
    draft = json.loads(source.read_text(encoding="utf-8"))
    track_qa = json.loads((QA / "tracking_track_qa_result.json").read_text(encoding="utf-8"))
    split_qa = json.loads((QA / "track_split_qa_result.json").read_text(encoding="utf-8"))
    unanswered = [track for track, entry in split_qa.items() if not entry["status"]]
    if unanswered:
        raise RuntimeError(f"Frame-level QA incomplete for tracks {unanswered}")
    if any(entry["switch_frames"] for entry in split_qa.values()):
        raise RuntimeError("ID-switch frames were marked; splitting is not implemented, handle them explicitly")
    removals = {(int(track), frame) for track, entry in split_qa.items() for frame in entry["extra_frames"]}
    same = [tuple(map(int, key.split("-"))) for key, entry in track_qa["pairs"].items() if entry["label"] == "same"]
    removed_tracks_in_window = {int(track) for track in split_qa}
    parent = {}
    merges, skipped = [], []
    for first, second in same:
        # A track whose only boxes in the QA window were removed as extra boxes cannot be merged with anything.
        if second in removed_tracks_in_window or first in removed_tracks_in_window:
            skipped.append((first, second))
            continue
        parent[find(parent, max(first, second))] = find(parent, min(first, second))
        merges.append((first, second))
    mapping = {track: find(parent, track) for track in list(parent)}
    result = copy.deepcopy(draft)
    removed, remapped = [], Counter()
    for frame in result["frames"]:
        kept = []
        for item in frame["objects"]:
            if (item["track_id"], frame["mot_frame_id"]) in removals:
                removed.append((item["track_id"], frame["mot_frame_id"]))
                continue
            if item["track_id"] in mapping and mapping[item["track_id"]] != item["track_id"]:
                remapped[(item["track_id"], mapping[item["track_id"]])] += 1
                item["track_id"] = mapping[item["track_id"]]
            kept.append(item)
        identities = Counter(item["track_id"] for item in kept)
        if any(count > 1 for count in identities.values()):
            raise RuntimeError(f"Merge creates duplicate IDs in frame {frame['mot_frame_id']}: {[i for i, c in identities.items() if c > 1]}")
        frame["objects"] = kept
    missing = removals - set(removed)
    if missing:
        raise RuntimeError(f"Marked boxes not found in the draft: {sorted(missing)}")
    plan = {"removed_boxes": len(removed), "removed_by_track": dict(Counter(track for track, _ in removed)),
            "merged_pairs": merges, "merge_skipped_because_track_removed": skipped, "boxes_reassigned": {f"{a}->{b}": n for (a, b), n in remapped.items()}}
    print(json.dumps(plan, indent=2))
    if not arguments.execute:
        return
    OUTPUT.mkdir(parents=True, exist_ok=False)
    result["draft_revision"] = "after_track_qa_201_230"
    result["source_draft_sha256"] = sha256(source)
    result["status"] = "UNREVIEWED"
    result["is_ground_truth"] = False
    (OUTPUT / "annotations_draft.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    (OUTPUT / "qa_applied.json").write_text(json.dumps({**plan, "scope": "boxes marked in frames 201-230 only; frames outside that window were not reviewed",
                                                        "limitation": "QA cannot reveal vehicles that are missing from the draft"}, indent=2), encoding="utf-8")
    print("WROTE", OUTPUT)


if __name__ == "__main__":
    main()
