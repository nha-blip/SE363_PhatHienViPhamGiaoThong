import hashlib
import json
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.verify_release_zip_deletion import EXPECTED, FINGERPRINT, file_hash, label_totals


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "data/vehicle_audit/cleanup/frozen_payload_cleanup_verification.json"


def inventory(directory):
    records = {}
    pending = [directory]
    while pending:
        folder = pending.pop()
        with os.scandir(folder) as entries:
            for entry in entries:
                metadata = entry.stat(follow_symlinks=False)
                if getattr(metadata, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                    raise RuntimeError(f"Reparse point rejected: {entry.path}")
                if entry.is_dir(follow_symlinks=False):
                    pending.append(Path(entry.path))
                elif entry.is_file(follow_symlinks=False):
                    records[Path(entry.path).relative_to(directory).as_posix()] = metadata.st_size
                else:
                    raise RuntimeError(f"Unexpected entry: {entry.path}")
    return records


def compare_file(arguments):
    source, retained, kind, relative = arguments
    source_path = source / relative
    retained_path = retained / relative
    source_digest = file_hash(source_path)
    if source_digest != file_hash(retained_path):
        raise RuntimeError(f"Content mismatch: {source_path}")
    boxes, classes = label_totals(retained_path.read_bytes()) if kind == "labels" else (0, {})
    return relative, source_digest, boxes, classes


def main():
    vehicle = ROOT / "data/Vehicle363-v1"
    if file_hash(vehicle / "FILE_HASHES.sha256") != FINGERPRINT:
        raise RuntimeError("Historical Vehicle release manifest fingerprint mismatch")
    report = {"status": "VERIFYING", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Only images/labels in the two frozen release directories; retained working copies are never deleted",
              "historical_vehicle_fingerprint": FINGERPRINT, "datasets": {}, "files_deleted": 0}
    for name, expected in EXPECTED.items():
        source = ROOT / "data" / name
        retained = ROOT / "data/YOLO363_training_working" / name
        details = {"source": str(source), "retained": str(retained), "payloads": {}, "boxes": 0,
                   "classes": {str(identity): 0 for identity in range(4)}, "protected_file_hashes": {}}
        for kind in ("images", "labels"):
            source_directory = source / kind
            retained_directory = retained / kind
            for directory in (source_directory, retained_directory):
                if directory.resolve(strict=True) != directory or directory.is_symlink():
                    raise RuntimeError(f"Unsafe directory: {directory}")
            source_files = inventory(source_directory)
            retained_files = inventory(retained_directory)
            if kind == "labels":
                retained_files = {relative: size for relative, size in retained_files.items() if not relative.endswith(".cache")}
            if source_files != retained_files or len(source_files) != expected["images"]:
                raise RuntimeError(f"Inventory/count mismatch: {name}/{kind}")
            payload_digest = hashlib.sha256()
            with ThreadPoolExecutor(max_workers=4) as executor:
                records = executor.map(compare_file, ((source_directory, retained_directory, kind, relative) for relative in sorted(source_files)))
                for position, (relative, digest, boxes, classes) in enumerate(records, 1):
                    payload_digest.update(f"{digest}  {relative}\n".encode())
                    details["boxes"] += boxes
                    for identity, count in classes.items():
                        details["classes"][str(identity)] += count
                    if position % 25000 == 0:
                        print(f"VERIFIED {name}/{kind}: {position}/{len(source_files)}", flush=True)
            details["payloads"][kind] = {"files": len(source_files), "logical_bytes": sum(source_files.values()),
                                         "manifest_sha256": payload_digest.hexdigest(), "all_file_hashes_match": True}
        if details["boxes"] != expected["boxes"]:
            raise RuntimeError(f"Box count mismatch: {name}")
        if "classes" in expected and details["classes"] != {str(identity): count for identity, count in expected["classes"].items()}:
            raise RuntimeError(f"Class count mismatch: {name}")
        for relative in inventory(source):
            if relative.split("/", 1)[0] not in {"images", "labels"}:
                details["protected_file_hashes"][relative] = file_hash(source / relative)
        report["datasets"][name] = details
        print(f"DATASET VERIFIED: {name}", flush=True)
    report["status"] = "PASS"
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"VERIFICATION_PASS: {REPORT}", flush=True)


if __name__ == "__main__":
    main()
