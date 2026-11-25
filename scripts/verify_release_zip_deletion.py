import hashlib
import json
import math
import sys
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
FINGERPRINT = "26c44c93d18755256eef9797d72b3d2231e990dc95d45775ba38183b8b902093"
EXPECTED = {
    "Vehicle363-v1": {"images": 153174, "boxes": 673687, "classes": {0: 327150, 1: 236236, 2: 21435, 3: 88866}},
    "TrafficLight363-v1": {"images": 7364, "boxes": 16674},
}
REPORT = ROOT / "data/vehicle_audit/cleanup/release_zip_deletion_verification_20261006.json"


def stream_hash(handle):
    digest = hashlib.sha256()
    for chunk in iter(lambda: handle.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def file_hash(path):
    with path.open("rb") as handle:
        return stream_hash(handle)


def label_totals(data):
    boxes = 0
    classes = Counter()
    for line in data.decode("utf-8").splitlines():
        if not line.strip():
            continue
        values = list(map(float, line.split()))
        if len(values) != 5 or not all(math.isfinite(value) for value in values):
            raise RuntimeError("Retained YOLO label malformed")
        identity, center_x, center_y, width, height = values
        if identity != int(identity) or not 0 <= identity <= 3 or not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 < width <= 1 and 0 < height <= 1):
            raise RuntimeError("Retained YOLO label values invalid")
        boxes += 1
        classes[int(identity)] += 1
    return boxes, classes


def main():
    vehicle = ROOT / "data/Vehicle363-v1"
    if file_hash(vehicle / "FILE_HASHES.sha256") != FINGERPRINT:
        raise RuntimeError("Vehicle frozen FILE_HASHES fingerprint mismatch; do not delete archives")
    if (vehicle / "DATASET_FINGERPRINT.sha256").read_text().split()[0] != FINGERPRINT:
        raise RuntimeError("Frozen fingerprint control mismatch")
    archives = []
    for name, expected in EXPECTED.items():
        folder = (ROOT / "data" / name).resolve(strict=True)
        path = (ROOT / "data" / f"{name}.zip").resolve(strict=True)
        if folder.parent != (ROOT / "data").resolve() or path.parent != folder.parent or not path.is_file():
            raise RuntimeError("Unexpected resolved archive/dataset path")
        information = path.stat()
        if getattr(information, "st_file_attributes", 0) & 0x400:
            raise RuntimeError("Archive is a linked/reparse path; do not delete")
        local_hashes = {}
        verified = 0
        boxes = 0
        classes = Counter()
        print(f"Fully comparing {name}.zip against retained frozen files...", flush=True)
        with zipfile.ZipFile(path) as archive:
            entries = [entry for entry in archive.infolist() if not entry.is_dir()]
            seen = set()
            def compare(entry):
                member = PurePosixPath(entry.filename)
                if member.parts[0] != name or ".." in member.parts or "\\" in entry.filename:
                    raise RuntimeError("Unexpected ZIP member; archive may contain unique data")
                relative = Path(*member.parts[1:])
                key = relative.as_posix()
                destination = (folder / relative).resolve(strict=True)
                if folder not in destination.parents or not destination.is_file() or destination.stat().st_size != entry.file_size:
                    raise RuntimeError(f"Missing or changed retained file: {entry.filename}")
                is_label = relative.parts[0] == "labels" and relative.suffix == ".txt"
                row_count, class_counts = 0, Counter()
                if is_label:
                    data = destination.read_bytes()
                    local_digest = hashlib.sha256(data).hexdigest()
                    row_count, class_counts = label_totals(data)
                else:
                    local_digest = file_hash(destination)
                with archive.open(entry) as handle:
                    archived_digest = stream_hash(handle)
                if local_digest != archived_digest:
                    raise RuntimeError(f"ZIP differs from retained file: {entry.filename}; preserve archive")
                return key, local_digest, row_count, class_counts
            with ThreadPoolExecutor(max_workers=8) as pool:
                for start in range(0, len(entries), 2048):
                    for key, local_digest, row_count, class_counts in pool.map(compare, entries[start:start + 2048]):
                        if key in seen:
                            raise RuntimeError("Duplicate archive member")
                        seen.add(key)
                        local_hashes[key] = local_digest
                        boxes += row_count
                        classes.update(class_counts)
                        verified += 1
                        if verified % 10000 == 0:
                            print(f"  {verified}/{len(entries)} files byte-verified", flush=True)
        local_files = {item.relative_to(folder).as_posix() for item in folder.rglob("*") if item.is_file()}
        if local_files != seen:
            raise RuntimeError("ZIP and retained dataset file sets differ; review before deletion")
        if name == "Vehicle363-v1":
            for line in (folder / "FILE_HASHES.sha256").read_text(encoding="utf-8").splitlines():
                recorded, relative = line.split(maxsplit=1)
                if local_hashes.get(relative.strip().lstrip("*")) != recorded:
                    raise RuntimeError("Frozen release payload hash mismatch; preserve archive")
        image_paths = list((folder / "images").rglob("*"))
        images = {item.relative_to(folder / "images").with_suffix("").as_posix() for item in image_paths if item.is_file() and item.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}}
        label_paths = list((folder / "labels").rglob("*.txt"))
        labels = {item.relative_to(folder / "labels").with_suffix("").as_posix() for item in label_paths}
        if images != labels or len(images) != expected["images"]:
            raise RuntimeError("Retained dataset pairing/count mismatch")
        if boxes != expected["boxes"] or "classes" in expected and dict(classes) != expected["classes"]:
            raise RuntimeError("Retained box/class totals mismatch")
        final_information = path.stat()
        if (information.st_size, information.st_mtime_ns) != (final_information.st_size, final_information.st_mtime_ns):
            raise RuntimeError("Archive changed during verification")
        archives.append({"path": str(path), "bytes": information.st_size, "mtime_ns": information.st_mtime_ns,
                         "sha256": file_hash(path), "retained_dataset": str(folder), "verified_files": verified,
                         "retained_images": len(images), "retained_boxes": boxes, "retained_class_counts": dict(classes),
                         "full_zip_and_retained_file_content_equal": True})
        print(f"  VERIFIED: {len(images)} images / {boxes} boxes", flush=True)
    report = {"status": "VERIFIED_NOT_DELETED", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "authorization": "Owner explicitly requested deletion of Vehicle363-v1.zip and TrafficLight363-v1.zip only",
              "frozen_vehicle_fingerprint": FINGERPRINT, "archives": archives,
              "total_archive_bytes": sum(item["bytes"] for item in archives), "files_deleted": 0,
              "raw_and_frozen_dataset_modified": False}
    REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


def record_deletion():
    verified = json.loads(REPORT.read_text(encoding="utf-8"))
    expected = {(ROOT / "data" / f"{name}.zip").as_posix() for name in EXPECTED}
    if verified["status"] != "VERIFIED_NOT_DELETED" or {Path(item["path"]).as_posix() for item in verified["archives"]} != expected:
        raise RuntimeError("Deletion verification does not match the exact authorized archives")
    if any(Path(item["path"]).exists() for item in verified["archives"]):
        raise RuntimeError("At least one archive still exists; cannot report deletion complete")
    if any(not (ROOT / "data" / name).is_dir() for name in EXPECTED):
        raise RuntimeError("Protected frozen dataset directory missing")
    if file_hash(ROOT / "data/Vehicle363-v1/FILE_HASHES.sha256") != FINGERPRINT:
        raise RuntimeError("Frozen fingerprint control changed after deletion")
    report = {"status": "PASS", "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "files_deleted": 2, "deleted_archives": verified["archives"],
              "deleted_archive_bytes": verified["total_archive_bytes"],
              "deleted_archive_size_gb_decimal": round(verified["total_archive_bytes"] / 1e9, 2),
              "protected_frozen_dataset_directories_present": True,
              "frozen_vehicle_fingerprint": FINGERPRINT, "raw_and_frozen_dataset_modified": False,
              "verification_report": REPORT.relative_to(ROOT).as_posix(),
              "note": "Only two explicit release ZIP paths removed with PowerShell; full ZIP/retained-file hash equivalence verified before removal. Original downloaded ZIPs, raw sources, working datasets and audit evidence retained."}
    destination = REPORT.parent / "release_zip_deletion_apply_20261006.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    destination.with_suffix(".md").write_text("# Release ZIP deletion\n\nSTATUS: PASS\nFILES DELETED: 2\n\n- data/Vehicle363-v1.zip\n- data/TrafficLight363-v1.zip\n\nArchive logical size removed: " + str(report["deleted_archive_size_gb_decimal"]) + " GB. Frozen directories and fingerprint control retained. No raw, frozen, working dataset or original downloaded source archive removed. Full byte/hash verification evidence: `release_zip_deletion_verification_20261006.json`.\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    if sys.argv[1:] == ["--record-deletion"]:
        record_deletion()
    elif sys.argv[1:]:
        raise SystemExit("Use no arguments for verification, or --record-deletion after authorized PowerShell removal")
    else:
        main()
