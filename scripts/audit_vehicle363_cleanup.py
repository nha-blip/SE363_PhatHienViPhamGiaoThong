import ctypes
import hashlib
import json
import os
import stat
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import apply_vehicle363_duplicate_policy as integrity
from scripts import audit_duplicate_resolution as audit
from scripts import dryrun_final_duplicate_resolution as duplicate_audit
from scripts import freeze_vehicle363_v1 as freeze
from scripts import simulate_duplicate_conflict_drop as statistics


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUTPUT = DATA / "vehicle_audit/cleanup"
EXPECTED_FINGERPRINT = "26c44c93d18755256eef9797d72b3d2231e990dc95d45775ba38183b8b902093"
CLASSES = dict(car=327150, motorcycle=236236, bus=21435, truck=88866)
SOURCES = {
    "archive", "archive (1)", "MIO-TCD-Localization", "dataset_train_rgb",
    "AICity21_Track1_Vehicle_Counting", "Bellevue_116th_NE12th",
    "Bellevue_150th_Eastgate", "Bellevue_150th_Newport", "Bellevue_150th_SE38th",
    "Dataset_Traffic_Evaluation", "imptc_samples", "Vehicle_Tracking",
}
SAFE_CACHES = {
    "data/vehicle_audit/vehicle363_build/build_plan.pkl",
    "data/vehicle_audit/vehicle363_build/parsed_annotations.pkl",
}
PRIORITY = {"SAFE_TO_DELETE": 0, "REVIEW_BEFORE_DELETE": 1, "KEEP_CRITICAL": 2}


def classify(relative):
    parts = Path(relative).parts
    top = parts[1]
    if top in {"Vehicle363-v1", "TrafficLight363-v1"}:
        return f"data/{top}", "KEEP_CRITICAL", "Protected finalized dataset; includes all previews and metadata", False
    if top in SOURCES or len(parts) == 2 and Path(top).suffix.lower() in {".zip", ".tar", ".gz", ".7z", ".rar"}:
        return f"data/{top}", "KEEP_CRITICAL", "Original/downloaded source data or source archive; deletion prohibited", False
    if relative in SAFE_CACHES:
        return relative, "SAFE_TO_DELETE", "Superseded generated build checkpoint; regenerate from raw annotations and preserved processing code into a NEW version, never in frozen v1", True
    if relative.startswith("data/vehicle_audit/duplicate_resolution/apply_transactions/") and "quarantine" in parts:
        position = parts.index("quarantine")
        return "/".join(parts[:position + 1]), "REVIEW_BEFORE_DELETE", "Quarantined former active duplicate image/label pairs; retain until explicit separate approval and reconstruction/rollback evidence assessment", False
    if top == "vehicle_audit":
        unit = "/".join(parts[:3]) if len(parts) > 3 else relative
        return unit, "KEEP_CRITICAL", "Preserved audit/provenance, QA review evidence, transaction backups or referenced image-scan evidence; do not remove", False
    return f"data/{top}", "REVIEW_BEFORE_DELETE", "Role not established; no deletion proposed without provenance review", False


def verify_dataset():
    if not freeze.DATASET.is_dir():
        raise RuntimeError("Vehicle363-v1 is missing")
    content = (freeze.DATASET / "FILE_HASHES.sha256").read_bytes()
    if hashlib.sha256(content).hexdigest() != EXPECTED_FINGERPRINT:
        raise RuntimeError("Frozen fingerprint differs from the authorized release")
    fingerprint, entries = freeze.verify_release()
    if fingerprint != EXPECTED_FINGERPRINT:
        raise RuntimeError("Release fingerprint mismatch")
    records = audit.load_json(freeze.DATASET / "metadata/source_manifest.json")
    actual = statistics.aggregate(records)
    if actual["images"] != 153174 or actual["boxes"] != 673687 or actual["class_counts"] != CLASSES:
        raise RuntimeError("Dataset count mismatch")
    expected_splits = {"train": (110478, 528720), "val": (21460, 78772), "test": (21236, 66195)}
    for split, expected in expected_splits.items():
        if (actual["per_split"][split]["images"], actual["per_split"][split]["boxes"]) != expected:
            raise RuntimeError(f"Split count mismatch: {split}")
    with ThreadPoolExecutor(max_workers=12) as executor:
        for offset in range(0, len(records), 4096):
            for result in executor.map(lambda record: integrity.inspect_record(record, decode_image=False), records[offset:offset + 4096]):
                if not result["label_sha256"]:
                    raise RuntimeError("Missing validated label hash")
            print(f"Read-only label validation: {min(offset + 4096, len(records))}/{len(records)}", flush=True)
    leakage = duplicate_audit.leakage(records)
    if any(leakage.values()):
        raise RuntimeError(f"Leakage: {leakage}")
    release = audit.load_json(freeze.DATASET / "RELEASE_MANIFEST.json")
    previous = audit.load_json(DATA / "vehicle_audit/final/work_4_9_freeze_report.json")
    human = audit.load_json(DATA / "vehicle_audit/final_qa/human_qa_confirmation.json")
    if release["status"] != "FROZEN" or previous["status"] != "PASS" or human["status"] != "PASS":
        raise RuntimeError("Frozen release or human confirmation is not PASS")
    if previous["actual"] != actual or previous["dataset_fingerprint"] != fingerprint:
        raise RuntimeError("Work 4.9 properties mismatch")
    return {"fingerprint": fingerprint, "file_hash_entries": entries, "actual": actual, "leakage": leakage, "fresh_label_validation": True, "fresh_full_release_hash_verification": True, "image_readability": "Every image byte matches the Work 4.9 fully decoded release; image decoding not repeated", "human_qa_scope": human["scope"]}


def allocation_reader():
    if os.name != "nt":
        return lambda path, information: getattr(information, "st_blocks", 0) * 512
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    class StandardInformation(ctypes.Structure):
        _fields_ = [("allocation_size", ctypes.c_int64), ("end_of_file", ctypes.c_int64), ("number_of_links", ctypes.c_uint32), ("delete_pending", ctypes.c_ubyte), ("directory", ctypes.c_ubyte)]

    kernel.CreateFileW.argtypes = [ctypes.c_wchar_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    kernel.CreateFileW.restype = ctypes.c_void_p
    kernel.GetFileInformationByHandleEx.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint32]
    kernel.GetFileInformationByHandleEx.restype = ctypes.c_int
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel.CloseHandle.restype = ctypes.c_int

    def read(path, information):
        handle = kernel.CreateFileW(str(path), 0x80, 7, None, 3, 0, None)
        if handle == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_last_error(), f"Cannot read allocated size: {path}")
        try:
            value = StandardInformation()
            if not kernel.GetFileInformationByHandleEx(handle, 1, ctypes.byref(value), ctypes.sizeof(value)):
                raise OSError(ctypes.get_last_error(), f"Cannot read allocation information: {path}")
            if value.end_of_file != information.st_size or value.number_of_links != information.st_nlink:
                raise RuntimeError(f"File changed during allocation measurement: {path}")
            return value.allocation_size
        finally:
            kernel.CloseHandle(handle)

    return read


def inventory():
    units = {}
    identities = {}
    errors = []
    read_allocation = allocation_reader()
    stack = [DATA]
    total_files = 0
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as iterator:
                for entry in iterator:
                    relative = Path(entry.path).relative_to(ROOT).as_posix()
                    information = os.stat(entry.path, follow_symlinks=False)
                    if entry.is_symlink() or getattr(information, "st_file_attributes", 0) & 0x400:
                        errors.append({"path": relative, "reason": "Reparse point not followed; size unknown and requires review"})
                        continue
                    if stat.S_ISDIR(information.st_mode):
                        stack.append(Path(entry.path))
                        continue
                    if not stat.S_ISREG(information.st_mode):
                        errors.append({"path": relative, "reason": "Non-regular file; not followed"})
                        continue
                    unit, category, reason, reproducible = classify(relative)
                    item = units.setdefault(unit, {"path": unit, "category": category, "reason": reason, "reproducible": reproducible, "safe_to_delete": category == "SAFE_TO_DELETE", "file_count": 0, "logical_bytes": 0, "exclusive_recoverable_allocated_bytes": 0})
                    item["file_count"] += 1
                    item["logical_bytes"] += information.st_size
                    identity = (information.st_dev, information.st_ino)
                    if not information.st_ino:
                        raise RuntimeError(f"File identity unavailable; cannot safely account for hardlinks: {relative}")
                    if identity not in identities:
                        identities[identity] = {"allocated": read_allocation(entry.path, information), "logical": information.st_size, "nlink": information.st_nlink, "seen": 0, "owners": set()}
                    stored = identities[identity]
                    if stored["logical"] != information.st_size or stored["nlink"] != information.st_nlink:
                        raise RuntimeError(f"File changed during storage audit: {relative}")
                    stored["seen"] += 1
                    stored["owners"].add(unit)
                    total_files += 1
                    if total_files % 100000 == 0:
                        print(f"Disk inventory: {total_files} files", flush=True)
        except OSError as error:
            errors.append({"path": directory.relative_to(ROOT).as_posix(), "reason": str(error)})
    physical = Counter()
    linked_files = 0
    unobserved_links = 0
    for stored in identities.values():
        owners = stored["owners"]
        category = max((units[owner]["category"] for owner in owners), key=PRIORITY.get)
        if stored["seen"] > stored["nlink"]:
            raise RuntimeError("Hardlink inventory inconsistent")
        if stored["nlink"] > 1:
            linked_files += 1
        if stored["seen"] < stored["nlink"]:
            unobserved_links += 1
        physical[category] += stored["allocated"]
        if category == "SAFE_TO_DELETE" and stored["seen"] == stored["nlink"]:
            owner = sorted(owners)[0]
            units[owner]["exclusive_recoverable_allocated_bytes"] += stored["allocated"]
    logical = Counter()
    for item in units.values():
        logical[item["category"]] += item["logical_bytes"]
        item["size_gb"] = round(item["logical_bytes"] / 1e9, 6)
        item["recoverable_gb"] = round(item["exclusive_recoverable_allocated_bytes"] / 1e9, 6)
    recoverable = sum(item["exclusive_recoverable_allocated_bytes"] for item in units.values())
    return {"candidates": sorted(units.values(), key=lambda item: (item["category"], -item["logical_bytes"], item["path"])), "logical_bytes_by_category": dict(logical), "allocated_bytes_by_category": dict(physical), "current_logical_bytes": sum(logical.values()), "current_unique_allocated_bytes": sum(physical.values()), "safe_logical_bytes": logical["SAFE_TO_DELETE"], "recoverable_allocated_bytes": recoverable, "estimated_logical_bytes_after_cleanup": sum(logical.values()) - logical["SAFE_TO_DELETE"], "estimated_allocated_bytes_after_cleanup": sum(physical.values()) - recoverable, "file_count": total_files, "unique_file_identities": len(identities), "hardlinked_unique_files": linked_files, "identities_with_unobserved_links": unobserved_links, "scan_errors": errors}


def gb(value):
    return f"{value / 1e9:.6f} GB"


def render(report):
    lines = ["# Post-Work-4.9 cleanup plan — DRY-RUN ONLY", "", f"CLEANUP DRY-RUN STATUS: {report['status']}", f"FROZEN DATASET VERIFIED: {report['frozen_dataset_verified']}", f"VEHICLE363-v1 FINGERPRINT: {report['fingerprint']}", "", "FILES TO BE DELETED: 0. No deletion, move, rename, compression or modification of dataset payload occurred. Only this plan and its generating script are new artifacts.", ""]
    if "inventory" not in report:
        return "\n".join(lines + [report.get("error", ""), "STOP. No deletion authorized.", ""])
    scan = report["inventory"]
    lines += [f"CURRENT PROJECT DATA SIZE (logical): {gb(scan['current_logical_bytes'])}", *[f"{category} (logical): {gb(scan['logical_bytes_by_category'].get(category, 0))}" for category in PRIORITY], f"ESTIMATED SIZE AFTER CLEANUP (logical): {gb(scan['estimated_logical_bytes_after_cleanup'])}", "", f"CURRENT UNIQUE ALLOCATED SIZE: {gb(scan['current_unique_allocated_bytes'])}", f"ACTUAL RECOVERABLE ALLOCATION ESTIMATE: {gb(scan['recoverable_allocated_bytes'])}", f"ESTIMATED UNIQUE ALLOCATED SIZE AFTER CLEANUP: {gb(scan['estimated_allocated_bytes_after_cleanup'])}", "", "GB is decimal (1,000,000,000 bytes). Logical sizes sum paths; hardlinks count repeatedly there. Unique allocation counts each Windows volume/file identity once using FileStandardInfo.AllocationSize; shared identities are attributed KEEP before REVIEW before SAFE. Only safe identities with ALL hardlinks accounted for in safe candidates are reclaimable. Directory/MFT metadata, future writes and volume-level compression/deduplication effects are outside this estimate.", "", "Scope: existing regular files beneath data/ at inventory time, before writing these plan files. Junctions/symlinks are never followed. Any scan error makes totals incomplete and blocks approval-ready status. Workspace scripts/notebooks outside data/ are protected, excluded from data totals, and must not be removed. No per-source archive is considered safe merely because an extracted copy exists.", "", "No large obsolete converted dataset/staging copy was identified as safely disposable. All QA previews are retained as human-review evidence, even where technically reproducible. image_scan.pkl is kept because historical duplicate audits explicitly reference its fingerprint. Transaction backups remain provenance; quarantine requires separate review, not blanket deletion.", "", "Reproduction of the two safe checkpoints uses preserved scripts/finish_vehicle363_v1.py parsing/planning routines and raw sources. The frozen build guards must not be bypassed: regenerate only in a separately configured NEW dataset version/work area if needed. They are not required to consume or explain the frozen release."]
    for category in ("KEEP_CRITICAL", "SAFE_TO_DELETE", "REVIEW_BEFORE_DELETE"):
        lines += ["", f"## {category}", "", "| PATH | SIZE_GB | CATEGORY | REASON | REPRODUCIBLE | SAFE_TO_DELETE | RECOVERABLE_GB |", "|---|---:|---|---|---|---|---:|"]
        for item in sorted((item for item in scan["candidates"] if item["category"] == category), key=lambda item: -item["logical_bytes"]):
            lines.append(f"| `{item['path']}` | {item['size_gb']:.6f} | {category} | {item['reason']} | {'YES' if item['reproducible'] else 'NO'} | {'YES' if item['safe_to_delete'] else 'NO'} | {item['recoverable_gb']:.6f} |")
    lines += ["", "Table rows partition files without double counting: directory rows exclude any separately listed descendant rows. A KEEP parent is not permission to remove its contents. Any future approval for SAFE applies only to the exact two checkpoint FILE paths, never their parent directory."]
    lines += ["", "## Protected datasets and evidence", "", "Vehicle363-v1; TrafficLight363-v1; HCMC raw (archive (1)); MIO-TCD raw; ALL other originals/downloads; all release manifests/hash controls/configs; class/split/source metadata; Work 4.8 human confirmation; Work 4.9 freeze reports; audit reports and important QA/provenance; processing scripts. No protected directory is a cleanup candidate.", "", f"Scan errors: {json.dumps(scan['scan_errors'], ensure_ascii=False)}", "", "NEXT: WAIT FOR HUMAN APPROVAL BEFORE DELETION. Any future approved cleanup must revalidate frozen hashes and candidate identity/sizes immediately before action. This script has no deletion/apply mode.", ""]
    return "\n".join(lines)


def write_plan(report):
    report["path_scope"] = "Non-overlapping file partitions; parent rows exclude separately listed descendants. Approval must name exact SAFE file paths, never parent folders."
    if "inventory" in report:
        report["KEEP"] = [item for item in report["inventory"]["candidates"] if item["category"] == "KEEP_CRITICAL"]
        report["SAFE_TO_DELETE"] = sorted([item for item in report["inventory"]["candidates"] if item["category"] == "SAFE_TO_DELETE"], key=lambda item: -item["logical_bytes"])
        report["REVIEW_BEFORE_DELETE"] = [item for item in report["inventory"]["candidates"] if item["category"] == "REVIEW_BEFORE_DELETE"]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "cleanup_plan.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "cleanup_plan.md").write_text(render(report), encoding="utf-8")


def main():
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "ABORTED_DATASET_NOT_VERIFIED", "frozen_dataset_verified": "NO", "fingerprint": EXPECTED_FINGERPRINT, "dry_run_only": True, "files_to_be_deleted": 0, "protected_workspace_code": True}
    try:
        report["verification"] = verify_dataset()
        report["frozen_dataset_verified"] = "YES"
        report["inventory"] = inventory()
        report["status"] = "REVIEW_REQUIRED_INCOMPLETE_SCAN" if report["inventory"]["scan_errors"] else "PASS"
        report["next"] = "WAIT FOR HUMAN APPROVAL BEFORE DELETION"
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        report["error"] = str(error)
        if report["frozen_dataset_verified"] == "YES":
            report["status"] = "REVIEW_REQUIRED_INCOMPLETE_SCAN"
    write_plan(report)
    print(render(report), flush=True)
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
