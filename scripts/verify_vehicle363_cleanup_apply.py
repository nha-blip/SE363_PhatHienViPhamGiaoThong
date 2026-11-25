import argparse
import hashlib
import json
import stat
from datetime import datetime, timezone
from pathlib import Path

import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import audit_vehicle363_cleanup as cleanup


REPORT = cleanup.OUTPUT / "cleanup_apply.json"


def save(report):
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def preflight():
    if REPORT.exists():
        raise RuntimeError("An application report already exists; inspect it before retrying")
    plan = cleanup.audit.load_json(cleanup.OUTPUT / "cleanup_plan.json")
    if plan["status"] != "PASS" or plan["fingerprint"] != cleanup.EXPECTED_FINGERPRINT:
        raise RuntimeError("Cleanup plan is not validated")
    approved = {entry["path"]: entry for entry in plan["SAFE_TO_DELETE"]}
    if set(approved) != cleanup.SAFE_CACHES:
        raise RuntimeError("Approval scope must be exactly the two cache files")
    verification = cleanup.verify_dataset()
    files = []
    parent = (cleanup.DATA / "vehicle_audit/vehicle363_build").resolve()
    read_allocation = cleanup.allocation_reader()
    for relative in sorted(cleanup.SAFE_CACHES):
        path = (cleanup.ROOT / relative).resolve(strict=True)
        if path.parent != parent or cleanup.ROOT.resolve() not in path.parents:
            raise RuntimeError(f"Candidate escaped approved workspace directory: {path}")
        information = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(information.st_mode) or information.st_nlink != 1 or getattr(information, "st_file_attributes", 0) & 0x400:
            raise RuntimeError(f"Candidate is not an independent regular file: {path}")
        allocated = read_allocation(path, information)
        if information.st_size != approved[relative]["logical_bytes"] or allocated != approved[relative]["exclusive_recoverable_allocated_bytes"]:
            raise RuntimeError(f"Candidate changed since dry-run: {path}")
        files.append({"path": relative, "absolute_path": str(path), "logical_bytes": information.st_size, "allocated_bytes": allocated, "sha256": cleanup.audit.digest_file(path), "file_identity": [information.st_dev, information.st_ino]})
    protected = cleanup.freeze.authoritative_paths() + [cleanup.DATA / relative for relative in ("vehicle_audit/final_qa/human_qa_confirmation.json", "vehicle_audit/final/work_4_9_freeze_report.json", "vehicle_audit/final/work_4_9_freeze_report.md")]
    protected += [cleanup.freeze.DATASET / name for name in cleanup.freeze.CONTROLS]
    report = {"status": "PREFLIGHT_PASS", "approval_scope": "User approved only the two SAFE checkpoint files from cleanup_plan", "preflight_at_utc": datetime.now(timezone.utc).isoformat(), "frozen_dataset_verified": "YES", "fingerprint": cleanup.EXPECTED_FINGERPRINT, "verification": verification, "approved_files": files, "files_deleted": 0, "protected_sha256": {str(path.relative_to(cleanup.ROOT)): cleanup.audit.digest_file(path) for path in protected}}
    save(report)
    print("PREFLIGHT PASS: only build_plan.pkl and parsed_annotations.pkl may be removed", flush=True)


def postcheck():
    report = cleanup.audit.load_json(REPORT)
    if report["status"] != "PREFLIGHT_PASS":
        raise RuntimeError("Missing valid preflight")
    absent = [entry for entry in report["approved_files"] if not (cleanup.ROOT / entry["path"]).exists()]
    if len(absent) != 2 or {entry["path"] for entry in absent} != cleanup.SAFE_CACHES:
        report.update(status="REVIEW_REQUIRED_PARTIAL_CLEANUP", files_deleted=len(absent))
        save(report)
        raise RuntimeError("Both approved cache files must be absent; inspect partial cleanup")
    for relative, digest in report["protected_sha256"].items():
        if cleanup.audit.digest_file(cleanup.ROOT / relative) != digest:
            raise RuntimeError(f"Protected metadata/evidence changed: {relative}")
    content = (cleanup.freeze.DATASET / "FILE_HASHES.sha256").read_bytes()
    if hashlib.sha256(content).hexdigest() != cleanup.EXPECTED_FINGERPRINT:
        raise RuntimeError("Frozen fingerprint changed")
    records = cleanup.audit.load_json(cleanup.freeze.DATASET / "metadata/source_manifest.json")
    cleanup.integrity.inventory_check(records)
    if cleanup.statistics.aggregate(records) != report["verification"]["actual"]:
        raise RuntimeError("Final dataset properties changed")
    for directory in ("Vehicle363-v1", "TrafficLight363-v1", "archive (1)", "MIO-TCD-Localization", "vehicle_audit/final_qa", "vehicle_audit/duplicate_resolution/apply_transactions/20261004T085116457895Z/quarantine"):
        if not (cleanup.DATA / directory).is_dir():
            raise RuntimeError(f"Protected directory missing: {directory}")
    report.update(status="PASS", completed_at_utc=datetime.now(timezone.utc).isoformat(), files_deleted=2, deleted_files=absent, recovered_logical_bytes=sum(entry["logical_bytes"] for entry in absent), recovered_allocated_bytes=sum(entry["allocated_bytes"] for entry in absent), postcheck="Frozen fingerprint, protected metadata/QA evidence, active image/label inventory and counts unchanged; quarantine and raw source roots retained", next="Cleanup complete; frozen dataset remains immutable")
    save(report)
    (cleanup.OUTPUT / "cleanup_apply.md").write_text("\n".join(["# Approved checkpoint cleanup", "", "STATUS: PASS", "FILES DELETED: 2", *[f"- {entry['path']} ({entry['logical_bytes']} bytes)" for entry in absent], "", f"Recovered allocation estimate: {report['recovered_allocated_bytes'] / 1e9:.6f} GB", f"Vehicle363-v1 fingerprint: {cleanup.EXPECTED_FINGERPRINT}", "", report["postcheck"], "No other deletion authorized or performed. Original dry-run plan remains historical evidence.", ""]), encoding="utf-8")
    print(f"CLEANUP APPLY: PASS; FILES DELETED: 2; RECOVERED: {report['recovered_allocated_bytes'] / 1e9:.6f} GB", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "postcheck"))
    arguments = parser.parse_args()
    {"preflight": preflight, "postcheck": postcheck}[arguments.phase]()
