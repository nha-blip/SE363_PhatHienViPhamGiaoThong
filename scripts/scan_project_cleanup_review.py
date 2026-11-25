import ast
import hashlib
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/vehicle_audit/cleanup"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".gif"}
RAW_ROOTS = {"archive", "archive (1)", "MIO-TCD-Localization", "dataset_train_rgb", "AICity21_Track1_Vehicle_Counting", "Bellevue_116th_NE12th", "Bellevue_150th_Eastgate", "Bellevue_150th_Newport", "Bellevue_150th_SE38th", "Dataset_Traffic_Evaluation", "imptc_samples", "Vehicle_Tracking"}
ROLES = {
    "audit_vehicle_datasets": "Audit nguồn HCMC/MIO",
    "audit_hcmc_class_semantics": "QA ngữ nghĩa class HCMC",
    "verify_hcmc_daynight": "Xác minh class HCMC theo ngày/đêm",
    "audit_mio_mapping": "QA mapping class MIO",
    "finish_vehicle363_v1": "Pipeline build/resume Vehicle363; có chặn sửa bản frozen",
    "build_vehicle363_v1": "Entry point gọi finish_vehicle363_v1",
    "audit_duplicate_resolution": "Audit xung đột duplicate và thư viện helper được import",
    "simulate_duplicate_conflict_drop": "Mô phỏng duplicate; thư viện thống kê được import",
    "dryrun_final_duplicate_resolution": "Mô phỏng chính sách 4.7C; thư viện kiểm tra leakage",
    "apply_vehicle363_duplicate_policy": "Áp dụng/rollback 4.7C; helper integrity; có guard frozen",
    "prepare_vehicle363_final_qa": "Tạo QA 4.8 và helper kiểm tra",
    "freeze_vehicle363_v1": "Freeze và xác minh hash release; vẫn cần cho --verify",
    "audit_vehicle363_cleanup": "Kế hoạch cleanup và helper đo hardlink/allocation",
    "verify_vehicle363_cleanup_apply": "Preflight/postcheck cleanup đã duyệt; lưu dấu vết thao tác",
    "convert_bosch_to_yolo": "Chuyển Bosch thành YOLO, phục vụ tái tạo TrafficLight363",
    "create_bosch_sequence_split": "Split Bosch theo sequence, phục vụ tái tạo",
    "qa_bosch_dataset": "Kiểm tra Bosch conversion",
    "render_bosch_qa_previews": "Render QA Bosch; pipeline/provenance TrafficLight363",
    "audit_kaggle_traffic_lights": "Audit nguồn traffic light Kaggle",
    "build_kaggle_tl_yolo_v1": "Chuyển nguồn Kaggle, phục vụ tái tạo TrafficLight363",
    "build_trafficlight363_v1": "Build TrafficLight363 từ các nguồn đã chuyển",
    "scan_project_cleanup_review": "Tạo danh sách duyệt script/ảnh; không có chế độ xóa",
}


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def image_group(relative):
    parts = Path(relative).parts
    if parts[0] != "data":
        return str(Path(relative).parent).replace("\\", "/"), "REVIEW_BEFORE_DELETE", "Ảnh ngoài dataset; chưa xác định mục đích"
    top = parts[1]
    if top in {"Vehicle363-v1", "TrafficLight363-v1"}:
        return "/".join(parts[:3]), "KEEP_CRITICAL", "Dataset được bảo vệ; QA nằm trong release cũng không được xóa"
    if top in RAW_ROOTS:
        return "/".join(parts[:2]), "KEEP_CRITICAL", "Nguồn gốc/download; không được xóa"
    if top == "vehicle_audit":
        if "quarantine" in parts:
            return "/".join(parts[:parts.index("quarantine") + 1]), "REVIEW_BEFORE_DELETE", "Quarantine của quyết định duplicate; chưa được duyệt xóa"
        if parts[2:4] == ("duplicate_resolution", "qa"):
            return "/".join(parts[:5]), "KEEP_QA_EVIDENCE", "QA duplicate/calibration hỗ trợ quyết định đã áp dụng; không coi là ảnh tạm"
        if len(parts) > 4 and parts[2] == "final_qa":
            return "/".join(parts[:4]), "KEEP_QA_EVIDENCE", "Bằng chứng human QA cuối; giữ nguyên"
        group = "/".join(parts[:3]) if len(parts) > 3 else "/".join(parts[:2])
        return group, "KEEP_QA_EVIDENCE", "Bằng chứng mapping/audit nguồn; cần giữ cho provenance"
    return "/".join(parts[:2]), "REVIEW_BEFORE_DELETE", "Chưa có bằng chứng là bản sao sinh lại được"


def scan():
    scripts = []
    cache_files = defaultdict(list)
    image_groups = {}
    instructions = []
    errors = []
    stack = [ROOT]
    file_count = 0
    while stack:
        directory = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    relative = path.relative_to(ROOT).as_posix()
                    if entry.is_symlink() or entry.is_junction():
                        errors.append({"path": relative, "reason": "Không đi theo symlink/junction"})
                        continue
                    if entry.is_dir(follow_symlinks=False):
                        if entry.name not in {".git", ".codex", ".agents", ".aws"}:
                            stack.append(path)
                        if relative in {"__pycache__", "scripts/__pycache__", ".ipynb_checkpoints", "notebook/.ipynb_checkpoints"}:
                            cache_files.setdefault(relative, [])
                        continue
                    information = os.stat(path, follow_symlinks=False)
                    file_count += 1
                    if entry.name == "AGENTS.md":
                        instructions.append(relative)
                    if path.suffix.lower() in {".py", ".ps1", ".sh", ".bat", ".cmd", ".ipynb"} and not any(part in {"__pycache__", ".ipynb_checkpoints"} for part in path.parts):
                        scripts.append({"path": relative, "size_bytes": information.st_size})
                    for prefix in ("__pycache__/", "scripts/__pycache__/", ".ipynb_checkpoints/", "notebook/.ipynb_checkpoints/"):
                        if relative.startswith(prefix):
                            cache_files[prefix[:-1]].append({"path": relative, "size_bytes": information.st_size})
                    if path.suffix.lower() in IMAGE_EXTENSIONS:
                        group, category, reason = image_group(relative)
                        item = image_groups.setdefault(group, {"path": group, "category": category, "reason": reason, "image_count": 0, "logical_bytes": 0})
                        item["image_count"] += 1
                        item["logical_bytes"] += information.st_size
                    if file_count % 150000 == 0:
                        print(f"Scanned {file_count} files", flush=True)
        except OSError as error:
            errors.append({"path": directory.relative_to(ROOT).as_posix(), "reason": str(error)})
    return scripts, cache_files, image_groups, instructions, errors, file_count


def main():
    release_path = ROOT / "data/Vehicle363-v1/RELEASE_MANIFEST.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    provenance = {entry["workspace_relative_path"].replace("\\", "/"): entry["sha256"] for entry in release["generation_and_audit_scripts"]}
    scripts, caches, images, instructions, errors, file_count = scan()
    modules = {Path(item["path"]).stem: item["path"] for item in scripts if item["path"].endswith(".py")}
    incoming = defaultdict(set)
    for item in scripts:
        if not item["path"].endswith(".py"):
            continue
        try:
            tree = ast.parse((ROOT / item["path"]).read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module == "scripts":
                    imports = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports = [node.module]
                else:
                    imports = []
                for module in imports:
                    module_name = module.removeprefix("scripts.").split(".")[0]
                    if module_name in modules:
                        incoming[modules[module_name]].add(item["path"])
        except (SyntaxError, UnicodeError) as error:
            errors.append({"path": item["path"], "reason": str(error)})
    for item in scripts:
        path = ROOT / item["path"]
        recorded_path = item["path"] if item["path"] in provenance else item["path"].removeprefix("scripts/")
        item.update(role=ROLES.get(path.stem, "Notebook/code khác; chưa chứng minh dư thừa"), category="KEEP_REPRODUCIBILITY" if path.stem in ROLES else "REVIEW_BEFORE_DELETE", imported_by=sorted(incoming[item["path"]]), recorded_in_vehicle_release=recorded_path in provenance, release_recorded_path=recorded_path if recorded_path in provenance else None, sha256=digest(path))
        if item["recorded_in_vehicle_release"]:
            item["release_script_hash_matches"] = item["sha256"] == provenance[recorded_path]
            if not item["release_script_hash_matches"]:
                errors.append({"path": item["path"], "reason": "Hash script khác provenance release; cần điều tra, không xóa"})
    candidates = []
    for directory, files in sorted(caches.items()):
        safe = directory in {"__pycache__", "scripts/__pycache__"} and all(Path(item["path"]).suffix == ".pyc" for item in files)
        evidence = "Bytecode Python có thể sinh lại; giữ nguyên script nguồn"
        if directory == ".ipynb_checkpoints":
            safe = not files
            evidence = "Thư mục checkpoint trống" if safe else "Checkpoint chưa xác định có dữ liệu duy nhất hay không"
        if directory == "notebook/.ipynb_checkpoints":
            expected = {"trans-checkpoint.txt", "Untitled-checkpoint.ipynb"}
            safe = {Path(item["path"]).name for item in files} == expected
            if safe:
                same_text = digest(ROOT / directory / "trans-checkpoint.txt") == digest(ROOT / "notebook/trans.txt")
                empty_notebook = json.loads((ROOT / directory / "Untitled-checkpoint.ipynb").read_text(encoding="utf-8")).get("cells") == []
                safe = same_text and empty_notebook
            evidence = "trans-checkpoint.txt giống hash trans.txt; Untitled-checkpoint.ipynb không có cell; notebook chính được giữ" if safe else "Checkpoint cần kiểm tra bản duy nhất"
        candidates.append({"id": f"C{len(candidates) + 1:02d}", "path": directory, "category": "SAFE_TO_DELETE_AFTER_APPROVAL" if safe else "REVIEW_BEFORE_DELETE", "reason": evidence, "file_count": len(files), "logical_bytes": sum(item["size_bytes"] for item in files), "files": files})
    report = {"date": "2026-10-05", "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "REVIEW_REQUIRED" if errors else "READY_FOR_HUMAN_REVIEW", "files_deleted": 0, "scope": "Workspace scripts, selected caches/checkpoints and grouped image directories; data sizes are LOGICAL, not hardlink-aware reclaim estimates", "full_frozen_hash_audit_repeated": False, "rules": "Không xóa scripts tái tạo, frozen datasets, raw nguồn hay QA/provenance. Không có thư mục ảnh đã chứng minh an toàn để xóa. Không import/chạy pipeline build hoặc training.", "scripts": sorted(scripts, key=lambda item: item["path"]), "approval_candidates": candidates, "image_directories": sorted(images.values(), key=lambda item: -item["logical_bytes"]), "instructions_found": instructions, "scan_errors": errors, "files_scanned": file_count, "previously_deleted_cache_files": ["data/vehicle_audit/vehicle363_build/build_plan.pkl", "data/vehicle_audit/vehicle363_build/parsed_annotations.pkl"]}
    lines = ["# Danh sách script/thư mục ảnh để duyệt — 2026-10-05", "", "CHƯA XÓA GÌ. Chỉ lập danh sách; cần xác nhận cụ thể ID/đường dẫn trước mọi thao tác.", "", report["rules"], "", "## 1. Có thể đề xuất xóa sau khi bạn xác nhận", "", "| ID | Đường dẫn | Phân loại | Số file | MB logical | Lý do |", "|---|---|---|---:|---:|---|"]
    for item in candidates:
        lines.append(f"| {item['id']} | `{item['path']}` | {item['category']} | {item['file_count']} | {item['logical_bytes'] / 1e6:.6f} | {item['reason']} |")
    lines += ["", "## 2. Script/notebook — giữ hoặc cần xem riêng", "", "Script chạy xong KHÔNG đồng nghĩa không cần nữa. Script không được import cũng có thể là CLI hoặc provenance. Không xóa các script cũ chỉ vì thư mục output trung gian không còn; Bosch/Kaggle vẫn là pipeline giải thích TrafficLight363.", "", "| Đường dẫn | KB | Phân loại | Vai trò | Được import bởi | Có trong release |", "|---|---:|---|---|---|---|"]
    for item in report["scripts"]:
        lines.append(f"| `{item['path']}` | {item['size_bytes'] / 1000:.3f} | {item['category']} | {item['role']} | {', '.join(item['imported_by']) or 'Không tìm thấy import tĩnh; không chứng minh unused'} | {'YES' if item['recorded_in_vehicle_release'] else 'NO'} |")
    lines += ["", "## 3. Thư mục chứa ảnh — kích thước chỉ tính ảnh", "", "Các hàng là nhóm không chồng nhau, gồm ảnh bên dưới. GB decimal; hardlink khiến logical size không bằng dung lượng thu hồi. Không đi theo junction/symlink. Không có thư mục ảnh nào được xếp SAFE_TO_DELETE.", "", "| Thư mục/nhóm | Số ảnh | GB logical | Phân loại | Lý do |", "|---|---:|---:|---|---|"]
    for item in report["image_directories"]:
        lines.append(f"| `{item['path']}` | {item['image_count']} | {item['logical_bytes'] / 1e9:.6f} | {item['category']} | {item['reason']} |")
    lines += ["", "## 4. Ghi chú", "", "- notebook/Untitled.ipynb có code PySpark/output; không coi là notebook trống chỉ dựa vào tên.", "- Quarantine vẫn cần duyệt riêng, không tự xóa hoặc đưa lại vào active dataset.", "- build_plan.pkl và parsed_annotations.pkl đã xóa theo phê duyệt trước; không đưa vào ứng viên lần này.", "- Chưa chạy lại full hash audit dataset: đây chỉ là danh sách, không phải xác nhận integrity mới. Trước một lần xóa mới cần preflight đúng phạm vi đã được phê duyệt.", f"- Scan errors: {json.dumps(errors, ensure_ascii=False)}", "", "NEXT: Bạn xác nhận các ID Cxx muốn xóa. Những mục KEEP không nằm trong phạm vi đề xuất xóa.", ""]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "script_image_cleanup_review.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "script_image_cleanup_review.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": report["status"], "files_scanned": file_count, "script_count": len(scripts), "image_groups": len(images), "image_count": sum(item["image_count"] for item in images.values()), "candidates": [{key: item[key] for key in ("id", "path", "category", "file_count", "logical_bytes")} for item in candidates], "errors": errors}, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
