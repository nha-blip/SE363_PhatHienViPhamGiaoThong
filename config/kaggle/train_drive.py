"""Launch independent Kaggle training workers with Drive backups."""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import yaml


TASKS = {"vehicle": 0, "traffic_light": 1}
MODELS = {"nano": "yolov8n.pt", "small": "yolov8s.pt"}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def remote_path(value):
    if not re.match(r"^[A-Za-z0-9_-]+:", value):
        raise argparse.ArgumentTypeError("Use a named rclone remote, e.g. gdrive:Project363/yolov8")
    return value.rstrip("/")


class DriveBackup:
    def __init__(self, executable, config, destination, directory):
        self.executable = executable
        self.config = config
        self.destination = destination.rstrip("/")
        self.directory = directory
        self.completed_epochs = []

    def copy(self, source, destination):
        subprocess.run(
            [self.executable, "--config", str(self.config), "copyto", str(source), destination,
             "--retries", "3", "--low-level-retries", "3", "--contimeout", "30s",
             "--timeout", "2m", "--stats", "0", "--log-level", "ERROR"],
            check=True, timeout=600,
        )

    def upload(self, trainer, kind):
        epoch = int(trainer.epoch) + 1
        folder = f"epoch_{epoch:04d}" if kind == "epoch" else "final"
        destination = f"{self.destination}/{folder}"
        status = self.directory / "backup_status.json"
        write_json(status, {"status": "UPLOADING", "epoch": epoch, "kind": kind, "completed_epochs": self.completed_epochs})
        try:
            records = {}
            last = Path(trainer.last)
            if not last.is_file() or not Path(trainer.best).is_file():
                raise FileNotFoundError("Both last.pt and best.pt must exist before backup")
            for path in (last, Path(trainer.best)):
                if not path.is_file():
                    continue
                records[path.name] = {"bytes": path.stat().st_size, "sha256": digest(path)}
                self.copy(path, f"{destination}/{path.name}")
            receipt = self.directory / "receipts" / f"{folder}.json"
            write_json(receipt, {"epoch": epoch, "kind": kind, "files": records,
                                 "created_at_utc": datetime.now(timezone.utc).isoformat(),
                                 "note": "best.pt is best so far, not necessarily this epoch; final weights may be stripped, use epoch last.pt for interrupted-run resume"})
            self.copy(receipt, f"{destination}/UPLOAD_COMPLETE.json")
            if kind == "epoch":
                self.completed_epochs.append(epoch)
            write_json(status, {"status": "UPLOADED", "epoch": epoch, "kind": kind, "destination": destination, "completed_epochs": self.completed_epochs})
            print(f"DRIVE_BACKUP_OK {folder}: {destination}", flush=True)
        except Exception as error:
            write_json(status, {"status": "FAILED", "epoch": epoch, "kind": kind,
                                "error_type": type(error).__name__, "completed_epochs": self.completed_epochs,
                                "note": "Training stops; local checkpoint remains. Remote folder without UPLOAD_COMPLETE.json is not a confirmed complete backup."})
            raise

    def after_save(self, trainer):
        self.upload(trainer, "epoch")

    def after_train(self, trainer):
        self.upload(trainer, "final")


def worker(arguments):
    from ultralytics import YOLO

    configuration = yaml.safe_load(Path(arguments.config).read_text(encoding="utf-8"))
    directory = Path(arguments.output)
    directory.mkdir(parents=True, exist_ok=True)
    configuration.update(model=arguments.weights, device=TASKS[arguments.task], save=True, save_period=-1, resume=False,
                         project=str(directory / "runs"), name=arguments.task, exist_ok=False)
    model_path = configuration.pop("model")
    configuration.pop("mode", None)
    configuration.pop("task", None)
    write_json(directory / "effective_train_config.json", {"model": model_path, **configuration})
    model = YOLO(model_path)
    backup = DriveBackup(shutil.which("rclone"), Path(arguments.rclone_config), arguments.destination, directory)
    model.add_callback("on_model_save", backup.after_save)
    model.add_callback("on_train_end", backup.after_train)
    state = directory / "training_status.json"
    write_json(state, {"status": "RUNNING", "task": arguments.task, "pid": os.getpid()})
    try:
        model.train(**configuration)
        write_json(state, {"status": "COMPLETE", "task": arguments.task, "completed_backup_epochs": backup.completed_epochs})
    except Exception as error:
        write_json(state, {"status": "FAILED", "task": arguments.task, "error_type": type(error).__name__})
        raise


def launch(arguments):
    import torch
    import ultralytics
    from ultralytics import YOLO

    executable = shutil.which("rclone")
    if not executable:
        raise RuntimeError("Install rclone before launching")
    source_config = Path(arguments.rclone_config).resolve(strict=True)
    working = Path(arguments.output_root).resolve()
    allowed = Path("/kaggle/working").resolve()
    if working != allowed and allowed not in working.parents:
        raise ValueError("Training outputs must stay under /kaggle/working, outside datasets")
    if source_config == working or working in source_config.parents or Path("/kaggle/working") in source_config.parents:
        raise ValueError("Secret rclone config must be outside /kaggle/working, e.g. /tmp/project363_rclone.conf")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Two visible CUDA GPUs required")
    configurations = {}
    for task in TASKS:
        path = (Path(arguments.config_dir) / f"{task}_{arguments.variant}_train.yaml").resolve(strict=True)
        configuration = yaml.safe_load(path.read_text(encoding="utf-8"))
        if configuration.get("resume"):
            raise ValueError("This launcher starts fresh runs, not resume")
        dataset_path = Path(configuration["data"]).resolve(strict=True)
        dataset = yaml.safe_load(dataset_path.read_text(encoding="utf-8"))
        root = Path(dataset["path"]).resolve(strict=True)
        for split in ("train", "val", "test"):
            for kind in ("images", "labels"):
                if not (root / kind / split).is_dir():
                    raise ValueError(f"Missing {root / kind / split}")
        if Path(str(configuration.get("model"))).name != MODELS[arguments.variant]:
            raise ValueError(f"Expected {MODELS[arguments.variant]} for {arguments.variant}")
        configurations[task] = path
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]
    session = working / run_id
    session.mkdir(parents=True, exist_ok=False)
    secret_directory = Path(tempfile.mkdtemp(prefix="project363-rclone-"))
    os.chmod(secret_directory, 0o700)
    check_config = secret_directory / "preflight.conf"
    shutil.copyfile(source_config, check_config)
    os.chmod(check_config, 0o600)
    destination = f"{arguments.destination}/{arguments.variant}/{run_id}"
    backup = DriveBackup(executable, check_config, destination, session)
    probe = session / "preflight.json"
    write_json(probe, {"run_id": run_id, "purpose": "Drive upload permission check; no credentials included"})
    backup.copy(probe, f"{destination}/preflight.json")
    weights = working / MODELS[arguments.variant]
    YOLO(str(weights))
    if not weights.is_file():
        raise FileNotFoundError(weights)
    write_json(session / "environment.json", {"torch": torch.__version__, "ultralytics": ultralytics.__version__,
                                               "gpus": [torch.cuda.get_device_name(index) for index in range(2)]})
    jobs = {}
    try:
        for task in TASKS:
            private_config = secret_directory / f"{task}.conf"
            shutil.copyfile(check_config, private_config)
            os.chmod(private_config, 0o600)
            directory = session / task
            directory.mkdir()
            handle = (directory / "train.log").open("w", encoding="utf-8")
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "worker", "--task", task,
                       "--config", str(configurations[task]), "--weights", str(weights),
                       "--rclone-config", str(private_config), "--destination", f"{destination}/{task}",
                       "--output", str(directory)]
            try:
                process = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                           cwd=str(directory), start_new_session=True,
                                           env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                                                "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
            finally:
                handle.close()
            jobs[task] = process
        write_json(session / "launch.json", {"run_id": run_id, "destination": destination,
                                            "pids": {task: process.pid for task, process in jobs.items()},
                                            "note": "Background processes still require an active Kaggle session"})
        print(json.dumps({"session": str(session), "drive": destination, "pids": {task: process.pid for task, process in jobs.items()}}, indent=2), flush=True)
        if arguments.wait:
            while any(process.poll() is None for process in jobs.values()):
                progress = {}
                for task, process in jobs.items():
                    status_path = session / task / "backup_status.json"
                    backup_status = json.loads(status_path.read_text()) if status_path.is_file() else {}
                    progress[task] = {
                        "process": "RUNNING" if process.poll() is None else f"EXIT={process.returncode}",
                        "backup": backup_status.get("status", "WAITING_FOR_FIRST_EPOCH"),
                        "epoch": backup_status.get("epoch"),
                    }
                print(progress, flush=True)
                time.sleep(30)
            failures = {task: process.returncode for task, process in jobs.items() if process.returncode != 0}
            if failures:
                raise RuntimeError(f"Workers failed: {failures}; inspect each train.log and backup_status.json")
            print("Both trainings and final Drive uploads COMPLETE", flush=True)
    except BaseException:
        for process in jobs.values():
            if process.poll() is None:
                process.terminate()
        raise


def main():
    parser = argparse.ArgumentParser(description="Two independent YOLOv8 jobs with synchronous per-epoch Drive backups; no source edits")
    commands = parser.add_subparsers(dest="command", required=True)
    launcher = commands.add_parser("launch")
    launcher.add_argument("--variant", choices=MODELS, required=True)
    launcher.add_argument("--config-dir", default="/kaggle/working/configs")
    launcher.add_argument("--output-root", default="/kaggle/working/yolo363_training")
    launcher.add_argument("--rclone-config", default="/tmp/project363_rclone.conf")
    launcher.add_argument("--destination", type=remote_path, default="gdrive2:Project363/yolov8")
    launcher.add_argument("--wait", action="store_true", help="Required final waiting cell for Kaggle Save & Run All")
    training = commands.add_parser("worker")
    training.add_argument("--task", choices=TASKS, required=True)
    for name in ("config", "weights", "rclone-config", "output"):
        training.add_argument(f"--{name}", required=True)
    training.add_argument("--destination", type=remote_path, required=True)
    arguments = parser.parse_args()
    if arguments.command == "launch":
        launch(arguments)
    else:
        worker(arguments)


if __name__ == "__main__":
    main()
