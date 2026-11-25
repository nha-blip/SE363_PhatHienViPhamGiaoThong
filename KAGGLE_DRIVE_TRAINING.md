# Two YOLOv8 models with per-epoch Google Drive backups

Use `scripts/kaggle_train_drive.py` with the four existing runtime YAMLs. Vehicle uses GPU 0, traffic lights GPU 1. This starts NEW training, not resume. Do not run the previous training launcher at the same time.

## 1. Prepare Drive authorization on your own computer

Install [rclone](https://rclone.org/downloads/), then run `rclone config`. Create a remote named `gdrive`, choose Google Drive, and authorize your own account through the browser. `rclone config file` displays the configuration file location.

Copy the complete, unencrypted configuration text into a Kaggle Secret named `RCLONE_CONFIG_TEXT` and enable that secret for this notebook. The config contains sensitive OAuth tokens: do NOT send it in chat, paste it in a public cell, upload it as a Dataset, or save it under `/kaggle/working`. Use a private notebook; users allowed to execute its secret-enabled code can access those credentials. Revoke authorization if leaked. An encrypted config requires additional password management and is not covered here.

Upload `scripts/kaggle_train_drive.py` and `scripts/kaggle_training_config.py` as a small private code Dataset, or recreate them as notebook files before the training cell. Replace `<your-code-dataset-path>` below with the actual mounted path, including any nested folders. The examples assume the files are uploaded at the Dataset root; include `/scripts/` if the folder is preserved.

## 2. Notebook setup cells

Select GPU T4 x2, enable Internet, and attach both frozen datasets plus your code Dataset.

```python
%pip install -q ultralytics PyYAML
!sudo apt-get update -qq && sudo apt-get install -y -qq rclone
```

Record and pin the Ultralytics version after validating your setup. Do not upgrade Torch/CUDA blindly.

Generate runtime configs again in this notebook so a fresh Save & Run All does not depend on previous interactive files:

```python
!python /kaggle/input/<your-code-dataset-path>/kaggle_training_config.py --check-pairs
```

Write authorization only to temporary storage, never notebook output:

```python
import os
from kaggle_secrets import UserSecretsClient

secret = UserSecretsClient().get_secret("RCLONE_CONFIG_TEXT")
descriptor = os.open(
    "/tmp/project363_rclone.conf",
    os.O_WRONLY | os.O_CREAT | os.O_TRUNC,
    0o600,
)
os.fchmod(descriptor, 0o600)
with os.fdopen(descriptor, "w") as handle:
    handle.write(secret)
del secret
print("Drive authorization configured; tokens not displayed")
```

## 3. Start training and leave it running

Recommended final notebook cell:

```python
!python /kaggle/input/<your-code-dataset-path>/kaggle_train_drive.py launch --destination gdrive:Project363/yolov8 --wait
```

The launcher performs a small actual Drive upload permission check BEFORE training, downloads shared `yolov8n.pt` once, then starts two separate background worker processes. The final `--wait` keeps this notebook run alive and reports worker failures. Select **Save Version -> Save & Run All** for a background Kaggle batch session; this re-executes the setup cells and starts a new run. Avoid an additional interactive training run consuming GPUs/quota. Background processes do NOT survive Kaggle session shutdown or quota/runtime termination. Do not run the launch cell twice.

Without `--wait`, the launcher returns immediately in an interactive session, but you must keep the Kaggle session alive yourself. Do not use this mode as the final cell of Save & Run All: the notebook may finish and terminate workers before training finishes.

## 4. Backup behavior

- Uses `on_model_save`, AFTER checkpoint writing/validation, not `on_train_epoch_end` (which is before validation/save).
- With `save=True`, uploads `last.pt`, existing `best.pt`, then `UPLOAD_COMPLETE.json` after each saved epoch. `best.pt` is best so far, not necessarily that epoch.
- Upload is synchronous inside each worker. Training pauses briefly while that worker uploads, avoiding copying a checkpoint while it is being rewritten. Two models still run independently on separate GPUs.
- Retries transient rclone errors. Persistent upload failure stops that model and retains its local checkpoint; it never silently reports successful backup. The other worker can continue.
- A folder without `UPLOAD_COMPLETE.json` is not a confirmed complete upload. The receipt records byte counts and SHA-256; validate downloaded bytes against it if resuming.
- Each model gets its own temporary OAuth config copy to avoid concurrent token-refresh writes. Secrets are outside saved notebook outputs; rclone commands contain only paths, not tokens. Do not enable debug/header dumps.
- Final export is uploaded separately after training. Ultralytics may strip optimizer state from final weights; use an `epoch_XXXX/last.pt` checkpoint for interrupted-run resume.
- Every epoch is retained on Drive, so provision enough space and expect upload time/quota use. Network errors, session termination during upload, or skipped checkpoint saves can prevent an epoch backup. A 100-epoch Vehicle run is not guaranteed to fit one Kaggle session.

Drive layout:

```text
Project363/yolov8/<unique-run-id>/
    preflight.json
    vehicle/
        epoch_0001/{last.pt,best.pt,UPLOAD_COMPLETE.json}
        epoch_0002/...
        final/{last.pt,best.pt,UPLOAD_COMPLETE.json}
    traffic_light/
        epoch_0001/...
        final/...
```

Local results:

```text
/kaggle/working/vehicle363_training/<unique-run-id>/
    launch.json
    environment.json
    vehicle/
        train.log
        training_status.json
        backup_status.json
        receipts/
        effective_train_config.json
        runs/vehicle/weights/
    traffic_light/...
```

Look for `DRIVE_BACKUP_OK epoch_0001` in each `train.log` and confirm `UPLOAD_COMPLETE.json` exists in Drive before leaving a long run unattended. `launch.json` records exact PIDs and destination. This script does not modify source images/labels or tracking QA, and never runs local training automatically merely by being uploaded.

## 5. Resume is a separate operation

The launcher intentionally starts fresh runs under unique IDs. Do not relaunch it expecting resume. Retrieve a verified per-epoch `last.pt` from Drive and use Ultralytics `resume=True` with the appropriate GPU and dataset paths; attach the backup callbacks again if continuing per-epoch uploads. Keep the recorded package version. Final/best inference weights are not necessarily resumable checkpoints.

References: [Ultralytics callbacks](https://docs.ultralytics.com/usage/callbacks/), [rclone copyto](https://rclone.org/commands/rclone_copyto/), [Kaggle notebook sessions](https://www.kaggle.com/docs/notebooks).
