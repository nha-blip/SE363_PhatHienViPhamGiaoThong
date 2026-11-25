# Automatic Kaggle2 continuation

Controller: `scripts/autorestart_yolo26_kaggle2.py` using the isolated Kaggle2
Python environment and existing Windows User `KAGGLE_API_TOKEN_2`. No token is
written into notebooks or upload files. The current GPU notebook is untouched.

The local machine must remain on, awake, and connected. This is a background
process, not an installed Windows service or scheduled task, and does not
survive a local reboot automatically. If the computer is off when Kaggle ends,
continuation waits until the controller is started again. Closing this chat
does not stop the independent hidden process.

Defaults in `autorestart.json`: enabled, poll every 90 seconds, watch for 72
hours, at most two new GPU sessions. Quota below two hours or another active
GPU allocation causes a wait, not a quota bypass or cancellation of other jobs.
New session/training budgets are capped by both the available quota and the
platform/session limits. No paid quota, account switching, or automatic quota
purchase is enabled.

Flow:

1. Poll the exact current notebook and wait for COMPLETE or ERROR. Respect a
   manual CANCEL_ACKNOWLEDGED; do not undo a user's Stop/Cancel.
2. Download rolling last/best, epoch status, results and logs from published
   outputs. HTTP Range handles aggregate ZIPs without downloading datasets.
   Verify checkpoint SHA256 against worker metadata before deserializing own
   checkpoints locally. Backups live in `weights/yolo26_autorestart_20261028/`.
3. Inspect epoch, optimizer, target, early-stop state and expected class names.
   Do not restart tasks that reached their target or early-stopped. Missing
   optimizer/checkpoint or fatal worker errors stop for manual diagnosis.
4. Upload only checkpoints/history for incomplete tasks as a new private
   dataset; reuse the existing immutable fullscene light and vehicle datasets.
5. Submit a uniquely named private T4x2 notebook. Resume exact optimizer/epoch,
   prior best and history, retaining the original splits/hyperparameters. If
   only one task remains, the other is not retrained.

`controller.lock` uses an OS file lock; process death releases it. A per-cycle
journal records dataset/submission intent before network writes. Ambiguous
submission outcomes are reconciled against the expected private notebook;
otherwise the controller stops rather than blindly creating duplicate jobs.
State updates are atomic. The exact submitted runtime is preserved per cycle.
Network failures never trigger a fresh pretrained restart.

Status: `runs/project363/yolo26_autorestart_20261028/status.json`.
Log: `runs/project363/yolo26_autorestart_20261028/controller.log`.

Disable by changing `enabled` to false in `autorestart.json`, or create the
file `runs/project363/yolo26_autorestart_20261028/STOP`. Neither action cancels
a currently running GPU notebook; it prevents subsequent continuation.
To restart the watcher after a reboot, with no other controller running:

```powershell
.\.tools\kaggle2_cli\Scripts\python.exe scripts\autorestart_yolo26_kaggle2.py
```

Stopping or sleeping the computer can delay checkpoint backup. Published
checkpoint recovery is not continuous external backup during an active session.
