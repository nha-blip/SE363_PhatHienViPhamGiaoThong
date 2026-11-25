# Fresh parallel training — supersedes checkpoint recovery

The user explicitly chose to abandon the inaccessible old checkpoints and
restart both tasks, then reported stopping the original notebook. The local
recovery helper was stopped. Do not restart it: its checkpoint-wait workflow
below is historical and has been superseded.

Current launcher: `scripts/start_yolo26_fresh_kaggle2.py`.
Current notebook: `v5nhdf/yolo26s-fresh-local-20261027`, private, T4 x2.
The launcher verifies the old notebook is terminal before submission; it does
not rely solely on the user's stop confirmation or submit duplicate notebooks.
Status: `runs/project363/yolo26_kaggle2_20261027/fresh_restart_status.json`.

Both independent workers start from COCO-pretrained `yolo26s.pt`, not old
checkpoints or randomly initialized weights. Vehicle uses physical GPU 0 and
light physical GPU 1. Vehicle extracts the SHA256-verified opaque ZIP into
local disk; light copies its smaller image/label set with eight I/O threads.
Both validate image/label pairs and use normal YOLO annotation/cache checks.
Both use imgsz 640, batch 16, 100 epochs maximum, patience 15 and two loader
workers; color augmentation is disabled/reduced for light as in the old run.

Each worker saves rolling optimizer-bearing `last.pt` and `best.pt` with SHA256
under `/kaggle/working/yolo26s_fresh_local_20261027/<task>/checkpoints/`.
These are session-local files, not continuous off-session backup.
Workers emit progress/stack diagnostics. A 45-minute startup watchdog stops
only a worker that has not entered training; a ten-hour session watchdog stops
remaining workers so the script can exit normally and publish completed files
before the twelve-hour platform limit. The epoch-end training budget is eight
hours after each worker's staging. Inspect `completion.json`: normal script
exit is used to preserve outputs even if a worker failed, not to claim success.
The local computer need not stay on after Kaggle confirms submission.

## Historical recovery-only workflow (not active)

The original GPU notebook remains untouched: its completed light checkpoints
are not yet published as downloadable outputs. Cancelling it now is unsafe.

`scripts/recover_yolo26_vehicle_kaggle2.py` runs on the local machine:

1. Checks the original vehicle ZIP SHA256; hard-links it into a separate upload
   folder as `vehicle_trainval.zip.bin`, without another 5.68 GB local copy.
2. Creates a private archive-only dataset. The `.bin` suffix is intended to
   preserve the ZIP as one mount object instead of auto-extracted tiny files.
   The replacement notebook refuses to run if the opaque file is unavailable.
3. Waits for the original notebook to publish light `last.pt` and `best.pt`.
   Downloads both to `weights/yolo26s_light_20261027_recovered/` and verifies
   their SHA256 values against the completed light worker's recorded hashes.
4. Waits for the old session to end naturally. It never calls cancel.
   If that session produces a vehicle epoch checkpoint, recovery stops for
   inspection instead of discarding trained work and starting over.
5. Only after those gates, submits a new private vehicle-only notebook.

The replacement script verifies the archive, safe member paths, split counts,
image/label stem pairing and available disk space before local extraction.
YOLO then performs its normal annotation/cache checks on `/kaggle/working`.
No annotations or cache hashes are bypassed. Validation remains unchanged.
Hyperparameters match the original vehicle run: YOLO26s pretrained, 640,
batch 16, 100 epochs, patience 15, two loader workers, GPU 0. Light is not retrained.
An 8-hour epoch-end budget leaves headroom for staging and publishing outputs.
The allocated accelerator may expose two T4s, but this vehicle-only worker
uses only GPU 0. Every epoch keeps rolling `last.pt` and `best.pt`; these are
local session checkpoints, not continuous off-session backups.

Progress: `runs/project363/yolo26_kaggle2_20261027/vehicle_recovery_status.json`.
Log: `runs/project363/yolo26_kaggle2_20261027/vehicle_recovery.log`.
The lock prevents duplicate launches; do not remove it while the helper is alive.
The local computer must remain on for the recovery helper to continue.
Upload/processing has a three-hour wait budget; checkpoint recovery waits at
most twelve hours. A safe stop requires inspection before restarting the helper.
