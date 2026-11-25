# Full-scene light rebuild and vehicle resume

- GPU 0: resume verified vehicle `last.pt` with optimizer at zero-based epoch
  10, so the next displayed epoch is 12/100. Keep original vehicle crop data,
  validation, hyperparameters and early-stop state for continuity. Restore the
  previous best checkpoint and 11-row results history before training.
- GPU 1: train light from COCO-pretrained YOLO26s, not the failed crop-only
  light checkpoint. Use full scenes, retain every annotation, downscale/pad
  to 1280x720 without upscaling; runtime `imgsz=640` matches the old baseline.
- Light train: 8,653 unique scenes; images containing yellow/off occur three
  times in the text sampler (10,211 effective entries), no duplicated payload.
  Unique boxes: red 9,357, yellow 540, green 9,300, off 650. Repetition does not
  add diversity and does not make these counts equal.
- Validation: same 1,898 images / 4,501 boxes, source bytes unchanged. Test
  payload is not uploaded. Source split groups are inherited, exact image
  hashes checked against validation and test. No new claims of complete
  manual GT verification or absence of all near-duplicate leakage are made.
- Source files and prior datasets are not modified. Prior crops remain as
  reproducibility artifacts, but are not part of the new light train sampler.
- `scripts/build_light_fullscene_20261028.py` builds the full-scene dataset.
  `--package-existing` validates pairs/labels and replaces only an interrupted
  invalid archive; it refuses to overwrite a completed ZIP.
- `scripts/recover_kaggle2_zip_checkpoints.py` uses HTTP Range to extract actual
  checkpoint files from Kaggle's aggregate output ZIP without downloading the
  entire 6.98 GB training payload. CRC and worker SHA256 values are verified.
- `scripts/deploy_yolo26_fullscene_resume_20261028.py` waits for both verified
  inputs, checks private upload allowlist and Kaggle2 identity, uploads only
  light archive plus vehicle checkpoints/history, then submits a private T4x2
  notebook if no other Kaggle2 GPU session is reserved.
- The submitted notebook bundles the shared runtime source into an in-memory
  `runtime_base` module. `__file__` points to the generated entry script so each
  GPU child runs the same self-contained code. Exact submitted source is saved
  to `runs/project363/yolo26_fullscene_resume_20261028/submitted_train.py`.
- Both datasets are staged to local working disk before normal YOLO cache and
  annotation checks. Per-worker startup watchdog is 45 minutes; epoch-end
  training budget 8 hours, parent hard budget 10 hours. Save rolling last/best
  each epoch with SHA256; these are session-local, not continuous Drive backup.
- After workers exit, remove only the resolved working staging directories,
  keeping weights, CSVs, logs and configuration. This avoids another huge ZIP
  containing the training dataset. Inspect `completion.json` for worker success;
  notebook COMPLETE alone does not prove training succeeded.

Progress: `runs/project363/yolo26_fullscene_resume_20261028/status.json`.
Local build and upload need this machine on until Kaggle submission is confirmed.
