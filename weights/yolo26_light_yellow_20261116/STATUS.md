# yolo26_light_yellow_20261116 — REJECTED for deployment

Fine-tune of light model A (+617 LISA yellow-focused frames), Kaggle2 notebook v5nhdf/yolo26-light-yellow-20261116, 13 epochs (best = epoch 1, early stop).

- A's validation set: mAP50-95 0.358 -> 0.355 (unchanged); yellow recall 45.8% -> 44.1%.
- LISA holdout sequences: yellow recall 11.4% -> 60.7%, AP50 0.07 -> 0.57.
- Vietnamese cameras (light_phase_eval_20261114, 6 iHanoi clips): the large near-camera heads (~130 px at C069) are almost no longer detected
  (C069_trial01: ~600 detections with A, ~30 with this model); ROI phase accuracy 96% -> 62% (missing detections).
- Bug: class 3 is stored as 'False' (unquoted YAML `off`); the index order is correct. Fixed in the training script for future runs.

Decision: keep model A in the pipeline. Do not use this checkpoint.
