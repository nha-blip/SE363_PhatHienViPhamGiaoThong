# light_colour_cls_20261117 — per-ROI signal colour classifier (yolo11s-cls, 64 px), CANDIDATE

Classes red, yellow, green, off, none. Kaggle2 v5nhdf/light-colour-classifier-20261117, crops data/LightCrops363_20261117.
- Val all (A val + VN val crops): 95.9% (red 96.8, green 97.0, none 97.5, off 86.1, yellow 66.2). VN val: 79.8% (yellow 5/18).
- Light-phase ROI eval (iHanoi + AICC22 + Bellevue, 188 s), voted accuracy / wrong seconds:
  A only 95.2% / 8; classifier on ROI only 81.4% / 25 (multi-head or loose ROIs, e.g. C148); K1 (A box re-read) 96.8% / 5;
  K1b (A box re-read, 'none' -> whole ROI crop) 95.7% / 7.
- Hard cases: seg13 small head red (A: nothing) -> K1/K1b red; seg08 red + green arrow (A: green) -> K1b red, K1 none;
  seg05 dim yellow 10.4-12.8 s -> none for every variant (no false red).
Proposed: K1b in the pipeline (scripts/build_roi_classifier_detections.py holds the reference logic).
