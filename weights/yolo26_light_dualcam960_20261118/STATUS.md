# yolo26_light_dualcam960_20261118 - NOT ADOPTED (18/11/2026)

Kaggle2 notebook v5nhdf/yolo26-light-dualcam960-20261118: fine-tune of light model A at imgsz 960, MuSGD lr0 0.002, no warmup,
no mosaic, 12 epochs (1.8 h). Train = A's balanced list + VN pool x3 + DualCam train x2. Weights: runs/project363/yolo26_light_dualcam960_20261118/output/out/best.pt

| val (imgsz 960) | A mAP50-95 | new | yellow R A -> new | red R | green R |
|---|---|---|---|---|---|
| A val | 0.382 | 0.391 | 0.48 -> 0.48 | 0.84 -> 0.85 | 0.89 -> 0.89 |
| DualCam val | 0.274 | 0.525 | 0.57 -> 0.80 | 0.80 -> 0.82 | 0.67 -> 0.69 |
| VN val (51 img) | 0.424 | 0.349 | 0.39 -> 0.33 | 0.85 -> 0.50 | 0.65 -> 0.47 |

Light-phase ROI eval (iHanoi/AICC/Bellevue, model alone, corrected truth): voted 95.2% -> 90.4%, wrong s 8 -> 12,
frame coverage 0.989 -> 0.809 (fewer detections on VN heads, e.g. C069_retry 33/39 -> 28/39, news_seg13 440 -> 308 detections).

Decision: keep model A (+ k1bpos reader). DualCam teaches its own domain (yellow +23 points there) but the model drifts
away from VN heads.
