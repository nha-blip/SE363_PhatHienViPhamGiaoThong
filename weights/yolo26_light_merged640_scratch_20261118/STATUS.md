# yolo26_light_merged640_scratch_20261118 - NOT ADOPTED (21/11/2026)

Kaggle2 notebook v5nhdf/yolo26-light-merged640-scratch-20261118: from yolo26s.pt with model A's exact recipe (640, batch 16,
MuSGD lr0 0.01, warmup 3, patience 15); only the data differs: A data + VN pool x3 + DualCam train x2. Early stop at epoch 45,
best epoch 30 (same as A), 3.2 h. Weights + every-5-epoch checkpoints: runs/project363/yolo26_light_merged640_scratch_20261118/output/out/

| val @960 | A | scratch640 best |
|---|---|---|
| A val mAP50-95 / yellow R | 0.382 / 0.48 | 0.404 / 0.50 |
| DualCam val mAP50-95 / yellow R | 0.274 / 0.57 | 0.567 / 0.80 |
| VN val mAP50-95 / red R / green R | 0.424 / 0.85 / 0.65 | 0.299 / 0.53 / 0.47 |
| light-phase ROI eval voted acc (coverage) | 95.2% (99.5%) | 69.1% (frame cov 60.6%), 0 wrong s |
| C069_retry large near-camera head | 33/39 | 0/39 (never detected) |

Decision: keep model A (+ k1bpos). Same pattern as the DualCam fine-tune: better on foreign data, worse on VN heads; the
large C069 head is lost even when training from scratch, so the cause is the data mix, not the fine-tune or the image size.
