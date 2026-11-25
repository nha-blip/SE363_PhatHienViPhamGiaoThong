# yolo26_light_yellowadd640_20261121 - NOT ADOPTED (22/11/2026)

Model A's exact recipe from yolo26s.pt (640, MuSGD, patience 15, selection on A val) + only 182 clean yellow images x3
(DualCam 154 without arrow/red-yellow heads, VN pool 28). 47 epochs (best 32), 2.6 h on Kaggle2.

| @960 | A | yellowadd |
|---|---|---|
| A val mAP50-95 / yellow R | 0.382 / 0.48 | 0.391 / 0.48 |
| DualCam val mAP50-95 / yellow R | 0.274 / 0.57 | 0.374 / 0.74 |
| VN val mAP50-95 / red R | 0.424 / 0.85 | 0.259 / 0.56 |
| light-phase ROI voted acc (frame coverage) | 95.2% (98.9%) | 69.1% (64.9%) |
| C069_retry large LED head | 33/39 | 0/39 |

C069 max confidence by checkpoint: epoch 10 0.18-0.43 (green phase only), epoch 20+ ~0 -> the large LED head is
learned away during training. Same loss as the merged scratch run, although only 182 images were added: the cause may be
run-to-run variance of A's recipe (A itself may be the lucky run), not the added data. Untested: a control run of A's
recipe on A's data alone.
