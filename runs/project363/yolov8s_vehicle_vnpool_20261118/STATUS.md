# YOLOv8s vehicle VN-pool fine-tune (Kaggle2 v5nhdf/yolov8s-vehicle-vnpool-20261118, 18/11/2026) - NOT ADOPTED

Recipe: from weights/kaggle_small_vehicle_20261019/best.pt, imgsz 960, SGD lr0 0.002 cos, mosaic (close 3), 9000 random
Vehicle363 train + 80 VN x15; val = 2000 Vehicle363 val + 20 VN val (video Kv2Z_NHz9KE). Early stop at epoch 11 (best epoch 5), 0.85 h.

| model | Vehicle363 val2000 mAP50-95 | moto R | VN val mAP50-95 | VN moto R |
|---|---|---|---|---|
| v8s (before) | 0.719 | 0.860 | 0.751 | 0.946 |
| fine-tuned best | 0.700 | 0.828 | 0.597 | 0.879 |

Epoch 1 already fell to 0.647 on the combined val and never recovered: the recipe (SGD + warmup at 960, original was AdamW at 640)
disturbed the model. VN val "before" is inflated (VN labels were pre-labelled by v8s), but Vehicle363 val also fell. Weights not downloaded.
