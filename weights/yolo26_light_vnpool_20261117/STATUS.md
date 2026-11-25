# yolo26_light_vnpool_20261117 — NOT ADOPTED (model A stays)

Fine-tune of A + 361 owner-QA'd VN images (x6), Kaggle2 v5nhdf/yolo26-light-vnpool-20261117, 14 epochs, best = epoch 2 (selection val = A val + VN val, dominated by A's large val).
- VN val (51 img): precision red 0.55->0.83, green 0.51->0.82 (fewer false lights); recall red 0.68->0.53, green 0.76->0.59; yellow recall 0.33 -> 0.33.
- Light-phase ROI eval (iHanoi, 188 s): @960 56.9% (large near heads lost); @640 93.6% per frame / 95.7% voted (A @640: 96.3% / 95.2%).
- Target cases unchanged: seg05 dim yellow 10.4-12.8 s not detected by A or this model (640 or 960); seg13 small head not detected; seg08 arrow still read green.

Finding (applies to the LISA fine-tune of 16/11 too): fine-tunes trained at imgsz 640 collapse on large heads at imgsz 960
(C069 max conf A 0.56-0.76 at 960; LISA 0.01-0.06 at 960 but 0.69-0.81 at 640; this model 0-0.01 at 960, 0.26-0.51 at 640).
Model A generalises to 960; the fine-tunes do not. Always evaluate a light model at the imgsz it will run with.
