import os
import sys
from pathlib import Path

os.environ["YOLO_AUTOINSTALL"] = "false"

ROOT = Path(__file__).resolve().parents[1]
LAST = ROOT / "runs/project363/yolo26_vehicle_local_resume_20261104/train/weights/last.pt"


def main():
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing silent CPU fallback")
    free, _ = torch.cuda.mem_get_info()
    if free < 4.5 * 2 ** 30:
        raise RuntimeError(f"Only {free / 2 ** 30:.1f} GiB GPU memory free; another process is using the GPU")
    YOLO(str(LAST)).train(resume=True, batch=8, workers=4, device=0)


if __name__ == "__main__":
    main()
