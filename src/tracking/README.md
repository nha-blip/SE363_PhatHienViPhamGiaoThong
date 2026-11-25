# Tracking363 pilot integration scaffold

This is interface preparation only. No detector or real tracker is created on import.

- `schemas.py`: validated, immutable pixel-xyxy prediction contracts, frame IDs and JSON round trips. Source frame = MOT frame - 1. Detector IDs retain their original model namespace; tracker IDs are predictions and are NOT ground-truth IDs.
- `yolo_adapter.py`: consumes an ALREADY produced axis-aligned Ultralytics Results object; never loads a model or runs inference. Filters by actual checkpoint class names: car/motorcycle/bus/truck only. Traffic-light classes are excluded even if their numeric IDs overlap those of a different model. No class-ID remapping. Native image coordinates and matching class metadata are mandatory.
- `bytetrack_adapter.py`: packages xyxy detections as NumPy `xywh` CENTER coordinates for the Ultralytics backend, including boolean slicing. Updates the injected backend on every frame, even an empty frame, and consumes axis-aligned Nx8 output. Tests use ONLY a stub, not ByteTrack. A new adapter/backend is required per sequence. Do not reset each frame or skip empty frames. `create_ultralytics_adapter` imports/creates the real backend only with explicit future execution authorization; its constructor feature-check supports the old optional frame_rate and current args-only signature.
- `io.py`: frame-map validation, exclusive JSONL output, serialization and approved snapshot verification.

`configs/tracking/bytetrack_pilot.yaml` is an untuned starting configuration, not a measured recommendation. Model path/device remain null until Work 5.4 selects a checkpoint. Its detector threshold must not exceed track_low_thresh if the ByteTrack low-score association stage is to receive those candidates. `execution_enabled` is false; no runner is provided/executed in Work 5.3.

Before actual Work 5.4: verify the integration snapshot, select and hash a vehicle checkpoint, verify its names metadata (or a unified model's original names), pin/check the Ultralytics backend version, and authorize execution explicitly. Runtime backend compatibility has NOT been smoke-tested with a real tracker here. Kalman/backend boxes outside image bounds cause a contract error rather than undocumented clipping. Keep GT classes null; predicted classes are not authoritative GT. Preserve the PILOT_DEV role, existing gaps and camera-exposure limitations; do not claim final unbiased benchmark metrics.

Contracts are also provided as JSON Schema files under `configs/tracking/`. Runtime validation additionally enforces frame offset, image bounds, class metadata correspondence and unique track IDs.

Tests: `python -m pytest tests/tracking -q -p no:cacheprovider`. They write only to pytest temporary directories and read the real pilot/provenance; no inference, model download, actual tracking, GT edits or tuning.

Primary interface references: [Ultralytics Results](https://docs.ultralytics.com/reference/engine/results/), [ByteTrack API](https://docs.ultralytics.com/reference/trackers/byte_tracker/), [upstream tracker configuration](https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/trackers/bytetrack.yaml).
