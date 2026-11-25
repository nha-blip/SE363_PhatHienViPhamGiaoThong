# Dataset Audit Report — Project 363

## Scope and method

Audit date: 2026-10-03. The audit inspected the nine requested directories under `data/` and the Bellevue ZIP archives at that same level. It performed only file enumeration, metadata reading, and annotation-schema parsing. No dataset files were modified, renamed, moved, converted, extracted, or used for training.

`UNKNOWN - NEED MANUAL REVIEW` is used where the dataset itself does not document a semantic mapping. Video resolution/FPS for datasets other than AICity is also marked unknown because no embedded, machine-readable probe output is present in the repository and no media-probe utility is available in the audit environment.

## Executive summary

| Dataset | Images | Videos | BBox | Classes | Track ID | Traffic Light State | Stop Line/ROI | Violation | Recommended Use | Need CVAT |
|---|---:|---:|---|---|---|---|---|---|---|---|
| `dataset_train_rgb` | 5,093 PNG | 0 | Yes, 10,756 | 13 TL labels | No | Yes | NOT AVAILABLE | NOT AVAILABLE | YOLO Traffic Light | NO LABELING NEEDED |
| `Vehicle_Tracking` | 0 extracted | 4 MP4 | Yes | NOT AVAILABLE | Yes | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | ByteTrack evaluation | PARTIAL LABELING |
| `AICity21_Track1_Vehicle_Counting` | 20 JPG | 32 MP4 | No per-object bbox | Car, Truck in counting GT | No | NOT AVAILABLE | ROI available; stop line NOT AVAILABLE | Counting GT only; no violation label | Heuristic / ROI / trajectory design | PARTIAL LABELING |
| `Dataset_Traffic_Evaluation` | 0 | 24 MP4 | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | Candidate end-to-end / VLM source after labels | FULL LABELING |
| `imptc_samples` | 742 PNG, 5 JPG | 8 MP4 | 3D cuboid; no 2D image bbox field | car, truck, person, bicycle, motorcycle, scooter | Yes | Timestamped signal codes; codebook UNKNOWN | NOT AVAILABLE | NOT AVAILABLE | Trajectory / temporal heuristic research | PARTIAL LABELING |
| `Bellevue_116th_NE12th` | 0 | 21 MP4 | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | End-to-end / VLM source after labels | FULL LABELING |
| `Bellevue_150th_Eastgate` | 0 | 7 MP4 | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | End-to-end / VLM source after labels | FULL LABELING |
| `Bellevue_150th_Newport` | 0 | 24 MP4 | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | End-to-end / VLM source after labels | FULL LABELING |
| `Bellevue_150th_SE38th` | 0 | 18 MP4 | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | End-to-end / VLM source after labels | FULL LABELING |

## Unextracted Bellevue archives

No extraction was performed. No existing extracted directory has the `Bellevue_NE8th` name, so correspondence with the existing Bellevue folders cannot be established from names alone.

| Archive | Size (bytes) | Size (GiB) |
|---|---:|---:|
| `Bellevue_Bellevue_NE8th-20261002T145509Z-1-001.zip` | 1,844,061,232 | 1.72 |
| `Bellevue_Bellevue_NE8th-20261002T145509Z-1-002.zip` | 1,836,300,414 | 1.71 |
| `Bellevue_Bellevue_NE8th-20261002T145509Z-1-003.zip` | 1,901,037,079 | 1.77 |
| `Bellevue_Bellevue_NE8th-20261002T145509Z-1-004.zip` | 2,092,820,512 | 1.95 |
| `Bellevue_Bellevue_NE8th-20261002T145509Z-1-005.zip` | 2,106,750,798 | 1.96 |
| `Bellevue_Bellevue_NE8th-20261002T145509Z-1-006.zip` | 2,046,642,183 | 1.91 |
| `Bellevue_Bellevue_NE8th-20261002T145509Z-1-007.zip` | 2,068,022,432 | 1.93 |
| `Bellevue_Bellevue_NE8th-20261002T145509Z-1-008.zip` | 1,356,028,287 | 1.26 |

Total: 15,251,662,937 bytes (14.20 GiB).

---

## Dataset: `dataset_train_rgb` (Bosch)

### A. Structure

```text
dataset_train_rgb/
├── train.yaml
├── non-commercial_license.docx
└── rgb/
    └── train/
        └── <7 recorded-sequence directories>/
            └── *.png
```

### B. Media

- 5,093 PNG images; 0 videos; total dataset size 6,677.06 MB (image files: 6,675.39 MB).
- Resolution distribution: `1280x720`: 5,093 images.
- The images are grouped by recorded sequence; there is only a `train` partition.

### C. Annotation

- One YAML annotation file (`train.yaml`, 1.66 MB), one record per image.
- Schema verified from content: `path`, `boxes[]`, and for each box `label`, `occluded`, `x_min`, `y_min`, `x_max`, `y_max`.
- Available: class and 2D bbox. Not available: track ID, frame ID as an explicit field, trajectory, ROI, stop line, timestamp, violation/event label.
- The `path` image references resolve to the `rgb/train/` layout. The YAML has 5,093 records, 3,153 records with at least one box, and 10,756 boxes.

### D. Classes

| Original class | Boxes | Images containing class |
|---|---:|---:|
| Green | 5,207 | 2,079 |
| Red | 3,057 | 1,527 |
| RedLeft | 1,092 | 861 |
| `'off'` | 726 | 507 |
| Yellow | 444 | 291 |
| GreenLeft | 178 | 137 |
| GreenStraight | 20 | 18 |
| GreenRight | 13 | 11 |
| RedStraight | 9 | 9 |
| RedRight | 5 | 5 |
| GreenStraightRight | 3 | 3 |
| GreenStraightLeft | 1 | 1 |
| RedStraightLeft | 1 | 1 |

The distribution is strongly imbalanced: `Green` has 5,207 boxes while each of `GreenStraightLeft` and `RedStraightLeft` has one.

### Bosch V1 mapping proposal (do not convert in this audit)

| Target V1 state | Original labels included | Boxes after mapping |
|---|---|---:|
| red | `Red*` | 4,164 |
| yellow | `Yellow*` | 444 |
| green | `Green*` | 5,422 |
| off | literal `'off'` | 726 |

The proposed mapping preserves all 10,756 boxes. Directional information is intentionally collapsed; retain the original YAML for traceability.

### E–G. Tracking, traffic light, violation

- Tracking / trajectory: NOT AVAILABLE.
- Traffic light: available as bbox plus original state label; V1 states are available through the stated mapping.
- Stop line, intersection ROI, vehicle crossing, time-aligned vehicle/light state, and violation labels: NOT AVAILABLE.

### Pipeline and CVAT decision

- Suitable for: **1. YOLO Traffic Light Detection** only.
- `NO LABELING NEEDED` for traffic-light detection. It is not suitable as-is for violation evaluation.
- Leakage: consecutive images from the same recording sequences are present in a single train directory. Split by recording-sequence directory, never randomly by image.

---

## Dataset: `Vehicle_Tracking`

### A. Structure

```text
Vehicle_Tracking/
├── AICC22-Custom/
│   ├── AICC22-Custom_*video.mp4
│   ├── AICC22-Custom_GroundTruth.json
│   └── AICC22-Custom_GroundTruth.txt
└── VNTraffic/
    ├── VNTraffic_*video.mp4
    ├── VNTraffic_GroundTruth.json
    └── VNTraffic_GroundTruth.txt
```

### B. Media

- 4 MP4 videos; no extracted images; total size 39.33 MB (video files: 37.54 MB).
- Resolution and FPS: `UNKNOWN - NEED MANUAL REVIEW`.

### C–E. Annotation and tracking

- Two JSON and two TXT files are available.
- JSON schema: root object keys are persistent track IDs; each value contains aligned `frames[]` and `boxes[]`. A box is `[x, y, width, height]`.
- Verified examples: AICC22 has 25 track IDs and VNTraffic has 205; the same object ID has multiple sequential frames, so tracking and image-plane trajectories are available.
- TXT uses ten comma-separated MOT-style fields; observed records have `frame,id,x,y,w,h,confidence,-1,-1,-1`. It contains 1,185 AICC22 and 16,036 VNTraffic detection records.
- The JSON frame paths reference an external `./train/.../img1/*.jpg` layout that is not present in this directory. The videos are present, but the source frame files referenced in JSON should be manually reconciled before direct frame-level evaluation.
- Class identity is NOT AVAILABLE in the parsed annotation: the TXT class field is `-1` and JSON has no class field.

### F–G. Traffic light and violation

Traffic-light annotations/states, stop lines, ROIs, timestamps, and violation/event labels are NOT AVAILABLE.

### Pipeline and CVAT decision

- Suitable for: **3. ByteTrack evaluation** and limited **4. Heuristic / trajectory** experiments.
- `PARTIAL LABELING`: keep existing bbox/track IDs; only annotate an ROI/stop line and time-aligned traffic-light state/event if a selected scene visibly supports the violation use case.
- Leakage: each ground-truth set is a consecutive frame sequence. Keep every source video wholly in one split.

---

## Dataset: `AICity21_Track1_Vehicle_Counting`

### A. Structure

```text
AICity21_Track1_Vehicle_Counting/
└── AIC21_Track1_Vehicle_Counting/
    ├── Dataset_A/                 # 31 benchmark videos and stats/class lists
    ├── counting_gt_sample/        # one annotated example video and CSV
    ├── ROIs/                      # 20 pixel-polygon text files
    ├── movement_description/      # 20 movement-of-interest text files
    └── screen_shot_with_roi_and_movement/  # 20 JPGs
```

### B. Media

- 32 MP4 videos (31 in `Dataset_A` plus one counting-GT sample video) and 20 JPG screenshots; total size 5,568.87 MB.
- JPG resolution distribution: `1280x960`: 20.
- `Dataset_A/datasetA_vid_stats.txt` supplies FPS: 27 videos at 10 FPS, 2 at 8 FPS, and 2 at 15 FPS. It also supplies frame counts per video.
- Video resolution: `UNKNOWN - NEED MANUAL REVIEW`.

### C–G. Annotation, classes, ROI, violation

- 44 TXT files, one CSV, and 20 ROI screenshots are present.
- `ROIs/cam_*.txt` contains polygon vertices in pixel coordinates. It is an ROI, not an explicitly identified stop line.
- `movement_description/cam_*.txt` describes movements of interest in words.
- The sample CSV schema is `video_clip_id,frame_id,movement_id,vehicle_class_id`; class list maps `1` to Car and `2` to Truck. It is a movement/counting event sample, not per-object bbox or track ground truth.
- Per-object bbox, persistent track ID, trajectory, traffic-light state, stop line, and violation/non-violation labels are NOT AVAILABLE.

### Pipeline and CVAT decision

- Suitable for: **4. Heuristic / trajectory** (ROI/MOI/counting logic) and implementation evaluation of ROI handling. It is not ground truth for ByteTrack or violation classification.
- `PARTIAL LABELING`: do not relabel provided ROI/MOI/counting assets. For a selected camera, annotate stop line, traffic-light bbox/state timeline, vehicle tracks, and violation events if it is to become an enforcement-evaluation scene.
- Leakage: videos include camera and weather/time variants (e.g., dawn/rain/snow). Split by camera and recording condition; do not split consecutive or variant clips across train/validation.

---

## Dataset: `Dataset_Traffic_Evaluation`

### A. Structure

```text
Dataset_Traffic_Evaluation/
└── *.mp4
```

### B–G. Media and annotation

- 24 MP4 videos, 0 images, total 8,187.62 MB.
- Resolution/FPS: `UNKNOWN - NEED MANUAL REVIEW`.
- No `.txt`, `.json`, `.xml`, `.yaml`, `.yml`, `.csv`, `.mat`, or `.pkl` annotation files are present.
- Therefore bbox, classes, tracks, trajectories, traffic-light states, ROI/stop lines, timestamps, and violation labels are NOT AVAILABLE.

### Pipeline and CVAT decision

- Potentially suitable for **5. Violation detection**, **6. VLM evaluation**, and **7. End-to-end evaluation** only after scene review and annotation.
- `FULL LABELING`: annotate traffic-light bbox and state timeline, vehicle bbox and track ID, intersection ROI/stop line, and normal/violation event labels. First confirm that each selected scene contains a visible signalized intersection.
- Leakage: filename timestamps indicate separate clips. Allocate complete videos, not sampled frames, to a split.

---

## Dataset: `imptc_samples`

### A. Structure

```text
imptc_samples/
├── samples_overview.csv
├── samples_vru_tracks.jpg
└── <4 sequence directories>/
    ├── overview.csv
    ├── sequence_overview_video.mp4
    ├── sequence_tracks_video.mp4
    ├── vehicle_trackinfo.csv / vru_trackinfo.csv
    ├── context/                   # timestamp, traffic signals, weather JSON
    ├── vehicles/<track-id>/        # track.json and summary PNG
    └── vrus/<track-id>/            # track.json and summary PNG
```

### B. Media

- 8 MP4 videos, 742 PNG images, 5 JPG images; total 463.24 MB.
- PNG resolution distribution: `1920x1080`: 371; `1672x1672`: 371. JPG resolution distribution was not computed as a full population: `UNKNOWN - NEED MANUAL REVIEW`.
- Video resolution/FPS: `UNKNOWN - NEED MANUAL REVIEW`.

### C–E. Annotation, classes, tracking, trajectory

- 383 JSON and 13 CSV files are present. Each vehicle/VRU has a `track.json` with `overview` and timestamp-keyed `track_data`.
- Verified track-point fields: `ts`, `coordinates`, `cuboid`, `velocity`, `ground_type`, `z_offset`, `source_type`, `class_prob`, `status`, and `volume`.
- This is persistent 3D trajectory ground truth. `cuboid` is a 3D geometry field; an image-plane 2D bbox field was not found.
- 371 annotated tracks: 290 vehicles and 81 VRUs. Class distribution by track: car 282, truck 8, person 61, bicycle 12, motorcycle 5, scooter 3.
- Sequence track counts: `0000`: 58 vehicles/5 VRUs; `0001`: 72/30; `0002`: 87/29; `0003`: 73/17.

### F–G. Traffic light and violation

- Each sequence has `context/traffic_light_signals.json` keyed by timestamp and signal IDs (`k*`, `f*`, `b*`) with numeric values.
- The mapping from signal IDs/numeric codes to red/yellow/green/off is not included in the inspected files: `UNKNOWN - NEED MANUAL REVIEW`. Do not infer state names from values.
- Master timestamp synchronization is available through `master_timestamp_sync.json`.
- Stop line, intersection ROI, vehicle-crossing event, and violation/non-violation labels are NOT AVAILABLE.

### Pipeline and CVAT decision

- Suitable for: **4. Heuristic / trajectory** and temporal alignment research. It can support signal/trajectory logic only after obtaining a signal-codebook and camera-calibration interpretation.
- `PARTIAL LABELING`: do not relabel tracks. Add camera-view 2D bbox only if 2D detector evaluation is required; add stop line/ROI and violation events for a selected sequence. Obtain or manually validate the traffic-signal codebook before labeling state names.
- Leakage: all tracks and derived PNGs within a sequence share a time interval. Split exclusively by sequence (`0000`–`0003`); do not split track points, track summaries, or frames from one sequence.

---

## Dataset: Bellevue video collections

The four directories have the same flat structure: one directory containing MP4 files and no sidecar annotation files. Their media totals are below.

| Dataset | Structure | Videos | Total video size | Annotation finding | CVAT decision |
|---|---|---:|---:|---|---|
| `Bellevue_116th_NE12th` | `*.mp4` | 21 | 6,358.65 MB | No supported annotation files | FULL LABELING |
| `Bellevue_150th_Eastgate` | `*.mp4` | 7 | 3,874.19 MB | No supported annotation files | FULL LABELING |
| `Bellevue_150th_Newport` | `*.mp4` | 24 | 16,164.00 MB | No supported annotation files | FULL LABELING |
| `Bellevue_150th_SE38th` | `*.mp4` | 18 | 7,479.55 MB | No supported annotation files | FULL LABELING |

For every Bellevue collection: images 0; video format MP4; video resolution/FPS `UNKNOWN - NEED MANUAL REVIEW`; bbox/classes/track IDs/trajectories/traffic-light states/ROI/stop line/timestamps/violation labels are NOT AVAILABLE.

They are potential **5. Violation detection**, **6. VLM evaluation**, and **7. End-to-end evaluation** sources once manually confirmed to depict the intended traffic events. They are not current training/evaluation ground truth. Required labels for a chosen clip are vehicle bbox+track ID, traffic-light bbox+state timeline, stop line or intersection ROI, crossing state, and normal/violation event label. All 70 videos are unlabeled annotation candidates; exact frame counts must be established only after choosing an annotation sampling policy, not by extracting the full collection during this audit.

Leakage risk is high for clips from a common camera and contiguous timestamps (for example, hourly-looking filenames). Split by camera/location and whole time block; never split frames from a clip between partitions.

---

## Pipeline mapping

## Dataset for YOLO Traffic Light

- Use `dataset_train_rgb` with the four-state mapping proposed above. Keep directional labels in the original source and document the collapsed V1 mapping.
- `imptc_samples` has time-series signal values but lacks a verified red/yellow/green/off codebook and 2D image boxes; it is not ready for YOLO traffic-light training.

## Dataset for YOLO Vehicle

- No dataset in the audited set provides a ready, verified vehicle 2D-bbox training corpus with class labels.
- `Vehicle_Tracking` supplies 2D bbox but no class; it may be used for class-agnostic vehicle detector/track evaluation after reconciling referenced frame paths.
- Bellevue and `Dataset_Traffic_Evaluation` require annotation first.

## Dataset for ByteTrack

- Primary: `Vehicle_Tracking`, with 230 persistent identities across the two JSON files and 17,221 MOT-style detection records.
- `imptc_samples` has trajectory IDs but 3D geometry rather than verified camera 2D boxes; use for trajectory logic, not direct ByteTrack metrics.

## Dataset for Heuristic

- `imptc_samples`: strongest temporal/trajectory asset; needs signal-codebook verification and enforcement geometry.
- `AICity21_Track1_Vehicle_Counting`: ROI/MOI and counting logic; lacks per-object trajectories and signal state.

## Dataset for VLM

- Candidate unlabeled sources: `Dataset_Traffic_Evaluation` and all Bellevue collections, after selecting and annotating evidence clips.
- No audited dataset currently has verified violation/non-violation evidence labels, so VLM evaluation ground truth is absent.

## Dataset not ready for current scope

- `AICity21_Track1_Vehicle_Counting` is not useful for direct traffic-light detection, ByteTrack ground-truth evaluation, or violation classification without additional labels.
- No dataset is categorically discarded; the video-only sets are retained as potential annotation sources rather than current ground truth.

## Gap analysis

### What Project 363 currently has

- **Traffic-light detection data:** 5,093 images / 10,756 labelled Bosch traffic-light boxes; four-state mapping counts are red 4,164, yellow 444, green 5,422, off 726.
- **Vehicle detection data:** vehicle boxes in `Vehicle_Tracking`, but no class labels and referenced source images are absent from its local layout.
- **Tracking data:** `Vehicle_Tracking` 2D MOT-style tracks and `imptc_samples` 371 persistent 3D tracks.
- **Temporal data:** IMPTC timestamps, trajectories, velocity, synchronization metadata, and raw traffic-signal codes; AICity frame IDs, ROI, MOI, and counting sample events.
- **Violation data:** none with verified violation/non-violation ground truth.
- **VLM evidence data:** raw candidate videos only; no evidence labels.

### Missing data

1. A verified vehicle 2D detection dataset with vehicle classes suitable for YOLO training/evaluation.
2. A traffic-light state timeline synchronized with camera frames and with an explicit red/yellow/green/off codebook in the same enforcement scene.
3. Stop line and intersection ROI geometry for scenes used for violations.
4. Crossing/temporal event labels and normal versus violation labels.
5. Curated VLM evidence clips and adjudicated ground truth.
6. A leakage-safe split manifest by camera/video/time sequence.

### Data to annotate manually

| Dataset | Existing labels to retain | Add in CVAT / review | Annotation unit / estimate |
|---|---|---|---|
| `dataset_train_rgb` | TL bbox and class | Nothing for traffic-light YOLO | 0 additional labels |
| `Vehicle_Tracking` | 2D bbox and track ID | ROI/stop line, traffic-light state timeline, crossing/event only for selected scenes | 2 source videos; frame total UNKNOWN |
| `AICity21_Track1_Vehicle_Counting` | ROI, MOI, sample counts | stop line, TL state, object tracks, violation event for selected cameras | 20 camera scenes; select a pilot camera first |
| `imptc_samples` | 3D tracks and timestamps | signal-codebook validation; ROI/stop line/event labels; 2D boxes only if needed | 4 sequences / 371 tracks; select 1 pilot sequence |
| `Dataset_Traffic_Evaluation` | None | vehicle tracks, TL state, ROI/stop line, normal/violation event | 24 unlabeled videos |
| Bellevue collections | None | vehicle tracks, TL state, ROI/stop line, normal/violation event | 70 unlabeled videos |

For frame-level volume, no exact estimate is defensible until the team chooses: (1) target camera/scene, (2) annotation cadence, and (3) event sampling rate. Extracting all videos to count frames is explicitly outside this audit. A practical next phase is one pilot video per candidate source, labelled at a stated cadence, followed by a cost estimate from observed frames/events.

## Split and leakage rules

1. Bosch: group split by recorded-sequence directory.
2. Vehicle tracking: group split by video/source set.
3. AICity: group split by camera, then keep weather/time variants from a camera in one partition where possible.
4. IMPTC: group split by the four sequence directories.
5. Bellevue and evaluation videos: group split by camera/location and chronological block.

Never randomly split frames or derived images from the same video between train, validation, and test.
