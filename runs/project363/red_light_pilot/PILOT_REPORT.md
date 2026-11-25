# Red-light pilot (M2) - 16/11/2026

Status: PILOT, predictions checked by the project owner by watching qa_overlay.mp4 of each camera.
Models: YOLOv8s @960 + ByteTrack tuned (configs/tracking/bytetrack_v8s_imgsz960_tuned_20261114.yaml), light model A @960.
Rules: src/rules/red_light.py; signal: src/signal/phase_voting.py; SceneConfigs: configs/scene_configs (mapping review: mapping_review_20261115.json).
Sources: existing iHanoi phone screen recordings (test only, timebase_verified=false), sampled at 10 fps.

| Camera | Approaches | Red crossing (pred) | No red crossing | Insufficient evidence | Scope uncertain | Owner: real violations |
|---|---|---|---|---|---|---|
| C040_survey_v2 | 3 | 0 | 2 | 1 | 0 | 0 |
| C042_survey_v5 | 3 | 0 | 0 | 1 | 2 | 0 |
| C057_retry_v2 | 20 | 0 | 6 | 12 | 2 | 0 |
| C069_retry_v2 | 23 | 0 | 17 | 6 | 0 | 0 |
| C069_trial01_v2 | 30 | 0 | 19 | 11 | 0 | 0 |
| C148_guarded_v2 | 2 | 0 | 2 | 0 | 0 | 0 |

False alerts: 0 (owner found no violation in any of the 6 clips). Event recall: NOT MEASURABLE (no positive event in the data).

Fixed during the pilot (each found by the owner's review):
- C040 tracks 61/70: a vehicle stopped at the line whose box nudged over it, and one that nosed over and was lost, were alerted. A crossing now has to be confirmed (reference point >= max(8 px, min(0.5 bbox height, 40 px)) past the line); a pixel-unit bug in the line distance was fixed.
- C069_trial01 track 65: crossed on yellow; the light model read the orange Vietnamese yellow as red. The first 3.5 s of red that directly follows green (no yellow observed) are now 'uncertain_colour'.

Limits: insufficient_evidence is high mainly because the phone recordings are interrupted by app screens and tracks are cut there (not stitched, by design). The reference point (bbox bottom centre) is not calibrated per camera. Yellow detection of the light model is weak.

## VN TV news segments (16/11/2026)

Source: data/vn_news_violation_clips_20261116 (one VN news report, 9 segments with a usable fixed view, 25 fps, sampled at 10 fps). SceneConfigs news_seg*_v2 drawn by the owner (copied lanes in seg00/04/07/10 confirmed by the owner). GT: news_ground_truth_20261116.json.

| Segment | Approaches | Red crossing (pred) | No red crossing | Insufficient | Out of scope | Owner |
|---|---|---|---|---|---|---|
| news_seg00 | 2 | 0 | 0 | 1 | 1 | not checked |
| news_seg02 | 2 | 0 | 1 | 1 | 0 | not checked |
| news_seg04 | 0 | 0 | 0 | 0 | 0 | not checked |
| news_seg05 | 7 | 0 | 5 | 2 | 0 | not checked |
| news_seg07 | 6 | 2 (tracks 29, 61) | 3 | 1 | 0 | 1 real violation (both alerts) |
| news_seg08 | 0 | 0 | 0 | 0 | 0 | not checked |
| news_seg10 | 7 | 0 | 5 | 2 | 0 | not checked |
| news_seg11 | 1 | 0 | 1 | 0 | 0 | not checked |
| news_seg13 | 5 | 0 | 2 | 3 | 0 | not checked |

- First positive event: news_seg07 red car (plate overlay 26A 206.XX) crosses on red at ~15.4 s. Track 61 is the news replay of the same scene (~33 s), so the 2 alerts count as 1 true event (spec 19.2), 0 false alerts.
- Owner review of all 9 segments: 8 real violation events (seg02 and seg04 are the same event). Event precision 1/1, **event recall 1/8 = 12.5%** (0 false alerts).
- Causes of the 7 misses (details in news_ground_truth_20261116.json):
  1. Stop line / approach ROI drawn at the bottom edge, 50-70 px below the real stop line (before the crosswalk), and the thin ROI strip leaves a gap under the line: seg08, seg10, seg13 tracks never enter the ROI.
  2. Direction checked on the whole track: seg00 car crossed straight then curved right -> out_of_scope.
  3. Track fragmentation at the line (ID switch / short track): seg00 15->17, seg02 19, seg10 17->19, seg13 31.
  4. Signal not observed at crossing time: seg05 (no head detected after 10 s), seg13 (head too small).
  5. Arrow signal: seg08 head = red lamp + green right-turn arrow; the light model reads it as green.
  6. Detector misses (seg08 bike on the crosswalk, seg11 left-turning bike, seg13 bike 31).
- Few vehicles are assessed because every news approach ROI is a thin strip at the bottom edge (stop lines at y ~316-327 of 360).

### Fixes after the owner review (16/11/2026, owner agreed 1+2+3)
- Direction is checked only up to 60 px past the stop line (RuleSettings.direction_past_px): crossing straight and curving away later stays in scope.
- Track stitching at the stop line (src/rules/track_stitch.py): ID switch joined when the old track ends <= 80 px from a stop line, the new one starts <= 0.6 s later
  (no quality gap in between) within max(25 px, 0.5 x height) of the motion prediction, same class family, similar height. Joins are listed in manifest.json (track_joins).
- Signal heads with a green turn arrow: SceneConfig signal flag has_turn_arrow; a not-red reading from such a head is insufficient evidence.
- scene_config_tool.py: buttons "Xóa hết làn" and "Đèn có mũi tên", --cameras filter, and a warning after each lane when the ROI does not reach the stop line or is thinner than 60 px (src.scene.config.approach_warnings).
- scripts/reevaluate_red_light.py: rules-only rerun of a finished run with a new SceneConfig version (same signal ROIs), output in rules_rerun/.

Rules-only rerun with the unchanged v2 configs (rules_rerun/): news red alerts seg00 track 17, seg02 track 19 (+20 joined), seg07 tracks 29/61, seg10 track 17 (+19 joined)
-> event recall 4/8, 0 false alerts. iHanoi 6 cameras: still 0 alerts (joins: C057 38->40, C069_trial01 83->105 and 61->98, C148 2->6).
Next: owner redraws real stop lines and full-depth approach ROIs for the news segments, marks arrow heads; then rerun.

### Rerun with owner-redrawn configs news_seg*_v3 (rules_rerun/news_seg*_v3, with qa_overlay.mp4)
The owner's stop lines are the real solid white lines (near the image bottom, a gap before the crosswalk): my earlier claim that they were drawn too low was wrong.
The thin-ROI warning is now skipped when the ROI reaches the image edge.

| Segment | Red alerts | Owner GT |
|---|---|---|
| seg00 | track 17 @8.88 s | violation (caught) |
| seg02 | track 19 (+20) @10.24 s | violation (caught) |
| seg04 | track 10 @3.28 s (left motorcycle, light yellow 2.4 s -> red 3.0 s) | owner confirmed: same violation as seg02 (caught) |
| seg05 | - | missed: car 34, no signal observation after 10 s (light model) |
| seg07 | tracks 29/61 (replay) | violation (caught, 1 event) |
| seg08 | - | missed: motorcycle first detected already past the line (detector), S1 head with green arrow not flagged |
| seg10 | track 17 (+19) @8.64 s | violation (caught) |
| seg11 | - | missed: left-turning motorcycle not tracked |
| seg13 | - | missed: motorcycle 31 first detected past the line; S1 never observed |

Event recall 4/8 (50%), false alerts 0 (owner confirmed). Remaining misses are detector / light-model limits, not rules.

news_seg08:v4: S1 flagged has_turn_arrow (owner: green arrow under the red lamp; the light model shows 'green' while the lamp is red). Rerun: no change (vehicle never observed before the line).

### Model diagnosis of the 4 remaining misses (16/11/2026, models re-run at low confidence)
- seg05: NOT a model error. S1 is a vertical 3-lamp head; the drawn ROI (y 28-52) covers only yellow+green, the red lamp (y 8-26) is outside. Model reads red 0.33-0.56 at 14-20 s. Fix: redraw the signal ROI over the whole head.
- seg13 light: model gives no detection at all (even conf 0.01) on the small VN head (red lamp + green arrow, ~9x31 px). Model error; head also has a turn arrow.
- seg13 motorcycle 31 (rider with raincoat/cargo, rear view): detected at conf 0.03-0.26 on the crosswalk, only 0.56 later -> below new_track_thresh 0.4. Model confidence error.
- seg08 motorcycle (rear view, in tree shadow): conf 0.04-0.18 or missed. Model error.
- seg11 left-turning motorcycle: not diagnosed yet.

### Step A (16/11/2026): news_seg05_v4 (S1 ROI over the whole head), news_seg13_v4 (S1 has_turn_arrow)
- seg05: red now observed from 13.12 s; car 34 crosses at 16.0 s but falls in the missing-yellow guard (13.12-16.62 s) -> insufficient.
  Owner (watching seg05 v3): the light did turn yellow, label unknown. Check: dim orange yellow lamp visible 10.4-12.8 s (~3 s);
  light model A gives NO detection even at conf 0.005 (news_seg05_yellow_phase_check.jpg). Yellow detection of VN lamps is a model failure.
- seg13: S1 still never detected (model).

### Light reader k1b (18/11/2026): model A box in each signal ROI re-read by the colour classifier (weights/light_colour_cls_20261117)
Runs: <config>__k1b (same configs: iHanoi v2/v5, news v3/v4). `--light-reader a` reproduces the previous behaviour.
- iHanoi 6 cameras: still 0 red alerts (owner GT: 0 violations).
- News: red alerts seg00 t17, seg02 t19, seg04 t10, seg07 t29, seg10 t17 -> event recall 4/8, 0 false alerts (unchanged).
  The seg07 replay (track 61) is now 'crossing not inside a single observed interval' (same event, already counted).
- Signal now right where A was wrong: seg08 S1 red 1.0-11.3 s (A: green, arrow); seg13 S1 red observed (A: never detected).
  The seg08/seg13 violations are still missed because the VEHICLE model detects those motorcycles only after the line.
- seg05 car 34: dim yellow 10.1-13.1 s still unread -> missing-yellow guard -> insufficient (no false red).
Next: retrain the vehicle model with the owner-QA'd VN set (data/vn_pool_qa_vehicle_20261117) to address the remaining misses.

## 18/11/2026: vehicle fine-tune not adopted + low-confidence probe on seg08/seg13

- VN vehicle fine-tune (Kaggle2 yolov8s-vehicle-vnpool-20261118): worse on both vals (Vehicle363 val2000 mAP50-95 0.719 -> 0.700,
  VN val 0.751 -> 0.597). Not adopted; see runs/project363/yolov8s_vehicle_vnpool_20261118/STATUS.md.
- Probe: v8s @960 at conf 0.05 on every 2nd frame, detections whose bottom-centre is inside the approach ROI.
  - seg08: no motorcycle at any confidence in the ROI. The violating bike enters from the bottom edge of the frame,
    only its upper half visible and motion-blurred (t 4.0-4.4 s), and is past the line by 4.8 s.
  - seg13: the bikes waiting/crossing at the bottom edge (rear view, raincoat, cut by the frame edge) get at most 0.055-0.075.
  - Conclusion: not a threshold/tracker problem. The approach strip is only ~30-35 px at the image's bottom edge, so the
    vehicle must be recognised while mostly out of frame. Lowering thresholds would not recover these events.

## 18/11/2026: lamp-position light reader (k1bpos) -> news recall 5/8, 0 false alerts

- Owner insight: on a 3-lamp head yellow is always the middle lamp. src/signal/lamp_position.py reads which third of the
  head ROI is lit (needs SceneConfig `lamp_layout` = vertical3 | horizontal3; only plain 3-lamp heads, no arrows/countdown).
  Pipeline default `--light-reader k1bpos`: position reading where a layout is set, otherwise k1b.
- Light-phase eval truth fixed (owner confirmed): C069_retry:H1 t=23-25 red -> yellow (middle lamp lit orange).
  Voted accuracy / wrong s: A 95.2%/8, k1b 94.7%/9, k1bpos 95.2%/8.
- Configs: 13 new versions with lamp_layout; seg02 v5 / seg04 v5 drop S2's layout (green ARROW in its bottom lamp was read as
  green during red in the first run, which lost both events).
- Results (`<config>__k1bpos`): iHanoi 6 cameras 0 alerts. News alerts: seg00 t17 8.88 s, seg02 t19 10.24 s, seg04 t10 3.28 s,
  seg05 t34 16.0 s (NEW: dim yellow 10.16-13.04 read, so the missing-yellow guard no longer blocks), seg07 t29 15.04 s + replay t61,
  seg10 t17 8.64 s. Event recall 5/8, 0 false alerts. Still missed: seg08, seg13 (bikes not detected before the line), seg11 (left turn).
