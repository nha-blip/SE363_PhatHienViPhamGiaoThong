# VN YouTube training pool (16/11/2026): labeling rules

Purpose: training data for the vehicle model (VN rear-view motorcycles: raincoats, cargo, shade) and the light model
(VN round lamps: orange yellow, small heads, heads with turn arrows / countdowns). Owner approved the source on 16/11/2026.

Source and split
- 16 public YouTube news videos (raw/*.mp4 + *.info.json keep URL, channel, upload date). YouTube standard licence:
  used for course research only, never redistributed (the Kaggle dataset stays private).
- The ĐTV channel is excluded, because the test news report (data/vn_news_violation_clips_20261116) comes from it and
  camera clips are reused across reports.
- Split by source VIDEO (all frames of one video go to the same split). Similar shots inside one video are
  near duplicates, so keep at most a few frames per shot.

Light classes (unchanged, spec §9.4): red, yellow, green, off. Decision 16/11 (option b, no new class):
- Label only ROUND lamps of vehicle signal heads, one box per head around the housing, class = the lit round lamp.
- A turn ARROW lamp (green/red/yellow arrow) is NOT a traffic-light target: do not box it, and do not include it in the
  round-lamp box. When a head has only an arrow lit and its round lamps are dark, label the head `off` only if the
  round lamps are clearly visible and dark; otherwise leave it out.
- Countdown displays and pedestrian signals are not targets.
- Yellow in VN is orange: label by lamp position and colour. When unsure between yellow and red, look at neighbouring
  frames; if still unsure, mark the frame for removal instead of guessing.
- Tiny heads: label every head whose lit lamp you can identify; skip heads where the colour cannot be told.

Vehicle classes (unchanged): car, motorcycle, bus, truck. Box the visible part; riders are part of the motorcycle box.
News overlays (presenter, plate boxes, logos) hide parts of the view: do not label behind them.

Update 17/11/2026 (owner): VN signal types — small roads: 3-colour round lamps + countdown; national/major roads: ARROW
lamps (straight / left / right) with countdown; special motorcycle-only lamps (straight, right turn). Decision (a): step 1 trains
the 4-class ROUND-lamp model now; arrows and motorcycle lamps become a separate step 2 (classes by shape, SceneConfig lane
movement + vehicle type, rule lookup by movement; spec §9.4 to be changed). The owner labels arrows with their colour;
each kept box carries `shape` = round | arrow | countdown (added after QA). Step 1 export uses only shape == round.

Vehicle class rule (owner, 17/11/2026): 16-seat minibuses and vans -> truck (same as van -> truck in Vehicle363/MIO);
bus = city buses and large coaches only. No new class for now (only needed for vehicle-type rules later).

Vehicle taxonomy change (owner, 21/11/2026): the spec needs the vehicle TYPE, so the detector moves to 6 classes:
car (sedan/SUV/taxi), motorcycle, pickup (bán tải), minibus (<=16 seats AND vans, including closed cargo vans: same body
shape), bus (city buses and large coaches), truck (trucks, tractor units, containers). Replaces the 17/11 rule
"16-seat/van -> truck". Spec §9.4 and every 4-class consumer must be updated before training. Missing classes (pickup,
minibus, VN examples) are to be sourced from public datasets first.
Owner 21/11/2026: MPV / minivan / 7-seat (Innova, Xpander, Sienna-like, car-like body) -> car. minibus = box-shaped bodies only
(Hiace/Transit/Solati-like <=16 seats, micro-vans, cargo vans). External "van"/"MPV" labels (UA-DETRAC van, HeteroTraffic MPV)
must be re-checked: box-shaped -> minibus, car-like -> car, wrong/unreadable -> drop.
