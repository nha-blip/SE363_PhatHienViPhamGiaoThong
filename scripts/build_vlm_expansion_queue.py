import hashlib
import html
import json
import os
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
SEGMENTS = [('video_05', 0, 30), ('video_05', 30, 60), ('video_20', 60, 90), ('video_03', 0, 30)]


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def main():
    area = ROOT / 'data/VLM363_working/expansion_queue/batch_002'
    output = ROOT / 'runs/project363/vlm/expansion_queue_batch_002'
    if area.exists() or output.exists():
        raise FileExistsError('Never overwrite review artifacts')
    manifest_path = ROOT / 'runs/project363/vlm/source_survey/source_manifest.json'
    sources = {row['survey_id']: row for row in json.loads(manifest_path.read_text(encoding='utf-8'))}
    hashes = {}
    for survey_id, _, _ in SEGMENTS:
        if survey_id not in hashes:
            source = sources[survey_id]
            hashes[survey_id] = digest(ROOT / source['source'])
            if hashes[survey_id] != source['sha256_before']:
                raise ValueError('Source changed since survey')
    area.mkdir(parents=True)
    output.mkdir(parents=True)
    entries, sections = [], []
    for ordinal, (survey_id, start_s, end_s) in enumerate(SEGMENTS, 1):
        source = sources[survey_id]
        metadata = source['decoder']
        fps = metadata['nominal_fps']
        start, end = round(start_s * fps), round(end_s * fps)
        if end > metadata['reported_frame_count']:
            raise ValueError('Segment beyond source')
        segment_id = f'vlm_queue_002_{ordinal:02d}'
        folder = area / segment_id
        folder.mkdir()
        clip = folder / 'clip.mp4'
        width = min(metadata['width'], 960)
        height = round(metadata['height'] * width / metadata['width'] / 2) * 2
        capture = cv2.VideoCapture(str(ROOT / source['source']))
        writer = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        keyframes = []
        selected = {start, start + (end-start)//4, start + (end-start)//2, start + 3*(end-start)//4, end-1}
        try:
            if not capture.isOpened() or not writer.isOpened() or not capture.set(cv2.CAP_PROP_POS_FRAMES, start):
                raise RuntimeError('Decoder/writer initialization failed')
            if abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - start) > 0.5:
                raise RuntimeError('Source seek mismatch')
            for frame_id in range(start, end):
                success, frame = capture.read()
                if not success:
                    raise RuntimeError(f'Missing frame {frame_id}')
                writer.write(cv2.resize(frame, (width, height)))
                if frame_id in selected:
                    image = folder / f'frame_{frame_id:08d}.jpg'
                    if not cv2.imwrite(str(image), frame):
                        raise RuntimeError('Image write failed')
                    keyframes.append({'frame_id':frame_id, 'image':image.relative_to(ROOT).as_posix(), 'sha256':digest(image)})
        finally:
            writer.release()
            capture.release()
        verify = cv2.VideoCapture(str(clip))
        decoded = 0
        try:
            while verify.read()[0]:
                decoded += 1
        finally:
            verify.release()
        if decoded != end-start:
            raise RuntimeError('Derived clip frame count mismatch')
        entry = {'segment_id':segment_id, 'survey_id':survey_id, 'source':source['source'],
                 'source_sha256':hashes[survey_id], 'source_start_frame':start, 'source_end_frame_exclusive':end,
                 'nominal_fps':fps, 'timebase_verified':False, 'clip':clip.relative_to(ROOT).as_posix(),
                 'clip_sha256':digest(clip), 'decoded_clip_frames':decoded, 'keyframes':keyframes,
                 'status':'PENDING_ALL_APPROACHES_ANNOTATION', 'assessment':None, 'targets':[],
                 'is_ground_truth':False, 'is_training_ready':False,
                 'split':None, 'usage_terms_verified':source['usage_terms_verified'],
                 'frame_mapping':'source_frame_id = source_start_frame + zero_based_clip_frame_id'}
        entries.append(entry)
        url = Path(os.path.relpath(clip, output)).as_posix()
        images = ''.join('<img style="width:32%" src="' + html.escape(Path(os.path.relpath(ROOT / frame['image'],output)).as_posix()) + '" title="Source frame ' + str(frame['frame_id']) + '">' for frame in keyframes)
        sections.append(f'<section><h2>{segment_id} — {survey_id}: {start_s}–{end_s}s nguồn</h2><video controls preload="none" style="width:100%;max-width:960px" src="{html.escape(url)}"></video><p><a href="{html.escape(url)}">Mở/tải clip</a></p>{images}</section>')
    for survey_id, expected in hashes.items():
        if digest(ROOT / sources[survey_id]['source']) != expected:
            raise RuntimeError('Source changed during extraction')
    manifest = {'batch_id':'batch_002', 'status':'SEGMENT_REVIEW_QUEUE_NOT_EVENT_DATASET',
                'scope_classes':['car','motorcycle','bus','truck'], 'source_manifest_sha256':digest(manifest_path),
                'raw_source_modified':False, 'new_event_labels':0, 'segments':entries}
    (area / 'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output / 'index.html').write_text('<!doctype html><html lang="vi"><meta charset="utf-8"><title>VLM batch 002 — hàng đợi mở rộng</title><body style="font:16px system-ui;background:#181818;color:#eee;padding:24px"><h1>4 đoạn mới để mở rộng pilot</h1><p>Chưa có nhãn sự kiện hoặc bbox. Xem toàn clip, liệt kê mọi lượt xe thuộc phạm vi; đánh dấu đúng xe, đèn, vạch và diễn biến. Không suy ra vi phạm từ tên video. Nếu MP4 không phát trong browser, dùng player local.</p>'+''.join(sections)+'</body></html>',encoding='utf-8')
    (output / 'report.md').write_text('# VLM expansion queue batch 002\n\n4 new 30-second segments, 3 source files, 120 seconds, 20 source keyframes.\nNo overlap with selected batch_001 intervals in the same source. This is not an independent train/test split.\nNight source video_03 included for review, not an automatically approved negative or positive.\nAll source SHA-256 checked before/after; raw sources unchanged.\nNo model run; no new event labels, vehicle IDs or classes inferred.\nNext: annotate all eligible vehicle approaches in chosen continuous segments; aim for 30–50 approaches across pilot, not 30–50 artificially generated labels.\n',encoding='utf-8')
    print(json.dumps({'segments':len(entries),'new_event_labels':0,'page':str(output / 'index.html')}))


if __name__ == '__main__':
    main()
