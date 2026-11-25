import argparse
import hashlib
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
AREA = ROOT / 'data/VLM363_working/expansion_queue/batch_002'
OUTPUT = AREA / 'vehicle_review_revision_001'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def survey():
    if OUTPUT.exists():
        raise FileExistsError('Never overwrite vehicle QA')
    OUTPUT.mkdir()
    manifest = json.loads((AREA / 'manifest.json').read_text(encoding='utf-8'))
    records = []
    for segment in manifest['segments']:
        clip = ROOT / segment['clip']
        if digest(clip) != segment['clip_sha256']:
            raise ValueError('Clip changed since extraction')
        folder = OUTPUT / segment['segment_id']
        folder.mkdir()
        capture = cv2.VideoCapture(str(clip))
        sheet = Image.new('RGB', (2560, 1878), '#181818')
        frames = []
        try:
            for clip_frame in range(segment['decoded_clip_frames']):
                success, frame = capture.read()
                if not success:
                    raise RuntimeError('Missing clip frame')
                if clip_frame % 30:
                    continue
                source_frame = segment['source_start_frame'] + clip_frame
                image_path = folder / f'source_{source_frame:08d}.jpg'
                if not cv2.imwrite(str(image_path), frame):
                    raise RuntimeError('Image write failed')
                ordinal = len(frames)
                left, top = (ordinal % 5) * 512, (ordinal // 5) * 313
                photo = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((512,288))
                sheet.paste(photo, (left,top+25))
                ImageDraw.Draw(sheet).text((left+5,top+5),f'clip {clip_frame/30:.0f}s | source {source_frame}', fill='white')
                frames.append({'source_frame_id':source_frame, 'clip_frame_id':clip_frame,
                               'image':image_path.relative_to(ROOT).as_posix()})
        finally:
            capture.release()
        sheet.save(folder / 'timeline.jpg')
        records.append({'segment_id':segment['segment_id'], 'frames':frames,
                        'inspection_method':'one-second sampled timeline, not full-frame manual review'})
    (OUTPUT / 'survey_frames.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    print(OUTPUT)


def validate_seed(seed, segment):
    start, end = seed['clip_seconds']
    fps = segment['nominal_fps']
    if not 0 <= start < end <= segment['decoded_clip_frames'] / fps:
        raise ValueError('Invalid context clip range')
    local_frame = seed['reference_frame'] - segment['source_start_frame']
    if not start * fps <= local_frame < end * fps:
        raise ValueError('Reference outside context clip')
    left, top, right, bottom = seed['bbox']
    if not 0 <= left < right <= 960 or not 0 <= top < bottom <= 540:
        raise ValueError('Invalid reference box')
    if seed['class'] not in {None, 'car', 'motorcycle', 'bus', 'truck'}:
        raise ValueError('Invalid proposed class')


def build_cases():
    manifest_path = AREA / 'manifest.json'
    owner_path = AREA / 'qa/human_segment_review.json'
    seed_path = OUTPUT / 'manual_vehicle_seeds.json'
    seeds = json.loads(seed_path.read_text(encoding='utf-8'))['cases']
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    segments = {row['segment_id'][-2:]:row for row in manifest['segments']}
    if len({row['id'] for row in seeds}) != len(seeds):
        raise ValueError('Duplicate candidate IDs')
    for seed in seeds:
        validate_seed(seed, segments[seed['segment']])
    cases_folder = OUTPUT / 'cases'
    if cases_folder.exists() and any(cases_folder.iterdir()):
        raise FileExistsError('Vehicle case outputs already exist')
    records, review_rows, lines, overview = [], [], [], []
    source_manifest = json.loads((ROOT / 'runs/project363/vlm/source_survey/source_manifest.json').read_text(encoding='utf-8'))
    native_sizes = {row['survey_id']:(row['decoder']['width'],row['decoder']['height'])
                    for row in source_manifest if row['survey_id'] in {entry['survey_id'] for entry in segments.values()}}
    cases_folder.mkdir(exist_ok=True)
    for seed in seeds:
        segment = segments[seed['segment']]
        folder = cases_folder / seed['id']
        folder.mkdir()
        reference = OUTPUT / segment['segment_id'] / f"source_{seed['reference_frame']:08d}.jpg"
        photo = Image.open(reference).convert('RGB')
        left, top, right, bottom = seed['bbox']
        marked = photo.copy()
        draw = ImageDraw.Draw(marked)
        draw.rectangle(seed['bbox'],outline='#00ff00',width=4)
        draw.rectangle((0,0,960,28),fill='black')
        draw.text((8,7),f"{seed['id']} | source frame {seed['reference_frame']} | MANUAL PROPOSAL ONLY",fill='white')
        marked.save(folder / 'reference_bbox.jpg',quality=95)
        photo.crop((max(0,left-20),max(0,top-20),min(960,right+20),min(540,bottom+20))).save(folder / 'target_crop.jpg',quality=95)
        native_width, native_height = native_sizes[segment['survey_id']]
        native_box = [left*native_width/960,top*native_height/540,right*native_width/960,bottom*native_height/540]
        context_start, context_end = [round(value*segment['nominal_fps']) for value in seed['clip_seconds']]
        capture = cv2.VideoCapture(str(ROOT / segment['clip']))
        writer = cv2.VideoWriter(str(folder / 'context.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),segment['nominal_fps'],(960,540))
        try:
            if not capture.isOpened() or not writer.isOpened() or not capture.set(cv2.CAP_PROP_POS_FRAMES,context_start):
                raise RuntimeError('Cannot open context video')
            for clip_frame in range(context_start,context_end):
                success, frame = capture.read()
                if not success:
                    raise RuntimeError('Context frame missing')
                source_frame = segment['source_start_frame'] + clip_frame
                cv2.rectangle(frame,(0,0),(960,27),(0,0,0),-1)
                cv2.putText(frame,f"{seed['id']} | source {source_frame} | ref bbox only at {seed['reference_frame']}",(8,19),cv2.FONT_HERSHEY_SIMPLEX,0.48,(255,255,255),1)
                if source_frame == seed['reference_frame']:
                    cv2.rectangle(frame,(left,top),(right,bottom),(0,255,0),3)
                writer.write(frame)
        finally:
            capture.release()
            writer.release()
        verify = cv2.VideoCapture(str(folder / 'context.mp4'))
        count = 0
        try:
            while verify.read()[0]:
                count += 1
        finally:
            verify.release()
        if count != context_end-context_start:
            raise RuntimeError('Context frame count mismatch')
        context_path = (folder / 'context.mp4').relative_to(ROOT).as_posix()
        record = {
            'candidate_id':seed['id'], 'canonical_vehicle_id':None, 'segment_id':segment['segment_id'],
            'description':seed['description'], 'class_proposed':seed['class'], 'class_confirmed':None,
            'motion_proposed':seed['motion'], 'review_status':'PENDING_VEHICLE_QA',
            'source':segment['source'], 'source_sha256':segment['source_sha256'],
            'reference_source_frame_id':seed['reference_frame'], 'bbox_xyxy_proposed_native':native_box,
            'bbox_xyxy_review_clip':seed['bbox'], 'bbox_annotation_method':'manual_reference_keyframe_only',
            'context_source_start_frame':segment['source_start_frame']+context_start,
            'context_source_end_frame_exclusive':segment['source_start_frame']+context_end,
            'context_range_is_exact_vehicle_presence':False, 'timebase_verified':False,
            'owner_segment_assessment':'no_red_light_crossing', 'owner_segment_qa':owner_path.relative_to(ROOT).as_posix(),
            'package_assessment':None, 'crossing_status':None, 'stop_line':None, 'signal_roi':None,
            'scope_status':'PENDING_TARGET_DIRECTION_QA', 'is_ground_truth':False, 'is_training_ready':False,
            'context_clip':context_path, 'context_clip_sha256':digest(folder / 'context.mp4'),
            'reference_bbox_image':(folder / 'reference_bbox.jpg').relative_to(ROOT).as_posix(),
            'target_crop':(folder / 'target_crop.jpg').relative_to(ROOT).as_posix(),
            'continuation_segment':seed.get('continuation_segment'), 'identity_link_confirmed':False,
        }
        records.append(record)
        review_rows.append({'candidate_id':seed['id'], 'decision':'UNREVIEWED', 'target_bbox_confirmed':False,
                            'final_class':None, 'final_scope_status':None, 'final_package_assessment':None,
                            'signal_mapping_confirmed':False, 'stop_line_confirmed':False,
                            'same_vehicle_as':None, 'notes':''})
        lines.append(f"| {seed['id']} | {seed['segment']} | {seed['clip_seconds'][0]}–{seed['clip_seconds'][1]} | {seed['description']} | {seed['class'] or 'PENDING'} |")
        overview.append(marked.resize((480,270)))
    for offset in range(0,len(overview),6):
        sheet = Image.new('RGB',(960,810),'black')
        for ordinal,photo in enumerate(overview[offset:offset+6]):
            sheet.paste(photo,((ordinal%2)*480,(ordinal//2)*270))
        sheet.save(OUTPUT / f'overview_{offset//6+1:02d}.jpg',quality=95)
    (OUTPUT / 'vehicle_candidates.jsonl').write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in records),encoding='utf-8')
    review = {'batch_id':'batch_002','status':'PENDING_VEHICLE_QA', 'manual_seeds_sha256':digest(seed_path),
              'candidate_manifest_sha256':digest(OUTPUT / 'vehicle_candidates.jsonl'), 'is_ground_truth':False,
              'reviews':review_rows, 'segment_04_review':{'vehicles_found_in_sampled_frames':0,
              'all_frames_manually_reviewed':False,'notes':'Night glare and browser-to-fullscreen layout change. No confident vehicle candidate; not proof that no vehicle exists.'}}
    (OUTPUT / 'human_vehicle_review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report = ('# QA lượt xe batch 002 — file local\n\n'
              '24 ứng viên thủ công, chưa phải GT hoặc toàn bộ lượt xe. Không chạy YOLO/ByteTrack/VLM; không tạo web.\n'
              'Đã xem timeline lấy mẫu mỗi 1 giây (120 ảnh), không duyệt bằng mắt mọi frame. Các lượt ngắn/xe xa có thể chưa được liệt kê.\n'
              'Clip 04: chưa có ứng viên xe nhận diện chắc chắn; có đổi layout browser/fullscreen. Không tạo lượt xe giả từ clip trống.\n'
              'B002_009 tiếp tục qua ranh giới clip 01/02, cần xác nhận; không tạo ID mới cho mảnh tiếp theo.\n'
              'Xe rẽ, xe vào mép khung và van có class chưa rõ phải QA scope/class, không ép thành lượt đi thẳng. Xe đỗ và các xe quá xa chưa đưa vào danh sách.\n\n'
              '## Cách xem\n'
              '1. Mở overview_01.jpg đến overview_04.jpg để nhận diện 24 ứng viên.\n'
              '2. Trong cases/B002_xxx, xem reference_bbox.jpg, target_crop.jpg rồi context.mp4.\n'
              '3. Bbox xanh chỉ là keyframe thủ công tham chiếu; không có tracker/interpolation trong video. Header video ghi frame nguồn.\n'
              '4. Điền human_vehicle_review.json: decision CONFIRM/CHANGE/REJECT; final_class car/motorcycle/bus/truck hoặc null nếu chưa rõ; final_scope_status in_scope/out_of_scope/scope_uncertain.\n'
              '5. final_package_assessment chỉ điền khi đã xác minh đủ: red_light_crossing/no_red_light_crossing/insufficient_evidence. Nhận xét cả clip âm tính không tự gán cho từng xe.\n'
              '6. Nếu trùng xe ghi same_vehicle_as; nếu bbox sai hoặc còn thiếu xe ghi notes/trao đổi ID với tôi. Gửi lại file QA sau khi xem.\n\n'
              'Khoảng giây trong bảng là clip context, không phải thời điểm xuất hiện chính xác hoặc lúc vượt vạch.\n\n'
              '| ID | Clip | Giây trong clip | Xe mục tiêu | Class đề xuất |\n|---|---|---|---|---|\n'+ '\n'.join(lines)+'\n')
    (OUTPUT / 'README_QA.md').write_text(report,encoding='utf-8')
    provenance = {'manifest_sha256':digest(manifest_path), 'segment_qa_sha256':digest(owner_path),
                  'seeds_sha256':digest(seed_path), 'vehicle_candidates_sha256':digest(OUTPUT / 'vehicle_candidates.jsonl'),
                  'candidate_count':len(records), 'all_approaches_covered':False, 'new_web_pages':0,
                  'raw_source_modified':False, 'model_executed':False}
    (OUTPUT / 'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(provenance,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['survey','build-cases'])
    arguments = parser.parse_args()
    if arguments.command == 'survey':
        survey()
    else:
        build_cases()
