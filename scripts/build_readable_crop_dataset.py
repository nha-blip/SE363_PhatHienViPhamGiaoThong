import argparse
import hashlib
import json
import math
import random
import sqlite3
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import cv2
import numpy as np
import psutil
import yaml

if __package__:
    from .filter_detector_datasets_720p import EXPANDED, NAMES, ORIGINAL, ROOT, write_json
    from .prepare_readable_crop_pilot import contains, intersects, safe_crop
else:
    from filter_detector_datasets_720p import EXPANDED, NAMES, ORIGINAL, ROOT, write_json
    from prepare_readable_crop_pilot import contains, intersects, safe_crop


AUDIT = ROOT / 'runs/project363/quality_filter_balanced_20261027/audit.sqlite'
OUTPUT = ROOT / 'data/YOLO363_readable_crops_20261027'
QA = ROOT / 'data/bbox_readability_qa_20261027/human_review.json'
CROP_QA = ROOT / 'data/readable_crop_pilot_20261027/human_crop_review.json'
FINAL_QA = OUTPUT / 'quality_review/human_review.json'
POLICY = {'vehicle': {'short_at_640': 12, 'crop_laplacian_min': 50},
          'light': {'short_at_640': 4, 'crop_laplacian_min': 30},
          'global_laplacian_min': 25, 'max_crops_per_source_frame': 2,
          'resize': 'Downscale and pad to 1280x720; crop without upscaling',
          'selection': 'Human QA overrides heuristic; unknown neighbors must pass size and blur',
          'geometry': 'No excluded annotated object may intersect crop; all intersecting boxes fully contained and labeled',
          'split': 'Only train is cropped; val and test remain original external references',
          'caveat': 'Draft automatic dataset, not manual GT; missing source annotations cannot be detected geometrically'}


def load_decisions():
    original = json.loads(QA.read_text(encoding='utf-8'))
    crop = json.loads(CROP_QA.read_text(encoding='utf-8'))
    if any(row['decision'] == 'PENDING' for row in original['reviews']) or crop['status'] != 'COMPLETE_CROP_READABILITY_REVIEW':
        raise ValueError('Human QA must be complete')
    decisions = {(row['audit_identity'], row['object_index']): row['decision'] for row in original['reviews']}
    by_id = {row['id']: row for row in original['reviews']}
    for row in crop['sample_reviews']:
        focus = by_id[row['sample_id']]
        decisions[(focus['audit_identity'], focus['object_index'])] = 'UNREADABLE' if row['decision'] == 'REJECT_UNREADABLE' else 'CLEAR'
    if FINAL_QA.exists():
        feedback = json.loads(FINAL_QA.read_text(encoding='utf-8'))
        for row in feedback['reviews']:
            if row['decision'] in {'CLEAR', 'UNREADABLE'}:
                decisions[(row['audit_identity'], row['focus_index'])] = row['decision']
    return decisions


def frame(row):
    if row.get('output_image'):
        picture = cv2.imread(row['output_image'])
        if picture is None or picture.shape[:2] != (720, 1280):
            raise ValueError('Missing/invalid balanced frame')
        return picture
    picture = cv2.imread(row['image'])
    if picture is None:
        raise ValueError('Cannot decode source')
    height, width = picture.shape[:2]
    ratio = min(1, 1280 / width, 720 / height)
    target_width, target_height = round(width * ratio), round(height * ratio)
    resized = cv2.resize(picture, (target_width, target_height), interpolation=cv2.INTER_AREA)
    target = np.full((720, 1280, 3), 114, dtype=np.uint8)
    left, top = (1280 - target_width) // 2, (720 - target_height) // 2
    target[top:top + target_height, left:left + target_width] = resized
    return target


def qualifies(kind, obj, decision):
    if decision in {'UNREADABLE', 'UNCERTAIN'}:
        return False
    if decision == 'CLEAR':
        return True
    box = obj['target_box']
    short = min(box[2] - box[0], box[3] - box[1]) / 2
    ordinary = short >= POLICY[kind]['short_at_640'] and obj['crop_laplacian_variance'] >= POLICY[kind]['crop_laplacian_min']
    extra = kind == 'light' and obj.get('class') in {0, 1, 2} and POLICY[kind].get('allow_color_cue', False) and short >= 2 and obj.get('color_cue', {}).get('supported', False)
    return ordinary or extra


def color_cue(region, identity):
    if identity not in {0, 1, 2} or region.size == 0:
        return {'supported': False}
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    visible = (saturation >= 80) & (value >= 110)
    masks = ((hue <= 12) | (hue >= 168), (hue > 12) & (hue <= 38), (hue > 38) & (hue <= 95))
    counts = [int(np.count_nonzero(visible & mask)) for mask in masks]
    target = counts[identity]
    total = int(np.count_nonzero(visible))
    return {'supported': target >= 2 and total > 0 and target/total >= 0.7,
            'red_yellow_green_pixels': counts, 'visible_color_pixels': total,
            'note': 'Supports inherited state only; heuristic, not verified signal identity'}


def yolo_rows(objects, indices, crop):
    left, top, right, bottom = crop
    rows = []
    for index in indices:
        box = objects[index]['target_box']
        if not contains(crop, box):
            raise ValueError('Partial annotation in crop')
        values = [(box[0] + box[2] - 2 * left) / (2 * (right - left)),
                  (box[1] + box[3] - 2 * top) / (2 * (bottom - top)),
                  (box[2] - box[0]) / (right - left), (box[3] - box[1]) / (bottom - top)]
        center_x, center_y, width, height = values
        if not (width > 0 and height > 0 and center_x-width/2 >= -1e-9 and center_y-height/2 >= -1e-9
                and center_x+width/2 <= 1+1e-9 and center_y+height/2 <= 1+1e-9):
            raise ValueError('Invalid normalized annotation')
        rows.append(str(objects[index]['class']) + ' ' + ' '.join(f'{value:.10f}' for value in values))
    return '\n'.join(rows) + '\n'


def process(row, decisions, output):
    result = {'identity': row['key'], 'kind': row['kind'], 'source': row['source'],
              'source_image': row['image'], 'status': 'REJECT', 'crops': [], 'reasons': []}
    if row['split'] != 'train':
        raise ValueError('Non-train input')
    if any(reason.startswith(('HUMAN_', 'INVALID_', 'IMAGE_DECODE')) for reason in row['reasons']):
        result['reasons'] = ['SOURCE_QA_OR_VALIDITY_HOLD']
        return result
    objects = row['objects']
    if not objects or any('target_box' not in obj for obj in objects):
        result['reasons'] = ['NO_USABLE_ANNOTATIONS']
        return result
    potential = [index for index, obj in enumerate(objects) if decisions.get((row['key'], index)) == 'CLEAR' or
                 (decisions.get((row['key'], index)) not in {'UNREADABLE', 'UNCERTAIN'} and
                  min(obj['target_box'][2]-obj['target_box'][0], obj['target_box'][3]-obj['target_box'][1])/2 >=
                  (2 if row['kind'] == 'light' and POLICY['light'].get('allow_color_cue', False) and obj['class'] != 3 else POLICY[row['kind']]['short_at_640']))]
    if not potential:
        result['reasons'] = ['NO_LARGE_OR_HUMAN_CLEAR_OBJECT']
        return result
    picture = frame(row)
    gray = cv2.cvtColor(picture, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=1)
    for obj in objects:
        if row['kind'] == 'light' and POLICY['light'].get('allow_color_cue', False):
            left, top, right, bottom = obj['target_box']
            obj['color_cue'] = color_cue(picture[max(0, math.floor(top)):min(720, math.ceil(bottom)),
                                               max(0, math.floor(left)):min(1280, math.ceil(right))], obj['class'])
        if 'crop_laplacian_variance' not in obj:
            left, top, right, bottom = obj['target_box']
            region = laplacian[max(0, math.floor(top)):min(720, math.ceil(bottom)),
                               max(0, math.floor(left)):min(1280, math.ceil(right))]
            obj['crop_laplacian_variance'] = float(region.var()) if region.size else 0
    good = {index for index, obj in enumerate(objects) if qualifies(row['kind'], obj, decisions.get((row['key'], index)))}
    excluded = set(range(len(objects))) - good
    result['excluded_indices'] = sorted(excluded)
    if not good:
        result['reasons'] = ['NO_SHARP_OBJECT']
        return result
    global_score = row.get('metrics', {}).get('global_laplacian_variance')
    if global_score is None:
        ratio = min(1, 1280/row['width'], 720/row['height'])
        width, height = round(row['width']*ratio), round(row['height']*ratio)
        left, top = (1280-width)//2, (720-height)//2
        global_score = float(laplacian[top:top+height, left:left+width].var())
    if global_score < POLICY['global_laplacian_min'] and not any(decisions.get((row['key'], index)) == 'CLEAR' or objects[index].get('color_cue', {}).get('supported', False) for index in good):
        result['reasons'] = ['GLOBAL_BLUR']
        return result
    boxes = [obj['target_box'] for obj in objects]
    frequencies = Counter(obj['class'] for obj in objects)
    focuses = sorted(good, key=lambda index: (decisions.get((row['key'], index)) != 'CLEAR',
                     frequencies[objects[index]['class']], -objects[index]['crop_laplacian_variance'], index))
    covered = set()
    for focus in focuses:
        if focus in covered:
            continue
        crop = safe_crop(boxes[focus], boxes, excluded)
        if crop is None:
            continue
        included = [index for index, box in enumerate(boxes) if intersects(crop, box)]
        if focus not in included or any(index in excluded or not contains(crop, boxes[index]) for index in included):
            raise ValueError('Unsafe export')
        left, top, right, bottom = crop
        target = picture[top:bottom, left:right]
        name = row['source'] + '__' + row['key'][:20] + f'__{focus}'
        image_path = output / row['kind'] / 'images/train' / (name + '.jpg')
        label_path = output / row['kind'] / 'labels/train' / (name + '.txt')
        content = yolo_rows(objects, included, crop)
        success, encoded = cv2.imencode('.jpg', target, [cv2.IMWRITE_JPEG_QUALITY, 95])
        if not success:
            raise OSError('JPEG encoding failed')
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape[:2] != (bottom-top, right-left):
            raise ValueError('Encoded crop dimensions wrong')
        image_path.write_bytes(encoded.tobytes())
        label_path.write_text(content, encoding='utf-8')
        result['crops'].append({'image': str(image_path), 'label': str(label_path), 'focus_index': focus,
                               'crop': crop, 'included_indices': included,
                               'class_counts': dict(Counter(NAMES[row['kind']][objects[index]['class']] for index in included)),
                               'focus_class': NAMES[row['kind']][objects[focus]['class']],
                               'focus_color_cue': objects[focus].get('color_cue'),
                               'focus_human_reviewed': decisions.get((row['key'], focus)) == 'CLEAR'})
        covered.update(included)
        if len(result['crops']) >= POLICY['max_crops_per_source_frame']:
            break
    result.update(status='KEEP' if result['crops'] else 'REJECT',
                  reasons=[] if result['crops'] else ['NO_SAFE_CROP_FOUND'],
                  eligible_objects=len(good), eligible_not_covered=len(good-covered))
    return result


def draw_preview(row, result, destination):
    picture = frame(row)
    crop_record = result['crops'][0] if result['crops'] else None
    focus = crop_record['focus_index'] if crop_record else 0
    box = row['objects'][focus]['target_box']
    if crop_record:
        left, top, right, bottom = crop_record['crop']
    else:
        margin = max(16, int(max(box[2]-box[0], box[3]-box[1]) * 0.5))
        left, top, right, bottom = max(0, math.floor(box[0])-margin), max(0, math.floor(box[1])-margin), min(1280, math.ceil(box[2])+margin), min(720, math.ceil(box[3])+margin)
    marked = picture.copy()
    for index, obj in enumerate(row['objects']):
        bounds = [round(value) for value in obj['target_box']]
        color = (0, 0, 255) if index in result.get('excluded_indices', []) else (0, 255, 255)
        cv2.rectangle(marked, tuple(bounds[:2]), tuple(bounds[2:]), color, 1)
    detail = marked[top:bottom, left:right]
    canvas = np.full((820, 1200, 3), 25, dtype=np.uint8)
    focus_name = NAMES[row['kind']][row['objects'][focus]['class']]
    cv2.putText(canvas, f"{result['kind']} {result['status']} | focus: {focus_name} | {row['source']} | not verified GT", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1)
    cv2.putText(canvas, ','.join(result['reasons']) or 'Yellow: retained candidates. Red: excluded bbox. Preview zoom adds no detail.', (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    canvas[95:455, 10:650] = cv2.resize(marked, (640, 360))
    factor = min(1120/detail.shape[1], 300/detail.shape[0])
    enlarged = cv2.resize(detail, (max(1, round(detail.shape[1]*factor)), max(1, round(detail.shape[0]*factor))), interpolation=cv2.INTER_NEAREST)
    canvas[500:500+enlarged.shape[0], 10:10+enlarged.shape[1]] = enlarged
    if not cv2.imwrite(str(destination), canvas):
        raise OSError('Preview write failed')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--light-color-supplement', action='store_true')
    args = parser.parse_args()
    base = OUTPUT
    if args.light_color_supplement:
        base = OUTPUT.with_name('YOLO363_light_color_supplement_20261027')
        POLICY['light']['allow_color_cue'] = True
        POLICY['light']['color_cue_policy'] = 'For red/yellow/green only: short >=2px at640, >=2 bright saturated same-state pixels, dominance >=70%; no off relaxation'
    output = base if not args.limit else base.with_name(base.name + '_smoke')
    decisions = load_decisions()
    output.mkdir(exist_ok=False)
    for kind in NAMES:
        for category in ('images/train', 'labels/train'):
            (output / kind / category).mkdir(parents=True)
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except (AttributeError, psutil.AccessDenied):
        pass
    write_json(output / 'policy.json', POLICY)
    input_paths = [QA, CROP_QA] + ([FINAL_QA] if FINAL_QA.exists() else [])
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths}
    source = sqlite3.connect(f'file:{AUDIT.as_posix()}?mode=ro', uri=True)
    selection = "split='train'"
    if args.light_color_supplement:
        source.execute('ATTACH DATABASE ? AS prior', (f'file:{(OUTPUT / "audit.sqlite").as_posix()}?mode=ro',))
        selection += " AND kind='light' AND identity IN (SELECT identity FROM prior.frames WHERE status='REJECT' AND kind='light')"
    expected = source.execute('SELECT count(*) FROM images WHERE ' + selection).fetchone()[0]
    query = 'SELECT record FROM images WHERE ' + selection
    if args.limit:
        query += ' LIMIT ' + str(args.limit)
        expected = min(expected, args.limit)
    audit = sqlite3.connect(output / 'audit.sqlite')
    audit.execute('CREATE TABLE frames (identity TEXT PRIMARY KEY, kind TEXT, source TEXT, status TEXT, record TEXT)')
    counts = defaultdict(Counter)
    sources = defaultdict(Counter)
    classes = defaultdict(Counter)
    reasons = defaultdict(Counter)
    processed = 0
    samples = {}
    seen = Counter()
    generator = random.Random(363)
    started = time.monotonic()

    def consume(future):
        nonlocal processed
        row, result = future.result()
        kind = row['kind']
        audit.execute('INSERT INTO frames VALUES (?,?,?,?,?)', (row['key'], kind, row['source'], result['status'], json.dumps(result)))
        counts[kind][result['status'] + '_source_frames'] += 1
        counts[kind]['crops'] += len(result['crops'])
        counts[kind]['eligible_not_covered'] += result.get('eligible_not_covered', 0)
        sources[kind + '/' + row['source']][result['status'] + '_source_frames'] += 1
        sources[kind + '/' + row['source']]['crops'] += len(result['crops'])
        for crop in result['crops']:
            classes[kind].update(crop['class_counts'])
            counts[kind]['bbox'] += sum(crop['class_counts'].values())
            sources[kind + '/' + row['source']]['bbox'] += sum(crop['class_counts'].values())
        reasons[kind].update(result['reasons'])
        if row['objects'] and all('target_box' in obj for obj in row['objects']):
            category = result['crops'][0]['focus_class'] if result['crops'] else ','.join(result['reasons'])
            group = (kind, result['status'], category)
            seen[group] += 1
            reservoir = samples.setdefault(group, [])
            index = len(reservoir) if len(reservoir) < 4 else generator.randrange(seen[group])
            if index < 4:
                if index == len(reservoir):
                    reservoir.append((row, result))
                else:
                    reservoir[index] = (row, result)
        processed += 1
        if processed % 100 == 0:
            audit.commit()
        if processed % 1000 == 0:
            state = {'status': 'RUNNING', 'processed': processed, 'expected': expected,
                     'elapsed_seconds': round(time.monotonic()-started),
                     'counts': {key: dict(value) for key, value in counts.items()}}
            write_json(output / 'status.json', state)
            print(json.dumps(state), flush=True)

    def worker(row):
        return row, process(row, decisions, output)

    write_json(output / 'status.json', {'status': 'RUNNING', 'processed': 0, 'expected': expected})
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            pending = set()
            for record, in source.execute(query):
                pending.add(pool.submit(worker, json.loads(record)))
                if len(pending) >= args.workers * 3:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        consume(future)
            for future in pending:
                consume(future)
        audit.commit()
        if processed != expected:
            raise ValueError('Input coverage mismatch')
        qa_folder = output / 'quality_review'
        qa_folder.mkdir()
        sample_rows = []
        for group in sorted(samples):
            for row, result in samples[group]:
                name = f"{row['kind']}_{result['status']}_{len(sample_rows)+1:03d}"
                destination = qa_folder / (name + '.png')
                draw_preview(row, result, destination)
                focus = result['crops'][0]['focus_index'] if result['crops'] else 0
                sample_rows.append({'id': name, 'kind': row['kind'], 'preview': str(destination),
                                    'audit_identity': row['key'], 'focus_index': focus,
                                    'bucket_at_imgsz640': 'crop QA', 'automatic_decision': result['status'], 'decision': 'PENDING'})
        write_json(qa_folder / 'human_review.json', {'status': 'PENDING_FINAL_SAMPLE_QA', 'reviews': sample_rows})
        (qa_folder / 'MO_QA.cmd').write_text('@echo off\r\n"C:\\Users\\NHA\\miniconda3\\envs\\my_env\\python.exe" "D:\\se363-bigdata\\scripts\\review_bbox_readability.py" --review --folder "%~dp0"\r\nif errorlevel 1 pause\r\n', encoding='ascii')
        (output / 'README.md').write_text('# Dataset crop rõ — bản nháp\n\nChỉ tập train được crop; val/test trỏ về nguồn gốc, không lọc lại benchmark.\nXem summary.json để biết số crop, bbox và phân bố lớp/nguồn.\nMở quality_review/MO_QA.cmd để kiểm tra mẫu giữ/bỏ.\nMẫu rejected cần đánh giá đối tượng focus có thật sự không rõ không, không mặc định đồng ý với máy.\nXe 134/135 bị loại theo QA crop mới nhất. Mỗi ảnh nguồn tối đa 2 crop, không bảo đảm giữ mọi đối tượng rõ.\nĐã kiểm tra hình học mọi crop và encode/decode JPEG trước khi ghi; chưa QA thủ công toàn bộ.\nKhông chứa negative crop tự tạo; cần theo dõi false positive khi train đối chứng.\nSố bbox là số lần xuất hiện trong các crop, có thể lặp cùng đối tượng nguồn giữa hai crop.\nNhận dạng kích thước/mờ là heuristic; không sửa nhãn lớp, không tạo thêm chi tiết bằng phóng to.\nCrop nhỏ thay đổi phân bố kích thước và ngữ cảnh khi YOLO resize, không bảo đảm tăng AP trên nguyên khung.\nKhông sửa dataset gốc, chưa train. Không dùng nếu status.json báo FAILED hoặc RUNNING.\n', encoding='utf-8')
        for kind in NAMES:
            if counts[kind]['crops'] == 0:
                continue
            settings = {'path': (output / kind).as_posix(), 'train': 'images/train',
                        'val': ((ORIGINAL / 'Vehicle363-v1/images/val').as_posix() if kind == 'vehicle' else (EXPANDED / 'val_combined.txt').as_posix()),
                        'test': ((ORIGINAL / 'Vehicle363-v1/images/test').as_posix() if kind == 'vehicle' else (EXPANDED / 'test_original.txt').as_posix()), 'names': NAMES[kind]}
            (output / kind / 'data.yaml').write_text(yaml.safe_dump(settings, sort_keys=False), encoding='utf-8')
            if kind == 'light':
                settings['val'] = (ORIGINAL / 'TrafficLight363-v1/images/val').as_posix()
                (output / kind / 'data_original_val.yaml').write_text(yaml.safe_dump(settings, sort_keys=False), encoding='utf-8')
        if args.light_color_supplement:
            old_summary = json.loads((OUTPUT / 'summary.json').read_text(encoding='utf-8'))
            for kind in NAMES:
                prior_config = yaml.safe_load((OUTPUT / kind / 'data.yaml').read_text(encoding='utf-8'))
                if kind == 'vehicle':
                    (output / kind / 'data.yaml').write_text(yaml.safe_dump(prior_config, sort_keys=False), encoding='utf-8')
                    continue
                original_images = sorted((OUTPUT / kind / 'images/train').glob('*.jpg'))
                new_images = sorted((output / kind / 'images/train').glob('*.jpg'))
                if len(original_images) != old_summary['counts']['light']['crops']:
                    raise ValueError('Previous light crop payload changed')
                train_list = output / kind / 'train_combined.txt'
                train_list.write_text('\n'.join(path.as_posix() for path in original_images + new_images)+'\n', encoding='utf-8')
                prior_config.update(path=(output / kind).as_posix(), train=train_list.as_posix())
                (output / kind / 'data.yaml').write_text(yaml.safe_dump(prior_config, sort_keys=False), encoding='utf-8')
                prior_config['val'] = (ORIGINAL / 'TrafficLight363-v1/images/val').as_posix()
                (output / kind / 'data_original_val.yaml').write_text(yaml.safe_dump(prior_config, sort_keys=False), encoding='utf-8')
        for kind in NAMES:
            if len(list((output / kind / 'images/train').glob('*.jpg'))) != counts[kind]['crops'] or len(list((output / kind / 'labels/train').glob('*.txt'))) != counts[kind]['crops']:
                raise ValueError('File pair count mismatch')
        if any(hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest for path, digest in hashes.items()):
            raise ValueError('QA input changed while building')
        result = {'status': 'DRAFT_COMPLETE_REQUIRES_SAMPLE_QA', 'processed_train_frames': processed,
                  'counts': {key: dict(value) for key, value in counts.items()},
                  'source_counts': {key: dict(value) for key, value in sources.items()},
                  'class_counts': {key: dict(value) for key, value in classes.items()},
                  'reject_reasons': {key: dict(value) for key, value in reasons.items()},
                  'review_previews': len(sample_rows), 'qa_input_sha256': hashes,
                  'crop_geometry_and_encoded_dimensions_checked': True, 'output_pair_counts_checked': True,
                  'source_modified': False, 'training_started': False, 'elapsed_seconds': time.monotonic()-started}
        if args.light_color_supplement:
            result['supplement_only'] = True
            result['previous_vehicle_payload_unchanged'] = True
            result['previous_light_payload_unchanged'] = True
            result['combined_light_crops'] = old_summary['counts']['light']['crops'] + counts['light']['crops']
            result['combined_light_bbox'] = old_summary['counts']['light']['bbox'] + counts['light']['bbox']
            combined_classes = Counter(old_summary['class_counts']['light'])
            combined_classes.update(classes['light'])
            result['combined_light_class_counts'] = dict(combined_classes)
        write_json(output / 'summary.json', result)
        write_json(output / 'status.json', result)
        lines = ['# Kết quả mở rộng crop — cần QA mẫu', '',
                 f'Đã khảo sát {processed:,} frame train. Không đổi val/test, không train model.', '',
                 '| Loại | Frame giữ | Crop | Bbox (lượt xuất hiện) |', '| --- | ---: | ---: | ---: |']
        for kind in NAMES:
            lines.append(f"| {kind} | {counts[kind]['KEEP_source_frames']} | {counts[kind]['crops']} | {counts[kind]['bbox']} |")
        lines.extend(['', '## Phân bố lớp', ''])
        for kind in NAMES:
            lines.append(f'- {kind}: ' + ', '.join(f'{name}: {value}' for name, value in classes[kind].items()))
        lines.extend(['', '## Phân bố nguồn', ''])
        for name, values in sources.items():
            lines.append(f"- {name}: {values['KEEP_source_frames']} frame giữ, {values['crops']} crop, {values['bbox']} bbox.")
        lines.extend(['', f'Có {len(sample_rows)} ảnh QA trong quality_review. Mở MO_QA.cmd ở đó.', '',
                      'Chỉ coi là bản nháp; hình học an toàn không chứng minh mọi đối tượng rõ hoặc nhãn đúng.',
                      'Đèn vàng/tắt và bus/truck vẫn cần cân đối theo thống kê, không tự tăng trọng số trong lần này.',
                      'Một frame tối đa 2 crop, ưu tiên lớp ít xuất hiện trong chính frame; không bảo đảm cân bằng toàn dataset.',
                      'Giữ test nguyên gốc; báo cáo AP phải dùng cùng benchmark với baseline.'])
        if args.light_color_supplement:
            lines.extend(['', '## Bổ sung riêng đèn', '',
                          f"Tổng train đèn kết hợp: {result['combined_light_crops']} crop, {result['combined_light_bbox']} bbox.",
                          'Train mới dùng light/train_combined.txt, gồm mọi crop đèn cũ và crop bổ sung từ frame trước đó bị loại.',
                          'Bộ xe và payload đèn cũ không thay đổi. Đèn off không được nới bằng tín hiệu màu.',
                          'HSV chỉ hỗ trợ màu nhãn kế thừa, không tự sửa nhãn hoặc xác nhận nguồn sáng là đèn giao thông.',
                          'QA đèn REJECT cũ còn thấy màu được xem như ứng viên xét lại, không gán CLEAR bằng suy đoán.',
                          'Không cần train từ ảnh trong thư mục smoke thử nghiệm.'])
        (output / 'BUILD_REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        audit.commit()
        write_json(output / 'status.json', {'status': 'FAILED', 'processed': processed, 'error': str(error)})
        raise
    finally:
        source.close()
        audit.close()


if __name__ == '__main__':
    main()
