import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'data/traffic_light_external_audit_20261022'
OUTPUT = AUDIT / 'revision_001_20261026'
CLASSES = {'red': 0, 'yellow': 1, 'green': 2, 'off': 3}
ADVERTISEMENTS = (
    '008932898d9407b30e90', '04704203f1c3704f1cd7', '35c6e35bfd2d13e20191',
    'be4e195042af1e050f0a', '0093f71333d0b11125d4', '00f03a0b805c0e156b23',
    '01726e8d5632f6fbba2c', '095d78cf0762ca0e3d77', '0c8de25ba19f9ffc0c53',
    '0f0d3a7e70bf17c7a217',
)


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def action_plan():
    plan = [{'id': 'S2TLD_1080x1920_' + suffix, 'object_number': 1,
             'action': 'REMOVE_NON_TRAFFIC_OBJECT', 'expected_class': 'wait_on'} for suffix in ADVERTISEMENTS]
    plan.extend({'id': 'S2TLD_1080x1920_' + suffix, 'object_number': 2,
                 'action': 'CHANGE_CLASS', 'expected_class': 'off', 'new_class': 'red'}
                for suffix in ('4dce7c2181c4af1bd0ef', 'e78a49c1954ccba34daf'))
    plan.append({'id': 'S2TLD_1080x1920_a229d4438409f056af14', 'object_number': 2,
                 'action': 'REMOVE_NON_TRAFFIC_OBJECT', 'expected_class': 'red'})
    plan.append({'id': 'ATLAS_sample_b241a608f0cbbaa93a81', 'object_number': 2,
                 'action': 'CLAMP_ROUNDING_AT_IMAGE_EDGE', 'expected_class': 'circle_green'})
    return plan


def apply_action(record, action):
    updated = copy.deepcopy(record)
    index = action['object_number'] - 1
    before = copy.deepcopy(updated['objects'][index])
    if before['source_class'] != action['expected_class']:
        raise ValueError('Source class changed since review: ' + record['id'])
    if action['action'] == 'REMOVE_NON_TRAFFIC_OBJECT':
        updated['objects'].pop(index)
        after = None
    elif action['action'] == 'CHANGE_CLASS':
        updated['objects'][index]['source_class'] = action['new_class']
        updated['objects'][index]['proposed_class'] = action['new_class']
        after = copy.deepcopy(updated['objects'][index])
    elif action['action'] == 'CLAMP_ROUNDING_AT_IMAGE_EDGE':
        box = before['bbox_xyxy']
        clamped = [max(0, min(value, bound)) for value, bound in zip(box, (record['width'], record['height'], record['width'], record['height']))]
        if max(abs(old - new) for old, new in zip(box, clamped)) > 0.01:
            raise ValueError('Not a subpixel rounding error')
        updated['objects'][index]['bbox_xyxy'] = clamped
        updated['objects'][index]['valid'] = True
        after = copy.deepcopy(updated['objects'][index])
    else:
        raise ValueError('Unknown correction')
    updated['source_audit_issues'] = record.get('source_audit_issues', record.get('issues', []))
    updated['issues'] = [issue for issue in record.get('issues', []) if issue not in {'INVALID_BBOX', 'UNMAPPED_CLASS_REQUIRES_REVIEW'}]
    if any(obj.get('proposed_class') not in CLASSES for obj in updated['objects']):
        updated['issues'].append('UNMAPPED_CLASS_REQUIRES_REVIEW')
    if any(not obj.get('valid', True) for obj in updated['objects']):
        updated['issues'].append('INVALID_BBOX')
    updated['eligible_after_human_qa'] = False
    updated['requires_correction_review'] = True
    return updated, {'action': action, 'before': before, 'after': after}


def yolo_lines(record):
    lines = []
    width, height = record['width'], record['height']
    for obj in record['objects']:
        if obj.get('proposed_class') not in CLASSES:
            raise ValueError('Unmapped object; whole image must be held')
        left, top, right, bottom = obj['bbox_xyxy']
        if not all(math.isfinite(value) for value in (left, top, right, bottom)) or not (0 <= left < right <= width and 0 <= top < bottom <= height):
            raise ValueError('Invalid bounding box')
        values = ((left + right) / (2 * width), (top + bottom) / (2 * height),
                  (right - left) / width, (bottom - top) / height)
        lines.append(str(CLASSES[obj['proposed_class']]) + ' ' + ' '.join(f'{value:.10f}' for value in values))
    if not lines:
        raise ValueError('Empty image requires explicit negative-image review')
    return lines


def duplicate_holds(records, fingerprints):
    lookup = np.array([value.bit_count() for value in range(256)], dtype=np.uint8)
    valid = [row for row in fingerprints if 'dhash' in row]
    matrix = np.array([list(bytes.fromhex(row['dhash'])) for row in valid], dtype=np.uint8)
    path_index = {row['path']: index for index, row in enumerate(valid)}
    new_paths = {row['image'] for row in records}
    existing = np.array([row['path'] not in new_paths for row in valid])
    result = {}
    for record in records:
        index = path_index.get(record['image'])
        if index is None:
            result[record['id']] = {'hold': True, 'reason': 'MISSING_FINGERPRINT'}
            continue
        distances = lookup[np.bitwise_xor(matrix, matrix[index])].sum(axis=1)
        matches = distances <= 4
        matches[index] = False
        result[record['id']] = {'hold': bool(matches.any()), 'new_candidates': int((matches & ~existing).sum()),
                                'existing_candidates': int((matches & existing).sum())}
    return result


def draw_boxes(image, objects, origin=(0, 0), scale=1, number_override=None):
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 18)
    for number, obj in enumerate(objects, 1):
        if number_override is not None:
            number = number_override
        box = [round((value - origin[position % 2]) * scale) for position, value in enumerate(obj['bbox_xyxy'])]
        if box[2] < box[0] or box[3] < box[1]:
            continue
        draw.rectangle(box, outline='black', width=6)
        draw.rectangle(box, outline='yellow', width=3)
        label = f'#{number} {obj["source_class"]}'
        anchor = (max(0, box[0]), max(0, box[1] - 24))
        draw.rectangle(draw.textbbox(anchor, label, font=font), fill='black')
        draw.text(anchor, label, font=font, fill='yellow')


def render_change(before, after, change, path):
    with Image.open(before['image']) as opened:
        source = opened.convert('RGB')
    factor = 800 / source.width
    overview_height = round(source.height * factor)
    canvas = Image.new('RGB', (1600, overview_height + 400), '#202020')
    for position, record in enumerate((before, after)):
        panel = source.resize((800, overview_height))
        draw_boxes(panel, record['objects'], scale=factor)
        canvas.paste(panel, (position * 800, 35))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf', 20)
    draw.text((10, 4), 'BEFORE - original labels', fill='white', font=font)
    draw.text((810, 4), 'AFTER - draft corrections (not GT)', fill='white', font=font)
    target = change['before']['bbox_xyxy']
    margin = max(20, math.ceil(max(target[2] - target[0], target[3] - target[1]) * 0.7))
    left = max(0, math.floor(target[0]) - margin)
    top = max(0, math.floor(target[1]) - margin)
    right = min(source.width, math.ceil(target[2]) + margin)
    bottom = min(source.height, math.ceil(target[3]) + margin)
    crop = source.crop((left, top, right, bottom))
    factor = min(760 / crop.width, 280 / crop.height)
    resized = crop.resize((max(1, round(crop.width * factor)), max(1, round(crop.height * factor))), Image.Resampling.NEAREST)
    for position, target_obj in enumerate((change['before'], change['after'])):
        panel = resized.copy()
        if target_obj is not None:
            draw_boxes(panel, [target_obj], origin=(left, top), scale=factor, number_override=change['action']['object_number'])
        canvas.paste(panel, (position * 800 + 10, overview_height + 100))
    draw.text((10, overview_height + 43), before['id'], fill='white', font=font)
    draw.text((810, overview_height + 72), change['action']['action'], fill='yellow', font=font)
    canvas.save(path, quality=98)


def main():
    paths = {name: AUDIT / name for name in ('human_review.json', 'records.json', 'fingerprints.json')}
    checksums = {name: sha256(path) for name, path in paths.items()}
    review = json.loads(paths['human_review.json'].read_text(encoding='utf-8'))
    reviews = {row['id']: row for row in review['reviews']}
    records = json.loads(paths['records.json'].read_text(encoding='utf-8'))
    originals = {row['id']: row for row in records}
    updated = copy.deepcopy(originals)
    logs = []
    for action in action_plan():
        identity = action['id']
        if not reviews[identity].get('notes', '').strip():
            raise ValueError('Expected human correction note missing: ' + identity)
        updated[identity], log = apply_action(updated[identity], action)
        logs.append({**log, 'id': identity, 'human_note': reviews[identity]['notes'],
                     'original_decision': reviews[identity]['decision']})
    OUTPUT.mkdir(exist_ok=False)
    for folder in ('labels_candidate', 'correction_previews', 'input_snapshots'):
        (OUTPUT / folder).mkdir()
    for name, path in paths.items():
        (OUTPUT / 'input_snapshots' / name).write_bytes(path.read_bytes())
    fingerprints = json.loads(paths['fingerprints.json'].read_text(encoding='utf-8'))
    holds = duplicate_holds(records, fingerprints)
    write_json(OUTPUT / 'duplicate_holds.json', {'fingerprints_sha256': checksums['fingerprints.json'],
               'policy': 'Cached dHash distance <=4; hold ALL candidates, never delete or call them proven duplicates', 'records': holds})
    changed_ids = {row['id'] for row in logs}
    new_reviews = []
    for log in logs:
        identity = log['id']
        image_path = OUTPUT / 'correction_previews' / (identity + '.jpg')
        render_change(originals[identity], updated[identity], log, image_path)
        new_reviews.append({'id': identity, 'qa_image': str(image_path), 'source_image': originals[identity]['image'],
                            'decision': 'PENDING', 'bbox_scope': 'UNREVIEWED', 'class_mapping_confirmed': False,
                            'notes': '', 'objects': updated[identity]['objects'], 'correction': log})
    manifest = []
    for identity, record in updated.items():
        blockers = []
        annotation_error = None
        try:
            lines = yolo_lines(record)
            (OUTPUT / 'labels_candidate' / (identity + '.txt')).write_text('\n'.join(lines) + '\n', encoding='utf-8')
        except (ValueError, KeyError) as error:
            annotation_error = str(error)
            blockers.append('UNMAPPED_OR_INVALID_ANNOTATION')
        if holds[identity]['hold']:
            blockers.append('NEAR_DUPLICATE_CANDIDATE_HOLD')
        human = reviews.get(identity)
        if identity in changed_ids:
            blockers.append('CORRECTION_PREVIEW_REQUIRES_QA')
        elif human and (human['decision'] != 'KEEP' or not human['class_mapping_confirmed'] or human['bbox_scope'] != 'SIGNAL_HEAD'):
            blockers.append('HUMAN_REVIEW_UNRESOLVED')
        manifest.append({'id': identity, 'image': record['image'], 'source': record['source'],
                         'source_member': record['image_member'],
                         'blockers': blockers, 'technical_pool_candidate': not blockers,
                         'individual_human_qa': identity in reviews, 'train_approved': False,
                         'annotation_error': annotation_error,
                         'label': str(OUTPUT / 'labels_candidate' / (identity + '.txt')) if 'UNMAPPED_OR_INVALID_ANNOTATION' not in blockers else None})
    summary = {'status': 'DRAFT_CORRECTIONS_NOT_MERGED', 'is_ground_truth': False, 'input_sha256': checksums,
               'corrected_images': len(changed_ids), 'actions': dict(Counter(row['action']['action'] for row in logs)),
               'candidate_label_files': len(list((OUTPUT / 'labels_candidate').glob('*.txt'))),
               'technical_pool_candidates': sum(row['technical_pool_candidate'] for row in manifest),
               'blocker_counts': dict(Counter(reason for row in manifest for reason in row['blockers'])),
               'near_candidates_with_existing': sum(row.get('existing_candidates', 0) > 0 for row in holds.values()),
               'split_policy': 'No split assigned; group by sequence/camera/time before any merge; existing test unchanged',
               'coordinate_policy': 'Preserve source pixel edge values, no guessed XML minus-one conversion; explicit logged clamp only',
               'source_labels_modified': False, 'current_light363_modified': False,
               'human_review_modified': False, 'training_started': False}
    write_json(OUTPUT / 'corrected_records.json', list(updated.values()))
    write_json(OUTPUT / 'correction_log.json', logs)
    write_json(OUTPUT / 'selection_manifest.json', manifest)
    write_json(OUTPUT / 'correction_review.json', {'status': 'PENDING_HUMAN_QA', 'is_ground_truth': False, 'reviews': new_reviews})
    write_json(OUTPUT / 'summary.json', summary)
    if checksums != {name: sha256(path) for name, path in paths.items()}:
        raise RuntimeError('Input changed during run; do not use outputs until reconciled')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
