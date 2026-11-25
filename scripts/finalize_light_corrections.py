import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

if __package__:
    from .light_bbox_editor import validate_box
    from .prepare_light_corrections import yolo_lines
    from .prepare_light_geometry_review import TASKS
    from .review_light_dataset import correction_focus
else:
    from light_bbox_editor import validate_box
    from prepare_light_corrections import yolo_lines
    from prepare_light_geometry_review import TASKS
    from review_light_dataset import correction_focus


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'data/traffic_light_external_audit_20261022'
FIRST = AUDIT / 'revision_001_20261026'
SECOND = AUDIT / 'revision_002_bbox_edit_20261026'
OUTPUT = AUDIT / 'revision_003_validated_20261026'


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def passed(row):
    return (row.get('decision') == 'KEEP' and row.get('bbox_scope') == 'SIGNAL_HEAD'
            and row.get('class_mapping_confirmed') is True and not row.get('notes', '').strip())


def validate_edit(baseline, row, number, task):
    if not passed(row):
        raise ValueError('Human QA unresolved: ' + row['id'])
    before = baseline['objects']
    after = row['objects']
    if len(before) != len(after):
        raise ValueError('Unexpected object count change')
    index = number - 1
    for position, obj in enumerate(after):
        if position != index and obj != before[position]:
            raise ValueError('Unexpected modification to another object')
    if task == 'geometry':
        edits = row.get('geometry_edits', [])
        if not edits:
            raise ValueError('Geometry was not redrawn')
        expected = before[index]['bbox_xyxy']
        for edit in edits:
            if edit['object_number'] != number or edit['before'] != expected or edit['coordinate_system'] != 'native_source_pixels':
                raise ValueError('Broken geometry edit history')
            expected = edit['after']
        if after[index]['bbox_xyxy'] != expected:
            raise ValueError('Final box does not match edit history')
        for key in ('source_class', 'proposed_class'):
            if before[index][key] != after[index][key]:
                raise ValueError('Unexpected class change during geometry edit')
    else:
        if before[index]['source_class'] != 'off' or after[index]['source_class'] != 'red' or after[index]['proposed_class'] != 'red':
            raise ValueError('Expected off to red correction')
        if before[index]['bbox_xyxy'] != after[index]['bbox_xyxy']:
            raise ValueError('Unexpected geometry change in color-only task')


def main():
    inputs = [AUDIT / 'human_review.json', FIRST / 'correction_review.json', SECOND / 'bbox_edit_review.json',
              FIRST / 'corrected_records.json', FIRST / 'selection_manifest.json', FIRST / 'duplicate_holds.json']
    checksums = {str(path): digest(path) for path in inputs}
    latest = read_json(SECOND / 'bbox_edit_review.json')
    if latest['previous_review_sha256'] != digest(FIRST / 'correction_review.json'):
        raise ValueError('Previous review changed; reconcile versions first')
    records = {row['id']: row for row in read_json(FIRST / 'corrected_records.json')}
    new_rows = {row['id']: row for row in latest['reviews']}
    if set(new_rows) != {task[0] for task in TASKS}:
        raise ValueError('Expected exactly the three targeted cases')
    for identity, number, task, hint in TASKS:
        row = new_rows[identity]
        validate_edit(records[identity], row, number, task)
        with Image.open(row['source_image']) as image:
            if image.size != (records[identity]['width'], records[identity]['height']):
                raise ValueError('Source image dimensions changed')
            for obj in row['objects']:
                validate_box(obj['bbox_xyxy'], *image.size)
        records[identity]['objects'] = copy.deepcopy(row['objects'])
        records[identity]['geometry_edits'] = copy.deepcopy(row.get('geometry_edits', []))
    effective = {}
    for path in inputs[:3]:
        effective.update({row['id']: row for row in read_json(path)['reviews']})
    corrected_ids = {row['id'] for row in read_json(FIRST / 'correction_log.json')}
    if not all(passed(effective[identity]) for identity in corrected_ids):
        raise ValueError('Some of the 14 corrected cases still require QA')
    OUTPUT.mkdir(exist_ok=False)
    for name in ('validated_correction_labels', 'final_previews', 'input_snapshots'):
        (OUTPUT / name).mkdir()
    for path in inputs[:3]:
        (OUTPUT / 'input_snapshots' / (path.parent.name + '_' + path.name)).write_bytes(path.read_bytes())
    for identity in sorted(corrected_ids):
        record = records[identity]
        (OUTPUT / 'validated_correction_labels' / (identity + '.txt')).write_text('\n'.join(yolo_lines(record)) + '\n', encoding='utf-8')
        record['requires_correction_review'] = False
        record['correction_qa_passed'] = True
    for identity, row in new_rows.items():
        correction_focus(row).save(OUTPUT / 'final_previews' / (identity + '.jpg'), quality=98)
    manifest = read_json(FIRST / 'selection_manifest.json')
    for entry in manifest:
        identity = entry['id']
        blockers = [reason for reason in entry['blockers'] if reason not in {'CORRECTION_PREVIEW_REQUIRES_QA', 'HUMAN_REVIEW_UNRESOLVED'}]
        if identity in effective and not passed(effective[identity]):
            blockers.append('HUMAN_REVIEW_UNRESOLVED')
        if identity in corrected_ids:
            entry['label'] = str(OUTPUT / 'validated_correction_labels' / (identity + '.txt'))
        entry.update(blockers=blockers, technical_pool_candidate=not blockers,
                     sample_qa_passed=identity in effective and passed(effective[identity]), train_approved=False)
    unresolved = [row['id'] for row in effective.values() if not passed(row)]
    summary = {'status': 'CORRECTIONS_QA_VALIDATED_NOT_MERGED', 'is_ground_truth': False,
               'input_sha256': checksums, 'latest_cases_passed': len(new_rows), 'corrected_cases_passed': len(corrected_ids),
               'effective_sample_reviews': len(effective), 'effective_sample_passed': len(effective) - len(unresolved),
               'unresolved_sample_ids': unresolved, 'technical_pool_candidates': sum(row['technical_pool_candidate'] for row in manifest),
               'blocker_counts': dict(Counter(reason for row in manifest for reason in row['blockers'])),
               'duplicate_policy': 'Prior cached dHash candidates held; no deletion or claim of confirmed duplicates',
               'current_light363_modified': False, 'source_reviews_modified': False, 'training_started': False}
    write_json(OUTPUT / 'corrected_records.json', list(records.values()))
    write_json(OUTPUT / 'effective_sample_reviews.json', {'is_ground_truth': False, 'reviews': list(effective.values())})
    write_json(OUTPUT / 'selection_manifest.json', manifest)
    write_json(OUTPUT / 'summary.json', summary)
    if checksums != {str(path): digest(path) for path in inputs}:
        raise RuntimeError('Input changed during validation; outputs need reconciliation')
    print(json.dumps(summary, ensure_ascii=True, indent=2), flush=True)


if __name__ == '__main__':
    main()
