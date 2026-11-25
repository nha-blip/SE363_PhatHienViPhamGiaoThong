import hashlib
import json
import re
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .prepare_light_corrections import yolo_lines
else:
    from prepare_light_corrections import yolo_lines


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / 'data/traffic_light_external_audit_20261022'
VALIDATED = AUDIT / 'revision_003_validated_20261026'
OUTPUT = ROOT / 'data/TrafficLight363_expansion_pool_20261026'
CLASSES = {'red': 0, 'yellow': 1, 'green': 2, 'off': 3}


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=True, indent=2), encoding='utf-8')


def sequence_group(record):
    source = record['source']
    member = record['image_member'].replace('\\', '/')
    if source == 'S2TLD_720x1280':
        match = re.search(r'/(normal_[^/]+)/', member)
        if not match:
            raise ValueError('Unknown sequence layout')
        return source + '/' + match.group(1)
    if source == 'S2TLD_1080x1920':
        match = re.search(r'(\d{4}-\d{2}-\d{2}) ', Path(member).stem)
        if not match:
            raise ValueError('Missing capture day')
        return source + '/' + match.group(1)
    if source == 'ATLAS_sample':
        match = re.search(r'front_(?:medium|tele|wide)_(\d{10})', Path(member).name)
        if not match:
            raise ValueError('Missing ATLAS capture timestamp')
        day = datetime.fromtimestamp(int(match.group(1)), timezone.utc).date().isoformat()
        return source + '/' + day
    raise ValueError('Unrecognized dataset source')


def verify_fingerprint(row):
    if 'sha256' not in row or 'dhash' not in row:
        raise ValueError('Incomplete fingerprint: ' + row['path'])
    actual = digest(row['path'])
    if actual != row['sha256']:
        raise ValueError('Image changed since duplicate audit: ' + row['path'])
    return row['path']


def main():
    summary_path = VALIDATED / 'summary.json'
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    for name, checksum in summary['input_sha256'].items():
        if digest(name) != checksum:
            raise ValueError('Review/annotation changed after validation: ' + name)
    fingerprints_path = AUDIT / 'fingerprints.json'
    duplicate_info = json.loads((AUDIT / 'revision_001_20261026/duplicate_holds.json').read_text(encoding='utf-8'))
    if digest(fingerprints_path) != duplicate_info['fingerprints_sha256']:
        raise ValueError('Fingerprint cache changed')
    fingerprints = json.loads(fingerprints_path.read_text(encoding='utf-8'))
    print(f'Verifying {len(fingerprints)} source/current image checksums', flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        verified = []
        for count, path in enumerate(pool.map(verify_fingerprint, fingerprints), 1):
            verified.append(path)
            if count % 2000 == 0:
                print(f'Checksum verification: {count}/{len(fingerprints)}', flush=True)
    fingerprint_map = {row['path']: row for row in fingerprints}
    records = {row['id']: row for row in json.loads((VALIDATED / 'corrected_records.json').read_text(encoding='utf-8'))}
    manifest = json.loads((VALIDATED / 'selection_manifest.json').read_text(encoding='utf-8'))
    selected = [row for row in manifest if row['technical_pool_candidate'] and not row['blockers']]
    selected_ids = {row['id'] for row in selected}
    held = [row for row in manifest if row['id'] not in selected_ids]
    if len(selected) != summary['technical_pool_candidates']:
        raise ValueError('Selection count inconsistent')
    checksums = {'validated_summary': digest(summary_path), 'fingerprints': digest(fingerprints_path),
                 'selection_manifest': digest(VALIDATED / 'selection_manifest.json'),
                 'corrected_records': digest(VALIDATED / 'corrected_records.json')}
    for entry in selected:
        record = records[entry['id']]
        if Path(entry['label']).read_text(encoding='utf-8').splitlines() != yolo_lines(record):
            raise ValueError('Label differs from validated annotation: ' + entry['id'])
        sequence_group(record)
    required = sum(Path(row['image']).stat().st_size for row in selected)
    if shutil.disk_usage(OUTPUT.parent).free < required + 1024 ** 3:
        raise OSError('Insufficient free disk space')
    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT / 'images/pool').mkdir(parents=True)
    (OUTPUT / 'labels/pool').mkdir(parents=True)
    write_json(OUTPUT / 'status.json', {'status': 'BUILDING_NOT_READY'})
    output_rows = []
    for count, entry in enumerate(selected, 1):
        record = records[entry['id']]
        source = Path(record['image'])
        image = OUTPUT / 'images/pool' / source.name
        shutil.copyfile(source, image)
        if digest(image) != fingerprint_map[str(source)]['sha256']:
            raise ValueError('Copied image checksum mismatch')
        label = OUTPUT / 'labels/pool' / (entry['id'] + '.txt')
        label.write_text('\n'.join(yolo_lines(record)) + '\n', encoding='utf-8')
        counts = dict(Counter(obj['proposed_class'] for obj in record['objects']))
        output_rows.append({'id': entry['id'], 'image': str(image), 'label': str(label),
                            'source_image': str(source), 'source_label': entry['label'],
                            'image_sha256': digest(image), 'label_sha256': digest(label),
                            'source': record['source'], 'sequence_group': sequence_group(record),
                            'class_counts': counts, 'priority_rare_class': bool(counts.get('yellow') or counts.get('off')),
                            'individual_sample_qa_passed': entry.get('sample_qa_passed', False),
                            'is_ground_truth': False, 'split': 'UNASSIGNED'})
        if count % 500 == 0:
            print(f'Export: {count}/{len(selected)}', flush=True)
    groups = defaultdict(list)
    for row in output_rows:
        groups[row['sequence_group']].append(row['id'])
    new_summary = {'status': 'FILTERED_POOL_NOT_MERGED_NOT_SPLIT', 'is_ground_truth': False,
                   'input_sha256': checksums, 'images': len(output_rows),
                   'class_counts': dict(Counter({name: sum(row['class_counts'].get(name, 0) for row in output_rows) for name in CLASSES})),
                   'priority_rare_class_images': sum(row['priority_rare_class'] for row in output_rows),
                   'individual_sample_qa_passed': sum(row['individual_sample_qa_passed'] for row in output_rows),
                   'held_images': len(held), 'hold_reason_counts': dict(Counter(reason for row in held for reason in row['blockers'])),
                   'source_images_rehashed': len(verified), 'sequence_groups': {key: len(value) for key, value in groups.items()},
                   'copied_image_bytes': required, 'original_light363_modified': False,
                   'train_val_test_assigned': False, 'training_started': False,
                   'duplicate_policy': 'All dHash<=4 candidates held, not deleted; verified original cache image SHA256 against current bytes',
                   'unknown_state_policy': 'Hold whole images containing wait_on/red_yellow; no forced mapping or dropped true signal boxes'}
    write_json(OUTPUT / 'pool_manifest.json', output_rows)
    write_json(OUTPUT / 'quarantine_manifest.json', held)
    write_json(OUTPUT / 'sequence_groups.json', groups)
    write_json(OUTPUT / 'class_mapping.json', CLASSES)
    write_json(OUTPUT / 'summary.json', new_summary)
    for name, checksum in summary['input_sha256'].items():
        if digest(name) != checksum:
            raise ValueError('Source input changed during export')
    for name, path in (('validated_summary', summary_path), ('fingerprints', fingerprints_path),
                       ('selection_manifest', VALIDATED / 'selection_manifest.json'),
                       ('corrected_records', VALIDATED / 'corrected_records.json')):
        if digest(path) != checksums[name]:
            raise ValueError('Input changed during export')
    write_json(OUTPUT / 'status.json', {'status': 'COMPLETE_FILTERED_POOL_NOT_MERGED'})
    print(json.dumps(new_summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
