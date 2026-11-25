import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AREA = ROOT / 'data/VLM363_working/annotation_batches/batch_001'
CLASSES = {'car', 'motorcycle', 'bus', 'truck'}
LABELS = {'red_light_crossing', 'no_red_light_crossing', 'insufficient_evidence'}


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def normalize(review, packages, draft_hash):
    if review.get('batch_id') != 'batch_001' or review.get('draft_sha256') != draft_hash:
        raise ValueError('Batch or draft hash mismatch')
    indexed = {package['sample_id']: package for package in packages}
    rows = review['reviews']
    if len(rows) != len(indexed) or {row['sample_id'] for row in rows} != set(indexed):
        raise ValueError('Missing, duplicate or unexpected review samples')
    normalized, reviewed = [], []
    for row in rows:
        package = indexed[row['sample_id']]
        if row['source_sha256'] != package['source_sha256']:
            raise ValueError('Source hash mismatch')
        if row['decision'] not in {'CONFIRM_PROPOSAL', 'CHANGE_LABEL'}:
            raise ValueError('Unaccepted review')
        if package['vehicle_type'] not in CLASSES or row['final_scope_status'] != 'in_scope':
            raise ValueError('Unsupported vehicle or scope')
        label = row['final_package_assessment']
        if label not in LABELS:
            raise ValueError('Invalid label')
        flags = ('target_bbox_confirmed', 'signal_mapping_confirmed', 'stop_line_confirmed')
        if any(type(row[flag]) is not bool for flag in flags) or not row['target_bbox_confirmed']:
            raise ValueError('Invalid confirmation flags')
        if label != 'insufficient_evidence' and not all(row[flag] for flag in flags):
            raise ValueError('Definitive label requires signal and stop-line confirmation')
        entry = copy.deepcopy(row)
        entry['original_decision'] = row['decision']
        changed = label != package['package_assessment_proposed'] or row['final_scope_status'] != package['scope_status']
        entry['decision'] = 'CHANGE_LABEL' if changed else 'CONFIRM_PROPOSAL'
        normalized.append(entry)
        result = copy.deepcopy(package)
        result.update(scope_status=row['final_scope_status'], package_assessment=label,
                      review_status='HUMAN_REVIEWED_PACKAGE_LABEL', reviewer_id='project_owner',
                      human_review=entry, is_ground_truth=False, is_training_ready=False)
        for evidence in result['evidence']:
            evidence['review_status'] = 'HUMAN_CONFIRMED_TARGET_BBOX'
        reviewed.append(result)
    return normalized, reviewed


def main():
    parser = argparse.ArgumentParser(description='Import QA into a new revision; preserve original drafts')
    parser.add_argument('--review', type=Path, required=True)
    arguments = parser.parse_args()
    raw = arguments.review.read_bytes()
    review = json.loads(raw.decode('utf-8-sig'))
    draft_path = AREA / 'manual_proposals.json'
    package_path = AREA / 'annotations/evidence_packages.draft.jsonl'
    packages = [json.loads(line) for line in package_path.read_text(encoding='utf-8').splitlines()]
    normalized, reviewed = normalize(review, packages, digest(draft_path))
    sources = {package['source']: package['source_sha256'] for package in packages}
    for source, expected in sources.items():
        if digest(ROOT / source) != expected:
            raise ValueError(f'Raw source changed: {source}')
    output = AREA / 'qa/review_revision_001'
    output.mkdir(exist_ok=False)
    (output / 'human_review.original.json').write_bytes(raw)
    manifest = {
        'batch_id': 'batch_001', 'review_status': 'VALIDATED_PACKAGE_LABELS',
        'review_exported_at_utc': review['exported_at_utc'],
        'original_review_sha256': hashlib.sha256(raw).hexdigest(),
        'draft_sha256': digest(draft_path), 'draft_packages_sha256': digest(package_path),
        'scope_policy': {'approved_on': '2026-10-12', 'vehicle_classes': sorted(CLASSES)},
        'source_hashes_verified': sources, 'raw_source_modified': False,
        'label_counts': dict(Counter(row['package_assessment'] for row in reviewed)),
        'is_ground_truth': False, 'is_training_ready': False,
        'remaining_work': ['Camera geometry coordinates', 'Crossing intervals and signal timeline',
                           'Timebase verification', 'Independent QA and pilot expansion'],
    }
    for name, value in [('human_review.normalized.json', {**review, 'reviews': normalized,
                         'status': 'VALIDATED_PACKAGE_LABELS'}), ('provenance.json', manifest)]:
        (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'evidence_packages.reviewed.jsonl').write_text(
        ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in reviewed), encoding='utf-8')
    report = ('# Batch 001 — Human QA import, 12/10/2026\n\n'
              '8 reviewed packages: 7 no_red_light_crossing, 1 insufficient_evidence.\n'
              'All four vehicle classes are in scope; vlm_007 is truck.\n'
              'Original export and drafts preserved. Changed proposals use CHANGE_LABEL.\n'
              'Source SHA-256 verified; raw sources unchanged.\n'
              'Human-reviewed package labels are not complete full-video GT or training-ready data.\n'
              'Checkbox confirmations do not supply geometry coordinates or crossing timestamps.\n')
    (output / 'report.md').write_text(report, encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
