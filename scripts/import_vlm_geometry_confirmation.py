import argparse
import copy
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AREA = ROOT / 'data/VLM363_working/annotation_batches/batch_001'


def validate_geometry(geometry, package):
    if geometry is None:
        return
    frame_ids = {entry['frame_id'] for entry in package['evidence']}
    if geometry['reference_frame_id'] not in frame_ids or geometry['coordinate_system'] != 'native_source_pixels':
        raise ValueError('Invalid geometry reference')
    points = geometry.get('points')
    if points is None:
        left, top, right, bottom = geometry['bbox_xyxy']
        if not left < right or not top < bottom:
            raise ValueError('Invalid signal ROI')
        points = [[left, top], [right, bottom]]
    if len(points) != 2 or points[0] == points[1]:
        raise ValueError('Invalid geometry points')
    for point in points:
        if len(point) != 2 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in point):
            raise ValueError('Invalid coordinates')
        if not 0 <= point[0] <= package['source_width'] or not 0 <= point[1] <= package['source_height']:
            raise ValueError('Coordinates outside source')


def main():
    parser = argparse.ArgumentParser(description='Record explicit owner confirmation: no vehicle crossed the line')
    parser.add_argument('--geometry', type=Path, required=True)
    arguments = parser.parse_args()
    raw = arguments.geometry.read_bytes()
    original = json.loads(raw.decode('utf-8-sig'))
    package_path = AREA / 'qa/review_revision_001/evidence_packages.reviewed.jsonl'
    package_bytes = package_path.read_bytes()
    packages = [json.loads(line) for line in package_bytes.decode('utf-8').splitlines()]
    expected = hashlib.sha256(package_bytes).hexdigest()
    if original['batch_id'] != 'batch_001' or original['review_sha256'] != expected:
        raise ValueError('Review hash mismatch')
    indexed = {package['sample_id']: package for package in packages}
    rows = original['reviews']
    if len(rows) != len(indexed) or {row['sample_id'] for row in rows} != set(indexed):
        raise ValueError('Missing or duplicate samples')
    corrected = copy.deepcopy(original)
    unresolved = []
    for row in corrected['reviews']:
        package = indexed[row['sample_id']]
        if row['source_sha256'] != package['source_sha256']:
            raise ValueError('Source provenance mismatch')
        validate_geometry(row['stop_line'], package)
        validate_geometry(row['signal_roi'], package)
        row.update(crossing_status='NO_CROSSING_OBSERVED', crossing_frame_before=None,
                   crossing_frame_after=None, signal_state_at_crossing='unknown',
                   signal_controls_target=row['signal_roi'] is not None,
                   signal_state_applicability='NOT_APPLICABLE_NO_CROSSING',
                   correction_basis='Project owner chat confirmation, 2026-10-12: signals control the target directions; no vehicle crossed the stop line')
        if row['stop_line'] is None or row['signal_roi'] is None:
            unresolved.append({'sample_id':row['sample_id'], 'field':'geometry',
                               'message':'Owner observation recorded; geometry still missing. Existing insufficient_evidence label retained.'})
        package['geometry_review'] = row
        package['is_ground_truth'] = False
        package['is_training_ready'] = False
    corrected.update(status='OWNER_CONFIRMED_NO_CROSSING_GEOMETRY_PARTIAL', validation_errors=unresolved,
                     correction_date='2026-10-12', correction_author='project_owner',
                     original_geometry_sha256=hashlib.sha256(raw).hexdigest())
    output = AREA / 'qa/geometry_revision_001'
    output.mkdir(exist_ok=False)
    (output / 'geometry_review.original.json').write_bytes(raw)
    (output / 'geometry_review.corrected.json').write_text(json.dumps(corrected, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (output / 'evidence_packages.geometry_reviewed.jsonl').write_text(
        ''.join(json.dumps(package, ensure_ascii=False) + '\n' for package in packages), encoding='utf-8')
    (output / 'report.md').write_text(
        '# Owner geometry confirmation — 12/10/2026\n\n'
        'Owner confirms no target crossed the stop line in the reviewed packages.\n'
        'Signal direction mappings confirmed for the seven provided ROIs; no ROI invented for vlm_008.\n'
        'Crossing timestamps remain null. Signal state at crossing is unknown/not applicable, not guessed green.\n'
        '7 no_red_light_crossing labels retained; vlm_008 remains insufficient_evidence due to missing geometry.\n'
        'Original export preserved byte-for-byte. Raw sources and earlier QA revisions unchanged.\n'
        'Labels remain package-level; no full-video assessment or training-ready claim.\n', encoding='utf-8')
    print(json.dumps({'output':str(output), 'no_crossing_confirmations':len(rows), 'remaining_geometry':unresolved}, ensure_ascii=False))


if __name__ == '__main__':
    main()
