import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    revision = ROOT / 'data/VLM363_working/annotation_batches/batch_001/qa/review_revision_001'
    source = revision / 'evidence_packages.reviewed.jsonl'
    samples = [json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
    output = ROOT / 'runs/project363/vlm/geometry_review_batch_001'
    if output.exists():
        raise FileExistsError('Never overwrite geometry QA output')
    for sample in samples:
        paths = [('clip_url', sample['media']['clip_uri'])]
        for field, path in paths:
            if not (ROOT / path).is_file():
                raise FileNotFoundError(path)
            sample[field] = Path(os.path.relpath(ROOT / path, output)).as_posix()
        for evidence in sample['evidence']:
            if not (ROOT / evidence['image']).is_file():
                raise FileNotFoundError(evidence['image'])
            evidence['image_url'] = Path(os.path.relpath(ROOT / evidence['image'], output)).as_posix()
    payload = {'batch_id': 'batch_001', 'review_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'samples': samples}
    template = (ROOT / 'scripts/vlm_geometry_review.html').read_text(encoding='utf-8')
    output.mkdir(parents=True)
    encoded = json.dumps(payload, ensure_ascii=False).replace('<', '\\u003c')
    (output / 'index.html').write_text(template.replace('__PAYLOAD__', encoded), encoding='utf-8')
    (output / 'README.md').write_text(
        '# Geometry review batch_001\n\n'
        'Open index.html. Green: reviewed target bbox; yellow: manually marked stop line; cyan: signal ROI.\n'
        'Two clicks on native-source canvas define geometry. Different approaches require separate mapping.\n'
        'Use source frame indexes (zero based), not clip-local frames. FPS timestamps remain nominal.\n'
        'No coordinate or crossing time is invented. Exported geometry_review.json must be validated before import.\n'
        'Existing reviewed package labels and raw source videos remain unchanged.\n', encoding='utf-8')
    print(output / 'index.html')


if __name__ == '__main__':
    main()
