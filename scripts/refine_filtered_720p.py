import hashlib
import json
import shutil
import sqlite3
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import psutil
import yaml

if __package__:
    from .filter_detector_datasets_720p import EXPANDED, NAMES, ORIGINAL, OUTPUT as BASE, REPORT, write_json
else:
    from filter_detector_datasets_720p import EXPANDED, NAMES, ORIGINAL, OUTPUT as BASE, REPORT, write_json


OUTPUT = BASE.with_name(BASE.name + '_strict')
STRICT = {'global_min': 50, 'vehicle': {'short': 16, 'long': 24, 'crop_min': 200},
          'light': {'short': 8, 'long': 16, 'crop_min': 100}}


def reasons(row):
    if row['decision'] != 'KEEP':
        return row['reasons']
    failures = []
    policy = STRICT[row['kind']]
    if row['metrics']['global_laplacian_variance'] < STRICT['global_min']:
        failures.append('STRICT_GLOBAL_BLUR_HEURISTIC')
    for obj in row['objects']:
        left, top, right, bottom = obj['target_box']
        if min(right - left, bottom - top) < policy['short'] or max(right - left, bottom - top) < policy['long']:
            failures.append('STRICT_SMALL_OBJECT')
        if obj['crop_laplacian_variance'] < policy['crop_min']:
            failures.append('STRICT_OBJECT_BLUR_HEURISTIC')
    return sorted(set(failures))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def copy_pair(row):
    image = OUTPUT / row['kind'] / 'images' / row['split'] / Path(row['output_image']).name
    label = OUTPUT / row['kind'] / 'labels' / row['split'] / Path(row['output_label']).name
    shutil.copyfile(row['output_image'], image)
    shutil.copyfile(row['output_label'], label)
    if digest(image) != digest(row['output_image']) or digest(label) != digest(row['output_label']):
        raise ValueError('Copy checksum mismatch')
    return row, str(image), str(label)


def main():
    if json.loads((BASE / 'validation.json').read_text())['status'] != 'EXPORTED_FILES_VALIDATED_REQUIRES_VISUAL_QA':
        raise ValueError('Initial export not validated')
    try:
        psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except (AttributeError, psutil.AccessDenied):
        pass
    OUTPUT.mkdir(exist_ok=False)
    for kind in NAMES:
        for split in ('train', 'val', 'test'):
            for category in ('images', 'labels'):
                (OUTPUT / kind / category / split).mkdir(parents=True)
    write_json(OUTPUT / 'status.json', {'status': 'BUILDING_STRICT_REVISION'})
    source = sqlite3.connect(f'file:{(REPORT / "audit.sqlite").as_posix()}?mode=ro', uri=True)
    audit = sqlite3.connect(OUTPUT / 'selection.sqlite')
    audit.execute('CREATE TABLE selection (identity TEXT PRIMARY KEY, kind TEXT, split TEXT, decision TEXT, reasons TEXT, image TEXT, label TEXT)')
    counts = defaultdict(Counter)
    classes = defaultdict(Counter)
    sources = defaultdict(Counter)
    reject_reasons = defaultdict(Counter)
    copied = 0
    pending = set()
    with ThreadPoolExecutor(max_workers=4) as pool:
        for record, in source.execute('SELECT record FROM images'):
            row = json.loads(record)
            failure = reasons(row)
            group = row['kind'] + '/' + row['split']
            if failure:
                counts[group]['REJECT'] += 1
                sources[row['kind'] + '/' + row['source']]['REJECT'] += 1
                reject_reasons[group].update(failure)
                audit.execute('INSERT INTO selection VALUES (?,?,?,?,?,?,?)', (row['key'], row['kind'], row['split'], 'REJECT', json.dumps(failure), None, None))
            else:
                pending.add(pool.submit(copy_pair, row))
            if len(pending) >= 12:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
            else:
                done = set()
            for future in done:
                kept, image, label = future.result()
                group = kept['kind'] + '/' + kept['split']
                counts[group]['KEEP'] += 1
                sources[kept['kind'] + '/' + kept['source']]['KEEP'] += 1
                classes[group].update(NAMES[kept['kind']][obj['class']] for obj in kept['objects'])
                audit.execute('INSERT INTO selection VALUES (?,?,?,?,?,?,?)', (kept['key'], kept['kind'], kept['split'], 'KEEP', '[]', image, label))
                copied += 1
                if copied % 1000 == 0:
                    audit.commit()
                    write_json(OUTPUT / 'status.json', {'status': 'COPYING_STRICT_REVISION', 'copied': copied})
                    print('Strict copies:', copied, flush=True)
        for future in pending:
            kept, image, label = future.result()
            group = kept['kind'] + '/' + kept['split']
            counts[group]['KEEP'] += 1
            sources[kept['kind'] + '/' + kept['source']]['KEEP'] += 1
            classes[group].update(NAMES[kept['kind']][obj['class']] for obj in kept['objects'])
            audit.execute('INSERT INTO selection VALUES (?,?,?,?,?,?,?)', (kept['key'], kept['kind'], kept['split'], 'KEEP', '[]', image, label))
            copied += 1
    audit.commit()
    for kind in NAMES:
        settings = {'path': (OUTPUT / kind).as_posix(), 'train': 'images/train', 'val': 'images/val', 'test': 'images/test', 'names': NAMES[kind]}
        (OUTPUT / kind / 'data.yaml').write_text(yaml.safe_dump(settings, sort_keys=False), encoding='utf-8')
        settings['test'] = ((ORIGINAL / 'Vehicle363-v1/images/test').as_posix() if kind == 'vehicle' else (EXPANDED / 'test_original.txt').as_posix())
        (OUTPUT / kind / 'data_original_test.yaml').write_text(yaml.safe_dump(settings, sort_keys=False), encoding='utf-8')
    total = audit.execute('SELECT count(*) FROM selection').fetchone()[0]
    if total != 164615:
        raise ValueError('Selection audit coverage mismatch')
    report = {'status': 'COMPLETE_STRICT_HEURISTIC_FILTER', 'processed': total, 'policy': STRICT,
              'base_policy': str(BASE / 'policy.json'), 'base_audit': str(REPORT / 'audit.sqlite'),
              'image_counts': {key: dict(value) for key, value in counts.items()},
              'kept_bbox_counts': {key: dict(value) for key, value in classes.items()},
              'source_counts': {key: dict(value) for key, value in sources.items()},
              'reject_reason_counts': {key: dict(value) for key, value in reject_reasons.items()},
              'image_and_label_copy_checksums_verified': True, 'original_data_modified': False,
              'active_ab_inputs_modified': False, 'training_started': False,
              'warning': 'Automatic heuristics; inspect examples. Filtered test is not the original benchmark.'}
    write_json(OUTPUT / 'summary.json', report)
    write_json(OUTPUT / 'status.json', {'status': report['status'], 'images_kept': copied})
    source.close()
    audit.close()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
