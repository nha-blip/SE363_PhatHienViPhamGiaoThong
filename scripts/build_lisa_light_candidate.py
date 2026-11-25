import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'data/LISA_traffic_light_raw_20261030/lisa-traffic-light-dataset.zip'
OUTPUT = ROOT / 'data/LISA_light_candidate_20261030'
NAMES = ['red', 'yellow', 'green', 'off']
TAG_TO_CLASS = {'stop': 0, 'warning': 1, 'go': 2}
ARROW_TAGS = {'stopLeft', 'goLeft', 'warningLeft', 'goForward'}
HOLDOUT_SEQUENCES = {'daySequence2', 'nightSequence2'}
WIDTH, HEIGHT = 1280, 960
STRIDE_DEFAULT = 32
STRIDE_YELLOW = 8


def sequence_of(annotation_path):
    """Return the sequence for one BOX file; the duplicated sample-* folders are rejected."""
    parts = PurePosixPath(annotation_path).parts
    if len(parts) < 4 or parts[0] != 'Annotations' or parts[-1] != 'frameAnnotationsBOX.csv':
        return None
    sequence = parts[-2]
    return None if sequence.startswith('sample-') else sequence


def read_annotations(text):
    """Group BOX rows by frame file name, keeping the source frame number."""
    frames = defaultdict(lambda: {'frame': None, 'rows': []})
    for row in csv.DictReader(io.StringIO(text), delimiter=';'):
        name = PurePosixPath(row['Filename'].replace('\\', '/')).name
        item = frames[name]
        item['frame'] = int(row['Origin frame number'])
        item['rows'].append((row['Annotation tag'], int(row['Upper left corner X']), int(row['Upper left corner Y']),
                             int(row['Lower right corner X']), int(row['Lower right corner Y'])))
    return frames


def usable_frame(rows):
    """Reject a whole frame with an arrow or unknown tag so no signal is left unlabeled as background."""
    return bool(rows) and all(row[0] in TAG_TO_CLASS for row in rows)


def thin_frames(frames, default_stride=STRIDE_DEFAULT, yellow_stride=STRIDE_YELLOW):
    """Keep sparse frames per sequence; yellow is short and rare, so it uses a smaller time stride."""
    selected = []
    last_default = last_yellow = None
    for name, item in sorted(frames.items(), key=lambda pair: pair[1]['frame']):
        if not usable_frame(item['rows']):
            continue
        frame = item['frame']
        if any(row[0] == 'warning' for row in item['rows']):
            if last_yellow is None or frame - last_yellow >= yellow_stride:
                selected.append(name)
                last_yellow = frame
        elif last_default is None or frame - last_default >= default_stride:
            selected.append(name)
            last_default = frame
    return selected


def yolo_rows(rows, width=WIDTH, height=HEIGHT):
    """Convert corner boxes to normalized YOLO rows; clipping is explicit and an empty result raises."""
    converted = []
    for tag, left, top, right, bottom in rows:
        left, right = max(0, left), min(width, right)
        top, bottom = max(0, top), min(height, bottom)
        if right <= left or bottom <= top:
            raise ValueError('Degenerate LISA box')
        converted.append((TAG_TO_CLASS[tag], (left + right) / 2 / width, (top + bottom) / 2 / height,
                          (right - left) / width, (bottom - top) / height))
    return converted


def short_side_bucket(row):
    short = min(row[3] * WIDTH, row[4] * HEIGHT) * 0.5
    return '<8' if short < 8 else '8-16' if short < 16 else '16-32' if short < 32 else '>=32'


def main():
    global OUTPUT, STRIDE_DEFAULT, STRIDE_YELLOW
    parser = argparse.ArgumentParser(description='Build a thinned LISA red/yellow/green candidate; never merged or trained')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--default-stride', type=int, default=STRIDE_DEFAULT, help='min frame gap between kept frames without yellow')
    parser.add_argument('--yellow-stride', type=int, default=STRIDE_YELLOW, help='min frame gap between kept frames with yellow')
    arguments = parser.parse_args()
    OUTPUT, STRIDE_DEFAULT, STRIDE_YELLOW = arguments.output, arguments.default_stride, arguments.yellow_stride
    expected = json.loads((ARCHIVE.parent / 'download_provenance.json').read_text(encoding='utf-8'))['sha256']
    digest = hashlib.sha256()
    with ARCHIVE.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 24), b''):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError('LISA archive SHA256 mismatch')
    OUTPUT.mkdir(exist_ok=False)
    for split in ('train', 'holdout'):
        (OUTPUT / 'images' / split).mkdir(parents=True)
        (OUTPUT / 'labels' / split).mkdir(parents=True)
    manifest, counts, sizes, sequences, rejected = [], Counter(), defaultdict(Counter), Counter(), Counter()
    with zipfile.ZipFile(ARCHIVE) as archive:
        images = defaultdict(list)
        for member in archive.namelist():
            path = PurePosixPath(member)
            if path.suffix == '.jpg' and not path.parts[0].startswith('sample-'):
                images[path.name].append(member)
        for member in sorted(archive.namelist()):
            sequence = sequence_of(member)
            if sequence is None:
                continue
            frames = read_annotations(archive.read(member).decode())
            rejected[sequence] = sum(1 for item in frames.values() if not usable_frame(item['rows']))
            split = 'holdout' if sequence in HOLDOUT_SEQUENCES else 'train'
            for name in thin_frames(frames, STRIDE_DEFAULT, STRIDE_YELLOW):
                if len(images[name]) != 1:
                    raise ValueError('Expected one image payload: ' + name)
                rows = yolo_rows(frames[name]['rows'])
                (OUTPUT / 'images' / split / name).write_bytes(archive.read(images[name][0]))
                (OUTPUT / 'labels' / split / (PurePosixPath(name).stem + '.txt')).write_text(
                    ''.join(f'{row[0]} ' + ' '.join(f'{value:.8f}' for value in row[1:]) + '\n' for row in rows))
                for row in rows:
                    counts[(split, NAMES[row[0]])] += 1
                    sizes[(split, NAMES[row[0]])][short_side_bucket(row)] += 1
                sequences[(split, sequence)] += 1
                manifest.append({'image': name, 'split': split, 'sequence': sequence, 'source_frame': frames[name]['frame'],
                                 'source_member': images[name][0], 'classes': dict(Counter(NAMES[row[0]] for row in rows))})
    (OUTPUT / 'manifest.json').write_text(json.dumps(manifest, indent=1), encoding='utf-8')
    summary = {'status': 'CANDIDATE_NOT_MERGED_NOT_TRAINED', 'is_ground_truth_verified': False, 'source': 'LISA Traffic Light Dataset',
               'license': 'CC BY-NC-SA 4.0', 'archive_sha256': expected, 'annotation': 'frameAnnotationsBOX, whole signal area',
               'class_map': TAG_TO_CLASS, 'arrow_policy': sorted(ARROW_TAGS), 'holdout_sequences': sorted(HOLDOUT_SEQUENCES),
               'stride_frames': {'default': STRIDE_DEFAULT, 'yellow': STRIDE_YELLOW}, 'images_written_unmodified': True,
               'frames_by_split': dict(Counter(row['split'] for row in manifest)),
               'bbox': {f'{key[0]}/{key[1]}': value for key, value in sorted(counts.items())},
               'short_side_at_640': {f'{key[0]}/{key[1]}': dict(value) for key, value in sorted(sizes.items())},
               'frames_by_sequence': {f'{key[0]}/{key[1]}': value for key, value in sorted(sequences.items())},
               'annotated_frames_rejected_arrow_by_sequence': dict(rejected)}
    (OUTPUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == '__main__':
    main()
