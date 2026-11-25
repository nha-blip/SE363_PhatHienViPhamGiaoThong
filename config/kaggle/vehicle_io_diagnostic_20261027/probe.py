import faulthandler
import hashlib
import json
import os
import random
import threading
import time
from pathlib import Path


OUTPUT = Path('/kaggle/working/vehicle_io_diagnostic')
SOURCE = Path('/kaggle/input/datasets/v5nhdf/yolo363-readable-crops-20261027/vehicle_trainval/vehicle')


def emit(data):
    print('VEHICLE_IO_PROBE ' + json.dumps(data), flush=True)


def bounded(name, function, seconds=180):
    result = {'stage': name}
    started = time.monotonic()

    def run():
        try:
            result['result'] = function()
        except Exception as error:
            result['error_type'] = type(error).__name__

    thread = threading.Thread(target=run, daemon=True)
    emit({'stage': name, 'state': 'START'})
    thread.start()
    thread.join(seconds)
    result.update(completed=not thread.is_alive(), elapsed_seconds=time.monotonic()-started)
    reported = dict(result)
    if isinstance(reported.get('result'), list):
        reported['result'] = {'item_count': len(reported['result'])}
    emit(reported)
    (OUTPUT / (name + '.json')).write_text(json.dumps(reported, indent=2))
    return result


def main():
    OUTPUT.mkdir()
    faulthandler.enable()
    faulthandler.dump_traceback_later(45, repeat=True)
    emit({'cpu_only': True, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'source': str(SOURCE)})
    listing = bounded('list_images', lambda: sorted(str(path) for path in (SOURCE / 'images/train').iterdir()), 60)
    if not listing['completed'] or 'result' not in listing:
        return
    images = listing.pop('result')
    (OUTPUT / 'list_images.json').write_text(json.dumps({**listing, 'image_count': len(images)}, indent=2))
    if len(images) != 165233:
        emit({'error': 'Unexpected source image count', 'count': len(images)})
        return
    generator = random.Random(363)
    sample = generator.sample(images, 500)
    labels = [path.replace('/images/', '/labels/').rsplit('.', 1)[0]+'.txt' for path in images]

    def stat_sample(paths):
        sizes = []
        for path in paths:
            sizes.append(os.stat(path).st_size)
        return {'files': len(paths), 'total_bytes': sum(sizes)}

    results = [bounded('stat_500_images', lambda: stat_sample(sample), 60),
               bounded('stat_500_labels', lambda: stat_sample([path.replace('/images/', '/labels/').rsplit('.', 1)[0]+'.txt' for path in sample]), 60)]
    cache = SOURCE / 'labels/train.cache'
    results.append(bounded('source_cache_presence', lambda: {'exists': cache.exists()}, 30))
    if not all(result['completed'] for result in results):
        emit({'state': 'STOP_AFTER_BLOCKED_SAMPLE', 'source_modified': False})
        return

    def native_hash():
        digest = hashlib.sha256()
        for index, path in enumerate(labels+images):
            digest.update(path.encode())
            digest.update(b'\0')
            try:
                metadata = os.stat(path)
            except OSError:
                digest.update(b'\0')
                continue
            digest.update(f'{metadata.st_size}:{metadata.st_mtime_ns}'.encode())
            digest.update(b'\0')
            if index % 10000 == 0:
                emit({'stage': 'exact_metadata_hash', 'processed': index, 'total': len(images)*2})
        return {'files': len(images)*2, 'sha256': digest.hexdigest()}

    results.append(bounded('exact_metadata_hash', native_hash, 240))

    def read_labels():
        total = 0
        for path in labels[:1000]:
            total += len(Path(path).read_bytes())
        return {'files': min(1000, len(labels)), 'bytes': total}

    results.append(bounded('read_1000_labels', read_labels, 90))
    (OUTPUT / 'summary.json').write_text(json.dumps({'results': results, 'source_modified': False}, indent=2))
    faulthandler.cancel_dump_traceback_later()
    emit({'state': 'COMPLETE', 'source_modified': False, 'gpu_training_touched': False})


if __name__ == '__main__':
    main()
