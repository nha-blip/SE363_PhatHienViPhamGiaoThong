import argparse
import concurrent.futures
import hashlib
import json
import shutil
import time
import zipfile
from pathlib import Path
from urllib.parse import quote

import requests


ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=True, indent=2), encoding='utf-8')
    temporary.replace(path)


def metadata(url):
    response = requests.get(url, timeout=90)
    response.raise_for_status()
    return response.json()


def digest(path, algorithm):
    checksum = hashlib.new(algorithm)
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def download(record, output):
    folder = output / record['dataset']
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / record['filename']
    partial = path.with_suffix(path.suffix + '.partial')
    provenance = path.with_suffix(path.suffix + '.provenance.json')
    record = dict(record, status='DOWNLOADING')
    write_json(provenance, record)
    try:
        if not path.exists():
            for attempt in range(1, 5):
                try:
                    offset = partial.stat().st_size if partial.exists() else 0
                    if offset == record['bytes']:
                        break
                    headers = {'Accept-Encoding': 'identity'}
                    if offset:
                        headers['Range'] = f'bytes={offset}-'
                    with requests.get(record['url'], headers=headers, stream=True, timeout=(30, 120)) as response:
                        response.raise_for_status()
                        if response.status_code == 206:
                            if not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-'):
                                raise ValueError('Unexpected Content-Range')
                        else:
                            offset = 0
                        with partial.open('ab' if offset else 'wb') as handle:
                            previous = time.monotonic()
                            for block in response.iter_content(4 * 1024 * 1024):
                                if block:
                                    handle.write(block)
                                    offset += len(block)
                                if time.monotonic() - previous > 15:
                                    print(f"{record['filename']}: {offset}/{record['bytes']} bytes", flush=True)
                                    previous = time.monotonic()
                    if partial.stat().st_size != record['bytes']:
                        raise ValueError('Downloaded size mismatch')
                    break
                except Exception:
                    if attempt == 4:
                        raise
                    time.sleep(attempt * 3)
            candidate = partial
        else:
            candidate = path
        if candidate.stat().st_size != record['bytes']:
            raise ValueError('Size mismatch; existing file left untouched')
        actual = digest(candidate, record['checksum_algorithm'])
        if actual != record['expected_checksum']:
            raise ValueError('Source checksum mismatch; file left for inspection')
        with zipfile.ZipFile(candidate) as archive:
            bad_member = archive.testzip()
            if bad_member:
                raise ValueError(f'Corrupt ZIP member: {bad_member}')
            entries = archive.infolist()
            record['archive'] = {
                'files': sum(not entry.is_dir() for entry in entries),
                'images': sum(Path(entry.filename).suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp'} for entry in entries),
                'xml_annotations': sum(entry.filename.lower().endswith('.xml') for entry in entries),
                'uncompressed_bytes': sum(entry.file_size for entry in entries),
                'sample_paths': [entry.filename for entry in entries[:20]],
            }
        if candidate == partial:
            partial.replace(path)
        record.update(status='VERIFIED', sha256=digest(path, 'sha256'), local_path=str(path))
    except Exception as error:
        record.update(status='FAILED', error=str(error))
    write_json(provenance, record)
    print(f"{record['filename']}: {record['status']}", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT / 'data/traffic_light_external_raw')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    repository = metadata('https://huggingface.co/api/datasets/yangxue/S2TLD')
    revision = repository['sha']
    files = metadata(f'https://huggingface.co/api/datasets/yangxue/S2TLD/tree/{revision}')
    atlas = metadata('https://zenodo.org/api/records/14846709')
    write_json(args.output / 'source_metadata.json', {'s2tld': repository, 'atlas': atlas})
    records = []
    for item in files:
        if item['path'].endswith('.zip'):
            resolution = '1080x1920' if '1080x1920' in item['path'] else '720x1280'
            records.append({
                'dataset': 'S2TLD', 'filename': f'S2TLD_{resolution}.zip',
                'original_filename': item['path'], 'revision': revision,
                'url': f"https://huggingface.co/datasets/yangxue/S2TLD/resolve/{revision}/{quote(item['path'])}",
                'bytes': item['size'], 'checksum_algorithm': 'sha256',
                'expected_checksum': item['lfs']['oid'],
            })
    item = next(item for item in atlas['files'] if item['key'] == 'ATLAS_sample.zip')
    records.append({
        'dataset': 'ATLAS_sample', 'filename': item['key'],
        'url': item['links']['self'], 'bytes': item['size'],
        'checksum_algorithm': 'md5', 'expected_checksum': item['checksum'].split(':', 1)[1],
    })
    if shutil.disk_usage(args.output).free < sum(item['bytes'] for item in records) + 1024 ** 3:
        raise OSError('Insufficient disk space')
    write_json(args.output / 'download_plan.json', records)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda record: download(record, args.output), records))
    write_json(args.output / 'manifest.json', {
        'status': 'COMPLETE' if all(item['status'] == 'VERIFIED' for item in results) else 'INCOMPLETE',
        'files': results, 'atlas_scope': 'SAMPLE_ONLY',
        'extracted': False, 'merged_into_training': False,
    })
    if any(item['status'] != 'VERIFIED' for item in results):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
