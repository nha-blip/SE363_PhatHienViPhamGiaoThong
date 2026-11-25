import argparse
import hashlib
import json
import re
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def digest(path, algorithm):
    result = hashlib.new(algorithm)
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def review(arguments):
    def remote(*command):
        result = subprocess.run(
            [arguments.rclone, *command, '--drive-root-folder-id', arguments.folder_id,
             '--retries', '3', '--contimeout', '20s', '--timeout', '2m',
             '--log-level', 'ERROR'],
            check=True, capture_output=True, text=True, encoding='utf-8', timeout=900,
        )
        return result.stdout

    def read_json(path):
        return json.loads(remote('cat', f'gdrive2:{path}'))

    items = json.loads(arguments.inventory.read_text(encoding='utf-8-sig'))
    paths = {item['Path']: item for item in items}
    runs = defaultdict(list)
    for item in items:
        parts = item['Path'].split('/')
        if len(parts) >= 4 and parts[0] in ('nano', 'small') and parts[2] in ('vehicle', 'traffic_light'):
            runs['/'.join(parts[:3])].append(item)

    candidates = defaultdict(list)
    for run, entries in runs.items():
        variant, run_id, task = run.split('/')
        epochs = []
        for item in entries:
            match = re.search(r'/epoch_(\d+)/last.pt$', item['Path'])
            if match and item['Path'].replace('last.pt', 'UPLOAD_COMPLETE.json') in paths:
                epochs.append(int(match.group(1)))
        markers = [item for item in entries if item['Name'] == 'TRAIN_COMPLETE.json'
                   or item['Path'].endswith('/final/UPLOAD_COMPLETE.json')]
        marker = max(markers, key=lambda item: item['ModTime']) if markers else None
        candidates[(variant, task)].append({
            'run': run, 'run_id': run_id, 'confirmed_epoch': max(epochs, default=None),
            'completion_marker': marker['Path'] if marker else None,
            'modified': max(item['ModTime'] for item in entries),
        })

    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'source_folder_id': arguments.folder_id, 'remote_files_modified': False,
              'notes': ['Completion markers do not establish the early-stopping reason.',
                        'No completion marker does not prove a job is currently running.',
                        'best.pt is selected by training fitness, not cross-run comparison.'],
              'models': []}
    for (variant, task), entries in sorted(candidates.items()):
        latest = max(entries, key=lambda entry: entry['modified'])
        result = {'variant': variant, 'task': task, 'latest_run': latest}
        completed = [entry for entry in entries if entry['completion_marker']]
        if not completed:
            result['status'] = 'NO_CONFIRMED_COMPLETION_NOT_DOWNLOADED'
            if latest['confirmed_epoch'] is not None:
                receipt_path = f"{latest['run']}/epoch_{latest['confirmed_epoch']:04d}/UPLOAD_COMPLETE.json"
                result['latest_receipt'] = read_json(receipt_path)
            report['models'].append(result)
            print(f"{variant}/{task}: no completion marker; latest confirmed epoch {latest['confirmed_epoch']}", flush=True)
            continue

        selected = max(completed, key=lambda entry: paths[entry['completion_marker']]['ModTime'])
        marker = read_json(selected['completion_marker'])
        result.update(selected_run=selected['run'], completion=marker)
        if selected['completion_marker'].endswith('/final/UPLOAD_COMPLETE.json'):
            best_path = f"{selected['run']}/final/best.pt"
            receipt = marker
        else:
            best_path = f"{selected['run']}/best.pt"
            epoch = selected['confirmed_epoch']
            receipt = read_json(f"{selected['run']}/epoch_{epoch:04d}/UPLOAD_COMPLETE.json")
        result['checkpoint_receipt'] = receipt
        metadata = json.loads(remote('lsjson', f'gdrive2:{best_path}', '--stat', '--hash'))
        target_dir = arguments.output / variant / task
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / 'best.pt'
        temporary = target_dir / 'best.pt.part'
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite existing model: {target}')
        remote('copyto', f'gdrive2:{best_path}', str(temporary))
        expected_md5 = metadata.get('Hashes', {}).get('md5')
        if not expected_md5:
            raise RuntimeError(f'Remote MD5 unavailable for {best_path}')
        if temporary.stat().st_size != metadata['Size'] or digest(temporary, 'md5') != expected_md5:
            raise RuntimeError(f'Remote size/MD5 verification failed: {best_path}')
        sha256 = digest(temporary, 'sha256')
        expected_sha256 = (receipt.get('best_sha256_at_backup')
                           or receipt.get('files', {}).get('best.pt', {}).get('sha256'))
        if expected_sha256 and sha256 != expected_sha256:
            raise RuntimeError(f'Receipt SHA256 mismatch: {best_path}')
        temporary.replace(target)
        result.update(status='DOWNLOADED_VERIFIED', remote_path=best_path,
                      local_path=str(target.resolve()), bytes=metadata['Size'],
                      sha256=sha256, md5=expected_md5,
                      receipt_sha256_verified=bool(expected_sha256))
        (target_dir / 'provenance.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        report['models'].append(result)
        print(f'{variant}/{task}: downloaded and verified {target}', flush=True)

    arguments.report.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    lines = ['# Drive training review', '', f"Checked: {report['checked_at_utc']}", '',
             '| Variant | Task | Latest confirmed epoch | Status |', '|---|---|---:|---|']
    for model in report['models']:
        lines.append(f"| {model['variant']} | {model['task']} | {model['latest_run']['confirmed_epoch']} | {model['status']} |")
    lines += ['', 'Remote files were not modified or deleted.',
              'Downloads verified against remote MD5 and size; receipt SHA256 checked when present.',
              'Completion may reflect early stopping; metrics/logs are needed to establish the reason.',
              'Training activity cannot be determined from backup files alone.']
    arguments.report.with_suffix('.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rclone', required=True)
    parser.add_argument('--folder-id', required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    review(parser.parse_args())
