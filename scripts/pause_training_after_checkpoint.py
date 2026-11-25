import argparse
import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--completed-epoch', type=int, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    process = psutil.Process(args.pid)
    creation = process.create_time()

    def status(value):
        value.update(pid=args.pid, target_completed_epoch=args.completed_epoch,
                     checked_at_utc=datetime.now(timezone.utc).isoformat())
        temporary = args.output / 'status.tmp'
        temporary.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
        temporary.replace(args.output / 'status.json')

    while True:
        try:
            if not process.is_running() or process.create_time() != creation:
                status({'status': 'TRAINING_ALREADY_EXITED'})
                return
        except psutil.NoSuchProcess:
            status({'status': 'TRAINING_ALREADY_EXITED'})
            return
        completed = 0
        if args.results.is_file():
            with args.results.open(encoding='utf-8-sig', newline='') as handle:
                for row in csv.DictReader(handle):
                    if row.get('epoch'):
                        completed = max(completed, int(row['epoch']))
        status({'status': 'WAITING_FOR_EPOCH_CHECKPOINT', 'last_csv_epoch': completed})
        if completed >= args.completed_epoch and args.checkpoint.is_file():
            import shutil
            import torch

            torch.set_num_threads(1)
            before = args.checkpoint.stat()
            time.sleep(2)
            after = args.checkpoint.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                continue
            snapshot = args.output / 'last.pt.part'
            shutil.copyfile(args.checkpoint, snapshot)
            source_hash = sha256(args.checkpoint)
            if sha256(snapshot) != source_hash:
                time.sleep(5)
                continue
            checkpoint = torch.load(snapshot, map_location='cpu', weights_only=False)
            checkpoint_epoch = int(checkpoint.get('epoch', -1)) + 1
            if checkpoint_epoch < args.completed_epoch or checkpoint.get('optimizer') is None:
                del checkpoint
                time.sleep(5)
                continue
            del checkpoint
            snapshot.replace(args.output / 'last.pt')
            if not process.is_running() or process.create_time() != creation:
                status({'status': 'CHECKPOINT_SAVED_TRAINING_ALREADY_EXITED',
                        'completed_epoch': checkpoint_epoch, 'sha256': source_hash})
                return
            process.terminate()
            process.wait(timeout=30)
            status({'status': 'STOPPED_AFTER_CONFIRMED_CHECKPOINT',
                    'completed_epoch': checkpoint_epoch, 'sha256': source_hash,
                    'checkpoint': str((args.output / 'last.pt').resolve()),
                    'optimizer_preserved': True,
                    'note': 'Any batches of the following epoch are not retained. Resume from this checkpoint.'})
            return
        time.sleep(10)


if __name__ == '__main__':
    main()
