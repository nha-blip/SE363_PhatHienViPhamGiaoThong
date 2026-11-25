import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

if __package__:
    from .filter_detector_datasets_720p import ROOT, write_json
else:
    from filter_detector_datasets_720p import ROOT, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--expected-keep', type=int, required=True)
    parser.add_argument('--expected-reject', type=int, required=True)
    args = parser.parse_args()
    folder = args.folder.resolve()
    folder.relative_to(ROOT)
    path = folder / 'human_review.json'
    original = path.read_bytes()
    review = json.loads(original)
    groups = Counter(row['automatic_decision'] for row in review['reviews'])
    if dict(groups) != {'KEEP': args.expected_keep, 'REJECT': args.expected_reject}:
        raise ValueError('Sample counts differ from confirmed batch')
    for row in review['reviews']:
        expected = 'CLEAR' if row['automatic_decision'] == 'KEEP' else 'UNREADABLE'
        if row['decision'] not in {'PENDING', expected}:
            raise ValueError('Individual feedback conflicts with batch confirmation')
    backup = folder / ('human_review.before_confirmation_' + hashlib.sha256(original).hexdigest()[:12] + '.json')
    if review.get('status') != 'COMPLETE_SAMPLE_READABILITY_QA':
        if not backup.exists():
            backup.write_bytes(original)
        for row in review['reviews']:
            row['decision'] = 'CLEAR' if row['automatic_decision'] == 'KEEP' else 'UNREADABLE'
            row['decision_source'] = 'User chat: retained samples are clear; rejected samples are not clear'
        review.update(status='COMPLETE_SAMPLE_READABILITY_QA', scope='Readability of displayed samples only; not full manual GT',
                      confirmed_at_utc=datetime.now(timezone.utc).isoformat(), dataset_modified=False)
        write_json(path, review)
    receipt = {'status': 'SAMPLE_QA_APPROVED_FOR_CONTROLLED_TRAINING',
               'review_file': str(path), 'review_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
               'sample_counts': dict(groups), 'readability_decisions': dict(Counter(row['decision'] for row in review['reviews'])),
               'scope': 'Sample review and filter policy approval, not manual verification of all images/classes/bboxes',
               'source_dataset_modified': False, 'training_started': False,
               'generation_status_preserved': True}
    write_json(folder.parent / 'sample_qa_approval.json', receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
