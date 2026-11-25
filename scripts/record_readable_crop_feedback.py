import hashlib
import json
import sqlite3
from collections import Counter
from pathlib import Path

if __package__:
    from .filter_detector_datasets_720p import ROOT, write_json
else:
    from filter_detector_datasets_720p import ROOT, write_json


FOLDER = ROOT / 'data/YOLO363_readable_crops_20261027/quality_review'
FEEDBACK = 'Đèn KEEP đều nhìn rõ; đèn REJECT tuy độ phân giải thấp vẫn đoán được nhờ màu. Xe KEEP nhìn rõ; xe REJECT hoàn toàn không nhìn thấy.'


def main():
    path = FOLDER / 'human_review.json'
    original = path.read_bytes()
    data = json.loads(original)
    groups = Counter((row['kind'], row['automatic_decision']) for row in data['reviews'])
    expected = {('light', 'KEEP'): 16, ('light', 'REJECT'): 16,
                ('vehicle', 'KEEP'): 16, ('vehicle', 'REJECT'): 12}
    if dict(groups) != expected:
        raise ValueError('Unexpected sample inventory; do not apply feedback to another batch')
    if data.get('chat_feedback') == FEEDBACK:
        print('Feedback already saved')
        return
    if any(row['decision'] != 'PENDING' for row in data['reviews']):
        raise ValueError('Existing individual decisions require reconciliation before batch update')
    backup = FOLDER / ('human_review.before_chat_' + hashlib.sha256(original).hexdigest()[:12] + '.json')
    if not backup.exists():
        backup.write_bytes(original)
    audit = sqlite3.connect(f'file:{(FOLDER.parent / "audit.sqlite").as_posix()}?mode=ro', uri=True)
    reason_counts = Counter()
    for row in data['reviews']:
        if row['automatic_decision'] == 'KEEP':
            row.update(decision='CLEAR', human_feedback='User confirms the displayed retained sample is clear')
        elif row['kind'] == 'vehicle':
            row.update(decision='UNREADABLE', human_feedback='User cannot see/recognize the displayed rejected vehicle')
        else:
            row.update(decision='UNCERTAIN', color_cue_visible=True,
                       human_feedback='User can infer color despite low resolution; exact state, object identity and mapping not individually confirmed',
                       needs_light_filter_policy_review=True)
            record = json.loads(audit.execute('SELECT record FROM frames WHERE identity=?', (row['audit_identity'],)).fetchone()[0])
            reason_counts.update(record['reasons'])
        row['decision_source'] = 'User chat feedback, interpreted per displayed sample, not all frames'
    audit.close()
    data.update(status='COMPLETE_SAMPLE_FEEDBACK_REQUIRES_LIGHT_POLICY_REVIEW', chat_feedback=FEEDBACK,
                light_reject_interpretation='UNCERTAIN is a conservative encoding of can infer from color, not explicit per-sample color-state confirmation',
                sample_only=True, dataset_modified=False, training_started=False)
    write_json(path, data)
    report = {'status': data['status'], 'sample_decisions': dict(Counter(row['decision'] for row in data['reviews'])),
              'light_rejected_sample_reasons': dict(reason_counts), 'prior_review_backup': str(backup),
              'feedback': FEEDBACK, 'dataset_modified': False, 'training_started': False,
              'recommendation': 'Vehicle policy supported by inspected samples; reconsider light size-only rejection, retaining real signals only when state is identifiable, without overriding QA/invalid geometry holds'}
    write_json(FOLDER / 'feedback_summary.json', report)
    print(json.dumps(report, ensure_ascii=True, indent=2))


if __name__ == '__main__':
    main()
