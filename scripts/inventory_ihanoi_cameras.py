import argparse
import json
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ADB = Path('C:/Program Files/BlueStacks_nxt/HD-Adb.exe')
OUTPUT = ROOT / 'data/iHanoi_capture/city_inventory_20261029'


def camera_rows(xml):
    tree = ET.fromstring(xml)
    nodes = list(tree.iter('node'))
    titles = {node.get('content-desc', '') or node.get('text', '') for node in nodes}
    if 'Camera giao thông' not in titles:
        raise ValueError('Expected camera listing; no navigation or login will be attempted')
    rows = {}
    for node in nodes:
        description = node.get('content-desc', '') or node.get('text', '')
        match = re.search(r'\bC\d{2,}(?:\.\d+)*(?:-[A-Za-z0-9]+)?\b', description)
        if not match or node.get('clickable') != 'true':
            continue
        lines = description.splitlines()
        identity = match.group()
        rows[identity] = {'camera_id': identity, 'name': lines[0],
                          'address': lines[1] if len(lines) > 1 else '',
                          'description': description, 'ui_bounds': node.get('bounds'),
                          'status': 'DISCOVERED_NOT_RECORDED'}
    return rows


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    temporary.replace(path)


def unopened_section(xml, opened):
    for node in ET.fromstring(xml).iter('node'):
        description = node.get('content-desc', '')
        if not description.startswith(('Phường ', 'Xã ')) or '\n' in description:
            continue
        bounds = [int(value) for value in re.findall(r'\d+', node.get('bounds', ''))]
        if (description not in opened and node.get('clickable') == 'true' and len(bounds) == 4
                and bounds[1] >= 284 and bounds[3] <= 1550 and bounds[3] - bounds[1] >= 50):
            return description, (bounds[0] + bounds[2]) // 2, (bounds[1] + bounds[3]) // 2
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--serial', default='emulator-5554')
    parser.add_argument('--max-pages', type=int, default=300)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'emulator-\d+', args.serial) or not 1 <= args.max_pages <= 500:
        raise ValueError('Unexpected device or page budget')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    catalog = OUTPUT / 'cameras.json'
    if catalog.exists() and not args.resume:
        raise ValueError('Inventory already exists; inspect it before starting another scan')
    known = json.loads((ROOT / 'data/iHanoi_capture/capture_catalog.json').read_text(encoding='utf-8'))
    decisions = {row['camera_id']: row for row in known['videos']}
    previous = json.loads(catalog.read_text(encoding='utf-8')) if catalog.exists() else {}
    rows = {row['camera_id']: row for row in previous.get('cameras', [])}
    opened = set(previous.get('sections_opened', ['Phường Ba Đình']))
    signatures = set()
    repeated = 0

    def adb(*arguments):
        result = subprocess.run([str(ADB), '-s', args.serial, *arguments],
                                capture_output=True, timeout=45)
        if result.returncode:
            raise RuntimeError('ADB command failed; inventory stopped')
        return result.stdout

    def progress(status, page, reason=None):
        save(catalog, {'status': status, 'updated_at_utc': datetime.now(timezone.utc).isoformat(),
             'serial': args.serial, 'pages_read': page, 'unique_cameras': len(rows),
             'sections_opened': sorted(opened),
             'all_city_coverage_verified': False, 'recording_started': False,
             'reason': reason, 'cameras': list(rows.values())})
        print(json.dumps({'status': status, 'pages': page, 'cameras': len(rows)}), flush=True)

    page = 0
    try:
        if args.resume:
            for attempt in range(8):
                adb('shell', 'input', 'swipe', '450', '450', '450', '1450', '300')
        start = previous.get('pages_read', 0) + 1
        for page in range(start, start + args.max_pages):
            if (OUTPUT / 'STOP').exists():
                progress('STOP_REQUESTED', page - 1)
                return
            adb('shell', 'uiautomator', 'dump', '/sdcard/ihanoi_city_inventory.xml')
            raw = adb('shell', 'cat', '/sdcard/ihanoi_city_inventory.xml')
            parsed = camera_rows(raw)
            (OUTPUT / f'page_{page:04d}.xml').write_bytes(raw)
            for identity, row in parsed.items():
                if identity not in rows:
                    row['first_page'] = page
                    if identity in decisions:
                        row['existing_review'] = decisions[identity]
                        row['status'] = decisions[identity]['status']
                    rows[identity] = row
            section = unopened_section(raw, opened)
            if section:
                name, horizontal, vertical = section
                opened.add(name)
                progress('EXPANDING_SECTION_LIST_ONLY', page)
                adb('shell', 'input', 'tap', str(horizontal), str(vertical))
                time.sleep(0.7)
                repeated = 0
                continue
            signature = tuple((node.get('content-desc', ''), node.get('bounds', ''))
                              for node in ET.fromstring(raw).iter('node') if node.get('content-desc'))
            repeated = repeated + 1 if signature in signatures else 0
            signatures.add(signature)
            progress('SCANNING_LIST_ONLY', page)
            if repeated >= 3:
                progress('STABLE_LIST_REACHED_COVERAGE_REQUIRES_REVIEW', page,
                         'Three repeated views; may be list end or a UI scroll limit. Not proof all city cameras were enumerated.')
                return
            adb('shell', 'input', 'swipe', '450', '1450', '450', '450', '500')
            time.sleep(0.7)
        progress('PAGE_BUDGET_REACHED', page)
    except Exception as error:
        progress('STOPPED_SAFELY', page, str(error))
        raise


if __name__ == '__main__':
    main()
