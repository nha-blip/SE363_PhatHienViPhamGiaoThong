import json
import os
import stat
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'runs/project363/project_cleanup_20261027'


def main():
    groups = Counter()
    cache = Counter()
    large = []
    skipped = []
    for folder, directories, files in os.walk(ROOT, followlinks=False):
        parent = Path(folder)
        for name in list(directories):
            path = parent / name
            try:
                attributes = path.lstat().st_file_attributes
            except (OSError, AttributeError):
                attributes = 0
            if attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                directories.remove(name)
                skipped.append(str(path.relative_to(ROOT)))
        for name in files:
            path = parent / name
            try:
                metadata = path.lstat()
                if stat.S_ISLNK(metadata.st_mode) or getattr(metadata, 'st_file_attributes', 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT:
                    continue
                size = metadata.st_size
            except OSError:
                skipped.append(str(path.relative_to(ROOT)))
                continue
            relative = path.relative_to(ROOT)
            depth = 2 if relative.parts[0] in {'data', 'weights', 'exports'} else 3 if relative.parts[:2] == ('runs', 'project363') else 1
            group = '/'.join(relative.parts[:min(depth, len(relative.parts))])
            groups[group] += size
            if size >= 1024 ** 3:
                large.append({'path': relative.as_posix(), 'bytes': size})
            if relative.parts[0] in {'scripts', 'src', 'tests', '__pycache__', '.pytest_cache'}:
                for index, part in enumerate(relative.parts):
                    if part in {'__pycache__', '.pytest_cache'}:
                        cache['/'.join(relative.parts[:index + 1])] += size
                        break
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = {'surveyed_at_utc': datetime.now(timezone.utc).isoformat(),
              'size_type': 'logical file bytes, not allocated disk usage',
              'total_bytes': sum(groups.values()), 'groups': dict(groups.most_common()),
              'regenerable_code_caches': dict(cache), 'large_files': sorted(large, key=lambda item: -item['bytes']),
              'skipped_links_or_unreadable': skipped, 'datasets_deleted': False}
    (OUTPUT / 'inventory.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    lines = ['# Khảo sát dọn dẹp dự án', '', 'Dung lượng logic; không đi theo symlink/junction.', '',
             '| Thư mục | GiB |', '| --- | ---: |']
    lines.extend(f'| `{name}` | {size / 1024 ** 3:.3f} |' for name, size in groups.most_common())
    lines.extend(['', '## Cache mã có thể tạo lại', ''])
    lines.extend(f'- `{name}`: {size / 1024 ** 2:.2f} MiB' for name, size in cache.items())
    lines.extend(['', '## Chưa xóa', '',
                  'Giữ dataset gốc, bản balanced, QA, checkpoint, mã nguồn, cấu hình và bài cá nhân.',
                  'Bản lọc sơ bộ/strict và gói zip cần xác nhận riêng trước khi xóa vì có thể đang được tham chiếu.',
                  'Không tự xóa notebook checkpoint hoặc file xác thực.'])
    (OUTPUT / 'cleanup_plan.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'total_GiB': sum(groups.values()) / 1024 ** 3,
                      'largest': dict(groups.most_common(18)), 'cache_bytes': sum(cache.values())}), flush=True)


if __name__ == '__main__':
    main()
