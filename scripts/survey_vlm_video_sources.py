import argparse
import hashlib
import html
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter

import cv2
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'runs/project363/vlm/source_survey'
FRACTIONS = (0, 0.01, 0.05, 0.15, 0.30, 0.50, 0.70, 0.90, 0.98)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path, algorithm='sha256'):
    result = hashlib.new(algorithm)
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def sample_video(source, folder):
    folder.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(source))
    result = {'opened': capture.isOpened(), 'samples': [], 'decode_windows': []}
    if not capture.isOpened():
        capture.release()
        write_json(folder / 'decoder.json', result)
        return
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    result.update(reported_frame_count=count, nominal_fps=fps, width=width, height=height,
                  estimated_duration_s=count / fps if fps > 0 else None,
                  backend=capture.getBackendName(), full_decode_verified=False,
                  timebase_verified=False)
    tiles = []
    for slot, fraction in enumerate(FRACTIONS):
        requested = min(max(count - 1, 0), int(max(count - 1, 0) * fraction))
        seek_ok = capture.set(cv2.CAP_PROP_POS_FRAMES, requested)
        decoded, frame = capture.read()
        sample = {'slot': slot, 'requested_frame_zero_based': requested,
                  'seek_ok': seek_ok, 'decoded': decoded,
                  'reported_frame_zero_based': capture.get(cv2.CAP_PROP_POS_FRAMES) - 1,
                  'decoder_timestamp_ms': capture.get(cv2.CAP_PROP_POS_MSEC)}
        tile = Image.new('RGB', (640, 400), '#181818')
        draw = ImageDraw.Draw(tile)
        if decoded:
            frame_path = folder / f'frame_{requested:08d}.jpg'
            picture = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            picture.save(frame_path, quality=94)
            sample['image'] = relative(frame_path)
            picture.thumbnail((640, 360))
            tile.paste(picture, ((640 - picture.width) // 2, 36))
        label = f'{source.stem} | frame {requested} | {requested / fps:.2f}s nominal' if fps > 0 else f'frame {requested}'
        draw.text((8, 8), label, fill='white')
        if not decoded:
            draw.text((20, 180), 'DECODE FAILED', fill='red')
        result['samples'].append(sample)
        tiles.append(tile)
    for position in sorted({0, count // 2, max(0, count - 5)}):
        capture.set(cv2.CAP_PROP_POS_FRAMES, position)
        window = {'requested_start_frame_zero_based': position, 'attempted': 0,
                  'decoded': 0, 'decoder_timestamps_ms': []}
        for offset in range(min(5, max(count - position, 0))):
            decoded, frame = capture.read()
            window['attempted'] += 1
            if not decoded:
                break
            window['decoded'] += 1
            window['decoder_timestamps_ms'].append(capture.get(cv2.CAP_PROP_POS_MSEC))
        result['decode_windows'].append(window)
    capture.release()
    sheet = Image.new('RGB', (1920, 1200), '#181818')
    for slot, tile in enumerate(tiles):
        sheet.paste(tile, ((slot % 3) * 640, (slot // 3) * 400))
    sheet_path = folder / 'contact_sheet.jpg'
    sheet.save(sheet_path, quality=92)
    result['contact_sheet'] = relative(sheet_path)
    write_json(folder / 'decoder.json', result)


def inventory():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sources = sorted((ROOT / 'data/Dataset_Traffic_Evaluation').glob('*.mp4'))
    sources.append(ROOT / 'data/Vehicle_Tracking/VNTraffic/VNTraffic_Original-video.mp4')
    records = []
    for index, source in enumerate(sources, start=1):
        folder = OUTPUT / f'video_{index:02d}'
        before = source.stat()
        source_hash = digest(source)
        print(f'{index}/{len(sources)} {source.name}', flush=True)
        record = {'survey_id': f'video_{index:02d}', 'source': relative(source),
                  'bytes': before.st_size, 'sha256_before': source_hash,
                  'mtime_ns_before': before.st_mtime_ns,
                  'country': None, 'usage_terms_verified': False,
                  'visual_status': 'PENDING_VISUAL_REVIEW'}
        try:
            completed = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                        '--worker', str(source), '--folder', str(folder)],
                                       capture_output=True, text=True, timeout=120)
            record['worker_exit_code'] = completed.returncode
            record['decoder_log'] = completed.stderr[-3000:]
            decoder_path = folder / 'decoder.json'
            if completed.returncode == 0 and decoder_path.is_file():
                record['decoder'] = json.loads(decoder_path.read_text(encoding='utf-8'))
        except subprocess.TimeoutExpired:
            record['decoder_error'] = 'TIMEOUT_120_SECONDS'
        after = source.stat()
        record['sha256_after'] = digest(source)
        record['source_unchanged'] = (source_hash == record['sha256_after']
                                      and before.st_size == after.st_size
                                      and before.st_mtime_ns == after.st_mtime_ns)
        records.append(record)
        write_json(OUTPUT / 'source_manifest.json', records)
    write_json(OUTPUT / 'survey_execution.json', {
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_count': len(records), 'source_files_modified': not all(item['source_unchanged'] for item in records),
        'method': '9 distributed frame seeks and three windows of up to 5 consecutive frames; not full decoding',
        'models_run': False, 'ground_truth_created': False})
    cards = []
    for record in records:
        decoder = record.get('decoder', {})
        sheet = decoder.get('contact_sheet')
        title = html.escape(record['source'])
        images = ''
        if sheet:
            sheet_link = Path(sheet).relative_to(OUTPUT.relative_to(ROOT)).as_posix()
            images += f'<a href="{sheet_link}"><img loading="lazy" src="{sheet_link}"></a>'
            images += '<p>Native sample frames: '
            for sample in decoder['samples']:
                if sample.get('image'):
                    link = Path(sample['image']).relative_to(OUTPUT.relative_to(ROOT)).as_posix()
                    images += f'<a href="{link}">{sample["requested_frame_zero_based"]}</a> '
            images += '</p>'
        cards.append(f'<section id="{record["survey_id"]}"><h2>{record["survey_id"]}: {title}</h2>'
                     f'<p>Opened: {decoder.get("opened", False)}; nominal FPS: {decoder.get("nominal_fps")}; '
                     f'duration estimate: {decoder.get("estimated_duration_s")}; SHA unchanged: {record["source_unchanged"]}</p>{images}</section>')
    document = ('<!doctype html><html lang="vi"><meta charset="utf-8"><title>VLM source survey</title>'
                '<style>body{background:#171717;color:#eee;font:16px sans-serif;margin:24px}'
                'img{width:100%;max-width:1440px}a{color:#7cdcff}section{border-top:1px solid #777;padding:12px 0}</style>'
                '<h1>VLM363 — source survey</h1><p>Sampling only, not complete event labels. '
                'Frame numbers are zero-based. Timestamps are decoder/nominal values, not independently verified PTS.</p>'
                '<p><a href="source_suitability_report.md">Suitability report</a> | '
                '<a href="source_manifest.json">Hash and decoder manifest</a></p>' + ''.join(cards) + '</html>')
    (OUTPUT / 'index.html').write_text(document, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--folder', type=Path)
    parser.add_argument('--details', action='store_true')
    parser.add_argument('--report', action='store_true')
    arguments = parser.parse_args()
    cv2.setNumThreads(1)
    if arguments.report:
        build_report()
    elif arguments.details:
        detailed_review()
    elif arguments.worker:
        sample_video(arguments.worker, arguments.folder)
    else:
        inventory()


def detailed_review():
    records = json.loads((OUTPUT / 'source_manifest.json').read_text(encoding='utf-8'))
    official = json.loads((OUTPUT / 'official_checksums.json').read_text(encoding='utf-8'))
    verifications = []
    for record in records:
        source = ROOT / record['source']
        expected = official['md5'].get(source.name)
        if expected:
            actual = digest(source, 'md5')
            verifications.append({'survey_id': record['survey_id'], 'source': record['source'],
                                  'expected_md5': expected, 'actual_md5': actual,
                                  'matches_published_file': actual == expected})
    write_json(OUTPUT / 'published_file_verification.json', {
        'reference_url': official['reference_url'], 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
        'files': verifications})
    settings = [('video_05', 60, 90), ('video_20', 0, 30), ('video_20', 30, 60),
                ('video_22', 0, None), ('video_25', 0, None)]
    results = []
    for survey_id, start_s, end_s in settings:
        record = next(item for item in records if item['survey_id'] == survey_id)
        source = ROOT / record['source']
        metadata = record['decoder']
        fps = metadata['nominal_fps']
        start = int(start_s * fps)
        end = min(metadata['reported_frame_count'], int(end_s * fps)) if end_s is not None else metadata['reported_frame_count']
        detail_name = 'detail_30_60' if survey_id == 'video_20' and start_s == 30 else 'detail'
        folder = OUTPUT / survey_id / detail_name
        folder.mkdir(parents=True, exist_ok=True)
        capture = cv2.VideoCapture(str(source))
        capture.set(cv2.CAP_PROP_POS_FRAMES, start)
        timestamps = []
        thumbnails = []
        images = []
        preview_path = folder / 'preview.mp4'
        preview_width = min(metadata['width'], 960)
        preview_height = int(metadata['height'] * preview_width / metadata['width']) // 2 * 2
        writer = cv2.VideoWriter(str(preview_path), cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                 (preview_width, preview_height))
        if not writer.isOpened():
            raise RuntimeError('Could not open QA video writer')
        previous = None
        repeated_adjacent = 0
        for frame_id in range(start, end):
            decoded, frame = capture.read()
            if not decoded:
                break
            timestamps.append(capture.get(cv2.CAP_PROP_POS_MSEC))
            small = cv2.resize(frame, (160, 90))
            if previous is not None and (small == previous).all():
                repeated_adjacent += 1
            previous = small
            writer.write(cv2.resize(frame, (preview_width, preview_height)))
            if (frame_id - start) % round(fps) == 0:
                picture = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                image_path = folder / f'frame_{frame_id:08d}.jpg'
                picture.save(image_path, quality=94)
                images.append({'frame_zero_based': frame_id, 'nominal_source_s': frame_id / fps,
                               'image': relative(image_path)})
                picture.thumbnail((480, 270))
                tile = Image.new('RGB', (480, 300), '#151515')
                tile.paste(picture, ((480 - picture.width) // 2, 28))
                ImageDraw.Draw(tile).text((6, 6), f'{survey_id}: frame {frame_id} | {frame_id / fps:.2f}s', fill='white')
                thumbnails.append(tile)
        writer.release()
        capture.release()
        sheet_paths = []
        for page in range((len(thumbnails) + 11) // 12):
            sheet = Image.new('RGB', (1920, 900), '#151515')
            for slot, tile in enumerate(thumbnails[page * 12:(page + 1) * 12]):
                sheet.paste(tile, ((slot % 4) * 480, (slot // 4) * 300))
            sheet_path = folder / f'contact_{page + 1:02d}.jpg'
            sheet.save(sheet_path, quality=92)
            sheet_paths.append(relative(sheet_path))
        verifier = cv2.VideoCapture(str(preview_path))
        preview_count = 0
        while True:
            decoded, frame = verifier.read()
            if not decoded:
                break
            preview_count += 1
        verifier.release()
        results.append({'survey_id': survey_id, 'source': record['source'],
                        'source_sha256': record['sha256_before'], 'start_frame_zero_based': start,
                        'end_frame_exclusive': end, 'expected_frames': end - start,
                        'decoded_frames': len(timestamps), 'all_requested_frames_decoded': len(timestamps) == end - start,
                        'timestamps_nondecreasing': all(later >= earlier for earlier, later in zip(timestamps, timestamps[1:])),
                        'timebase_verified': False, 'repeated_adjacent_downsampled_frames': repeated_adjacent,
                        'images': images, 'contact_sheets': sheet_paths, 'qa_preview': relative(preview_path),
                        'qa_preview_codec': 'mp4v', 'qa_preview_decoded_frames': preview_count,
                        'source_hash_unchanged': digest(source) == record['sha256_before']})
    write_json(OUTPUT / 'detailed_windows.json', results)


def build_report():
    records = json.loads((OUTPUT / 'source_manifest.json').read_text(encoding='utf-8'))
    review = json.loads((OUTPUT / 'visual_review.json').read_text(encoding='utf-8'))
    details = json.loads((OUTPUT / 'detailed_windows.json').read_text(encoding='utf-8'))
    verification = json.loads((OUTPUT / 'published_file_verification.json').read_text(encoding='utf-8'))
    review_by_id = {item['survey_id']: item for item in review['videos']}
    verification_by_id = {item['survey_id']: item for item in verification['files']}
    assert set(review_by_id) == {item['survey_id'] for item in records}
    for record in records:
        record['visual_review'] = review_by_id[record['survey_id']]
        record['published_file_verification'] = verification_by_id.get(record['survey_id'])
        record['visual_status'] = record['visual_review']['status']
    statuses = Counter(item['visual_status'] for item in records)
    summary = {
        'survey_execution_status': 'COMPLETE_BOUNDED_SOURCE_SURVEY',
        'dataset_readiness': 'REVIEW_REQUIRED', 'vietnam_red_light_core_ready': False,
        'source_count': len(records),
        'openable_video_count': sum(item.get('decoder', {}).get('opened', False) for item in records),
        'distributed_sample_frames_decoded': sum(sample['decoded'] for item in records for sample in item.get('decoder', {}).get('samples', [])),
        'short_probe_frames_decoded': sum(window['decoded'] for item in records for window in item.get('decoder', {}).get('decode_windows', [])),
        'detailed_window_count': len(details), 'detailed_frames_decoded': sum(item['decoded_frames'] for item in details),
        'published_checksum_matches': sum(item['matches_published_file'] for item in verification['files']),
        'published_checksum_mismatches': sum(not item['matches_published_file'] for item in verification['files']),
        'visual_status_counts': dict(statuses),
        'source_files_modified': not all(item['source_unchanged'] for item in records),
        'all_detail_sources_unchanged': all(item['source_hash_unchanged'] for item in details),
        'models_run': False, 'event_ground_truth_created': False,
        'human_event_qa_complete': False, 'approved_red_light_positive_count': None,
        'usage_terms_verified': False, 'source_timebase_verified': False,
        'full_collection_decode_verified': False,
    }
    assert not summary['source_files_modified'] and summary['all_detail_sources_unchanged']
    assert all(item['all_requested_frames_decoded'] and item['qa_preview_decoded_frames'] == item['decoded_frames'] for item in details)
    report = {'date_local': review['review_date_local'], 'summary': summary,
              'review_scope': review['method'], 'videos': records, 'detail_windows': details,
              'limitations': review['default_requirements_before_dataset_acceptance'],
              'official_reference': verification['reference_url']}
    write_json(OUTPUT / 'source_suitability_report.json', report)
    lines = [
        '# VLM363 — Báo cáo khảo sát video nguồn',
        '', 'Ngày: 09/10/2026. Khảo sát nguồn: COMPLETE_BOUNDED_SOURCE_SURVEY. Dataset: REVIEW_REQUIRED.',
        '', '## Kết quả chính', '',
        f'- Phạm vi: {summary["source_count"]} file: 24 Traffic Evaluation và 1 VNTraffic gốc; không phải mọi video trong dự án.',
        f'- Mở được {summary["openable_video_count"]}/25 file; xem trực tiếp 23 contact sheet, {summary["distributed_sample_frames_decoded"]} ảnh phân bố theo thời gian.',
        f'- Giải mã {summary["short_probe_frames_decoded"]} frame trong các cửa sổ đầu/giữa/cuối; không chứng minh toàn bộ video liên tục.',
        f'- Khảo sát sâu {len(details)} cửa sổ, giải mã đủ {summary["detailed_frames_decoded"]} frame; video QA cũng giải mã lại đủ số frame.',
        '- Video gốc giữ nguyên: SHA-256 trước/sau và size/mtime không đổi cho cả 25 file. Không chạy YOLO, ByteTrack hay VLM; không sửa GT.',
        '- 21/24 file Traffic Evaluation khớp MD5 công bố; 3 file không khớp. MD5 dùng đối chiếu bản tải, SHA-256 dùng provenance nội bộ.',
        '', '## Kết luận tuyển nguồn', '',
        '**Chưa có nguồn được nghiệm thu làm lõi benchmark vượt đỏ Việt Nam.** Có nguồn quốc tế để thử cấu hình camera, gói bằng chứng và prompt; cần bổ sung/khảo sát nguồn Việt Nam phù hợp.',
        '', 'Ưu tiên tiếp theo: `2024-04-30 15-59-50.mp4` (Jackson Hole) và `video2.mp4` (Canmore), sau xác minh quyền dùng, vạch chính xác, ánh xạ đèn–hướng và timebase. Đây là ứng viên kỹ thuật, chưa phải GT hoặc tập test được duyệt.',
        '', '`video4.mp4` chỉ là đoạn thử thách/bổ sung: đủ 536 frame giải mã nhưng clip ngắn, có quan hệ với compilation, có thể cắt cụt lượt. Xe tiến vào vùng ảnh khi đèn đỏ không tự động là vượt vạch lúc đỏ.',
        '', 'VNTraffic 17 giây: đủ 510 frame giải mã, có xe máy/ô tô và che khuất; đèn xa/khó ánh xạ và vạch chưa cấu hình. Dùng cho hiểu cảnh, định vị đối tượng hoặc thiếu bằng chứng; không gán nhãn đỏ/không đỏ từ GT tracking.',
        '', '## Các vấn đề phải xử lý trước gán nhãn', '',
        '1. `2024-05-07 18-24-33.mp4` không khớp checksum, báo 4000 frame nhưng cửa sổ cuối lỗi. `video3.mp4` và `video6.mp4` không mở được và không khớp checksum. Giữ nguyên file; cần lấy lại bản đúng khi được phép, không nghiệm thu bản hiện tại.',
        '2. `2024-05-08 00-48-11.mp4` và `2024-05-08 16-26-55.mp4` đổi từ Canmore sang Jackson Hole. Phải tìm ranh giới đổi cảnh; không dùng một cấu hình camera cho toàn file.',
        '3. Nhiều file là ghi màn hình YouTube: browser, player controls, pause/resize/fullscreen. Chọn đoạn ổn định, kiểm tra đứng hình/cắt cảnh; không xem đây là luồng camera nguyên bản đã được bảo đảm.',
        '4. Ngày trong tên file có thể khác timestamp hiển thị trong ảnh. Không chia phiên/luật theo tên file. Timestamp OpenCV và FPS 30 chỉ là dữ kiện decoder, chưa kiểm chứng PTS/timebase thực.',
        '5. Phần lớn cảnh là Canmore (Canada) hoặc Jackson Hole (Mỹ), không phải Việt Nam. Quốc gia ghi theo địa danh nhìn thấy; nơi công tác tác giả Zenodo không xác định quốc gia của video.',
        '6. Chưa xác minh giấy phép: trang Zenodo trả về mục License không có tên cụ thể trong nội dung đọc được. Public download/checksum đúng không đồng nghĩa được tái phân phối hoặc huấn luyện.',
        '', '## Chống rò rỉ chia tập', '',
        '- `teste2_6.mp4` và `video1.mp4`: cùng cảnh/timestamp March 20, phần quan sát chồng nhau.',
        '- `video2.mp4` và đoạn Canmore của `2024-05-08 00-48-11.mp4`: cùng May 07 khoảng 17:50.',
        '- `video4.mp4` và `rlr_occasions.mp4`: cùng cảnh/chu kỳ October 15 khoảng 15:31–15:32.',
        '- `video5.mp4` và `2024-11-04 12-00-31.mp4`: cùng cảnh November 03 khoảng 17:35–17:37.',
        '', 'Đây là nhóm trùng/chồng nội dung cần giữ chung trước khi xác định ranh giới chính xác. Chưa tạo split. Các group trong JSON là gợi ý bảo thủ, chưa phải ID phiên đã nghiệm thu; sau tách cảnh phải giải quyết cả quan hệ giữa các group.',
        '', '## Danh mục từng file', '',
        '| ID | File | Giây danh nghĩa | Trạng thái | Bối cảnh |',
        '| --- | --- | ---: | --- | --- |',
    ]
    cards = []
    for record in records:
        visual = record['visual_review']
        decoder = record.get('decoder', {})
        duration = decoder.get('estimated_duration_s')
        duration_text = f'{duration:.2f}' if duration is not None else 'N/A'
        lines.append(f'| {record["survey_id"]} | {Path(record["source"]).name} | {duration_text} | {visual["status"]} | {visual["camera_context"] or "chưa xác minh"} |')
        body = f'<h2>{record["survey_id"]}: {html.escape(Path(record["source"]).name)}</h2>'
        body += f'<p><strong>{visual["status"]}</strong> | {html.escape(visual["camera_context"] or "Not visually available")}</p>'
        body += f'<p>{html.escape(visual["observations"])}</p>'
        if decoder.get('contact_sheet'):
            link = Path(decoder['contact_sheet']).relative_to(OUTPUT.relative_to(ROOT)).as_posix()
            body += f'<a href="{link}"><img loading="lazy" src="{link}"></a><p>Native frames: '
            for sample in decoder['samples']:
                if sample.get('image'):
                    link = Path(sample['image']).relative_to(OUTPUT.relative_to(ROOT)).as_posix()
                    body += f'<a href="{link}">{sample["requested_frame_zero_based"]}</a> '
            body += '</p>'
        for detail in details:
            if detail['survey_id'] == record['survey_id']:
                link = Path(detail['qa_preview']).relative_to(OUTPUT.relative_to(ROOT)).as_posix()
                body += f'<h3>Continuous decode: [{detail["start_frame_zero_based"]}, {detail["end_frame_exclusive"]})</h3>'
                body += f'<video controls preload="none" src="{link}"></video><p><a href="{link}">Open/download QA MP4</a> (mp4v: browser support varies; use local player if needed)</p>'
                for sheet in detail['contact_sheets']:
                    link = Path(sheet).relative_to(OUTPUT.relative_to(ROOT)).as_posix()
                    body += f'<details><summary>Dense 1-second frames</summary><a href="{link}"><img loading="lazy" src="{link}"></a></details>'
                body += '<details><summary>Native dense frames</summary><p>'
                for sample in detail['images']:
                    link = Path(sample['image']).relative_to(OUTPUT.relative_to(ROOT)).as_posix()
                    body += f'<a href="{link}">{sample["frame_zero_based"]}</a> '
                body += '</p></details>'
        cards.append(f'<section id="{record["survey_id"]}">{body}</section>')
    lines.extend(['', '## Các cửa sổ giải mã sâu', '', '| File ID | Frame nguồn [đầu, cuối) | Giải mã | Tình trạng |', '| --- | --- | ---: | --- |'])
    for detail in details:
        lines.append(f'| {detail["survey_id"]} | [{detail["start_frame_zero_based"]}, {detail["end_frame_exclusive"]}) | {detail["decoded_frames"]}/{detail["expected_frames"]} | đủ frame, không sửa nguồn |')
    lines.extend([
        '', 'Giải mã liên tục đạt chỉ chứng minh đọc được frame trong cửa sổ. Xem dày ở 1 frame/giây không thay thế QA mọi frame quanh thời điểm vượt/chuyển đèn. Không có event label hoặc số lượt vi phạm được xác nhận trong khảo sát này.',
        '', '## Đầu ra và bước tiếp theo', '',
        '- `index.html`: ảnh mẫu, ảnh dày, clip QA và liên kết ảnh độ phân giải nguồn.',
        '- `source_manifest.json`: metadata và SHA-256 trước/sau.',
        '- `published_file_verification.json`: MD5 bản tải so với bản phát hành.',
        '- `visual_review.json`: nhận xét từng nguồn, không phải human event GT.',
        '- `detailed_windows.json`: provenance, số frame và kiểm tra clip QA.',
        '- `source_suitability_report.json`: báo cáo tổng hợp máy đọc được.',
        '', 'Bước tiếp theo: người dùng xem hai ứng viên ưu tiên, xác minh quyền dùng và cấu hình xe–vạch–đèn. Gán mọi lượt trong đoạn liên tục đã chọn; bổ sung nguồn Việt Nam nhìn rõ vạch/đèn để đáp ứng mục tiêu chính. Chưa cần vẽ bbox mọi frame hoặc train VLM.',
        '', 'Nguồn công bố/checksum: [Dataset for Traffic Evaluation, Zenodo v2](https://zenodo.org/records/14041920). Trang này mô tả dữ liệu phục vụ phân tích giao thông và phát hiện vượt đỏ; không dùng mô tả đó thay kiểm chứng nhãn từng clip.',
    ])
    (OUTPUT / 'source_suitability_report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    document = ('<!doctype html><html lang="vi"><meta charset="utf-8"><title>VLM363 source suitability</title>'
                '<style>body{background:#171717;color:#eee;font:16px sans-serif;margin:24px}img{width:100%;max-width:1440px}'
                'video{max-width:960px;width:100%}a{color:#7cdcff}section{border-top:1px solid #777;padding:16px 0}details{margin:12px 0}</style>'
                '<h1>VLM363 — Khảo sát nguồn video</h1><p>25 files surveyed; 23 openable; 3 published-checksum mismatches. '
                'Dataset readiness: REVIEW_REQUIRED. No models run, no source changes, no event GT created.</p>'
                '<p>Frame numbers: zero-based. Sampling and decode checks are not complete event QA.</p>'
                '<p><a href="source_suitability_report.md">Báo cáo</a> | <a href="source_suitability_report.json">JSON</a> | '
                '<a href="#video_05">Priority: Jackson Hole</a> | <a href="#video_20">Priority: Canmore</a> | '
                '<a href="#video_25">VNTraffic</a></p>' + ''.join(cards) + '</html>')
    (OUTPUT / 'index.html').write_text(document, encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
