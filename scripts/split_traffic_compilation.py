import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import cv2
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'data/AQOcupdZzFNYCBBIWIfFt8OJZqHhMHevwUC2lqdB4Z04cyur5Ocf7ep3aRW3dl4UGFz7sU0Mp34IYwZ8E8PEU0aMh8EWdH7Ct_yi9gEPGw.mp4'
OUTPUT = ROOT / 'data/VLM363_working/external_compilation_15'
BOUNDARY_GUESSES = [137,168,198,228,268,302,333,363,393,423,489,523,543,568,593,618]
CUT_FRAMES = [3485,4189,4979,5682,6635,7574,8251,9058,9863,10618,12231,13012,13619,14180,14759,15480]
DESCRIPTIONS = [
    'Xe trắng, cảnh ban đêm, overlay 29C 665.XX',
    'Xe máy, overlay 29L5 263.XX',
    'Xe máy áo xanh, overlay 29N1 004.XX',
    'SUV bạc, overlay 30A 117.XX',
    'Ô tô đỏ, overlay 26A 206.XX',
    'Xe máy, overlay 26AA 145.XX',
    'Xe máy trong vùng bóng râm, overlay 26B1 895.XX',
    'Xe tải/van nhỏ, overlay 15C 071.XX',
    'Ô tô, cảnh ban đêm, overlay 15K 522.XX',
    'Đoạn nhiều xe máy: overlay 27AA 104.XX, 27N1 035.XX, 17B1 386.XX',
    'Ô tô tối màu, overlay 89A 020.XX',
    'Xe máy, overlay 25AA 050.XX',
    'Xe máy, overlay 98AL 011.XX',
    'Xe máy, overlay 27B1 143.XX',
    'Xe máy, overlay 27B1 164.XX',
]


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            checksum.update(chunk)
    return checksum.hexdigest()


def survey():
    folder = OUTPUT / 'scene_survey'
    folder.mkdir(parents=True,exist_ok=False)
    capture = cv2.VideoCapture(str(SOURCE))
    fps, count = capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    records, sheets = [], []
    try:
        for frame_id in range(0,count,round(fps*5)):
            capture.set(cv2.CAP_PROP_POS_FRAMES,frame_id)
            success, frame = capture.read()
            if not success:
                raise RuntimeError(f'Cannot read frame {frame_id}')
            ordinal = len(records)
            if ordinal % 30 == 0:
                sheets.append(Image.new('RGB',(1600,1220),'black'))
            local = ordinal % 30
            left, top = (local%5)*320,(local//5)*203
            photo = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((320,180))
            sheets[-1].paste(photo,(left,top+23))
            ImageDraw.Draw(sheets[-1]).text((left+4,top+5),f'{frame_id/fps:.0f}s / frame {frame_id}',fill='white')
            records.append({'frame_id':frame_id,'seconds_nominal':frame_id/fps})
    finally:
        capture.release()
    for ordinal,sheet in enumerate(sheets,1):
        sheet.save(folder / f'overview_{ordinal:02d}.jpg',quality=95)
    (folder / 'metadata.json').write_text(json.dumps({'source':SOURCE.relative_to(ROOT).as_posix(),'source_sha256':digest(SOURCE),'fps':fps,'frame_count':count,'samples':records},indent=2)+'\n',encoding='utf-8')
    print(folder)


def boundaries():
    capture = cv2.VideoCapture(str(SOURCE))
    fps = capture.get(cv2.CAP_PROP_FPS)
    results = []
    try:
        for guess in BOUNDARY_GUESSES:
            start, end = round((guess-4)*fps), round((guess+4)*fps)
            capture.set(cv2.CAP_PROP_POS_FRAMES,start)
            previous, scores = None, []
            for frame_id in range(start,end):
                success, frame = capture.read()
                if not success:
                    raise RuntimeError('Boundary frame missing')
                small = cv2.resize(frame[:290,:],(128,58))
                if previous is not None:
                    scores.append((float(cv2.absdiff(previous,small).mean()),frame_id))
                previous = small
            strongest = sorted(scores,reverse=True)[:5]
            print(guess, [(round(score,2),frame_id,round(frame_id/fps,2)) for score,frame_id in strongest])
            sheet = Image.new('RGB',(1280,812),'black')
            for ordinal,second in enumerate(range(guess-3,guess+5)):
                capture.set(cv2.CAP_PROP_POS_FRAMES,round(second*fps))
                success, frame = capture.read()
                if not success:
                    raise RuntimeError('Boundary thumbnail missing')
                left, top = (ordinal%2)*640,(ordinal//2)*203
                photo = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((640,180))
                sheet.paste(photo,(left,top+23))
                ImageDraw.Draw(sheet).text((left+5,top+4),f'{second}s',fill='white')
            sheet.save(OUTPUT / 'scene_survey' / f'boundary_{guess:03d}.jpg',quality=95)
            results.append({'guess':guess,'strongest_changes':strongest})
    finally:
        capture.release()
    (OUTPUT / 'scene_survey/boundary_scores.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')


def split(ffmpeg):
    survey_metadata = json.loads((OUTPUT / 'scene_survey/metadata.json').read_text(encoding='utf-8'))
    source_hash = digest(SOURCE)
    if source_hash != survey_metadata['source_sha256']:
        raise ValueError('Source changed since survey')
    fps = survey_metadata['fps']
    if len(CUT_FRAMES) != 16 or len(DESCRIPTIONS) != 15 or any(start >= end for start,end in zip(CUT_FRAMES,CUT_FRAMES[1:])):
        raise ValueError('Invalid fifteen-part cut plan')
    clips = OUTPUT / 'clips'
    clips.mkdir(exist_ok=False)
    records = []
    overview = Image.new('RGB',(960,1070),'black')
    for ordinal,(start,end,description) in enumerate(zip(CUT_FRAMES,CUT_FRAMES[1:],DESCRIPTIONS),1):
        clip_path = clips / f'case_{ordinal:02d}.mp4'
        command = [str(ffmpeg),'-hide_banner','-loglevel','error','-nostdin','-n',
                   '-ss',f'{start/fps:.6f}','-i',str(SOURCE),'-t',f'{(end-start)/fps:.6f}',
                   '-map','0:v:0','-map','0:a:0','-c:v','libx264','-preset','fast','-crf','18',
                   '-threads','2','-c:a','aac','-b:a','96k','-movflags','+faststart',str(clip_path)]
        subprocess.run(command,check=True,capture_output=True,timeout=300)
        capture = cv2.VideoCapture(str(clip_path))
        count = 0
        selected_images = {}
        try:
            if abs(capture.get(cv2.CAP_PROP_FPS)-fps) > 0.01:
                raise RuntimeError('Output FPS mismatch')
            for frame_id in range(end-start):
                success, frame = capture.read()
                if not success:
                    raise RuntimeError(f'Clip {ordinal}: missing frame {frame_id}')
                if frame.shape[:2] != (360,640):
                    raise RuntimeError('Output resolution mismatch')
                count += 1
                if frame_id in {0,(end-start)//2,end-start-1}:
                    selected_images[frame_id] = frame
            if capture.read()[0]:
                raise RuntimeError('Extra output frame')
        finally:
            capture.release()
        stream_check = subprocess.run([str(ffmpeg),'-hide_banner','-i',str(clip_path)],capture_output=True,text=True)
        if 'Audio:' not in stream_check.stderr:
            raise RuntimeError('Audio stream missing')
        frames_sheet = Image.new('RGB',(960,203),'black')
        for slot,(frame_id,frame) in enumerate(sorted(selected_images.items())):
            photo = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((320,180))
            frames_sheet.paste(photo,(slot*320,23))
            ImageDraw.Draw(frames_sheet).text((slot*320+4,5),f'case {ordinal:02d} | source {(start+frame_id)/fps:.2f}s',fill='white')
        frames_sheet.save(clips / f'case_{ordinal:02d}_preview.jpg',quality=95)
        center = selected_images[(end-start)//2]
        overview.paste(Image.fromarray(cv2.cvtColor(center,cv2.COLOR_BGR2RGB)).resize((320,180)),(((ordinal-1)%3)*320,((ordinal-1)//3)*214+30))
        ImageDraw.Draw(overview).text((((ordinal-1)%3)*320+5,((ordinal-1)//3)*214+8),f'case_{ordinal:02d}: {start/fps:.2f}-{end/fps:.2f}s',fill='white')
        records.append({'case_id':f'case_{ordinal:02d}','clip':clip_path.relative_to(ROOT).as_posix(),
                        'description':description,'source_start_frame_zero_based':start,
                        'source_end_frame_exclusive':end,'source_start_seconds':start/fps,
                        'source_end_seconds':end/fps,'decoded_frames':count,'nominal_fps':fps,
                        'clip_sha256':digest(clip_path),'audio_present':True,'video_codec':'h264','audio_codec':'aac',
                        'assessment':None,'review_status':'PENDING_EVENT_QA','is_ground_truth':False})
        print(f'Created case_{ordinal:02d}: {start/fps:.2f}–{end/fps:.2f}s',flush=True)
    if digest(SOURCE) != source_hash:
        raise RuntimeError('Raw source changed during split')
    overview.save(OUTPUT / 'overview_15_cases.jpg',quality=95)
    manifest = {'source':SOURCE.relative_to(ROOT).as_posix(),'source_sha256':source_hash,'raw_source_modified':False,
                'cut_method':'manual visual editorial sections with hard cuts and crossfade boundaries',
                'case_count':15,'source_total_frames':survey_metadata['frame_count'],
                'omitted_source_ranges_frames':[[0,CUT_FRAMES[0]],[CUT_FRAMES[-1],survey_metadata['frame_count']]],
                'audio_retained':True,'reencoded':True,'ffmpeg_sha256':digest(ffmpeg),
                'no_new_web_pages':True,'cases':records,
                'note':'15 editorial sections, not necessarily 15 unique vehicles. case_10 contains multiple highlighted vehicles. Broadcast zoom/replay/overlays are preserved, not independent raw evidence.'}
    (OUTPUT / 'split_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines = ['# 15 clip tình huống — QA local','',
             'Mở clips/case_01.mp4 đến case_15.mp4. Có âm thanh, 640×360, 25 FPS; đã kiểm tra số frame từng clip.',
             'Giữ nguyên file gốc. Bỏ phần mở đầu trước 139,40s và phần kết sau 619,20s. Không chia đều thời lượng.',
             'Đây là 15 phần biên tập; case_10 có nhiều xe được đánh dấu. Không mặc định mỗi clip là một xe hoặc là GT vi phạm.',
             'Video có người dẫn, chữ/biển số che, zoom, replay và chuyển cảnh; chưa dùng làm input đánh giá VLM sạch.',
             'Xem overview_15_cases.jpg hoặc clips/case_xx_preview.jpg để chọn clip. Nhãn assessment vẫn null, cần QA.',
             '', '| Clip | Khoảng giây nguồn | Mô tả |','|---|---|---|']
    lines.extend(f"| {row['case_id']} | {row['source_start_seconds']:.2f}–{row['source_end_seconds']:.2f} | {row['description']} |" for row in records)
    (OUTPUT / 'README_QA.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command',choices=['survey','boundaries','split'])
    parser.add_argument('--ffmpeg',type=Path)
    arguments = parser.parse_args()
    if arguments.command == 'survey':
        survey()
    elif arguments.command == 'boundaries':
        boundaries()
    else:
        if not arguments.ffmpeg or not arguments.ffmpeg.is_file():
            parser.error('--ffmpeg must point to an existing executable')
        split(arguments.ffmpeg)
