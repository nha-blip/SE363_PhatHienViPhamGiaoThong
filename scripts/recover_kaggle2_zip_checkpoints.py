import binascii
import hashlib
import io
import json
import re
import shutil
import time
import zipfile
from pathlib import Path

import requests


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'weights/yolo26s_fresh_20261027'
REPORT = ROOT / 'runs/project363/yolo26_kaggle2_20261027'


class RangeReader(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        self.position = 0
        self.offset = 0
        self.buffer = b''
        self.downloaded = 0
        with requests.get(url, headers={'Range': 'bytes=0-0'}, timeout=(30, 120)) as response:
            if response.status_code != 206:
                raise RuntimeError('Server does not support safe range reads')
            self.length = int(response.headers['Content-Range'].split('/')[-1])

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        position = offset if whence == 0 else self.position + offset if whence == 1 else self.length + offset
        if position < 0:
            raise ValueError('Negative seek')
        self.position = position
        return position

    def read(self, size=-1):
        size = self.length - self.position if size < 0 else min(size, self.length - self.position)
        if size <= 0:
            return b''
        if not (self.offset <= self.position and self.position + size <= self.offset + len(self.buffer)):
            end = min(self.length - 1, self.position + max(size, 4 * 1024**2) - 1)
            for attempt in range(5):
                try:
                    with requests.get(self.url, headers={'Range': f'bytes={self.position}-{end}'},
                                      timeout=(30, 180)) as response:
                        if response.status_code != 206:
                            raise RuntimeError('Server ignored requested range')
                        expected = f'bytes {self.position}-{end}/{self.length}'
                        if response.headers.get('Content-Range') != expected:
                            raise RuntimeError('Wrong byte range')
                        self.buffer = response.content
                        if len(self.buffer) != end - self.position + 1:
                            raise RuntimeError('Incomplete range read')
                    break
                except requests.RequestException:
                    if attempt == 4:
                        raise RuntimeError('Range download failed after five attempts') from None
                    time.sleep(2**attempt)
            self.offset = self.position
            self.downloaded += len(self.buffer)
        start = self.position - self.offset
        self.position += size
        return self.buffer[start:start + size]


def main():
    import winreg
    from kagglesdk import KaggleClient, KaggleEnv
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest
    from kagglesdk.security.types.oauth_service import IntrospectTokenRequest

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as key:
        token = winreg.QueryValueEx(key, 'KAGGLE_API_TOKEN_2')[0].strip()
    with KaggleClient(env=KaggleEnv.PROD, api_token=token, verbose=False) as client:
        query = IntrospectTokenRequest()
        query.token = token
        identity = client.security.oauth_client.introspect_token(query)
        if not identity.active or identity.username != 'v5nhdf':
            raise RuntimeError('Wrong Kaggle2 identity')
        query = ApiListKernelSessionOutputRequest()
        query.user_name = 'v5nhdf'
        query.kernel_slug = 'yolo26s-fresh-local-20261027'
        files = client.kernels.kernels_api_client.list_kernel_session_output(query).files
        bundle = next(row for row in files if row.file_name == '_output_.zip')
        reader = RangeReader(bundle.url)
        records = []
        OUTPUT.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(reader) as archive:
            selected = [name for name in archive.namelist() if re.fullmatch(
                r'yolo26s_fresh_local_20261027/(?:completion\.json|progress\.json|'
                r'(?:vehicle|light)/(?:epoch_status\.json|train\.log|checkpoints/(?:last|best)\.pt|'
                r'train/(?:results\.csv|args\.yaml)))', name)]
            if len([name for name in selected if name.endswith('.pt')]) != 4:
                raise RuntimeError('Expected four rolling checkpoints in output archive')
            for name in selected:
                target = OUTPUT / Path(name).relative_to('yolo26s_fresh_local_20261027')
                target.parent.mkdir(parents=True, exist_ok=True)
                metadata = archive.getinfo(name)
                valid = (target.is_file() and target.stat().st_size == metadata.file_size
                         and binascii.crc32(target.read_bytes()) & 0xffffffff == metadata.CRC)
                if not valid:
                    partial = target.with_suffix(target.suffix + '.partial')
                    with archive.open(name) as source, partial.open('wb') as destination:
                        shutil.copyfileobj(source, destination, length=1024**2)
                    partial.replace(target)
                records.append({'file': str(target.relative_to(OUTPUT)), 'bytes': target.stat().st_size,
                                'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
                print(json.dumps(records[-1]), flush=True)
        for kind in ('vehicle', 'light'):
            status = json.loads((OUTPUT / kind / 'epoch_status.json').read_text())
            for name, expected in status['sha256'].items():
                record = next(row for row in records if row['file'].replace('\\', '/') == f'{kind}/checkpoints/{name}')
                if record['sha256'] != expected:
                    raise RuntimeError('Checkpoint SHA256 differs from worker epoch status')
        result = {'status': 'FOUR_CHECKPOINTS_CRC_SHA256_VERIFIED', 'archive_bytes': reader.length,
                  'range_downloaded_bytes': reader.downloaded, 'files': records}
        (OUTPUT / 'recovery_verified.json').write_text(json.dumps(result, indent=2))
        (REPORT / 'checkpoint_recovery.json').write_text(json.dumps(result, indent=2))
        print(json.dumps({'status': result['status'], 'downloaded_bytes': reader.downloaded}), flush=True)


if __name__ == '__main__':
    main()
