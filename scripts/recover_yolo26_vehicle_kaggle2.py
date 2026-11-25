import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'runs/project363/yolo26_kaggle2_20261027'
PACKAGE = ROOT / 'exports/yolo26_vehicle_local_20261027'
NOTEBOOK = ROOT / 'config/kaggle/yolo26_vehicle_local_20261027'
BACKUP = ROOT / 'weights/yolo26s_light_20261027_recovered'
OLD_SLUG = 'yolo26s-vehicle-light-20261027'
DATA_SLUG = 'yolo363-vehicle-archive-20261027'


def state(status, **details):
    record = {'status': status, 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'old_notebook_cancelled': False, **details}
    temporary = REPORT / 'vehicle_recovery_status.tmp'
    temporary.write_text(json.dumps(record, indent=2))
    temporary.replace(REPORT / 'vehicle_recovery_status.json')
    print(json.dumps(record), flush=True)


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    import winreg
    from kagglesdk import KaggleClient, KaggleEnv
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest, ApiGetDatasetStatusRequest
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiGetKernelRequest, ApiGetKernelSessionStatusRequest,
        ApiListKernelSessionOutputRequest, ApiSaveKernelRequest)
    from kagglesdk.kernels.types.kernels_enums import KernelExecutionType
    from kagglesdk.security.types.oauth_service import IntrospectTokenRequest
    from requests import HTTPError
    import requests

    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / 'vehicle_recovery.lock').open('x') as handle:
        handle.write(str(os.getpid()))
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as key:
        token = winreg.QueryValueEx(key, 'KAGGLE_API_TOKEN_2')[0].strip()
    os.environ.update(KAGGLE_API_TOKEN=token, KAGGLE_API_ENVIRONMENT='PROD',
                      KAGGLE_CONFIG_DIR=str(ROOT / '.tools/kaggle2_profile'))
    for name in ('KAGGLE_USERNAME', 'KAGGLE_KEY'):
        os.environ.pop(name, None)
    with KaggleClient(env=KaggleEnv.PROD, api_token=token, verbose=False) as client:
        request = IntrospectTokenRequest()
        request.token = token
        identity = client.security.oauth_client.introspect_token(request)
        if not identity.active or identity.username != 'v5nhdf':
            raise RuntimeError('Wrong Kaggle2 identity')
        inventory = json.loads((ROOT / 'exports/yolo26_kaggle2_20261027/package_status.json').read_text())
        vehicle = next(row for row in inventory['archives'] if row['kind'] == 'vehicle')
        source = Path(vehicle['archive'])
        state('VERIFY_LOCAL_ARCHIVE')
        if checksum(source) != vehicle['sha256']:
            raise ValueError('Local archive hash mismatch')
        PACKAGE.mkdir(parents=True, exist_ok=True)
        opaque = PACKAGE / 'vehicle_trainval.zip.bin'
        if not opaque.exists():
            os.link(source, opaque)
        if not os.path.samefile(source, opaque):
            raise ValueError('Unexpected existing archive payload')
        metadata = {'id': 'v5nhdf/' + DATA_SLUG, 'title': 'YOLO363 Vehicle Opaque Archive 20261027',
                    'licenses': [{'name': 'other'}],
                    'description': 'Private research archive; same reviewed train/full validation as original. '
                                   'ZIP stored with .bin suffix for sequential local extraction, no test data. '
                                   'Original source licenses remain applicable.'}
        (PACKAGE / 'dataset-metadata.json').write_text(json.dumps(metadata, indent=2))
        if {path.name for path in PACKAGE.iterdir()} != {'vehicle_trainval.zip.bin', 'dataset-metadata.json'}:
            raise ValueError('Unexpected upload payload')
        get_dataset = ApiGetDatasetRequest()
        get_dataset.owner_slug = 'v5nhdf'
        get_dataset.dataset_slug = DATA_SLUG
        try:
            dataset = client.datasets.dataset_api_client.get_dataset(get_dataset)
        except HTTPError as error:
            if error.response.status_code not in (403, 404):
                raise
            state('UPLOADING_PRIVATE_OPAQUE_ARCHIVE', bytes=vehicle['bytes'])
            from kaggle import api
            result = api.dataset_create_new(str(PACKAGE), public=False, quiet=True, dir_mode='skip')
            if result.error:
                raise RuntimeError('Dataset upload rejected')
        else:
            if not dataset.is_private:
                raise RuntimeError('Refusing public dataset')
        query = ApiGetDatasetStatusRequest()
        query.owner_slug = 'v5nhdf'
        query.dataset_slug = DATA_SLUG
        deadline = time.monotonic() + 3 * 3600
        while time.monotonic() < deadline:
            try:
                name = client.datasets.dataset_api_client.get_dataset_status(query).status.name
            except HTTPError as error:
                if error.response.status_code not in (403, 404, 429, 500, 502, 503):
                    raise
                name = 'PENDING_API'
            state('WAITING_ARCHIVE_DATASET', dataset_status=name)
            if name in {'READY', 'COMPLETE'}:
                break
            if 'FAIL' in name or 'ERROR' in name:
                raise RuntimeError('Archive dataset processing failed')
            time.sleep(60)
        else:
            raise TimeoutError('Dataset processing timeout')
        state('ARCHIVE_READY_WAITING_LIGHT_CHECKPOINTS', dataset='v5nhdf/' + DATA_SLUG)
        deadline = time.monotonic() + 12 * 3600
        expected = json.loads((REPORT / 'live_stream_status.json').read_text())['latest_progress']['light']
        hashes = expected['sha256']
        if set(hashes) != {'last.pt', 'best.pt'}:
            raise ValueError('Missing expected light checkpoint hashes')
        BACKUP.mkdir(parents=True, exist_ok=True)
        while time.monotonic() < deadline:
            status_query = ApiGetKernelSessionStatusRequest()
            status_query.user_name = 'v5nhdf'
            status_query.kernel_slug = OLD_SLUG
            old_status = client.kernels.kernels_api_client.get_kernel_session_status(status_query).status.name
            files = {}
            page = ''
            while True:
                query = ApiListKernelSessionOutputRequest()
                query.user_name = 'v5nhdf'
                query.kernel_slug = OLD_SLUG
                query.page_size = 200
                query.page_token = page
                outputs = client.kernels.kernels_api_client.list_kernel_session_output(query)
                files.update({row.file_name: row.url for row in outputs.files})
                page = outputs.next_page_token
                if not page:
                    break
            wanted = {name: 'yolo26s_20261027/light/checkpoints/' + name for name in hashes}
            if not all(path in files for path in wanted.values()):
                state('WAITING_PUBLISHED_LIGHT_CHECKPOINTS', old_status=old_status,
                      output_files=len(files), new_training_started=False)
                time.sleep(60)
                continue
            for name, remote_path in wanted.items():
                destination = BACKUP / name
                if destination.is_file() and checksum(destination) == hashes[name]:
                    continue
                partial = destination.with_suffix('.partial')
                with requests.get(files[remote_path], stream=True, timeout=(30, 180)) as response:
                    response.raise_for_status()
                    with partial.open('wb') as handle:
                        for block in response.iter_content(1024 * 1024):
                            handle.write(block)
                if checksum(partial) != hashes[name]:
                    raise ValueError('Light checkpoint hash mismatch; refusing restart')
                partial.replace(destination)
            (BACKUP / 'backup_verified.json').write_text(json.dumps({
                'notebook': 'v5nhdf/' + OLD_SLUG, 'epoch_completed': expected['epoch_completed'],
                'sha256': hashes, 'verified_at_utc': datetime.now(timezone.utc).isoformat()}, indent=2))
            if old_status not in {'COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED'}:
                state('LIGHT_BACKUP_VERIFIED_WAITING_OLD_SESSION', old_status=old_status)
                time.sleep(60)
                continue
            vehicle_progress = 'yolo26s_20261027/vehicle/epoch_status.json'
            if vehicle_progress in files:
                raise RuntimeError('Original vehicle has checkpoint; inspect and resume instead of starting from scratch')
            break
        else:
            raise TimeoutError('Light backup not safely recoverable within 12 hours; old job left untouched')
        metadata = json.loads((NOTEBOOK / 'kernel-metadata.json').read_text())
        query = ApiGetKernelRequest()
        query.user_name, query.kernel_slug = metadata['id'].split('/')
        try:
            client.kernels.kernels_api_client.get_kernel(query)
        except HTTPError as error:
            if error.response.status_code not in (403, 404):
                raise
        else:
            raise RuntimeError('Replacement notebook exists; refusing duplicate submission')
        request = ApiSaveKernelRequest()
        request.slug = metadata['id']
        request.new_title = metadata['title']
        request.text = (NOTEBOOK / 'train.py').read_text()
        request.language = 'python'
        request.kernel_type = 'script'
        request.is_private = True
        request.enable_gpu = True
        request.enable_tpu = False
        request.enable_internet = True
        request.machine_shape = 'NvidiaTeslaT4'
        request.session_timeout_seconds = 43200
        request.dataset_data_sources = metadata['dataset_sources']
        request.kernel_execution_type = KernelExecutionType.SAVE_AND_RUN_ALL
        state('SUBMITTING_VEHICLE_LOCAL_NOTEBOOK', light_backup_verified=True)
        result = client.kernels.kernels_api_client.save_kernel(request)
        if result.error or result.invalid_dataset_sources:
            raise RuntimeError('Replacement notebook rejected')
        (REPORT / 'vehicle_local_submission.json').write_text(json.dumps({
            'url': result.url, 'version': result.version_number, 'private': True}, indent=2))
        state('VEHICLE_LOCAL_SUBMITTED', notebook_url=result.url, light_backup_verified=True,
              training_worker_start_verified=False)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        state('RECOVERY_STOPPED_SAFELY', error_type=type(error).__name__,
              http_status=getattr(getattr(error, 'response', None), 'status_code', None),
              reason=str(error) if isinstance(error, (RuntimeError, ValueError, TimeoutError)) else None)
        sys.exit(1)
