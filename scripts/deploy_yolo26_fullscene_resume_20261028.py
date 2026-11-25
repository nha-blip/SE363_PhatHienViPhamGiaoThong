import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[1]
EXPORT = ROOT / 'exports/yolo26_fullscene_resume_20261028'
UPLOAD = EXPORT / 'upload'
SOURCE = ROOT / 'config/kaggle/yolo26_fullscene_resume_20261028'
BASE = ROOT / 'config/kaggle/yolo26_vehicle_local_20261027/train.py'
RECOVERED = ROOT / 'weights/yolo26s_fresh_20261027'
REPORT = ROOT / 'runs/project363/yolo26_fullscene_resume_20261028'
DATA_SLUG = 'yolo363-fullscene-resume-20261028'


def canonical_notebook(url):
    parsed = urlsplit(url)
    if parsed.scheme != 'https' or parsed.hostname != 'www.kaggle.com':
        raise ValueError('Unexpected notebook URL')
    reference = parsed.path.removeprefix('/code/').rstrip('/')
    if not re.fullmatch(r'v5nhdf/[a-z0-9-]+', reference):
        raise ValueError('Unexpected notebook owner or slug')
    return reference


def read_call(function, query):
    from requests import ConnectionError, HTTPError, Timeout

    for attempt in range(3):
        try:
            return function(query)
        except HTTPError as error:
            if error.response.status_code not in (429, 500, 502, 503, 504) or attempt == 2:
                raise
        except (ConnectionError, Timeout):
            if attempt == 2:
                raise
        time.sleep(3 * (attempt + 1))


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024**2), b''):
            result.update(block)
    return result.hexdigest()


def state(status, **details):
    result = {'status': status, 'updated_at_utc': datetime.now(timezone.utc).isoformat(),
              'account': 'v5nhdf', 'gpu_assignment': {'vehicle_resume': 0, 'light_fullscene': 1}, **details}
    REPORT.mkdir(parents=True, exist_ok=True)
    temporary = REPORT / 'status.tmp'
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(REPORT / 'status.json')
    print(json.dumps(result), flush=True)


def main():
    import winreg
    from kagglesdk import KaggleClient, KaggleEnv
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest, ApiGetDatasetStatusRequest
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiGetKernelRequest, ApiGetKernelSessionStatusRequest, ApiSaveKernelRequest,
        ApiGetAcceleratorQuotaStatisticsRequest)
    from kagglesdk.kernels.types.kernels_enums import KernelExecutionType
    from kagglesdk.security.types.oauth_service import IntrospectTokenRequest
    from requests import HTTPError

    REPORT.mkdir(parents=True, exist_ok=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--retry-after-safe-stop', action='store_true')
    args = parser.parse_args()
    if args.retry_after_safe_stop:
        prior = json.loads((REPORT / 'status.json').read_text())
        if prior['status'] != 'STOPPED_SAFELY' or (REPORT / 'submission.json').exists():
            raise RuntimeError('Retry requires safe stop without an existing notebook submission')
    lock = 'deployment_retry_20261028.lock' if args.retry_after_safe_stop else 'deployment.lock'
    with (REPORT / lock).open('x') as handle:
        handle.write(str(os.getpid()))
    state('WAITING_LOCAL_BUILD_AND_VERIFIED_CHECKPOINTS')
    deadline = time.monotonic()+3*3600
    while not (EXPORT / 'light_package.json').is_file() or not (RECOVERED / 'recovery_verified.json').is_file():
        if time.monotonic() > deadline:
            raise TimeoutError('Local preparation incomplete; no GPU submitted')
        time.sleep(20)
    light = json.loads((EXPORT / 'light_package.json').read_text())
    if not light['crc_verified'] or light['validation_images'] != 1898:
        raise ValueError('Unverified light package')
    if digest(UPLOAD / light['file']) != light['sha256']:
        raise ValueError('Light archive SHA256 mismatch')
    resume_status = json.loads((RECOVERED / 'vehicle/epoch_status.json').read_text())
    if resume_status['epoch_completed'] != 11:
        raise ValueError('Unexpected vehicle resume epoch')
    files = {'vehicle_last.pt': RECOVERED / 'vehicle/checkpoints/last.pt',
             'vehicle_best.pt': RECOVERED / 'vehicle/checkpoints/best.pt',
             'vehicle_results.csv': RECOVERED / 'vehicle/train/results.csv'}
    hashes = {}
    for name, path in files.items():
        hashes[name] = digest(path)
        if name.endswith('.pt') and hashes[name] != resume_status['sha256'][name.replace('vehicle_', '')]:
            raise ValueError('Vehicle checkpoint SHA256 mismatch')
        target = UPLOAD / name
        if not target.exists():
            os.link(path, target)
        if not os.path.samefile(path, target):
            raise ValueError('Unexpected existing upload file')
    package = {'light': light, 'vehicle_resume': {'epoch_completed': 11, 'sha256': hashes,
               'strategy': 'Optimizer-bearing last.pt resume; original vehicle data retained'}}
    (UPLOAD / 'training_package.json').write_text(json.dumps(package, indent=2))
    (UPLOAD / 'dataset-metadata.json').write_text(json.dumps({
        'id': 'v5nhdf/'+DATA_SLUG, 'title': 'YOLO363 Fullscene Light Vehicle Resume 20261028',
        'licenses': [{'name': 'other'}], 'description': 'Private research full-scene light training, '
        'unchanged validation, no test images. Own vehicle last/best checkpoints for optimizer resume. '
        'Original source licenses remain applicable.'}, indent=2))
    if {path.name for path in UPLOAD.iterdir()} != {
            'light_fullscene.zip.bin', 'vehicle_last.pt', 'vehicle_best.pt', 'vehicle_results.csv',
            'training_package.json', 'dataset-metadata.json'}:
        raise ValueError('Unexpected private upload payload')
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as key:
        token = winreg.QueryValueEx(key, 'KAGGLE_API_TOKEN_2')[0].strip()
    os.environ.update(KAGGLE_API_TOKEN=token, KAGGLE_API_ENVIRONMENT='PROD',
                      KAGGLE_CONFIG_DIR=str(ROOT / '.tools/kaggle2_profile'))
    for name in ('KAGGLE_USERNAME', 'KAGGLE_KEY'):
        os.environ.pop(name, None)
    with KaggleClient(env=KaggleEnv.PROD, api_token=token, verbose=False) as client:
        query = IntrospectTokenRequest()
        query.token = token
        identity = read_call(client.security.oauth_client.introspect_token, query)
        if not identity.active or identity.username != 'v5nhdf':
            raise RuntimeError('Wrong Kaggle2 identity')
        metadata = json.loads((SOURCE / 'kernel-metadata.json').read_text())
        if not metadata['is_private'] or metadata['id'] != 'v5nhdf/yolo26-fullscene-light-vehicle-resume-20261028':
            raise RuntimeError('Unexpected notebook metadata')
        query = ApiGetKernelRequest()
        query.user_name, query.kernel_slug = metadata['id'].split('/')
        try:
            read_call(client.kernels.kernels_api_client.get_kernel, query)
        except HTTPError as error:
            if error.response.status_code not in (403, 404):
                raise
        else:
            raise RuntimeError('Notebook exists; refusing duplicate submission')
        query = ApiGetDatasetRequest()
        query.owner_slug, query.dataset_slug = 'v5nhdf', DATA_SLUG
        try:
            dataset = read_call(client.datasets.dataset_api_client.get_dataset, query)
        except HTTPError as error:
            if error.response.status_code not in (403, 404):
                raise
            state('UPLOADING_PRIVATE_FULLSCENE_AND_CHECKPOINTS',
                  upload_bytes=sum(path.stat().st_size for path in UPLOAD.iterdir()))
            from kaggle import api
            result = api.dataset_create_new(str(UPLOAD), public=False, quiet=True, dir_mode='skip')
            if result.error:
                raise RuntimeError('Private dataset upload rejected')
        else:
            if not dataset.is_private:
                raise RuntimeError('Refusing public dataset')
        query = ApiGetDatasetStatusRequest()
        query.owner_slug, query.dataset_slug = 'v5nhdf', DATA_SLUG
        deadline = time.monotonic()+3*3600
        while time.monotonic() < deadline:
            try:
                name = read_call(client.datasets.dataset_api_client.get_dataset_status, query).status.name
            except HTTPError as error:
                if error.response.status_code not in (403, 404, 429, 500, 502, 503):
                    raise
                name = 'PENDING_API'
            state('WAITING_PRIVATE_DATASET_READY', dataset_status=name)
            if name in {'READY', 'COMPLETE'}:
                break
            if 'ERROR' in name or 'FAIL' in name:
                raise RuntimeError('Dataset processing failed')
            time.sleep(30)
        else:
            raise TimeoutError('Dataset processing timeout')
        quota = read_call(client.kernels.kernels_api_client.get_accelerator_quota_statistics,
                          ApiGetAcceleratorQuotaStatisticsRequest())
        if quota.gpu_quota is None or quota.gpu_quota.time_reserved.total_seconds() != 0:
            raise RuntimeError('Kaggle2 GPU reserved by another job; refuse interference')
        base_source = BASE.read_text()
        prefix = ('import sys, types\n_runtime_base = types.ModuleType("runtime_base")\n'
                  '_runtime_base.__file__ = __file__\nsys.modules["runtime_base"] = _runtime_base\n'
                  f'exec(compile({base_source!r}, "runtime_base.py", "exec"), _runtime_base.__dict__)\n')
        code = prefix + (SOURCE / 'train.py').read_text()
        compile(code, 'submitted_train.py', 'exec')
        (REPORT / 'submitted_train.py').write_text(code)
        request = ApiSaveKernelRequest()
        request.slug, request.new_title = metadata['id'], metadata['title']
        request.text, request.language, request.kernel_type = code, 'python', 'script'
        request.is_private = True
        request.enable_gpu, request.enable_tpu, request.enable_internet = True, False, True
        request.machine_shape, request.session_timeout_seconds = 'NvidiaTeslaT4', 43200
        request.dataset_data_sources = metadata['dataset_sources']
        request.kernel_execution_type = KernelExecutionType.SAVE_AND_RUN_ALL
        state('SUBMITTING_VEHICLE_RESUME_LIGHT_FULLSCENE')
        saved = client.kernels.kernels_api_client.save_kernel(request)
        if saved.error or saved.invalid_dataset_sources:
            raise RuntimeError('Notebook rejected')
        actual_reference = canonical_notebook(saved.url)
        (REPORT / 'submission.json').write_text(json.dumps({'url': saved.url,
               'version': saved.version_number, 'notebook': actual_reference, 'private': True}, indent=2))
        query = ApiGetKernelSessionStatusRequest()
        query.user_name, query.kernel_slug = actual_reference.split('/')
        for attempt in range(60):
            status = read_call(client.kernels.kernels_api_client.get_kernel_session_status, query).status.name
            state('KAGGLE_'+status, notebook_url=saved.url, training_worker_start_verified=False)
            if status in {'RUNNING', 'COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED'}:
                return
            time.sleep(20)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        state('STOPPED_SAFELY', error_type=type(error).__name__,
              reason=str(error) if isinstance(error, (ValueError, RuntimeError, TimeoutError)) else None,
              http_status=getattr(getattr(error, 'response', None), 'status_code', None))
        sys.exit(1)
