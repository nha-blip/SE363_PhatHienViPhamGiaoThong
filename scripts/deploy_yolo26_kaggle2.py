import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'exports/yolo26_kaggle2_20261027'
NOTEBOOK = ROOT / 'config/kaggle/yolo26_parallel_20261027'
REPORT = ROOT / 'runs/project363/yolo26_kaggle2_20261027'
DATASET_SLUG = 'yolo363-readable-crops-20261027'
KERNEL_SLUG = 'yolo26s-vehicle-light-20261027'


def write_state(status, **details):
    result = {'status': status, 'updated_at_utc': datetime.now(timezone.utc).isoformat(),
              'account': 'v5nhdf', 'model': 'yolo26s.pt', 'gpu_assignment': {'vehicle': 0, 'light': 1},
              'dataset': 'v5nhdf/' + DATASET_SLUG, 'notebook': 'v5nhdf/' + KERNEL_SLUG, **details}
    temporary = REPORT / 'deployment_status.tmp'
    temporary.write_text(json.dumps(result, indent=2), encoding='utf-8')
    temporary.replace(REPORT / 'deployment_status.json')
    print(json.dumps(result), flush=True)


def check_upload_folder():
    expected = {'dataset-metadata.json', 'vehicle_trainval.zip', 'light_trainval.zip'}
    actual = {path.name for path in (PACKAGE / 'upload').iterdir()}
    if actual != expected:
        raise ValueError('Upload folder contains unexpected files or secrets')
    manifest = json.loads((PACKAGE / 'package_status.json').read_text(encoding='utf-8'))
    for archive in manifest['archives']:
        path = Path(archive['archive'])
        if path.parent != PACKAGE / 'upload' or path.stat().st_size != archive['bytes']:
            raise ValueError('Package archive does not match inventory')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume-ready-dataset', action='store_true')
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    lock = REPORT / 'deployment.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with os.fdopen(descriptor, 'w') as handle:
        handle.write(str(os.getpid()))
    stage = 'WAITING_FOR_PACKAGE'
    write_state(stage, training_started=False)
    deadline = time.monotonic() + 3*3600
    try:
        while True:
            path = PACKAGE / 'package_status.json'
            if path.exists():
                manifest = json.loads(path.read_text(encoding='utf-8'))
                if manifest['status'] == 'COMPLETE_CRC_SHA256_VERIFIED' and (PACKAGE / 'upload/dataset-metadata.json').is_file():
                    break
            if time.monotonic() > deadline:
                raise TimeoutError('Packaging did not complete within 3 hours')
            time.sleep(20)
        manifest = check_upload_folder()
        import winreg
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as key:
            token = winreg.QueryValueEx(key, 'KAGGLE_API_TOKEN_2')[0].strip()
        os.environ['KAGGLE_API_TOKEN'] = token
        os.environ['KAGGLE_CONFIG_DIR'] = str(ROOT / '.tools/kaggle2_profile')
        os.environ['KAGGLE_API_ENVIRONMENT'] = 'PROD'
        for name in ('KAGGLE_USERNAME', 'KAGGLE_KEY'):
            os.environ.pop(name, None)
        from kagglesdk import KaggleClient, KaggleEnv
        from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest, ApiGetDatasetStatusRequest
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiGetKernelSessionStatusRequest, ApiSaveKernelRequest
        from kagglesdk.kernels.types.kernels_enums import KernelExecutionType
        from kagglesdk.security.types.oauth_service import IntrospectTokenRequest
        from requests import HTTPError
        with KaggleClient(env=KaggleEnv.PROD, api_token=token, verbose=False) as client:
            request = IntrospectTokenRequest()
            request.token = token
            identity = client.security.oauth_client.introspect_token(request)
            if not identity.active or identity.username != 'v5nhdf':
                raise ValueError('Wrong Kaggle2 account')
            dataset_request = ApiGetDatasetRequest()
            dataset_request.owner_slug = 'v5nhdf'
            dataset_request.dataset_slug = DATASET_SLUG
            try:
                existing_dataset = client.datasets.dataset_api_client.get_dataset(dataset_request)
            except HTTPError as error:
                if args.resume_ready_dataset:
                    raise
                if error.response.status_code not in (403, 404):
                    raise
            else:
                if not args.resume_ready_dataset:
                    raise ValueError('Dataset already exists; refuse overwrite')
                if not existing_dataset.is_private:
                    raise ValueError('Resume requires private dataset')
            kernel_request = ApiGetKernelRequest()
            kernel_request.user_name = 'v5nhdf'
            kernel_request.kernel_slug = KERNEL_SLUG
            try:
                client.kernels.kernels_api_client.get_kernel(kernel_request)
            except HTTPError as error:
                if error.response.status_code not in (403, 404):
                    raise
            else:
                raise ValueError('Notebook already exists; refuse overwrite')
            if not args.resume_ready_dataset:
                stage = 'UPLOADING_PRIVATE_DATASET'
                write_state(stage, training_started=False, archive_bytes=sum(row['bytes'] for row in manifest['archives']))
                from kaggle import api
                uploaded = api.dataset_create_new(str(PACKAGE / 'upload'), public=False, quiet=False, dir_mode='skip')
                if uploaded.error:
                    raise RuntimeError('Kaggle rejected dataset creation')
            stage = 'WAITING_FOR_DATASET_PROCESSING'
            write_state(stage, training_started=False)
            dataset_status = ApiGetDatasetStatusRequest()
            dataset_status.owner_slug = 'v5nhdf'
            dataset_status.dataset_slug = DATASET_SLUG
            deadline = time.monotonic() + 3*3600
            while time.monotonic() < deadline:
                try:
                    result = client.datasets.dataset_api_client.get_dataset_status(dataset_status)
                except HTTPError as error:
                    if error.response.status_code not in (403, 404, 429, 500, 502, 503):
                        raise
                    write_state(stage, training_started=False, transient_http_status=error.response.status_code)
                    time.sleep(30)
                    continue
                name = result.status.name
                write_state(stage, training_started=False, dataset_processing_status=name)
                if name in {'READY', 'COMPLETE'}:
                    break
                if 'FAIL' in name or 'ERROR' in name:
                    raise RuntimeError('Dataset processing failed: ' + name)
                time.sleep(30)
            else:
                raise TimeoutError('Dataset processing exceeded 3 hours')
            dataset = client.datasets.dataset_api_client.get_dataset(dataset_request)
            if not dataset.is_private:
                raise ValueError('Refuse training from accidentally public dataset')
            metadata = json.loads((NOTEBOOK / 'kernel-metadata.json').read_text())
            if metadata['id'] != 'v5nhdf/' + KERNEL_SLUG or not metadata['is_private']:
                raise ValueError('Notebook metadata mismatch')
            request = ApiSaveKernelRequest()
            request.slug = metadata['id']
            request.new_title = metadata['title']
            request.text = (NOTEBOOK / 'train.py').read_text(encoding='utf-8')
            request.language = 'python'
            request.kernel_type = 'script'
            request.is_private = True
            request.enable_gpu = True
            request.enable_internet = True
            request.enable_tpu = False
            request.machine_shape = 'NvidiaTeslaT4'
            request.session_timeout_seconds = 43200
            request.dataset_data_sources = metadata['dataset_sources']
            request.kernel_execution_type = KernelExecutionType.SAVE_AND_RUN_ALL
            stage = 'SUBMITTING_GPU_NOTEBOOK'
            write_state(stage, training_started=False)
            saved = client.kernels.kernels_api_client.save_kernel(request)
            if saved.error or saved.invalid_dataset_sources:
                (REPORT / 'submission_rejection.json').write_text(json.dumps({'error': saved.error,
                         'invalid_dataset_sources': saved.invalid_dataset_sources}, indent=2), encoding='utf-8')
                raise RuntimeError('Notebook submission rejected')
            (REPORT / 'submission.json').write_text(json.dumps({'url': saved.url, 'version': saved.version_number,
                         'notebook': metadata['id'], 'private': True, 'accelerator': 'NvidiaTeslaT4'}, indent=2), encoding='utf-8')
            status_request = ApiGetKernelSessionStatusRequest()
            status_request.user_name = 'v5nhdf'
            status_request.kernel_slug = KERNEL_SLUG
            stage = 'NOTEBOOK_SUBMITTED'
            for attempt in range(120):
                status = client.kernels.kernels_api_client.get_kernel_session_status(status_request)
                name = status.status.name
                write_state('KAGGLE_' + name, notebook_url=saved.url, version=saved.version_number,
                            training_started=False, training_worker_start_verified=False,
                            note='RUNNING means notebook session, not proof both training workers have started')
                if name in {'RUNNING', 'COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED'}:
                    break
                time.sleep(20)
    except Exception as error:
        response = getattr(error, 'response', None)
        write_state('FAILED', failed_stage=stage, error_type=type(error).__name__,
                    http_status=getattr(response, 'status_code', None), training_started=False,
                    note='Do not blindly retry submission; inspect submission.json and Kaggle notebook first')
        raise SystemExit(1)


if __name__ == '__main__':
    main()
