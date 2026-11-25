import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'runs/project363/yolo26_kaggle2_20261027'
SOURCE = ROOT / 'config/kaggle/yolo26_vehicle_local_20261027'


def state(status, **details):
    data = {'status': status, 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
            'old_checkpoint_wait_skipped_by_user': True, **details}
    temporary = REPORT / 'fresh_restart_status.tmp'
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(REPORT / 'fresh_restart_status.json')
    print(json.dumps(data), flush=True)


def main():
    import os
    import winreg
    from kagglesdk import KaggleClient, KaggleEnv
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiGetKernelRequest, ApiGetKernelSessionStatusRequest, ApiSaveKernelRequest,
        ApiGetAcceleratorQuotaStatisticsRequest)
    from kagglesdk.kernels.types.kernels_enums import KernelExecutionType
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetStatusRequest
    from kagglesdk.security.types.oauth_service import IntrospectTokenRequest
    from requests import HTTPError

    parser = argparse.ArgumentParser()
    parser.add_argument('--user-stopped-old-session', action='store_true')
    args = parser.parse_args()
    lock_name = 'fresh_restart_after_user_stop.lock' if args.user_stopped_old_session else 'fresh_restart.lock'
    with (REPORT / lock_name).open('x') as handle:
        handle.write(str(os.getpid()))
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as key:
        token = winreg.QueryValueEx(key, 'KAGGLE_API_TOKEN_2')[0].strip()
    with KaggleClient(env=KaggleEnv.PROD, api_token=token, verbose=False) as client:
        query = IntrospectTokenRequest()
        query.token = token
        identity = client.security.oauth_client.introspect_token(query)
        if not identity.active or identity.username != 'v5nhdf':
            raise RuntimeError('Wrong Kaggle2 account')
        metadata = json.loads((SOURCE / 'kernel-metadata.json').read_text())
        if not metadata['is_private'] or metadata['id'] != 'v5nhdf/yolo26s-fresh-local-20261027':
            raise RuntimeError('Unexpected notebook metadata')
        query = ApiGetKernelRequest()
        query.user_name, query.kernel_slug = metadata['id'].split('/')
        try:
            client.kernels.kernels_api_client.get_kernel(query)
        except HTTPError as error:
            if error.response.status_code not in (403, 404):
                raise
        else:
            raise RuntimeError('Fresh notebook already exists; refusing duplicate submission')
        for reference in metadata['dataset_sources']:
            query = ApiGetDatasetStatusRequest()
            query.owner_slug, query.dataset_slug = reference.split('/')
            if client.datasets.dataset_api_client.get_dataset_status(query).status.name not in {'READY', 'COMPLETE'}:
                raise RuntimeError('Dataset not ready')
        query = ApiGetKernelSessionStatusRequest()
        query.user_name = 'v5nhdf'
        query.kernel_slug = 'yolo26s-vehicle-light-20261027'
        deadline = time.monotonic() + 20 * 60
        while time.monotonic() < deadline:
            try:
                status = client.kernels.kernels_api_client.get_kernel_session_status(query).status.name
            except HTTPError as error:
                if not args.user_stopped_old_session or error.response.status_code not in (403, 404):
                    raise
                quota = client.kernels.kernels_api_client.get_accelerator_quota_statistics(
                    ApiGetAcceleratorQuotaStatisticsRequest())
                if quota.gpu_quota is None or quota.gpu_quota.time_reserved.total_seconds() != 0:
                    raise RuntimeError('Old notebook unavailable but GPU still reserved; refusing duplicate job')
                state('USER_STOP_CONFIRMED_GPU_IDLE', old_status='UNAVAILABLE', gpu_reserved_seconds=0)
                break
            state('WAITING_OLD_SESSION_STOP', old_status=status)
            if status in {'COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED'}:
                break
            time.sleep(20)
        else:
            raise TimeoutError('Old session still running; fresh GPU job not submitted')
        request = ApiSaveKernelRequest()
        request.slug = metadata['id']
        request.new_title = metadata['title']
        request.text = (SOURCE / 'train.py').read_text()
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
        state('SUBMITTING_FRESH_PARALLEL', model='yolo26s.pt', gpu_assignment={'vehicle': 0, 'light': 1})
        result = client.kernels.kernels_api_client.save_kernel(request)
        if result.error or result.invalid_dataset_sources:
            raise RuntimeError('Fresh notebook submission rejected')
        (REPORT / 'fresh_submission.json').write_text(json.dumps({
            'url': result.url, 'version': result.version_number,
            'notebook': metadata['id'], 'private': True}, indent=2))
        query.kernel_slug = metadata['id'].split('/')[1]
        for attempt in range(30):
            status = client.kernels.kernels_api_client.get_kernel_session_status(query).status.name
            state('FRESH_' + status, url=result.url, training_worker_start_verified=False)
            if status in {'RUNNING', 'COMPLETE', 'ERROR', 'CANCEL_ACKNOWLEDGED'}:
                return
            time.sleep(20)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        state('FRESH_RESTART_STOPPED', error_type=type(error).__name__,
              reason=str(error) if isinstance(error, (RuntimeError, TimeoutError)) else None,
              http_status=getattr(getattr(error, 'response', None), 'status_code', None))
        sys.exit(1)
