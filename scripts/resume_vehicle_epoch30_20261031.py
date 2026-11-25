import json
import os
from pathlib import Path

from scripts import autorestart_yolo26_kaggle2 as controller
from scripts.resume_vehicle_epoch27_20261030 import safe_budgets


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'runs/project363/yolo26_vehicle_resume_20261031'
CHECKPOINTS = ROOT / 'weights/yolo26_vehicle_epoch30_20261031'
COMPLETED_EPOCHS = 30
PREVIOUS_RESTARTS = 2


def require_resumable(metadata):
    names = metadata['names']
    if isinstance(names, dict):
        names = [names[key] for key in sorted(names, key=int)]
    if (metadata['completed_epochs'] != COMPLETED_EPOCHS or not metadata['optimizer_present']
            or names != ['car', 'motorcycle', 'bus', 'truck']
            or controller.resume_reason(metadata) != 'RESUME'):
        raise RuntimeError(f'Vehicle checkpoint is not resumable at epoch {COMPLETED_EPOCHS+1}')


def main():
    import winreg
    from kagglesdk import KaggleClient, KaggleEnv
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiGetAcceleratorQuotaStatisticsRequest, ApiGetKernelSessionStatusRequest)
    from kagglesdk.security.types.oauth_service import IntrospectTokenRequest

    REPORT.mkdir(parents=True, exist_ok=True)
    submission_path = REPORT / 'submission.json'
    if submission_path.exists():
        print(submission_path.read_text(), flush=True)
        return
    controller.REPORT = REPORT
    controller.DATASETS = ['v5nhdf/yolo363-vehicle-archive-20261027']
    metadata = controller.inspect_checkpoint(CHECKPOINTS / 'vehicle/checkpoints/last.pt')
    require_resumable(metadata)
    saved = json.loads((CHECKPOINTS / 'vehicle/epoch_status.json').read_text())
    for name, expected in saved['sha256'].items():
        if controller.digest(CHECKPOINTS / 'vehicle/checkpoints' / name) != expected:
            raise RuntimeError('Vehicle checkpoint hash mismatch')
    controller.write_json(REPORT / 'manual_resume_approval.json', {
        'tasks': ['vehicle'], 'light_enabled': False, 'checkpoint': metadata,
        'authorization': 'User explicitly requested continuing vehicle training from epoch 30 on 2026-10-31',
        'policy': 'One manual restart, no automatic retry on failure; preserve optimizer and log exit code'})
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as key:
        token = winreg.QueryValueEx(key, 'KAGGLE_API_TOKEN_2')[0].strip()
    os.environ.update(KAGGLE_API_TOKEN=token, KAGGLE_API_ENVIRONMENT='PROD',
                      KAGGLE_CONFIG_DIR=str(ROOT / '.tools/kaggle2_profile'))
    for name in ('KAGGLE_USERNAME', 'KAGGLE_KEY'):
        os.environ.pop(name, None)
    with KaggleClient(env=KaggleEnv.PROD, api_token=token, verbose=False) as client:
        query = IntrospectTokenRequest()
        query.token = token
        identity = controller.read_call(client.security.oauth_client.introspect_token, query)
        if not identity.active or identity.username != 'v5nhdf':
            raise RuntimeError('Wrong Kaggle2 account')
        quota = controller.read_call(client.kernels.kernels_api_client.get_accelerator_quota_statistics,
                                     ApiGetAcceleratorQuotaStatisticsRequest()).gpu_quota
        if quota is None or quota.time_reserved.total_seconds() != 0:
            raise RuntimeError('GPU already reserved; no existing job will be interrupted')
        free = quota.total_time_allowed.total_seconds() - quota.time_used.total_seconds()
        budget = safe_budgets(free)
        if budget is None:
            raise RuntimeError('Insufficient GPU quota')
        controller.write_json(REPORT / 'quota_before.json', {'remaining_seconds': free, **budget})
        result = controller.restart(client, {'restart_count': PREVIOUS_RESTARTS}, CHECKPOINTS, {'vehicle': metadata}, budget)
        controller.write_json(submission_path, result)
        query = ApiGetKernelSessionStatusRequest()
        query.user_name, query.kernel_slug = result['reference'].split('/')
        status = controller.read_call(client.kernels.kernels_api_client.get_kernel_session_status, query)
        controller.write_json(REPORT / 'status.json', {'status': status.status.name,
                              'url': 'https://www.kaggle.com/code/' + result['reference'],
                              'tasks': ['vehicle'], 'light_enabled': False,
                              'training_worker_start_verified': False})
        print((REPORT / 'status.json').read_text(), flush=True)


if __name__ == '__main__':
    main()
