import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'runs/project363/kaggle2_connection_20261027/account_status.json'


def load_token():
    import winreg

    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as key:
        token = winreg.QueryValueEx(key, 'KAGGLE_API_TOKEN_2')[0].strip()
    if not token.startswith('KGAT') or len(token) < 20:
        raise ValueError('Invalid token format')
    return token


def quota_hours(quota):
    if quota is None:
        return None
    used = quota.time_used.total_seconds() / 3600
    total = quota.total_time_allowed.total_seconds() / 3600
    return {'used_hours': used, 'total_hours': total, 'remaining_hours': max(0, total-used)}


def main():
    from kagglesdk import KaggleClient, KaggleEnv
    from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest, ApiListKernelsRequest
    from kagglesdk.kernels.types.kernels_enums import KernelsListViewType
    from kagglesdk.security.types.oauth_service import IntrospectTokenRequest

    report = {'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'cli_version': version('kaggle'), 'sdk_version': version('kagglesdk'),
              'credential_variable': 'KAGGLE_API_TOKEN_2', 'token_written_to_disk': False,
              'first_account_modified': False, 'remote_writes_performed': False,
              'training_started': False}
    stage = 'load_token'
    try:
        token = load_token()
        with KaggleClient(env=KaggleEnv.PROD, api_token=token, verbose=False) as client:
            http = client.http_client()
            original_send = http._session.send

            def send(request, **kwargs):
                kwargs.setdefault('timeout', 35)
                return original_send(request, **kwargs)

            http._session.send = send
            stage = 'token_introspection'
            request = IntrospectTokenRequest()
            request.token = token
            identity = client.security.oauth_client.introspect_token(request)
            report['token_active'] = identity.active
            report['username'] = identity.username
            if not identity.active or not identity.username:
                raise ValueError('Token introspection did not confirm identity')
            stage = 'accelerator_quota'
            try:
                quota = client.kernels.kernels_api_client.get_accelerator_quota_statistics(ApiGetAcceleratorQuotaStatisticsRequest())
                report['gpu_quota'] = quota_hours(quota.gpu_quota)
                report['tpu_quota'] = quota_hours(quota.tpu_quota)
                report['quota_refresh_time'] = quota.quota_refresh_time.isoformat() if quota.quota_refresh_time else None
            except Exception as error:
                response = getattr(error, 'response', None)
                report['quota_error'] = {'type': type(error).__name__, 'http_status': getattr(response, 'status_code', None)}
            stage = 'list_own_notebooks'
            request = ApiListKernelsRequest()
            request.group = KernelsListViewType.PROFILE
            request.page_size = 1
            request.page = 1
            notebooks = client.kernels.kernels_api_client.list_kernels(request)
            report['private_profile_read_verified'] = True
            report['notebooks_returned_limited_to_one'] = len(notebooks.kernels)
            report['status'] = 'ACCOUNT_READ_AND_QUOTA_VERIFIED' if 'gpu_quota' in report else 'ACCOUNT_READ_VERIFIED_QUOTA_UNAVAILABLE'
    except Exception as error:
        response = getattr(error, 'response', None)
        report.update(status='CHECK_FAILED', failed_stage=stage,
                      error_type=type(error).__name__, http_status=getattr(response, 'status_code', None))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2), encoding='utf-8')
    temporary.replace(OUTPUT)
    print(json.dumps(report, indent=2), flush=True)
    return 1 if report['status'] == 'CHECK_FAILED' else 0


if __name__ == '__main__':
    raise SystemExit(main())
