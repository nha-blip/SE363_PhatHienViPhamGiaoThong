import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / '.tools/kaggle2_cli/Scripts/python.exe'


def main():
    import winreg

    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

    arguments = sys.argv[1:]
    if not arguments or arguments[0] not in {'datasets', 'kernels', 'quota'}:
        raise SystemExit('Allowed command groups: datasets, kernels, quota')
    if any(value in {'delete', '--debug', '--verbose'} for value in arguments):
        raise SystemExit('Destructive or verbose credential-sensitive commands are disabled')
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as key:
        token = winreg.QueryValueEx(key, 'KAGGLE_API_TOKEN_2')[0].strip()
    if not token.startswith('KGAT'):
        raise SystemExit('Missing Kaggle2 token')
    from kagglesdk import KaggleClient, KaggleEnv
    from kagglesdk.security.types.oauth_service import IntrospectTokenRequest

    with KaggleClient(env=KaggleEnv.PROD, api_token=token, verbose=False) as client:
        request = IntrospectTokenRequest()
        request.token = token
        identity = client.security.oauth_client.introspect_token(request)
        if not identity.active or identity.username != 'v5nhdf':
            raise SystemExit('Token is not the approved Kaggle2 account v5nhdf')
    environment = os.environ.copy()
    environment['KAGGLE_API_TOKEN'] = token
    environment['KAGGLE_API_ENVIRONMENT'] = 'PROD'
    environment['KAGGLE_CONFIG_DIR'] = str(ROOT / '.tools/kaggle2_profile')
    environment['PYTHONIOENCODING'] = 'utf-8'
    for name in ('KAGGLE_USERNAME', 'KAGGLE_KEY'):
        environment.pop(name, None)
    result = subprocess.run([str(PYTHON), '-m', 'kaggle', *arguments], env=environment,
                            capture_output=True, text=True, encoding='utf-8', errors='replace')
    if result.stdout:
        print(result.stdout.replace(token, '[REDACTED]'), end='', flush=True)
    if result.stderr:
        print(result.stderr.replace(token, '[REDACTED]'), end='', file=sys.stderr, flush=True)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
