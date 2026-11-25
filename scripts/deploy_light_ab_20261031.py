import argparse
import hashlib
import json
import os
import time
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ACCOUNT = 'nhatquang6'
TOKEN_NAME = 'KAGGLE_API_TOKEN_3'
DATASET = 'yolo363-light-ab-20261031'
KERNEL = 'yolo26-light-ab-20261031'
REPORT = ROOT / 'runs/project363/yolo26_light_ab_20261031'
UPLOAD = ROOT / 'exports/yolo26_light_ab_20261031/upload'
FULLSCENE = ROOT / 'exports/yolo26_fullscene_resume_20261028/upload/light_fullscene.zip.bin'
FULLSCENE_SHA256 = '30ba2e5fd64c6c0f2ab59bed1b860ddb9332af740d108bb76733764f91137952'
LISA = ROOT / 'data/LISA_light_candidate_20261030'
CODE = ROOT / 'config/kaggle/yolo26_light_ab_20261031/train.py'
SESSION_TIMEOUT_SECONDS = 11 * 3600


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 24), b''):
            result.update(block)
    return result.hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def lisa_members(root):
    """List candidate files under the a_lisa archive root, without manifests that could diverge from the images."""
    members = []
    for split in ('train', 'holdout'):
        for category in ('images', 'labels'):
            for path in sorted((root / category / split).iterdir()):
                members.append((path, f'a_lisa/{category}/{split}/{path.name}'))
    return members


def package():
    UPLOAD.mkdir(parents=True, exist_ok=True)
    if digest(FULLSCENE) != FULLSCENE_SHA256:
        raise RuntimeError('Full-scene archive changed')
    target = UPLOAD / FULLSCENE.name
    if not target.exists():
        os.link(FULLSCENE, target)
    archive_path = UPLOAD / 'lisa_candidate.zip.bin'
    if not archive_path.exists():
        temporary = archive_path.with_suffix('.partial')
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED) as archive:
            for path, name in lisa_members(LISA):
                archive.write(path, name)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise RuntimeError('LISA archive CRC failure')
        temporary.replace(archive_path)
    write_json(UPLOAD / 'light_ab_package.json', {'fullscene_sha256': FULLSCENE_SHA256, 'lisa_sha256': digest(archive_path),
                                                  'lisa_license': 'CC BY-NC-SA 4.0', 'test_uploaded': False})
    write_json(UPLOAD / 'dataset-metadata.json', {'id': f'{ACCOUNT}/{DATASET}', 'title': 'YOLO363 Light AB 20261031',
                                                  'licenses': [{'name': 'other'}],
                                                  'description': 'Private project experiment data; no credentials.'})
    expected = {FULLSCENE.name, archive_path.name, 'light_ab_package.json', 'dataset-metadata.json'}
    if {path.name for path in UPLOAD.iterdir()} != expected:
        raise RuntimeError('Unexpected upload payload')


def main():
    import winreg

    parser = argparse.ArgumentParser(description='Upload private A/B data and submit one light training notebook')
    parser.add_argument('--package-only', action='store_true')
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    package()
    print('PACKAGED', flush=True)
    if args.package_only:
        return
    submission = REPORT / 'submission.json'
    if submission.exists():
        print(submission.read_text(), flush=True)
        return
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, 'Environment') as key:
        token = winreg.QueryValueEx(key, TOKEN_NAME)[0].strip()
    os.environ.update(KAGGLE_API_TOKEN=token, KAGGLE_API_ENVIRONMENT='PROD', KAGGLE_CONFIG_DIR=str(ROOT / '.tools/kaggle3_profile'))
    (ROOT / '.tools/kaggle3_profile').mkdir(parents=True, exist_ok=True)
    for name in ('KAGGLE_USERNAME', 'KAGGLE_KEY'):
        os.environ.pop(name, None)
    from kagglesdk import KaggleClient, KaggleEnv
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest, ApiGetDatasetStatusRequest
    from kagglesdk.kernels.types.kernels_api_service import (
        ApiGetAcceleratorQuotaStatisticsRequest, ApiGetKernelRequest, ApiSaveKernelRequest)
    from kagglesdk.kernels.types.kernels_enums import KernelExecutionType
    from kagglesdk.security.types.oauth_service import IntrospectTokenRequest
    from requests import HTTPError

    journal_path = REPORT / 'journal.json'
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else {}
    with KaggleClient(env=KaggleEnv.PROD, api_token=token, verbose=False) as client:
        query = IntrospectTokenRequest()
        query.token = token
        identity = client.security.oauth_client.introspect_token(query)
        if not identity.active or identity.username != ACCOUNT:
            raise RuntimeError('Wrong Kaggle account')
        query = ApiGetDatasetRequest()
        query.owner_slug, query.dataset_slug = ACCOUNT, DATASET
        try:
            dataset = client.datasets.dataset_api_client.get_dataset(query)
        except HTTPError as error:
            if error.response.status_code not in (403, 404):
                raise
            if journal.get('dataset_create_intent'):
                print('DATASET_UPLOAD_INTENT_RECORDED_WAITING', flush=True)
            else:
                journal['dataset_create_intent'] = True
                write_json(journal_path, journal)
                print('UPLOADING', flush=True)
                from kaggle import api
                result = api.dataset_create_new(str(UPLOAD), public=False, quiet=True, dir_mode='skip')
                if result.error:
                    raise RuntimeError('Dataset rejected')
        else:
            if not dataset.is_private:
                raise RuntimeError('Dataset is not private')
        query = ApiGetDatasetStatusRequest()
        query.owner_slug, query.dataset_slug = ACCOUNT, DATASET
        deadline = time.monotonic() + 3600
        while time.monotonic() < deadline:
            try:
                name = client.datasets.dataset_api_client.get_dataset_status(query).status.name
            except HTTPError as error:
                if error.response.status_code not in (403, 404):
                    raise
                name = 'PENDING'
            print('DATASET_STATUS', name, flush=True)
            if name in {'READY', 'COMPLETE'}:
                break
            if 'ERROR' in name or 'FAIL' in name:
                raise RuntimeError('Dataset processing failed')
            time.sleep(30)
        else:
            raise RuntimeError('Dataset not ready; no notebook submitted')
        query = ApiGetDatasetRequest()
        query.owner_slug, query.dataset_slug = ACCOUNT, DATASET
        if not client.datasets.dataset_api_client.get_dataset(query).is_private:
            raise RuntimeError('Dataset is not private')
        quota = client.kernels.kernels_api_client.get_accelerator_quota_statistics(ApiGetAcceleratorQuotaStatisticsRequest()).gpu_quota
        if (quota is None or quota.time_reserved.total_seconds() != 0
                or quota.total_time_allowed.total_seconds() - quota.time_used.total_seconds() < SESSION_TIMEOUT_SECONDS):
            raise RuntimeError('GPU quota or reservation changed; no notebook submitted')
        if journal.get('submission_intent'):
            raise RuntimeError('Prior submission outcome ambiguous; inspect before retrying')
        journal['submission_intent'] = True
        write_json(journal_path, journal)
        query = ApiSaveKernelRequest()
        query.slug, query.new_title = f'{ACCOUNT}/{KERNEL}', 'YOLO26 Light AB 20261031'
        query.text, query.language, query.kernel_type = CODE.read_text(encoding='utf-8'), 'python', 'script'
        query.is_private, query.enable_gpu, query.enable_tpu, query.enable_internet = True, True, False, True
        query.machine_shape, query.session_timeout_seconds = 'NvidiaTeslaT4', SESSION_TIMEOUT_SECONDS
        query.dataset_data_sources = [f'{ACCOUNT}/{DATASET}']
        query.kernel_execution_type = KernelExecutionType.SAVE_AND_RUN_ALL
        result = client.kernels.kernels_api_client.save_kernel(query)
        if result.error or result.invalid_dataset_sources:
            raise RuntimeError('Notebook rejected')
        query = ApiGetKernelRequest()
        query.user_name, query.kernel_slug = ACCOUNT, KERNEL
        metadata = client.kernels.kernels_api_client.get_kernel(query).metadata
        record = {'url': result.url, 'version': result.version_number, 'is_private': metadata.is_private,
                  'sources': list(metadata.dataset_data_sources), 'machine_shape': metadata.machine_shape}
        write_json(submission, record)
        print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
