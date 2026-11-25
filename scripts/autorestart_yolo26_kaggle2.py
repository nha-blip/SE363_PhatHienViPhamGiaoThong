import argparse
import binascii
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests

if __package__:
    from .deploy_yolo26_fullscene_resume_20261028 import canonical_notebook, read_call
    from .recover_kaggle2_zip_checkpoints import RangeReader
else:
    from deploy_yolo26_fullscene_resume_20261028 import canonical_notebook, read_call
    from recover_kaggle2_zip_checkpoints import RangeReader


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'runs/project363/yolo26_autorestart_20261028'
CONFIG = ROOT / 'config/kaggle/yolo26_fullscene_resume_20261028/autorestart.json'
SOURCE = ROOT / 'config/kaggle/yolo26_fullscene_resume_20261028'
WEIGHTS = ROOT / 'weights/yolo26_autorestart_20261028'
PREFIX = 'yolo26_fullscene_resume_20261028/'
DATASETS = ['v5nhdf/yolo363-vehicle-archive-20261027', 'v5nhdf/yolo363-fullscene-resume-20261028']


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2))
    temporary.replace(path)


def emit(status, **details):
    result = {'status': status, 'updated_at_utc': datetime.now(timezone.utc).isoformat(),
              'pid': os.getpid(), **details}
    write_json(REPORT / 'status.json', result)
    print(json.dumps(result), flush=True)


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024**2), b''):
            result.update(block)
    return result.hexdigest()


def resume_reason(metadata):
    completed = metadata['completed_epochs']
    if completed <= 0:
        raise ValueError('No completed checkpoint epoch')
    if completed >= metadata['target_epochs']:
        return 'TARGET_REACHED'
    best = metadata.get('best_epoch')
    patience = metadata.get('patience') or 0
    if patience > 0 and best is not None and completed - best >= patience:
        return 'EARLY_STOPPED'
    if not metadata['optimizer_present']:
        raise ValueError('Incomplete training has no optimizer state')
    return 'RESUME'


def budgets(free_seconds):
    if free_seconds < 2*3600:
        return None
    return {'session_timeout_seconds': int(min(12*3600, free_seconds-300)),
            'session_budget_seconds': int(min(10*3600, free_seconds-900)),
            'train_budget_seconds': int(min(8*3600, free_seconds-1800))}


def selected_member(name, tasks):
    if not name.startswith(PREFIX):
        return False
    relative = name[len(PREFIX):]
    if relative in {'completion.json', 'progress.json'}:
        return True
    return any(re.fullmatch(re.escape(kind)+r'/(?:epoch_status\.json|train\.log|'
                           r'checkpoints/(?:last|best)\.pt|train/(?:results\.csv|args\.yaml))', relative)
               for kind in tasks)


def recover(client, reference, tasks, destination):
    from kagglesdk.kernels.types.kernels_api_service import ApiListKernelSessionOutputRequest

    page = ''
    files = []
    while True:
        query = ApiListKernelSessionOutputRequest()
        query.user_name, query.kernel_slug = reference.split('/')
        query.page_size, query.page_token = 200, page
        result = read_call(client.kernels.kernels_api_client.list_kernel_session_output, query)
        files.extend(result.files)
        page = result.next_page_token
        if not page:
            break
    bundle = next((row for row in files if row.file_name == '_output_.zip'), None)
    destination.mkdir(parents=True, exist_ok=True)
    if bundle:
        with zipfile.ZipFile(RangeReader(bundle.url)) as archive:
            for name in archive.namelist():
                if not selected_member(name, tasks):
                    continue
                info = archive.getinfo(name)
                if info.file_size > 512*1024**2:
                    raise RuntimeError('Unexpectedly large output artifact')
                target = destination / name[len(PREFIX):]
                target.parent.mkdir(parents=True, exist_ok=True)
                valid = (target.is_file() and target.stat().st_size == info.file_size
                         and binascii.crc32(target.read_bytes()) & 0xffffffff == info.CRC)
                if not valid:
                    partial = target.with_suffix('.partial')
                    with archive.open(name) as source, partial.open('wb') as handle:
                        shutil.copyfileobj(source, handle, length=1024**2)
                    partial.replace(target)
    else:
        for row in files:
            if not selected_member(row.file_name, tasks):
                continue
            target = destination / row.file_name[len(PREFIX):]
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_suffix('.partial')
            with requests.get(row.url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with partial.open('wb') as handle:
                    for block in response.iter_content(1024**2):
                        handle.write(block)
            partial.replace(target)
    states = {}
    for kind in tasks:
        path = destination / kind / 'epoch_status.json'
        if not path.is_file():
            return None
        state = json.loads(path.read_text())
        if set(state['sha256']) != {'last.pt', 'best.pt'}:
            raise RuntimeError('Rolling checkpoint pair incomplete')
        for name, expected in state['sha256'].items():
            if digest(destination / kind / 'checkpoints' / name) != expected:
                raise RuntimeError('Output checkpoint SHA256 mismatch')
        states[kind] = state
    write_json(destination / 'recovery_verified.json', {'source': reference, 'tasks': states})
    return states


def inspect_checkpoint(path):
    python = Path('C:/Users/NHA/miniconda3/envs/my_env/python.exe')
    result = subprocess.run([str(python), str(ROOT / 'scripts/inspect_yolo_resume_checkpoint.py'), str(path)],
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=180)
    if result.returncode:
        raise RuntimeError('Local checkpoint inspection failed; refusing restart')
    return json.loads(result.stdout)


def prepare_resume(destination, states):
    completion = destination / 'completion.json'
    if completion.is_file():
        failed = set(json.loads(completion.read_text()).get('failed_tasks', []))
        if failed.intersection(states):
            raise RuntimeError('Worker failure recorded; diagnose before automatic restart')
    result = {}
    for kind, checkpoint in states.items():
        metadata = inspect_checkpoint(destination / kind / 'checkpoints/last.pt')
        names = metadata.get('names')
        names = [names[key] for key in sorted(names, key=int)] if isinstance(names, dict) else names
        expected = ['car', 'motorcycle', 'bus', 'truck'] if kind == 'vehicle' else ['red', 'yellow', 'green', 'off']
        if names != expected:
            raise RuntimeError('Checkpoint classes do not match task')
        if metadata['completed_epochs'] != checkpoint['epoch_completed']:
            raise RuntimeError('Checkpoint epoch differs from saved metadata')
        reason = resume_reason(metadata)
        if reason != 'RESUME':
            continue
        log = destination / kind / 'train.log'
        if log.is_file() and ('Traceback (most recent call last):' in log.read_text(errors='replace')
                             or 'CUDA out of memory' in log.read_text(errors='replace')):
            raise RuntimeError('Worker has fatal error; manual diagnosis required')
        result[kind] = metadata
    return result


def restart(client, current, destination, tasks, budget):
    from kagglesdk.datasets.types.dataset_api_service import ApiGetDatasetRequest, ApiGetDatasetStatusRequest
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelRequest, ApiSaveKernelRequest
    from kagglesdk.kernels.types.kernels_enums import KernelExecutionType
    from requests import HTTPError

    index = current['restart_count']+1
    cycle = REPORT / f'restart_{index:02d}'
    upload = cycle / 'upload'
    upload.mkdir(parents=True, exist_ok=True)
    journal_path = cycle / 'journal.json'
    journal = json.loads(journal_path.read_text()) if journal_path.exists() else {}
    slug = current.get('restart_slug') or f'yolo26-auto-resume-20261028-{index:02d}'
    dataset_slug = slug+'-checkpoints'
    config = {'tasks': {}, **budget}
    for kind, metadata in tasks.items():
        hashes = {}
        for suffix, source in [('last.pt', destination / kind / 'checkpoints/last.pt'),
                               ('best.pt', destination / kind / 'checkpoints/best.pt'),
                               ('results.csv', destination / kind / 'train/results.csv')]:
            name = f'resume_{kind}_{suffix}'
            target = upload / name
            hashes[name] = digest(source)
            if not target.exists():
                os.link(source, target)
            if digest(target) != hashes[name]:
                raise RuntimeError('Resume upload changed')
        config['tasks'][kind] = {'completed_epochs': metadata['completed_epochs'], 'sha256': hashes}
    if journal.get('dataset_create_intent'):
        uploaded_config = json.loads((upload / 'resume_state.json').read_text())
        if uploaded_config['tasks'] != config['tasks']:
            raise RuntimeError('Immutable resume dataset differs from current checkpoints')
        config = uploaded_config
    else:
        write_json(upload / 'resume_state.json', config)
    write_json(upload / 'dataset-metadata.json', {'id': 'v5nhdf/'+dataset_slug,
               'title': f'YOLO26 Auto Resume 20261028 {index:02d} Checkpoints',
               'licenses': [{'name': 'other'}], 'description': 'Private own-model optimizer checkpoints; no credentials.'})
    expected = {'resume_state.json', 'dataset-metadata.json'} | {
        f'resume_{kind}_{suffix}' for kind in tasks for suffix in ('last.pt', 'best.pt', 'results.csv')}
    if {path.name for path in upload.iterdir()} != expected:
        raise RuntimeError('Unexpected checkpoint upload payload')
    query = ApiGetDatasetRequest()
    query.owner_slug, query.dataset_slug = 'v5nhdf', dataset_slug
    try:
        dataset = read_call(client.datasets.dataset_api_client.get_dataset, query)
    except HTTPError as error:
        if error.response.status_code not in (403, 404):
            raise
        if not journal.get('dataset_create_intent'):
            journal['dataset_create_intent'] = True
            write_json(journal_path, journal)
            emit('UPLOADING_RESUME_CHECKPOINTS', restart_index=index)
            from kaggle import api
            result = api.dataset_create_new(str(upload), public=False, quiet=True, dir_mode='skip')
            if result.error:
                raise RuntimeError('Resume dataset rejected')
    else:
        if not dataset.is_private:
            raise RuntimeError('Resume dataset is not private')
    query = ApiGetDatasetStatusRequest()
    query.owner_slug, query.dataset_slug = 'v5nhdf', dataset_slug
    deadline = time.monotonic()+1800
    while time.monotonic() < deadline:
        try:
            name = read_call(client.datasets.dataset_api_client.get_dataset_status, query).status.name
        except HTTPError as error:
            if error.response.status_code not in (403, 404):
                raise
            name = 'PENDING'
        if name in {'READY', 'COMPLETE'}:
            break
        if 'ERROR' in name or 'FAIL' in name:
            raise RuntimeError('Resume dataset processing failed')
        time.sleep(30)
    else:
        raise RuntimeError('Resume dataset not ready; inspect journal before retry')
    sources = DATASETS+['v5nhdf/'+dataset_slug]
    query = ApiGetKernelRequest()
    query.user_name, query.kernel_slug = 'v5nhdf', slug
    try:
        existing = read_call(client.kernels.kernels_api_client.get_kernel, query)
    except HTTPError as error:
        if error.response.status_code not in (403, 404):
            raise
    else:
        if (not journal.get('submission_intent') or not existing.metadata.is_private
                or set(existing.metadata.dataset_data_sources) != set(sources)):
            raise RuntimeError('Unexpected existing restart notebook')
        return {'reference': 'v5nhdf/'+slug, 'tasks': list(tasks), 'restart_count': index}
    if journal.get('submission_intent'):
        raise RuntimeError('Prior submission outcome ambiguous; do not submit again')
    from kagglesdk.kernels.types.kernels_api_service import ApiGetAcceleratorQuotaStatisticsRequest
    quota = read_call(client.kernels.kernels_api_client.get_accelerator_quota_statistics,
                      ApiGetAcceleratorQuotaStatisticsRequest()).gpu_quota
    if (quota is None or quota.time_reserved.total_seconds() != 0
            or quota.total_time_allowed.total_seconds()-quota.time_used.total_seconds() < config['session_timeout_seconds']):
        raise RuntimeError('GPU quota or reservations changed during upload; no job submitted')
    code = 'import sys, types\n'
    for module_name, path in [('runtime_base', ROOT / 'config/kaggle/yolo26_vehicle_local_20261027/train.py'),
                              ('runtime_fullscene', SOURCE / 'train.py')]:
        code += (f'_module=types.ModuleType({module_name!r})\n_module.__file__=__file__\n'
                 f'sys.modules[{module_name!r}]=_module\n'
                 f'exec(compile({path.read_text()!r}, {module_name!r}, "exec"), _module.__dict__)\n')
    code += (SOURCE / 'autoresume.py').read_text()
    compile(code, 'autoresume_submitted.py', 'exec')
    (cycle / 'submitted_train.py').write_text(code)
    journal['submission_intent'] = True
    write_json(journal_path, journal)
    query = ApiSaveKernelRequest()
    query.slug, query.new_title = 'v5nhdf/'+slug, f'YOLO26 Auto Resume 20261028 {index:02d}'
    query.text, query.language, query.kernel_type = code, 'python', 'script'
    query.is_private, query.enable_gpu, query.enable_tpu, query.enable_internet = True, True, False, True
    query.machine_shape, query.session_timeout_seconds = 'NvidiaTeslaT4', config['session_timeout_seconds']
    query.dataset_data_sources = sources
    query.kernel_execution_type = KernelExecutionType.SAVE_AND_RUN_ALL
    result = client.kernels.kernels_api_client.save_kernel(query)
    if result.error or result.invalid_dataset_sources:
        raise RuntimeError('Auto-resume notebook rejected')
    reference = canonical_notebook(result.url)
    write_json(cycle / 'submission.json', {'url': result.url, 'reference': reference, 'version': result.version_number})
    return {'reference': reference, 'tasks': list(tasks), 'restart_count': index}


def watch(client, check_only):
    from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest, ApiGetAcceleratorQuotaStatisticsRequest

    config = json.loads(CONFIG.read_text())
    pointer = REPORT / 'current.json'
    current = json.loads(pointer.read_text()) if pointer.exists() else {
        'reference': config['source_notebook'], 'tasks': ['vehicle', 'light'], 'restart_count': 0}
    canonical_notebook('https://www.kaggle.com/code/'+current['reference'])
    deadline = time.monotonic()+config['watch_hours']*3600
    while time.monotonic() < deadline:
        config = json.loads(CONFIG.read_text())
        if not config['enabled'] or (REPORT / 'STOP').exists():
            emit('DISABLED_NO_RUNNING_JOB_CANCELLED')
            return
        query = ApiGetKernelSessionStatusRequest()
        query.user_name, query.kernel_slug = current['reference'].split('/')
        try:
            status = read_call(client.kernels.kernels_api_client.get_kernel_session_status, query).status.name
            emit('WATCHING_EXISTING_SESSION', **current, session_status=status)
            if check_only:
                return
            if status == 'CANCEL_ACKNOWLEDGED':
                emit('MANUAL_CANCEL_RESPECTED', **current)
                return
            if status not in {'COMPLETE', 'ERROR'}:
                time.sleep(config['poll_seconds'])
                continue
            destination = WEIGHTS / f'cycle_{current["restart_count"]:02d}'
            states = recover(client, current['reference'], current['tasks'], destination)
            if states is None:
                emit('WAITING_PUBLISHED_CHECKPOINTS', **current)
                time.sleep(config['poll_seconds'])
                continue
            tasks = prepare_resume(destination, states)
            if not tasks:
                emit('TRAINING_FINISHED_NO_RESTART', **current, backup=str(destination))
                return
            if current['restart_count'] >= config['max_restarts']:
                emit('RESTART_LIMIT_REACHED_CHECKPOINTS_SAVED', **current)
                return
            quota = read_call(client.kernels.kernels_api_client.get_accelerator_quota_statistics,
                              ApiGetAcceleratorQuotaStatisticsRequest()).gpu_quota
            budget = budgets(quota.total_time_allowed.total_seconds()-quota.time_used.total_seconds()) if quota else None
            if budget is None or quota.time_reserved.total_seconds() != 0:
                emit('WAITING_QUOTA_OR_OTHER_GPU_JOB', **current)
                time.sleep(600)
                continue
            current = restart(client, current, destination, tasks, budget)
            write_json(pointer, current)
            emit('AUTO_RESUME_SUBMITTED', **current)
        except requests.RequestException as error:
            if getattr(getattr(error, 'response', None), 'status_code', None) in (401, 403, 404):
                raise RuntimeError('Account or notebook access unavailable; refusing restart') from None
            emit('NETWORK_RETRY_NO_DUPLICATE_SUBMISSION', **current, error_type=type(error).__name__)
            if check_only:
                raise RuntimeError('Preflight network unavailable') from None
            time.sleep(config['poll_seconds'])
    emit('WATCH_WINDOW_EXPIRED_NO_JOB_CANCELLED', **current)


def main():
    import msvcrt
    import winreg
    from kagglesdk import KaggleClient, KaggleEnv
    from kagglesdk.security.types.oauth_service import IntrospectTokenRequest

    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    REPORT.mkdir(parents=True, exist_ok=True)
    with (REPORT / 'controller.lock').open('a+b') as lock:
        lock.write(b'0')
        lock.flush()
        lock.seek(0)
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            print('Controller already active; no duplicate started', flush=True)
            return
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
            watch(client, args.check_only)


if __name__ == '__main__':
    try:
        main()
    except BlockingIOError:
        print('Controller already active; no duplicate started', flush=True)
        sys.exit(1)
    except Exception as error:
        emit('AUTO_RESTART_STOPPED_SAFELY', error_type=type(error).__name__,
             reason=str(error) if isinstance(error, (ValueError, RuntimeError)) else None)
        sys.exit(1)
