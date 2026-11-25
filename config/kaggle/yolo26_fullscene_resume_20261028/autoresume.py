import argparse
import json
import os
import shutil

import runtime_base as base
import runtime_fullscene as fullscene


original_worker = base.worker


def settings():
    return json.loads(fullscene.input_file('resume_state.json').read_text())


def configure():
    configuration = settings()
    base.TASKS = tuple(configuration['tasks'])
    if not base.TASKS or any(kind not in {'vehicle', 'light'} for kind in base.TASKS):
        raise RuntimeError('Invalid resume tasks')
    base.TRAIN_BUDGET_SECONDS = configuration['train_budget_seconds']
    base.SESSION_BUDGET_SECONDS = configuration['session_budget_seconds']
    base.stage_light = fullscene.stage_light
    base.worker = worker


def worker(kind):
    configure()
    configuration = settings()['tasks'][kind]
    files = {}
    for suffix in ('last.pt', 'best.pt', 'results.csv'):
        name = f'resume_{kind}_{suffix}'
        source = fullscene.input_file(name)
        fullscene.verify(source, configuration['sha256'][name])
        files[suffix] = source
    original_worker(kind, resume_checkpoint=files['last.pt'], resume_best=files['best.pt'],
                    resume_results=files['results.csv'], resume_start_epoch=configuration['completed_epochs'])


if __name__ == '__main__':
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = '1'
    configure()
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=('vehicle', 'light'))
    args = parser.parse_args()
    if args.task:
        worker(args.task)
    else:
        base.main()
        for kind in base.TASKS:
            staging = (base.WORK / kind / 'dataset').resolve()
            if staging.parent != (base.WORK / kind).resolve() or not staging.is_relative_to(base.WORK.resolve()):
                raise RuntimeError('Unsafe staging cleanup')
            if staging.is_dir():
                shutil.rmtree(staging)
        base.emit('STAGING_REMOVED_OUTPUTS_KEPT')
