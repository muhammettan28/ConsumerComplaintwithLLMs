"""Persistent, cached, sequential GPU jobs for reviewer-requested replications."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/sadhana/replication_20261008'


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'checkpoints/.qlora_queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((ROOT / 'dataset/finetuning/manifest.json').read_text())
    for relative, expected in manifest['sha256'].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Dataset changed: {relative}')
    encoder_revision = json.loads((ROOT / 'results/sadhana/distilbert/n20000/seed1/training_report.json').read_text())['hf_revision']
    jobs = []
    def add(key, command, completed=None, live=None, requires=()):
        jobs.append({'key': key, 'command': [sys.executable, '-u', *map(str, command)],
                     'completed_artifact': str(completed) if completed else None,
                     'live_status': str(live) if live else None, 'requires': list(requires)})
    for seed in [1, 2, 3]:
        out = ROOT / f'checkpoints/distilbert/n500/seed{seed}'
        add(f'distilbert_n500_s{seed}', [ROOT/'utils/train_encoder_baseline.py', '--train-size', 500,
            '--seed', seed, '--revision', encoder_revision, '--resume'],
            ROOT/f'results/sadhana/distilbert/n500/seed{seed}/training_report.json', out/'run_status.json')
    for seed in [1, 2, 3]:
        out = ROOT / f'checkpoints/distilbert/n500_convergence/seed{seed}'
        add(f'distilbert_n500_convergence_s{seed}', [ROOT/'utils/train_encoder_baseline.py', '--train-size', 500,
            '--seed', seed, '--revision', encoder_revision, '--epochs', 30,
            '--run-tag', 'convergence', '--early-stopping-patience', 5, '--resume'],
            ROOT/f'results/sadhana/distilbert/n500_convergence/seed{seed}/training_report.json', out/'run_status.json')
    for name in ['gemma4', 'qwen3', 'phi4']:
        revision = json.loads((ROOT/f'checkpoints/{name}/n500/seed1/training_report.json').read_text())['hf_revision']
        for seed in [2, 3]:
            train_key = f'{name}_n500_s{seed}_train'
            out = ROOT/f'checkpoints/{name}/n500/seed{seed}'
            add(train_key, [ROOT/'utils/train_qlora.py', '--model', name, '--train-size', 500,
                '--seed', seed, '--revision', revision, '--resume'], out/'training_report.json', out/'run_status.json')
            live = RUN/f'evaluation/{name}_s{seed}/run_status.json'
            adapted = ROOT/f'results/sadhana/qlora/{name}_n500_s{seed}'
            base = ROOT/f'results/sadhana/hf_base_repeats/n500/seed{seed}/{name}'
            add(f'{name}_n500_s{seed}_evaluate', [ROOT/'utils/evaluate_qlora_pair.py', '--adapter', out,
                '--base-output', base, '--adapter-output', adapted, '--status', live,
                '--batch-size', 32, '--serial-samples', 100, '--resume'],
                adapted/'evaluation_complete.json', live, [train_key])
    for seed in [2, 3]:
        base = ROOT/f'results/sadhana/hf_base_repeats/n500/seed{seed}'
        comparison = RUN/f'seed{seed}_comparison'
        summary_key = f'seed{seed}_statistics'
        add(summary_key, [ROOT/'utils/summarize_qlora_comparison.py', '--training-seed', seed,
            '--base-root', base, '--output', comparison, '--bootstrap', 5000],
            requires=[f'{name}_n500_s{seed}_evaluate' for name in ['gemma4','qwen3','phi4']])
        add(f'seed{seed}_verify', [ROOT/'utils/verify_qlora_evaluation.py', '--training-seed', seed,
            '--base-root', base, '--comparison-dir', comparison], requires=[summary_key])
    add('aggregate_replications', [ROOT/'utils/summarize_replications.py'],
        requires=['seed2_verify', 'seed3_verify', *[f'distilbert_n500_s{s}' for s in [1,2,3]],
                  *[f'distilbert_n500_convergence_s{s}' for s in [1,2,3]]])
    plan = {'train_size': 500, 'validation_size': 2000, 'training_seeds': [1,2,3],
            'new_encoder_fits': 6, 'new_llm_fits': 6, 'new_llm_test_predictions': 144000,
            'original_test_sha256': {k:v for k,v in manifest['sha256'].items() if 'subset' in k},
            'protocol': 'Same splits, pinned HF commits and hyperparameters; seed1 LLM outputs retained; each fit evaluated separately on the same tests',
            'network': 'Offline HF cache; no Ollama pulls',
            'temporal_evaluation': 'Unavailable: local CSV has no dates or original CFPB IDs', 'jobs': jobs}
    plan['encoder_convergence_control'] = {'max_epochs': 30, 'early_stopping_patience': 5,
        'selection': 'validation weighted F1 only',
        'reason': 'Seed1 three-epoch validation F1 and loss were still improving; convergence control chosen before inspecting new test scores',
        'preservation': 'Three-epoch controls retained; convergence fits stored separately'}
    atomic_json(RUN/'study_plan.json', plan)
    previous = json.loads((RUN/'status.json').read_text()) if args.resume and (RUN/'status.json').exists() else {}
    state = {'phase': 'running', 'started_local': previous.get('started_local', datetime.now().astimezone().isoformat()),
             'jobs': previous.get('jobs', {}), 'train_size': 500, 'validation_size': 2000,
             'training_seeds': [1,2,3], 'n_jobs': len(jobs), 'active_job': None}
    def update():
        state['updated_local'] = datetime.now().astimezone().isoformat()
        atomic_json(RUN/'status.json', state)
    env = os.environ.copy()
    env.update({'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'HF_HUB_DISABLE_PROGRESS_BARS': '1',
                'TOKENIZERS_PARALLELISM': 'false', 'PYTHONUNBUFFERED': '1',
                'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '4'})
    update()
    for job in jobs:
        key = job['key']
        if any(state['jobs'].get(dep, {}).get('phase') != 'complete' for dep in job['requires']):
            state['jobs'][key] = {'phase': 'blocked_by_failed_job', 'requires': job['requires']}; update(); continue
        if job['completed_artifact'] and Path(job['completed_artifact']).exists():
            state['jobs'][key] = {'phase': 'complete', 'already_completed': True}; update(); continue
        if args.resume and not job['completed_artifact'] and state['jobs'].get(key, {}).get('phase') == 'complete':
            continue
        log = RUN/'logs'/f'{key}.log'; log.parent.mkdir(exist_ok=True)
        state['active_job'] = key
        started = time.monotonic()
        state['jobs'][key] = {'phase': 'running', 'log': str(log), 'live_status': job['live_status'],
                              'started_local': datetime.now().astimezone().isoformat()}
        update(); print('Launching', key, ' '.join(job['command']), flush=True)
        with log.open('a') as output:
            child = subprocess.Popen(job['command'], cwd=ROOT, env=env, stdout=output, stderr=subprocess.STDOUT)
            while child.poll() is None:
                state['jobs'][key].update({'pid': child.pid, 'elapsed_seconds': time.monotonic()-started})
                if job['live_status'] and Path(job['live_status']).exists():
                    state['jobs'][key]['progress'] = json.loads(Path(job['live_status']).read_text())
                update(); time.sleep(5)
        state['jobs'][key].update({'phase': 'complete' if child.returncode == 0 else 'failed',
                                  'exit_code': child.returncode, 'elapsed_seconds': time.monotonic()-started})
        update(); print(key, state['jobs'][key]['phase'], flush=True)
    state['active_job'] = None
    failed = any(item['phase'] != 'complete' for item in state['jobs'].values())
    state['phase'] = 'completed_with_failures' if failed else 'complete'
    update()
    if failed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
