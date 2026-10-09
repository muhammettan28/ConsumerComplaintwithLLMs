"""Persistent sequential paired evaluation and automatic comparison reports."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if __package__:
    from .evaluate_qlora_pair import atomic_json
else:
    from evaluate_qlora_pair import atomic_json

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=['gemma4','qwen3','phi4'],default=['gemma4','qwen3','phi4'])
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--serial-samples',type=int,default=100)
    args=parser.parse_args()
    lock=(ROOT/'checkpoints/.qlora_queue.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=ROOT/'results/sadhana/evaluation_run';run.mkdir(parents=True,exist_ok=True)
    state={'phase':'running','started_local':datetime.now().astimezone().isoformat(),
        'batch_size':args.batch_size,'serial_samples':args.serial_samples,'jobs':{m:{'phase':'queued'} for m in args.models}}
    def update(name=None,detail=None,**changes):
        if name:state['jobs'][name]=detail
        state.update(changes);state['updated_local']=datetime.now().astimezone().isoformat()
        atomic_json(run/'status.json',state)
    update()
    env=os.environ.copy();env.update({'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_PROGRESS_BARS':'1',
        'TOKENIZERS_PARALLELISM':'false','PYTHONUNBUFFERED':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'4'})
    def summarize(replicates):
        return subprocess.run([sys.executable,'-u',str(ROOT/'utils/summarize_qlora_comparison.py'),
            '--bootstrap',str(replicates)],cwd=ROOT,env=env).returncode
    last_pairs=-1
    for stage, name in [(stage,name) for stage in ['primary','full'] for name in args.models]:
        update(phase='primary_comparisons' if stage=='primary' else 'remaining_tests')
        base=ROOT/f'results/sadhana/hf_base/{name}'
        adapted=ROOT/f'results/sadhana/qlora/{name}_n500_s1'
        logfile=run/f'{name}.log';live=run/name/'run_status.json'
        if (base/'evaluation_complete.json').exists() and (adapted/'evaluation_complete.json').exists():
            update(name,{'phase':'complete','already_completed':True});continue
        if stage=='primary' and all((out/f'preds_{seed}_subset_balanced_500.csv').exists()
            for out in [base,adapted] for seed in [1,2,3]):
            update(name,{'phase':'primary_complete','already_completed':True});continue
        command=[sys.executable,'-u',str(ROOT/'utils/evaluate_qlora_pair.py'),'--adapter',str(ROOT/f'checkpoints/{name}/n500/seed1'),
            '--base-output',str(base),'--adapter-output',str(adapted),'--status',str(live),
            '--batch-size',str(args.batch_size),'--serial-samples',str(args.serial_samples),'--resume']
        if stage=='primary':command.extend(['--stop-after-datasets','3'])
        print('Launching',' '.join(command),flush=True)
        with logfile.open('a') as output:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT)
            while child.poll() is None:
                detail=json.loads(live.read_text()) if live.exists() else {'phase':'initializing'}
                update(name,{**detail,'pid':child.pid,'log':str(logfile)})
                pairs=sum(1 for model in args.models for p in (ROOT/f'results/sadhana/hf_base/{model}').glob('preds_*.csv')
                          if (ROOT/f'results/sadhana/qlora/{model}_n500_s1'/p.name).exists())
                if pairs!=last_pairs:
                    if pairs and summarize(0):print('Partial summary failed; inspect queue journal',flush=True)
                    last_pairs=pairs
                time.sleep(5)
        success=child.returncode==0 and (stage=='primary' or
            ((base/'evaluation_complete.json').exists() and (adapted/'evaluation_complete.json').exists()))
        phase=('primary_complete' if stage=='primary' else 'complete') if success else 'failed'
        detail=json.loads(live.read_text()) if live.exists() else {}
        update(name,{**detail,'phase':phase,'exit_code':child.returncode,'log':str(logfile)})
        print(name,state['jobs'][name]['phase'],flush=True)
    failed=any(j['phase']=='failed' for j in state['jobs'].values())
    update(phase='summarizing')
    summary_exit=summarize(5000)
    update(phase='completed_with_failures' if failed or summary_exit else 'complete',summary_exit_code=summary_exit)
    if failed or summary_exit:raise SystemExit(1)


if __name__=='__main__':main()
