"""Show live downloads, training, paired evaluation, and available F1 results."""
import csv
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/sadhana/finetuning_run"


def show_thinking_status():
    folder = ROOT/'results/sadhana/thinking_control_20261009'
    if not (folder/'status.json').exists():
        return
    state = json.loads((folder/'status.json').read_text())
    print(f"Thinking control: {state['phase']} | updated {state.get('updated_local', '')}")
    print('  Gemma 4 / Qwen3: base GGUF, same 500 balanced records, <=500 characters; no training')
    if state['phase'] == 'pilot':
        print(f"  Excluded technical pilot: {state.get('pilot_done', 0)}/{state.get('pilot_total', 40)} calls | cap {state.get('output_cap')}")
    print(f"  Test predictions: {state.get('done_predictions', 0)}/{state.get('total_predictions', 2000)}")
    if state.get('active_model'):
        print(f"  Active: {state['active_model']} | thinking {state.get('think', '')}")
    if state.get('estimated_remaining_minutes') and state['phase']=='evaluation':
        print(f"  Rough remaining time from completed calls: {state['estimated_remaining_minutes']:.0f} min (mode/model speeds differ)")
    if state.get('error'):
        print(f"  Error: {state['error']}")
    if (folder/'summary.csv').exists():
        with (folder/'summary.csv').open() as stream:
            for row in csv.DictReader(stream):
                print(f"  {row['model']} thinking={row['think']}: F1={float(row['f1_weighted']):.4f}, mean {float(row['mean_latency_s']):.2f}s, invalid {row['invalid']}")
    print('  Results: results/sadhana/thinking_control_20261009/\n')


def show_replication_status():
    path = ROOT/'results/sadhana/replication_20261008/status.json'
    if not path.exists():
        return
    status = json.loads(path.read_text())
    jobs = status['jobs']
    complete = sum(j.get('phase') == 'complete' for j in jobs.values())
    print(f"Reviewer replications: {status['phase']} | {complete}/{status['n_jobs']} jobs complete")
    print('  Training: 500 fixed records; validation: 2,000; training seeds: 1, 2, 3')
    print('  DistilBERT: three-epoch and validation-stopped controls; LLMs: two additional fits each')
    if status.get('active_job'):
        key = status['active_job']; job = jobs[key]
        progress = job.get('progress', {})
        if job.get('live_status') and Path(job['live_status']).exists():
            progress = json.loads(Path(job['live_status']).read_text())
        heading = 'Active' if status['phase']=='running' else 'Last job'
        print(f"  {heading}: {key} | {progress.get('phase', 'initializing')}", end='')
        if progress.get('max_steps'):
            print(f" | step {progress.get('global_step', 0)}/{progress['max_steps']}", end='')
        if progress.get('total_rows'):
            print(f" | predictions {progress.get('done_rows', 0)}/{progress['total_rows']}", end='')
        if progress.get('total_datasets'):
            print(f" | test files {progress.get('done_datasets', 0)}/{progress['total_datasets']}", end='')
        print()
        if progress.get('dataset'):
            print(f"    {progress['dataset']}")
        values = progress.get('latest_metrics', {})
        chosen = ['loss', 'eval_loss', 'eval_f1_weighted']
        print('    '+', '.join(f'{k}={values[k]:.5g}' for k in chosen if k in values))
        print(f"    Log: {job['log']}")
    for key, job in jobs.items():
        if job.get('phase') in ['failed', 'blocked_by_failed_job']:
            print(f"  {key}: {job['phase']} | {job.get('log', '')}")
    print('  Run: results/sadhana/replication_20261008/')
    summary = path.parent/'qlora_seed_summary.csv'
    if summary.exists():
        with summary.open() as stream:
            rows = [r for r in csv.DictReader(stream) if r['condition']=='balanced' and r['size']=='500']
        print('  Verified balanced 500-character results (available training fits):')
        for row in rows:
            deviation = row['adapter_f1_weighted_std'] or '--'
            if deviation != '--':
                deviation = f'{float(deviation):.4f}'
            print(f"    {row['model']}: {row['training_seeds_available']} fit(s), F1={float(row['adapter_f1_weighted_mean']):.4f}, std={deviation}")
    print()
    return True


def show_evaluation_status():
    path = ROOT / 'results/sadhana/evaluation_run/status.json'
    if not path.exists():
        return
    status = json.loads(path.read_text())
    print(f"\nPaired HF base/adapter evaluation: {status['phase']}")
    print(f"  Maximum batch size: {status['batch_size']}; serial benchmark: {status['serial_samples']} per mode/model")
    for name, item in status['jobs'].items():
        print(f"  {name}: {item['phase']}", end='')
        if item.get('effective_batch_size'):
            print(f" | batch {item['effective_batch_size']}",end='')
        done, total = item.get('done_rows', 0), item.get('total_rows', 0)
        stage_total = item.get('stage_total_rows', 6000 if status['phase']=='primary_comparisons' else total)
        if total:
            print(f" | {done}/{stage_total} stage predictions ({100*done/stage_total:.1f}%)", end='')
        fresh = item.get('fresh_rows', done)
        seconds = item.get('inference_seconds', item.get('elapsed_seconds', 0))
        if fresh >= 256 and done < stage_total and seconds and item['phase']=='inference':
            remaining = seconds*(stage_total-done)/fresh/60
            print(f" | roughly {remaining:.0f} min remaining in this stage", end='')
        print()
        if item.get('dataset'):
            print(f"    {item['dataset']}, {item.get('mode', '')}")
        if item['phase'] == 'failed':
            print(f"    Inspect {item.get('log')}")
    print('  Results: results/sadhana/qlora_comparison/pooled_comparison.csv')
    report = ROOT / 'results/sadhana/qlora_comparison/pooled_comparison.csv'
    if report.exists():
        with report.open() as stream:
            rows = [r for r in csv.DictReader(stream) if r['condition']=='balanced' and r['size']=='500']
        if rows:
            print('\nBalanced 500-character tests; training: 500 records, seed 1:')
            print(f"  {'Model':<9} {'N':>5} {'HF base F1':>11} {'QLoRA F1':>10} {'Gain (pp)':>10} {'Test subsets':>13}")
            for row in rows:
                base, adapted = float(row['base_f1_weighted']), float(row['adapter_f1_weighted'])
                print(f"  {row['model']:<9} {row['n']:>5} {base:>11.4f} {adapted:>10.4f} {(adapted-base)*100:>+10.2f} {row['test_seeds_available']+'/3':>13}")


def download_lines(path):
    path = Path(path)
    if not path.exists():
        return []
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - 65536))
        tail = stream.read().decode(errors="replace")
    tail = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", tail)
    latest = {}
    for line in tail.replace("\r", "\n").splitlines():
        match = re.search(r"pulling ([a-f0-9]+):\s*(\d+)%", line)
        if match:
            latest[match[1]] = line.strip()
    return list(latest.values())


def main():
    show_thinking_status()
    if show_replication_status():
        return
    if not (RUN / "status.json").exists():
        print("No run status yet.")
        show_evaluation_status()
        return
    status = json.loads((RUN / "status.json").read_text())
    print(f"Queue: {status['phase']} | Updated: {status['updated_local']}")
    print(f"Train: {status['train_size']} records | Seeds: {status['training_seeds']}")
    print("\nOllama downloads:")
    for name, item in status["ollama"].items():
        print(f"  {item['tag']}: {item['phase']}")
        if item["phase"] == "downloading" and item.get("log"):
            for line in download_lines(item["log"]):
                print(f"    {line}")
        if item.get("error"):
            print(f"    {item['error']}")
    print("\nHF QLoRA training:")
    for name, item in status["training"].items():
        phase = item["phase"]
        print(f"  {name}: {phase}", end="")
        if "global_step" in item:
            print(f" | step {item['global_step']}/{item['max_steps']}", end="")
        if "peak_allocated_vram_gib" in item:
            print(f" | peak {item['peak_allocated_vram_gib']:.2f} GiB", end="")
        print()
        metrics = item.get("latest_metrics", {})
        values = [f"{key}={metrics[key]:.5g}" for key in ["loss", "eval_loss", "grad_norm"] if key in metrics]
        if values:
            print("    " + ", ".join(values))
        if phase == "loading_weights":
            print("    Downloading/loading HF weights; no training step recorded yet.")
        if phase == "failed":
            print(f"    Inspect {item.get('log')}")
    show_evaluation_status()


if __name__ == "__main__":
    main()
