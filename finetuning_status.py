"""Show live downloads, training, paired evaluation, and available F1 results."""
import csv
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "results/sadhana/finetuning_run"


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
