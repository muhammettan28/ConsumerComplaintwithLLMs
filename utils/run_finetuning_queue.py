"""Download the requested Ollama builds and train HF QLoRA models sequentially.

Run under a systemd user service so downloads/training continue after this chat
turn. Only one training subprocess runs at a time; no complaint data are sent
to remote services. HF requests download public model artifacts only.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
TAGS = {"gemma4": "gemma4:12b", "qwen3": "qwen3:14b", "phi4": "phi4:14b"}


def api(endpoint, data=None):
    request = urllib.request.Request("http://127.0.0.1:11434/api/"+endpoint,
        data=json.dumps(data).encode() if data is not None else None,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=TAGS, default=list(TAGS))
    parser.add_argument("--train-size", type=int, choices=[500, 2000, 10000, 20000], default=500)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1])
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/sadhana/finetuning_run")
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "checkpoints").mkdir(exist_ok=True)
    lockfile = (ROOT / "checkpoints/.qlora_queue.lock").open("w")
    fcntl.flock(lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = json.loads((ROOT / "dataset/finetuning/manifest.json").read_text())
    for relative, expected in manifest["sha256"].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Dataset/test hash changed: {relative}")
    condition = threading.Lock()
    state = {"train_size": args.train_size, "training_seeds": args.seeds,
        "started_local": datetime.now(ZoneInfo("Europe/Istanbul")).isoformat(),
        "phase": "running",
        "ollama": {name: {"phase": "queued", "tag": TAGS[name]} for name in args.models},
        "training": {f"{name}_n{args.train_size}_seed{seed}": {"phase": "queued"}
                     for name in args.models for seed in args.seeds},
        "protocol": "NF4 + BF16, r=8, 3 epochs, completion-only loss; fixed original tests"}

    def update(group=None, key=None, value=None, **changes):
        with condition:
            if group:
                state[group][key] = value
            state.update(changes)
            state["updated_local"] = datetime.now(ZoneInfo("Europe/Istanbul")).isoformat()
            tmp = args.run_dir / "status.tmp"
            tmp.write_text(json.dumps(state, indent=2))
            tmp.replace(args.run_dir / "status.json")

    update()

    def pull_models():
        for name in args.models:
            logfile = args.run_dir / f"ollama_{name}.log"
            update("ollama", name, {"phase": "downloading", "tag": TAGS[name], "log": str(logfile)})
            try:
                with logfile.open("a") as output:
                    result = subprocess.run(["ollama", "pull", TAGS[name]], stdout=output, stderr=subprocess.STDOUT)
                if result.returncode:
                    raise RuntimeError(f"ollama pull exit code {result.returncode}")
                metadata = {"tag": TAGS[name], "version": api("version"),
                    "show": api("show", {"model": TAGS[name]}),
                    "tags": api("tags")}
                metadata_path = args.run_dir / f"ollama_{name}_metadata.json"
                metadata_path.write_text(json.dumps(metadata, indent=2))
                update("ollama", name, {"phase": "complete", "tag": TAGS[name], "metadata": str(metadata_path)})
                print(f"Ollama ready: {TAGS[name]}", flush=True)
            except Exception as error:
                update("ollama", name, {"phase": "failed", "tag": TAGS[name], "error": str(error), "log": str(logfile)})
                print(f"Ollama failed: {name}: {error}", flush=True)

    pulls = threading.Thread(target=pull_models, daemon=False)
    pulls.start()
    environment = os.environ.copy()
    environment.update({"PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                        "TOKENIZERS_PARALLELISM": "false"})
    for name in args.models:
        preflight = json.loads((ROOT / f"results/sadhana/{name}_preflight.json").read_text())
        for seed in args.seeds:
            key = f"{name}_n{args.train_size}_seed{seed}"
            destination = ROOT / f"checkpoints/{name}/n{args.train_size}/seed{seed}"
            report_path = destination / "training_report.json"
            if report_path.exists():
                update("training", key, {"phase": "complete", "report": str(report_path), "already_completed": True})
                continue
            logfile = args.run_dir / f"{key}.log"
            command = [sys.executable, "-u", str(ROOT / "utils/train_qlora.py"),
                "--model", name, "--revision", preflight["revision"],
                "--train-size", str(args.train_size), "--seed", str(seed)]
            if destination.exists() and any(destination.iterdir()):
                command.append("--resume")
            update("training", key, {"phase": "initializing", "log": str(logfile),
                "live_status": str(destination / "run_status.json"), "hf_revision": preflight["revision"]})
            print(f"Launching {' '.join(command)}", flush=True)
            with logfile.open("a") as output:
                child = subprocess.Popen(command, cwd=ROOT, env=environment,
                    stdout=output, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    run_status = destination / "run_status.json"
                    detail = json.loads(run_status.read_text()) if run_status.exists() else {"phase": "initializing"}
                    update("training", key, {**detail, "pid": child.pid,
                        "log": str(logfile), "live_status": str(run_status), "hf_revision": preflight["revision"]})
                    time.sleep(5)
            if child.returncode == 0 and report_path.exists():
                update("training", key, {"phase": "complete", "report": str(report_path), "log": str(logfile)})
                print(f"Training complete: {key}", flush=True)
            else:
                update("training", key, {"phase": "failed", "exit_code": child.returncode, "log": str(logfile)})
                print(f"Training failed: {key}; inspect {logfile}", flush=True)
    update(phase="waiting_for_ollama" if pulls.is_alive() else "finishing")
    pulls.join()
    failed = any(item.get("phase") == "failed" for group in ["ollama", "training"] for item in state[group].values())
    update(phase="completed_with_failures" if failed else "complete")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
