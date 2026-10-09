"""Fine-tune DistilBERT using the same split as QLoRA, select only on validation.

python utils/train_encoder_baseline.py --train-size 20000 --seed 1
"""
import argparse
import hashlib
import json
import math
import re
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer, EarlyStoppingCallback,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

if __package__:
    from .sadhana_audit import CATEGORIES, ROOT, matrix, metrics
else:
    from sadhana_audit import CATEGORIES, ROOT, matrix, metrics
if __package__:
    from .train_qlora import ProgressCallback
else:
    from train_qlora import ProgressCallback


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, choices=[500, 2000, 10000, 20000], default=20000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--revision", help="Pin the same HF commit as existing encoder fits")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument('--run-tag', default='', help='Separate validation-guided convergence controls')
    parser.add_argument('--early-stopping-patience', type=int)
    args = parser.parse_args()
    if args.run_tag and not re.fullmatch(r'[a-z0-9_]+', args.run_tag):
        parser.error('Invalid run tag')
    if args.early_stopping_patience is not None and args.early_stopping_patience < 1:
        parser.error('Early-stopping patience must be positive')
    set_seed(args.seed)
    from huggingface_hub import HfApi
    model_id = "distilbert/distilbert-base-uncased"
    revision = args.revision or HfApi().model_info(model_id).sha
    manifest = json.loads((ROOT / 'dataset/finetuning/manifest.json').read_text())
    for relative, digest in manifest['sha256'].items():
        if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Data changed: {relative}')
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    label_map = {c: i for i, c in enumerate(CATEGORIES)}

    def tokenize(frame, labels=True):
        data = tokenizer(frame.narrative.tolist(), truncation=True, max_length=256)
        if labels:
            data["labels"] = frame.true_label.map(label_map).tolist()
        return Dataset.from_dict(data)

    train = tokenize(pd.read_csv(ROOT / f"dataset/finetuning/train_{args.train_size}.csv"))
    val = tokenize(pd.read_csv(ROOT / "dataset/finetuning/validation.csv"))
    budget_tag = f'n{args.train_size}' + (f'_{args.run_tag}' if args.run_tag else '')
    out = ROOT / f"checkpoints/distilbert/{budget_tag}/seed{args.seed}"
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    destination = ROOT / f"results/sadhana/distilbert/{budget_tag}/seed{args.seed}"
    if (destination / 'training_report.json').exists():
        raise FileExistsError(f'Already completed: {destination}')
    config_path = out / 'run_config.json'
    run_config = {'train_size': args.train_size, 'seed': args.seed,
                  'epochs': args.epochs, 'hf_model_id': model_id,
                  'hf_revision': revision, 'split_sha256': manifest['sha256']}
    if args.run_tag or args.early_stopping_patience:
        run_config.update({'run_tag': args.run_tag, 'early_stopping_patience': args.early_stopping_patience})
    if config_path.exists() and json.loads(config_path.read_text()) != run_config:
        raise ValueError('Incompatible encoder resume configuration')
    atomic_json(config_path, run_config)
    progress = ProgressCallback(out / 'run_status.json')
    fit_report_path = out / 'fit_complete.json'
    fitted = json.loads(fit_report_path.read_text()) if fit_report_path.exists() else None
    start = time.perf_counter()
    model = AutoModelForSequenceClassification.from_pretrained(str(out / 'best') if fitted else model_id,
        **({} if fitted else {'revision': revision}),
        num_labels=5, id2label=dict(enumerate(CATEGORIES)), label2id=label_map)

    def score(result):
        return metrics(matrix([CATEGORIES[i] for i in result.label_ids],
            [CATEGORIES[i] for i in np.argmax(result.predictions, axis=-1)]))

    training = TrainingArguments(output_dir=str(out), per_device_train_batch_size=16,
        per_device_eval_batch_size=32, learning_rate=2e-5, num_train_epochs=args.epochs,
        weight_decay=.01, warmup_steps=math.ceil(math.ceil(len(train)/16)*args.epochs*.1),
        bf16=torch.cuda.is_available(),
        eval_strategy="epoch", save_strategy="epoch", save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="f1_weighted",
        logging_steps=100, seed=args.seed, data_seed=args.seed, report_to="none", disable_tqdm=True)
    trainer = Trainer(model=model, args=training, train_dataset=train,
        eval_dataset=val, processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=score,
        callbacks=[progress] + ([EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)]
                               if args.early_stopping_patience else []))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    if fitted is None:
        from transformers.trainer_utils import get_last_checkpoint
        checkpoint = get_last_checkpoint(str(out)) if args.resume else None
        result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(str(out / 'best'))
        fitted = {'fit_seconds': time.perf_counter()-start,
                  'peak_allocated_vram_gb': torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else None,
                  'train_metrics': result.metrics, 'selected_checkpoint': trainer.state.best_model_checkpoint}
        atomic_json(fit_report_path, fitted)
    fit_seconds, peak_vram = fitted['fit_seconds'], fitted['peak_allocated_vram_gb']
    destination.mkdir(parents=True, exist_ok=True)
    summaries = []
    model.eval()
    for number, path in enumerate(sorted((ROOT / "dataset").glob("*subset*.csv")), 1):
        frame = pd.read_csv(path)
        atomic_json(out / 'run_status.json', {'phase': 'evaluation', 'dataset': path.stem,
                    'done_datasets': number-1, 'total_datasets': 12, 'updated_unix': time.time()})
        prediction_path = destination / f'preds_distilbert_{path.stem}.csv'
        saved_metrics = destination / f'metrics_{path.stem}.json'
        if args.resume and prediction_path.exists() and saved_metrics.exists():
            old = pd.read_csv(prediction_path)
            if (old['index'].tolist() != list(range(len(frame))) or
                    old.true_label.tolist() != frame.true_label.tolist() or
                    old.source_hash.tolist() != [hashlib.sha256(t.encode()).hexdigest() for t in frame.narrative]):
                raise ValueError(f'Invalid saved encoder predictions: {prediction_path}')
            summaries.append(json.loads(saved_metrics.read_text()))
            continue
        # Batched predictions for scores; latency is measured separately, serially.
        prediction = trainer.predict(tokenize(frame)).predictions.argmax(axis=-1)
        preds = [CATEGORIES[i] for i in prediction]
        with torch.inference_mode():
            warmup = tokenizer(frame.narrative.iloc[0], return_tensors="pt").to(model.device)
            model(**warmup)
            elapsed = []
            for text in frame.narrative:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                begin = time.perf_counter()
                inputs = tokenizer(text, truncation=True, max_length=256, return_tensors="pt").to(model.device)
                model(**inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                elapsed.append(time.perf_counter()-begin)
        pd.DataFrame({"index": range(len(frame)), "true_label": frame.true_label,
            "predicted_label": preds, "status": "ok", "latency_seconds": elapsed,
            "source_hash": [hashlib.sha256(t.encode()).hexdigest() for t in frame.narrative]}).to_csv(
                prediction_path.with_suffix('.tmp'), index=False)
        prediction_path.with_suffix('.tmp').replace(prediction_path)
        summaries.append({"test_dataset": path.stem, **metrics(matrix(frame.true_label, preds)),
            "mean_serial_latency_s": float(np.mean(elapsed)), "training_seconds": fit_seconds,
            "off_label": 0, "peak_allocated_vram_gb": peak_vram})
        atomic_json(saved_metrics, summaries[-1])
    pd.DataFrame(summaries).to_csv(destination / "metrics.csv", index=False)
    (destination / "training_report.json").write_text(json.dumps({**vars(args),
        "hf_model_id": model_id, "hf_revision": revision,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "fit_seconds": fit_seconds, "train_metrics": fitted['train_metrics'],
        "selected_checkpoint": fitted['selected_checkpoint'],
        "selection": "best validation weighted F1", "latency": "warm serial tokenization + forward pass with CUDA sync",
        "training_manifest": manifest,
        "prediction_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.glob('preds_*.csv')}}, indent=2))
    atomic_json(out / 'run_status.json', {'phase': 'complete', 'done_datasets': 12,
                'total_datasets': 12, 'updated_unix': time.time()})


if __name__ == "__main__":
    main()
