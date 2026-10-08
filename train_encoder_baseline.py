"""Fine-tune DistilBERT using the same split as QLoRA, select only on validation.

python train_encoder_baseline.py --train-size 20000 --seed 1
"""
import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments, set_seed)

from sadhana_audit import CATEGORIES, ROOT, matrix, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=3)
    args = parser.parse_args()
    set_seed(args.seed)
    from huggingface_hub import HfApi
    model_id = "distilbert/distilbert-base-uncased"
    revision = HfApi().model_info(model_id).sha
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    label_map = {c: i for i, c in enumerate(CATEGORIES)}

    def tokenize(frame, labels=True):
        data = tokenizer(frame.narrative.tolist(), truncation=True, max_length=256)
        if labels:
            data["labels"] = frame.true_label.map(label_map).tolist()
        return Dataset.from_dict(data)

    train = tokenize(pd.read_csv(ROOT / f"dataset/finetuning/train_{args.train_size}.csv"))
    val = tokenize(pd.read_csv(ROOT / "dataset/finetuning/validation.csv"))
    out = ROOT / f"checkpoints/distilbert/n{args.train_size}/seed{args.seed}"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}")
    start = time.perf_counter()
    model = AutoModelForSequenceClassification.from_pretrained(model_id, revision=revision,
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
        data_collator=DataCollatorWithPadding(tokenizer), compute_metrics=score)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    trainer.save_model(str(out / "best"))
    fit_seconds = time.perf_counter()-start
    peak_vram = torch.cuda.max_memory_allocated()/2**30 if torch.cuda.is_available() else None
    destination = ROOT / f"results/sadhana/distilbert/n{args.train_size}/seed{args.seed}"
    destination.mkdir(parents=True, exist_ok=True)
    summaries = []
    model.eval()
    for path in sorted((ROOT / "dataset").glob("*subset*.csv")):
        frame = pd.read_csv(path)
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
            "predicted_label": preds, "status": "ok", "latency_seconds": elapsed}).to_csv(
                destination / f"preds_distilbert_{path.stem}.csv", index=False)
        summaries.append({"test_dataset": path.stem, **metrics(matrix(frame.true_label, preds)),
            "mean_serial_latency_s": float(np.mean(elapsed)), "training_seconds": fit_seconds,
            "off_label": 0, "peak_allocated_vram_gb": peak_vram})
    pd.DataFrame(summaries).to_csv(destination / "metrics.csv", index=False)
    (destination / "training_report.json").write_text(json.dumps({**vars(args),
        "hf_model_id": model_id, "hf_revision": revision,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "fit_seconds": fit_seconds, "train_metrics": result.metrics,
        "selection": "best validation weighted F1", "latency": "warm serial tokenization + forward pass with CUDA sync",
        "training_manifest": json.loads((ROOT / "dataset/finetuning/manifest.json").read_text())}, indent=2))


if __name__ == "__main__":
    main()
