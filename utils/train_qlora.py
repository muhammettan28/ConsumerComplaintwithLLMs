"""Completion-only QLoRA training on the preserved-test protocol.

python utils/train_qlora.py --model qwen3 --train-size 500 --seed 1
python utils/train_qlora.py --model gemma4 --preflight
Run preflight before downloading weights. Base HF NF4 inference must be measured
separately from historical Ollama inference before attributing changes to LoRA.
"""
import argparse
import importlib.metadata
import json
import math
import time
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoConfig, AutoModelForCausalLM, AutoProcessor,
                          AutoTokenizer, BitsAndBytesConfig, Trainer,
                          TrainingArguments, TrainerCallback, set_seed)

if __package__:
    from .run_inference import build_prompt
else:
    from run_inference import build_prompt
if __package__:
    from .sadhana_audit import ROOT
else:
    from sadhana_audit import ROOT

MODELS = {"gemma4": "google/gemma-4-12B-it", "qwen3": "Qwen/Qwen3-14B",
          "phi4": "microsoft/phi-4"}
END_TOKENS = {"gemma4": "<turn|>", "qwen3": "<|im_end|>", "phi4": "<|im_end|>"}
TARGET_MODULES = {
    "gemma4": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "qwen3": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "phi4": ["qkv_proj", "o_proj", "gate_up_proj", "down_proj"],
}


def prepare_quantized_model(model):
    """Freeze NF4 base weights; keep large frozen matrices in BF16 on 16 GB.

    PEFT's default preparation casts embeddings and heads to FP32 too. Here
    one-dimensional parameters (including norms) use FP32, frozen matrices
    retain BF16, and non-reentrant checkpointing supports frozen inputs.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False
        if parameter.ndim < 2 and parameter.dtype in (torch.float16, torch.bfloat16):
            parameter.data = parameter.data.float()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return model


class ProgressCallback(TrainerCallback):
    def __init__(self, path):
        self.path = Path(path)
        self.latest = {}

    def write(self, state, phase, logs=None):
        if logs:
            self.latest.update({k: v for k, v in logs.items() if isinstance(v, (int, float))})
        status = {"phase": phase, "global_step": state.global_step,
            "max_steps": state.max_steps, "epoch": state.epoch, "latest_metrics": self.latest,
            "allocated_vram_gib": torch.cuda.memory_allocated()/2**30,
            "peak_allocated_vram_gib": torch.cuda.max_memory_allocated()/2**30,
            "updated_unix": time.time()}
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(json.dumps(status, indent=2))
        temporary.replace(self.path)

    def on_train_begin(self, args, state, control, **kwargs):
        self.write(state, "training")

    def on_step_end(self, args, state, control, **kwargs):
        self.write(state, "training")

    def on_log(self, args, state, control, logs=None, **kwargs):
        self.write(state, "training", logs)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        self.write(state, "validation", metrics)

    def on_train_end(self, args, state, control, **kwargs):
        self.write(state, "saving")


def model_loader(config):
    if config.model_type.startswith("gemma4"):
        import transformers
        loader = getattr(transformers, "AutoModelForMultimodalLM", None)
        if loader is None:
            loader = transformers.AutoModelForImageTextToText
        return loader
    return AutoModelForCausalLM


def load_frontend(model_id, revision):
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    if config.model_type.startswith("gemma4"):
        processor = AutoProcessor.from_pretrained(model_id, revision=revision)
        return config, processor.tokenizer, processor
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    return config, tokenizer, tokenizer


def prompt_ids(frontend, text):
    rendered = frontend.apply_chat_template([{"role": "user", "content": build_prompt(text)}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    tokenizer = getattr(frontend, "tokenizer", frontend)
    return tokenizer(rendered, add_special_tokens=False)["input_ids"]


class CompletionCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, rows):
        width = max(len(r["input_ids"]) for r in rows)
        batch = {key: torch.tensor([r[key] + [fill] * (width-len(r[key])) for r in rows])
                for key, fill in [("input_ids", self.pad_id), ("attention_mask", 0), ("labels", -100)]}
        valid = torch.nonzero(batch["labels"] != -100)
        if not len(valid):
            raise ValueError("Batch contains no supervised answer tokens")
        # Project only the tail needed to predict the supervised completion.
        # This is the same causal loss as full logits, without a large prompt x vocab tensor.
        start = max(0, int(valid[:, 1].min()) - 1)
        batch["logits_to_keep"] = width-start
        shifted = torch.nn.functional.pad(batch["labels"], (0, 1), value=-100)[:, 1:]
        batch["shift_labels"] = shifted[:, start:].contiguous()
        return batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--revision", help="HF commit; default resolves current commit and records it")
    parser.add_argument("--train-size", type=int, choices=[500, 2000, 10000, 20000], default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=-1, help="Smoke test only; record separately from full experiments")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted compatible run")
    args = parser.parse_args()
    from huggingface_hub import HfApi
    revision = args.revision or HfApi().model_info(MODELS[args.model]).sha
    config, tokenizer, frontend = load_frontend(MODELS[args.model], revision)
    loader = model_loader(config)
    supported_class = loader._model_mapping[type(config)]
    test_ids = prompt_ids(frontend, "my credit report contains an incorrect account")
    if not test_ids:
        raise ValueError("Empty rendered chat prompt")
    print(json.dumps({"model": MODELS[args.model], "revision": revision, "model_type": config.model_type,
                      "prompt_tokens": len(test_ids), "thinking_requested": False,
                      "loader_class": supported_class.__name__,
                      "target_end_token": END_TOKENS[args.model],
                      "cuda": torch.cuda.is_available()}, indent=2))
    if args.preflight:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires the CUDA GPU in this experiment")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This protocol requires BF16 support")
    set_seed(args.seed)
    out = ROOT / f"checkpoints/{args.model}/n{args.train_size}/seed{args.seed}"
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite {out}")
    if (out / "training_report.json").exists():
        raise FileExistsError(f"Run already completed: {out}")
    out.mkdir(parents=True, exist_ok=True)
    run_config = {**vars(args), "hf_model_id": MODELS[args.model], "hf_revision": revision,
                  "target_modules": TARGET_MODULES[args.model]}
    config_path = out / "run_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        for key in ["model", "hf_revision", "train_size", "seed", "rank", "max_length", "epochs", "max_steps", "target_modules"]:
            if previous[key] != run_config[key]:
                raise ValueError(f"Incompatible resume configuration: {key}")
    config_path.write_text(json.dumps(run_config, indent=2))
    (out / "run_status.json").write_text(json.dumps({"phase": "loading_weights", "updated_unix": time.time()}, indent=2))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode(row):
        prefix = prompt_ids(frontend, row["narrative"])
        answer = json.dumps({"product": row["true_label"]})
        target = tokenizer(answer + END_TOKENS[args.model], add_special_tokens=False)["input_ids"]
        if len(prefix) + len(target) > args.max_length:
            raise ValueError("Example exceeds max length; increase it, do not silently truncate labels")
        ids = prefix + target
        return {"input_ids": ids, "attention_mask": [1]*len(ids), "labels": [-100]*len(prefix)+target}

    def dataset(path):
        frame = pd.read_csv(path)
        return Dataset.from_list([encode(row) for row in frame.to_dict("records")])

    train = dataset(ROOT / f"dataset/finetuning/train_{args.train_size}.csv")
    val = dataset(ROOT / "dataset/finetuning/validation.csv")
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    model = loader.from_pretrained(MODELS[args.model], revision=revision,
        quantization_config=quant, device_map={"": 0}, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    model = prepare_quantized_model(model)
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2*args.rank,
        lora_dropout=.05, bias="none", task_type="CAUSAL_LM",
        target_modules=TARGET_MODULES[args.model]))
    model.print_trainable_parameters()
    training = TrainingArguments(output_dir=str(out), per_device_train_batch_size=1,
        per_device_eval_batch_size=1, gradient_accumulation_steps=16,
        learning_rate=2e-4, num_train_epochs=args.epochs, max_steps=args.max_steps,
        bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, optim="paged_adamw_8bit",
        warmup_steps=math.ceil((args.max_steps if args.max_steps > 0 else math.ceil(len(train)/16)*args.epochs)*.03),
        lr_scheduler_type="cosine", eval_strategy="epoch",
        save_strategy="epoch", save_total_limit=1, load_best_model_at_end=True,
        metric_for_best_model="eval_loss", greater_is_better=False,
        logging_steps=1, seed=args.seed, data_seed=args.seed, report_to="none",
        prediction_loss_only=True, disable_tqdm=True)
    progress = ProgressCallback(out / "run_status.json")
    trainer = Trainer(model=model, args=training, train_dataset=train, eval_dataset=val,
        data_collator=CompletionCollator(tokenizer.pad_token_id), callbacks=[progress])
    from transformers.trainer_utils import get_last_checkpoint
    resume_checkpoint = get_last_checkpoint(str(out)) if args.resume else None
    result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(str(out / "adapter"))
    frontend.save_pretrained(str(out / "adapter"))
    report = {**vars(args), "hf_model_id": MODELS[args.model], "hf_revision": revision,
        "gpu": torch.cuda.get_device_name(0), "cuda_runtime": torch.version.cuda,
        "peak_allocated_vram_gb": torch.cuda.max_memory_allocated()/2**30,
        "load_and_training_seconds": time.perf_counter()-start, "metrics": result.metrics,
        "validation_selection": "minimum completion-only validation loss",
        "selected_checkpoint": trainer.state.best_model_checkpoint,
        "best_validation_loss": trainer.state.best_metric,
        "target_end_token": END_TOKENS[args.model],
        "quantization": "NF4 double quantization, BF16 compute",
        "frozen_matrix_dtype": "BF16; one-dimensional parameters FP32",
        "loss_projection": "completion tail logits with matching shifted labels",
        "target_modules": TARGET_MODULES[args.model],
        "train_manifest": json.loads((ROOT / "dataset/finetuning/manifest.json").read_text()),
        "versions": {p: importlib.metadata.version(p) for p in ["torch", "transformers", "peft", "bitsandbytes", "datasets"]}}
    (out / "training_report.json").write_text(json.dumps(report, indent=2))
    progress.write(trainer.state, "complete", result.metrics)


if __name__ == "__main__":
    main()
