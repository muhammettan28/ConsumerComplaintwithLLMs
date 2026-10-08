"""Generate comparable base/adapter predictions in the same HF NF4 runtime.

python evaluate_qlora.py --model qwen3 --revision COMMIT --output results/sadhana/hf_base/qwen3
python evaluate_qlora.py --adapter checkpoints/qwen3/n500/seed1 --output results/sadhana/qlora/qwen3_n500_s1
Historical Ollama scores remain useful context; base vs adapter in this runtime
is the controlled adaptation comparison. Timing includes serial generation.
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig

from sadhana_audit import ROOT, CATEGORIES, matrix, metrics, strict_label, text_hash
from train_qlora import MODELS, END_TOKENS, load_frontend, model_loader, prompt_ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", choices=MODELS)
    source.add_argument("--adapter", type=Path, help="Run directory containing training_report.json and adapter/")
    parser.add_argument("--revision", help="Required immutable HF commit for base inference")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="Pilot only; label output as a pilot")
    args = parser.parse_args()
    if args.adapter:
        report = json.loads((args.adapter / "training_report.json").read_text())
        model_name, revision = report["model"], report["hf_revision"]
    else:
        model_name, revision = args.model, args.revision
        if not revision:
            parser.error("Base inference requires --revision matching the adapter training report")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by this protocol")
    config, tokenizer, frontend = load_frontend(MODELS[model_name], revision)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = model_loader(config).from_pretrained(MODELS[model_name], revision=revision,
        quantization_config=quant, device_map={"": 0}, dtype=torch.bfloat16,
        attn_implementation="sdpa")
    if args.adapter:
        model = PeftModel.from_pretrained(model, str(args.adapter / "adapter"))
    model.eval()
    model.config.use_cache = True
    end_id = tokenizer.convert_tokens_to_ids(END_TOKENS[model_name])
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    with torch.inference_mode():
        warmup = torch.tensor([prompt_ids(frontend, "incorrect credit account on report")], device="cuda")
        model.generate(input_ids=warmup, max_new_tokens=16, do_sample=False, eos_token_id=end_id,
                       pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        for path in sorted((ROOT / "dataset").glob("*subset*.csv")):
            frame = pd.read_csv(path)
            if args.limit:
                frame = frame.head(args.limit)
            rows = []
            for idx, row in frame.iterrows():
                torch.cuda.synchronize()
                start = time.perf_counter()
                ids = torch.tensor([prompt_ids(frontend, row.narrative)], device="cuda")
                result = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                    max_new_tokens=64, do_sample=False, eos_token_id=end_id,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
                torch.cuda.synchronize()
                elapsed = time.perf_counter()-start
                generated = result[0, ids.shape[-1]:]
                raw = tokenizer.decode(generated, skip_special_tokens=True)
                pred = strict_label(raw)
                status = "ok" if pred in CATEGORIES else "parse_error" if pred == "parse_error" else "off_label"
                rows.append({"index": idx, "text_hash": text_hash(row.narrative),
                    "true_label": row.true_label, "predicted_label": pred, "status": status,
                    "latency_seconds": elapsed, "raw_response": raw, "output_tokens": len(generated),
                    "reached_output_limit": bool(len(generated) == 64 and int(generated[-1]) != end_id)})
            df = pd.DataFrame(rows)
            df.to_csv(args.output / f"preds_{path.stem}.csv", index=False)
            summaries.append({"test_dataset": path.stem, "n": len(df),
                **metrics(matrix(df.true_label, df.predicted_label)),
                "mean_latency_s": float(df.latency_seconds.mean()),
                "invalid": int(df.status.ne("ok").sum()), "limited": bool(args.limit)})
            print(path.stem, summaries[-1], flush=True)
    pd.DataFrame(summaries).to_csv(args.output / "metrics.csv", index=False)
    (args.output / "evaluation_method.json").write_text(json.dumps({
        "model": MODELS[model_name], "hf_revision": revision,
        "adapter": str(args.adapter) if args.adapter else None,
        "thinking": False, "quantization": "NF4 double quantization, BF16",
        "max_new_tokens": 64, "do_sample": False,
        "gpu": torch.cuda.get_device_name(0), "limit": args.limit,
        "latency": "warm serial prompt rendering/tokenization + CUDA generation; output decoding excluded"}, indent=2))


if __name__ == "__main__":
    main()
