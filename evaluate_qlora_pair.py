"""Evaluate one pinned NF4 model with LoRA disabled/enabled, with resumable rows."""
import argparse
from contextlib import nullcontext
from datetime import datetime
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

import pandas as pd
import torch
from peft import PeftModel
from transformers import BitsAndBytesConfig

from sadhana_audit import ROOT, CATEGORIES, matrix, metrics, strict_label, text_hash
from train_qlora import MODELS, END_TOKENS, load_frontend, model_loader, prompt_ids, prepare_quantized_model


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_journal(path, frame):
    rows = {}
    if not path.exists():
        return rows
    good_end = 0
    with path.open('rb') as stream:
        while line := stream.readline():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if stream.read():
                    raise ValueError(f'Malformed interior journal line: {path}')
                break
            idx = row['index']
            if idx in rows or idx < 0 or idx >= len(frame):
                raise ValueError(f'Duplicate/unexpected index in {path}: {idx}')
            source = frame.iloc[idx]
            raw_hash = hashlib.sha256(source.narrative.encode()).hexdigest()
            if row['source_hash'] != raw_hash or row['true_label'] != source.true_label:
                raise ValueError(f'Source mismatch in {path}: {idx}')
            rows[idx] = row
            good_end = stream.tell()
    # Only an interrupted final JSON record can be discarded.
    if path.stat().st_size != good_end:
        with path.open('r+b') as stream:
            stream.truncate(good_end)
    return rows


def save_csv(path, rows):
    frame = pd.DataFrame([rows[i] for i in sorted(rows)])
    temp = path.with_suffix('.tmp')
    frame.to_csv(temp, index=False)
    temp.replace(path)
    return frame


def rewrite_journal(path, rows):
    temporary=path.with_suffix('.tmp')
    with temporary.open('w') as stream:
        for idx in sorted(rows):stream.write(json.dumps(rows[idx])+'\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--adapter', type=Path, required=True)
    parser.add_argument('--base-output', type=Path, required=True)
    parser.add_argument('--adapter-output', type=Path, required=True)
    parser.add_argument('--status', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--max-new-tokens', type=int, default=64)
    parser.add_argument('--serial-samples', type=int, default=100)
    parser.add_argument('--limit', type=int, help='Smoke test only; not a full evaluation')
    parser.add_argument('--datasets', nargs='+', help='Optional test stems; default evaluates all 12 files')
    parser.add_argument('--stop-after-datasets', type=int, help='Stage a full-protocol run; later --resume completes the remaining tests')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if (args.batch_size < 1 or args.serial_samples < 0 or (args.limit is not None and args.limit < 1)
            or (args.stop_after_datasets is not None and args.stop_after_datasets < 1)):
        parser.error('Invalid batch/serial/limit count')
    args.adapter = args.adapter.resolve()
    report = json.loads((args.adapter/'training_report.json').read_text())
    name, revision = report['model'], report['hf_revision']
    paths = sorted((ROOT/'dataset').glob('*subset*.csv'),
                   key=lambda p: (not p.stem.endswith('_subset_balanced_500'), p.name))
    if args.datasets:
        unknown=set(args.datasets)-{p.stem for p in paths}
        if unknown:parser.error(f'Unknown datasets: {unknown}')
        paths=[p for p in paths if p.stem in args.datasets]
    test_hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    for key, value in test_hashes.items():
        if value != report['train_manifest']['sha256'][key]:
            raise ValueError(f'Test changed after training: {key}')
    frames = {p.stem: pd.read_csv(p, keep_default_na=False).head(args.limit)
              if args.limit else pd.read_csv(p, keep_default_na=False) for p in paths}
    outputs = {'base': args.base_output.resolve(), 'adapter': args.adapter_output.resolve()}
    protocol = {'model_key': name, 'model': MODELS[name], 'hf_revision': revision,
        'adapter_run': str(args.adapter), 'adapter_sha256': sha(args.adapter/'adapter/adapter_model.safetensors'),
        'adapter_config_sha256': sha(args.adapter/'adapter/adapter_config.json'),
        'test_sha256': test_hashes, 'batch_size': args.batch_size,
        'serial_samples_requested': args.serial_samples, 'max_new_tokens': args.max_new_tokens,
        'thinking': False, 'do_sample': False, 'limit': args.limit,
        'quantization': 'NF4 double quantization, BF16 compute',
        'frozen_parameter_dtype': 'matrices BF16, one-dimensional norms FP32',
        'adapter_compute': 'BF16 autocast; saved adapter parameter dtypes retained',
        'base_mode': 'Same PeftModel instance with adapters disabled; no KV cache shared across calls',
        'latency_seconds': 'Amortized batch wall time per row; prompt rendering/tokenization and CUDA generation included, output decoding excluded',
        'serial_latency': 'Separate stratified serial benchmark on first balanced 500-character test, same timing boundaries',
        'gpu': torch.cuda.get_device_name(0),
        'versions': {p: importlib.metadata.version(p) for p in ['torch','transformers','peft','bitsandbytes']}}
    fingerprint = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    limit_path=args.status.parent/'batch_limit.json'
    batch_limit=min(args.batch_size,json.loads(limit_path.read_text())['max_batch_size']) if limit_path.exists() else args.batch_size
    for mode, out in outputs.items():
        method = out/'evaluation_method.json'
        if out.exists() and any(out.iterdir()):
            if not args.resume:
                raise FileExistsError(f'Use --resume for {out}')
            previous = json.loads(method.read_text())
            if previous['fingerprint'] != fingerprint or previous['mode'] != mode:
                raise ValueError(f'Incompatible evaluation resume: {out}')
        out.mkdir(parents=True, exist_ok=True)
        atomic_json(method, {**protocol, 'fingerprint': fingerprint, 'mode': mode,
            'batch_size_is_maximum':True, 'execution_batch_limit':batch_limit,
            'memory_policy':'Paired batches are halved after CUDA OOM; successful paired rows retained; actual per-row batch size recorded'})
    total = sum(len(f) for f in frames.values())*2
    done = 0
    fresh_rows = 0
    inference_seconds = 0.0
    stage_total = sum(len(f) for f in list(frames.values())[:args.stop_after_datasets])*2 if args.stop_after_datasets else total
    started = time.perf_counter()

    def status(phase, **detail):
        atomic_json(args.status, {'phase': phase, 'model': name, 'done_rows': done,
            'total_rows': total, 'stage_total_rows': stage_total, 'fresh_rows': fresh_rows,
            'effective_batch_size':batch_limit,
            'inference_seconds': inference_seconds, 'elapsed_seconds': time.perf_counter()-started,
            'peak_allocated_vram_gib': torch.cuda.max_memory_allocated()/2**30,
            'updated_local': datetime.now().astimezone().isoformat(), **detail})

    status('loading_model')
    config, tokenizer, frontend = load_frontend(MODELS[name], revision)
    end_id = tokenizer.convert_tokens_to_ids(END_TOKENS[name])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else end_id
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    base = model_loader(config).from_pretrained(MODELS[name], revision=revision,
        quantization_config=quant, device_map={'':0}, dtype=torch.bfloat16,
        attn_implementation='sdpa', local_files_only=True)
    base = prepare_quantized_model(base)
    base.gradient_checkpointing_disable()
    base.config.use_cache = True
    model = PeftModel.from_pretrained(base, str(args.adapter/'adapter'), is_trainable=False)
    model.eval()
    torch.manual_seed(42)

    def generate(mode, selected):
        context = model.disable_adapter() if mode == 'base' else nullcontext()
        with context, torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            torch.cuda.synchronize()
            start = time.perf_counter()
            lists = [prompt_ids(frontend, row.narrative) for _, row in selected]
            width = max(map(len, lists))
            ids = torch.tensor([[pad_id]*(width-len(x))+x for x in lists], device='cuda')
            mask = torch.tensor([[0]*(width-len(x))+[1]*len(x) for x in lists], device='cuda')
            generated = model.generate(input_ids=ids, attention_mask=mask,
                max_new_tokens=args.max_new_tokens, do_sample=False, eos_token_id=end_id,
                pad_token_id=pad_id, logits_to_keep=1)
            torch.cuda.synchronize()
            elapsed = time.perf_counter()-start
        result = []
        for (idx, row), sequence in zip(selected, generated):
            tokens = sequence[width:].tolist()
            if end_id in tokens:
                tokens = tokens[:tokens.index(end_id)+1]
            raw = tokenizer.decode(tokens, skip_special_tokens=True)
            pred = strict_label(raw)
            result.append({'index': int(idx), 'text_hash': text_hash(row.narrative),
                'source_hash': hashlib.sha256(row.narrative.encode()).hexdigest(),
                'true_label': row.true_label, 'predicted_label': pred,
                'status': 'ok' if pred in CATEGORIES else 'parse_error' if pred=='parse_error' else 'off_label',
                'raw_response': raw, 'input_tokens': len(lists[len(result)]),
                'output_tokens': len(tokens), 'batch_size': len(selected),
                'batch_latency_seconds': elapsed, 'latency_seconds': elapsed/len(selected),
                'reached_output_limit': bool(len(tokens)==args.max_new_tokens and tokens[-1]!=end_id)})
        return result

    dummy = [(0, pd.Series({'narrative':'incorrect credit account on my report', 'true_label':'credit_reporting'}))]
    for mode in outputs:
        generate(mode, dummy)
    batch_number = 0
    for dataset_number, (dataset, frame) in enumerate(frames.items(), start=1):
        journals = {mode: out/f'preds_{dataset}.jsonl' for mode, out in outputs.items()}
        saved = {mode: read_journal(path, frame) for mode, path in journals.items()}
        matched=set(saved['base']) & set(saved['adapter'])
        # An interruption between journal writes can leave one-sided rows.
        # Repeat only those rows so base and adapter use the same batch context.
        for mode in outputs:
            if set(saved[mode])!=matched:
                saved[mode]={i:r for i,r in saved[mode].items() if i in matched}
                rewrite_journal(journals[mode],saved[mode])
        done += sum(len(rows) for rows in saved.values())
        start=0
        while start<len(frame):
            indices=list(range(start,min(start+batch_limit,len(frame))))
            if all(i in matched for i in indices):
                start+=len(indices)
                continue
            order = ['base','adapter'] if batch_number%2==0 else ['adapter','base']
            batch_number += 1
            selected=[(i,frame.iloc[i]) for i in indices]
            paired_rows={}
            oom_message=None
            try:
                for mode in order:
                    status('inference', dataset=dataset, mode=mode)
                    paired_rows[mode]=generate(mode,selected)
            except torch.OutOfMemoryError as error:
                if len(indices)==1:raise
                oom_message=str(error)
            if oom_message is not None:
                paired_rows.clear()
                batch_limit=max(1,min(batch_limit,len(indices)//2))
                event={'max_batch_size':batch_limit,'dataset':dataset,'first_index':start,
                    'failed_batch_size':len(indices),'updated_local':datetime.now().astimezone().isoformat(),
                    'reason':oom_message}
                atomic_json(limit_path,event)
                with (args.status.parent/'oom_events.jsonl').open('a') as stream:
                    stream.write(json.dumps(event)+'\n')
                if hasattr(base,'_cache'):delattr(base,'_cache')
                gc.collect();torch.cuda.empty_cache()
                status('retrying_smaller_batch',dataset=dataset,mode=mode)
                print(f'{name}: CUDA OOM; retry paired rows from {start} with batch limit {batch_limit}',flush=True)
                for mode,out in outputs.items():
                    method=json.loads((out/'evaluation_method.json').read_text())
                    atomic_json(out/'evaluation_method.json',{**method,'execution_batch_limit':batch_limit})
                continue
            # Commit only after both modes succeed. A failed pair is retried together.
            for mode in order:
                rows=[r for r in paired_rows[mode] if r['index'] not in saved[mode]]
                with journals[mode].open('a') as stream:
                    for row in rows:
                        stream.write(json.dumps(row)+'\n')
                        saved[mode][row['index']] = row
                    stream.flush()
                done += len(rows)
                fresh_rows += len(rows)
                inference_seconds += paired_rows[mode][0]['batch_latency_seconds']
                status('inference', dataset=dataset, mode=mode)
            matched.update(indices)
            start+=len(indices)
            if batch_number%25==0:
                print(f'{name}: {done}/{total} paired rows', flush=True)
        for mode, out in outputs.items():
            assert len(saved[mode])==len(frame)
            table = save_csv(out/f'preds_{dataset}.csv', saved[mode])
            estimate = {'test_dataset': dataset, 'n': len(table),
                **metrics(matrix(table.true_label, table.predicted_label)),
                'mean_amortized_latency_s': float(table.latency_seconds.mean()),
                'invalid': int(table.status.ne('ok').sum()),
                'output_limited': int(table.reached_output_limit.sum()), 'limited': bool(args.limit)}
            atomic_json(out/f'metrics_{dataset}.json', estimate)
            print(name, mode, json.dumps(estimate), flush=True)
        status('dataset_complete', dataset=dataset)
        if args.stop_after_datasets and dataset_number >= args.stop_after_datasets:
            for out in outputs.values():
                estimates=[json.loads(p.read_text()) for p in sorted(out.glob('metrics_*.json'))]
                pd.DataFrame(estimates).to_csv(out/'metrics.csv',index=False)
            status('primary_complete', dataset=dataset)
            return

    # Benchmark independent serial calls on a class-balanced subset; these do not replace full predictions.
    first_name = next(iter(frames))
    first = frames[first_name]
    serial_ids = sorted(i for category in CATEGORIES
        for i in first[first.true_label.eq(category)].index[:args.serial_samples//len(CATEGORIES)])
    for mode, out in outputs.items():
        if serial_ids:
            journal = out/'serial_benchmark.jsonl'
            saved = read_journal(journal, first)
            for idx in serial_ids:
                if idx in saved:
                    continue
                status('serial_benchmark', mode=mode, serial_done=len(saved), serial_total=len(serial_ids))
                row = generate(mode, [(idx, first.iloc[idx])])[0]
                with journal.open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
                    stream.flush()
                saved[idx] = row
            serial = save_csv(out/'serial_benchmark.csv', saved)
            full = pd.read_csv(out/f'preds_{first_name}.csv', keep_default_na=False).set_index('index')
            atomic_json(out/'serial_benchmark_summary.json', {'dataset':first_name,'n':len(serial),
                'selection':'first equal count per class, sorted back to original indices',
                'mean_latency_s':float(serial.latency_seconds.mean()),
                'median_latency_s':float(serial.latency_seconds.median()),
                'p95_latency_s':float(serial.latency_seconds.quantile(.95)),
                'batch_prediction_agreement':float((serial.predicted_label.to_numpy()==full.loc[serial['index'],'predicted_label'].to_numpy()).mean())})
        estimates = [json.loads(p.read_text()) for p in sorted(out.glob('metrics_*.json'))]
        pd.DataFrame(estimates).to_csv(out/'metrics.csv', index=False)
        atomic_json(out/'evaluation_complete.json', {'phase':'complete','model':name,
            'mode':mode,'fingerprint':fingerprint,'n_datasets':len(frames),
            'n_predictions':sum(len(f) for f in frames.values()),'limit':args.limit,
            'prediction_sha256':{p.name:sha(p) for p in sorted(out.glob('preds_*.csv'))},
            'completed_local':datetime.now().astimezone().isoformat()})
    status('complete')


if __name__=='__main__':
    main()
