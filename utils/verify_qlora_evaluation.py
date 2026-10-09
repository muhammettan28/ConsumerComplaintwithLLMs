"""Verify completed paired HF predictions against source files and saved reports."""
import hashlib
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

if __package__:
    from .sadhana_audit import ROOT, CATEGORIES, strict_label, text_hash, matrix, metrics
else:
    from sadhana_audit import ROOT, CATEGORIES, strict_label, text_hash, matrix, metrics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_rows(frame, source, expected_indices):
    assert frame['index'].tolist() == expected_indices, 'Missing, repeated or reordered indices'
    selected = source.loc[expected_indices]
    assert frame.true_label.tolist() == selected.true_label.tolist(), 'Labels changed'
    assert frame.source_hash.tolist() == [hashlib.sha256(t.encode()).hexdigest() for t in selected.narrative], 'Source text changed'
    assert frame.text_hash.tolist() == [text_hash(t) for t in selected.narrative], 'Normalized text hashes changed'
    assert frame.predicted_label.tolist() == [strict_label(t) for t in frame.raw_response], 'Raw response parsing mismatch'
    expected = ['ok' if p in CATEGORIES else 'parse_error' if p == 'parse_error' else 'off_label'
                for p in frame.predicted_label]
    assert frame.status.tolist() == expected, 'Output status mismatch'
    assert np.isfinite(frame.latency_seconds).all() and (frame.latency_seconds > 0).all(), 'Invalid timing'
    assert np.allclose(frame.latency_seconds * frame.batch_size, frame.batch_latency_seconds), 'Amortization mismatch'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-dir', type=Path, default=ROOT/'results/sadhana/qlora_comparison')
    parser.add_argument('--base-root', type=Path, default=ROOT/'results/sadhana/hf_base')
    parser.add_argument('--training-seed', type=int, default=1)
    args = parser.parse_args()
    out = args.comparison_dir
    pooled = pd.read_csv(out / 'pooled_comparison.csv')
    assert len(pooled) == 12 and pooled.test_seeds_complete.all()
    assert (pooled.n == 3000).all()
    assert pooled.training_seed.eq(args.training_seed).all()
    checked, differences = [], []
    prediction_count = serial_count = 0
    for name in ['gemma4', 'qwen3', 'phi4']:
        directories = {'base': args.base_root/name,
                       'adapter': ROOT / f'results/sadhana/qlora/{name}_n500_s{args.training_seed}'}
        methods = {mode: json.loads((p / 'evaluation_method.json').read_text()) for mode, p in directories.items()}
        assert methods['base']['fingerprint'] == methods['adapter']['fingerprint']
        protocol = methods['base']
        run = Path(protocol['adapter_run'])
        training = json.loads((run / 'training_report.json').read_text())
        assert training['seed'] == args.training_seed and training['train_size'] == 500
        for rel in ['dataset/finetuning/train_500.csv', 'dataset/finetuning/validation.csv']:
            assert sha(ROOT / rel) == training['train_manifest']['sha256'][rel], f'Training/validation changed: {rel}'
        assert sha(run / 'adapter/adapter_model.safetensors') == protocol['adapter_sha256']
        assert sha(run / 'adapter/adapter_config.json') == protocol['adapter_config_sha256']
        for rel, digest in protocol['test_sha256'].items():
            assert sha(ROOT / rel) == digest, f'Test changed: {rel}'
        reports = {mode: json.loads((p / 'evaluation_complete.json').read_text()) for mode, p in directories.items()}
        groups = {mode: {} for mode in directories}
        for mode, report in reports.items():
            assert report['phase'] == 'complete' and report['n_datasets'] == 12
            assert report['n_predictions'] == 12000 and report['limit'] is None
            assert report['fingerprint'] == protocol['fingerprint'] and report['mode'] == mode
            assert len(report['prediction_sha256']) == 12
        for stem in sorted(Path(p).stem for p in protocol['test_sha256']):
            source = pd.read_csv(ROOT / 'dataset' / f'{stem}.csv', keep_default_na=False)
            seed, _, condition, size = stem.split('_')
            pair = {}
            for mode, directory in directories.items():
                path = directory / f'preds_{stem}.csv'
                assert sha(path) == reports[mode]['prediction_sha256'][path.name], f'Prediction file changed: {path}'
                frame = pd.read_csv(path, keep_default_na=False).sort_values('index')
                assert len(frame) == len(source) == 1000
                check_rows(frame, source, list(range(1000)))
                pair[mode] = frame
                groups[mode].setdefault((condition, int(size)), []).append(frame)
                prediction_count += len(frame)
            assert pair['base'].batch_size.tolist() == pair['adapter'].batch_size.tolist(), 'Unpaired batch sizes'
            checked.append({'model': name, 'test': stem, 'n_per_mode': 1000,
                'actual_batch_sizes': sorted(set(pair['base'].batch_size)),
                **{f'invalid_{mode}': int(frame.status.ne('ok').sum()) for mode, frame in pair.items()}})
        for key in groups['base']:
            selected = pooled[(pooled.model == name) & (pooled.condition == key[0]) & (pooled['size'] == key[1])]
            assert len(selected) == 1
            row = selected.iloc[0]
            for mode in directories:
                frame = pd.concat(groups[mode][key])
                values = metrics(matrix(frame.true_label, frame.predicted_label))
                for score in ['accuracy', 'f1_weighted', 'f1_macro']:
                    assert np.isclose(values[score], row[f'{mode}_{score}'], rtol=0, atol=1e-12), 'Pooled score mismatch'
                for category in CATEGORIES:
                    precision, recall = values[f'precision_{category}'], values[f'recall_{category}']
                    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                    for score, value in [('precision', precision), ('recall', recall), ('f1', f1)]:
                        assert np.isclose(value, row[f'{mode}_{score}_{category}'], rtol=0, atol=1e-12), 'Pooled class score mismatch'
                assert int(frame.status.ne('ok').sum()) == row[f'invalid_{mode}']
        stem = '1_subset_balanced_500'
        source = pd.read_csv(ROOT / 'dataset' / f'{stem}.csv', keep_default_na=False)
        indices = sorted(i for c in CATEGORIES for i in source.index[source.true_label.eq(c)][:20])
        for mode, directory in directories.items():
            frame = pd.read_csv(directory / 'serial_benchmark.csv', keep_default_na=False).sort_values('index')
            check_rows(frame, source, indices)
            assert frame.batch_size.eq(1).all() and len(frame) == 100
            report = json.loads((directory / 'serial_benchmark_summary.json').read_text())
            batched = pd.read_csv(directory / f'preds_{stem}.csv', keep_default_na=False).set_index('index')
            agreement = frame.predicted_label.to_numpy() == batched.loc[indices].predicted_label.to_numpy()
            assert np.isclose(report['batch_prediction_agreement'], agreement.mean())
            for key, value in [('mean_latency_s', frame.latency_seconds.mean()),
                               ('median_latency_s', frame.latency_seconds.median()),
                               ('p95_latency_s', frame.latency_seconds.quantile(.95))]:
                assert np.isclose(report[key], value)
            for idx in frame.loc[~agreement, 'index']:
                serial = frame.set_index('index').loc[idx]; batch = batched.loc[idx]
                differences.append({'model': name, 'mode': mode, 'index': int(idx),
                    'source_hash': serial.source_hash, 'true_label': serial.true_label,
                    'serial_label': serial.predicted_label, 'batch_label': batch.predicted_label,
                    'serial_response': serial.raw_response, 'batch_response': batch.raw_response,
                    'serial_input_tokens': int(serial.input_tokens), 'batch_input_tokens': int(batch.input_tokens)})
            assert frame.input_tokens.tolist() == batched.loc[indices].input_tokens.tolist(), 'Serial prompt length mismatch'
            serial_count += len(frame)
    pd.DataFrame(differences).to_csv(out / 'serial_batch_disagreements.csv', index=False)
    result = {'verified_at': datetime.now().astimezone().isoformat(), 'result': 'pass',
        'training_seed': args.training_seed,
        'n_full_test_predictions': prediction_count, 'n_serial_predictions': serial_count,
        'checks': ['All six completion manifests and prediction SHA256 values', 'Original training/validation, test and adapter SHA256 values',
                   'Source/index/label and normalized text alignment', 'Raw-response reparsing and invalid counts',
                   'Equal paired batch sizes; finite positive and correctly amortized timing', 'Recomputed pooled accuracy/weighted/macro F1',
                   'Serial 20-per-class selection, input token counts, latency summaries and batch agreement', 'Recomputed pooled per-class precision/recall/F1'],
        'comparison_sha256': {name: sha(out / name) for name in
                              ['pooled_comparison.csv', 'per_test_comparison.csv', 'comparison_method.json']},
        'serial_batch_disagreements': len(differences), 'datasets': checked}
    assert prediction_count == 72000 and serial_count == 600
    (out / 'verification.json').write_text(json.dumps(result, indent=2))
    print(f'PASS: {prediction_count:,} full-test + {serial_count} serial predictions; {len(differences)} serial/batch label differences.')


if __name__ == '__main__':
    main()
