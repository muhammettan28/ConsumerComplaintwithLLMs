"""Verify new fits and report training-seed variability without pooling model fits."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

from sadhana_audit import ROOT, CATEGORIES
from summarize_qlora_comparison import measures, paired_bootstrap

RUN = ROOT/'results/sadhana/replication_20261008'
NAMES = ['gemma4', 'qwen3', 'phi4']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paths_for(name, seed):
    base = ROOT/f'results/sadhana/hf_base/{name}' if seed == 1 else ROOT/f'results/sadhana/hf_base_repeats/n500/seed{seed}/{name}'
    return base, ROOT/f'results/sadhana/qlora/{name}_n500_s{seed}'


def source_checked(path, expected_sha=None):
    if expected_sha is not None:
        assert sha(path) == expected_sha, f'Changed predictions: {path}'
    stem = path.stem
    test = stem[stem.index('1_subset'):] if '1_subset' in stem else stem[stem.index('2_subset'):] if '2_subset' in stem else stem[stem.index('3_subset'):]
    source = pd.read_csv(ROOT/'dataset'/f'{test}.csv', keep_default_na=False)
    frame = pd.read_csv(path, keep_default_na=False).sort_values('index')
    assert len(frame) == len(source) == 1000 and frame['index'].tolist() == list(range(1000))
    assert frame.true_label.tolist() == source.true_label.tolist()
    if 'source_hash' in frame:
        assert frame.source_hash.tolist() == [hashlib.sha256(t.encode()).hexdigest() for t in source.narrative]
    return frame


def pool(directory, prefix, condition, size, hashes=None):
    parts = []
    for subset in [1, 2, 3]:
        path = directory/f'{prefix}{subset}_subset_{condition}_{size}.csv'
        frame = source_checked(path, hashes[path.name] if hashes else None)
        frame['test_subset'] = subset
        parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def mean_std(frame, by, columns):
    rows = []
    for key, group in frame.groupby(by, sort=False):
        if not isinstance(key, tuple):
            key = (key,)
        row = dict(zip(by, key))
        row['training_seeds_available'] = int(group.training_seed.nunique())
        row['test_n_per_fit'] = 3000
        for column in columns:
            row[column+'_mean'] = float(group[column].mean())
            row[column+'_std'] = float(group[column].std(ddof=1)) if len(group)>1 else None
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--bootstrap', type=int, default=5000)
    args = parser.parse_args()
    if not args.allow_partial and args.bootstrap < 100:
        parser.error('Full reports require at least 100 bootstrap replicates')
    RUN.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT/'dataset/finetuning/manifest.json').read_text())
    for path, digest in manifest['sha256'].items():
        assert sha(ROOT/path) == digest, f'Data changed: {path}'
    encoder_rows, encoder_predictions = [], {}
    for variant, folder in [('three_epochs','n500'), ('convergence','n500_convergence')]:
        for seed in [1,2,3]:
            directory = ROOT/f'results/sadhana/distilbert/{folder}/seed{seed}'
            if not (directory/'training_report.json').exists():
                if args.allow_partial:
                    continue
                raise FileNotFoundError(directory/'training_report.json')
            report = json.loads((directory/'training_report.json').read_text())
            assert report['train_size'] == 500 and report['seed'] == seed
            assert report['training_manifest']['sha256'] == manifest['sha256']
            assert len(report['prediction_sha256']) == 12
            actual = pd.read_csv(directory/'metrics.csv').set_index('test_dataset')
            for file in directory.glob('preds_*.csv'):
                single = source_checked(file, report['prediction_sha256'][file.name])
                assert single.predicted_label.isin(CATEGORIES).all()
                test = file.stem.replace('preds_distilbert_', '')
                for score, value in measures(single.true_label, single.predicted_label).items():
                    if score in actual:
                        assert np.isclose(actual.loc[test,score], value, rtol=0, atol=1e-12)
            for condition in ['balanced','imbalanced']:
                for size in [250,500]:
                    frame = pool(directory,'preds_distilbert_',condition,size,report['prediction_sha256'])
                    encoder_predictions[(variant,seed,condition,size)] = frame
                    encoder_rows.append({'variant':variant,'training_seed':seed,'train_size':500,
                        'validation_size':2000,'condition':condition,'size':size,'test_n':3000,
                        **measures(frame.true_label,frame.predicted_label),
                        'serial_latency_s':float(frame.latency_seconds.mean()),
                        'fit_seconds':report['fit_seconds'], 'epochs_run':report['train_metrics']['epoch'],
                        'selected_checkpoint':report['selected_checkpoint'], 'hf_revision':report['hf_revision']})
    encoder = pd.DataFrame(encoder_rows)
    encoder.to_csv(RUN/'encoder_per_fit.csv',index=False)
    if len(encoder):
        encoder_summary = mean_std(encoder,['variant','condition','size'],
            ['f1_weighted','f1_macro','accuracy','serial_latency_s','fit_seconds'])
        encoder_summary.to_csv(RUN/'encoder_summary.csv',index=False)
    fits = []
    for seed in [1,2,3]:
        directory = ROOT/'results/sadhana/qlora_comparison' if seed==1 else RUN/f'seed{seed}_comparison'
        if not (directory/'verification.json').exists():
            if args.allow_partial:
                continue
            raise FileNotFoundError(directory/'verification.json')
        verification = json.loads((directory/'verification.json').read_text())
        assert verification['result']=='pass'
        # Revalidate the original pilot once to anchor its exported tables too.
        if 'comparison_sha256' not in verification:
            base_root = ROOT/'results/sadhana/hf_base' if seed==1 else ROOT/f'results/sadhana/hf_base_repeats/n500/seed{seed}'
            subprocess.run([sys.executable,str(ROOT/'verify_qlora_evaluation.py'),
                            '--training-seed',str(seed),'--base-root',str(base_root),
                            '--comparison-dir',str(directory)],check=True)
            verification = json.loads((directory/'verification.json').read_text())
        for name, digest in verification['comparison_sha256'].items():
            assert sha(directory/name)==digest, f'Changed verified comparison: {directory/name}'
        table = pd.read_csv(directory/'pooled_comparison.csv')
        assert len(table)==12 and table.training_seed.eq(seed).all()
        fits.append(table)
    allfits = pd.concat(fits,ignore_index=True) if fits else pd.DataFrame()
    baseline_tests = []
    if len(allfits):
        allfits['mcnemar_p_holm_across_training_seeds'] = multipletests(allfits.mcnemar_p,method='holm')[1]
        allfits.to_csv(RUN/'qlora_per_fit.csv',index=False)
        columns = [f'{mode}_{score}' for mode in ['base','adapter','delta'] for score in ['f1_weighted','f1_macro','accuracy']]
        columns += ['serial_latency_s_base','serial_latency_s_adapter']
        mean_std(allfits,['model','condition','size'],columns).to_csv(RUN/'qlora_seed_summary.csv',index=False)
        if not args.allow_partial:
            assert len(allfits)==36 and allfits.test_seeds_complete.all()
            for row in allfits.itertuples():
                base, adapted = paths_for(row.model,int(row.training_seed))
                completion = json.loads((adapted/'evaluation_complete.json').read_text())
                predictions = pool(adapted,'preds_',row.condition,int(row.size),completion['prediction_sha256'])
                for baseline in ['tfidf_svm','distilbert_convergence']:
                    other = (pool(ROOT/'results/sadhana/classical','preds_tfidf_svc_n500_',row.condition,int(row.size))
                             if baseline=='tfidf_svm' else encoder_predictions[('convergence',int(row.training_seed),row.condition,int(row.size))])
                    assert predictions[['test_subset','index','true_label']].equals(other[['test_subset','index','true_label']])
                    group = predictions[['test_subset','index','source_hash','text_hash','true_label']].copy()
                    group.rename(columns={'test_subset':'seed'},inplace=True)
                    group['predicted_label_base'] = other.predicted_label.to_numpy()
                    group['predicted_label_adapter'] = predictions.predicted_label.to_numpy()
                    stats = paired_bootstrap(group,args.bootstrap)
                    consistent = group.groupby('source_hash').true_label.nunique().eq(1)
                    selected = group[group.source_hash.isin(consistent[consistent].index)].sort_values(['seed','index']).drop_duplicates('source_hash')
                    correct_other = selected.predicted_label_base.eq(selected.true_label)
                    correct_llm = selected.predicted_label_adapter.eq(selected.true_label)
                    only_llm = int((correct_llm & ~correct_other).sum())
                    only_other = int((correct_other & ~correct_llm).sum())
                    a,b = measures(group.true_label,group.predicted_label_base),measures(group.true_label,group.predicted_label_adapter)
                    baseline_tests.append({'model':row.model,'training_seed':int(row.training_seed),
                        'baseline':baseline,'condition':row.condition,'size':int(row.size),'test_n':3000,
                        'baseline_f1_weighted':a['f1_weighted'],'adapter_f1_weighted':b['f1_weighted'],
                        'delta_f1_weighted':b['f1_weighted']-a['f1_weighted'],
                        'delta_f1_weighted_ci_low':stats['delta_f1_weighted_ci_low'],
                        'delta_f1_weighted_ci_high':stats['delta_f1_weighted_ci_high'],
                        'only_adapter_correct':only_llm,'only_baseline_correct':only_other,
                        'mcnemar_p':binomtest(only_llm,only_llm+only_other,.5).pvalue if only_llm+only_other else 1.0})
            tests = pd.DataFrame(baseline_tests)
            assert len(tests)==72
            tests['mcnemar_p_holm'] = multipletests(tests.mcnemar_p,method='holm')[1]
            tests.to_csv(RUN/'label_matched_comparisons.csv',index=False)
    complete = len(encoder)==24 and len(allfits)==36 and len(baseline_tests)==72
    method = {'phase':'complete' if complete else 'partial','updated_local':datetime.now().astimezone().isoformat(),
        'encoder_fits_complete':len(encoder)//4,'qlora_fits_complete':len(allfits)//4,
        'seed_aggregation':'Compute metrics separately for each fit on the same 3,000 records; report mean and sample std over 3 training fits. Never pool fits into 9,000 independent records.',
        'mcnemar_families':'36 base/adapter comparisons across training seeds; 72 separate label-matched classifier comparisons; Holm within each family',
        'bootstrap':f'{args.bootstrap:,} paired normalized-text-cluster draws for label-matched comparisons; base/adapter intervals retained from verified per-fit summaries; training variability reported separately',
        'encoder_selection':'3 epochs retained; additional max-30-epoch/patience-5 control selected by validation only',
        'label_budget':'All n500 methods use the same 500 training and 2,000 validation records',
        'temporal_limit':'Processed CSV has no timestamps or original complaint IDs; this is not temporal validation'}
    (RUN/'replication_method.json').write_text(json.dumps(method,indent=2))
    report = ['# Reviewer-requested replications', '', f"Status: **{method['phase']}**. Updated: {method['updated_local']}.", '',
              '500 fixed training records + the same 2,000 validation records; pinned HF weights; original tests unchanged.', '',
              'Each fit is evaluated on the same 3,000 records. Means and sample standard deviations describe training seeds, not 9,000 independent test records.', '']
    if len(encoder):
        chosen = encoder_summary[(encoder_summary.condition=='balanced')&(encoder_summary['size']==500)]
        report += ['| Encoder protocol | Fits | Weighted F1 mean | Training-seed std |', '|---|---:|---:|---:|']
        for row in chosen.itertuples():
            std = f'{row.f1_weighted_std:.4f}' if pd.notna(row.f1_weighted_std) else '--'
            report.append(f'| {row.variant} | {row.training_seeds_available} | {row.f1_weighted_mean:.4f} | {std} |')
        report += ['']
    if len(allfits):
        means = pd.read_csv(RUN/'qlora_seed_summary.csv')
        report += ['| LLM, balanced 500-character test | Fits | Adapted F1 mean | Training-seed std | Gain (pp) |', '|---|---:|---:|---:|---:|']
        for row in means[(means.condition=='balanced')&(means['size']==500)].itertuples():
            std = f'{row.adapter_f1_weighted_std:.4f}' if pd.notna(row.adapter_f1_weighted_std) else '--'
            report.append(f'| {row.model} | {row.training_seeds_available} | {row.adapter_f1_weighted_mean:.4f} | {std} | {100*row.delta_f1_weighted_mean:.2f} |')
    report += ['', 'Temporal generalisation is not tested: the local processed CSV contains no submission dates or original CFPB IDs.',
               'Full per-fit outputs, seed summaries, verification logs and baseline comparisons are retained in this directory.']
    (RUN/'report.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(method,indent=2),flush=True)
    if not args.allow_partial and not complete:
        raise SystemExit('Incomplete replication study')
    if complete and not args.allow_partial:
        subprocess.run([sys.executable, str(ROOT/'export_replication_manuscript.py')], cwd=ROOT, check=True)


if __name__=='__main__':
    main()
