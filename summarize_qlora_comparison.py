"""Paired HF base/QLoRA metrics, cluster-bootstrap changes and unique-input McNemar."""
import argparse
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

from sadhana_audit import ROOT, CATEGORIES, matrix, metrics

PATTERN = re.compile(r'preds_([123])_subset_(balanced|imbalanced)_(250|500)\.csv$')
SCORES = ['accuracy','f1_weighted','f1_macro']


def measures(true, pred):
    values=metrics(matrix(true,pred))
    for category in CATEGORIES:
        p,r=values[f'precision_{category}'],values[f'recall_{category}']
        values[f'f1_{category}']=2*p*r/(p+r) if p+r else 0.0
    return values


def paired_bootstrap(group, replicates):
    clusters, codes = np.unique(group.text_hash, return_inverse=True)
    label_ids = {c:i for i,c in enumerate(CATEGORIES)}
    cells = []
    for mode in ['base','adapter']:
        cells.append(np.array([label_ids[t]*6+label_ids.get(p,5)
            for t,p in zip(group.true_label, group[f'predicted_label_{mode}'])]))
    cm = np.zeros((len(clusters),60))
    np.add.at(cm,(codes,cells[0]),1)
    np.add.at(cm,(codes,cells[1]+30),1)
    labels = group.groupby('text_hash').true_label
    stratified = bool((labels.nunique()==1).all())
    if stratified:
        ordered = labels.first().reindex(clusters).to_numpy()
        strata = [np.flatnonzero(ordered==c) for c in CATEGORIES]
    else:
        strata = [np.arange(len(clusters))]
    rng = np.random.default_rng(42)
    draws = {f'{mode}_{score}':[] for mode in ['base','adapter','delta'] for score in SCORES}
    for _ in range(replicates):
        weights = np.zeros(len(clusters),dtype=int)
        for ids in strata:
            if len(ids): weights[ids]=rng.multinomial(len(ids),np.full(len(ids),1/len(ids)))
        combined = weights@cm
        a,b = metrics(combined[:30].reshape(5,6)),metrics(combined[30:].reshape(5,6))
        for score in SCORES:
            for mode,v in [('base',a[score]),('adapter',b[score]),('delta',b[score]-a[score])]:
                draws[f'{mode}_{score}'].append(v)
    return {**{f'{key}_ci_{side}':float(value) for key,values in draws.items()
        for side,value in zip(['low','high'],np.quantile(values,[.025,.975]))},
        'bootstrap_clusters':len(clusters),'bootstrap_class_stratified':stratified}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bootstrap',type=int,default=5000)
    parser.add_argument('--output',type=Path,default=ROOT/'results/sadhana/qlora_comparison')
    parser.add_argument('--training-seed',type=int,default=1)
    parser.add_argument('--base-root',type=Path,default=ROOT/'results/sadhana/hf_base')
    args=parser.parse_args()
    if args.bootstrap and args.bootstrap<100:parser.error('Use 0 or at least 100 replicates')
    args.output.mkdir(parents=True,exist_ok=True)
    per_test,pooled=[],[]
    for name in ['gemma4','qwen3','phi4']:
        base=args.base_root/name
        adapted=ROOT/f'results/sadhana/qlora/{name}_n500_s{args.training_seed}'
        if not (base/'evaluation_method.json').exists() or not (adapted/'evaluation_method.json').exists():continue
        method=json.loads((base/'evaluation_method.json').read_text())
        other=json.loads((adapted/'evaluation_method.json').read_text())
        if method['fingerprint']!=other['fingerprint']:raise ValueError(f'Protocol mismatch: {name}')
        if method['limit'] is not None:raise ValueError('Pilot-limited predictions are not full test results')
        pairs=[]
        for path in sorted(base.glob('preds_*.csv')):
            match=PATTERN.fullmatch(path.name)
            if not match or not (adapted/path.name).exists():continue
            seed,condition,size=match.groups()
            a=pd.read_csv(path,keep_default_na=False);b=pd.read_csv(adapted/path.name,keep_default_na=False)
            if len(a)!=1000 or len(b)!=1000:raise ValueError(f'Incomplete dataset: {path}')
            join=a.merge(b,on=['index','source_hash','text_hash','true_label'],suffixes=('_base','_adapter'),validate='one_to_one')
            if len(join)!=1000:raise ValueError(f'Prediction/source alignment failed: {path}')
            join['seed']=int(seed);join['condition']=condition;join['size']=int(size);pairs.append(join)
            va,vb=measures(join.true_label,join.predicted_label_base),measures(join.true_label,join.predicted_label_adapter)
            per_test.append({'model':name,'seed':int(seed),'training_seed':args.training_seed,'condition':condition,'size':int(size),'n':len(join),
                **{f'{mode}_{key}':value for mode,values in [('base',va),('adapter',vb)] for key,value in values.items()},
                **{f'delta_{key}':vb[key]-va[key] for key in SCORES},
                'invalid_base':int(join.status_base.ne('ok').sum()),'invalid_adapter':int(join.status_adapter.ne('ok').sum())})
        if not pairs:continue
        full=pd.concat(pairs,ignore_index=True)
        for (condition,size),group in full.groupby(['condition','size']):
            va,vb=measures(group.true_label,group.predicted_label_base),measures(group.true_label,group.predicted_label_adapter)
            row={'model':name,'condition':condition,'size':int(size),'train_size':500,'training_seed':args.training_seed,
                'test_seeds_available':int(group.seed.nunique()),'test_seeds_complete':bool(group.seed.nunique()==3),'n':len(group),
                **{f'{mode}_{key}':value for mode,values in [('base',va),('adapter',vb)] for key,value in values.items()},
                **{f'delta_{key}':vb[key]-va[key] for key in SCORES},
                'invalid_base':int(group.status_base.ne('ok').sum()),'invalid_adapter':int(group.status_adapter.ne('ok').sum()),
                'output_limited_base':int(group.reached_output_limit_base.sum()),'output_limited_adapter':int(group.reached_output_limit_adapter.sum()),
                'mean_amortized_latency_s_base':float(group.latency_seconds_base.mean()),
                'mean_amortized_latency_s_adapter':float(group.latency_seconds_adapter.mean())}
            if args.bootstrap and row['test_seeds_complete']:
                row.update(paired_bootstrap(group,args.bootstrap))
                consistent=group.groupby('source_hash').true_label.nunique().eq(1)
                valid=group[group.source_hash.isin(consistent[consistent].index)]
                conflicts=valid.groupby('source_hash')[['predicted_label_base','predicted_label_adapter']].nunique().gt(1).any(axis=1)
                unique=valid.sort_values(['seed','index']).drop_duplicates('source_hash')
                ca=unique.predicted_label_base.eq(unique.true_label);cb=unique.predicted_label_adapter.eq(unique.true_label)
                n01,n10=int((~ca&cb).sum()),int((ca&~cb).sum())
                row.update({'mcnemar_unique_inputs':len(unique),'only_adapter_correct':n01,'only_base_correct':n10,
                    'mcnemar_p':float(binomtest(n01,n01+n10,.5).pvalue) if n01+n10 else 1.0,
                    'duplicate_prediction_conflict_inputs':int(conflicts.sum()),
                    'conflicting_label_inputs_excluded_from_mcnemar':int((~consistent).sum())})
            for mode,out in [('base',base),('adapter',adapted)]:
                bench=out/'serial_benchmark_summary.json'
                if bench.exists():
                    values=json.loads(bench.read_text())
                    row[f'serial_latency_s_{mode}']=values['mean_latency_s']
                    row[f'serial_batch_agreement_{mode}']=values['batch_prediction_agreement']
            pooled.append(row)
    table=pd.DataFrame(pooled)
    if len(table) and 'mcnemar_p' in table:
        chosen=table.mcnemar_p.notna()
        table.loc[chosen,'mcnemar_p_holm']=multipletests(table.loc[chosen,'mcnemar_p'],method='holm')[1]
    for filename,frame in [('per_test_comparison.csv',pd.DataFrame(per_test)),('pooled_comparison.csv',table)]:
        path=args.output/filename;temporary=path.with_suffix('.tmp');frame.to_csv(temporary,index=False);temporary.replace(path)
    method={'training':f'500 balanced records, seed {args.training_seed}, best validation-loss adapter; within-fit uncertainty only',
        'test':'12 preserved files, three test subsets, balanced/imbalanced, 250/500 characters; all invalid outputs retained as errors',
        'estimator':'Pooled confusion-matrix metrics; partial groups explicitly marked',
        'paired_bootstrap':'Same normalized-text-cluster weights for base and adapter; class-stratified if cluster labels consistent, otherwise unstratified',
        'bootstrap_replicates':args.bootstrap,'rng_seed':42,
        'mcnemar':'Exact paired accuracy test after selecting first seed/index for each exact source input; conflicting-label inputs excluded only from this test; Holm across available model/condition/length comparisons',
        'timing':'Full-test amortized batch latency (max32, paired memory fallback, actual batch size recorded per row) and separate stratified serial benchmark; serial benchmark belongs to balanced 500-character seed-1 dataset, not each condition',
        'historical_ollama':'Separate runtime/quantization; not used as the base for adaptation deltas'}
    (args.output/'comparison_method.json').write_text(json.dumps(method,indent=2))
    if args.bootstrap and len(table) and 'base_f1_weighted_ci_low' in table:
        import matplotlib
        matplotlib.use('Agg')
        models=['gemma4','qwen3','phi4']
        complete=table[(table['size']==500)&table.test_seeds_complete]
        conditions=[c for c in ['balanced','imbalanced']
            if set(complete.loc[complete.condition==c,'model'])==set(models)]
        if not conditions:
            print('Figure deferred until a 500-character condition is complete for all three models.',flush=True)
        else:
            make_figure(complete,conditions,args.output)
    if len(table):
        chosen=table[(table.condition=='balanced')&(table['size']==500)]
        print(chosen[['model','n','test_seeds_available','base_f1_weighted','adapter_f1_weighted','delta_f1_weighted','base_accuracy','adapter_accuracy']].to_string(index=False),flush=True)


def make_figure(table,conditions,output):
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,len(conditions),figsize=(4.5*len(conditions),3.7),sharey=True,squeeze=False)
    for axis,condition in zip(axes[0],conditions):
        part=table[table.condition==condition].set_index('model').reindex(['gemma4','qwen3','phi4'])
        x=np.arange(len(part))
        for mode,offset,color,label in [('base',-.19,'#64748b','HF base'),('adapter',.19,'#2563eb','QLoRA (500 training records)')]:
            y=part[f'{mode}_f1_weighted'].to_numpy()
            low=part[f'{mode}_f1_weighted_ci_low'].to_numpy();high=part[f'{mode}_f1_weighted_ci_high'].to_numpy()
            axis.bar(x+offset,y,width=.36,color=color,label=label)
            axis.errorbar(x+offset,y,yerr=np.stack([np.maximum(0,y-low),np.maximum(0,high-y)]),fmt='none',color='#111827',capsize=3,lw=1)
        axis.set_xticks(x,['Gemma 4 12B','Qwen3 14B','Phi-4 14B'],rotation=12)
        axis.set_title('Balanced tests' if condition=='balanced' else 'Reduced-imbalance tests')
        axis.set_ylim(0,1);axis.grid(axis='y',alpha=.2);axis.set_axisbelow(True)
    axes[0,0].set_ylabel('Pooled weighted F1')
    axes[0,0].legend(fontsize=8,loc='lower left')
    figure.suptitle('Same HF NF4 runtime; 500-character tests\nPaired cluster-bootstrap 95% intervals',fontsize=10)
    figure.tight_layout()
    figure.savefig(output/'hf_qlora_f1_comparison.pdf',bbox_inches='tight')
    figure.savefig(output/'hf_qlora_f1_comparison.png',dpi=200,bbox_inches='tight')
    plt.close(figure)

if __name__=='__main__':main()
