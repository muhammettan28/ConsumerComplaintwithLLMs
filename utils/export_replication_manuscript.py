"""Build and audit a complete three-seed manuscript before replacing live files."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np
import pandas as pd

if __package__:
    from .sadhana_audit import ROOT
else:
    from sadhana_audit import ROOT

RUN = ROOT/'results/sadhana/replication_20261008'
TEX = ROOT/'SPRINGER'
MODELS = {'gemma4':'Gemma 4','qwen3':'Qwen3','phi4':'Phi-4'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def estimate(mean, std, digits=4, scale=1):
    return f'${mean*scale:.{digits}f}\\pm{std*scale:.{digits}f}$'


def change(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'Manuscript changed; cannot safely replace: {old[:90]}')
    return text.replace(old,new,1)


def main():
    method = json.loads((RUN/'replication_method.json').read_text())
    assert method['phase']=='complete' and method['encoder_fits_complete']==6 and method['qlora_fits_complete']==9
    result = RUN/'manuscript_export.json'
    if result.exists() and json.loads(result.read_text()).get('phase')=='complete':
        print('Multiseed manuscript was already exported; preserving subsequent edits')
        return
    summary = pd.read_csv(RUN/'qlora_seed_summary.csv')
    fits = pd.read_csv(RUN/'qlora_per_fit.csv')
    enc = pd.read_csv(RUN/'encoder_summary.csv')
    assert len(summary)==12 and summary.training_seeds_available.eq(3).all()
    assert len(fits)==36
    primary = summary[(summary.condition=='balanced')&(summary['size']==500)].set_index('model')
    encoder_primary = enc[(enc.condition=='balanced')&(enc['size']==500)].set_index('variant')
    assert encoder_primary.training_seeds_available.eq(3).all()
    best = primary.adapter_f1_weighted_mean.idxmax()
    lo, hi = primary.delta_f1_weighted_mean.min()*100,primary.delta_f1_weighted_mean.max()*100
    positive = int((fits.delta_f1_weighted_ci_low>0).sum())
    significant = int((fits.mcnemar_p_holm_across_training_seeds<.05).sum())
    invalid = int(fits.invalid_adapter.sum())
    base_invalid = int(fits.invalid_base.sum())
    serial_differences = sum(json.loads((ROOT/'results/sadhana/qlora_comparison/verification.json'
        if seed==1 else RUN/f'seed{seed}_comparison/verification.json').read_text())['serial_batch_disagreements']
        for seed in [1,2,3])
    low_latency, high_latency = primary.serial_latency_s_adapter_mean.min(),primary.serial_latency_s_adapter_mean.max()
    snapshots = {name:(TEX/name).read_text() for name in
                 ['main.tex','qlora_methods.tex','qlora_results.tex','supervised_baselines.tex','supplementary_material.tex']}
    original_hashes = {name:sha(TEX/name) for name in snapshots}
    backup = RUN/'manuscript_before_multiseed';backup.mkdir(exist_ok=True)
    for name in snapshots:
        if not (backup/name).exists():
            shutil.copy2(TEX/name,backup/name)
    generated = {}
    tables = [r'''\begin{table}[tb]
\centering\scriptsize
\caption{Matched HF base/adapted 500-character results. Each fit uses the
same 3,000 test records; entries are mean $\pm$ sample standard deviation
over three training seeds, not test-population confidence intervals.
Training/validation budgets are 500/2,000 records.}
\label{tab:qlora_paired}
\setlength{\tabcolsep}{2pt}
\begin{tabular}{llrrrr}\toprule
Model & Test & Base F1 & Adapted F1 & Gain (pp) & Adapted macro-F1 \\\midrule''']
    for condition,tag in [('balanced','A'),('imbalanced','B')]:
        for name,label in MODELS.items():
            row=summary[(summary.model==name)&(summary.condition==condition)&(summary['size']==500)].iloc[0]
            values=[label,tag,estimate(row.base_f1_weighted_mean,row.base_f1_weighted_std),
                    estimate(row.adapter_f1_weighted_mean,row.adapter_f1_weighted_std),
                    estimate(row.delta_f1_weighted_mean,row.delta_f1_weighted_std,2,100),
                    estimate(row.adapter_f1_macro_mean,row.adapter_f1_macro_std)]
            tables.append(' & '.join(values)+r' \\')
    tables.append(r'\botrule\end{tabular}\end{table}')
    tables.append(r'''\begin{table}[tb]
\centering\scriptsize
\caption{Three-fit training and warm serial timing. Means $\pm$ sample
standard deviations describe three fits. Peak allocated VRAM is the
maximum over fits; training time includes validation but excludes loading.}
\label{tab:qlora_resources}
\begin{tabular}{lrrrr}\toprule
Model & Train/val (min) & Peak (GiB) & Base serial (s) & Adapted serial (s) \\\midrule''')
    resources=[]
    for name,label in MODELS.items():
        reports=[json.loads((ROOT/f'checkpoints/{name}/n500/seed{seed}/training_report.json').read_text()) for seed in [1,2,3]]
        minutes=[r['metrics']['train_runtime']/60 for r in reports]
        memory=max(r['peak_allocated_vram_gb'] for r in reports)
        row=primary.loc[name]
        values=[label,estimate(np.mean(minutes),np.std(minutes,ddof=1),2),f'{memory:.2f}',
                estimate(row.serial_latency_s_base_mean,row.serial_latency_s_base_std,3),
                estimate(row.serial_latency_s_adapter_mean,row.serial_latency_s_adapter_std,3)]
        tables.append(' & '.join(values)+r' \\')
        resources.append({'model':name,'training_minutes_mean':np.mean(minutes),'training_minutes_std':np.std(minutes,ddof=1),'peak_vram_max_gib':memory})
    tables.append(r'\botrule\end{tabular}\end{table}')
    generated['qlora_main_tables.tex']='\n'.join(tables)+'\n'
    pd.DataFrame(resources).to_csv(RUN/'training_resources.csv',index=False)
    supplementary=[r'''\begin{longtable}{llrrrrrr}
\caption{Complete three-training-seed paired results, each pooled over
the same 3,000 test records. A/B denote balanced/reduced imbalance;
the length column is a character limit, not a sample size.}
\label{tab:replication_full}\\
\toprule Model & Test & Length & Fit seed & Base F1 & Adapted F1 & Gain (pp) & Holm $p$ \\\midrule
\endfirsthead
\toprule Model & Test & Length & Fit seed & Base F1 & Adapted F1 & Gain (pp) & Holm $p$ \\\midrule
\endhead''']
    for row in fits.sort_values(['model','condition','size','training_seed']).itertuples():
        supplementary.append(f"{MODELS[row.model]} & {'A' if row.condition=='balanced' else 'B'} & {row.size} & {row.training_seed} & {row.base_f1_weighted:.4f} & {row.adapter_f1_weighted:.4f} & {row.delta_f1_weighted*100:.2f} & {row.mcnemar_p_holm_across_training_seeds:.3g}"+r' \\')
    supplementary.append(r'\bottomrule\end{longtable}')
    generated['replication_supplementary_tables.tex']='\n'.join(supplementary)+'\n'
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(9,3.8),sharey=True)
    for axis,condition in zip(axes,['balanced','imbalanced']):
        selected=summary[(summary.condition==condition)&(summary['size']==500)].set_index('model').reindex(list(MODELS))
        x=np.arange(3)
        for mode,offset,color,label in [('base',-.19,'#64748b','HF base'),('adapter',.19,'#2563eb','QLoRA')]:
            axis.bar(x+offset,selected[f'{mode}_f1_weighted_mean'],width=.36,color=color,label=label,
                     yerr=selected[f'{mode}_f1_weighted_std'],capsize=3)
        axis.set_xticks(x,list(MODELS.values()));axis.set_ylim(0,1)
        axis.set_title('Balanced' if condition=='balanced' else 'Reduced imbalance')
        axis.grid(axis='y',alpha=.2);axis.set_axisbelow(True)
    axes[0].set_ylabel('Weighted F1, mean over 3 fits');axes[0].legend(loc='lower left')
    figure.suptitle('500 training + 2,000 validation records; 500-character tests\nError bars: training-seed standard deviation, not confidence intervals',fontsize=10)
    figure.tight_layout();figure.savefig(RUN/'hf_qlora_multiseed_f1.pdf',bbox_inches='tight');plt.close(figure)
    results=rf'''\subsection{{QLoRA Adaptation with Three Training Seeds}}
\label{{sec:qlora_results}}
Nine adapter fits completed all twelve preserved tests, giving 216,000
paired base/adapted predictions and 1,800 serial benchmark predictions.
All source, label, response, pairing and artifact checks passed.
Each fit is evaluated separately on the same 3,000 records per condition;
Table~\ref{{tab:qlora_paired}} and Fig.~\ref{{fig:qlora_paired}} report
means and sample standard deviations over three training seeds.
Balanced mean weighted-F1 gains range from {lo:.2f} to {hi:.2f} percentage
points. {MODELS[best]} has the highest adapted mean point estimate,
{primary.loc[best,'adapter_f1_weighted_mean']:.4f}; model-to-model superiority
is not established by this ranking.
\input{{qlora_main_tables}}

Across the 36 model--training-seed--condition--length comparisons,
{positive} paired F1-change intervals exclude zero in the positive direction.
{significant} accuracy comparisons pass Holm correction across this family
of 36 tests. These within-fit intervals measure uncertainty conditional
on saved test outputs; between-fit variation is reported separately.
Overlapping conditions and narrative variants are not independent replications.
There are {invalid} invalid adapted predictions among 108,000 outputs,
compared with {base_invalid} invalid base outputs. Invalid responses remain errors.
Supplementary Table~\ref{{S-tab:replication_full}} reports every fit and condition.

\begin{{figure}}[tb]\centering
\includegraphics[width=\textwidth]{{figures/v2/hf_qlora_multiseed_f1.pdf}}
\caption{{Matched HF base/adapted mean weighted F1 over three training
seeds on identical 3,000-record tests. Error bars show sample standard
deviation over fits, not bootstrap confidence intervals.}}
\label{{fig:qlora_paired}}\end{{figure}}

The label-matched classifier comparison uses the same 500 training and
2,000 validation records. The validation-stopped DistilBERT control
has balanced weighted F1 {encoder_primary.loc['convergence','f1_weighted_mean']:.4f}
$\pm$ {encoder_primary.loc['convergence','f1_weighted_std']:.4f}; the fixed
three-epoch control scores {encoder_primary.loc['three_epochs','f1_weighted_mean']:.4f}
$\pm$ {encoder_primary.loc['three_epochs','f1_weighted_std']:.4f}.
TF-IDF/SVM scores 0.7546 at the same training-label budget.
Per-fit adapter--classifier F1 differences use paired cluster bootstrap;
accuracy tests correct a separate family of 72 comparisons covering
three models, three fit seeds, two conditions, two lengths and two classifiers.
The complete differences and adjusted tests are supplied as reproducibility CSVs;
no superiority claim is inferred from point estimates alone.

Table~\ref{{tab:qlora_resources}} reports training and serial timing.
Adapted balanced serial means range from {low_latency:.3f} to
{high_latency:.3f}~s across model averages. These measurements concern
unmerged adapters and a fixed 100-record warm serial benchmark per fit,
not concurrent throughput. Serial/batch differences remain explicitly
recorded by the verifier; full-test F1 uses batched outputs.
'''
    generated['qlora_results.tex']=results
    methods=snapshots['qlora_methods.tex']
    methods=change(methods,'Training uses seed 1, three epochs,', 'Training uses seeds 1, 2 and 3 in separate fits, each with three epochs,')
    methods=change(methods,"All runs complete\n96 optimiser steps, with Gemma's selected checkpoint at epoch 2 and\nQwen/Phi at epoch 3.", 'Each fit completes 96 optimiser steps. Validation loss selects the\ncheckpoint separately for every fit; selected checkpoints are recorded.')
    methods=change(methods,'The full design yields 72,000 predictions: three models, two modes and\ntwelve preserved 1,000-record files.', 'The full design yields 216,000 predictions: three models, three training\nseeds, two modes and twelve preserved 1,000-record files.')
    methods=change(methods,'Holm correction covers all twelve\nmodel--condition--length comparisons.', 'Holm correction covers all 36 model--training-seed--condition--length\ncomparisons. Metrics are calculated per fit, then mean and sample\nstandard deviation are reported over the three fits.')
    methods=change(methods,'Confidence\nintervals describe these test outputs, not variation over training seeds.', 'Bootstrap confidence intervals describe test-output uncertainty within\na fit; training-seed variability is reported separately. Repeated model\nfits are never pooled into 9,000 independent test observations.')
    generated['qlora_methods.tex']=methods
    supervised=snapshots['supervised_baselines.tex']
    extra=rf'''We additionally train DistilBERT on the identical 500-record subset
with three training seeds and the same 2,000-record validation set.
A fixed three-epoch control scores balanced weighted F1
{encoder_primary.loc['three_epochs','f1_weighted_mean']:.4f}$\pm$
{encoder_primary.loc['three_epochs','f1_weighted_std']:.4f}.
Its validation trajectory was still improving, motivating a separate
convergence control before inspecting its test scores. This control
allows at most 30 epochs, uses patience 5 on validation weighted F1,
and selects the best validation checkpoint. All other encoder settings
are retained; original three-epoch outputs are preserved separately.
It scores {encoder_primary.loc['convergence','f1_weighted_mean']:.4f}$\pm$
{encoder_primary.loc['convergence','f1_weighted_std']:.4f} on the same tests.
This matches label budgets, not training-update or compute budgets.

'''
    supervised=change(supervised,'Table~\\ref{tab:supervised} summarises',extra+'Table~\\ref{tab:supervised} summarises')
    newrows=[]
    for variant,label in [('three_epochs','DistilBERT (3 epochs)'),('convergence','DistilBERT (early stop)')]:
        row=encoder_primary.loc[variant]
        newrows.append(f"{label} & 500 & {estimate(row.f1_weighted_mean,row.f1_weighted_std)} & {1000*row.serial_latency_s_mean:.2f}"+r' \\')
    supervised=change(supervised,'DistilBERT (GPU) & 20,000', '\n'.join(newrows)+'\nDistilBERT (GPU) & 20,000')
    generated['supervised_baselines.tex']=supervised
    supp=snapshots['supplementary_material.tex']
    supp=change(supp,r'\usepackage{graphicx,booktabs,amsmath,url,hyperref}',r'\usepackage{graphicx,booktabs,amsmath,url,hyperref,longtable}')
    start=supp.index(r'\section{Controlled HF adaptation:')
    end=supp.index(r'\section{Extended related work}',start)
    section=r'''\section{Controlled HF adaptation: complete results and reproducibility}
Three models use the same 500 balanced training and 2,000 validation records,
with three training seeds each. Nine base/adapted pairs use identical
pinned weights within each fit. Exact revisions are:
\input{qlora_revisions}
All 216,000 test predictions and 1,800 serial predictions pass the same
source, label, hash, response and pairing checks. Mean and sample standard
deviation describe three fits on the same tests, not 9,000 independent
records. Per-fit paired confidence intervals and unique-input McNemar tests
use the original cluster and duplicate handling. Holm correction covers
36 base/adapter comparisons; a separate family of 72 tests compares
adapters with label-matched SVM and validation-stopped DistilBERT.
Serial timing uses the same first 20 records per class from the first
balanced 500-character file; full-test scores use batched predictions.

The initial seed-1 results are retained for traceability in
Table~\ref{tab:qlora_full} (full metrics), Table~\ref{tab:qlora_class}
(class metrics), and Table~\ref{tab:qlora_serial} (timing and agreement).
These three tables describe seed 1, not averages over fits.
\input{qlora_supplementary_tables}
Table~\ref{tab:replication_full} reports all three training seeds.
\begingroup\scriptsize
\input{replication_supplementary_tables}
\endgroup
Full seed summaries, class scores, baseline differences, bootstrap
intervals, adjusted tests and verification logs are supplied as CSV/JSON
reproducibility artifacts in the replication directory.
\clearpage

'''
    generated['supplementary_material.tex']=supp[:start]+section+supp[end:]
    main=snapshots['main.tex']
    abstract=rf'''Consumer complaint classification supports financial-service routing,
but selecting a local model depends on available labels and response-time
requirements. We compare eleven open-weight models in a zero-shot
benchmark and study lightweight adaptation of Gemma~4, Qwen3 and Phi-4
on a five-category Consumer Financial Protection Bureau task. Adaptation
uses 500 balanced training records, a separate 2,000-record validation set
and three training seeds per model. Each fit is evaluated on the same
3,000 balanced test records. Mean weighted-F1 gains range from {lo:.2f}
to {hi:.2f} percentage points; {MODELS[best]} reaches the highest adapted
mean, {primary.loc[best,'adapter_f1_weighted_mean']:.4f}. A label-matched
support vector machine scores 0.7546, while a validation-stopped
DistilBERT control averages {encoder_primary.loc['convergence','f1_weighted_mean']:.4f}.
At 20,000 training records, these direct classifiers reach 0.8291 and
0.8244, respectively. Adaptation does not consistently reduce latency.
Separate reporting of test uncertainty and training-seed variability
supports conditional deployment choices, rather than a universal model
ranking. Findings remain limited to the preserved short-narrative corpus;
temporal generalisation and concurrent serving were not evaluated.'''
    main=re.sub(r'\\abstract\{.*?\}\s*\\keywords',lambda _:r'\abstract{'+abstract+'}\n\n'+r'\keywords',main,count=1,flags=re.S)
    main=change(main,'is evaluated at 500 labels and one training seed.', 'is evaluated at 500 training labels and three training seeds.')
    main=change(main,'identifiers are distinct from the single LLM adaptation training seed.', 'identifiers are distinct from the three LLM adaptation training seeds.')
    begin=main.index('For local LLM adaptation, Phi-4 has the largest balanced F1 gain')
    end=main.index('The following archived Ollama recommendations',begin)
    main=main[:begin]+rf'''For local LLM adaptation, {MODELS[best]} has the highest three-fit
balanced mean point estimate ({primary.loc[best,'adapter_f1_weighted_mean']:.4f}).
Model-to-model superiority was not tested. Training-seed variation and
label-matched classifier comparisons are reported in
Section~\ref{{sec:qlora_results}}; point estimates alone do not establish
operational dominance. Serial means concern unmerged adapters and a fixed
warm benchmark; concurrent throughput was not evaluated.
''' +main[end:]
    old='''Eighth, LLM adaptation uses
only 500 training records and one training seed. Its three test subsets
and bootstrap intervals do not measure variation across independent
adapter fits; larger training budgets and additional training seeds remain
unevaluated for LLMs.'''
    main=change(main,old,'''Eighth, LLM adaptation uses only 500 training records and three
training seeds. Between-fit sample standard deviations complement
within-fit bootstrap intervals, but three fits do not establish robustness
across training corpora or larger LLM label budgets.''')
    main=change(main,'Batch and serial outputs\ndiffer on three of 600 benchmark predictions;',
                f'Batch and serial outputs differ on {serial_differences} of 1,800\nbenchmark predictions;')
    start=main.index('We compared archived zero-shot LLM configurations',main.index(r'\section{Conclusion}'))
    end=main.index(r'\section*{Declarations}',start)
    conclusion=rf'''We compared archived zero-shot configurations, controlled three-seed
QLoRA adaptation and direct classifiers on preserved CFPB tests.
At 500 training and 2,000 validation records, mean balanced F1 gains
range from {lo:.2f} to {hi:.2f} percentage points. Training variability
and within-fit test uncertainty are reported separately. The additional
500-record encoder controls use the same labels and validation split,
including validation-guided early stopping.

The larger-budget SVM and DistilBERT results show that available labels
and serving constraints must inform local model selection. Weighted
scores under reduced imbalance require macro-F1 and class-level inspection.
The historical API comparison provides zero-shot context without
establishing statistical equivalence or isolating reasoning effects.
Future work should evaluate larger LLM training budgets, longer narratives,
dated corpora and concurrent serving; these are not established by
three fits on the current short-narrative tests.

'''
    generated['main.tex']=main[:start]+conclusion+main[end:]
    preview=RUN/'manuscript_preview';preview.mkdir(exist_ok=True)
    for path in TEX.iterdir():
        if path.is_file() and path.suffix in ['.tex','.bib','.cls','.bst']:
            shutil.copy2(path,preview/path.name)
    for folder in ['figures','bst']:
        shutil.copytree(TEX/folder,preview/folder,dirs_exist_ok=True)
    for name,content in generated.items():
        (preview/name).write_text(content)
    shutil.copy2(RUN/'hf_qlora_multiseed_f1.pdf',preview/'figures/v2/hf_qlora_multiseed_f1.pdf')
    for document in ['supplementary_material','main']:
        with (RUN/f'{document}_build.log').open('w') as log:
            subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',document+'.tex'],
                           cwd=preview,stdout=log,stderr=subprocess.STDOUT,check=True)
        log=(preview/f'{document}.log').read_text()
        if 'undefined' in log.lower() or 'Overfull' in log:
            raise ValueError(f'Unresolved references or overfull text in {document}; preview retained')
    if __package__:
        from . import audit_manuscript
    else:
        import audit_manuscript
    audit_manuscript.TEX=preview
    audit=audit_manuscript.audit()
    for name,digest in original_hashes.items():
        if sha(TEX/name)!=digest:
            raise ValueError(f'Manuscript edited during export: {name}; validated preview retained')
    for name in generated:
        shutil.copy2(preview/name,TEX/name)
    shutil.copy2(RUN/'hf_qlora_multiseed_f1.pdf',TEX/'figures/v2/hf_qlora_multiseed_f1.pdf')
    pages={}
    for document in ['supplementary_material','main']:
        for suffix in ['pdf','aux','bbl','log']:
            shutil.copy2(preview/f'{document}.{suffix}',TEX/f'{document}.{suffix}')
        info=subprocess.check_output(['pdfinfo',str(TEX/f'{document}.pdf')],text=True)
        pages[document]=int(re.search(r'Pages:\s+(\d+)',info).group(1))
    stamp=datetime.now().astimezone().isoformat()
    notice=f'''\n\n## Ek deneyler tamamlandı — {stamp}\n\n500 eğitim / 2.000 doğrulama kaydıyla üç-seed DistilBERT kontrolleri\nve üç LLM'nin üç eğitim seed'i tamamlandı. Dokuz LLM fit'inin 216.000\ntest + 1.800 seri tahmini doğrulandı. Üç epoch encoder F1: {encoder_primary.loc['three_epochs','f1_weighted_mean']:.4f};\nerken durdurmalı encoder F1: {encoder_primary.loc['convergence','f1_weighted_mean']:.4f}.\nLLM balanced ortalama F1 artışları {lo:.2f}–{hi:.2f} yüzde puan.\nMakalede özet/yöntem/tablo/şekil/sonuç/tartışma üç eğitim seed'ine\ngöre güncellendi; yukarıdaki tek-seed açıklamalar önceki pilotu belgeler.\nSon ana/ek PDF: {pages['main']}/{pages['supplementary_material']} sayfa.\nZamansal test yapılmadı: yerel CSV tarih/özgün CFPB kimliği içermiyor.\nTam rapor: `results/sadhana/replication_20261008/report.md`.\n'''
    for name in ['sadhana_durum_raporu.md','hakem_degerlendirmesi_2026-10-08.md','yol_haritasi_sadhana.md']:
        with (TEX/name).open('a') as stream:stream.write(notice)
    result.write_text(json.dumps({'phase':'complete','updated_local':stamp,'pages':pages,
        'audit':audit,'backup':str(backup),'generated_files':list(generated),
        'checks':'Complete verified fits; isolated preview build; reference/object audit; source-change guard before live update'},indent=2))
    print(json.loads(result.read_text()),flush=True)


if __name__=='__main__':
    main()
