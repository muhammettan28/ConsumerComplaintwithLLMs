"""Resumeable, serial Ollama thinking ablation on preserved <=500-char inputs.

python utils/run_thinking_control.py --prepare-only
python utils/run_thinking_control.py --resume
No pulls, training, altered narratives, or grammar constraints are used.
"""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

if __package__:
    from .run_inference import PROMPT_TEMPLATE, build_prompt
else:
    from run_inference import PROMPT_TEMPLATE, build_prompt
if __package__:
    from .sadhana_audit import CATEGORIES, strict_label, text_hash
else:
    from sadhana_audit import CATEGORIES, strict_label, text_hash
if __package__:
    from .summarize_qlora_comparison import measures, paired_bootstrap
else:
    from summarize_qlora_comparison import measures, paired_bootstrap

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/sadhana/thinking_control_20261009'
SOURCE = ROOT / 'dataset/1_subset_balanced_500.csv'
MODELS = {'gemma4': 'gemma4:12b', 'qwen3': 'qwen3:14b'}
SEED = 20261009
HOST = 'http://localhost:11434'


def stamp():
    return datetime.now(ZoneInfo('Europe/Istanbul')).isoformat()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def api(endpoint, payload=None, timeout=30):
    request = urllib.request.Request(HOST + '/api/' + endpoint,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=timeout) as stream:
        return json.load(stream)


def metadata():
    tags = {r['name']: r for r in api('tags')['models']}
    result = {'version': api('version')['version'], 'models': {}}
    for name, tag in MODELS.items():
        show = api('show', {'model': tag})
        assert show.get('thinking', {}).get('values') == [False, True], (tag, show.get('thinking'))
        result['models'][name] = {'tag': tags[tag], 'show': show}
    return result


def verify_metadata(plan):
    current = metadata()
    assert current['version'] == plan['metadata']['version'], 'Ollama version changed'
    for name in MODELS:
        old = plan['metadata']['models'][name]
        now = current['models'][name]
        assert now['tag']['digest'] == old['tag']['digest'], f'Artifact changed: {name}'
        for key in ['template', 'parameters', 'thinking']:
            assert now['show'].get(key) == old['show'].get(key), f'Model metadata changed: {name}:{key}'


def prepare():
    RUN.mkdir(parents=True, exist_ok=True)
    if (RUN / 'study_plan.json').exists():
        plan = json.loads((RUN / 'study_plan.json').read_text())
        assert plan['source_sha256'] == sha(SOURCE)
        assert plan['prompt_sha256'] == hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()
        for name, expected in plan['selection_sha256'].items():
            assert sha(RUN / name) == expected, f'Selection changed: {name}'
        verify_metadata(plan)
        return plan
    source = pd.read_csv(SOURCE).reset_index(names='source_index')
    assert len(source) == 1000 and source.true_label.value_counts().eq(200).all()
    assert source.narrative.str.len().between(1, 500).all()
    source['text_hash'] = source.narrative.map(text_hash)
    conflicts = set(source.groupby('text_hash').true_label.nunique().loc[lambda s: s > 1].index)
    eligible = source[~source.text_hash.isin(conflicts)].drop_duplicates('text_hash')
    rng = np.random.default_rng(SEED)
    pilot, test = [], []
    for category in CATEGORIES:
        indices = eligible.index[eligible.true_label == category].to_numpy()
        picked = rng.choice(indices, 102, replace=False)
        pilot.extend(picked[:2]); test.extend(picked[2:])
    selections = {'pilot_inputs.csv': source.loc[sorted(pilot)],
                  'evaluation_inputs.csv': source.loc[sorted(test)]}
    assert not set(selections['pilot_inputs.csv'].text_hash) & set(selections['evaluation_inputs.csv'].text_hash)
    for name, df in selections.items():
        df.to_csv(RUN / name, index=False)
    plan = {'created_local': stamp(), 'source': str(SOURCE.relative_to(ROOT)),
        'source_sha256': sha(SOURCE), 'selection_seed': SEED,
        'evaluation_records': 500, 'records_per_class': 100, 'pilot_records': 10,
        'pilot_excluded_by_normalized_text_hash': True,
        'selection_sha256': {name: sha(RUN / name) for name in selections},
        'prompt_sha256': hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest(),
        'metadata': metadata(), 'models': MODELS, 'modes': [False, True],
        'options_without_output_cap': {'temperature': 0, 'seed': SEED, 'num_ctx': 8192, 'num_batch': 256, 'num_gpu': 99},
        'output_cap_candidates': [1024, 2048, 4096],
        'cap_selection': 'First cap with no pilot length termination or empty final response in either model/mode; class scores never used',
        'warmup': 'One excluded pilot record per mode after each model load/restart; stored separately',
        'ordering': 'Serial pairs, alternating which mode goes first by selected-record position',
        'parser': 'JSON decoder, last explicit product string from final response only; reasoning excluded',
        'statistics': {'bootstrap_replicates': 5000, 'bootstrap': 'paired normalized-text clusters, stratified by class',
                       'accuracy': 'exact two-sided McNemar; Holm family of two model comparisons',
                       'intervals': 'unadjusted 95% percentile F1-change intervals, conditional on this fixed subset'},
        'runtime': {'python': os.sys.version, 'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'], text=True).strip()},
        'scope': 'Base GGUF models only; same preserved <=500-character inputs. No training or narrative-length/prompt/classifier experiments.',
        'historical_mode_verified': False}
    save(RUN / 'study_plan.json', plan)
    return plan


def journal(path):
    if not path.exists():
        return []
    result = []
    good_bytes = 0
    with path.open('rb') as stream:
        while line := stream.readline():
            try:
                result.append(json.loads(line))
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Only a final interrupted write may be repaired.
                assert not stream.read(), f'Corrupt non-final journal line: {path}'
                break
            good_bytes += len(line)
    if path.stat().st_size != good_bytes:
        with path.open('r+b') as stream:
            stream.truncate(good_bytes)
    return result


def append(path, record):
    with path.open('a') as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def parse_answer(body):
    if body.get('done_reason') == 'length':
        return 'output_limit', 'output_limit'
    label = strict_label(body.get('response', ''))
    return label, 'ok' if label in CATEGORIES else 'parse_error' if label == 'parse_error' else 'off_label'


def call(name, row, think, cap, role):
    payload = {'model': MODELS[name], 'prompt': build_prompt(row['narrative']),
               'think': think, 'stream': False, 'keep_alive': '10m',
               'options': {**PLAN['options_without_output_cap'], 'num_predict': cap}}
    begin = time.monotonic()
    body = api('generate', payload, timeout=600)
    elapsed = time.monotonic() - begin
    assert body.get('done') is True, 'Incomplete Ollama response'
    assert isinstance(body.get('response', ''), str) and isinstance(body.get('thinking', ''), str)
    if not think:
        assert not body.get('thinking'), 'Thinking emitted when disabled'
    label, status = parse_answer(body)
    return {'model': name, 'think': think, 'role': role, 'source_index': int(row['source_index']),
            'text_hash': row['text_hash'], 'true_label': row['true_label'],
            'request': payload, 'response': body, 'predicted_label': label,
            'status': status,
            'latency_seconds': elapsed, 'recorded_local': stamp()}


def unload(name):
    api('generate', {'model': MODELS[name], 'keep_alive': 0})


def progress(phase, **extra):
    state = json.loads((RUN / 'status.json').read_text()) if (RUN / 'status.json').exists() else {}
    if phase != 'failed':
        state.pop('error', None)
    state.update({'phase': phase, 'updated_local': stamp(), 'pid': os.getpid(), **extra})
    save(RUN / 'status.json', state)


def pilot_cap(plan):
    if (RUN / 'protocol.json').exists():
        protocol = json.loads((RUN / 'protocol.json').read_text())
        assert protocol['study_plan_sha256'] == sha(RUN / 'study_plan.json')
        if protocol.get('amendment_sha256'):
            assert protocol['amendment_sha256'] == sha(RUN/'protocol_amendment.json')
        return protocol['output_cap']
    inputs = pd.read_csv(RUN / 'pilot_inputs.csv').to_dict('records')
    for cap in plan['output_cap_candidates']:
        path = RUN / f'pilot_{cap}.jsonl'
        records = journal(path)
        done = {(r['model'], r['source_index'], r['think']) for r in records}
        for name in MODELS:
            verify_metadata(plan)
            progress('pilot', active_model=name, output_cap=cap, pilot_done=len(records), pilot_total=40)
            for pos, row in enumerate(inputs):
                for think in ([False, True] if pos % 2 == 0 else [True, False]):
                    if (name, row['source_index'], think) in done:
                        continue
                    record = call(name, row, think, cap, 'technical_pilot')
                    append(path, record); records.append(record)
                    progress('pilot', active_model=name, output_cap=cap, pilot_done=len(records), pilot_total=40)
            unload(name)
        assert len(records) == 40
        for name in MODELS:
            assert any(r['response'].get('thinking') for r in records if r['model'] == name and r['think']), f'Thinking not observed: {name}'
        failures = sum(r['response'].get('done_reason') == 'length' or not r['response'].get('response', '').strip() for r in records)
        save(RUN / f'pilot_{cap}_termination_summary.json', {'cap': cap, 'n': 40, 'length_or_empty_final': failures,
             'no_accuracy_based_selection': True})
        amendment_path = RUN/'protocol_amendment.json'
        bounded_fallback = False
        if failures and cap == plan['output_cap_candidates'][-1] and amendment_path.exists():
            amendment = json.loads(amendment_path.read_text())
            assert amendment['study_plan_sha256'] == sha(RUN/'study_plan.json')
            assert amendment['before_any_main_predictions'] and amendment['no_classification_accuracy_used']
            assert amendment['highest_prespecified_cap'] == cap
            bounded_fallback = True
        if failures == 0 or bounded_fallback:
            protocol = {'frozen_local': stamp(), 'study_plan_sha256': sha(RUN / 'study_plan.json'),
                        'output_cap': cap, 'pilot_source_sha256': sha(path),
                        'total_main_predictions': 2000, 'same_cap_for_both_modes_and_models': True,
                        'residual_pilot_length_or_empty_final': failures,
                        'bounded_fallback_used': bounded_fallback,
                        'amendment_sha256': sha(amendment_path) if bounded_fallback else None,
                        'definition': 'Total generated tokens, including reasoning; limit failures remain evaluation errors'}
            save(RUN / 'protocol.json', protocol)
            return cap
    raise RuntimeError('Pilot still reaches output limit; no main-test budget chosen')


def evaluate(plan, cap):
    inputs = pd.read_csv(RUN / 'evaluation_inputs.csv').to_dict('records')
    path = RUN / 'predictions.jsonl'
    records = journal(path)
    valid = {(name, row['source_index'], think) for name in MODELS for row in inputs for think in [False, True]}
    done = {(r['model'], r['source_index'], r['think']) for r in records}
    assert len(done) == len(records) and done <= valid
    for r in records:
        assert r['request']['options'] == {**plan['options_without_output_cap'], 'num_predict': cap}
    fresh = 0; elapsed = 0
    for name in MODELS:
        if all((name, row['source_index'], think) in done for row in inputs for think in [False, True]):
            continue
        verify_metadata(plan)
        pilot = pd.read_csv(RUN / 'pilot_inputs.csv').iloc[0].to_dict()
        for think in [False, True]:
            append(RUN / 'warmups.jsonl', call(name, pilot, think, cap, 'warmup_excluded'))
        save(RUN / f'{name}_loaded_runtime.json', api('ps'))
        for pos, row in enumerate(inputs):
            if pos % 25 == 0:
                verify_metadata(plan)
            for think in ([False, True] if pos % 2 == 0 else [True, False]):
                if (name, row['source_index'], think) in done:
                    continue
                progress('evaluation', active_model=name, think=think, done_predictions=len(records),
                         total_predictions=2000, fresh_predictions=fresh, inference_seconds=elapsed,
                         output_cap=cap)
                record = call(name, row, think, cap, 'evaluation')
                append(path, record); records.append(record)
                done.add((name, row['source_index'], think))
                fresh += 1; elapsed += record['latency_seconds']
                progress('evaluation', active_model=name, think=think, done_predictions=len(records),
                         total_predictions=2000, fresh_predictions=fresh, inference_seconds=elapsed,
                         output_cap=cap, estimated_remaining_minutes=(2000-len(records))*elapsed/fresh/60)
        unload(name)
    assert len(records) == 2000
    verify_metadata(plan)
    return records


def summarize(records):
    inputs = pd.read_csv(RUN / 'evaluation_inputs.csv').set_index('source_index')
    rows = []
    for r in records:
        source = inputs.loc[r['source_index']]
        assert r['text_hash'] == source.text_hash and r['true_label'] == source.true_label
        assert r['request']['prompt'] == build_prompt(source.narrative)
        assert r['request']['think'] == r['think'] and r['request']['model'] == MODELS[r['model']]
        assert (r['predicted_label'], r['status']) == parse_answer(r['response'])
        rows.append({k: r[k] for k in ['model','think','source_index','text_hash','true_label','predicted_label','status','latency_seconds']} | {
            'output_tokens': r['response']['eval_count'], 'input_tokens': r['response']['prompt_eval_count'],
            'done_reason': r['response'].get('done_reason'), 'reasoning_characters': len(r['response'].get('thinking', '')),
            'output_limit': r['response'].get('done_reason') == 'length'})
    data = pd.DataFrame(rows); data.to_csv(RUN / 'predictions.csv', index=False)
    modes, comparisons = [], []
    for name in MODELS:
        for think in [False, True]:
            d = data[(data.model == name) & (data.think == think)]
            assert len(d) == 500 and d.true_label.value_counts().eq(100).all()
            assert d.source_index.is_unique
            modes.append({'model': name, 'think': think, 'n': 500,
                **measures(d.true_label, d.predicted_label), 'invalid': int((d.status != 'ok').sum()),
                'length_terminations': int(d.output_limit.sum()),
                'empty_final': sum(not r['response'].get('response','').strip() for r in records if r['model']==name and r['think']==think),
                'mean_latency_s': float(d.latency_seconds.mean()), 'median_latency_s': float(d.latency_seconds.median()),
                'p95_latency_s': float(d.latency_seconds.quantile(.95)), 'mean_generated_tokens': float(d.output_tokens.mean())})
        off = data[(data.model == name) & ~data.think]
        on = data[(data.model == name) & data.think]
        pair = off.merge(on, on=['source_index','text_hash','true_label'], suffixes=('_base','_adapter'), validate='one_to_one')
        a = pair.predicted_label_base.eq(pair.true_label); b = pair.predicted_label_adapter.eq(pair.true_label)
        n10 = int((a & ~b).sum()); n01 = int((~a & b).sum())
        ci = paired_bootstrap(pair, 5000)
        comparisons.append({'model': name, 'off_only_correct': n10, 'on_only_correct': n01,
            'delta_accuracy': float(b.mean()-a.mean()),
            'mcnemar_p': float(binomtest(n01, n01+n10, .5).pvalue) if n10+n01 else 1.,
            **{k.replace('base_', 'off_').replace('adapter_', 'on_'): v for k,v in ci.items()}})
    summary = pd.DataFrame(modes); comparison = pd.DataFrame(comparisons)
    comparison['mcnemar_p_holm'] = multipletests(comparison.mcnemar_p, method='holm')[1]
    summary.to_csv(RUN / 'summary.csv', index=False); comparison.to_csv(RUN / 'paired_comparison.csv', index=False)
    save(RUN / 'verification.json', {'verified_local': stamp(), 'predictions': 2000, 'unique_test_inputs': 500,
        'source_sha256': sha(SOURCE), 'selection_sha256': sha(RUN/'evaluation_inputs.csv'),
        'prediction_sha256': sha(RUN/'predictions.jsonl'), 'pairs': 1000,
        'pilot_test_overlap': 0, 'strict_reparse_and_request_pairing': 'pass',
        'model_metadata': 'digest/template/parameters/version unchanged'})
    return summary, comparison


def export(summary, comparison, install=True):
    tex = ROOT / 'SPRINGER'
    lines = [r'\subsection{Controlled Thinking-On/Off Comparison}', r'\label{sec:thinking_control}',
        'We separately compare thinking policies for the currently installed',
        'Gemma~4~12B and Qwen3~14B GGUF artifacts. The same 500 preserved',
        'balanced short-narrative inputs (100 per class) are evaluated in both',
        'modes; an excluded ten-record technical pilot selects a common total',
        'generation cap before evaluation. Temperature is zero; prompts,',
        'artifact, context, runtime and generation cap are fixed within pairs.',
        'Requests run serially after warmup, with alternating mode order.',
        'Only final-answer JSON is parsed; invalid outputs remain errors.',
        'Table~\\ref{tab:thinking_control} reports this additional base-model',
        'experiment; it does not recover the unrecorded modes of archived runs.',
        r'\begin{table}[tb]\centering\small',
        r'\caption{Thinking policies on the same 500 balanced inputs. Latency',
        r'is warm serial HTTP wall time, including full-response transfer and',
        r'excluding initial loading; it is separate from HF generation timing.',
        r'Generated tokens include reasoning. Invalid outputs remain in F1.}',
        r'\label{tab:thinking_control}', r'\begin{tabular}{llrrrr}\toprule',
        r'Model & Thinking & F1 & Invalid & Latency (s) & Tokens \\\midrule']
    for row in summary.itertuples():
        name = 'Gemma 4' if row.model == 'gemma4' else 'Qwen3'
        lines.append(f'{name} & {"On" if row.think else "Off"} & {row.f1_weighted:.4f} & {row.invalid} & {row.mean_latency_s:.2f} & {row.mean_generated_tokens:.1f}'+r' \\')
    lines.append(r'\botrule\end{tabular}\end{table}')
    for row in comparison.itertuples():
        name = 'Gemma~4' if row.model == 'gemma4' else 'Qwen3'
        delta = (summary[(summary.model==row.model)&summary.think].iloc[0].f1_weighted -
                 summary[(summary.model==row.model)&~summary.think].iloc[0].f1_weighted)*100
        lines.append(f'For {name}, the thinking-on minus thinking-off weighted-F1 change is '
            f'{delta:+.2f} percentage points; the unadjusted paired 95\\,\\% interval is '
            f'[{row.delta_f1_weighted_ci_low*100:+.2f}, {row.delta_f1_weighted_ci_high*100:+.2f}]. '
            f'The two-model Holm-adjusted accuracy-test $p$ value is {row.mcnemar_p_holm:.4g}.')
    lines += ['These intervals condition on the fixed test subset and current',
              'artifacts; no new training-seed or prompt robustness is established.',
              'The comparison estimates thinking policy under a fixed total-token',
              'budget, including any limit terminations. Complete protocol and',
              'class metrics are given in Supplementary Section~\\ref{S-sec:thinking_protocol}.']
    protocol = json.loads((RUN/'protocol.json').read_text())
    supplementary = [r'\subsection{Thinking-control protocol and class metrics}', r'\label{sec:thinking_protocol}',
        f'The evaluation uses selection seed {SEED}, a total generation cap of {protocol["output_cap"]} tokens',
        'in both modes and both models, an 8,192-token context, temperature zero',
        f'and Ollama {PLAN["metadata"]["version"]}. No JSON grammar is forced.',
        'The ten balanced pilot records are excluded from the 500 evaluation',
        'records by normalised-text hash. Selection precedes all pilot outputs;',
        'the cap is selected from termination/empty-final flags, never accuracy.',
        'Each request stores its full prompt/options, separate reasoning and final',
        'response, token counts, termination reason, loading/generation duration',
        'and serial HTTP wall time. Cluster-stratified paired bootstrap uses',
        '5,000 draws; two exact two-sided McNemar tests use Holm correction.',
        'Raw outputs and manifests are in', r'\path{results/sadhana/thinking_control_20261009/}.',
        r'\begin{itemize}']
    if protocol.get('bounded_fallback_used'):
        supplementary[-1:-1] = ['At the highest prespecified cap, one excluded Gemma thinking-on',
            'pilot input still terminates at the length limit. A technical',
            'amendment recorded before any main predictions fixes this finite',
            'budget instead of extending the search. All length-terminated main',
            'responses count as errors, including any embedded JSON; the two modes',
            'retain identical budgets and unchanged input selection.']
    for name in MODELS:
        m = PLAN['metadata']['models'][name]['tag']
        supplementary.append(r'\item '+m['name'].replace('_',r'\_')+' ('+m['details']['quantization_level'].replace('_',r'\_')+r'): artifact digest \newline\texttt{'+m['digest']+'}.')
    supplementary.extend([r'\end{itemize}',r'Table~\ref{tab:thinking_class} reports per-class F1.',
        r'\begin{table}[tb]\centering\small',r'\caption{Class F1 in the controlled thinking experiment.}',
        r'\label{tab:thinking_class}',r'\begin{tabular}{llrrrrr}\toprule',
        r'Model & Mode & Retail & CreditRep. & Card & Mortg. & Debt \\\midrule'])
    for row in summary.to_dict('records'):
        supplementary.append(('Gemma 4' if row['model']=='gemma4' else 'Qwen3')+' & '+('On' if row['think'] else 'Off')+' & '+
            ' & '.join(f'{row[f"f1_{c}"]:.4f}' for c in CATEGORIES)+r' \\')
    supplementary.append(r'\bottomrule\end{tabular}\end{table}')
    # Stage exact sources and validate both PDFs before replacing live files.
    import shutil
    protected = [*tex.glob('*.tex'), tex/'references.bib']
    before_sha256 = {str(p): sha(p) for p in protected}
    preview = RUN/'manuscript_preview'
    if preview.exists():
        shutil.rmtree(preview)
    shutil.copytree(tex, preview)
    (preview/'thinking_control.tex').write_text('\n'.join(lines)+'\n')
    (preview/'thinking_control_supplementary.tex').write_text('\n'.join(supplementary)+'\n')
    main = (preview/'main.tex').read_text()
    main = main.replace('the benefit of reasoning. A controlled thinking-on/off comparison on\nidentical complaints is needed to support that causal claim.',
        'the benefit of reasoning. The additional paired thinking-policy\ncomparison in Section~\\ref{sec:thinking_control} uses current pinned\nartifacts and identical complaints; it does not retrospectively identify\nthe modes or artifacts of the archived benchmark.')
    if r'\input{thinking_control}' not in main:
        assert main.count(r'\input{qlora_results}')==1
        main = main.replace(r'\input{qlora_results}', r'\input{qlora_results}'+'\n'+r'\input{thinking_control}')
    (preview/'main.tex').write_text(main)
    supp = (preview/'supplementary_material.tex').read_text()
    if r'\input{thinking_control_supplementary}' not in supp:
        assert supp.count(r'\input{serial_token_analysis}')==1
        supp = supp.replace(r'\input{serial_token_analysis}', r'\input{serial_token_analysis}'+'\n'+r'\input{thinking_control_supplementary}')
    (preview/'supplementary_material.tex').write_text(supp)
    if __package__:
        from .audit_manuscript import audit
    else:
        from audit_manuscript import audit
    if __package__:
        from . import audit_manuscript
    else:
        import audit_manuscript
    old = audit_manuscript.TEX
    try:
        audit_manuscript.TEX=preview
        checked = audit()
    finally:
        audit_manuscript.TEX=old
    for stem in ['supplementary_material','main']:
        with (RUN/f'{stem}_build.log').open('w') as log:
            subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',stem+'.tex'], cwd=preview,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        log = (preview/f'{stem}.log').read_text()
        assert not any(w in log for w in ['Overfull', 'undefined references', 'undefined on input', 'Undefined control sequence']), f'Build warnings: {stem}'
    assert all(sha(Path(p)) == expected for p, expected in before_sha256.items()), 'Manuscript edited during preview build; retry using latest sources'
    if not install:
        return checked
    backup=RUN/'manuscript_before_thinking';backup.mkdir(exist_ok=True)
    for name in ['main.tex','supplementary_material.tex']:
        if not (backup/name).exists():
            shutil.copy2(tex/name, backup/name)
    for name in ['main.tex','supplementary_material.tex','thinking_control.tex','thinking_control_supplementary.tex']:
        shutil.copy2(preview/name,tex/name)
    for stem in ['main','supplementary_material']:
        for ext in ['pdf','aux','bbl','blg','log','out','fdb_latexmk','fls']:
            p=preview/f'{stem}.{ext}'
            if p.exists():
                shutil.copy2(p,tex/p.name)
    save(RUN/'manuscript_export.json', {'exported_local':stamp(), 'phase':'complete', 'audit':checked,
          'note':'New controlled artifact experiment; historical unverified modes unchanged'})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--export-manuscript', action='store_true',
                        help='Also export to locally available manuscript sources')
    args = parser.parse_args()
    if args.export_manuscript and not all(path.exists() for path in [
            ROOT/'SPRINGER/main.tex', ROOT/'utils/audit_manuscript.py']):
        parser.error('--export-manuscript requires the local manuscript files and audit utility')
    (ROOT/'checkpoints').mkdir(exist_ok=True)
    with (ROOT/'checkpoints/.qlora_queue.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        global PLAN
        if not args.prepare_only and (RUN/'status.json').exists() and json.loads((RUN/'status.json').read_text()).get('phase')=='complete':
            print('Thinking control already complete.', flush=True)
            return
        PLAN=prepare()
        if args.prepare_only:
            progress('prepared', done_predictions=0, total_predictions=2000)
            print('Frozen inputs and model metadata prepared.', flush=True)
            return
        if (RUN/'status.json').exists() and json.loads((RUN/'status.json').read_text()).get('phase')=='complete':
            print('Thinking control already complete.', flush=True)
            return
        if not args.resume:
            assert not (RUN/'predictions.jsonl').exists(), 'Use --resume for existing outputs'
        try:
            cap=pilot_cap(PLAN)
            records=evaluate(PLAN,cap)
            progress('analysis',done_predictions=2000,total_predictions=2000)
            summary,comparison=summarize(records)
            if args.export_manuscript:
                progress('manuscript_export',done_predictions=2000,total_predictions=2000)
                export(summary,comparison)
            progress('complete',active_model=None,done_predictions=2000,total_predictions=2000,
                     manuscript_updated=args.export_manuscript,completed_local=stamp())
            print(summary.to_string(index=False),flush=True)
            subprocess.run(['systemctl','--user','disable','consumercomplaint-thinking-control.service'],
                           capture_output=True, check=False)
        except BaseException as exc:
            progress('failed',error=f'{type(exc).__name__}: {exc}')
            raise


if __name__=='__main__':
    main()
