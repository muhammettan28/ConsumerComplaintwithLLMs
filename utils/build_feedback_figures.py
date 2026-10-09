"""Build reviewer-requested views from saved data; no training or inference.

python utils/build_feedback_figures.py
Source hashes and plotted values are saved beside the feedback audit.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np
import pandas as pd

if __package__:
    from .sadhana_audit import CATEGORIES
else:
    from sadhana_audit import CATEGORIES

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / 'SPRINGER'
RUN = ROOT / 'results/sadhana/replication_20261008'
OUT = ROOT / 'results/sadhana/reviewer_audit/post_replication/feedbacks'
MODELS = {'gemma4': 'Gemma 4', 'qwen3': 'Qwen3', 'phi4': 'Phi-4'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [ROOT / 'dataset/complaints.csv',
               ROOT / 'dataset/1_subset_balanced_500.csv',
               ROOT / 'dataset/1_subset_imbalanced_500.csv',
               ROOT / 'results/sadhana/supervised_baseline_summary.csv',
               RUN / 'qlora_seed_summary.csv', RUN / 'qlora_per_fit.csv',
               RUN / 'encoder_summary.csv']
    raw = pd.read_csv(sources[0]).dropna(subset=['product', 'narrative'])
    raw = raw[raw['product'].isin(CATEGORIES)].copy()
    raw['narrative'] = raw.narrative.str.strip()
    filtered = raw[raw.narrative.str.len().between(1, 500)]
    pool = filtered.drop_duplicates('narrative')
    assert len(filtered) == 103915 and len(pool) == 72611
    counts = pd.DataFrame({
        'Length-filtered (N=103,915)': filtered['product'].value_counts(),
        'Deduplicated pool (N=72,611)': pool['product'].value_counts(),
        'Reduced imbalance (N=1,000)': pd.read_csv(sources[2]).true_label.value_counts(),
        'Balanced (N=1,000)': pd.read_csv(sources[1]).true_label.value_counts(),
    }).reindex(CATEGORIES)
    counts.to_csv(OUT / 'class_distribution_counts.csv')
    percentages = counts.div(counts.sum(axis=0), axis=1) * 100
    percentages.to_csv(OUT / 'class_distribution_percentages.csv')
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(5)
    for i, (name, color) in enumerate(zip(counts.columns, ['#94a3b8', '#14b8a6', '#60a5fa', '#1d4ed8'])):
        ax.bar(x + (i - 1.5) * .19, percentages[name], width=.18, color=color, label=name)
    ax.set_xticks(x, ['Retail\nbanking', 'Credit\nreporting', 'Credit\ncard', 'Mortgages\n& loans', 'Debt\ncollection'])
    ax.set_ylabel('Class proportion (%)')
    ax.set_ylim(0, 75)
    ax.legend(fontsize=8, ncol=2, loc='upper right')
    ax.grid(axis='y', alpha=.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(TEX / 'figures/v2/class_distribution.png', dpi=300)
    plt.close(fig)

    summary = pd.read_csv(RUN / 'qlora_seed_summary.csv')
    fits = pd.read_csv(RUN / 'qlora_per_fit.csv')
    balanced = summary[(summary.condition == 'balanced') & (summary['size'] == 500)].set_index('model')
    classical = pd.read_csv(sources[3])
    svm = classical[(classical.model == 'tfidf_linear_svc') & (classical.condition == 'balanced')].sort_values('train_n')
    encoder = pd.read_csv(RUN / 'encoder_summary.csv')
    low_encoder = encoder[(encoder.variant == 'convergence') & (encoder.condition == 'balanced') & (encoder['size'] == 500)]
    assert len(low_encoder) == 1
    high_encoder = classical[(classical.model == 'distilbert') & (classical.condition == 'balanced') & (classical.train_n == 20000)]
    assert len(high_encoder) == 3
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(svm.train_n, svm.f1_weighted, 'o-', color='#111827', label='TF-IDF/SVM')
    ax.scatter([500, 20000], [low_encoder.iloc[0].f1_weighted_mean, high_encoder.f1_weighted.mean()],
               color='#9333ea', marker='D', label='DistilBERT (observed budgets)', zorder=6)
    colors = ['#2563eb', '#d97706', '#059669']
    for (model, name), color, marker in zip(MODELS.items(), colors, ['s', '^', 'o']):
        row = balanced.loc[model]
        ax.axhline(row.base_f1_weighted_mean, color=color, linestyle=':', linewidth=1,
                   label=f'{name} HF base (zero training labels)')
        ax.scatter([500], [row.adapter_f1_weighted_mean], color=color, marker=marker,
                   s=65, label=f'{name} QLoRA (500 training labels)', zorder=5)
    ax.set_xscale('log')
    ax.set_xlim(350, 27000)
    ax.set_xticks([500, 2000, 10000, 20000], ['500', '2,000', '10,000', '20,000'])
    ax.set_xlabel('Training labels (all supervised points use 2,000 validation labels)')
    ax.set_ylabel('Balanced weighted F1')
    ax.set_ylim(.65, .86)
    top = ax.secondary_xaxis('top', functions=(lambda n: n + 2000, lambda n: n - 2000))
    top.set_xticks([2500, 4000, 12000, 22000], ['2,500', '4,000', '12,000', '22,000'])
    top.xaxis.set_minor_locator(NullLocator())
    top.set_xlabel('Total labels: training + validation (supervised points)')
    ax.grid(alpha=.2)
    ax.legend(fontsize=7, ncol=2, loc='lower right')
    fig.tight_layout()
    fig.savefig(TEX / 'figures/v2/label_budget_tradeoff.pdf')
    fig.savefig(OUT / 'label_budget_tradeoff.png', dpi=160)
    plt.close(fig)
    plotted = []
    for row in svm.itertuples():
        plotted.append({'model': 'TF-IDF/SVM', 'train_n': row.train_n, 'validation_n': 2000, 'f1': row.f1_weighted})
    for n, value in [(500, low_encoder.iloc[0].f1_weighted_mean), (20000, high_encoder.f1_weighted.mean())]:
        plotted.append({'model': 'DistilBERT', 'train_n': n, 'validation_n': 2000, 'f1': value})
    for model, row in balanced.iterrows():
        plotted.extend([{'model': f'{model} QLoRA', 'train_n': 500, 'validation_n': 2000, 'f1': row.adapter_f1_weighted_mean},
                        {'model': f'{model} HF base', 'train_n': 0, 'validation_n': 0, 'f1': row.base_f1_weighted_mean}])
    pd.DataFrame(plotted).to_csv(OUT / 'label_budget_figure_data.csv', index=False)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.7), sharey=True)
    for ax, condition in zip(axes, ['balanced', 'imbalanced']):
        for i, (model, name) in enumerate(MODELS.items()):
            row = summary[(summary.model == model) & (summary.condition == condition) & (summary['size'] == 500)].iloc[0]
            selected = fits[(fits.model == model) & (fits.condition == condition) & (fits['size'] == 500)].sort_values('training_seed')
            assert len(selected) == 3
            for mode, offset, color in [('base', -.13, '#64748b'), ('adapter', .13, '#2563eb')]:
                ax.errorbar(i + offset, row[f'{mode}_f1_weighted_mean'], yerr=row[f'{mode}_f1_weighted_std'],
                            fmt='o', color=color, capsize=3, label=('HF base' if mode == 'base' else 'QLoRA') if i == 0 else None)
                ax.scatter(i + offset + np.array([-.035, 0, .035]), selected[f'{mode}_f1_weighted'],
                           s=14, facecolors='none', edgecolors=color, zorder=4)
        ax.set_xticks(np.arange(3), list(MODELS.values()))
        ax.set_ylim(.65, .85)
        ax.set_title('Balanced' if condition == 'balanced' else 'Reduced imbalance')
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Weighted F1')
    axes[0].legend(fontsize=8, loc='upper left')
    fig.suptitle('500 training + 2,000 validation labels\nFilled: mean ± training-seed SD; hollow: individual fits', fontsize=9)
    fig.tight_layout()
    fig.savefig(TEX / 'figures/v2/hf_qlora_multiseed_f1.pdf')
    fig.savefig(OUT / 'hf_qlora_multiseed_f1.png', dpi=160)
    plt.close(fig)

    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (OUT / 'figure_sources.json').write_text(json.dumps({'source_sha256': hashes, 'new_training_or_inference': False}, indent=2) + '\n')
    print('Built class-prior, label-budget and per-fit figures from frozen artifacts.')


if __name__ == '__main__':
    main()
