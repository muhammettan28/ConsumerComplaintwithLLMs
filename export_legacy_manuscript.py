"""Refresh archived 500-character result tables from unchanged predictions."""
from pathlib import Path
import argparse
import re

import pandas as pd

from analyze_seeds import CATEGORIES, REASONING, aggregate, load_all


def refreshed_text(text):
    raw = load_all()
    assert len(raw) == 132 and raw['n'].eq(1000).all()
    groups = {c: aggregate(raw, c, 500).set_index('model')
              for c in ['balanced', 'imbalanced']}
    for condition, label in [('balanced', 'tab:results_balanced'),
                             ('imbalanced', 'tab:results_imbalanced')]:
        pattern = re.compile(r'\\begin\{table\}.*?\\end\{table\}', re.S)
        matches = [m for m in pattern.finditer(text) if label in m.group()]
        assert len(matches) == 1
        match = matches[0]
        table = match.group()
        def row(m):
            model = m.group(1).replace(':', '_')
            values = groups[condition].loc[model]
            mode = 'U' if model == 'gemma4_12b' else ('R' if model in REASONING else 'S')
            def estimate(metric):
                std = f"{values[metric + '_std']:.3f}"[1:]
                return f"{values[metric + '_mean']:.3f}" + r'{\scriptsize($\pm$' + std + ')}'
            cells = [r'\texttt{' + m.group(1) + '}', mode,
                     estimate('accuracy'), estimate('f1_weighted'),
                     f"{values['f1_macro_mean']:.3f}",
                     f"{values['mean_latency_s_mean']:.2f}",
                     f"{values['off_label_mean']:.1f}"]
            return '    ' + ' & '.join(cells) + r' \\'
        table, n = re.subn(r'^\s*\\texttt\{([^}]+)\}[^\n]*\\\\', row, table, flags=re.M)
        assert n == 11
        table = table.replace('mean of 3 seeds; std in parentheses',
                              'mean of 3 test subsets; sample std in parentheses')
        text = text[:match.start()] + table + text[match.end():]
    pattern = re.compile(r'^(\s*)([AB]) & \\texttt\{([^}]+)\}[^\n]*\\\\', re.M)
    def class_row(m):
        condition = 'balanced' if m.group(2) == 'A' else 'imbalanced'
        values = groups[condition].loc[m.group(3).replace(':', '_')]
        cells = [m.group(2), r'\texttt{' + m.group(3) + '}']
        cells += [f"{values['f1_' + c + '_mean']:.4f}" for c in CATEGORIES]
        return m.group(1) + ' & '.join(cells) + r' \\'
    text, n = pattern.subn(class_row, text)
    assert n == 6
    text = text.replace('Exp.~A and~B, $N{=}1{,}000$)',
                        'Exp.~A and~B, mean over 3 subsets of $N{=}1{,}000$)')
    return text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', help='Verify without writing')
    args = parser.parse_args()
    path = Path('SPRINGER/main.tex')
    current = path.read_text()
    updated = refreshed_text(current)
    if args.check:
        if current != updated:
            raise SystemExit('Archived tables do not match saved predictions')
        print('All 22 archived model/condition rows and 6 class-metric rows match predictions')
    else:
        path.write_text(updated)
        print('Archived result tables refreshed; prediction files unchanged')
