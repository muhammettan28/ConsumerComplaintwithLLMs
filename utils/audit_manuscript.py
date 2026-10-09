"""Read-only checks of manuscript objects, citations and external references."""
from pathlib import Path
import json
import re

ROOT = Path(__file__).resolve().parents[1]
TEX = ROOT / 'SPRINGER'


def expand(path, stack=()):
    assert path not in stack, f'Recursive input: {path}'
    text = re.sub(r'(?<!\\)%[^\n]*', '', path.read_text())
    def include(match):
        child = path.parent / (match.group(1) + '.tex')
        return expand(child, stack + (path,))
    return re.sub(r'\\input\{([^}]+)\}', include, text)


def audit():
    documents = {name: expand(TEX / name) for name in
                 ['main.tex', 'supplementary_material.tex']}
    labels = {name: re.findall(r'\\label\{([^}]+)\}', text)
              for name, text in documents.items()}
    bibkeys = set(re.findall(r'@\w+\{([^,]+),', (TEX / 'references.bib').read_text()))
    result = {'documents': {}, 'uncited_objects': [], 'undefined_refs': [],
              'undefined_citations': [], 'duplicate_labels': []}
    allrefs = {name: re.findall(r'\\(?:ref|eqref|pageref)\{([^}]+)\}', text)
               for name, text in documents.items()}
    for name, text in documents.items():
        counts = {'figure': 0, 'table': 0}
        for kind, block in re.findall(r'\\begin\{(figure|table|longtable)\}\s*(.*?)\\end\{\1\}', text, re.S):
            if kind == 'longtable':
                kind = 'table'
            counts[kind] += 1
            object_labels = re.findall(r'\\label\{([^}]+)\}', block)
            assert object_labels and r'\caption' in block, (name, kind)
            cited = any(label in allrefs[name] or
                        (name == 'supplementary_material.tex' and 'S-' + label in allrefs['main.tex'])
                        for label in object_labels)
            if not cited:
                result['uncited_objects'].append({'document': name, 'labels': object_labels})
        result['documents'][name] = counts
        result['duplicate_labels'] += [f'{name}:{x}' for x in set(labels[name]) if labels[name].count(x) > 1]
        for ref in allrefs[name]:
            resolved = (ref in labels[name] or
                        (name == 'main.tex' and ref.startswith('S-') and ref[2:] in labels['supplementary_material.tex']))
            if not resolved:
                result['undefined_refs'].append(f'{name}:{ref}')
        for cite in re.findall(r'\\cite\{([^}]+)\}', text):
            result['undefined_citations'] += [f'{name}:{key}' for key in cite.split(',') if key.strip() not in bibkeys]
    abstract = re.search(r'\\abstract\{(.*?)\}\s*\\keywords', documents['main.tex'], re.S).group(1)
    result['abstract_word_count_whitespace'] = len(abstract.split())
    result['abstract_hardware_terms'] = re.findall(r'\b(?:GPU|RTX|CUDA|NF4|BF16|VRAM|PyTorch|Transformers)\b', abstract)
    result['bibliography_entry_count'] = len(bibkeys)
    for key in ['uncited_objects', 'undefined_refs', 'undefined_citations', 'duplicate_labels', 'abstract_hardware_terms']:
        assert not result[key], f'{key}: {result[key]}'
    return result


if __name__ == '__main__':
    report = audit()
    output = ROOT / 'results/sadhana/reviewer_audit/manuscript_structure_after.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
