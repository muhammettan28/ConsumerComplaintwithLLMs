"""Preserve all saved test subsets and create balanced, nested training subsets.

Near-duplicate screening: approximate MinHash candidate retrieval followed by
exact word-shingle Jaccard verification. This is not a proof that all semantic
paraphrases have been removed. All outputs carry stable text hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH

if __package__:
    from .sadhana_audit import CATEGORIES, ROOT, normalize, text_hash
else:
    from sadhana_audit import CATEGORIES, ROOT, normalize, text_hash


def shingles(text):
    words = text.split()
    return {" ".join(words[i:i+3]) for i in range(len(words)-2)} if len(words) >= 3 else {text}


def signature(tokens):
    mh = MinHash(num_perm=128, seed=42)
    mh.update_batch([s.encode() for s in sorted(tokens)])
    return mh


class NearIndex:
    def __init__(self, threshold):
        self.threshold = threshold
        self.lsh = MinHashLSH(threshold=threshold, num_perm=128)
        self.tokens = {}

    def add(self, text):
        key = text_hash(text)
        if key not in self.tokens:
            tokens = shingles(text)
            self.tokens[key] = tokens
            self.lsh.insert(key, signature(tokens))

    def matches(self, text):
        tokens = shingles(text)
        for key in self.lsh.query(signature(tokens)):
            other = self.tokens[key]
            if len(tokens & other) / len(tokens | other) >= self.threshold:
                return True
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-size", type=int, default=20000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--near-threshold", type=float, default=.85)
    args = parser.parse_args()
    if args.train_size % 5 or args.val_size % 5:
        parser.error("Sizes must be divisible by five")
    if not 0 < args.near_threshold <= 1:
        parser.error("near-threshold must lie in (0, 1]")
    out = ROOT / "dataset/finetuning"
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite existing splits in {out}")
    test_files = sorted((ROOT / "dataset").glob("*subset*.csv"))
    reserved = set()
    index = NearIndex(args.near_threshold)
    for path in test_files:
        for text in pd.read_csv(path).narrative.map(normalize):
            reserved.add(text_hash(text))
            index.add(text)
    raw = pd.read_csv(ROOT / "dataset/complaints.csv", dtype=str)
    raw = raw.dropna(subset=["product", "narrative"])
    raw = raw[raw["product"].isin(CATEGORIES)].copy()
    raw["narrative"] = raw.narrative.str.strip()
    raw = raw[raw.narrative.str.len().between(1, 500)].copy()
    raw["source_id"] = raw.iloc[:, 0]
    raw["narrative"] = raw.narrative.map(normalize)
    raw = raw[raw.narrative.str.len() > 0].rename(columns={"product": "true_label"})
    raw["text_hash"] = raw.narrative.map(text_hash)
    conflicts = raw.groupby("text_hash").true_label.nunique()
    conflicting_hashes = set(conflicts[conflicts > 1].index)
    raw = raw[~raw.text_hash.isin(conflicting_hashes)].drop_duplicates("text_hash")
    exact_test = raw.text_hash.isin(reserved) | raw.narrative.str[:250].str.strip().map(text_hash).isin(reserved)
    excluded_exact = int(exact_test.sum())
    pool = raw[~exact_test].sample(frac=1, random_state=args.seed)
    counts = {c: 0 for c in CATEGORIES}
    val_rows, remaining, near_test = [], [], 0
    # Screen every candidate against test sets first; then make a fixed validation set.
    for _, row in pool.iterrows():
        if index.matches(row.narrative):
            near_test += 1
            continue
        if counts[row.true_label] < args.val_size // 5:
            val_rows.append(row)
            counts[row.true_label] += 1
        else:
            remaining.append(row)
    if any(v != args.val_size // 5 for v in counts.values()):
        raise ValueError(f"Insufficient validation samples: {counts}")
    val = pd.DataFrame(val_rows)
    validation_index = NearIndex(args.near_threshold)
    for text in val.narrative:
        validation_index.add(text)
    val_prefixes = set(val.narrative.str[:250].str.strip().map(text_hash))
    counts = {c: 0 for c in CATEGORIES}
    train_rows, near_val = [], 0
    for row in remaining:
        if counts[row.true_label] >= args.train_size // 5:
            continue
        if text_hash(row.narrative[:250].strip()) in val_prefixes or validation_index.matches(row.narrative):
            near_val += 1
            continue
        train_rows.append(row)
        counts[row.true_label] += 1
    if any(v != args.train_size // 5 for v in counts.values()):
        raise ValueError(f"Insufficient train samples after leakage screening: {counts}")
    train = pd.DataFrame(train_rows)
    assert not set(train.text_hash) & set(val.text_hash)
    assert not set(train.text_hash) & reserved
    assert not set(val.text_hash) & reserved
    # Interleave class-specific random orders so every nested prefix is balanced.
    rng = np.random.default_rng(args.seed)
    order = [rng.permutation(train.index[train.true_label == c]) for c in CATEGORIES]
    train = train.loc[np.column_stack(order).ravel()].reset_index(drop=True)
    cols = ["source_id", "text_hash", "true_label", "narrative"]
    out.mkdir(parents=True, exist_ok=True)
    val[cols].to_csv(out / "validation.csv", index=False)
    sizes = sorted({n for n in [500, 2000, 10000, args.train_size] if n <= args.train_size})
    for n in sizes:
        train.iloc[:n][cols].to_csv(out / f"train_{n}.csv", index=False)
    manifest = {"seed": args.seed, "train_size": len(train), "validation_size": len(val),
        "distribution": "balanced, fixed validation, nested balanced train sizes",
        "max_narrative_chars": 500, "training_seeds_planned": [1, 2, 3],
        "original_test_files_unchanged": [str(p.relative_to(ROOT)) for p in test_files],
        "reserved_unique_normalized_texts": len(reserved),
        "conflicting_label_hashes_excluded": len(conflicting_hashes),
        "exact_and_250_prefix_test_records_excluded": excluded_exact,
        "approx_near_test_records_excluded": near_test, "approx_near_validation_records_excluded": near_val,
        "near_duplicate_method": "128-permutation MinHash LSH + exact word-3-shingle Jaccard",
        "near_duplicate_threshold": args.near_threshold,
        "near_duplicate_limit": "LSH retrieval is approximate; undetected paraphrases may remain",
        "counts_train": train.true_label.value_counts().to_dict(),
        "counts_validation": val.true_label.value_counts().to_dict(),
        "source_sha256": hashlib.sha256((ROOT / "dataset/complaints.csv").read_bytes()).hexdigest(),
        "sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in test_files + sorted(out.glob("*.csv"))}}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
