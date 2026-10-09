"""Read-only audit of saved splits; never constructs splits or runs a model.

Writes a separate evidence report. MinHash retrieval remains approximate,
even when all retrieved candidates are checked with exact Jaccard similarity.
"""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import pandas as pd

if __package__:
    from .prepare_finetuning_data import NearIndex
else:
    from prepare_finetuning_data import NearIndex
if __package__:
    from .sadhana_audit import ROOT, normalize, text_hash
else:
    from sadhana_audit import ROOT, normalize, text_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((ROOT / "dataset/finetuning/manifest.json").read_text())
    report = {"checked_at": datetime.now().astimezone().isoformat(),
              "scope": "Saved data only; no training, inference or split changes",
              "sha256_match": {}, "splits": {}, "overlaps": {},
              "test_clusters": {}, "length_pairs": {}}
    for name, expected in manifest["sha256"].items():
        report["sha256_match"][name] = hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected
    report["source_sha256_match"] = hashlib.sha256((ROOT / "dataset/complaints.csv").read_bytes()).hexdigest() == manifest["source_sha256"]
    frames = {p.stem: pd.read_csv(p, dtype=str, keep_default_na=False)
              for p in sorted((ROOT / "dataset").glob("*subset*.csv"))}
    for name in ["train_500", "train_2000", "train_10000", "train_20000", "validation"]:
        frames[name] = pd.read_csv(ROOT / f"dataset/finetuning/{name}.csv", dtype=str, keep_default_na=False)
    texts = {name: set(df.narrative.map(normalize)) for name, df in frames.items()}
    prefixes = {name: {s[:250].strip() for s in values} for name, values in texts.items()}
    test_names = [name for name in frames if "subset" in name]
    test_texts = set().union(*(texts[name] for name in test_names))
    test_prefixes = {s[:250].strip() for s in test_texts}
    raw = pd.read_csv(ROOT / "dataset/complaints.csv", dtype=str, keep_default_na=False)
    raw_by_id = raw.set_index(raw.columns[0])
    assert raw_by_id.index.is_unique
    for name, frame in frames.items():
        entry = {"n": len(frame), "counts": frame.true_label.value_counts().to_dict(),
                 "unique_normalized_texts": len(texts[name])}
        if "source_id" in frame:
            original = raw_by_id.loc[frame.source_id]
            entry.update({"stored_text_hashes_match": frame.text_hash.equals(frame.narrative.map(text_hash)),
                          "source_texts_match": original.narrative.map(normalize).tolist() == frame.narrative.map(normalize).tolist(),
                          "source_labels_match": original["product"].tolist() == frame.true_label.tolist(),
                          "unique_source_ids": frame.source_id.nunique()})
        report["splits"][name] = entry
    for name in ["train_500", "train_2000", "train_10000", "train_20000", "validation"]:
        entry = {"test_exact": len(texts[name] & test_texts),
                 "test_250_prefix": len(prefixes[name] & test_prefixes)}
        if name != "validation":
            entry.update({"validation_exact": len(texts[name] & texts["validation"]),
                          "validation_250_prefix": len(prefixes[name] & prefixes["validation"]),
                          "validation_source_id": len(set(frames[name].source_id) & set(frames["validation"].source_id))})
        report["overlaps"][name] = entry
    for condition in ["balanced", "imbalanced"]:
        for length in [250, 500]:
            group = pd.concat([frames[f"{s}_subset_{condition}_{length}"] for s in [1, 2, 3]])
            labels = group.groupby(group.narrative.map(text_hash)).true_label.nunique()
            report["test_clusters"][f"{condition}_{length}"] = {
                "n": len(group), "normalized_text_clusters": len(labels),
                "conflicting_label_clusters": int(labels.gt(1).sum())}
        for seed in [1, 2, 3]:
            a, b = (frames[f"{seed}_subset_{condition}_{n}"] for n in [250, 500])
            report["length_pairs"][f"{condition}_{seed}"] = {
                "same_labels_order": a.true_label.equals(b.true_label),
                "same_truncated_texts_order": a.narrative.equals(b.narrative.str[:250].str.strip())}
    a = set().union(*(texts[f"{s}_subset_balanced_500"] for s in [1, 2, 3]))
    b = set().union(*(texts[f"{s}_subset_imbalanced_500"] for s in [1, 2, 3]))
    report["test_AB_500"] = {"shared_normalized_texts": len(a & b), "union": len(a | b)}
    report["nested_training_subsets"] = all(frames[f"train_{n}"].equals(frames["train_20000"].iloc[:n].reset_index(drop=True)) for n in [500, 2000, 10000])
    print("Exact/prefix, source identity and manifest checks completed", flush=True)
    test_index = NearIndex(manifest["near_duplicate_threshold"])
    for text in sorted(test_texts):
        test_index.add(text)
    val_index = NearIndex(manifest["near_duplicate_threshold"])
    for text in sorted(texts["validation"]):
        val_index.add(text)
    print("Near-duplicate indices ready; checking saved training and validation", flush=True)
    report["near_duplicate_recheck"] = {
        "method": manifest["near_duplicate_method"],
        "threshold": manifest["near_duplicate_threshold"],
        "limit": "Approximate retrieval; not proof of absence of all near duplicates or semantic paraphrases",
        "train_20000_to_test_matches": sum(test_index.matches(t) for t in sorted(texts["train_20000"])),
        "validation_to_test_matches": sum(test_index.matches(t) for t in sorted(texts["validation"])),
        "train_20000_to_validation_matches": sum(val_index.matches(t) for t in sorted(texts["train_20000"]))}
    report["passed"] = (all(report["sha256_match"].values()) and report["source_sha256_match"]
        and all(not any(v.values()) for v in report["overlaps"].values())
        and report["nested_training_subsets"]
        and all(all(v.values()) for v in report["length_pairs"].values())
        and all(v.get("stored_text_hashes_match", True) and v.get("source_texts_match", True)
                and v.get("source_labels_match", True) for v in report["splits"].values())
        and all(v == 0 for k, v in report["near_duplicate_recheck"].items() if k.endswith("matches")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({"passed": report["passed"], "overlaps": report["overlaps"],
                      "near_duplicate_recheck": report["near_duplicate_recheck"],
                      "report": str(args.output)}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
