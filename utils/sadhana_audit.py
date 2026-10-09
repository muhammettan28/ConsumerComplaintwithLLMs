"""Audit saved experiments without changing their data or recorded predictions.

python utils/sadhana_audit.py --bootstrap 5000
Intervals condition on these test sets, not model-training or prompt uncertainty.
"""
import argparse
import hashlib
import json
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = ["retail_banking", "credit_reporting", "credit_card",
              "mortgages_and_loans", "debt_collection"]
PATTERN = re.compile(r"preds_(.+)_([123])_subset_(balanced|imbalanced)_(250|500)\.csv$")


def normalize(text):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", str(text).lower())).strip()


def text_hash(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()


def strict_label(text):
    # Use JSON decoding, ignore mentions of category names in prose or reasoning.
    decoder = json.JSONDecoder()
    matches = []
    for start in re.finditer(r"\{", text):
        try:
            value, _ = decoder.raw_decode(text[start.start():])
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(value, dict) and isinstance(value.get("product"), str):
            matches.append(value["product"].lower().strip())
    return matches[-1] if matches else "parse_error"


def matrix(true, pred, weights=None):
    mapping = {c: i for i, c in enumerate(CATEGORIES)}
    a = np.array([mapping[c] for c in true])
    b = np.array([mapping.get(c, 5) for c in pred])
    return np.bincount(a * 6 + b, weights=weights, minlength=30).reshape(5, 6)


def metrics(cm):
    support = cm.sum(axis=1)
    tp = np.diag(cm[:, :5])
    predicted = cm[:, :5].sum(axis=0)
    precision = np.divide(tp, predicted, out=np.zeros(5), where=predicted > 0)
    recall = np.divide(tp, support, out=np.zeros(5), where=support > 0)
    f1 = np.divide(2 * tp, support + predicted, out=np.zeros(5), where=support + predicted > 0)
    return {"accuracy": float(tp.sum() / support.sum()),
            "f1_weighted": float(f1 @ support / support.sum()),
            "f1_macro": float(f1.mean()),
            **{f"precision_{c}": float(v) for c, v in zip(CATEGORIES, precision)},
            **{f"recall_{c}": float(v) for c, v in zip(CATEGORIES, recall)}}


def bootstrap(df, replicates, seed):
    # Resample full normalized narrative clusters: repeated texts share a weight.
    # Stratify by true class, preserving the empirical class mix and cluster count.
    keys = df["text_hash"]
    clusters, codes = np.unique(keys, return_inverse=True)
    group_label = df.groupby("text_hash")["true_label"].agg(lambda x: x.iloc[0])
    if (df.groupby("text_hash")["true_label"].nunique() > 1).any():
        raise ValueError("Conflicting labels within a bootstrap text cluster")
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in ["accuracy", "f1_weighted", "f1_macro"]}
    strata = [np.flatnonzero(group_label.reindex(clusters).to_numpy() == c) for c in CATEGORIES]
    label_ids = {c: i for i, c in enumerate(CATEGORIES)}
    cells = np.array([label_ids[t] * 6 + label_ids.get(p, 5)
                      for t, p in zip(df.true_label, df.predicted_label)])
    cluster_matrices = np.zeros((len(clusters), 30))
    np.add.at(cluster_matrices, (codes, cells), 1)
    for _ in range(replicates):
        counts = np.zeros(len(clusters), dtype=int)
        for ids in strata:
            counts[ids] = rng.multinomial(len(ids), np.full(len(ids), 1 / len(ids)))
        values = metrics((counts @ cluster_matrices).reshape(5, 6))
        for k in draws:
            draws[k].append(values[k])
    return {f"{k}_ci_{side}": float(v) for k, vals in draws.items()
            for side, v in zip(["low", "high"], np.quantile(vals, [.025, .975]))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    if args.bootstrap < 100:
        parser.error("Use at least 100 bootstrap replicates")
    out = ROOT / "results/sadhana"
    out.mkdir(parents=True, exist_ok=True)
    datasets = {p.stem: pd.read_csv(p, keep_default_na=False)
                for p in sorted((ROOT / "dataset").glob("*subset*.csv"))}
    data_rows, overlaps, length_checks = [], [], []
    for name, df in datasets.items():
        assert len(df) == 1000, name
        assert df.true_label.isin(CATEGORIES).all(), name
        data_rows.append({"dataset": name, "n": len(df),
                          "unique_texts": df.narrative.map(text_hash).nunique(),
                          "max_chars": int(df.narrative.str.len().max()),
                          **df.true_label.value_counts().to_dict()})
    for a, b in combinations(datasets, 2):
        if a.rsplit("_", 1)[1] == b.rsplit("_", 1)[1]:
            common = set(datasets[a].narrative.map(text_hash)) & set(datasets[b].narrative.map(text_hash))
            overlaps.append({"dataset_a": a, "dataset_b": b, "shared_normalized_texts": len(common)})
    for condition in ["balanced", "imbalanced"]:
        for seed in [1, 2, 3]:
            a, b = (datasets[f"{seed}_subset_{condition}_{size}"] for size in [250, 500])
            length_checks.append({"condition": condition, "seed": seed,
                "same_labels_and_order": bool(a.true_label.equals(b.true_label)),
                "all_are_250_char_truncations": bool(a.narrative.equals(b.narrative.str[:250].str.strip()))})
    all_runs, examples, parse_rows, file_checks = [], [], [], []
    for path in sorted((ROOT / "preds").glob("*.csv")):
        match = PATTERN.fullmatch(path.name)
        if not match:
            continue
        model, seed, condition, size = match.groups()
        df = pd.read_csv(path, keep_default_na=False)
        source = datasets[f"{seed}_subset_{condition}_{size}"]
        ids = pd.to_numeric(df["index"], errors="raise").to_numpy(dtype=int)
        if len(df) != len(source) or len(set(ids)) != len(ids) or set(ids) != set(range(len(source))):
            raise ValueError(f"Missing/duplicate/unexpected row indices: {path}")
        source = source.iloc[ids].reset_index(drop=True)
        if not source.true_label.equals(df.true_label):
            raise ValueError(f"Prediction/source label mismatch: {path}")
        df["text_hash"] = source.narrative.map(text_hash)
        for k, v in {"model": model, "seed": int(seed), "condition": condition, "size": int(size)}.items():
            df[k] = v
        file_checks.append({"file": str(path.relative_to(ROOT)), "n": len(df),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "labels_aligned": True,
            "identity_limit": "predictions store only positional indices and labels; no original narrative hash"})
        strict = df.raw_response.map(strict_label)
        original = df.predicted_label.where(df.predicted_label.isin(CATEGORIES), "invalid")
        parsed = strict.where(strict.isin(CATEGORIES), "invalid")
        parse_rows.append({"model": model, "seed": int(seed), "condition": condition, "size": int(size),
            "n": len(df), "legacy_invalid": int((original == "invalid").sum()),
            "strict_invalid": int((parsed == "invalid").sum()),
            "changed_by_strict_parser": int((original != parsed).sum()),
            "raw_empty": int(df.raw_response.eq("").sum()),
            "raw_at_2000_char_limit": int(df.raw_response.str.len().eq(2000).sum()),
            "legacy_accuracy": float((original == df.true_label).mean()),
            "strict_accuracy": float((parsed == df.true_label).mean()),
            **{f"status_{k}": int(v) for k, v in df.status.value_counts().items()}})
        for status, group in df[df.status != "ok"].groupby("status"):
            for _, row in group.head(2).iterrows():
                examples.append({"model": model, "seed": seed, "condition": condition, "size": size,
                    "index": row["index"], "true_label": row.true_label, "status": status,
                    "predicted_label": row.predicted_label, "raw_response": row.raw_response,
                    "latency_seconds": row.latency_seconds})
        all_runs.append(df)
    runs = pd.concat(all_runs, ignore_index=True)
    if len(all_runs) != 132:
        raise ValueError(f"Expected 132 files, found {len(all_runs)}")
    estimates, per_run, reweighted, sensitivity = [], [], [], []
    for keys, group in runs.groupby(["model", "condition", "size"]):
        model, condition, size = keys
        for seed, part in group.groupby("seed"):
            per_run.append({"model": model, "condition": condition, "size": size, "seed": int(seed),
                            **metrics(matrix(part.true_label, part.predicted_label))})
        if size == 500:
            values = metrics(matrix(group.true_label, group.predicted_label))
            estimates.append({"model": model, "condition": condition, "size": size, "n": len(group),
                "unique_text_clusters": group.text_hash.nunique(), **values,
                **bootstrap(group, args.bootstrap, 42),
                "mean_latency_s": float(group.latency_seconds.mean()),
                "serial_days_per_million": float(group.latency_seconds.mean() * 1e6 / 86400)})
            if condition == "imbalanced":
                cm = matrix(group.true_label, group.predicted_label)
                weights = 1 / cm.sum(axis=1)
                adjusted = metrics(cm * weights[:, None])
                reweighted.append({"model": model, **{f"observed_{k}": values[k] for k in ["accuracy", "f1_weighted", "f1_macro"]},
                                   **{f"equal_prior_{k}": adjusted[k] for k in ["accuracy", "f1_weighted", "f1_macro"]}})
        if size == 250:
            longer = runs[(runs.model == model) & (runs.condition == condition) & (runs["size"] == 500)]
            paired = group.merge(longer, on=["seed", "index"], suffixes=("_250", "_500"), validate="one_to_one")
            a = paired.predicted_label_250.eq(paired.true_label_250).to_numpy()
            b = paired.predicted_label_500.eq(paired.true_label_500).to_numpy()
            n01, n10 = int((~a & b).sum()), int((a & ~b).sum())
            sensitivity.append({"model": model, "condition": condition, "n": len(a),
                "accuracy_250": float(a.mean()), "accuracy_500": float(b.mean()),
                "delta_accuracy": float(b.mean() - a.mean()), "only_500_correct": n01, "only_250_correct": n10,
                "mcnemar_p": binomtest(n01, n01+n10, .5).pvalue if n01+n10 else 1.0})
    sens = pd.DataFrame(sensitivity)
    sens["mcnemar_p_holm"] = multipletests(sens.mcnemar_p, method="holm")[1]
    outputs = {"dataset_inventory": data_rows, "dataset_overlap": overlaps,
        "length_pairing": length_checks, "prediction_integrity": file_checks,
        "parser_audit": parse_rows, "error_examples": examples, "pooled_bootstrap_ci": estimates,
        "per_run_metrics": per_run, "equal_prior_reweighting": reweighted}
    for name, rows in outputs.items():
        pd.DataFrame(rows).to_csv(out / f"{name}.csv", index=False)
    sens.to_csv(out / "paired_length_sensitivity.csv", index=False)
    (out / "audit_method.json").write_text(json.dumps({
        "bootstrap_replicates": args.bootstrap, "rng_seed": 42,
        "estimator": "pooled metrics, NOT mean of per-seed metrics",
        "bootstrap": "normalized narrative cluster bootstrap stratified by true class",
        "invalid_predictions": "retained as errors in accuracy, recall and F1; extra column in confusion matrix",
        "reweighting": "confusion rows rescaled to equal class prior, then precision/recall/F1 recomputed",
        "mcnemar": "250 vs 500, paired by seed and positional index, Holm correction over 22 tests",
        "limits": ["Intervals condition on the saved evaluation sets and predictions",
                   "Old prediction files omit original text hashes, thinking, done_reason and token counters",
                   "Strict-parser audit uses saved responses, which may be truncated to 2000 characters",
                   "A and B share records: their results are not independent replications"]}, indent=2))
    print(pd.DataFrame(estimates).query("condition == 'balanced'")[["model", "f1_weighted", "f1_weighted_ci_low", "f1_weighted_ci_high"]].sort_values("f1_weighted", ascending=False).to_string(index=False))
    print(f"Wrote audit outputs to {out}")


if __name__ == "__main__":
    main()
