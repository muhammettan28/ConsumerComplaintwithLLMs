"""Summarise supervised baselines and pair them with saved zero-shot runs.

Keep training seeds separate: three model fits evaluated on the same 3,000
records are not 9,000 independent test observations.
"""
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests

from sadhana_audit import ROOT, matrix, metrics


def main():
    root = ROOT / "results/sadhana"
    cases = []
    for n in [500, 2000, 10000, 20000]:
        paths = root / "classical"
        if (paths / f"preds_tfidf_svc_n{n}_1_subset_balanced_500.csv").exists():
            cases.append(("tfidf_linear_svc", n, 42, paths, f"preds_tfidf_svc_n{n}_"))
    for seed in [1, 2, 3]:
        paths = root / f"distilbert/n20000/seed{seed}"
        if (paths / "metrics.csv").exists():
            cases.append(("distilbert", 20000, seed, paths, "preds_distilbert_"))
    summaries, tests = [], []
    for model, n, train_seed, path, prefix in cases:
        for condition in ["balanced", "imbalanced"]:
            parts = []
            for subset in [1, 2, 3]:
                df = pd.read_csv(path / f"{prefix}{subset}_subset_{condition}_500.csv")
                df["subset"] = subset
                parts.append(df)
            pooled = pd.concat(parts, ignore_index=True)
            summaries.append({"model": model, "train_n": n, "training_seed": train_seed,
                "condition": condition, "test_n": len(pooled),
                **metrics(matrix(pooled.true_label, pooled.predicted_label)),
                "mean_serial_latency_s": float(pooled.latency_seconds.mean())})
            for reference in ["gemma4_12b", "qwen3_14b", "phi4_14b"]:
                legacy = []
                for subset in [1, 2, 3]:
                    df = pd.read_csv(ROOT / f"preds/preds_{reference}_{subset}_subset_{condition}_500.csv")
                    df["subset"] = subset
                    legacy.append(df)
                paired = pooled.merge(pd.concat(legacy), on=["subset", "index"],
                    suffixes=("_baseline", "_llm"), validate="one_to_one")
                assert paired.true_label_baseline.equals(paired.true_label_llm)
                a = paired.predicted_label_baseline.eq(paired.true_label_baseline)
                b = paired.predicted_label_llm.eq(paired.true_label_llm)
                n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
                tests.append({"baseline": model, "train_n": n, "training_seed": train_seed,
                    "condition": condition, "zero_shot_model": reference,
                    "only_baseline_correct": n10, "only_llm_correct": n01,
                    "delta_accuracy": float(a.mean()-b.mean()),
                    "mcnemar_p": binomtest(n10, n10+n01, .5).pvalue if n10+n01 else 1.0})
    results, paired = pd.DataFrame(summaries), pd.DataFrame(tests)
    paired["mcnemar_p_holm"] = multipletests(paired.mcnemar_p, method="holm")[1]
    results.to_csv(root / "supervised_baseline_summary.csv", index=False)
    paired.to_csv(root / "supervised_vs_zero_shot_mcnemar.csv", index=False)
    print(results[["model", "train_n", "training_seed", "condition", "f1_weighted", "mean_serial_latency_s"]].to_string(index=False))


if __name__ == "__main__":
    main()
