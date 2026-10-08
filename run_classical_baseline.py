"""Fit TF-IDF + LinearSVC on the preserved-test training protocol.

Choose C on validation weighted F1 only. Never tune on the existing test sets.
"""
import argparse
import json
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

from sadhana_audit import CATEGORIES, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[500, 2000, 10000, 20000])
    args = parser.parse_args()
    out = ROOT / "results/sadhana/classical"
    out.mkdir(parents=True, exist_ok=True)
    val = pd.read_csv(ROOT / "dataset/finetuning/validation.csv")
    summaries = []
    for n in args.sizes:
        train = pd.read_csv(ROOT / f"dataset/finetuning/train_{n}.csv")
        start = time.perf_counter()
        candidates = []
        for c in [.1, 1., 10.]:
            model = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                max_features=100000, sublinear_tf=True), LinearSVC(C=c, random_state=42))
            model.fit(train.narrative, train.true_label)
            score = f1_score(val.true_label, model.predict(val.narrative), labels=CATEGORIES,
                             average="weighted", zero_division=0)
            candidates.append((score, c, model))
        best_score, c, model = max(candidates, key=lambda v: v[0])
        train_seconds = time.perf_counter() - start
        for path in sorted((ROOT / "dataset").glob("*subset*.csv")):
            test = pd.read_csv(path)
            # Measure warm, serial end-to-end CPU inference, including vectorisation.
            model.predict(test.narrative.iloc[:1])
            preds, latencies = [], []
            for text in test.narrative:
                begin = time.perf_counter()
                preds.append(model.predict([text])[0])
                latencies.append(time.perf_counter() - begin)
            df = pd.DataFrame({"index": range(len(test)), "true_label": test.true_label,
                "predicted_label": preds, "status": "ok", "latency_seconds": latencies})
            df.to_csv(out / f"preds_tfidf_svc_n{n}_{path.stem}.csv", index=False)
            summaries.append({"model": "tfidf_linear_svc", "train_n": n, "test_dataset": path.stem,
                "selected_C": c, "validation_f1": best_score, "tuning_and_training_seconds": train_seconds,
                "accuracy": accuracy_score(test.true_label, preds),
                "f1_weighted": f1_score(test.true_label, preds, labels=CATEGORIES, average="weighted", zero_division=0),
                "f1_macro": f1_score(test.true_label, preds, labels=CATEGORIES, average="macro", zero_division=0),
                "mean_latency_s_cpu": sum(latencies)/len(latencies), "off_label": 0})
        print(f"n={n} C={c} validation weighted F1={best_score:.4f}", flush=True)
    pd.DataFrame(summaries).to_csv(out / "metrics.csv", index=False)
    (out / "method.json").write_text(json.dumps({
        "hyperparameter_selection": "C in [0.1, 1, 10], highest validation weighted F1; ties choose first",
        "features": "word TF-IDF 1/2-grams, min_df=2, max_features=100000, sublinear_tf=True",
        "hardware": "CPU; serial latency includes vectorization; no GPU or output generation",
        "seed": 42, "training_split": "same fixed balanced nested splits intended for QLoRA",
        "limits": "A classical baseline; does not replace a supervised transformer baseline"}, indent=2))
    print(pd.DataFrame(summaries).query("test_dataset.str.endswith('balanced_500') and not test_dataset.str.contains('imbalanced')", engine="python")[["train_n", "test_dataset", "f1_weighted"]].to_string(index=False))


if __name__ == "__main__":
    main()
