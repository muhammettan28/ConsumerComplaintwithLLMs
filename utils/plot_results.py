"""
analyze.py — Run locally (conda env: kangal)
Analyzes balanced and imbalanced experiment results, produces publication-ready figures.

Usage:
  python3 analyze.py              # both experiments + comparison
  python3 analyze.py --exp balanced
  python3 analyze.py --exp imbalanced
"""

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score,
)

# ── Constants ─────────────────────────────────────────────────────────────
CATEGORIES = [
    "retail_banking", "credit_reporting", "credit_card",
    "mortgages_and_loans", "debt_collection",
]
CAT_SHORT = {
    "retail_banking":      "Retail Banking",
    "credit_reporting":    "Credit Reporting",
    "credit_card":         "Credit Card",
    "mortgages_and_loans": "Mortgages & Loans",
    "debt_collection":     "Debt Collection",
}
REASONING_MODELS = {"deepseek-r1_14b", "deepseek-r1_8b", "qwen3_8b", "gpt-oss_20b"}
COLOR_REASONING  = "#1565C0"
COLOR_STANDARD   = "#90CAF9"

# Reference paper results (Roumeliotis et al. 2025, Electronics)
REFERENCE = {
    "o1 (paid API)":               0.777,
    "gpt-4.5 (paid API)":          0.774,
    "claude-3.7-sonnet (paid API)": 0.765,
    "gemini-2.0-flash (paid API)":  0.758,
    "gpt-4o (paid API)":            0.736,
}

EXPERIMENTS = {
    "balanced": {
        "dir":    "balanced_preds",
        "prefix": r"^preds_(?:pilot_)?",
        "label":  "Experiment A — Balanced (200 samples/class)",
    },
    "imbalanced": {
        "dir":    "imbalanced_preds",
        "prefix": r"^preds_imb_",
        "label":  "Experiment B — Imbalanced (original distribution)",
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────

def clean_name(path: str, prefix_pat: str) -> str:
    base = os.path.basename(path)
    name = re.sub(prefix_pat, "", base)
    return re.sub(r"\.csv$", "", name)


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["true_label"]      = df["true_label"].str.lower().str.strip()
    df["predicted_label"] = df["predicted_label"].str.lower().str.strip()
    # Mark anything outside the 5 categories as off_label
    df.loc[~df["predicted_label"].isin(CATEGORIES), "predicted_label"] = "off_label"
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    """
    All metrics use the full dataset (n=1000).
    off_label predictions count as wrong for accuracy, precision, recall, F1.
    labels=CATEGORIES in sklearn ensures off_label acts as a misclassification,
    not a legitimate sixth class.
    """
    y_true = df["true_label"]
    y_pred = df["predicted_label"]

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=CATEGORIES,
                           average="weighted", zero_division=0)
    rec  = recall_score(   y_true, y_pred, labels=CATEGORIES,
                           average="weighted", zero_division=0)
    f1   = f1_score(       y_true, y_pred, labels=CATEGORIES,
                           average="weighted", zero_division=0)

    lat = df["latency_seconds"].astype(float)
    return {
        "accuracy":        round(acc,  4),
        "precision":       round(prec, 4),
        "recall":          round(rec,  4),
        "f1":              round(f1,   4),
        "mean_latency_s":  round(lat.mean(), 2),
        "total_latency_s": round(lat.sum(),  1),
        "off_label":       int((y_pred == "off_label").sum()),
        "errors":          int((df.get("status", pd.Series()) == "error").sum()),
        "n":               len(df),
    }


def model_color(name: str) -> str:
    return COLOR_REASONING if name in REASONING_MODELS else COLOR_STANDARD


# ── Plot functions ────────────────────────────────────────────────────────

def plot_confusion(df: pd.DataFrame, model: str, out_dir: str,
                   exp_label: str = "") -> None:
    y_true = df["true_label"]
    y_pred = df["predicted_label"].replace("off_label", "other")
    labels = CATEGORIES + (["other"] if "other" in y_pred.values else [])
    short  = [CAT_SHORT.get(l, "Other") for l in labels]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df   = pd.DataFrame(cm, index=short, columns=short)
    cm_norm = cm_df.div(cm_df.sum(axis=1).replace(0, 1), axis=0).round(2)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                linewidths=0.4, ax=ax, vmin=0, vmax=1,
                annot_kws={"size": 9})
    subtitle = f"\n{exp_label}" if exp_label else ""
    ax.set_title(f"{model}{subtitle}", fontsize=11, pad=10)
    ax.set_xlabel("Predicted Label", fontsize=9)
    ax.set_ylabel("True Label", fontsize=9)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    path = os.path.join(out_dir, f"confusion_{model}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    -> {path}")


def plot_f1_bar(summary: pd.DataFrame, out_dir: str, title: str) -> None:
    df = summary.sort_values("f1", ascending=True)
    colors = [model_color(m) for m in df["model"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(df["model"], df["f1"], color=colors,
                   edgecolor="white", height=0.6)
    for bar, val in zip(bars, df["f1"]):
        ax.text(bar.get_width() + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)
    ax.set_xlabel("Weighted F1-Score", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0.45, 0.95)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    leg = [mpatches.Patch(color=COLOR_REASONING, label="Reasoning"),
           mpatches.Patch(color=COLOR_STANDARD,  label="Standard")]
    ax.legend(handles=leg, loc="lower right", fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "f1_bar.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    -> {path}")


def plot_latency_acc(summary: pd.DataFrame, out_dir: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    for _, row in summary.iterrows():
        color  = model_color(row["model"])
        marker = "D" if row["model"] in REASONING_MODELS else "o"
        ax.scatter(row["mean_latency_s"], row["accuracy"],
                   s=120, color=color, zorder=3, marker=marker)
        ax.annotate(row["model"],
                    (row["mean_latency_s"], row["accuracy"]),
                    textcoords="offset points", xytext=(7, 4), fontsize=8)
    ax.set_xlabel("Mean Latency (s / prediction)", fontsize=10)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    leg = [mpatches.Patch(color=COLOR_REASONING, label="Reasoning  (◆)"),
           mpatches.Patch(color=COLOR_STANDARD,  label="Standard   (●)")]
    ax.legend(handles=leg, fontsize=9)
    plt.tight_layout()
    path = os.path.join(out_dir, "latency_vs_accuracy.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    -> {path}")


def plot_comparison_ab(sum_a: pd.DataFrame, sum_b: pd.DataFrame,
                       out_dir: str) -> None:
    """Side-by-side bar chart: Experiment A vs B for each model."""
    idx_a = sum_a.set_index("model")["accuracy"]
    idx_b = sum_b.set_index("model")["accuracy"]
    models = sorted(set(idx_a.index) & set(idx_b.index))
    a_vals = [idx_a[m] for m in models]
    b_vals = [idx_b[m] for m in models]

    x = range(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    bars_a = ax.bar([i - w/2 for i in x], a_vals, w,
                    label="Experiment A — Balanced",   color="#42A5F5", edgecolor="white")
    bars_b = ax.bar([i + w/2 for i in x], b_vals, w,
                    label="Experiment B — Imbalanced", color="#EF5350", edgecolor="white")

    for bar in bars_a:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                f"{bar.get_height():.3f}", ha="center", fontsize=7.5)
    for bar in bars_b:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                f"{bar.get_height():.3f}", ha="center", fontsize=7.5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title("Experiment A vs. Experiment B — Balanced vs. Imbalanced Data",
                 fontsize=11)
    ax.set_ylim(0.45, 0.92)
    ax.axhline(0.777, color="gray", linestyle="--", linewidth=1.2,
               label="Reference best — o1 (0.777, paid API)")
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    path = os.path.join(out_dir, "comparison_A_vs_B.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    -> {path}")


def plot_vs_reference(sum_a: pd.DataFrame, out_dir: str) -> None:
    """Our Experiment A models + reference paper paid-API models."""
    our = sum_a[["model", "accuracy"]].copy()
    our["type"] = our["model"].apply(
        lambda m: "Reasoning (open-source)"
        if m in REASONING_MODELS else "Standard (open-source)"
    )
    ref = pd.DataFrame([
        {"model": k, "accuracy": v, "type": "Paid API (reference paper)"}
        for k, v in REFERENCE.items()
    ])
    combined = pd.concat([our, ref], ignore_index=True).sort_values("accuracy")

    palette = {
        "Reasoning (open-source)":  COLOR_REASONING,
        "Standard (open-source)":   COLOR_STANDARD,
        "Paid API (reference paper)":"#E57373",
    }
    colors = [palette[t] for t in combined["type"]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(combined["model"], combined["accuracy"],
                   color=colors, edgecolor="white", height=0.6)
    for bar, val in zip(bars, combined["accuracy"]):
        ax.text(bar.get_width() + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8)
    ax.set_xlabel("Accuracy", fontsize=10)
    ax.set_title("Open-Source Models vs. Paid API — Zero-Shot Classification Accuracy",
                 fontsize=11)
    ax.set_xlim(0.48, 0.86)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    leg = [mpatches.Patch(color=c, label=l) for l, c in palette.items()]
    ax.legend(handles=leg, fontsize=9, loc="lower right")
    plt.tight_layout()
    path = os.path.join(out_dir, "vs_reference.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"    -> {path}")


# ── Runner ────────────────────────────────────────────────────────────────

def run_experiment(exp_key: str) -> pd.DataFrame:
    cfg     = EXPERIMENTS[exp_key]
    src     = cfg["dir"]
    pfix    = cfg["prefix"]
    label   = cfg["label"]
    out_dir = os.path.join("figures", exp_key)
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(src, "*.csv")))
    if not files:
        print(f"  [WARNING] No CSV files found in {src}/")
        return pd.DataFrame()

    print(f"\n{'='*60}")
    print(f"  {label}  ({len(files)} models)")
    print(f"{'='*60}")
    print(f"  {'Model':<26}  {'Acc':>6}  {'Prec':>6}  {'Rec':>6}  "
          f"{'F1':>6}  {'Lat(s)':>6}  {'Off':>4}")
    print(f"  {'-'*70}")

    records = []
    for path in files:
        model = clean_name(path, pfix)
        df    = load(path)
        m     = compute_metrics(df)
        m["model"] = model
        records.append(m)
        print(f"  {model:<26}  {m['accuracy']:>6.4f}  {m['precision']:>6.4f}  "
              f"{m['recall']:>6.4f}  {m['f1']:>6.4f}  "
              f"{m['mean_latency_s']:>6.2f}  {m['off_label']:>4}")
        plot_confusion(df, model, out_dir, exp_label=label)

    cols = ["model", "accuracy", "precision", "recall", "f1",
            "mean_latency_s", "total_latency_s", "off_label", "errors", "n"]
    summary = (pd.DataFrame(records)[cols]
               .sort_values("accuracy", ascending=False)
               .reset_index(drop=True))

    csv_path = os.path.join(out_dir, "results_table.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    plot_f1_bar(summary, out_dir,
                title=f"F1-Score Comparison — {label}")
    plot_latency_acc(summary, out_dir,
                     title=f"Accuracy vs. Latency — {label}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", choices=["balanced", "imbalanced", "both"],
                        default="both")
    args = parser.parse_args()

    os.makedirs("figures", exist_ok=True)
    sum_a = sum_b = pd.DataFrame()

    if args.exp in ("balanced", "both"):
        sum_a = run_experiment("balanced")

    if args.exp in ("imbalanced", "both"):
        sum_b = run_experiment("imbalanced")

    if args.exp == "both" and not sum_a.empty and not sum_b.empty:
        cmp_dir = os.path.join("figures", "comparison")
        os.makedirs(cmp_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print("  Comparison figures")
        print(f"{'='*60}")
        plot_comparison_ab(sum_a, sum_b, cmp_dir)
        plot_vs_reference(sum_a, cmp_dir)

    print("\nAll figures ready -> see figures/ folder.")


if __name__ == "__main__":
    main()
