"""
analyze_seeds.py — Multi-seed analysis for the revised paper.

Reads all 132 prediction files from preds/ (3 seeds × 2 sizes × 2 conditions × 11 models),
computes per-run metrics, aggregates across seeds (mean ± std), and produces:
  - results/table_balanced.csv
  - results/table_imbalanced.csv
  - results/table_size_sensitivity.csv
  - results/table_vs_reference.csv
  - figures/v2/f1_balanced_errbar.png
  - figures/v2/f1_imbalanced_errbar.png
  - figures/v2/comparison_A_vs_B.png
  - figures/v2/vs_reference.png
  - figures/v2/size_sensitivity.png
  - figures/v2/confusion_{model}_{condition}_{size}.png  (averaged confusion matrix)

Usage:
  python3 analyze_seeds.py
"""

import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score,
)

# ── Constants ──────────────────────────────────────────────────────────────
CATEGORIES = [
    "retail_banking", "credit_reporting", "credit_card",
    "mortgages_and_loans", "debt_collection",
]
CAT_SHORT = {
    "retail_banking":      "Retail\nBanking",
    "credit_reporting":    "Credit\nReporting",
    "credit_card":         "Credit\nCard",
    "mortgages_and_loans": "Mortgages\n& Loans",
    "debt_collection":     "Debt\nCollection",
}
MODELS = [
    "deepseek-r1_8b", "deepseek-r1_14b", "gemma2_9b", "gemma4_12b",
    "gpt-oss_20b", "llama3.1_8b", "mistral_7b", "mistral-nemo_12b",
    "phi4_14b", "qwen3_8b", "qwen3_14b",
]
MODEL_PAT = "|".join(re.escape(m) for m in MODELS)
FNAME_PAT = re.compile(
    rf"preds_({MODEL_PAT})_(\d+)_subset_(balanced|imbalanced)_(\d+)\.csv"
)
REASONING = {"deepseek-r1_14b", "deepseek-r1_8b", "qwen3_8b", "qwen3_14b", "gpt-oss_20b"}
COLOR_R = "#1565C0"
COLOR_S = "#90CAF9"

REFERENCE = {
    "o1 (paid API)":                {"acc": 0.777, "f1": 0.7764, "cost": "$24.24"},
    "GPT-4.5 (paid API)":           {"acc": 0.774, "f1": 0.7763, "cost": "$12.32"},
    "Claude 3.7 Sonnet (paid API)": {"acc": 0.765, "f1": 0.7626, "cost": "$0.64"},
    "Gemini 2.0 Flash (paid API)":  {"acc": 0.758, "f1": 0.7586, "cost": "$0.04"},
    "GPT-4o (paid API)":            {"acc": 0.736, "f1": 0.7358, "cost": "$0.45"},
}

PREDS_DIR   = "preds"
RESULTS_DIR = "results"
FIGS_DIR    = os.path.join("figures", "v2")


# ── Helpers ────────────────────────────────────────────────────────────────

def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["true_label"]      = df["true_label"].str.lower().str.strip()
    df["predicted_label"] = df["predicted_label"].str.lower().str.strip()
    df.loc[~df["predicted_label"].isin(CATEGORIES), "predicted_label"] = "off_label"
    return df


def compute_metrics(df: pd.DataFrame) -> dict:
    y_true = df["true_label"]
    y_pred = df["predicted_label"]
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, labels=CATEGORIES, average="weighted", zero_division=0)
    rec  = recall_score(   y_true, y_pred, labels=CATEGORIES, average="weighted", zero_division=0)
    f1w  = f1_score(       y_true, y_pred, labels=CATEGORIES, average="weighted", zero_division=0)
    f1m  = f1_score(       y_true, y_pred, labels=CATEGORIES, average="macro",    zero_division=0)
    per_class = f1_score(  y_true, y_pred, labels=CATEGORIES, average=None,       zero_division=0)
    lat = df["latency_seconds"].astype(float)
    return {
        "accuracy":       acc,
        "precision":      prec,
        "recall":         rec,
        "f1_weighted":    f1w,
        "f1_macro":       f1m,
        "mean_latency_s": lat.mean(),
        "off_label":      int((y_pred == "off_label").sum()),
        "n":              len(df),
        **{f"f1_{cat}": v for cat, v in zip(CATEGORIES, per_class)},
    }


def mcolor(model: str) -> str:
    return COLOR_R if model in REASONING else COLOR_S


# ── Load all files ─────────────────────────────────────────────────────────

def load_all() -> pd.DataFrame:
    records = []
    for path in sorted(glob.glob(os.path.join(PREDS_DIR, "*.csv"))):
        m = FNAME_PAT.match(os.path.basename(path))
        if not m:
            continue
        model, seed, condition, size = m.groups()
        df = load(path)
        metrics = compute_metrics(df)
        records.append({
            "model": model,
            "seed": int(seed),
            "condition": condition,
            "size": int(size),
            "path": path,
            "df": df,
            **metrics,
        })
    return pd.DataFrame(records)


# ── Aggregate across seeds ─────────────────────────────────────────────────

METRIC_COLS = ["accuracy", "f1_weighted", "f1_macro", "precision", "recall",
               "mean_latency_s", "off_label"] + [f"f1_{c}" for c in CATEGORIES]


def aggregate(raw: pd.DataFrame, condition: str, size: int) -> pd.DataFrame:
    sub = raw[(raw["condition"] == condition) & (raw["size"] == size)]
    rows = []
    for model in MODELS:
        g = sub[sub["model"] == model]
        if g.empty:
            continue
        row = {"model": model, "n_seeds": len(g)}
        for col in METRIC_COLS:
            vals = g[col].values.astype(float)
            row[f"{col}_mean"] = vals.mean()
            row[f"{col}_std"]  = vals.std(ddof=1) if len(vals) > 1 else 0.0
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("accuracy_mean", ascending=False).reset_index(drop=True)
    df["type"] = df["model"].apply(lambda m: "Reasoning" if m in REASONING else "Standard")
    return df


# ── Plots ──────────────────────────────────────────────────────────────────

def plot_f1_errbar(agg: pd.DataFrame, title: str, fname: str) -> None:
    df = agg.sort_values("f1_weighted_mean")
    colors = [mcolor(m) for m in df["model"]]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(df["model"], df["f1_weighted_mean"],
            xerr=df["f1_weighted_std"], color=colors,
            edgecolor="white", height=0.6, capsize=4, error_kw={"elinewidth": 1.2})
    for _, row in df.iterrows():
        ax.text(row["f1_weighted_mean"] + row["f1_weighted_std"] + 0.005,
                list(df["model"]).index(row["model"]),
                f'{row["f1_weighted_mean"]:.4f} ± {row["f1_weighted_std"]:.4f}',
                va="center", fontsize=7.5)
    ax.set_xlabel("Weighted F1-Score (mean ± std, 3 seeds)", fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(0.40, 1.02)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    leg = [mpatches.Patch(color=COLOR_R, label="Reasoning"),
           mpatches.Patch(color=COLOR_S, label="Standard")]
    ax.legend(handles=leg, loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  -> {fname}")


def plot_comparison_ab(agg_a: pd.DataFrame, agg_b: pd.DataFrame, fname: str) -> None:
    models = MODELS
    a_vals = [agg_a.loc[agg_a.model == m, "accuracy_mean"].values[0] for m in models]
    b_vals = [agg_b.loc[agg_b.model == m, "accuracy_mean"].values[0] for m in models]
    a_std  = [agg_a.loc[agg_a.model == m, "accuracy_std"].values[0]  for m in models]
    b_std  = [agg_b.loc[agg_b.model == m, "accuracy_std"].values[0]  for m in models]

    x = np.arange(len(models))
    w = 0.35
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - w/2, a_vals, w, yerr=a_std, label="Experiment A — Balanced",
           color="#42A5F5", edgecolor="white", capsize=4)
    ax.bar(x + w/2, b_vals, w, yerr=b_std, label="Experiment B — Imbalanced",
           color="#EF5350", edgecolor="white", capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("Accuracy (mean ± std, 3 seeds)", fontsize=10)
    ax.set_title("Experiment A vs. B — Balanced vs. Imbalanced (N=500)", fontsize=11)
    ax.axhline(0.777, color="gray", linestyle="--", linewidth=1.2,
               label="Reference best — o1 (0.777, paid API)")
    ax.set_ylim(0.40, 0.95)
    ax.legend(fontsize=8.5)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  -> {fname}")


def plot_size_sensitivity(raw: pd.DataFrame, condition: str, fname: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    for model in MODELS:
        sub = raw[(raw["condition"] == condition) & (raw["model"] == model)]
        for size, marker, ls in [(250, "o", "--"), (500, "s", "-")]:
            g = sub[sub["size"] == size]["f1_weighted"].values
            if len(g) == 0:
                continue
            color = mcolor(model)
            ax.errorbar(model, g.mean(), yerr=g.std(ddof=1) if len(g) > 1 else 0,
                        fmt=marker, color=color, linestyle=ls,
                        capsize=4, markersize=7,
                        label=f"N={size}" if model == MODELS[0] else "")
    ax.set_ylabel("Weighted F1 (mean ± std)", fontsize=10)
    ax.set_title(f"Narrative Length Sensitivity — {condition.capitalize()} Condition", fontsize=11)
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    leg_size = [
        mpatches.Patch(color="gray", label="N=250 (●)"),
        mpatches.Patch(color="gray", label="N=500 (■)"),
        mpatches.Patch(color=COLOR_R, label="Reasoning"),
        mpatches.Patch(color=COLOR_S, label="Standard"),
    ]
    ax.legend(handles=leg_size, fontsize=8.5)
    plt.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  -> {fname}")


def plot_vs_reference(agg_a: pd.DataFrame, fname: str) -> None:
    ref_rows = [
        {"model": k, "acc": v["acc"], "type": "Paid API"}
        for k, v in REFERENCE.items()
    ]
    our_rows = [
        {"model": row["model"], "acc": row["accuracy_mean"],
         "acc_std": row["accuracy_std"],
         "type": "Reasoning (open-source)" if row["model"] in REASONING else "Standard (open-source)"}
        for _, row in agg_a.iterrows()
    ]
    ref_df = pd.DataFrame(ref_rows)
    our_df = pd.DataFrame(our_rows)
    combined = pd.concat(
        [ref_df.assign(acc_std=0), our_df],
        ignore_index=True
    ).sort_values("acc")

    palette = {
        "Reasoning (open-source)": COLOR_R,
        "Standard (open-source)":  COLOR_S,
        "Paid API":                "#E57373",
    }
    colors = [palette[t] for t in combined["type"]]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(combined["model"], combined["acc"],
            xerr=combined["acc_std"], color=colors,
            edgecolor="white", height=0.6, capsize=3)
    for _, row in combined.iterrows():
        std_str = f" ± {row['acc_std']:.4f}" if row["acc_std"] > 0 else ""
        ax.text(row["acc"] + row["acc_std"] + 0.004,
                list(combined["model"]).index(row["model"]),
                f'{row["acc"]:.4f}{std_str}', va="center", fontsize=7.5)
    ax.set_xlabel("Accuracy (mean ± std for open-source; point estimate for paid API)", fontsize=9)
    ax.set_title("Open-Source Models vs. Paid API — Zero-Shot Accuracy", fontsize=11)
    ax.set_xlim(0.44, 0.95)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    leg = [mpatches.Patch(color=c, label=l) for l, c in palette.items()]
    ax.legend(handles=leg, fontsize=9, loc="lower right")
    plt.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  -> {fname}")


def plot_confusion_averaged(raw: pd.DataFrame, model: str,
                            condition: str, size: int, fname: str) -> None:
    sub = raw[(raw["model"] == model) & (raw["condition"] == condition) & (raw["size"] == size)]
    cms = []
    for _, row in sub.iterrows():
        df = row["df"]
        y_true = df["true_label"]
        y_pred = df["predicted_label"].replace("off_label", CATEGORIES[0])
        cm = confusion_matrix(y_true, y_pred, labels=CATEGORIES).astype(float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(1)
        cms.append(cm_norm)
    avg_cm = np.mean(cms, axis=0)

    import seaborn as sns
    short = [CAT_SHORT[c] for c in CATEGORIES]
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(avg_cm, annot=True, fmt=".2f", cmap="Blues",
                linewidths=0.4, ax=ax, vmin=0, vmax=1,
                xticklabels=short, yticklabels=short,
                annot_kws={"size": 9})
    ax.set_title(f"{model} — {condition} N={size} (avg. {len(sub)} seeds)", fontsize=10, pad=8)
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("True", fontsize=9)
    plt.xticks(rotation=30, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    print(f"  -> {fname}")


# ── Table builders ─────────────────────────────────────────────────────────

def build_summary_table(agg: pd.DataFrame) -> pd.DataFrame:
    cols = {
        "model":                "Model",
        "type":                 "Type",
        "accuracy_mean":        "Acc (mean)",
        "accuracy_std":         "Acc (std)",
        "f1_weighted_mean":     "F1w (mean)",
        "f1_weighted_std":      "F1w (std)",
        "f1_macro_mean":        "F1m (mean)",
        "f1_macro_std":         "F1m (std)",
        "mean_latency_s_mean":  "Lat (s)",
        "off_label_mean":       "Off-label",
    }
    out = agg[[c for c in cols if c in agg.columns]].rename(columns=cols)
    for c in ["Acc (mean)", "Acc (std)", "F1w (mean)", "F1w (std)",
              "F1m (mean)", "F1m (std)", "Lat (s)"]:
        if c in out.columns:
            out[c] = out[c].round(4)
    out["Off-label"] = out["Off-label"].round(1)
    return out


def build_per_class_table(agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in agg.iterrows():
        r = {"Model": row["model"], "Type": row["type"]}
        for cat in CATEGORIES:
            col = f"f1_{cat}_mean"
            r[cat] = round(row[col], 4) if col in row else None
        rows.append(r)
    return pd.DataFrame(rows)


def build_vs_reference_table(agg_a: pd.DataFrame) -> pd.DataFrame:
    ref_rows = [
        {"Model": k, "Source": "Reference (paid API)",
         "Acc": v["acc"], "Acc std": "—",
         "F1w": v["f1"], "F1w std": "—", "Cost/1k": v["cost"]}
        for k, v in REFERENCE.items()
    ]
    our_rows = [
        {"Model": row["model"], "Source": "Ours (open-source)",
         "Acc":     round(row["accuracy_mean"],    4),
         "Acc std": round(row["accuracy_std"],     4),
         "F1w":     round(row["f1_weighted_mean"], 4),
         "F1w std": round(row["f1_weighted_std"],  4),
         "Cost/1k": "$0.00"}
        for _, row in agg_a.iterrows()
    ]
    combined = pd.DataFrame(ref_rows + our_rows)
    combined["Acc_sort"] = pd.to_numeric(combined["Acc"], errors="coerce")
    return combined.sort_values("Acc_sort", ascending=False).drop(columns="Acc_sort").reset_index(drop=True)


def build_size_sensitivity_table(raw: pd.DataFrame, condition: str) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for size in [250, 500]:
            sub = raw[(raw["model"] == model) & (raw["condition"] == condition) & (raw["size"] == size)]
            if sub.empty:
                continue
            rows.append({
                "Model":    model,
                "N":        size,
                "Acc mean": round(sub["accuracy"].mean(),    4),
                "Acc std":  round(sub["accuracy"].std(ddof=1) if len(sub) > 1 else 0, 4),
                "F1w mean": round(sub["f1_weighted"].mean(), 4),
                "F1w std":  round(sub["f1_weighted"].std(ddof=1) if len(sub) > 1 else 0, 4),
            })
    return pd.DataFrame(rows)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGS_DIR, exist_ok=True)

    print("Loading prediction files...")
    raw = load_all()
    print(f"  {len(raw)} files loaded.")
    print(f"  Models:     {sorted(raw['model'].unique())}")
    print(f"  Seeds:      {sorted(raw['seed'].unique())}")
    print(f"  Conditions: {sorted(raw['condition'].unique())}")
    print(f"  Sizes:      {sorted(raw['size'].unique())}")

    # ── Aggregated tables ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  Experiment A — Balanced, N=500")
    print("="*60)
    agg_a500 = aggregate(raw, "balanced", 500)
    tbl_a = build_summary_table(agg_a500)
    print(tbl_a.to_string(index=False))
    tbl_a.to_csv(os.path.join(RESULTS_DIR, "table_balanced_500.csv"), index=False)

    print("\n" + "="*60)
    print("  Experiment B — Imbalanced, N=500")
    print("="*60)
    agg_b500 = aggregate(raw, "imbalanced", 500)
    tbl_b = build_summary_table(agg_b500)
    print(tbl_b.to_string(index=False))
    tbl_b.to_csv(os.path.join(RESULTS_DIR, "table_imbalanced_500.csv"), index=False)

    print("\n" + "="*60)
    print("  Per-Class F1 — Balanced N=500")
    print("="*60)
    pc_a = build_per_class_table(agg_a500)
    print(pc_a.to_string(index=False))
    pc_a.to_csv(os.path.join(RESULTS_DIR, "table_perclass_balanced_500.csv"), index=False)

    print("\n" + "="*60)
    print("  vs. Reference Paper")
    print("="*60)
    ref_tbl = build_vs_reference_table(agg_a500)
    print(ref_tbl.to_string(index=False))
    ref_tbl.to_csv(os.path.join(RESULTS_DIR, "table_vs_reference.csv"), index=False)

    print("\n" + "="*60)
    print("  Sample Size Sensitivity — Balanced")
    print("="*60)
    size_tbl = build_size_sensitivity_table(raw, "balanced")
    print(size_tbl.to_string(index=False))
    size_tbl.to_csv(os.path.join(RESULTS_DIR, "table_size_sensitivity.csv"), index=False)

    # ── Figures ────────────────────────────────────────────────────────────
    print("\nGenerating figures...")
    plot_f1_errbar(
        agg_a500,
        title="Weighted F1-Score — Balanced Experiment (N=500, mean ± std, 3 seeds)",
        fname=os.path.join(FIGS_DIR, "f1_balanced_errbar.png"),
    )
    plot_f1_errbar(
        agg_b500,
        title="Weighted F1-Score — Imbalanced Experiment (N=500, mean ± std, 3 seeds)",
        fname=os.path.join(FIGS_DIR, "f1_imbalanced_errbar.png"),
    )
    plot_comparison_ab(
        agg_a500, agg_b500,
        fname=os.path.join(FIGS_DIR, "comparison_A_vs_B.png"),
    )
    plot_vs_reference(
        agg_a500,
        fname=os.path.join(FIGS_DIR, "vs_reference.png"),
    )
    plot_size_sensitivity(
        raw, "balanced",
        fname=os.path.join(FIGS_DIR, "size_sensitivity_balanced.png"),
    )
    plot_size_sensitivity(
        raw, "imbalanced",
        fname=os.path.join(FIGS_DIR, "size_sensitivity_imbalanced.png"),
    )

    # Averaged confusion matrices for top models
    for model in ["gpt-oss_20b", "phi4_14b", "deepseek-r1_14b"]:
        for cond in ["balanced", "imbalanced"]:
            plot_confusion_averaged(
                raw, model, cond, 500,
                fname=os.path.join(FIGS_DIR, f"confusion_{model}_{cond}.png"),
            )

    print("\nDone. Results -> results/   Figures -> figures/v2/")


if __name__ == "__main__":
    main()
