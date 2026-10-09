"""
utils/generate_dataset_figures.py
Produces two figures for the paper:
  1. figures/v2/char_length_histogram.png  — narrative character length distribution (3 seeds)
  2. figures/v2/class_distribution.png     — balanced vs imbalanced class counts
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

FIGS_DIR = ROOT / "figures/v2"
os.makedirs(FIGS_DIR, exist_ok=True)

CATEGORIES = [
    "retail_banking", "credit_reporting", "credit_card",
    "mortgages_and_loans", "debt_collection",
]
CAT_LABELS = {
    "retail_banking":      "Retail\nBanking",
    "credit_reporting":    "Credit\nReporting",
    "credit_card":         "Credit\nCard",
    "mortgages_and_loans": "Mortgages\n& Loans",
    "debt_collection":     "Debt\nCollection",
}

# ── Figure 1: Character length histogram ───────────────────────────────────

print("Generating char_length_histogram.png ...")

RAW_CSV = ROOT / "dataset/complaints.csv"
df_raw = pd.read_csv(RAW_CSV)
df_raw = df_raw.dropna(subset=["narrative", "product"])
df_raw = df_raw[df_raw["product"].isin(CATEGORIES)]
df_raw["char_len"] = df_raw["narrative"].str.len()

# Use full unfiltered dataset to show complete narrative length range
# Clip at 1000 chars for display; anything beyond goes to the last bin
MAX_DISPLAY = 1000
df_raw["char_len_clipped"] = df_raw["narrative"].str.len().clip(upper=MAX_DISPLAY)

# Three random seeds from the full dataset (N=5000 each) to show consistency
rng = np.random.default_rng(0)
idx_all = np.arange(len(df_raw))
seed_lengths = {}
for seed in [1, 2, 3]:
    rng_s = np.random.default_rng(seed)
    chosen = rng_s.choice(idx_all, size=min(5000, len(df_raw)), replace=False)
    seed_lengths[seed] = df_raw["char_len_clipped"].values[chosen]

fig, ax = plt.subplots(figsize=(7, 3.5))

colors = ["#1565C0", "#42A5F5", "#90CAF9"]
bins = np.arange(0, MAX_DISPLAY + 25, 25)

for (seed, lengths), color in zip(seed_lengths.items(), colors):
    ax.hist(lengths, bins=bins, alpha=0.55, color=color,
            label=f"Sample {seed}", edgecolor="white", linewidth=0.3)

ax.axvline(x=250, color="red",     linestyle="--", linewidth=1.5, label="250-char limit")
ax.axvline(x=500, color="darkred", linestyle="-",  linewidth=1.5, label="500-char limit")

# shade excluded region
ax.axvspan(500, MAX_DISPLAY, alpha=0.07, color="grey", label="Excluded (>500 chars)")

ax.set_xlabel("Narrative Length (characters)", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.set_title("CFPB Complaint Narrative Length Distribution", fontsize=11)
ax.legend(fontsize=9, loc="upper right")
ax.set_xlim(0, MAX_DISPLAY)
ax.set_xticks(np.arange(0, MAX_DISPLAY + 1, 100))
plt.tight_layout()
out1 = os.path.join(FIGS_DIR, "char_length_histogram.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> {out1}")

# ── Figure 2: Class distribution — raw vs balanced vs imbalanced ───────────

print("Generating class_distribution.png ...")

# Raw full dataset distribution (after length filter)
df_raw["char_len"] = df_raw["narrative"].str.len()
df_filtered = df_raw[df_raw["char_len"] <= 500]
raw_counts = df_filtered["product"].value_counts()
raw_pct = (raw_counts / raw_counts.sum() * 100).reindex(CATEGORIES)

# Balanced N=500 (seed 1 representative)
bal = pd.read_csv(ROOT / "dataset/1_subset_balanced_500.csv")
bal_counts = bal["true_label"].value_counts().reindex(CATEGORIES)
bal_pct = (bal_counts / bal_counts.sum() * 100)

# Imbalanced N=500 (seed 1 representative)
imb = pd.read_csv(ROOT / "dataset/1_subset_imbalanced_500.csv")
imb_counts = imb["true_label"].value_counts().reindex(CATEGORIES)
imb_pct = (imb_counts / imb_counts.sum() * 100)

x = np.arange(len(CATEGORIES))
width = 0.28

fig, ax = plt.subplots(figsize=(8, 4))

bars1 = ax.bar(x - width, raw_pct.values, width, label="Full dataset (N=103,915)",
               color="#B0BEC5", edgecolor="grey", linewidth=0.5)
bars2 = ax.bar(x,          imb_pct.values, width, label="Imbalanced subset (N=1,000)",
               color="#42A5F5", edgecolor="#1565C0", linewidth=0.5)
bars3 = ax.bar(x + width,  bal_pct.values, width, label="Balanced subset (N=1,000)",
               color="#1565C0", edgecolor="#0D47A1", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels([CAT_LABELS[c] for c in CATEGORIES], fontsize=9)
ax.set_ylabel("Class Proportion (%)", fontsize=11)
ax.set_title("Class Distribution: Full Dataset vs Experimental Subsets", fontsize=11)
ax.legend(fontsize=9)
ax.set_ylim(0, 75)

# annotate dominant class
for bar in bars1:
    h = bar.get_height()
    if h > 40:
        ax.text(bar.get_x() + bar.get_width() / 2, h + 1,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=7.5, color="grey")

plt.tight_layout()
out2 = os.path.join(FIGS_DIR, "class_distribution.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  -> {out2}")

print("Done.")
