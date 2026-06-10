"""
preprocess_subsets.py
─────────────────────
Çalıştır: python3 preprocess_subsets.py

Tasarım:
  • 500-char subsets → master: 500-char filtreli, dedup'lanmış korpustan
    örtüşmeyen 3 balanced + 3 imbalanced subset.
  • 250-char subsets → AYNI kayıtlar, narrative'ler 250 karakterde kesilmiş.
    (Kontrollü ablation: içerik sabit, bağlam uzunluğu değişiyor.)

Çıktılar (12 CSV):
  dataset/{1,2,3}_subset_balanced_500.csv
  dataset/{1,2,3}_subset_imbalanced_500.csv
  dataset/{1,2,3}_subset_balanced_250.csv   ← 500 olanların kısaltılmışı
  dataset/{1,2,3}_subset_imbalanced_250.csv ← 500 olanların kısaltılmışı

  figures/narrative_length_distribution.png
"""

import glob, os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Sabitler ──────────────────────────────────────────────────────────────────
CATEGORIES = [
    "retail_banking", "credit_reporting", "credit_card",
    "mortgages_and_loans", "debt_collection",
]
MAX_LEN        = 500   # master filtre
TRUNC_LEN      = 250   # 250-char versiyonlar için kırpma uzunluğu
N_PER_CLASS    = 200   # balanced: her kategoriden
N_TOTAL_IMB    = 1000  # imbalanced: toplam subset büyüklüğü
N_SUBSETS      = 3
MASTER_SEED    = 42
SRC            = "dataset/complaints.csv"
OUT_DIR        = "dataset"
FIG_DIR        = "figures"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)


# ── 1. Ham veri yükle ─────────────────────────────────────────────────────────
print("Veri yükleniyor…")
raw = pd.read_csv(SRC)
raw = raw.dropna(subset=["product", "narrative"])
raw = raw[raw["product"].isin(CATEGORIES)].copy()
raw["narrative"] = raw["narrative"].str.strip()
raw = raw[raw["narrative"].str.len() > 0]
raw["nar_len"] = raw["narrative"].str.len()
print(f"  Ham kayıt (5 kategori, boş yok): {len(raw):,}")


# ── 2. Uzunluk histogramı ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
clipped = raw["nar_len"].clip(upper=2000)
ax.hist(clipped, bins=80, color="#90CAF9", edgecolor="white", alpha=0.85)
for limit, color, ls, lbl in [
    (250, "#E53935", "--", f"250-char truncation point"),
    (500, "#43A047", "-",  f"500-char filter ({(raw['nar_len']<=500).mean()*100:.1f}%)"),
]:
    ax.axvline(limit, color=color, linestyle=ls, linewidth=1.8, label=lbl)
ax.set_xlabel("Narrative Length (characters)", fontsize=10)
ax.set_ylabel("Number of Complaints", fontsize=10)
ax.set_title("Narrative Length Distribution — Full Corpus", fontsize=11)
ax.legend(fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.4)

ax = axes[1]
data_by_cat = [raw.loc[raw["product"] == c, "nar_len"].clip(upper=2000).values
               for c in CATEGORIES]
bp = ax.boxplot(data_by_cat, patch_artist=True,
                medianprops=dict(color="black", linewidth=1.5))
for patch, color in zip(bp["boxes"],
                        ["#42A5F5","#66BB6A","#FFA726","#AB47BC","#EF5350"]):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax.set_xticklabels(["Retail\nBanking","Credit\nReporting","Credit\nCard",
                    "Mortgages\n& Loans","Debt\nCollection"], fontsize=8.5)
ax.axhline(250, color="#E53935", linestyle="--", linewidth=1.5, label="250-char truncation")
ax.axhline(500, color="#43A047", linestyle="-",  linewidth=1.5, label="500-char filter")
ax.set_ylabel("Narrative Length (characters, capped at 2000)", fontsize=10)
ax.set_title("Narrative Length by Category", fontsize=11)
ax.legend(fontsize=9); ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()
hist_path = os.path.join(FIG_DIR, "narrative_length_distribution.png")
fig.savefig(hist_path, dpi=150)
plt.close(fig)
print(f"  Histogram: {hist_path}")

print("\nFiltre karşılaştırması (ham veri):")
print(f"  {'Eşik':>5}  {'Kayıt':>8}  {'%':>6}  {'Retail(min)':>12}  {'CreditRep(max)':>15}")
for lim in [250, 350, 500, 750]:
    s  = raw[raw["nar_len"] <= lim]
    vc = s["product"].value_counts()
    print(f"  {lim:>5}  {len(s):>8,}  {len(s)/len(raw)*100:>5.1f}%"
          f"  {vc.get('retail_banking',0):>12,}  {vc.get('credit_reporting',0):>15,}")


# ── 3. Master filtre + deduplication (500-char) ───────────────────────────────
print(f"\n{'='*62}")
filtered = raw[raw["nar_len"] <= MAX_LEN].copy()
before   = len(filtered)
filtered = filtered.drop_duplicates(subset=["narrative"]).reset_index(drop=True)
after    = len(filtered)

# Metin normalleştirme
filtered["narrative"] = (filtered["narrative"]
    .str.lower()
    .str.replace(r"[^a-z0-9\s]", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip())
filtered = filtered.rename(columns={"product": "true_label"})[["true_label","narrative"]]

print(f"Master pool (≤{MAX_LEN} chars, dedup): {before:,} → {after:,} kayıt "
      f"(-{before-after:,} duplicate)")
print("Kategori dağılımı:")
for cat, cnt in filtered["true_label"].value_counts().items():
    print(f"  {cat:28s}: {cnt:,}")
print(f"{'='*62}")

# Karıştır (sabit seed)
filtered = filtered.sample(frac=1, random_state=MASTER_SEED).reset_index(drop=True)
cat_idx  = {cat: filtered.index[filtered["true_label"] == cat].tolist()
            for cat in CATEGORIES}


# ── 4. Overlap kontrol fonksiyonu ─────────────────────────────────────────────
def overlap_check(files, label):
    dfs = [pd.read_csv(fp) for fp in files]
    ok  = True
    for i in range(len(dfs)):
        for j in range(i+1, len(dfs)):
            m = pd.merge(dfs[i], dfs[j], on=["true_label","narrative"])
            if len(m):
                print(f"  UYARI {label} S{i+1}↔S{j+1}: {len(m)} ortak!"); ok=False
    if ok:
        print(f"  Örtüşme [{label}]: YOK ✓")


# ── 5. 500-char master subsets ────────────────────────────────────────────────
print(f"\nBalanced subsets ({N_PER_CLASS}/kat × 5 = {N_PER_CLASS*5}, 500-char):")
for cat, idxs in cat_idx.items():
    if len(idxs) < N_PER_CLASS * N_SUBSETS:
        raise ValueError(f"Yetersiz: {cat}")

bal500_files = []
for s in range(N_SUBSETS):
    rows  = [filtered.loc[cat_idx[cat][s*N_PER_CLASS:(s+1)*N_PER_CLASS]]
             for cat in CATEGORIES]
    sub   = pd.concat(rows).sample(frac=1, random_state=MASTER_SEED+s).reset_index(drop=True)
    fname = os.path.join(OUT_DIR, f"{s+1}_subset_balanced_500.csv")
    sub.to_csv(fname, index=False)
    bal500_files.append(fname)
    print(f"  Subset {s+1}: {os.path.basename(fname)}  n={len(sub)}")
overlap_check(bal500_files, "Balanced-500")

# Doğal dağılım oranları
nat   = filtered["true_label"].value_counts(normalize=True)
pcn   = (nat * N_TOTAL_IMB).round().astype(int)
pcn[pcn.idxmax()] += N_TOTAL_IMB - pcn.sum()

print(f"\nImbalanced subsets (N={N_TOTAL_IMB}, 500-char):")
print("  Hedef dağılım:")
for cat, n in pcn.items():
    print(f"    {cat:28s}: {n}  ({n/N_TOTAL_IMB*100:.1f}%)")

imb500_files = []
for s in range(N_SUBSETS):
    rows = []
    for cat in CATEGORIES:
        n   = pcn[cat]
        idx = cat_idx[cat][s*n:(s+1)*n]
        if len(idx) < n:
            raise ValueError(f"Yetersiz (imb): {cat} subset {s+1}")
        rows.append(filtered.loc[idx])
    sub   = pd.concat(rows).sample(frac=1, random_state=MASTER_SEED+s+10).reset_index(drop=True)
    fname = os.path.join(OUT_DIR, f"{s+1}_subset_imbalanced_500.csv")
    sub.to_csv(fname, index=False)
    imb500_files.append(fname)
    print(f"  Subset {s+1}: {os.path.basename(fname)}  n={len(sub)}")
overlap_check(imb500_files, "Imbalanced-500")


# ── 6. 250-char subsets → AYNI kayıtlar, narrative'ler 250'de kırpılmış ───────
print(f"\n250-char versiyonlar (500-char subsetlerin kırpılmış hali):")
bal250_files = []
for fname500 in bal500_files:
    df       = pd.read_csv(fname500)
    df["narrative"] = df["narrative"].str[:TRUNC_LEN].str.strip()
    df = df[df["narrative"].str.len() > 0].reset_index(drop=True)
    fname250 = fname500.replace("_500.csv", "_250.csv")
    df.to_csv(fname250, index=False)
    bal250_files.append(fname250)
    print(f"  {os.path.basename(fname500)} → {os.path.basename(fname250)}  "
          f"n={len(df)}  median_len={df['narrative'].str.len().median():.0f}")

imb250_files = []
for fname500 in imb500_files:
    df       = pd.read_csv(fname500)
    df["narrative"] = df["narrative"].str[:TRUNC_LEN].str.strip()
    df = df[df["narrative"].str.len() > 0].reset_index(drop=True)
    fname250 = fname500.replace("_500.csv", "_250.csv")
    df.to_csv(fname250, index=False)
    imb250_files.append(fname250)
    print(f"  {os.path.basename(fname500)} → {os.path.basename(fname250)}  "
          f"n={len(df)}  median_len={df['narrative'].str.len().median():.0f}")

# 250-char subsetlerde kayıt sayısı 500-char ile aynı olmalı
# (kırpma sonrası boş kalan olmadıkça)
print("\n  250-char subsetlerde overlap kontrolü (narrative farklı ama kayıt aynı):")
print("  → Kontrol atlandı: 250-char aynı kayıtların kırpılmış hali, overlap beklenir.")


# ── 7. Özet ───────────────────────────────────────────────────────────────────
print(f"\n{'='*62}")
print("Oluşturulan dosyalar:")
for fp in sorted(glob.glob(os.path.join(OUT_DIR, "*_subset_*.csv"))):
    rows = sum(1 for _ in open(fp)) - 1
    print(f"  {os.path.basename(fp):50s}  {rows} satır")
print(f"\nNot: *_250.csv dosyaları, *_500.csv ile AYNI kayıtları içerir;")
print(f"     narrative'ler ilk {TRUNC_LEN} karakterde kırpılmıştır.")
print("\nTamamlandı.")
