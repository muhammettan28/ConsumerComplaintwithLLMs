"""
preprocess.py — Yerel makinede çalışır (conda env: kangal)
Girdi : complaints.csv
Çıktı : balanced_1000.csv  (200 satır × 5 kategori)
"""

import re
import pandas as pd
from collections import Counter

SEED = 42
INPUT_FILE = "complaints.csv"
OUTPUT_FILE = "balanced_1000.csv"
MAX_CHARS = 500
N_PER_CLASS = 200

CATEGORIES = [
    "retail_banking",
    "credit_reporting",
    "credit_card",
    "mortgages_and_loans",
    "debt_collection",
]


def standardize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # özel karakterleri boşluğa çevir
    text = re.sub(r"\s+", " ", text).strip()    # fazla boşlukları temizle
    return text


def main():
    print(f"Okununuyor: {INPUT_FILE}")
    df = pd.read_csv(INPUT_FILE, index_col=0)
    print(f"Ham satır sayısı: {len(df):,}")

    # 1. Boş product/narrative satırlarını düşür
    before = len(df)
    df = df.dropna(subset=["product", "narrative"])
    df = df[df["product"].str.strip() != ""]
    df = df[df["narrative"].str.strip() != ""]
    print(f"Boş alan temizliği sonrası: {len(df):,}  (düşürülen: {before - len(df):,})")

    # 2. Yalnızca 5 hedef kategoriyi tut
    df = df[df["product"].isin(CATEGORIES)].copy()
    print(f"5 kategori filtresi sonrası: {len(df):,}")

    # Ön dağılım
    print("\n--- Ham Dağılım (5 kategori) ---")
    for cat, cnt in sorted(Counter(df["product"]).items()):
        print(f"  {cat:<30} {cnt:>7,}  ({cnt/len(df)*100:.1f}%)")

    # 3. Narrative uzunluğunu filtrele (> 500 karakter olanları çıkar)
    df["narrative"] = df["narrative"].astype(str)
    before = len(df)
    df = df[df["narrative"].str.len() <= MAX_CHARS].copy()
    print(f"\n>500 karakter filtresi sonrası: {len(df):,}  (düşürülen: {before - len(df):,})")

    # 4. Narrative metni standardize et
    df["narrative"] = df["narrative"].apply(standardize)
    # Standardizasyon sonrası boş kalanları çıkar
    df = df[df["narrative"].str.len() > 0]

    # 5. Dengeli örnekleme: her kategoriden 200 satır
    print(f"\nDengeli örnekleme: {N_PER_CLASS} × {len(CATEGORIES)} = "
          f"{N_PER_CLASS * len(CATEGORIES)} satır  (seed={SEED})")

    samples = []
    for cat in CATEGORIES:
        pool = df[df["product"] == cat]
        if len(pool) < N_PER_CLASS:
            raise ValueError(
                f"'{cat}' kategorisinde yalnızca {len(pool)} satır var, "
                f"{N_PER_CLASS} gerekli."
            )
        samples.append(pool.sample(n=N_PER_CLASS, random_state=SEED))

    balanced = pd.concat(samples, ignore_index=True)
    balanced = balanced.rename(columns={"product": "true_label"})[["narrative", "true_label"]]
    balanced = balanced.sample(frac=1, random_state=SEED).reset_index(drop=True)

    # Doğrulama
    print("\n--- Dengeli Dağılım ---")
    vc = balanced["true_label"].value_counts()
    for cat in CATEGORIES:
        print(f"  {cat:<30} {vc[cat]:>4}")
    assert all(vc == N_PER_CLASS), "Her kategoride tam 200 satır olmalı!"
    assert len(balanced) == N_PER_CLASS * len(CATEGORIES)

    balanced.to_csv(OUTPUT_FILE, index=True)
    print(f"\nKaydedildi: {OUTPUT_FILE}  ({len(balanced)} satır)")
    print("Hazır — kümeye gönderilebilir: balanced_1000.csv")


if __name__ == "__main__":
    main()
