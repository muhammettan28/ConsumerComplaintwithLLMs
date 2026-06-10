# Sonraki Adımlar — ConsumerComplaint Projesi

> Son güncelleme: 2026-06-08 (model listesi güncellendi)

## Genel Strateji — İki Aşamalı Deney Tasarımı

| Deney | Veri | Amaç |
|---|---|---|
| **Deney A** (ana) | `balanced_1000.csv` — 200/kategori × 5 | Referans makaleyle birebir karşılaştırma |
| **Deney B** (katkı) | `imbalanced_1000.csv` — orijinal dağılım (~56% credit_reporting) | Gerçek dünya koşulları; referans makale bunu yapmadı → özgün katkı |

**Neden B mantıklı?**
Referans makale yalnızca dengeli alt küme kullandı. Dengesiz koşullarda modellerin `credit_reporting`'e bias yapıp yapmadığını göstermek özgün ve tartışma bölümü için somut bulgu sağlar. Veri boyutu aynı (1000 satır) → küme maliyeti aynı.

---

## Aşama 0 — Yerel Pilot Test (şu an burada)

- [x] `test_local.py`'yi çalıştır: qwen2.5:3b ile 10 satır, yerel Ollama ✓
  - Doğruluk: 4/10 (3B model, beklenen düşük); off-label ("credit_collection") yakalandı
  - JSON çıkarımı, off-label tespiti, latency kaydı → tümü doğrulandı
- [x] Pipeline uçtan uca çalışıyor — kümeye göndermeye hazır ✓

---

## Aşama 1 — Kümeye Hazırlık

- [ ] Dosyaları kümeye gönder (yerel PC'den):
  ```bash
  scp balanced_1000.csv imbalanced_1000.csv classify.py run_ollama.slurm \
      muhtan@172.16.6.11:/arf/scratch/muhtan/ccc/
  ```
- [x] Login node'da (arf-ui1) modelleri pull et ✓ — **8 model hazır:**

  | Model | Tür | Boyut | Aile |
  |---|---|---|---|
  | `gemma2:9b` | Standart | 5.4 GB | Google |
  | `mistral:7b` | Standart | 4.4 GB | Mistral |
  | `llama3.1:8b` | Standart | 4.9 GB | Meta |
  | `phi4:14b` | Standart | 9.1 GB | Microsoft |
  | `deepseek-r1:8b` | Reasoning | 5.2 GB | DeepSeek |
  | `deepseek-r1:14b` | Reasoning | 9.0 GB | DeepSeek |
  | `qwen3:8b` | Reasoning | 5.2 GB | Alibaba |
  | `gpt-oss:20b` | Reasoning | 13 GB | OpenAI |

  > ⚠️ `gpt-oss:20b` 13 GB → V100 16 GB'a sığar ama yeterince dar; bu model için `--time=04:00:00` kullan.

---

## Aşama 2 — Küme Pilot Testi

- [ ] `run_ollama.slurm` içindeki `python3 classify.py` satırına `--limit 50` ekle (geçici)
- [ ] İlk pilot iş (mistral hızlı ve hafif, iyi başlangıç):
  ```bash
  # ⚠️ ZORUNLU: önce scratch dizinine geç, home'dan sbatch kabul edilmez!
  cd /arf/scratch/muhtan/ccc/
  sbatch --export=MODEL=mistral:7b,OUTFILE=preds_pilot_mistral_7b.csv run_ollama.slurm
  squeue -u muhtan   # durum takibi
  ```
- [ ] `cls-<JOBID>.out` log dosyasını incele, hata yoksa `--limit` satırını kaldır

---

## Aşama 3 — Tam Çalıştırma (1000 satır × tüm modeller)

```bash
# ⚠️ ZORUNLU: önce scratch dizinine geç, home'dan sbatch kabul edilmez!
cd /arf/scratch/muhtan/ccc/

# Standart modeller (2 saatlik limit yeterli)
for m in "gemma2:9b" "mistral:7b" "llama3.1:8b" "phi4:14b"; do
    safe=$(echo $m | tr ':' '_')
    sbatch --export=MODEL=$m,OUTFILE=preds_${safe}.csv run_ollama.slurm
done

# Reasoning modeller — deepseek:8b ve qwen3 için 2 saat yeterli
for m in "deepseek-r1:8b" "deepseek-r1:14b" "qwen3:8b"; do
    safe=$(echo $m | tr ':' '_')
    sbatch --export=MODEL=$m,OUTFILE=preds_${safe}.csv run_ollama.slurm
done

# gpt-oss:20b — 13 GB model, daha uzun süre gerekebilir
sbatch --time=04:00:00 \
    --export=MODEL=gpt-oss:20b,OUTFILE=preds_gpt-oss_20b.csv run_ollama.slurm
```

- [ ] Her iş için `cls-*.out` loglarını gözden geçir
- [ ] Tamamlananları kontrol et: `ls -lh preds_*.csv`
- [ ] 8 dosya bekleniyor: `gemma2_9b`, `mistral_7b`, `llama3.1_8b`, `phi4_14b`, `deepseek-r1_8b`, `deepseek-r1_14b`, `qwen3_8b`, `gpt-oss_20b`

---

## Aşama 4 — Analiz (yerel)

- [ ] Tahmin CSV'lerini kopyala:
  ```bash
  scp muhtan@172.16.6.11:/arf/scratch/muhtan/ccc/preds_*.csv .
  ```
- [ ] `python3 analyze.py`
  - Çıktılar: `results_table.csv`, `figures/confusion_*.png`,
    `figures/accuracy_vs_latency.png`, `figures/f1_comparison.png`
- [ ] `results_table.csv`'yi referans makalenin Tablo 1'iyle karşılaştır

---

## Aşama 4B — Deney B: Dengesiz Veri (özgün katkı)

- [x] `imbalanced_1000.csv` oluşturuldu ✓ — orijinal dağılım korundu:

  | Kategori | Satır | Oran |
  |---|---|---|
  | `credit_reporting` | 636 | %63.6 |
  | `debt_collection` | 141 | %14.1 |
  | `mortgages_and_loans` | 77 | %7.7 |
  | `credit_card` | 75 | %7.5 |
  | `retail_banking` | 71 | %7.1 |

- [x] `run_ollama.slurm` güncellendi ✓ — `INPUT` parametresi eklendi (varsayılan: `balanced_1000.csv`)
- [ ] Kümeye gönder (yerel PC'den, `balanced_1000.csv` ile birlikte):
  ```bash
  scp imbalanced_1000.csv muhtan@172.16.6.11:/arf/scratch/muhtan/ccc/
  ```
- [ ] Aynı 8 model ile Deney B'yi çalıştır:
  ```bash
  # ⚠️ ZORUNLU: önce scratch dizinine geç, home'dan sbatch kabul edilmez!
  cd /arf/scratch/muhtan/ccc/

  for m in "gemma2:9b" "mistral:7b" "llama3.1:8b" "phi4:14b" \
           "deepseek-r1:8b" "deepseek-r1:14b" "qwen3:8b"; do
      safe=$(echo $m | tr ':' '_')
      sbatch --export=MODEL=$m,OUTFILE=preds_imb_${safe}.csv,INPUT=imbalanced_1000.csv run_ollama.slurm
  done
  sbatch --time=04:00:00 \
      --export=MODEL=gpt-oss:20b,OUTFILE=preds_imb_gpt-oss_20b.csv,INPUT=imbalanced_1000.csv run_ollama.slurm
  ```
- [ ] `analyze.py` ile Deney A ve B sonuçlarını karşılaştır
  - Beklenen bulgu: modeller `credit_reporting`'e bias yapar → karışıklık matrisinde görünür
  - Bu bulgu makalenin tartışma bölümünü zenginleştirir

---

## Aşama 5 — Opsiyonel: Quantization Alt Çalışması

- [ ] Aynı modeli farklı quant seviyelerinde çek ve karşılaştır:
  ```bash
  ollama pull qwen2.5:7b-instruct-q4_K_M
  ollama pull qwen2.5:7b-instruct-q8_0
  ollama pull qwen2.5:7b-instruct-fp16   # VRAM yeterliyse
  ```
- [ ] Her quant için ayrı `sbatch` + `analyze.py`'de karşılaştırma grafiği

---

## Aşama 6 — Makale Yazımı

- [ ] Yöntem bölümü: model listesi, Ollama versiyonu, GPU bilgisi (V100 16 GB), pull tarihleri
- [ ] Tablo 1 (model × metrik) + Şekil 1 (karışıklık matrisleri) + Şekil 2 (doğruluk-latency)
- [ ] Tartışma: açık modeller vs ücretli modeller; reasoning modeli farkı; maliyet = $0 argümanı
- [ ] TR Dizin dergi seçimi ve makale gönderimi
