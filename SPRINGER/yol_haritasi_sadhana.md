# Sādhanā Gönderimi İçin Yol Haritası

> **8 Ekim 2026 uygulama notu:** Ortam kuruldu; Aşama 1 düzeltmeleri,
> bootstrap/reweighting/eşleştirilmiş uzunluk analizleri, testlerden ayrılmış
> eğitim bölmesi, TF-IDF/SVM öğrenme eğrisi ve üç seed DistilBERT tamamlandı.
> QLoRA sonuçları başlık/özet/yöntem/sonuç/tartışmaya işlendi; ana PDF 24,
> ek PDF 14 sayfa. Son biçim ve 20 sayfaya kısaltma en sona bırakıldı.
> Üç büyük modelin 500 kayıt/seed 1 QLoRA pilotları tamamlandı;
> Aynı HF/NF4 ayarlarıyla 3.000 dengeli, 500-karakter test kaydında adaptersiz
> → adapterli weighted F1: Gemma 0.7520 → 0.7803, Qwen3 0.7099 → 0.7891,
> Phi-4 0.6883 → 0.7845. Tüm 12 koşul ve seri gecikme tamamlandı;
> 72.000 tahmin + 600 seri ölçüm doğrulandı. 500 eğitim etiketi yanında
> ortak 2.000 validation etiketi kullanıldı; tek adapter eğitim seed'i var.
> Gemma 4 düşünme modu
> ve tarihsel model kimliği hâlâ doğrulama gerektiriyor.
> Yeni bulgular ve kalan işler: [sadhana_durum_raporu.md](sadhana_durum_raporu.md).

**Makale:** Local Consumer Complaint Classification: Zero-Shot, Lightweight Adaptation and Label Budgets
**Hedef dergi:** Sādhanā – Academy Proceedings in Engineering Sciences (Springer, SCIE, JCR Q3)
**Kaynaklar:** Claude geri bildirimi + ChatGPT'nin ince ayar (fine-tuning) önerileri

> **Ana strateji:** Zero-shot kısmı korunur. Makale "zero-shot kıyaslaması"ndan **"zero-shot → QLoRA ince ayar → doğruluk / gecikme / maliyet karşılaştırması"** (deployment stratejisi çalışması) yapısına genişletilir. "Yalnızca mevcut modeller + mevcut veri seti üzerinde kıyaslama" eleştirisini azaltmak hedeflenir.

---

## Aşama 0 — Önce doğrulanması gerekenler (diğer her şeyi etkiler)

Bu maddeler sonuçları değiştirebilir. Düzeltme ve ince ayar deneylerinden **önce** yapılmalı.

- [ ] **gemma4:12b düşünme (thinking) modu kontrolü.** 17.1 s gecikme "standart model" için şüpheli. Ham çıktıları inceleyin; Ollama sürümünüzde `think` parametresi varsa kapalı/açık iki koşulda küçük bir örnekle (örn. 200 kayıt) test edin. Sonuç makalenin ana iddiasını ("reasoning gerekli değil") doğrudan etkiler.
- [ ] **Reasoning modeller için varsayılanları gözden geçirin.** qwen3 ve deepseek-r1'de temperature=0 (greedy) tekrar döngülerine yol açabilir; varsayılan bağlam/çıktı limiti düşünme izini kesmiş olabilir. Kesilen çıktılar off-label sayılıyorsa bunu raporlayın.
- [ ] **llama3.1:8b (67 parse hatası) ve mistral:7b (59 off-label)**: 10–20 ham çıktıyı elle inceleyin. Prompt şablonu / `generate` vs `chat` endpoint sorunu olabilir.
- [ ] **gpt-oss:20b nicemleme formatı**: Q4_K_M mi MXFP4 mü? Tablo I ve metindeki ifadeyi buna göre düzeltin.
- [ ] **gemma4 ve diğer model etiketlerinin model kartlarını** kaydedin (kaynak/sürüm); kaynakçaya eklenecek.

---

## Aşama 1 — Kesin tutarsızlıkların düzeltilmesi

| # | Konum | Sorun | Yapılacak |
|---|-------|-------|-----------|
| 1 | Bölüm V-C | "En büyük artış mistral:7b ve llama3.1:8b" deniyor, aynı paragrafta mistral-nemo +0.094 ile en büyük | Cümleyi düzeltin (nemo en büyük artış) |
| 2 | Bölüm VI-F | llama3.1:8b için "+4.2 pp" | Tablo II/III'e göre **+5.4 pp** (0.578 → 0.632) |
| 3 | Şekil 4 alt yazısı ve metin | gpt-oss:20b "top reasoning model" | En iyi reasoning model **qwen3:14b** (0.750 > 0.749) |
| 4 | Şekil 6 alt yazısı | deepseek-r1:14b "second among reasoning" | Tabloya göre **üçüncü** |
| 5 | Bölüm VI-B | "original eight-model design" | Eski sürümden kalma ifadeyi silin |
| 6 | Bölüm VI-E | $12,320 "en iyi ücretli API" için verilmiş | Bu GPT-4.5'in rakamı; o1 için **$24,240** (1M tahmin). Hangisini kastettiğinizi netleştirin |
| 7 | Bölüm III-D, Şekil 1 | %63.6 tekrar çıkarma öncesine, %51.1 tekrar çıkarma sonrasına ait | 8 Ekim: 72.611 kayıtlık deduplikasyon sonrası dağılım ve bütün test dosyalarının yeniden üretilebilirliği doğrulandı; yöntem açıklandı |
| 8 | Bölüm VI-C | mistral:7b (59→42) "iyileşti", llama (67→57) "çözülmedi" | Mantığı tutarlı hale getirin veya karşılaştırmayı kaldırın |

---

## Aşama 2 — Ek analizler (mevcut verilerle, yeni model eğitimi gerektirmez)

- [x] **Bootstrap güven aralıkları:** Her model için üç test altkümesinin birleştirilmiş 3.000 tahmini üzerinden %95 CI hesaplandı; API eşdeğerliği iddiaları yumuşatıldı. Yeni HF karşılaştırmasında eşleştirilmiş metin-kümesi bootstrap kullanıldı.
  - qwen3:14b ve gpt-oss:20b'nin GPT-4o'yu "geçtiği" fark seed std (~0.016) içinde; ifade buna uygun olmalı.
- [x] **Deney B yeniden ağırlıklandırma:** Confusion matrix eşit sınıf öncülleriyle yeniden ağırlıklandırıldı; aggregate artışın sınıf oranlarıyla ilişkisi makaleye işlendi.
  - Düşük off-label, sınıf öncülü etkisini dışlamaz; Nemo için yeniden ağırlıklandırma raporlandı.
- [x] **Nitel hata analizi:** Parse hatası/ret yanıtı ve sözlük dışı JSON etiket örnekleri CSV'ye kaydedildi; ayrım makalede yorumlandı.
- [x] **250 vs 500 karakter duyarlılığı:** Dosyalar incelendi; aynı kayıtların aynı sırada 250 karaktere kısaltılmış sürümleri olduğu doğrulandı. Eşleştirilmiş McNemar ve 22 karşılaştırma için Holm düzeltmesi yapıldı; yorumlar buna göre güncellendi.
- [x] **Maliyet / verim argümanı:** 17.1 s × 1M ≈ 198 seri GPU-günü ve API ücretinin toplam sahip olma maliyeti olmadığı açıklandı. Yeni HF seri ölçümler ile amortize batch süreleri ayrıldı; gerçek serving-engine ve çoklu GPU throughput ölçülmedi.
- [x] **"Parametre-eşleştirilmiş" iddiası:** Qwen/DeepSeek içindeki 8B/14B çiftleri ile nesil/boyut değişimini karıştıran diğer karşılaştırmalar ayrıldı.

---

## Aşama 3 — İnce ayar (QLoRA) deneyi

### 3.1 Model seçimi (ChatGPT ve Claude önerileri örtüşüyor)

| Model | Gerekçe |
|-------|---------|
| gemma4:12b | Zero-shot'ta #1; ince ayar tavanını gösterir |
| qwen3:14b (veya 8b) | Reasoning temsilcisi; ince ayar sonrası düşünme modu kapalı doğrudan etiket üretimi → gecikme kazancı olabilir |
| phi4:14b | En hızlı; "hızlı model hafif adaptasyonla rekabetçi olur mu?" sorusu |
| *(opsiyonel)* llama3.1:8b / mistral:7b | İnce ayarın off-label sorununu bitirip bitirmediğini gösterir |

- 11 modelin hepsini ince ayarlamayın; üç iyi seçilmiş model yeterli. gpt-oss:20b'yi 16 GB'ta zorlamayın.

### 3.2 Teknik ayarlar (başlangıç noktası)

- 4-bit NF4 QLoRA, LoRA rank r = 8/16, gradient checkpointing
- batch size 1–2 + gradient accumulation, dizi uzunluğu 512–1024
- Kartınızı ve belleğinizi kontrol edin (ChatGPT RTX 5060 Ti 16 GB varsaydı; siz sadece 16 GB yazdınız)
- Gemma4 için seçtiğiniz kütüphanenin model desteğini önceden doğrulayın
- Tüm hiperparametreleri (rank, öğrenme oranı, epoch, seed) makalede raporlayın

### 3.3 Veri bölme (kritik karar)

**Önerim (Claude):** Mevcut 3 seed × 2 koşul test setlerini **aynen test seti olarak koruyun.**
1. Bu test kayıtlarını havuzdan çıkarın.
2. Kalan veriden train ve validation (ayrı) çekin.
3. Böylece zero-shot sonuçlarınızla ve Roumeliotis protokolüyle doğrudan karşılaştırılabilirlik korunur.

**ChatGPT'nin önerisi:** Yeni bir sabit train/val/test (örn. 20.000 / 2.000 / 5.000). Bu yolu seçerseniz zero-shot sonuçlarını da **aynı yeni test setinde yeniden koşmanız** gerekir; aksi halde tabloda elma-armut karşılaştırması olur.

- [x] Mevcut 12 test dosyası korundu; train/validation havuzdan ayrı çekildi ve yöntem açıklandı.
- [x] Dengeli iç içe eğitim altkümeleri ve sabit dengeli validation kullanıldı.
- [x] Tam metin/250-önek eşleşmeleri, çelişkili etiketler ve tespit edilen yakın kopyalar dışlandı; MinHash taramasının yaklaşık olduğu belirtildi.

### 3.4 Deney matrisi

| Boyut | Değerler |
|-------|----------|
| Model | gemma4:12b, qwen3:14b, phi4:14b |
| Eğitim verisi boyutu (öğrenme eğrisi) | 500 / 2.000 / 10.000 / 20.000 |
| Seed | 1, 2, 3 |
| Test koşulu | Dengeli (A), azaltılmış dengesiz (B) |
| Raporlanacak | Acc, weighted F1, macro-F1, sınıf bazlı F1, off-label, gecikme, eğitim süresi/VRAM |

**Tamamlanan kapsam:** Üç LLM için 500 eğitim kaydı/seed 1; tüm 12 test
koşulu, ayrıca her model/mod için 100 seri ölçüm. LLM 2.000/10.000/20.000
öğrenme eğrisi ve ek eğitim seed'leri henüz çalıştırılmadı. Klasik öğrenme
eğrisi ve 20.000 kayıtlı DistilBERT'in üç eğitim seed'i tamamlandı.

### 3.5 Karşılaştırma tablosu (makalenin merkez tablosu)

| Model | Eğitim kaydı | Matched HF base F1 | Adapterli/supervised F1 | Δ (yüzde puan) | Seri gecikme | Eğitim/doğrulama |
|---|---:|---:|---:|---:|---:|---:|
| Gemma 4 12B | 500 | 0.7520 | 0.7803 | +2.83 | 0.647 s | 34.75 dk |
| Qwen3 14B | 500 | 0.7099 | 0.7891 | +7.92 | 0.539 s | 41.15 dk |
| Phi-4 14B | 500 | 0.6883 | 0.7845 | +9.62 | 0.435 s | 40.68 dk |
| TF-IDF/SVM | 500 | — | 0.7546 | — | 0.29 ms CPU | — |
| TF-IDF/SVM | 20.000 | — | 0.8291 | — | 0.29 ms CPU | — |
| DistilBERT, üç eğitim seed'i | 20.000 | — | 0.8244 ± 0.0046 | — | 2.07 ms GPU | yaklaşık 80 s/fit |

Her supervised bütçeye ayrıca ortak 2.000 validation etiketi eklenir.
F1 aynı 3.000 dengeli 500-karakter testinden hesaplanır. Eski Ollama
GGUF veya literatürdeki API skorları adaptasyon tabanı değildir.

- [x] **Baselines:** TF-IDF/SVM öğrenme eğrisi ve üç eğitim seed'i DistilBERT çalıştırıldı.
- [x] **Eşleştirilmiş test:** Aynı HF base/adapter çıktıları için exact accuracy McNemar ve 12 karşılaştırmalık Holm düzeltmesi tamamlandı.
- [x] **Sıralama değişimi:** Dengeli matched HF base sırası Gemma/Qwen/Phi; adapterli nokta tahmini Qwen/Phi/Gemma. Adapterli modeller arası anlamlı üstünlük iddia edilmedi.
- [x] **Literatür referansı:** Güllü ve ark. farklı bölme/ayar bağlamında ele alındı.

> **Uyarı:** ChatGPT'nin verdiği ".855 / .872 / .861" ve "0.88–0.90" gibi değerler **tahmindir, hedef değil.** Etiketler örtüşüyor ve gürültülü; plato görmeniz olası. Makale, sonuç ne olursa olsun sağlam durmalı.

---

## Aşama 4 — Makalenin yeniden yapılandırılması

- [x] **Başlık ve özet:** Yerel sınıflandırma, hafif adaptasyon, etiket bütçesi ve ölçülen donanım/serving maliyeti etrafında güncellendi.
- [x] **Katkılar:** Kontrollü adaptasyon, klasik öğrenme eğrisi, supervised baselines, gecikme/VRAM ve eşleştirilmiş istatistikler eklendi.
- [x] **Limitations:** Tek prompt ve tek adapter eğitim seed'i, ek 2.000 validation etiketi, tarihsel kimlik/batch-seri farkları, nicemleme ve contamination sınırları açıklandı.
- [x] **İlişkili çalışmalar:** Bankacılık/regülasyon uygulaması ile LoRA/QLoRA'nın mühendislik rolü eklendi.
- [x] **Sonuç:** QLoRA pilotu tamamlanan bulgu olarak yazıldı; daha büyük eğitim bütçeleri/seed'ler ve serving çalışmaları geleceğe bırakıldı.

---

## Aşama 5 — Kaynakça ve biçim

- [ ] **Eksik atıflar:** Gemma, Llama 3.1, Mistral/Mistral-Nemo, Phi-4, gpt-oss, Ollama, QLoRA/LoRA (ve eklenecek BERT tabanlı baseline).
- [ ] **Tüm referansları tek tek doğrulayın** (yazar listesi, sayfa, DOI, arXiv numarası); "et al." ile kısaltılmış ve eksik görünen girdileri tamamlayın.
- [ ] **Şablon:** IEEE şablonundan Sādhanā / Springer şablonuna geçin ("Index Terms", "Manuscript received" gibi IEEE öğelerini kaldırın).
- [ ] Derginin güncel "Instructions for Authors" sayfasını okuyun (uzunluk, şekil, veri/kod beyanı, etik beyanı, ücret/açık erişim seçenekleri).
- [ ] Şekil alt yazılarını ve tablo numaralarını yeniden tutarlılık için gözden geçirin.
- [ ] Kod ve eğitim betiklerini GitHub deposuna ekleyin; kullanılan model sürümlerini ve seed'leri yazın.

---

## Aşama 6 — Gönderim

- [ ] İngilizce dil kontrolü (özellikle yeni bölümler).
- [ ] Cover letter: Sādhanā'nın kapsamıyla (bilgisayar bilimi dahil mühendislik, uygulama odaklı çalışma) bağlantıyı kurun; yerel dağıtım ve veri gizliliği vurgusu.
- [ ] Olası hakem sorularına önceden hazırlıklı olun:
  1. Novelty nedir? → ince ayar + öğrenme eğrisi + deployment analizi
  2. Neden LLM, klasik baseline yetmez mi? → BERT/RoBERTa karşılaştırması
  3. Ücretli API karşılaştırması adil mi? → protokol eşleşmesi, CI, sınırlamalar
  4. Eğitim kirlenmesi / veri sızıntısı? → bölme şeması ve kontroller

---

## Önerilen sıra (özet)

1. Aşama 0 (doğrulamalar)
2. Aşama 1 (tutarsızlık düzeltmeleri)
3. Aşama 2 (ek analizler)
4. Aşama 3 (QLoRA deneyleri)
5. Aşama 4–5 (yeniden yazım, kaynakça, şablon)
6. Aşama 6 (gönderim)

Aşama 0–2 hâlâ mevcut verilerle yapılabilir; yeni deneyler Aşama 3'te başlar. Aşama 0 sonuçları (özellikle gemma4 düşünme modu) zero-shot rakamlarını değiştirirse, Aşama 3'teki karşılaştırma tablosu bu yeni rakamlarla kurulmalıdır.

## 8 Ekim hakem incelemesi

[Detaylı rapor](hakem_degerlendirmesi_2026-10-08.md): kaynakça,
veri akışı, tablo metrikleri, geçersiz çıktı matrisleri ve bütün şekil/tablo
göndermeleri kontrol edildi. Kesin hatalar düzeltildi; tek adapter eğitim
seed'i, aynı bütçede küçük encoder kontrolü ve tarihsel Ollama kimliği
açık bilimsel sınırlılıklar olarak ayrıldı. Son ana/ek PDF: 24/14 sayfa.


## Ek deneyler — 8 Ekim 2026, çalışıyor

Hakem incelemesindeki eşit etiket bütçesi ve eğitim-seed değişkenliği
eksikleri için kuyruk başlatıldı. Yukarıdaki tek-seed sonuçlar ilk pilotu
belgeliyor; üç-seed LLM sonuçları henüz tamamlanmadı.

- Aynı 500 eğitim / 2.000 doğrulama kaydıyla DistilBERT: üç epoch ve
  doğrulama F1'ına göre en fazla 30 epoch / patience 5 kontrolleri,
  her biri üç eğitim seed'i. Altı fit ve 72.000 tahmin tamamlandı.
  Dengeli 500-karakter testlerinde weighted F1: üç epoch
  **0.5329 ± 0.1522**, erken durdurmalı kontrol **0.7150 ± 0.0208**.
  Bunlar üç fit'in ortalaması ve örnek standart sapmasıdır.
- Gemma 4, Qwen3 ve Phi-4 için seed 2 ve 3 ekleniyor. Her yeni fit'in
  base/adapter tahminleri aynı yüklenmiş ağırlıklarda ayrı ayrı üretiliyor.
  Altı yeni LLM fit'i, 144.000 eşleştirilmiş test çıktısı ve 1.200
  seri çıktı planlandı. Mevcut seed-1 ağırlıkları ve tahminleri korunuyor.
- Her fit aynı 3.000 test kaydında ayrı değerlendirilir. Eğitim seed'leri
  9.000 bağımsız test kaydı gibi birleştirilmez. Base/adapter için
  36, etiket bütçesi eşit classifier karşılaştırmaları için ayrı 72
  testlik Holm düzeltmesi ve 5.000 eşleştirilmiş küme bootstrap'i kullanılır.
- Modeller sabit HF commit'lerinden offline önbellekle yükleniyor;
  bu kuyruk model indirmez. GPU işleri sırayla çalışır.
- Tarih/özgün CFPB kimliği içeren ek dosya yok: zamansal genelleme
  yapılmayacak ve bu sınırlılık makalede kalacak.

Canlı durum: `python finetuning_status.py`. Servis:
`consumercomplaint-replication-n500.service`. Ayrıntılı plan, log ve
kısmi rapor: `results/sadhana/replication_20261008/`. Tüm fit'ler ve
tahminler doğrulandığında makale ayrı dizinde derlenip atıf/tablo/şekil
denetiminden geçecek; ardından yedek alınarak üç-seed sonuçları işlenecek.
Ana makale bu tamamlanma öncesinde hâlâ tek-seed pilotunu raporlar.
