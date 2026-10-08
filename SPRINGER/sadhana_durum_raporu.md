# Sādhanā revizyonu — çalışma durumu

Son kontrol: 8 Ekim 2026. Ortam: `ConsumerComplaint`; RTX 5060 Ti 16 GB.

## Ana bulgu ve önerilen yön

Makalenin güçlü yönü artık yalnızca yerel zero-shot model sıralaması değil,
**etiket bütçesine göre yerel sınıflandırıcı seçimi** olmalıdır.
Mevcut testlerde supervised baselines zero-shot LLM'lerden daha başarılı ve
çok daha hızlı çıktı. Tamamlanan 500 kayıtlık QLoRA pilotları aynı HF
ayarlarıyla F1 artışı sağladı. Tüm test koşulları ve seri gecikme ölçümleri
tamamlandı; sonuçlar makalenin yöntem, sonuç ve tartışmasına işlendi.
Bir kabul olasılığı veya beklenen fine-tuning F1 değeri hesaplanmadı.

| Yöntem | Eğitim kaydı | Dengeli weighted F1 | Ölçülen seri gecikme |
|---|---:|---:|---:|
| Gemma 4 12B, eski Ollama zero-shot | 0 | 0.7738 | 17.12 s |
| Qwen3 14B, eski Ollama zero-shot | 0 | 0.7498 | 5.73 s |
| Phi-4 14B, eski Ollama zero-shot | 0 | 0.7057 | 0.54 s |
| TF-IDF + LinearSVC, CPU | 500 | 0.7546 | 0.29 ms |
| TF-IDF + LinearSVC, CPU | 2.000 | 0.7908 | 0.29 ms |
| TF-IDF + LinearSVC, CPU | 10.000 | 0.8241 | 0.28 ms |
| TF-IDF + LinearSVC, CPU | 20.000 | 0.8291 | 0.29 ms |
| DistilBERT, GPU, üç eğitim seed'i | 20.000 | 0.8244 ± 0.0046 | 2.07 ms |

F1 değerleri aynı 3.000 dengeli test kaydının birleştirilmesiyle hesaplandı.
DistilBERT için her model fit'i ayrı değerlendirildi, ardından üç eğitim
seed'inin ortalaması/std'si alındı; 9.000 bağımsız test kaydı varsayılmadı.
Supervised yöntemler eğitim etiketleri kullanır; zero-shot ile etiket bütçeleri
eşit değildir. SVM gecikmesi CPU, DistilBERT gecikmesi GPU ölçümüdür;
eski Ollama süreleri ayrı bir serving/runtime protokolüne aittir.

## Tamamlanan HF adaptersiz / adapterli ana karşılaştırma

Her model aynı 3.000 dengeli, 500-karakter test kaydında değerlendirildi.
Eğitim 500 dengeli kayıt, ortak ayrı 2.000 doğrulama kaydı ve tek eğitim
seed'i (1) kullanır; üç test altkümesi
üç ayrı eğitim seed'i değildir. Aynı HF commit'i/NF4/BF16 ve düşünme kapalı
şablonla adapter etkinleştirilip devre dışı bırakıldı.

| Model | Adaptersiz weighted F1 | Adapterli weighted F1 | Artış (yüzde puan) | Artış için %95 CI |
|---|---:|---:|---:|---:|
| Gemma 4 12B | 0.7520 | 0.7803 | +2.83 | [+1.75, +3.96] |
| Qwen3 14B | 0.7099 | 0.7891 | +7.92 | [+6.55, +9.34] |
| Phi-4 14B | 0.6883 | 0.7845 | +9.62 | [+8.21, +11.07] |

Üç artışın da 5.000 tekrarlı eşleştirilmiş bootstrap güven aralığı pozitiftir.
Accuracy: Gemma %75.13 → %78.07; Qwen %71.57 → %78.83;
Phi %68.73 → %78.40. Adapterli üç modelde geçersiz çıktı yoktur;
Phi'nin adaptersiz çıktılarında 6 geçersiz cevap ve 5 çıktı-limitine takılma
vardır. 18.000 tahminde kaynak/etiket/indeks/ham cevap ve eşit çiftli batch
boyutları doğrulandı. Büyük artış yalnızca JSON biçim hatalarının düzelmesiyle
açıklanamaz: Gemma ve Qwen'in adaptersiz çıktılarında da geçersiz cevap yoktur.

Sabit rapor, CSV, şekil ve doğrulama kaydı:
`results/sadhana/qlora_primary_comparison_20261006/`.
Tüm 250/500-karakter, dengeli/dengesiz koşullar ve gerçek batch-1 seri
benchmark 6 Ekim 2026 saat 18:24'te tamamlandı. 8 Ekim kontrolünde 72.000
test tahmini + 600 seri ölçüm; altı completion manifesti ve SHA256'ları,
train/validation/test/adapter hash'leri, kaynak/etiket/indeks eşleşmesi,
ham cevaplar, çiftli batch boyutları ve pooled metrikler doğrulandı.
12 F1 artış CI'sının tamamı pozitif; accuracy için 12 karşılaştırmalık
Holm düzeltmesinden sonra tüm p değerleri < 0.00142.
Tam rapor: `results/sadhana/qlora_comparison/report.md`.

| Model | Azaltılmış dengesiz 500-karakter HF base F1 | Adapterli F1 | Seri base (s) | Seri adapter (s) |
|---|---:|---:|---:|---:|
| Gemma 4 | 0.7730 | 0.7948 | 0.475 | 0.647 |
| Qwen3 | 0.7278 | 0.7946 | 0.421 | 0.539 |
| Phi-4 | 0.7456 | 0.7994 | 0.487 | 0.435 |

Seri süreler aynı 100 dengeli 500-karakter örneğe aittir; dengesiz koşul
süreleri değildir. Adapterli 36.000 test tahmininde geçersiz cevap yok;
base Gemma 2, Qwen 0, Phi 21 geçersiz cevap üretti. 600 seri/batch
karşılaştırmasının üçünde etiket farkı var (Gemma adapter 1, Qwen base 2);
metin hash ve token sayıları eşleşir, hesaplama nedeni izole edilmedi.
F1 tam batched çıktıdan, süre bağımsız seri benchmark'tan hesaplanır.
Phi uzun girişlerde batch 32 bellek aşımından sonra batch 16 ile sürdürüldü;
aynı çiftin iki modu aynı batch boyutuyla değerlendirilir. Tek eğitim seed'i
ile elde edilen bu bulgu daha büyük etiket bütçeleri ve eğitim seed'leriyle
tekrar edilebilir; şimdiki sonuçlar yalnızca 500 kayıt pilotuna aittir.

## Tamamlanan işler

- `ConsumerComplaint` aktive edilerek analiz ve eğitim paketleri kuruldu;
  Jupyter kernel'i eklendi. Paket sürümleri `requirements-lock.txt` ve
  `environment-consumercomplaint.yml` içinde kaydedildi.
- `pip check`, CUDA matris işlemi, BF16, NF4 forward/backward ve küçük bir
  Qwen3 üzerinde yalnızca cevap token'larına loss uygulayan LoRA backward
  doğrulandı. Bu küçük test, gerçek 12B/14B eğitimi değildir.
- 132 tahmin dosyasında eksik/tekrarlı sıra numarası ve gerçek etiket
  eşleşmesi kontrol edildi. Kaynak test dosyaları ve eski tahminler değiştirilmedi.
- 250/500 çiftlerinin tamamı aynı kayıtların aynı sırada kısaltılmış
  sürümleri. Dengeli/dengesiz kümeler arasında örtüşme var; bağımsız tekrar
  oldukları iddiası kaldırıldı. Eski tahminlerde metin hash'i bulunmadığından
  tarihsel narrative kimliği yalnızca etiket/sıra kontrolüyle kesin kanıtlanamaz.
- 5.000 tekrarlı, sınıfa göre tabakalı metin-kümesi bootstrap CI hesaplandı.
  Gemma 4 dengeli pooled F1: **0.7738, %95 CI [0.7586, 0.7882]**.
  CI'lar kayıtlı test/çıktılara koşulludur; prompt ve model eğitimi belirsizliği
  ile ücretli API referansının belirsizliğini içermez.
- Deney B confusion matrix'i eşit sınıf öncüllerine göre yeniden
  ağırlıklandırıldı. Mistral-Nemo accuracy 0.6530 → 0.5562; gpt-oss
  0.7820 → 0.7480. Artışların büyük bölümü sınıf oranlarıyla açıklanıyor.
- 250/500 accuracy farkı için eşleştirilmiş McNemar ve 22 karşılaştırmada
  Holm düzeltmesi yapıldı. Dengeli koşulda Llama ve Qwen3 14B farkları
  anlamlı; DeepSeek-R1 14B farkı düzeltilmiş eşikten geçmedi.
- JSON decoder ile yeniden ayrıştırma kayıtlı geçerli tahminleri değiştirmedi.
  Gemma 4'te tüm 12 koşul boyunca 37 boş yanıt var. Llama hatalarının
  örnekleri çoğunlukla ret yanıtları; Mistral etiketleri çoğunlukla geçerli
  JSON içinde sözlük dışı kategoriler. Nitel örnekler CSV'ye kaydedildi.
- Mevcut testlerin birleşimi eğitim havuzundan çıkarıldı. 20.000 dengeli
  eğitim ve 2.000 ayrı dengeli doğrulama kaydı hazırlandı; 500/2.000/10.000
  eğitim altkümeleri iç içe. 133 çelişkili etiket grubu, 4.444 tam eşleşme/önek
  kaydı, 496 yakın test kopyası ve 28 yakın validation kopyası dışlandı.
  MinHash taraması yaklaşıktır; anlamsal kopyaların tümünü bulma garantisi yoktur.
- TF-IDF/SVM öğrenme eğrisi ve DistilBERT üç seed ile gerçekten çalıştırıldı.
  Her DistilBERT eğitimi yaklaşık 80 saniye ve 1.30 GiB tepe allocated VRAM.
  20.000 kayıtlı SVM ve her DistilBERT fit'i Gemma 4'e karşı dengeli accuracy
  testinde üstün: 42 karşılaştırma için Holm-düzeltilmiş p < 0.001.
- Makaledeki kesin sayı, sıralama ve maliyet tutarsızlıkları düzeltildi;
  API eşdeğerliği ve reasoning mekanizması hakkındaki aşırı iddialar yumuşatıldı.
  İstatistikler, supervised yöntemler ve sonuçlar TeX'e eklendi.
- Tekrarlayan grafikler ve ayrıntılar `supplementary_material.tex` dosyasına
  taşındı; ilk revizyon 20/13 sayfa idi. QLoRA entegrasyonuyla ana metin
  **24 sayfa**, ek materyal **14 sayfa** olarak yeniden derlendi.
  İki PDF'de de çözümlenmemiş atıf/referans kalmadı. Bu sayfa sayısı mevcut
  genel Springer şablonuna aittir; derginin nihai biçim onayı değildir.

## Üç LLM QLoRA için hazır olanlar ve sınırlar

Seçim: **Gemma 4 12B, Qwen3 14B, Phi-4 14B**. Model sınıfları ve thinking
kapalı sohbet şablonları yüklendi; commit'ler `*_preflight.json` içinde.
`train_qlora.py` ve `evaluate_qlora.py` hazır; komutlar ana `README.md` içinde.
6 Ekim 2026'da `ollama pull gemma4:12b`, `qwen3:14b`, `phi4:14b` indirmeleri
ve HF ağırlıklarıyla 500 kayıt/seed 1 QLoRA pilot kuyruğu başlatıldı.
Üç Ollama indirmesi ve üç 500 kayıt/seed 1 pilotu saat 13:59'da tamamlandı;
`results/sadhana/finetuning_run/status.json` tamamlanmayı kaydeder.
NF4/BF16, rank 8, 3 epoch, batch 1/accumulation 16 ve completion-only loss
kullanılır. Phi-4'ün birleşik QKV/MLP katmanları hedeflenir; büyük dondurulmuş
embedding matrisleri 16 GB GPU için BF16 tutulur. Her model 96 eğitim adımını
ve 3 epoch'u tamamladı; adapter dosyalarındaki bütün tensörler sonlu bulundu.
Eğitim + doğrulama süreleri Gemma/Qwen/Phi için sırasıyla 34,75 / 41,15 /
40,68 dakika; tepe ayrılmış VRAM 7,77 / 10,04 / 9,32 GiB. İndirme/yükleme
süreleri bu eğitim sürelerinden ayrıdır. Ayrıntılar
`results/sadhana/finetuning_run/pilot_summary.json` içindedir.
Pilotların adaptersiz/adapterli HF değerlendirmesi tüm koşullarda tamamlandı.
İlerleme `results/sadhana/evaluation_run/status.json`, kısmi ve son ölçümler
`results/sadhana/qlora_comparison/pooled_comparison.csv` içindedir;
`test_seeds_complete` tamamlanan üç test altkümesini belirtir.
Önce dengeli 500 karakter koşulu, sonra kalan testler değerlendirilir.
En fazla batch 32 (Phi bellek sınırı sonrası 16) amortize süre ile ayrıca
100 kayıtlık seri benchmark süresi ayrı
raporlanır. Son aşamada paired cluster bootstrap ve McNemar/Holm hesaplanır.
Üç mimarinin küçük rastgele ağırlıklı kopyalarında NF4/BF16 ile eğitim,
doğrulama, checkpoint seçimi ve adapter kaydı geçti
(`results/sadhana/qlora_stack_smoke.json`). Bu testler büyük model performansı
veya VRAM ölçümü değildir. Pilot ve doğrulama kayıtları her tokenizer ile
kontrol edildi; en uzun örnek 204 token, hiçbir hedef cevap kesilmiyor
(`results/sadhana/qlora_tokenization_check.json`).

500 kayıt pilotları tamamlandı; ardından kaynaklara göre 2.000/10.000/
20.000 ve seed 1/2/3. Eğitim sonucu ne olursa olsun SVM ve DistilBERT
karşılaştırmaları tabloda kalmalıdır. Adapter etkisini izole etmek için
aynı HF commit'i, NF4, şablon ve thinking modu altında **adaptersiz ve
adapterli** çıkarım yapılmalıdır. Eski Ollama GGUF → yeni HF NF4 farkının
tamamı fine-tuning etkisi diye sunulmamalıdır.

## Gönderim öncesi hâlâ gerekli

1. **Gemma 4 thinking-on/off kontrolü ve tarihsel model kimliği.** Resmî
   model kartı thinking destekliyor. Eski CSV'lerde ayrı thinking alanı,
   `done_reason` ve token sayısı yok; Gemma tag'i şimdi bu bilgisayarda kurulu,
   güncel digest/metadata kaydedildi; bu eski artifact kimliğini kanıtlamaz.
   Eski deneylerin digest/metadata kayıtları veya kontrollü yeni pilot gerekir.
   “Non-reasoning model birinci” ana iddiası bu nedenle korunmadı.
2. **Tarihsel nicemleme ve altyapı.** Güncel yerel gpt-oss metadata'sı MXFP4;
   Qwen3 8B Q4_K_M. Bu, bütün eski koşulların aynı artifact olduğu kanıtı
   değildir. `run_inference.py` başlığı TRUBA'dan söz ederken eski makale
   yerel workstation diyor; eski yürütme/GPU loglarıyla netleştirilmelidir.
3. **QLoRA deneylerinin kapsamı ve son makale çerçevesi.** 500 kayıt/seed 1
   pilotlarının tüm koşulları, 100 örneklik seri gecikme ölçümleri ve
   istatistikleri hazır. Başlık/özet/katkılar/yöntem/sonuç/tartışma güncellendi;
   tablolar doğrudan doğrulanmış CSV'lerden üretildi. Daha büyük LLM eğitim
   bütçeleri ve ek eğitim seed'leri henüz çalıştırılmadı; şimdiki metin bu
   sınırlamayı açıkça belirtir. Biçimsel düzenlemeler en sona bırakıldı.
4. **Dergi biçimi.** Resmî yazar kılavuzu araştırma makalelerinde 20 sayfa,
   en fazla yaklaşık altı keyword, 60 karakterden kısa running title,
   çift aralıklı metin ve renkli/grayscale şekillerde 300 dpi TIFF/JPEG
   istiyor. Keyword/running-title/“received” satırı düzeltildi. Springer
   genel şablonu ile derginin IAS stili ve satır aralığı ayrıca kontrol edilmeli;
   mevcut PNG'lerin gerçek 300 dpi çıktılarına dönüşümü tamamlanmadı.
   QLoRA içeriği sonrası 24 sayfalık ana metin ayrıca kısaltılmalıdır.
5. **Kaynakça doğrulaması ve gönderim belgeleri.** Gemma model kartı ve
   DistilBERT referansı eklendi. Bütün eski referansların DOI/yazar/sayfa
   kontrolü bu hakem incelemesinde önemli yanlışlıkları giderdi; bütün
   tam metinlerle cümle eşleştirmesi, dil kontrolü ve cover letter tamamlanmadı.
   LoRA/QLoRA atıfları özgün arXiv kayıtlarından doğrulanarak eklendi.
   Yeni yerel analiz/eğitim kodlarının ve artifact'lerin kamuya açık sürümü
   commit/push yapılmadığı için gönderim öncesinde tamamlanmalıdır.
   Gönderim yapılmadı.

## Kaynaklar ve çıktı konumları

- [Sādhanā resmî gönderim sayfası](https://link.springer.com/journal/12046/submission-guidelines)
- [2026 Instructions for Authors PDF](https://media.springer.com/full/springer-instructions-for-authors-assets/pdf/367298_173709708_12046-Information-Authors_up_2026.pdf)
- [Gemma 4 model kartı](https://huggingface.co/google/gemma-4-12B-it)
- [Ollama thinking belgelendirmesi](https://docs.ollama.com/capabilities/thinking)
- [Qwen3 14B model kartı](https://huggingface.co/Qwen/Qwen3-14B)
- [Phi-4 model kartı](https://huggingface.co/microsoft/phi-4)
- [DistilBERT makalesi](https://arxiv.org/abs/1910.01108)

Analizler: `results/sadhana/`; eğitim bölmeleri: `dataset/finetuning/`;
gerçek DistilBERT checkpoint'leri: `checkpoints/distilbert/n20000/seed{1,2,3}/`.

## 8 Ekim hakem denetimi

Ayrıntılı rapor: [hakem_degerlendirmesi_2026-10-08.md](hakem_degerlendirmesi_2026-10-08.md).
Özet 179 kelimeye sadeleştirildi; donanım ve nicemleme ayrıntıları çıkarıldı.
Eski tablolar tahminlerden yeniden üretildi; iki yönlü exact McNemar için
12 testlik Bonferroni ailesi kullanıldı. Geçersiz yanıtları retail banking'e
aktaran matris kodu düzeltildi ve üç ek şekil yeniden çizildi. Eksik tablo
göndermeleri ve S1 açıklaması düzeltildi; yanlış DOI/arXiv künyeleri
birincil kayıtlarla güncellendi. Deney B dağılımının tekrarları çıkarılmış
72.611 kayıtlık havuzdan geldiği doğrulandı. A/B uzun testleri 2.067 kayıt
paylaşır; birleşim 3.933 benzersiz şikâyettir. Yeni LaTeX kaynaklarının
`.gitignore` tarafından gizlenmesi giderildi. Ana PDF 24, ek PDF 14 sayfa;
biçimsel kısaltma ve ek eğitim seed'leri hâlâ açık.


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
