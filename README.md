# Consumer complaint classification: Sādhanā revision

Mevcut zero-shot tahminlerini değiştirmeden doğrulama, istatistiksel analiz ve
aynı test kayıtları üzerinde supervised karşılaştırma için betikler.

Python betikleri `utils/` klasöründedir. Komutları proje ana dizininden
`python utils/betik_adi.py` biçiminde çalıştırın; paket olarak
`python -m utils.betik_adi` kullanımı da desteklenir. Örneğin mevcut
deney durumunu `python utils/finetuning_status.py` gösterir.

## Ortam

```bash
source /home/tan/anaconda3/etc/profile.d/conda.sh
conda activate ConsumerComplaint
python -m pip check
```

Jupyter kernel: `Python (ConsumerComplaint)`. Kurulan sürümler
`requirements-lock.txt`, conda ortamı `environment-consumercomplaint.yml` içinde.
Başka bir makinede CUDA/PyTorch paketleri GPU ile uyumlu kurulmalıdır.

Mevcut splitleri değiştirmeden, eğitim veya model çıkarımı başlatmadan
veri ayrımlarını denetlemek için:

```bash
python utils/audit_split_integrity.py --output results/sadhana/reviewer_audit/split_integrity.json
```

Bu kontrol kaynak/split hash'lerini, kaynak satırı eşleşmelerini, tam metin
ve 250 karakter öneki çakışmalarını, iç içe eğitim kümelerini ve kayıtlı
MinHash/Jaccard yakın kopya yöntemini denetler. Yakın kopya aday bulma
işlemi yaklaşık olduğundan tüm anlamsal benzerliklerin yokluğunu kanıtlamaz.

## Deney dosyalarının Git takibi

`dataset/finetuning/` içindeki eğitim/validation CSV'leri ve manifest Git'e
dahildir. Küçük metrik CSV/JSON'leri, model revision/deney ayarları ve son
split doğrulama raporları da saklanır. `results/sadhana/` altındaki ham
JSONL yanıtları, `preds_*.csv` tahminleri, `serial_benchmark.csv`, thinking
tahmin CSV'si, loglar ve geçici makale/denetim yedekleri ignore edilir.
`preds/` altındaki sıfır-atış benchmark dosyaları takipte kalır.

Takipten çıkarma diskteki sonuçları silmez; eski commit'lerin boyutunu da
küçültmez. Ham deney çıktıları yeni bir klonda bulunmayacağından, bunları
okuyan yeniden hesaplama/doğrulama betikleri için ayrı sonuç arşivi gerekir.
Makale için paylaşılacak yeniden üretim paketine bu arşiv dahil edilmelidir;
yalnızca metrik özetleri ham tahmin denetiminin yerine geçmez.

## Tamamlanan analizleri yeniden üretme

```bash
python utils/sadhana_audit.py --bootstrap 5000
python utils/prepare_finetuning_data.py
python utils/run_classical_baseline.py
python utils/train_encoder_baseline.py --train-size 20000 --seed 1
python utils/train_encoder_baseline.py --train-size 20000 --seed 2
python utils/train_encoder_baseline.py --train-size 20000 --seed 3
python utils/summarize_sadhana_baselines.py
```

Veri hazırlama ve eğitim betikleri mevcut çıktıları üzerine yazmayı reddeder.
Yeni deney için çıktı dizinini ayrıca arşivleyin. Mevcut test altkümelerini
`utils/build_experiment_subsets.py` ile yeniden üretmeyin: dosyaları değiştirir ve
kayıtlı tahminlerle eşleşmeyi bozabilir. Tahminler yalnızca sıra numarası ve
etiket içerdiğinden etiket eşleşmesi, tarihsel metin kimliğinin kesin kanıtı değildir.

Eğitim/doğrulama dosyaları `dataset/finetuning/` altında, checkpoint'ler
`checkpoints/` altında tutulur ve Git dışında bırakılmıştır. Manifest,
orijinal test dosyalarının ve yeni bölmelerin SHA-256 değerlerini içerir.
Eğitimde 500/2.000/10.000/20.000 iç içe dengeli altkümeler; sabit, ayrı
2.000 doğrulama kaydı kullanılır. Test kayıtları, 250 karakterlik aynı
önekler ve tespit edilen yakın kopyalar dışlanır. MinHash yakın kopya taraması
yaklaşıktır; tüm anlamsal benzerlikleri dışladığı iddia edilmez.

## Üç LLM ile QLoRA

Seçilen modeller: `google/gemma-4-12B-it`, `Qwen/Qwen3-14B`, `microsoft/phi-4`.
Model sınıfı ve sohbet şablonu kontrolü ağırlıkları indirmez:

```bash
python utils/train_qlora.py --model gemma4 --preflight
python utils/train_qlora.py --model qwen3 --preflight
python utils/train_qlora.py --model phi4 --preflight
```

Ollama indirmeleri ve 500 kayıtlık üç QLoRA pilotu 6 Ekim 2026 tarihinde
`consumercomplaint-qlora-n500-s1` kullanıcı servisiyle başlatıldı. Kuyruk
Gemma 4 → Qwen3 → Phi-4 sırasındadır; aynı anda bir GPU eğitimi çalışır.
Üç pilot ve Ollama indirmeleri 6 Ekim 2026 saat 13:59'da tamamlandı.
Eğitim/doğrulama süreleri sırasıyla 34,75 / 41,15 / 40,68 dakika;
tepe ayrılmış VRAM 7,77 / 10,04 / 9,32 GiB. Adapter dosyaları sonlu
ağırlıklar açısından doğrulandı. Özet: `results/sadhana/finetuning_run/pilot_summary.json`.
Geçici servis tamamlanınca systemd listesinden kaldırılır; durum dosyası korunur.
Ollama GGUF dosyalarına ek olarak eğitim için HF ağırlıkları indirilir.
Güncel aşama, loss, adım ve VRAM bilgisi `results/sadhana/finetuning_run/status.json`
içindedir. `loading_weights` aşaması henüz bir eğitim adımı yapıldığını göstermez.

```bash
systemctl --user status consumercomplaint-qlora-n500-s1 --no-pager
python utils/finetuning_status.py
python -m json.tool results/sadhana/finetuning_run/status.json
journalctl --user -u consumercomplaint-qlora-n500-s1 -f
```

Aktif kuyruk varken aşağıdaki bağımsız eğitim komutlarını başlatmayın.

```bash
python utils/train_qlora.py --model gemma4 --train-size 500 --seed 1
python utils/train_qlora.py --model qwen3 --train-size 500 --seed 1
python utils/train_qlora.py --model phi4 --train-size 500 --seed 1
```

NF4, BF16, rank 8, batch 1, gradient accumulation 16, 512 token,
3 epoch; yalnızca assistant cevabı loss'a katılır. Checkpoint seçimi
doğrulama loss'una dayanır. Model commit'i, süre ve tepe VRAM kaydedilir.
Pilot başarılıysa aynı ayarlarla 2.000/10.000/20.000 kayıt ve seed 1/2/3
matrisine genişletilebilir. Tamamlanan eğitimlerin raporu
`checkpoints/MODEL/n500/seed1/training_report.json`, adapteri aynı dizindeki
`adapter/` altında oluşur. Pilot eğitimleri ve üç modelin dengeli
500-karakter testlerinde adapterli/adaptersiz karşılaştırması tamamlandı.
Tüm 12 koşul ve seri gecikme ölçümleri de 6 Ekim 2026 saat 18:24'te tamamlandı.
Kesilen uyumlu eğitimler `--resume` ile son checkpoint'ten sürdürülebilir.
Kuyruğu yeniden başlatmak için, mevcut servis durmuşken:

```bash
python utils/run_finetuning_queue.py --train-size 500 --seeds 1
```

Kuyruk tamamlanmış eğitimleri atlar ve veri/test SHA256 kayıtlarını kontrol
eder. Bu kullanıcı servisi sohbet sona erince çalışmaya devam eder;
bilgisayar yeniden başlatılırsa yukarıdaki komutla sürdürülmelidir.

## Adil zero-shot / adaptasyon karşılaştırması

6 Ekim 2026'da her model aynı 3.000 dengeli, 500-karakter test kaydında
değerlendirildi. Eğitim: 500 dengeli kayıt, eğitim seed'i 1; ayrıca ortak
2.000 doğrulama kaydı. Weighted F1:

| Model | HF adaptersiz | HF adapterli | Artış (yüzde puan) | Artış için %95 CI |
|---|---:|---:|---:|---:|
| Gemma 4 12B | 0.7520 | 0.7803 | +2.83 | [+1.75, +3.96] |
| Qwen3 14B | 0.7099 | 0.7891 | +7.92 | [+6.55, +9.34] |
| Phi-4 14B | 0.6883 | 0.7845 | +9.62 | [+8.21, +11.07] |

8 Ekim kontrolünde **72.000 test tahmini ve 600 seri ölçümün tamamı**
doğrulandı. Train/validation/test/adapter ve tamamlanan tahmin SHA256
değerleri, metin/etiket/indeks eşleşmeleri, ham cevaplar, eşit çiftli batch
boyutları, pooled skorlar ve seri süre/uyum değerleri kontrol edildi.
36.000 adapterli test çıktısında geçersiz cevap yoktur.
Tüm 12 model/koşul/uzunluk karşılaştırmasının F1 artış güven aralığı
pozitiftir. Tam rapor:
[`results/sadhana/qlora_comparison/report.md`](results/sadhana/qlora_comparison/report.md).
İlk ana karşılaştırmanın tarihsel anlık görüntüsü
`qlora_primary_comparison_20261006/` altında korunur.

Seri ortalamalar (base → adapter): Gemma 0.475 → 0.647 s;
Qwen 0.421 → 0.539 s; Phi 0.487 → 0.435 s. Fine-tuning her modeli
hızlandırmaz. 600 seri/batch karşılaştırmasının üçünde etiket farkı vardır;
ayrıntılar `serial_batch_disagreements.csv` içinde, kaynak hash ve token
sayıları eşleşir. F1 tam batched tahminlerden hesaplanır.

Makaleye yöntem, sonuç ve kaynak tüketimi tabloları, şekil, sınıf bazlı
ek sonuçlar, başlık/özet/katkılar, tartışma ve sonuçlar eklendi.
Ana PDF 24, ek PDF 15 sayfadır. Dergi biçimi ve 20 sayfaya kısaltma
son aşamaya bırakıldı; gönderim yapılmadı.

Tamamlanan sonuçları doğrulama ve makale tablolarını yeniden üretme:

```bash
python utils/verify_qlora_evaluation.py
python utils/export_qlora_manuscript.py
```

Üç 500 kayıt/seed 1 adapterinin HF karşılaştırması `utils/run_qlora_evaluation_queue.py`
ile yürütülür. Önce üç modelin dengeli 500 karakter testleri, ardından kalan
koşullar tamamlanır. Aynı NF4 model örneğinde adapter kapalı/açık tutulur;
her çağrı bağımsız KV cache kullanır. En fazla batch 32 ile tüm 12 test dosyası
değerlendirilir. Gecikme ayrıca dengeli seed-1/500-karakter testinden her
sınıfa 20 kayıt alınan 100 örnekte, batch 1 ile ölçülür. Toplu çıkarımın
kayıt başına süresi seri gecikme olarak raporlanmaz.
CUDA bellek aşımında aynı eşleşmiş batch iki mod için birlikte küçültülür;
Phi-4 uzun girişlerde bellek aşımından sonra batch 16 ile devam eder.
Her tahminin gerçek batch boyutu kaydedilir. Tamamlanan eşleşmiş tahminler
korunur; kesinti sırasında yalnızca bir modu yazılmış kayıtlar birlikte
tekrar üretilir. Bellek sınırı `evaluation_run/MODEL/batch_limit.json` içindedir.

```bash
python utils/finetuning_status.py
python utils/run_qlora_evaluation_queue.py --batch-size 32 --serial-samples 100
```

Aktif değerlendirme servisi varken ikinci kuyruk başlatmayın. İlerleme
`results/sadhana/evaluation_run/status.json`, adaptersiz tahminler
`results/sadhana/hf_base/`, adapterli tahminler `results/sadhana/qlora/`,
karşılaştırma `results/sadhana/qlora_comparison/` altındadır. Model dosyaları
offline önbellekten alınır. JSONL kayıtları kesilen değerlendirmeyi sürdürür;
tamamlanan tahminler tekrar üretilmez. Pooled tablo test seed'leri tamamlanana
kadar kısmi durumu gösterir. Son rapor 5.000 eşleştirilmiş metin-kümesi bootstrap
örneği, unique-input McNemar/Holm ve sınıf bazlı F1 içerir. Eğitim seed'i hâlâ
yalnızca 1'dir; test güven aralıkları eğitim seed değişkenliğini ölçmez.

Eski Ollama GGUF sonuçları ile yeni HF NF4 sonuçları, nicemleme ve şablon
farklarını da içerir. Adaptasyon etkisi için aynı commit/runtime ile
adaptersiz ve adapterli tahminleri ayrıca üretin; geçmiş Ollama sonuçlarını
bağlam olarak raporlayın.

```bash
python utils/evaluate_qlora.py --model qwen3 --revision MODEL_COMMIT --output results/sadhana/hf_base/qwen3
python utils/evaluate_qlora.py --adapter checkpoints/qwen3/n500/seed1 --output results/sadhana/qlora/qwen3_n500_s1
```

Commit'i `training_report.json` içinden alın. HF değerlendirmede thinking
kapalıdır, full çıktı ve output-limit bilgisi saklanır. Ollama thinking-on/off
deneyi ayrıca gereklidir: önceki CSV'lerde ayrı thinking alanı,
`done_reason` ve token sayaçları yoktur.

Güncel bulgular ve gönderim öncesi kalan işler:
[`SPRINGER/sadhana_durum_raporu.md`](SPRINGER/sadhana_durum_raporu.md).

## Hakem denetimi — 8 Ekim 2026

[Detaylı inceleme](SPRINGER/hakem_degerlendirmesi_2026-10-08.md),
veri bütünlüğü, kaynakça, istatistik ve tablo/şekil denetimini içerir.
Özet donanım ayrıntılarından arındırıldı. Denetim kayıtları
`results/sadhana/reviewer_audit/` altında tutulur.

```bash
python utils/verify_qlora_evaluation.py
python utils/export_legacy_manuscript.py --check
python utils/audit_manuscript.py
```

Eski tablo hücrelerini tahminlerden yenilemek için
`python utils/export_legacy_manuscript.py` kullanılır. Bu işlem tahminleri
değiştirmez. Çizimlerde geçersiz yanıtlar ayrı hata sütununda kalır.
Yeni LaTeX input dosyaları ve kullanılan QLoRA şekli `.gitignore`
istisnalarıyla görünür kılındı; son kamuya açık sürüm için commit/push
yine gereklidir.


## Ek eğitim seed'leri ve 500 etiket kontrolü

8 Ekim 2026'da aynı 500 eğitim / 2.000 doğrulama kaydı için ek
deneyler başlatıldı. Üç-epoch ve doğrulama ile erken durdurmalı
DistilBERT kontrollerinin üçer seed'i tamamlandı. Gemma 4, Qwen3 ve
Phi-4'ün ikinci/üçüncü eğitim seed'leri ve eşleşmiş base/adapter
değerlendirmeleri kuyrukta çalışıyor. Mevcut HF ağırlıkları offline
yüklenir. İlk pilot sonuçları yerlerinde korunur.

```bash
source /home/tan/anaconda3/etc/profile.d/conda.sh
conda activate ConsumerComplaint
python utils/finetuning_status.py
systemctl --user status consumercomplaint-replication-n500.service
```

Deney planı ve durum: `results/sadhana/replication_20261008/`.
`report.md` yalnızca tamamlanıp doğrulanmış fit'leri gösterir.
Tamamlanan fit'ler ayrı ayrı değerlendirilir; üç eğitim seed'inin
ortalama/standart sapması ile test belirsizliği ayrı raporlanır.
Zamansal test için tarih ve CFPB kayıt kimliği bulunmuyor.

Kuyruk durmuşsa ve başka değerlendirme/eğitim kuyruğu yoksa devam:

```bash
python utils/run_replication_queue.py --resume
```

Ara raporu yenilemek için `python utils/summarize_replications.py
--allow-partial --bootstrap 0` kullanılır. Tam kuyruk sonunda
`utils/verify_qlora_evaluation.py` her seed'i doğrular;
`utils/summarize_replications.py` 36 base/adapter ve ayrı 72
etiket-eşit classifier karşılaştırmasını hesaplar.
`utils/export_replication_manuscript.py` ayrı dizinde derleme, atıf ve
nesne denetimi başarılı olunca makaleyi yedekleyerek günceller.
Bu aşamadan sonra eski tek-seed `utils/export_qlora_manuscript.py`
komutu üç-seed tablolarını yeniden üretmek için kullanılmamalıdır;
yeni exporter'ın çıktıları esas alınır. Dergi biçimi son aşamadadır.


## Yeniden başlatma için durdurma — 8 Ekim 2026

Kuyruk kullanıcı isteğiyle yeniden başlatma öncesinde durduruldu.
Altı DistilBERT fit'i, Gemma seed-2/seed-3 eğitimleri ve Gemma seed-2
değerlendirmesi tamamlandı. Gemma seed-3 değerlendirmesinde 8.640/24.000
tahmin kaydedildi; iki modun eşleşmesi, kaynak metin/etiket/indeksleri
doğrulandı. Ayrıntılar `reboot_checkpoint.json` içinde.

Geçici servis kalıcı kullanıcı servisine dönüştürüldü:
`~/.config/systemd/user/consumercomplaint-replication-n500.service`.
Servis etkin, şu anda durmuş durumdadır. Sonraki açılış/oturumda
`utils/run_replication_queue.py --resume` ile otomatik devam eder.
Tamamlanan fit'ler atlanır; Gemma seed-3 kaydedilmiş tahminlerinden
devam eder. Servisin kopyası `recovery.service` dosyasındadır.

Kontrol: `python utils/finetuning_status.py`; manuel başlatma:
`systemctl --user start consumercomplaint-replication-n500.service`.
Servis logu: `results/sadhana/replication_20261008/service.log`.


## Üç-seed deneyler ve makale aktarımı tamamlandı — 8 Ekim 2026

Kuyruğun 23/23 işi 22:10'da başarıyla tamamlandı. Üç LLM'nin üçer
fit'i için 216.000 test ve 1.800 seri çıktı doğrulandı. Altı yeni
DistilBERT kontrolü de tamamlandı. Dengeli 500-karakter testinde
weighted F1 (üç eğitim seed'inin ortalama ± örnek standart sapması):
Gemma 0.7841 ± 0.0098; Qwen3 0.7833 ± 0.0092;
Phi-4 0.7857 ± 0.0011. Ortalamaların yakınlığı model üstünlüğü kanıtı değildir.

Makaleye üç-seed yöntem/özet/sonuç/tablo/şekil ve classifier kontrolleri
işlendi; ana PDF 25, ek PDF 14 sayfa. Özet 155 kelime ve donanım
terimi içermiyor. Atıf/nesne kontrolü başarılı. 36 base/adapter
karşılaştırmasının 34'ünde F1-artış aralığı pozitif; kalan iki karşılaştırma
dahil tüm sonuçlar raporlandı. Eğitim servisi durmuş ve tamamlanmış
kuyruğun tekrar açılışta çalışmasını önlemek için otomatik başlatma kapatılmıştır.

Son rapor: `results/sadhana/replication_20261008/report.md`.
Dergi biçimi, metni kısaltma ve son gönderim kontrolleri devam eden işlerdir.
Gönderim yapılmadı.


## Üç-seed sonrası hakem denetimi

[İkinci ayrıntılı rapor](SPRINGER/hakem_degerlendirmesi_uc_seed_2026-10-08.md)
mevcut sonuçların sınırlarını, düzeltilen metin hatalarını ve şema/sayfa
denemesini içerir. Güncel ana/ek PDF 25/15 sayfa. Denetim:
`results/sadhana/reviewer_audit/post_replication/`.
Yeni `replication_class_tables.tex` input dosyası üç fit'in sınıf-F1
ortalama/std değerlerini gösterir. Referee/iki-sütun denemeleri ayrı
dizindedir; Sādhanā'ya uygunluğu doğrulanmış gönderim PDF'leri değildir.

## Claude ve DeepSeek geri bildirimleri — 8 Ekim 2026

[Ayrıntılı değerlendirme](SPRINGER/claude_deepseek_feedback_degerlendirmesi.md). Ana mesaj ve toplam etiket bütçesi
açıklaştırıldı; classifier yöntemleri setup bölümüne taşındı. Mevcut
sonuçlardan bütçe–F1 grafiği, deduplicated-pool barı ve fit noktaları
eklendi. Seri çıktı tokenları analiz edildi; ek tablo oluşturuldu. Yeni
training/inference yapılmadı. Son ana/ek PDF: **26/15 sayfa**, 39 kaynak;
tanımsız referans/atıf veya taşma uyarısı yok. Yeni GPU deneyleri için
öncelik ve gerekçeler değerlendirme dosyasında.

## Thinking kontrolü — 9 Ekim 2026

Bu revizyonun ek deney kapsamı kullanıcı kararıyla Gemma 4 / Qwen3
thinking açık–kapalı karşılaştırmasıdır. Aynı 500 dengeli test girdisi,
aynı ≤500-karakter koşulu, aynı prompt ve her iki modda aynı çıktı
bütçesi kullanılıyor. Eğitim veya başka veri/prompt/classifier deneyleri
başlatılmadı. Canlı durum: `results/sadhana/thinking_control_20261009/status.json`;
`python utils/finetuning_status.py` ilerlemeyi gösterir. Servis:
`consumercomplaint-thinking-control.service`; kesintide tamamlanan
JSONL kayıtlarından devam eder.

Teknik pilot, aynı bir Gemma-thinking girdisinin 1.024/2.048/4.096
çıktı tokenında da son yanıta ulaşmadığını gösterdi. Ana test öncesinde
kaydedilen protokol ek notuyla bütçe 4.096 tokenda sabitlendi; sınıra
ulaşan ana yanıtlar hata sayılacak. Bu token bütçesi şikâyet metninin
karakter uzunluğu değildir. Tüm pilot girdileri ana testten dışlanmıştır.
Sonuçlar, tam doğrulama ve ayrı PDF derlemesi sonrasında makaleye aktarılır.

## Thinking kontrolü tamamlandı — 9 Ekim 2026, 11:34 (Türkiye)

2.000/2.000 ana tahmin tamamlandı. Aynı 500 benzersiz, sınıf başına 100
ve ≤500 karakter girdi iki modelin iki modunda kullanıldı. Bağımsız son
kontrol ham JSONL kayıtlarını yeniden ayrıştırdı; kaynak/seçim/tahmin
hash'leri, eşleştirilmiş istekler, sklearn metrikleri ve exact
McNemar/Holm sonuçları doğrulandı. Eksik veya çift kayıt ve pilot/test
çakışması yok. Rapor: `results/sadhana/thinking_control_20261009/completion_check.json`.

| Model | Thinking kapalı F1 | Thinking açık F1 | Ortalama süre kapalı → açık |
|---|---:|---:|---:|
| Gemma 4 12B | 0.7298 | 0.7447 | 0.28 → 6.19 s |
| Qwen3 14B | 0.6926 | 0.7449 | 0.25 → 8.43 s |

Gemma'nın weighted-F1 değişimi +1.49 yüzde puan; düzeltilmemiş eşleşmiş
%95 güven aralığı [-1.31, +4.19], dolayısıyla iyileşme kanıtı yeterli değil.
Thinking açıkken 14/500 (%2.8) yanıt ortak 4.096 çıktı-token sınırına
ulaşıp son yanıt üretmedi; tamamı hata olarak değerlendirmede tutuldu.
Qwen3'ün değişimi +5.23 yüzde puan; aynı tür aralık [+1.91, +8.51].
İki accuracy testi için Holm-düzeltilmiş p değerleri sırasıyla 0.7798 ve
0.004372. Bu p değerleri F1 testleri veya modeller arası üstünlük testleri
değildir. Sonuçlar mevcut artefaktlara ve sabit alt kümeye koşulludur;
tarihsel çalışmaların kaydedilmemiş thinking ayarlarını geriye dönük belirlemez.

Sonuçlar `SPRINGER/thinking_control.tex` ve
`SPRINGER/thinking_control_supplementary.tex` üzerinden ana/ek metne
ve PDF'lere işlendi. Ana PDF 27, ek PDF 15 sayfa; atıfsız nesne,
tanımsız atıf/referans, yinelenen etiket veya taşma uyarısı yok.
Deney servisi durmuş ve açılışta yeniden çalışması kapatılmıştır.

## Son hakem taraması ve kısaltma — 9 Ekim 2026

Ana metin baştan tarandı; tekrarlar ve koşulsuz deployment önerileri
budandı, uzun paragraflar bölündü. Thinking bulguları özet/katkı/tartışma/
sonuca işlendi; test-subset/eğitim-seed ve accuracy/F1 testleri ayrıldı.
Aynı şablonda ana PDF 27 → 23 sayfa; ek 15 sayfa. Ana metin sözcükleri
texcount ile 7.940 → 5.931 (yaklaşık %25 azalma); özet 177 sözcük.
Tabloların sayısal satırları ve sonuç dosyaları korunuyor; referans/nesne
ve derleme kontrolleri başarılı. Yeni deney başlatılmadı.
Ayrıntılı rapor: `SPRINGER/hakem_degerlendirmesi_editorial_2026-10-09.md`.
Denetim: `results/sadhana/reviewer_audit/editorial_20261009/`.

## Deney kolları ve sınıf düzeyi yorum — 9 Ekim 2026

Experimental Setup'a üç LLM deney kolunun farklı amaç ve karşılaştırma
sınırlarını açıklayan paragraf eklendi. Thinking sonuçlarına Qwen'in beş
sınıftaki pozitif nokta değişimleri ile Gemma'nın mortgages/loans ve
credit-card düşüşleri eklendi; bunların sınıf bazında anlamlılık testi
olmayan betimsel gözlemler olduğu belirtildi. Değerler CSV'den doğrulandı.
Son ana PDF **24 sayfa**, ek PDF 15 sayfa; derleme/atıf kontrolü başarılı.
Denetim: `results/sadhana/reviewer_audit/scope_clarifications_20261009/`.
