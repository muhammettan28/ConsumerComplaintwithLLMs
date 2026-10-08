# Consumer complaint classification: Sādhanā revision

Mevcut zero-shot tahminlerini değiştirmeden doğrulama, istatistiksel analiz ve
aynı test kayıtları üzerinde supervised karşılaştırma için betikler.

## Ortam

```bash
source /home/tan/anaconda3/etc/profile.d/conda.sh
conda activate ConsumerComplaint
python -m pip check
```

Jupyter kernel: `Python (ConsumerComplaint)`. Kurulan sürümler
`requirements-lock.txt`, conda ortamı `environment-consumercomplaint.yml` içinde.
Başka bir makinede CUDA/PyTorch paketleri GPU ile uyumlu kurulmalıdır.

## Tamamlanan analizleri yeniden üretme

```bash
python sadhana_audit.py --bootstrap 5000
python prepare_finetuning_data.py
python run_classical_baseline.py
python train_encoder_baseline.py --train-size 20000 --seed 1
python train_encoder_baseline.py --train-size 20000 --seed 2
python train_encoder_baseline.py --train-size 20000 --seed 3
python summarize_sadhana_baselines.py
```

Veri hazırlama ve eğitim betikleri mevcut çıktıları üzerine yazmayı reddeder.
Yeni deney için çıktı dizinini ayrıca arşivleyin. Mevcut test altkümelerini
`build_experiment_subsets.py` ile yeniden üretmeyin: dosyaları değiştirir ve
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
python train_qlora.py --model gemma4 --preflight
python train_qlora.py --model qwen3 --preflight
python train_qlora.py --model phi4 --preflight
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
python finetuning_status.py
python -m json.tool results/sadhana/finetuning_run/status.json
journalctl --user -u consumercomplaint-qlora-n500-s1 -f
```

Aktif kuyruk varken aşağıdaki bağımsız eğitim komutlarını başlatmayın.

```bash
python train_qlora.py --model gemma4 --train-size 500 --seed 1
python train_qlora.py --model qwen3 --train-size 500 --seed 1
python train_qlora.py --model phi4 --train-size 500 --seed 1
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
python run_finetuning_queue.py --train-size 500 --seeds 1
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
python verify_qlora_evaluation.py
python export_qlora_manuscript.py
```

Üç 500 kayıt/seed 1 adapterinin HF karşılaştırması `run_qlora_evaluation_queue.py`
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
python finetuning_status.py
python run_qlora_evaluation_queue.py --batch-size 32 --serial-samples 100
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
python evaluate_qlora.py --model qwen3 --revision MODEL_COMMIT --output results/sadhana/hf_base/qwen3
python evaluate_qlora.py --adapter checkpoints/qwen3/n500/seed1 --output results/sadhana/qlora/qwen3_n500_s1
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
python verify_qlora_evaluation.py
python export_legacy_manuscript.py --check
python audit_manuscript.py
```

Eski tablo hücrelerini tahminlerden yenilemek için
`python export_legacy_manuscript.py` kullanılır. Bu işlem tahminleri
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
python finetuning_status.py
systemctl --user status consumercomplaint-replication-n500.service
```

Deney planı ve durum: `results/sadhana/replication_20261008/`.
`report.md` yalnızca tamamlanıp doğrulanmış fit'leri gösterir.
Tamamlanan fit'ler ayrı ayrı değerlendirilir; üç eğitim seed'inin
ortalama/standart sapması ile test belirsizliği ayrı raporlanır.
Zamansal test için tarih ve CFPB kayıt kimliği bulunmuyor.

Kuyruk durmuşsa ve başka değerlendirme/eğitim kuyruğu yoksa devam:

```bash
python run_replication_queue.py --resume
```

Ara raporu yenilemek için `python summarize_replications.py
--allow-partial --bootstrap 0` kullanılır. Tam kuyruk sonunda
`verify_qlora_evaluation.py` her seed'i doğrular;
`summarize_replications.py` 36 base/adapter ve ayrı 72
etiket-eşit classifier karşılaştırmasını hesaplar.
`export_replication_manuscript.py` ayrı dizinde derleme, atıf ve
nesne denetimi başarılı olunca makaleyi yedekleyerek günceller.
Bu aşamadan sonra eski tek-seed `export_qlora_manuscript.py`
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
`run_replication_queue.py --resume` ile otomatik devam eder.
Tamamlanan fit'ler atlanır; Gemma seed-3 kaydedilmiş tahminlerinden
devam eder. Servisin kopyası `recovery.service` dosyasındadır.

Kontrol: `python finetuning_status.py`; manuel başlatma:
`systemctl --user start consumercomplaint-replication-n500.service`.
Servis logu: `results/sadhana/replication_20261008/service.log`.
