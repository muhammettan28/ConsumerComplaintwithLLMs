# Perplexity geri bildirimi: kanıt ve düzeltme değerlendirmesi

9 Ekim 2026. İncelenenler: gönderilen geri bildirimin tamamı, güncel ana metin
ve ek materyal, veri hazırlama/eğitim/değerlendirme/istatistik kodları ve
kayıtlı veri/sonuç dosyaları. Yeni eğitim veya model çıkarımı yapılmadı.

## Veri sızıntısı hakkında sonuç

Perplexity bir eğitim–test sızıntısı örneği göstermiyor. İlk maddesi, test
koşullarının örtüşmesini ve veri akışının daha açık anlatılmasını istiyor.
Bu iki konu ayrılmalıdır: test koşullarının örtüşmesi bağımsızlık/genelleme
sorunudur; eğitim veya validation ile testin karıştığını tek başına göstermez.
Yeni denetimde, aşağıdaki tanımlı kontrollerde sızıntı bulunmadı.

| Yeniden çalıştırılan kontrol | Sonuç |
|---|---|
| Kaynak CSV ve manifestteki 17 split dosyasının SHA-256 doğrulaması | Hepsi eşleşiyor |
| Dört eğitim bütçesi–test normalleştirilmiş tam metin kesişimi | Her bütçede 0 |
| Dört eğitim bütçesi–test ortak 250 karakter öneki | Her bütçede 0 |
| Dört eğitim bütçesi–validation tam metin/önek kesişimi | Her bütçede 0 |
| Eğitim–validation yerel kaynak satırı kesişimi | Her bütçede 0 |
| Validation–test tam metin/önek kesişimi | 0 |
| Eğitim/validation kayıtlarının kaynak satırına göre metin, etiket ve hash uyumu | Uyumlu |
| 500/2.000/10.000 kayıtlık eğitim kümelerinin 20.000'lik kümenin dengeli önekleri olması | Doğrulandı |
| 20.000 eğitim–test, validation–test ve 20.000 eğitim–validation yakın kopya taraması | Tanımlı yöntemle 0 eşleşme |
| 250/500 dosyalarında etiket, kayıt sırası ve kısaltma eşleşmesi | Altı çiftte doğrulandı |

Yakın kopya kontrolü yeniden çalıştırıldı: 128-permutation MinHash LSH aday
bulma ve sözcük üçlüleri üzerinde tam Jaccard ≥0,85 doğrulaması.
Yaklaşık aday bulma tüm yakın kopyaların veya anlamsal parafrazların yokluğunu
kanıtlamaz. Kamuya açık CFPB metinlerinin temel modellerin ön eğitiminde
bulunmuş olması da bu split denetimiyle dışlanamaz.

TF-IDF sözlüğü ve IDF yalnızca ilgili eğitim kümesinde öğreniliyor;
validation/test dönüşümünde yeniden fit edilmiyor. SVM'nin C seçimi
validation F1, DistilBERT checkpoint seçimi validation F1, adapter
checkpoint seçimi validation completion-only loss üzerinden yapılıyor.
HF test girdisi oluşturulurken gerçek sınıf etiketi prompt'a eklenmiyor.

### Gerçek anlatım hatası: model seçimi ile checkpoint seçimi

Introduction'daki “keeping test outcomes outside model selection” ifadesi
fazla genişti. Gemma/Qwen/Phi model seçimi benchmark sonuçları ve dağıtım
özelliklerinden yararlanıyor; bunu testten tamamen bağımsız bir model
seçimi gibi sunmak doğru değil. Cümle, yalnızca fit içindeki checkpoint'in
ayrı validation kayıtlarında seçildiğini söyleyecek şekilde düzeltildi.
Methods ayrıca model seçiminin keşif niteliğinde olduğunu ve uyarlama
sonuçlarının seçilen modellere koşullu olduğunu belirtiyor.

Bu ayrım split sızıntısı kontrolleriyle kapanmaz: aynı benchmark üzerinde
model tercihi ve raporlama yapıldığından bütün açık ağırlıklı modeller için
bağımsız doğrulama veya seçim sonrası üstünlük iddiası kurulamaz. Eşleştirilmiş
base/adapter farkları ise seçilmiş modeller ve kayıtlı testler için raporlanır.

## Bütün geri bildirim maddelerinin değerlendirmesi

| Madde | Değerlendirme ve işlem |
|---|---|
| Veri akışı, sınıf adetleri ve seed'ler | Ek materyale “Saved splits and statistical units” bölümü ve split tablosu eklendi. Benchmark master seed 42; altküme sıra shuffle seed'leri 42/43/44; supervised split seed'i 42; eğitim seed'leri 1/2/3 açıklandı. |
| CFPB kayıt kimliği ile ayırma | Elimizdeki alan yerel CSV satır indeksidir; CFPB şikâyet kimliği gibi sunulmadı. Split kimliği normalleştirilmiş metin hash'i ve önek kontrolleriyle tanımlandı. |
| A/B ve 250/500 bağımsızlığı | A/B tam metin kesişimi 2.067, birleşimi 3.933. 250/500 aynı sıralı kayıtlar. Bunlar bağımsız deney tekrarı olarak yorumlanmıyor. |
| Zamansal veya kaynak gruplarına genelleme | Rastgele corpus değerlendirmesinin bunları ölçmediği ek bölümde netleştirildi. Çıkarılması istenen limitation cümlesi geri eklenmedi. |
| “Label-matched” ve toplam etiket bütçesi | Karşılaştırmada gerçekten aynı 500 eğitim ve 2.000 validation kaydı kullanılıyor. Dolayısıyla aynı toplam kayıt sayısı var; fakat validation amacı, hesaplama ve fit sayıları eşdeğer değil. Bu ayrım ana metin ve ekte açıklandı. |
| SVM tek fit, diğerleri üç fit | Açıkça belirtildi. SVM nokta tahmini üç seed standart sapması gibi sunulmuyor. Ek eğitim yapılmadı. |
| Üç seed farklı eğitim örneklemi değil | Aynı eğitim/validation kayıtlarında optimizasyon rastgeleliği olduğu yöntem, ek materyal ve sınırlılıklarda netleştirildi. |
| Bootstrap birimi ve çatışan etiketler | Birim, aynı koşul/uzunlukta üç dosyada görülen aynı normalleştirilmiş metin. 500 karakterde her koşulda 3.000; 250 karakterde A 2.997, B 2.986 küme. Bu gruplarda çatışan etiket yok. Kaynak havuzundan çıkarılan 133 çelişkili grup ile olası McNemar dışlamaları birbirinden ayrıldı. |
| 34/36 anlamlı sonucu | Mevcut 36 satırdan tekrar doğrulandı: 34 pozitif F1 aralığı, 34 Holm anlamlı doğruluk testi. 36 bağımsız tekrar iddiası yok. F1 aralıkları düzeltilmemiş, Holm p doğruluk içindir. |
| Etki, güven aralığı ve p tam tablo | Üç seed'in 36 satırlık ek tablosuna mevcut CSV'deki F1 farkı %95 aralıkları eklendi. Base/adapted F1 ve Holm doğruluk p değerleri korunuyor. |
| Ticari API karşılaştırmasının bağlamsal olması | Eşleştirilmiş üstünlük veya eşdeğerlik kurulmadığı mevcut metinde açık. Aynı test kayıtları doğrulanmış gibi sunulmadı. |
| Phi-4 0,54 s ve F1 0,706 | Rakam değiştirilmedi; bunun A koşulu, 500 karakter, istek başına ortalama GGUF süresi olduğu açıklaştırıldı. |
| GGUF ölçüm sınırı | Kodda timer HTTP header dönüşünde, body okuma/JSON çözümlemesinden önce duruyor. Son deneme süresi kaydediliyor; önceki denemeler ve beklemeler dahil değil. Ek materyalde bu sınır, HF ve thinking sınırlarından ayrıldı. Hata/geçersiz çıktı satırları ortalamalarda tutuluyor. |
| Table 8 “cost” başlığı | Üçüncü taraf API ücreti olarak daraltıldı; sütun “API fee/1k”. Ücretlerin referans çalışmadan geldiği ve yerel donanım/enerji/bakımı içermediği caption'da da açık. |
| Azınlık sınıfı tanımı | B'de credit reporting %51,1; diğer dört sınıf %10,0–18,2 destekli olarak tanımlandı. Retail banking de dahil. |
| Eşit-prior F1 yeniden ağırlıklandırması | Ek materyalde C_ij / n_i dönüşümü verildi; kesirli sayılar korunuyor, metrik hesabından önce yuvarlama yok. Tahminler değişmiyor. |
| “Aynı prompt” | Aynı görev talimatı/taksonomi, deney koluna özgü template/thinking/üretim sınırı şeklinde düzeltildi. Gerçek token dizilerinin aynı olduğu iddia edilmiyor. |
| Gemma 2/Gemma 4 atfı | Model aileleri cümlesi Gemma 2 ve Gemma 4'ü kendi kaynaklarına birlikte bağlıyor. Gemma 4 adı ve thinking özelliği resmi model kartıyla kontrol edildi. |
| Latency dağılımı, eşzamanlılık ve TCO | Bu ölçümler eklenmedi. Conclusion'da concurrent throughput, tail latency ve toplam sahip olma maliyetinin değerlendirilmediği açıklaştırıldı. Var olan serial protokollerinin sınırları korunuyor. |
| Sınıf bazında precision/recall/F1 ve macro-F1 aralıkları | Bu değerler mevcut per-fit CSV'lerde bulunuyor; ek tablo sınıf F1 özetlerini içeriyor. Her sınıf için yeni güven aralığı hesaplanmadı; varmış gibi yazılmadı. |
| Etik ve yönetişim | Public corpus, katılımcı toplama/müdahale olmamasıyla gerekçelendirildi. Public metinlerin tanımlayıcı/finansal bilgi içerebileceği belirtildi; anonimlik denetimi veya veri koruma onayı uydurulmadı. |
| Operasyonel yönlendirme | Çekimser kalma, insana aktarma ve asimetrik yanlış yönlendirme maliyetlerinin ölçülmediği kısa bir cümleyle belirtildi. |
| Tek dil ve LoRA ayarları | İngilizce kapsamı ve rank/hedef modül seçimlerinin ablation yapılmadığı eklendi. |
| Tüm bibliyografya | Atıf anahtarları/çapraz referanslar kontrol edildi ve somut Gemma atfı düzeltildi. 39 kaynağın tüm DOI/yazar/sayfa metadatası bağımsız olarak yeniden doğrulanmış değildir. Güllü DOI sayfasının içeriği bu turda alınamadı; yeni ayrıntılı literatür iddiası eklenmedi. |

## Açık kalan gönderim işleri

Sonraki Git düzenlemesi (9 Ekim 2026): `dataset/finetuning/` ignore'dan
çıkarıldı ve split/manifest dosyaları Git index'ine eklendi. Aşağıdaki
ilk madde inceleme anındaki durumu kaydeder; ham model yanıtları ve
üretilmiş tahminler artık ayrı sonuç arşivinde paylaşılacak dosyalardır.
Kalıcı arşiv veya çevrimiçi yayın tamamlanmış değildir.

1. **Erişilebilir deney paketi:** Yerel manifest mevcut fakat
   `dataset/finetuning/` gitignore kapsamında. Thinking betiği ve yeni audit
   betiği henüz untracked. Yerelde mevcut olmak, hakemin çevrimiçi erişebildiğini
   göstermez. Commit/sürüm etiketi ve kalıcı arşiv yayımlanmış gibi yazılmadı;
   bu turda push veya dışarı veri yükleme yapılmadı.
2. **Sayfa/şablon:** Ana PDF 24, ek PDF 16 sayfa. Springer'ın erişilebilir
   güncel yazar talimatları araştırma/review/survey için 20 sayfa sınırı
   belirtiyor. Farklı draft şeması bu gerekliliği otomatik kaldırmıyor;
   nihai biçimlendirme aşamasında çözülmeli. Bu tur şablon değiştirilmedi.
3. **ORCID:** Resmi talimatta kuvvetle öneriliyor, zorunlu denmiyor.
   Doğrulanmış ORCID verilmediğinden kimlik uydurulmadı.
4. **Daha geniş bilimsel doğrulama:** Daha büyük LLM eğitim bütçesi,
   bağımsız eğitim örneklemleri, temporal/source-group holdout ve gerçek
   servis yükü ölçülmedi. Bunları açıklamak deneysel olarak ölçmüş olmak
   anlamına gelmez; kapsam buna göre sınırlandırıldı.

## Kontrol dosyaları

### Takip JSON'larından learning curve kontrolü

Dokuz LLM fit'inin `training_report.json` dosyası ayrı ayrı okundu:
Gemma/Qwen/Phi × seed 1/2/3 için gerçek `train_size` her durumda 500.
İç içe `train_manifest.train_size = 20000` alanı mevcut eğitim havuzunu
tanımlar; LLM'nin 20.000 örnekle eğitildiğini göstermez. Kayıtlı
loss/epoch noktaları optimizasyon sürecini gösterebilir, fakat farklı
benzersiz eğitim örneği sayılarında test F1 eğrisini üretmez. Seçilen
checkpoint'in `trainer_state.json` geçmişi 64 adımda da bitebildiğinden
tam 96-adımlık loss eğrisi için eğitim logları da gerekir.

Gerçek bir LLM etiket-bütçesi eğrisi ve eşit-bütçe kesişimi bu kayıtlarla
tamamlanmış sayılamaz. Bu kapsam sınırı Abstract ve Conclusion'da
açıklaştırıldı. Abstract'a adaptation sonrasında azınlık sınıflarındaki
zayıflığın sürdüğü eklendi; Gemma'nın sıfır-atış nokta sıralamasının
kayıtsız thinking ayarına atfedilemeyeceği yöntem bölümünde belirtildi.
Güncel PDF 24 sayfa, abstract 187 sözcük; katkıların dördü de 2. sayfada.
Kanıt ve kontrol: `results/sadhana/reviewer_audit/learning_curve_scope_20261009/`.
Yeni eğitim veya çıkarım başlatılmadı.

- Yeniden kullanılabilir salt okunur denetim: `utils/audit_split_integrity.py`.
- Yeni kanıt: `results/sadhana/reviewer_audit/perplexity_20261009/split_integrity.json`.
- Değişiklik öncesi kaynak/PDF yedekleri ve korunan veri hash'leri aynı dizinde.
- Split/istatistik açıklaması: `SPRINGER/supplementary_material.tex`.
- Tam 36 satırlık etki/aralık/p tablosu: `SPRINGER/replication_supplementary_tables.tex`.

## Resmî kaynak kontrolleri

- [Springer Sādhanā gönderim sayfası](https://link.springer.com/journal/12046/submission-guidelines)
  üzerinden bağlı [2026 yazar talimatı PDF](https://media.springer.com/full/springer-instructions-for-authors-assets/pdf/367298_173709708_12046-Information-Authors_up_2026.pdf)
  indirildi: 20 sayfa, yaklaşık en çok altı anahtar kelime ve önerilen ORCID
  ifadeleri doğrulandı. IAS web sayfaları doğrudan erişimde 403 verdi.
- [Resmî Gemma 4 12B model kartı](https://huggingface.co/google/gemma-4-12B-it)
  isimlendirme ve yapılandırılabilir thinking desteğini doğruluyor.
  Güncel kart, kaydı bulunmayan bir benchmark isteğinin ayarını kanıtlamaz.
