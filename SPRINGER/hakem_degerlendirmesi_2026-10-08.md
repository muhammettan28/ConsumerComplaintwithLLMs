# Hakem değerlendirmesi — 8 Ekim 2026

İncelenen sürüm: `main.tex`, bütün `\input` dosyaları, ek materyal, kaynakça,
ham veri, kayıtlı test dosyaları, eski Ollama tahminleri ve yeni HF sonuçları.
Bu rapor, **inceleme öncesinde bulunan hataları**, **uygulanan düzeltmeleri**
ve **hâlâ açık bilimsel sınırlılıkları** ayırır. Bir kabul olasılığı tahmini değildir.

## Genel hakem görüşü

İnceleme öncesi sürüm için önerim **major revision** olurdu. Bunun nedeni
GPU bilgisinin özette bulunmasından çok kaynak künyelerindeki yanlışlıklar,
yeniden hesaplanan değerlerle uyuşmayan tablo hücreleri, karmaşıklık matrisi
çizimindeki hata ve anlamlılık iddialarıdır. Bunlar bu incelemede düzeltildi.
Geriye özellikle tek adapter eğitim seed'i, sınırlı etiket bütçesi kontrolü ve
tarihsel Ollama koşullarının eksik kaydı kalıyor. Mevcut bulgular pilot
adaptasyon ve yerel sınıflandırıcı seçimi olarak savunulabilir; geniş bir
üstünlük veya genellenebilirlik iddiası için yeterli değil.

Makalenin en güçlü sorusu, “Hangi LLM birinci?” yerine “Etiket mevcutken
üretken model kullanmanın doğruluk ve süre karşılığı nedir?” sorusudur.
SVM'nin yüksek bütçede daha başarılı ve çok hızlı olması çalışmayı
zayıflatmak zorunda değil: dürüst bir mühendislik seçim çalışmasının önemli
sonucudur. Buna karşılık yalnızca bilinen modelleri aynı veri üzerinde
sıralamak, yöntemsel yenilik bakımından sınırlı kalır. QLoRA'nın kendisi
bu makalenin yeni yöntemi değildir; katkı kontrollü karşılaştırma ve uygulama
kararlarıdır. Giriş ve sonuç bu sınırı korumalıdır.

## Öncelikli bulgular ve durumları

| Önem | Bulgu | Bu incelemede yapılan / kalan |
|---|---|---|
| Kritik | Bazı DOI/arXiv kayıtları yazılan başlık ve yazarlarla uyuşmuyor | Doğrulanan yanlış künyeler düzeltildi; doğrulanmamış kesin literatür yüzdeleri çıkarıldı |
| Yüksek | Karmaşıklık matrisleri geçersiz cevapları retail banking'e aktarıyor | Kod düzeltildi, makaledeki üç matris yeniden çizildi |
| Yüksek | Tablo ve metindeki bazı sayılar tahminlerden yeniden hesaplananlarla farklı | 22 model/koşul satırı ve altı sınıf-F1 satırı tahminlerden üretildi |
| Yüksek | Tek yönlü/post-hoc McNemar yorumu ve düzeltme ailesi sorunlu | İki yönlü exact test, 12 karşılaştırmalık Bonferroni ve keşifsel yorum kullanıldı |
| Yüksek | “500 etiket” yanında 2.000 validation etiketi kullanılıyor | Özet, yöntem ve yorumlarda açık; toplam 2.500 etiket maliyeti unutulmamalı |
| Yüksek | Adapter başına yalnızca bir eğitim seed'i var | Açık sınırlılık; bootstrap bunu telafi etmez |
| Yüksek | A/B testleri büyük ölçüde ortak kayıtlardan oluşuyor | 2.067 ortak kayıt ve 3.933 benzersiz uzun görünüm açıklandı |
| Orta | Dengesiz dağılımın kaynağı eksik anlatılmış | Tekrar çıkarma sonrası 72.611 kayıtlık havuzun oranlarıyla açıklanıp doğrulandı |
| Orta | S1 açıklaması çizilen veriyle uyuşmuyor | 5.000'lik ham veri örnekleri ve 1.000 karakterdeki son kutu doğru tanımlandı |
| Orta | İki ana tablo ve üç ek tabloya açık gönderme yok | Göndermeler eklendi; ek şekiller zaten elle yazılmış numaralarla anılıyordu |
| Orta | API ücretinin sıfır olması, toplam maliyet veya eşit kalite kanıtı gibi okunabilir | API ücreti, yerel kaynak tüketimi ve bağımsız literatür örneklemesi ayrıldı |
| Orta | Yeni LaTeX bağımlılıkları `.gitignore` yüzünden sürüme eklenemiyor | Gerekli kaynaklar ve yeni şekil için istisnalar eklendi; commit/push yapılmadı |

## Özet değerlendirmesi

Özette GPU modeli, bellek, NF4 ve ayrıntılı runtime ölçümleri bilimsel mesajı
gölgeliyordu. Bunların yeri yöntem ve yeniden üretilebilirlik bölümüdür.
Özet şu sırayla yeniden yazıldı: problem → karşılaştırma → eğitim ve
doğrulama bütçesi → başlıca F1 bulguları → hız/baseline sonucu → sınırlılık.
Yeni özet boşluk temelli sayımla **179 kelime**; GPU/RTX/CUDA/NF4/BF16/VRAM
terimleri yok. Hafif adaptasyon ifadesi korunurken uygulama ayrıntıları çıkarıldı.

Ana mesaj artık üç modelin kendi kontrolüne göre iyileşmesi ve etiket
mevcudiyetinin tercihleri değiştirmesi. Qwen3'ün 0.7891 sonucu,
2.83–9.62 yüzde puan artış ve SVM/DistilBERT karşılaştırması korunuyor.
“En iyi model”, “üstün yöntem” veya “ücretsiz çözüm” gibi kapsamı aşan bir
özet iddiası kullanılmamalı. Sayısal yoğunluk hâlâ var, fakat bunlar donanım
ayrıntıları yerine araştırma sonuçlarıdır.

## Veri bütünlüğü: doğrudan doğrulananlar

| İşlem / kontrol | Sonuç |
|---|---:|
| Ham CSV | 162.421 kayıt |
| Eksik product/narrative çıkarıldıktan sonra | 162.411 |
| En fazla 500 karakter, tekrar çıkarmadan önce | 103.915 |
| Exact narrative tekrarları çıkarıldıktan sonra | 72.611 |
| Bu havuzda normalleştirme sonrası benzersiz metin | 72.611 |
| Altı 500-karakter test dosyasının kaynak metin + etiket eşleşmesi | Her biri 1.000/1.000 |
| Özgün oluşturma betiğinin bellekte yeniden yürütülmesi | Altı dosya da metin, etiket ve sıra bakımından aynı |
| 20.000'lik eğitim / 2.000'lik validation arasında exact normalleştirilmiş örtüşme | 0 |
| Eğitim veya validation ile testler arasında exact/normalleştirilmiş örtüşme | 0 |
| Eğitim veya validation ile testler arasında 250-karakter önek örtüşmesi | 0 |
| 500/2.000/10.000 eğitim altkümelerinin 20.000'lik eğitim kümesinde bulunması | Doğrulandı |
| Eski 132 tahmin dosyasının önceki denetim hash'lerine göre değişmesi | 0 dosya |
| Yeni HF değerlendirme | 72.000 test + 600 seri tahmin doğrulamayı geçti |
| SVM ve üç DistilBERT fit'inin tahminleri | 84 dosya / 84.000 satır; indeks/etiket ve yeniden hesaplanan metrikler uyumlu |

Dosyaları yeniden oluşturarak üzerine yazmadım. Özgün örnekleme sürecini
bellekte tekrar uygulayıp mevcut dosyalarla karşılaştırdım. Kaynakların
kontrol edilebilir olması olumlu bir bulgu; bu kontrollerde veri sızıntısı
gösteren exact veya aynı-önek eşleşme bulunmadı.

Deney B'nin kredi raporlama payı **%51,1** iken 103.915 kayıtlık havuzda
**%63,58** bulunması tek başına tutarsızlık değil. Arada 31.304 exact tekrar
çıkarılıyor. Son havuzun dağılımı kredi raporlama %51,15, borç tahsilatı
%18,17, mortgage/kredi %10,70, retail banking %10,02 ve kredi kartı %9,97.
Bunlar 1.000 kayda yuvarlanınca **511/182/107/100/100** sayımlarını veriyor.
Dolayısıyla Deney B, tekrarları çıkarılmış havuzun dağılımını temsil ediyor.
Ham CFPB evreninin ya da bütün uzunluktaki şikâyetlerin dağılımını temsil
ettiği söylenemez. Şekil 1'in ilk çubukları tekrar çıkarma öncesi aşamaya ait;
açıklama bu aşama farkını belirtmek üzere düzeltildi.

**Doğrulamanın sınırı:** Eski Ollama CSV'lerinde özgün narrative hash'i yok;
indeks ve etiket uyumu, tarihsel çalıştırmada aynı metnin kullanıldığını
kesin kanıtlamaz. Yeni HF kayıtlarında bu izlenebilirlik mevcut.
Supervised tahmin CSV'leri de satır bazlı
metin hash'i saklamıyor; bunlarda eğitim raporlarının kaynak/test/bölme
hash'leri ve güncel indeks/etiket uyumu doğrulandı. MinHash
ve trigram Jaccard taraması yaklaşık bir benzerlik araması; bütün
parafrazların dışlandığı söylenemez. Eğitim hazırlığında 133 çelişkili
etiket grubunun ve tespit edilen yakın kopyaların dışlanması faydalı,
fakat veri etiketlerinin tamamının insan denetiminden geçtiği anlamına gelmez.
Ön eğitimde CFPB verisinin görülüp görülmediği de bu dosyalardan belirlenemez.

CSV'de şikâyet tarihi veya özgün CFPB complaint ID bulunmuyor.
Mart 2020–Mart 2021 kapsamı kaynak çalışmanın açıklamasına dayanıyor;
yerel dosyadan bağımsız doğrulanmış bir zaman aralığı olarak sunulmamalı.
Makaledeki ifade buna göre düzeltildi.

## Örneklem bağımsızlığı ve istatistik

500-karakter dengeli testlerin birleşimi 3.000 benzersiz metin; dengesiz
testlerin birleşimi de 3.000 benzersiz metin içeriyor. İki koşulun
**2.067 metni ortak**, birleşimleri **3.933 benzersiz metin**. Dolayısıyla
6.000 test satırı, 6.000 farklı şikâyet anlamına gelmiyor. 250-karakter
görünümler de aynı kayıtların kısaltılmış halleri. “12 koşulda pozitif”
bulgusu değerli, ancak 12 bağımsız tekrar olarak yorumlanamaz.

Bootstrap metin kümelerini koruyor; bu, yinelenen girişlerden kaynaklanan
belirsizlik sorununu ele alıyor. Yine de güven aralıkları eldeki test
metinleri ve sabit model fit'leri için koşullu aralıklar. Eğitim seed'i,
prompt veya gelecek dönem belirsizliğini ölçmüyor. Ana tablolardaki eski
sonuçlar üç altkümenin ortalama/std'si, yeni sonuçlar ise pooled confusion
matrix üzerinden hesaplanan F1; bunların küçük farklarla ayrılması normal.
Bu iki kestiriciyi farkında olmadan aynı hücreye aktarmak hata.

Eski paragraftaki “DeepSeek-R1 14B için her seed'de p<0.001” iddiası yanlıştı:
seed 1'de eski tek yönlü exact p **0,001479**. Ayrıca dört karşılaştırmayı
her altkümede ayrı düzeltip üç altkümenin tamamı için iddia kurmak zayıf bir
çoklu test tanımı. Yeni metinde iki yönlü exact McNemar ve toplam 12 test
için Bonferroni kullanıldı. Gemma–DeepSeek 14B en büyük ham p **0,002959**,
Gemma–Qwen3 8B en büyük p **0,000811**; iki karşılaştırma bütün altkümelerde
düzeltmeyi geçiyor. **Qwen3 14B ve gpt-oss, hiçbir altkümede bu eşiği geçmiyor.**
Gemma referansı sonuçları gördükten sonra seçildiği için bu analiz yine
keşifsel. McNemar doğruluğu test eder; doğrudan weighted-F1 üstünlüğü testi değil.

Yeni HF base/adapter testleri ayrı: 12 karşılaştırmalık Holm düzeltmesinden
sonra bütün accuracy testleri anlamlı, bütün eşleştirilmiş F1 artış
aralıkları pozitif. Bu sonuçları eski Ollama testlerinin p-değerleriyle
karıştırmamak gerekiyor. Adapterli modellerin birbirlerine karşı
üstünlüğü ayrıca test edilmedi.

## Tablolar ve şekiller

Yeniden hesaplama, küçük görünse de güvenilirliği etkileyen farklar buldu:

| Hücre | Önce | Doğru raporlama |
|---|---:|---:|
| Gemma 4, A accuracy | 0.774 | 0.773; gerçek ortalama 0.773333 |
| Gemma 4, A latency | 17.11 s | 17.12 s; gerçek ortalama 17.11528 |
| gpt-oss, B weighted F1 | 0.783 | 0.782; altküme ortalaması 0.78248 |
| Gemma 2, A F1 std | 0.009 | 0.008; gerçek örneklem std 0.008459 |
| Phi-4, B accuracy std | 0.004 | 0.003; gerçek örneklem std 0.003464 |
| Gemma 4, B retail F1 | 0.7676 | 0.7687 |
| Qwen3 14B, B debt F1 | 0.6727 | 0.6719 |

`export_legacy_manuscript.py` bütün eski ana sonuç ve sınıf-F1 satırlarını
değişmemiş tahminlerden üretir; `--check` sayıları yazmadan karşılaştırır.
API karşılaştırma tablosundaki Gemma accuracy hücresi de düzeltildi.
“Ortalama altküme F1” ile “pooled F1” ayrımı korunmalı.

`analyze_seeds.py` karmaşıklık matrisi fonksiyonunda `off_label` önce
`retail_banking` ile değiştiriliyordu. Bu, geçersiz yanıtların yanlış bir
sınıfa yazılması demek. **F1 hesaplama fonksiyonunda bu dönüşüm yoktu**;
temel F1 sonuçlarını değiştiren bir hata bulunmadı. Çizim kodu ayrı bir
geçersiz çıktı sütunu kullanacak biçimde düzeltildi. S3/S4/S5 yeniden
üretildi; her altkümenin matris toplamı 1.000, her sınıf desteği 200 ve
geçersiz sütunu gerçek hata sayısıyla eşleşiyor. Hücrelerin normalize
oranlar olduğu da açıklamaya eklendi. Matris başlıklarındaki `N=500`
yanıltıcılığı `500-char, N=1,000` olarak düzeltildi. Küçük hata oranları
sıfıra yuvarlanmasın diye üç ondalık gösterildi; Phi/DeepSeek açıklamaları
gözlenen baskın hata yönlerine göre düzeltildi.

S1 için “üç dengeli 1.000'lik altküme” ve “çoğu anlatı 300 karakterden
kısa” açıklamaları yanlıştı. Şekil betiği filtrelenmemiş veriden üç
5.000'lik rastgele örnek çiziyor ve 1.000'in üzerini son kutuda topluyor.
Ham verinin medyanı 361 karakter; en fazla 300 karakter oranı **%42,69**.
Bu şekil 250 karakter tercihini kendi başına gerekçelendirmiyor.
500-karakter filtresi ve 250-karakter duyarlılığı, maliyet/kapsam tercihi
olarak açıklanmalı. Açıklama düzeltildi.

S8'de açıklama/metin accuracy derken çizilen y ekseni weighted F1 idi;
grafik metindeki accuracy duyarlılığına uygun olarak yeniden üretildi.
Legend'deki `N=250` etiketi de `250-char` olarak düzeltildi: her test
altkümesinin örneklem büyüklüğü 1.000, değişen şey karakter sınırı.

Şekil 2 accuracy çizerken metindeki bir cümle F1 dağılımından söz
ediyordu; accuracy olarak düzeltildi. “Phi en iyi accuracy/latency oranına
sahip” iddiasının tanımlanmış bir oran hesabı yoktu; gözlenen koşullarda
bir saniyenin altındaki en yüksek F1 olarak daraltıldı. Qwen3~8B için
“iki koşulda da sıfır geçersiz çıktı” cümlesi de düzeltilerek dengesiz
testteki tek hata açıklandı. Model grubu için open-weight terimi
kullanıldı; ağırlık erişimi, bütün modellerin aynı açık kaynak lisansına
sahip olduğu anlamına gelmez. S6'da Gemma'nın belirsiz modu ayrı gösterildi.

## Atıf ve gönderme denetimi

Ana belgede **3 şekil + 9 tablo**, ek belgede **8 şekil + 3 tablo** var.
İnceleme öncesinde ana belgenin bootstrap ve supervised tablolarına açık
gönderme yoktu. Ek belgenin üç QLoRA tablosu da numarasıyla anılmıyordu.
Bunlar düzeltildi. Ek şekiller ana metinde S1–S8 olarak zaten anılıyordu;
yalnızca `\ref` arayan kontrolün bunları “atıfsız” sayması yanlış olurdu.
Sabit yazılmış numaralar belge etiketlerinden çözümlenen göndermelere
çevrildi; ek tablolar S1/S2/S3 olarak numaralandırıldı.

**Metinden gönderme ile bibliyografik atıf farklıdır.** Kendi deneyimizden
üretilen tabloya başka bir yayın atfı eklemek gerekmez. Dış kaynaktan
alınmış API sayıları ve referans çizgisi kaynak ister. API tablosunda ve
ek karşılaştırma şeklinin açıklamasında kaynak zaten vardı; ana karşılaştırma
şeklinin o1 referans çizgisine de açık kaynak atfı eklendi.

Son yapısal kontrolde sahipsiz tablo/şekil, tanımsız referans, tanımsız
bibliyografik anahtar ve yinelenen etiket kalmadı. Bu, bir kaynağın
atıf yapılan bütün cümleleri anlamsal olarak desteklediğinin otomatik kanıtı değil.

## Kaynakça ve literatür iddiaları

DOI kayıtları, arXiv birincil sayfaları ve yakın referans makalenin yayıncı
PDF'i incelendi. Kesin yanlışlık ile erişilemediği için doğrulanamayan
iddia aynı şey değil. Crossref 429 yanıtı veya Crossref'te bulunmayan
DataCite/Zenodo kaydı, kaynağın sahte olduğunu göstermez.

Önemli düzeltmeler:

- Bulgakov/Turchin kaydı yanlış başlık ve yazar adları içeriyordu.
  Gerçek çalışma *Reasoning-Enhanced Rare-Event Prediction with Balanced
  Outcome Correction*; yazarlar Vitaly Bulgakov ve Alexander Turchin.
  Eski DeepSeek/binary outcome anlatımı yerine doğrulanmış kapsam kullanıldı.
  [Birincil kayıt](https://arxiv.org/abs/2601.16406).
- Shin kaydının başlığı ve ilk yazarı yanlıştı; gerçek kayıt Minkyu Shin,
  Jin Kim ve Jiwoong Shin'in finansal şikâyetlerde LLM benimsemesi çalışması.
  [Birincil kayıt](https://arxiv.org/abs/2311.16466).
- Liang kaydı yanlış ilk yazar ve başlık içeriyordu; gerçek çalışma
  Weixin Liang ve arkadaşlarının toplum genelinde LLM destekli yazı çalışması.
  [Birincil kayıt](https://arxiv.org/abs/2502.09747).
- Güllü'nün ilk adı/yazar listesi ve başlığı, Correa'nın yazarları/başlığı,
  Das'ın ilk yazarı/başlığı ve Vairetti'nin başlığı düzeltildi.
  [Güllü](https://doi.org/10.1109/ACDSA65407.2025.11166338),
  [Correa](https://doi.org/10.1109/ANDESCON61840.2024.10755794),
  [Das](https://doi.org/10.1109/TCSS.2023.3338357),
  [Vairetti](https://doi.org/10.1016/j.ejor.2023.08.027).
- Raju kaydındaki başlık, yazar ve konferans yanlışlığı düzeltildi.
  Gerçek CFPB topic-modelling çalışması Sangaraju ve arkadaşlarına ait.
  [Birincil kayıt](https://arxiv.org/abs/2205.07259).
- Jondhale başlığına AIOPS eklendi ve Shlok Patil adı düzeltildi;
  Li makalesinde Yin Xu adı ve 2025 basılı yayın yılı düzeltildi.
  [Jondhale](https://doi.org/10.1109/ICAIT61638.2024.10690778),
  [Li](https://doi.org/10.1007/s13042-024-02432-9).
- BERTopic yöntem atfı özgün 2022 makalesine yöneltildi;
  Llama 3.1 ailesi için Llama 2 yerine uygun Llama 3 makalesi eklendi.
  [BERTopic](https://arxiv.org/abs/2203.05794),
  [Llama 3](https://arxiv.org/abs/2407.21783).

Tam metin/protokol eşleşmesini doğrulayamadığım **89,99%, 92,1%, 88%,
67%/60,6%, 9,8%, 18% ve 684.653 kayıt** gibi literatür iddialarını
ana/ek anlatımdan çıkardım. Bunların yanlış olduğu sonucuna varmadım;
güvenilir bir karşılaştırma kurmak için dayanakları bu incelemede yeterince
doğrulanmadı. Kısa ve doğrulanmış kapsam açıklamaları bırakıldı.
Kaynakçada 36 giriş bulunuyor; bu inceleme bütün girişlerin bütün tam
metinlerini eksiksiz doğruladığı iddiasında değil. Gönderimden önce kalan
literatür cümleleri için tam metin–cümle eşleştirmesi yapılmalı.

## Gecikme, maliyet ve karşılaştırma adaleti

Eski Ollama gecikmesi sunucu isteği süresi; yeni HF seri gecikmesi warmup
sonrası prompt/tokenisation ve senkronize üretim süresi. Yeni ölçümde
decode, ağ ve ilk yükleme hariç. Batch süresinin batch boyutuna bölünmesi
tek istek gecikmesi değildir. Bunlar aynı tabloda açıklamasız birbirinin
yerine konamaz. Eski donanım/serving loglarının eksikliği ayrıca çözülmedi.

Seri/batch çıktılarının üçünde etiket farkı var: adapterli Gemma'da bir,
adaptersiz Qwen'de iki. Giriş hash'leri ve token sayıları aynı; nedeni
izole edilmedi. F1 bütün batch çıktılarından, süre ayrı 100'lik seri
örneklemden hesaplanıyor. Bu ayrım açık bırakıldı. Mean latency <1 s,
p95/p99 veya kuyruk altında <1 s garantisi değildir.

13 GB gpt-oss artifact boyutu, toplam VRAM tüketimi diye yorumlanamaz.
“110 tahmin/dakika” seri ortalama sürenin tersidir; ölçülmüş concurrent
throughput değildir. “0 dolar”, API ücretinin olmamasıdır; GPU, enerji,
personel ve fırsat maliyetini içeren toplam sahip olma maliyeti değildir.
Bir milyon kayıt için 198 seri GPU-günü hesabı da hızlanma ölçmeden ekonomik
üstünlük sonucuna çevrilmemeli.

Ücretli API kıyası aynı protokolü hedefliyor, fakat aynı test kimlikleri
bilinmiyor. Ücretler referans makalenin tarihsel ücretleri. Benzer F1
noktaları eşdeğerlik kanıtı değildir; dış örneklem üzerinde alınmış
ticari skorlarla yerel paired test yapılamaz.

## Yeni deney yapılacaksa en değerli ekler

1. **500 eğitim + aynı 2.000 validation ile üç-seed DistilBERT kontrolü.**
   Mevcut 20.000'lik encoder güçlü bir baseline, ancak aynı eğitim bütçesinde
   LLM avantajını sınamıyor. “Neden 14B LLM?” sorusuna en doğrudan kontrol bu.
2. **Üç LLM adapterini iki ek eğitim seed'iyle tekrarlamak.** Böylece her
   model için üç fit olur; örneklem bootstrap'ından ayrı eğitim kararlılığı
   ölçülür. Üç test altkümesi zaten üç eğitim tekrarının yerine geçmiyor.
3. **Zaman ve kapsam genellemesi.** Tam uzunlukta ve daha yeni kayıtlarda
   kontrollü ek test, tek veri kümesindeki sıralamadan daha güçlü kanıt sağlar.
   Özgün tarih/kimlik içeren veri gerekir; mevcut CSV'den üretilemez.
4. **Tarihsel Ollama'yı savunmak gerekiyorsa kontrollü yeniden koşu.** Model
   digest'i, chat template, quantisation, düşünme modu, çıktı limiti ve
   donanım kaydedilmeden mevcut sınırlılık kesin olarak giderilemez.

Bu incelemede yeni eğitim başlatılmadı. Yeni deney olmadan da çalışmanın
iddiaları mevcut pilot kapsamına daraltılabilir; ek deneylerin sağlayacağı
güç ile çalışma süresi ayrı bir karar olarak değerlendirilmelidir.

## Gönderim ve son doğrulama

Ana PDF **24**, ek PDF **14** sayfa. İkisinde de tanımsız atıf/referans veya
overfull kutu bulunmadı; ana şablonun font substitution ve bookmark
uyarıları var. Bunlar sonuç bütünlüğü hatası değil, son biçim kontrolü konusu.
Son kaynaklar ve değişen şekiller derlenerek kontrol edildi.

Güncel resmî yönerge araştırma makalesini 20 sayfa içinde istiyor;
mevcut ana PDF'nin son biçim aşamasında kısaltılması gerekiyor.
[Dergi yönergesi](https://link.springer.com/journal/12046/submission-guidelines).
Şablon, çift satır aralığı ve şekil baskı çözünürlüğü kontrolleri son aşamaya
bırakıldı. Özellikle şekilleri yalnızca ek dosyaya taşımak, ana metindeki
tekrarları azaltmanın yerini tutmaz.

Tekrarlanabilirlik için kod ve makale bağımlılıklarının gerçekten kamuya
açık sürümde bulunması gerekir. `.gitignore` daha önce yeni LaTeX input
dosyalarını ve yeni QLoRA şeklinin sürüme eklenmesini engelliyordu; dar
istisnalar eklendi. Eğitim veri manifestleri/adapter erişim planı ve son
kamuya açık sürüm yine gönderim öncesi tamamlanmalı. Commit, push veya
dergi gönderimi yapılmadı.

Denetimin ölçülebilir kayıtları `results/sadhana/reviewer_audit/` altında:
`data_integrity.json`, `builder_replay.json`, `test_population_overlap.json`,
`recomputed_legacy_metrics.csv`, `legacy_mcnemar.csv`,
`figure_and_hash_checks.json`, `bibliography_checks.json`,
`supervised_integrity.json`,
`verified_dois.json`, `additional_verified_dois.json`,
`object_references_before.json`, `manuscript_structure_after.json` ve
`build_verification.json`.
`bibliography_checks.json` inceleme öncesi künyeleri de saklar; son künyelerin
doğru olmadığını söyleyen bir güncel sonuç dosyası olarak okunmamalı.

Tekrar kontrol komutları:

```bash
conda activate ConsumerComplaint
python verify_qlora_evaluation.py
python export_legacy_manuscript.py --check
python audit_manuscript.py
cd SPRINGER
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary_material.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```


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
