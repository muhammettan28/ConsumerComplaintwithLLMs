tek subset yerine 3 ayrı subset kullan, alınan sonuçlar güvenilir ve inandırıcı olsun.


max length şuanki versionda 500 ama justification zayıf bence. bence seçtiğimiz 3 subset için metion uzunluklarına bakalım, makalede histogram ile gösterelim, gerekirse birden fazla length ile test edelim.(örnek 250 - 500)

şimdi birçok şeyi değiştireceğimiz için önceki sonuçlara dair görsel, excel verisigibi herşeyi belki lazım olur diye yedeklememiz gerekiyor.

Bundan sonra veriseti içinde şuanki versiyondaki zaman diliminde 3 farklı subset oluşturup(mükerrer kayıt olmadan)
balanced ve unbalanced şekilde verisetleri oluşturalım.şuan dataset klasöründeki balanced_1000.csv ve imbalanced_1000 i de diğer sonuçlar gibi yedeklediğimiz klasöre koyalım.
yeni verisetleri aşağıdaki isimlerle oluşsun. asıl verisetimiz zaten biliyorsun complaint.csv

örnek 1_subset_balanced_100.csv , 2_subset_balanced_100.csv, 3_subset_balanced_100.csv
örnek 1_subset_imbalanced_100.csv , 2_subset_imbalanced_100.csv, 3_subset_imbalanced_100.csv

tüm klasör yapımızda şuanki versiyona dair sonuç, görüntü dosyası gibi herşey yedek dosyasında olacak, arşivleyeceğiz yani.