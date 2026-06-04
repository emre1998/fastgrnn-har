# Günlük Rapor — 23 Mayıs 2026

## Bugün ne yaptık?

FastGRNN projesinin temellerini attık. Boş bir bilgisayardan başlayıp, gün sonunda
çalışan bir veri hattımız ve karşılaştırma için bir "yenmen gereken sayımız" oldu.

Aşağıda bütün adımları sıradan bir dille anlatıyorum.

---

## 1. Çalışma ortamını kurduk

Bilgisayara projeye özel bir "kutu" yaptık (`venv` denen sanal Python ortamı).
İçine ihtiyaç duyacağımız tüm araçları kurduk: PyTorch (sinir ağları için),
NumPy ve Pandas (matematik ve veri için), Matplotlib (grafik için), scikit-learn
(makine öğrenmesi ölçümleri için).

Projeye git başlattık ki her değişiklik kayıt altında olsun.

**Sonuç:** `C:\Users\EMRE CAN\Desktop\fastgrnn-har\` klasöründe çalışan,
temiz, takip edilen bir proje.

## 2. Projenin "anayasasını" yazdık

`contract.md` adında bir dosyaya projenin kurallarını yazdık:
- Hangi sınıfları tanıyacağız (yürüyor, duruyor, koşuyor, vb.)
- Ham veriyi nasıl pencerelere böleceğiz (128 örnek, %50 örtüşme)
- Hangi metriklerle ölçeceğiz (doğruluk, macro-F1)
- Hangi karta yükleyeceğiz (Arduino Uno → MSP430)
- "Test verisini ayar yapmak için kullanmayacağız" gibi bilimsel disiplin kuralları

Bu dosya değişebilir ama her değişiklik bilinçli yapılır.

## 3. FastGRNN'i sıfırdan, saf NumPy ile yazdık

Önce paper'ı okuyup matematiğini anladık (RNN, GRU, FastGRNN'in farkı).

Sonra denklemleri tek tek koda çevirdik:
- **sigmoid**: bir sayıyı 0 ile 1 arasına sıkıştıran S-eğrili fonksiyon. Kapılar için.
- **tanh**: sayıyı -1 ile +1 arasına sıkıştırır. Hafıza için.
- **fastgrnn_step**: bir zaman adımı için "eski hafızayı ne kadar tut, yeni
  bilgiyi ne kadar al" kararı.
- **run_sequence**: 128 örnekten oluşan tam pencereyi tek seferde çevirir.

Bunu yaparken FastGRNN'in zekice tasarımını görmüş olduk: aynı ağırlık matrisi
hem kapı hem hafıza için kullanılıyor (bu yüzden çok küçük). `ζ` ve `ν` adında
sadece iki sayı, RNN'lerin "patlama/sönme" sorununu çözüyor.

### Bonus: bir reproduction bulgusu yakaladık

Modeli henüz eğitmeden, `ζ = ν = 0.5` seçip 128 örneklik bir gerçek ivme
sinyali çalıştırdık. Hafızanın bir hücresi 27'ye fırladı — aşırı büyüdü.

Bu **hata değil, doğal bir gözlem**: paper'ın `ζ` ve `ν`'yi neden öğrenmek
zorunda olduğunu, neden belirli bir aralıkta tutulması gerektiğini
**deneysel olarak görmüş olduk**. Bu bulgu reproduction raporumuzun ilk
satırı olacak. Paper okumadan asla bulamayacağımız bir şeyi, kodu sıfırdan
yazdığımız için bedavaya öğrendik.

## 4. Gerçek veri setini (UCI HAPT) indirdik ve tanıdık

Microsoft EdgeML ekibinin de kullandığı, 30 kişinin akıllı telefonla yaptığı
aktivitelerin kayıtlı olduğu hazır bir veri seti indirdik.

Veriyi inceledik:
- 30 gönüllü, her biri farklı aktiviteler yapmış (yürüme, oturma, merdiven
  çıkma, vb.)
- 50 Hz örnekleme (saniyede 50 ölçüm)
- Her aktivitenin başlangıç-bitiş indeksi etiketlerle işaretli

Bir kullanıcının verisini grafikleştirdik. İki şey gördük:
- **Durağan aktiviteler** (oturma, ayakta durma) düz çizgiler — sinyal sakin.
- **Hareketli aktiviteler** (yürüme, merdiven) belirgin periyodik salınım —
  adım ritmi gözle görülüyor.

Bu, modelin neyi öğrenmeye çalışacağı hakkında sezgi verdi.

## 5. Veriyi modelin yiyebileceği hale getirdik

Tüm 61 oturumu işledik:
- Her etiketli aralığı 128 örneklik (~2.5 saniye) pencerelere böldük.
- Her pencerenin etiketi, kullanıcı kimliği ile birlikte kaydedildi.

**İlk denemede 12 sınıf vardı.** Ama 6 tanesi "geçiş" aktiviteleriydi
(oturur-pozisyondan-kalk gibi) ve çok kısa süreliydi — en küçüğü sadece 33
örnek. Bu **aşırı sınıf dengesizliği** demek. Model bunları öğrenemez,
ortalama metrikler yanlış görünür.

Bu yüzden geçişleri attık, 6 temel aktivite ile çalışmaya karar verdik.
Bu aynı zamanda FastGRNN paper'ının da kullandığı setupla uyumlu.

**Sonuç:** 10,411 pencere, 6 sınıf, dengeli dağılım. 21 kullanıcı eğitim,
9 kullanıcı test (sızıntı yok — bir kişi hem eğitimde hem testte değil).

## 6. MLP baseline modelini eğittik

FastGRNN'i karşılaştıracağımız "minimum çıta" için en basit modeli yazdık:
**MLP (Çok Katmanlı Algılayıcı)** — 384 girdi → 32 nöron → 6 çıktı. 12,518
parametre.

Bu modelin **bir zayıflığı var**: zamanı görmüyor. 128 örneklik pencereyi
düz bir vektör gibi alıyor — örneklerin sırasını umursamıyor. Yürürken
adımların ritmik tekrarı onun için sadece "384 sayı".

Bu zayıflık önemli, çünkü FastGRNN'in **en büyük iddiası** zamanı görmek.
Aradaki farkı bu test ölçecek.

30 epoch eğittik, sonuç:

| Metrik | Değer |
|---|---|
| Test accuracy | **%85.47** |
| Test macro-F1 | **%84.73** |

Bunlar "yenmemiz gereken sayılar." FastGRNN bunlardan yüksek çıkmazsa,
proje hipotezimiz çökmüş demektir.

## 7. Confusion matrix'i okuduk — projenin yol haritası buradan çıktı

Hangi sınıflar karışıyor diye baktık:

| Sınıf | F1 | Yorum |
|---|---|---|
| LAYING (uzanma) | %100 | Trivial — telefon yatay, yerçekimi farklı eksende. |
| WALKING | %84 | Solid. |
| DOWNSTAIRS | %88 | İyi. |
| SITTING | %82 | STANDING ile karışıyor. |
| STANDING | %81 | SITTING ile karışıyor. |
| **UPSTAIRS** | **%73** | **En zayıf halka.** |

UPSTAIRS pencerelerinin **%23'ü WALKING** olarak yanlış tahmin edilmiş.
Sebep: ikisi de yürüme deseni, ama merdiven çıkarken adım daha dik ve uzun
süreli, düz yürürken simetrik. **Bu fark zamanın içinde**, ama MLP zamanı
görmediği için ayırt edemiyor.

**İşte FastGRNN'in parlaması beklenen yer:** o ardışıklığı yakalayabildiği
için UPSTAIRS-WALKING ayrımını yapabilmeli.

LAYING'de zaten %100, daha iyileşemez. SITTING-STANDING ayrımı fiziksel
olarak da zor (ikisi de durağan). Asıl mücadelenin alanı: 3 yürüme sınıfının
birbirinden ayrılması.

---

## Bir cümlede bugün

> Sıfırdan bir AI projesi başlattık. Matematiği koda çevirdik, gerçek veriyle
> tanıştık, ilk modeli eğittik (%85 doğruluk), ve sıradaki adımda asıl
> modelimizin (FastGRNN) **hangi sınıflarda kazanım sağlaması gerektiğini**
> şimdiden saptadık.

## Yarın / sonraki oturumda

FastGRNN'i PyTorch'ta `nn.Module` olarak yeniden yazıp aynı veriyle eğiteceğiz.
Hedef: %85.47'yi aşmak, özellikle UPSTAIRS sınıfında ciddi iyileşme görmek.

## Proje klasöründeki dosyalar

```
fastgrnn-har/
├── contract.md              # Proje kuralları
├── README.md                # Kısa tanıtım
├── fastgrnn_numpy.py        # FastGRNN saf NumPy implementasyonu
├── scratch_sigmoid.py       # Sigmoid testi
├── scratch_step.py          # fastgrnn_step testi (kararlılık dahil)
├── scratch_sequence.py      # run_sequence testi
├── download_hapt.py         # HAPT indirici
├── explore_hapt.py          # Veri görselleştirme
├── build_dataset.py         # Pencere oluşturucu
├── train_mlp_baseline.py    # MLP eğitim ve değerlendirme
├── mlp_baseline_best.pt     # Eğitilmiş MLP ağırlıkları
├── experiments/
│   └── mlp_baseline.json    # Baseline sonuçları
├── data/
│   ├── hapt/                # Ham UCI HAPT
│   └── processed/
│       └── hapt_windows.npz # İşlenmiş veri (X, y, subjects)
├── explore_session.png      # Bir oturumun görseli
├── explore_walking.png      # WALKING + pencere görseli
└── notes/
    ├── memory_oturum3.md    # Oturum kaydı
    └── gunluk_rapor_2026-05-23.md   # Bu dosya
```
