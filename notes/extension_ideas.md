# Extension Ideas — Paper'in Otesine Gececek Fikirler

> Bu dosya: standart L-S-Q reproduction tamamlandiktan sonra (Hafta 8 sonrasi)
> denenebilecek arastirma katkilari. Hicbiri reproduction'a girmez —
> hepsi raporun "Beyond Reproduction" / "Our Extensions" bolumune adaydir.

---

## Fikir 1 — Statik/Dinamik icin Farkli Efektif Rank (2026-05-25)

**Motivasyon:** r_u sweep'inde gozlemledigimiz tradeoff:
- r_u=4 dinamik siniflar (WALKING, UPSTAIRS, DOWNSTAIRS) icin tatli nokta
  (regulariser gibi davraniyor, ortalama F1'leri ~0.92)
- r_u=4 statik siniflarda (SITTING, STANDING) bilgi kaybi yaratiyor
  (~0.80 F1'lere kadar dusus)
- r_u=6 tam tersi — statik kurtuluyor, dinamik kaybediyor
- Tek bir r_u ikisini birden alamiyor

**Hipotez:** Sinyalin karakterine gore farkli rank alt-uzaylari kullanmak
ikisini birden kazandirabilir.

### Olasi yaklasimlar

**A) Paralel iki dal:**
  U_efektif = U_dynamic (rank r_d=4) + U_static (rank r_s=8)
  - Matematiksel rank toplami; parametre cifte ciktigi icin saf
    verimlilik acisindan kotu, ama farkli induktif onyargi verir.

**B) Kapi tabanli dal secimi (en umut verici):**
  g = sigmoid(w_g . h_prev + b_g)
  h_combined = g . h_static + (1-g) . h_dynamic
  - Model sinyalden ogrenip kapiyi ayarlar.
  - Maliyet: ekstra parametre + compute. Sikistirma vaadini zayiflatabilir.

**C) Sinyal enerjisiyle kosullu yol:**
  Yumusak gating, sinyal varyansi/RMS kullan.
  - EE-tarzi, sezgisel.
  - if/else'i yumusak yapmak Secenek B'ye benzer.

**D) Hiyerarsik low-rank + diagonal artık (en ucuzu):**
  U_efektif = LowRank(r=4) + diag(alpha)
  - Sadece H=16 ekstra parametre.
  - Statik DC bilesenleri diagonal yoldan, dinamik desen lowrank yoldan.
  - Hafta 8 sonrasi hizli denenebilir.

### Neden simdi DEGIL

1. L-S-Q pipeline'i (Hafta 5-6-7) sparsity ve quantization ekleyecek.
   Dual-path'i simdi kurarsan, sonraki katmanlarin etkisi izlenemez hale gelir.
2. Reproduction'in temel iddiasi "FastGRNN'i sifirdan dogru implement ettim".
   Bunu varyantlarla bulandirma.
3. Bu fikir sonra "Beyond reproduction" eki olarak rapora girerse cok daha
   degerli — CV'de "ben sadece reproduce etmedim, paper'in yapmadigi bir
   varyant da denedim ve su sonucu aldim" cumlesi guclu.

### Hafta 8 sonrasi action plan
1. Standard L-S-Q reproduction tamamla.
2. Q15 model Arduino'ya indirildi, rakamlar raporlandi.
3. Sonra: Secenek D (en ucuz) ile baslayip ablasyon yap.
4. Per-class kazanc gozlemlenirse Secenek B'yi de dene.
5. Raporun "Extensions" bolumune ek olarak yaz.

---

---

## Fikir 2 — DOWNSTAIRS Sinifi Kazanim Stratejileri (2026-05-25)

**Motivasyon:** L-S-Q sonrasi DOWNSTAIRS hâlâ FastGRNN'in en zayıf dinamik
sinif kategorilerinden biri. MLP baseline (0.877) bu sınıfta hâlâ rekabet
edebilir durumda — paper'in açıklamadığı bir gerçek. Gemini analizinden
geçirip degerlendirdigimiz stratejiler:

### A) Butterworth low-pass filtre (en dusuk maliyet)
**Ne:** 0.3 Hz cutoff ile yer çekimi DC bileşenini ayrı al, sadece vücut
ivmesini modele besle. UCI HAR standart pipeline'ında zaten var; bizde
sadece z-score normalize ettik, frekansa dokunmadık.

**Maliyet:** scipy.signal ile ~10 satir Python. Model değismez.
**Beklenen kazanim:** DOWNSTAIRS dikey impuls daha belirgin → +2-5 puan F1.
**Risk:** TÜM L-S-Q sonuçlarini invalide eder (pre-processing değişiyor).
Tüm pipeline yeniden koşulmali.

### B) FFT / spektral öznitelikler (yuksek maliyet)
**Ne:** Pencere üstünde FFT al, 16-32 spectral bin'i ham veriye ekle.
**Maliyet:** Input D = 3 → ~35. W matrisi 12× büyür (48 → 560 param).
Compression hikayesi (44× az param) bozulur.
**Beklenen kazanim:** MLP'nin DOWNSTAIRS'te güçlü olmasının sebebi
büyük olasılıkla düzlestirilmis ozelliklerin dolayli frekans bilgisi
taşıması — açıkça FFT eklemek bu farkı kapatabilir.
**Karar:** Compression-accuracy trade-off ablasyon olarak değerli.
Rapora "Pareto front, compression vs frequency-domain features" bölümü.

### C) EMI-RNN egitim rutini (yanlis fit)
**Ne:** Uzun pencere içinde kritik kısa imzayı bulup ona odaklan.
**Karar:** Bizim problem için yanlış tool — DOWNSTAIRS sürekli periyodik,
bir anlık imza değil. EMI-RNN düşme tespiti gibi anlık olaylar için.
ATLA.

### D) ReLU vs tanh swap (paper-uyumluluk riski)
**Ne:** Cell içindeki tanh'ı ReLU ile değiştir.
**Karar:** FastGRNN'in kararlılık ispatı tanh'in sınırlı çıkışına dayanır.
Kalibrasyon bulgumuza göre h_t zaten ~60'a kadar çıkıyor; tanh OLMAZSA
exploding garanti. Reproduction kapsamında uygulanamaz.
"FastGRNN-variant" olarak ayrı bir paragrafta tartışılabilir.

### E) MSC-RNN (Multi-scale cascaded RNN)
**Ne:** Farklı zaman ölçeklerinde paralel RNN dalları.
**Maliyet:** Yeni mimari, sıfırdan inşa. Hafta 8 deploy hedefini öldürür.
**Karar:** Uzun-vadeli extension olarak değerli ama bu projeden ayrı bir
çalışma. Future work bölümünde mention.

### Onerilen sira (Hafta 8 sonrasi)
1. **Butterworth filtre + L-S-Q yeniden koş** (~5 saat compute, ama temiz
   karşılaştırma)
2. **FFT öznitelik ablasyonu**: Pareto front (compression vs accuracy)
3. **MSC-RNN literatür notu**: future work paragrafı
4. ReLU varyantı: opsiyonel deneme

### Hafta 8 SONRASI action plan (guncellenmis)
1. Standard L-S-Q reproduction tamamla (Hafta 1-7) ✓
2. Arduino deploy (Hafta 8) — manşet çıktısı, demo
3. **"Beyond Reproduction" bölümü:**
   a. Fikir 1 (dual-rank: statik/dinamik için farklı r_u)
   b. Fikir 2.A (Butterworth filtre)
   c. Fikir 2.B (FFT öznitelikler) — compression Pareto
   d. Multi-seed teknik notlari
4. CV açıklaması: "FastGRNN paper'ını sıfırdan reproduce ettim, donanıma
   indirdim, paper'da yer almayan 8 deploy gotcha'sı keşfettim, üzerine
   4 farklı varyant ablasyonu yaptım."

---

---

## Fikir 3 — Pure Q15 Integer Inference (2026-05-27 ertelendi)

**Motivasyon:** Hafta 8 LUT optimizasyonu sonrasi MSP430 13.85 ms/sample, Arduino
9.7 ms/sample. Her ikisi de 50Hz icin real-time (<20 ms). Pure Q15 integer
inference ile teorik olarak Arduino 5-6 ms, MSP430 1.5-3 ms olur (~2-10× ek hizlanma).

### Niye ertelendi — tasarim dirençi
Generate_q15_assets.py ile combined scale'ler hesaplanmaya calisilinca:
- XW_REAL_SCALE = XN_SCALE × W2_S × W1_S = (1/4096) × 3e-5 × 4.24e-5 ≈ 3.1e-13
- Bu degeri Q15 multiplier'a sikistirmak = 0 (sifira yuvarlaniyor)
- CLS_BIAS_MULT int32 araliginda taşmıştı (18,371,659)

**Kök sebep:** Per-tensor weight scale'leri tiny (~10⁻⁵). Chain'lenince ultra-tiny
(10⁻¹³). Q15 multiplier'da temsil edilemiyor. Pure integer pipeline icin:
- Custom scale-tracking (Q23+ multiplier + per-stage shift)
- int64 software accumulator (MSP430'da pahali)
- ya da QAT (quantization-aware training) ile baştan calibrate

### Karar mantigi (cost-benefit)
**Risk:**
- Iki inference path (toggle) → maintenance + test cifte
- Scale tracking bug'lari → sessiz veri bozulmasi, argmax flip
- 128 step icinde numerical drift birikir → ~%1-3 F1 dususu olası
- Live demo bug riski artar (sensor gelince debug zorlasiyor)
- SRAM gerilim (int64 buffer'lar 512 byte sinirini zorlar)
- Reproduction story bulanir (iki path acikla)

**Kazanc:**
- Real-time? ZATEN var (LUT yetiyor)
- Daha kucuk Flash? Şu an %62 kullanim, gerek yok
- Dusuk guc? Marjinal, olculmedi
- Sadece "2-3x daha hizli" — kullanim yeri yok

**Sonuc:** Mevcut LUT versiyonu "çözülmüş problem" noktası. Pure Q15 daha çok
solution-looking-for-a-problem.

### Eger ileride yapilirsa
1. Önce QAT yapmak: PyTorch'ta integer-aware training, scale alignment ogren
2. ONNX Runtime / TFLite Micro export ile cross-validation
3. Custom Q-format spec ile int64 accumulator pipeline
4. Tek dosyada toggle yerine ayri proje (clean separation)
5. Tahmini sure: 6-10 saat realistic

### Hafta 8 LUT bulgusu (Q15 yerine)
LUT degisikligi tek basina:
- Arduino: 1.51× hizlanma (1877 → 1245 ms full window)
- MSP430: 30.5× hizlanma (53996 → 1773 ms) ← projenin biggest win
- Iki kart birebir ayni logits (cross-platform consistency)
- Build size 10726 → 10200 (LUT 2KB ekledi, expf/tanhf cikti, NET +tasarruf)

LUT zaten optimization sweet-spot'unu yakaladi. Pure Q15 ileride hafta-sonu
extension olarak yapilabilir, simdi ertelenir.

---

## Yeri gelirse buraya not edilecek diger fikirler
- (Bos)
