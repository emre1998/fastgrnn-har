# Makale Şema Raporu — v2
*Claude + Codex ortak analizi, 28 Haziran 2026. Tüm sayılar `experiments/`'ten ölçülü.*

## 0. Kilitlenen tez (tek cümle)

> **Bayt bütçesi protagonist, FastGRNN değil.** Eşit recurrent kapasitede GRU daha iyi
> hücre; gerçek MCU bayt bütçesinde FastGRNN kazanır, çünkü sıkıştırma yapısı gizli-durum
> kapasitesini korurken GRU/LSTM küçülmek zorunda. Değerlendirme rejimi sonucu değiştirir.

FastGRNN = **test edilen hipotez**, kahraman değil.

## 1. Başlık yönü
- **Birincil:** *"When the Byte Budget Changes the Winner: Recurrent HAR on Sub-Kilobyte Microcontrollers"*
- **Alternatif:** *"Capacity, Compression, and Fixed-Point Recurrent HAR on 512 B–2 KB MCUs"*
- **KAÇIN:** "FastGRNN for Ultra-Constrained HAR" (zayıf v1 çerçevesi, hakem davet eder)

## 2. Ana bulgular (makalenin ampirik çekirdeği)

| Rejim | HAPT | WISDM | PAMAP2 | Sonuç |
|---|---|---|---|---|
| **Eşit kapasite (H=16)** | GRU 0.917 | GRU 0.764 | GRU 0.389 | GRU önde (net sadece HAPT) |
| **Eşit bayt (deployment)** | berabere | **FastGRNN 0.800** | **FastGRNN 0.444** | FastGRNN 2/3 kazanır |

- **Tersine dönüş** = makalenin nontrivial sonucu.
- **Mekanizma:** bütçeye sığarken GRU H16→H6 çöker (WISDM -0.10), FastGRNN H=16 korur.
- **Q15 near-lossless** üç dataset üç hücre (ΔF1~0.000).
- **L-S-Q çöküşü** HAPT'a özgü (std 0.081, worst 0.708); WISDM'de FastGRNN EN kararlı (0.009).

## 3. Bölüm yapısı + ELİMİZDE NE VAR / NE LAZIM

| Bölüm | İçerik | Durum |
|---|---|---|
| **Abstract** | Reversal ile başla (implementasyon değil) | ✏️ yeniden yaz |
| **1. Intro** | Düzeltilmiş premise; negatif sonucu erken söyle (eşit-H'de GRU kazanır); 5 katkı | ✏️ yeniden yaz |
| **2. Related Work** | *Değerlendirme rejimleri* etrafında (eşit-H / eşit-param / eşit-bayt / ölçülü deployment) | 🔧 var ama yeniden organize (5 alt bölüm güçlü) |
| **3. Methods** | Regime A (eşit kapasite) + Regime B (eşit bayt) ayrı; sıkıştırma reçeteleri; MCU deployment | ✏️ yeni deneyleri ekle |
| **4. Experiments** | E1 eşit-kapasite → E2 eşit-bayt → E3 mekanizma → E4 Pareto → E5 quantization → E6 deployment | ✏️ yeni tablolar |
| **5. Failure Analysis** | L-S-Q çöküş modu, seed varyansı, IHT geri-dönülmezliği, per-class, dataset-özgüllüğü | 🆕 YENİ bölüm |
| **6. Discussion** | "eşit-kapasite ≠ eşit-bütçe = farklı sorular"; FastGRNN avantajı koşullu | 🔧 var, tezi güncelle |
| **7. Conclusion** | Tek paragraf, zafer turu yok | ✏️ |

## 4. Reviewer #2 — 3 saldırı + savunma

1. **"Yeni değil, takas bariz."** → Novelty soyut takas değil, gerçek MCU kısıtında **kontrollü ampirik tersine dönüş**, 3 dataset, deterministik Q15, ölçülü enerji. "Sistem değerlendirme + deployment çalışması" de, "yeni mimari" deme.

2. **"Eşit-bayt HAKSIZ — FastGRNN özel sıkıştırma, GRU/LSTM sadece küçültülüyor."** ← EN ZOR.
   - Savunma: "modelin amaçladığı sıkıştırma yolu" + her modelin bütçeye neden sığdığını gösteren tablo (H, nonzero, bayt, aktivasyon belleği).
   - **Elimizde:** HAPT'ta GRU-pruned (0.886) ölçtük → shrink-H'den (0.903) düşük; yani GRU'ya en iyi yolunu verdik, FastGRNN bütçede yine önde.
   - **Karar gerek:** bu pruned ablation'ı 3 dataset'e yaymalı mıyız? VEYA "kahraman = yapısal sıkıştırma" reframe'i.

3. **"İstatistik zayıf: 5 tohum, HAPT çöküşü, PAMAP2 düşük."** → Varyansa yaslan (mean/std/worst), HAPT'ta aşırı iddia etme, PAMAP2'yi "kasıtlı zor tek-bilek" çerçevele.

## 5. Gerçek eksik analizi (codex fazla kötümser — yarısı elimizde)

| İhtiyaç | Durum |
|---|---|
| MLP baseline | ✅ VAR (0.847) |
| INA226 enerji | ✅ VAR (FastGRNN) |
| Bit-tam çift-platform | ✅ VAR (Arduino+MSP430) |
| Warm-up | ✅ VAR |
| Sıkıştırma ablation parçaları | ✅ kısmi (sparse sweep, low-rank ckpt) |
| **GRU/LSTM MCU latency/enerji** | ❌ EKSİK (sadece FastGRNN deploy) |
| **Temiz ablation tablosu** (dense→LR→+sparse→+Q15) | ❌ derlenmeli |
| **Formal bellek-bütçe tanımı** | ❌ yazılmalı |
| **3-dataset GRU/LSTM pruned ablation** | ⚠️ opsiyonel (saldırı #2 için) |

## 6. Senin karar noktaların (kapsam)

- **K1:** GRU/LSTM'i gerçek MCU'ya port edip latency/enerji ölçelim mi (büyük iş) yoksa "byte/MAC tahmini + FastGRNN ölçümü" diye dürüstçe mi çerçeveleyelim?
- **K2:** Saldırı #2 için GRU/LSTM pruned ablation'ı 3 dataset'e yayalım mı, yoksa "yapısal sıkıştırma kahraman" reframe'iyle mi gidelim?
- **K3:** Bellek-bütçe tanımı: sadece ağırlık baytı mı, SRAM-resident model mi, aktivasyon dahil mi?

## 7. v1→v2 anlatısı
Geri çekme DEĞİL. "İddia **underspecified**'dı — hücre kalitesini deployable sıkıştırma-kapasite
takasıyla karıştırmıştı; bu revizyon premise'i doğrudan test ediyor, ölçüm iddiayı rafine ediyor."
Özür yok, gömme yok, dramatik "yanlıştı" yok. Intro'da veya kısa "Relation to Prior Version" notunda.
