# FastGRNN Project — Hafta 7 Quantization (Q15 PTQ)

## Oturum — Hafta 7 (L-S-Q'nun "Q" fazi) — 2026-05-25

### Yapilanlar
1. **quantize.py yazildi:**
   - `q15_round(x, scale)`: per-tensor agirlik kuantizasyonu, scale = max_abs/32767
   - `quantize_weights(model)`: cell + classifier'in tum agirliklarini in-place Q15
   - `q15_activation(x, max_abs)`: aktivasyon kuantizasyonu, esnek max_abs (Q15/Q14)
   - `calibrate_activations(model, loader, n_batches)`: float pass ile gercek
     aktivasyon max'larini olc
   - `wrap_cell_with_calibrated_quantization(cell, stats, headroom)`: kalibre
     edilmis scale'lerle cell forward'i sar
   - `model_size_bytes(model, dtype_bits)`: deploy boyut hesabi (sparse + Q15)
2. **ptq_eval.py:** tek model PTQ, weights only
3. **ptq_full_eval.py:** multi-seed PTQ + aktivasyon kuantizasyon (3 mode)

### Sonuc Headline

| Stage              | Effective params | Boyut    | F1 (multi-seed)    |
|--------------------|-----------------:|---------:|-------------------:|
| L (low-rank r_u=8) |           430   |  1.7 KB  | 0.879 ± 0.056      |
| LS (+ sparsity 50%)|           283   |  1.1 KB  | 0.853 ± 0.107      |
| **LSQ (+ Q15)**    |       **283**   | **0.55 KB** | **0.853 ± 0.107** |

L -> LSQ: **3x boyut kucuklemesi**, F1 quantization fazinda etkisiz.

MLP karsilastirmasi:
- MLP: 12,518 param, F1 0.847
- **FastGRNN LSQ: 283 efektif param, Q15 (0.55 KB), F1 0.853**
- **44x daha az parametre, ~ayni accuracy**

### Donanim butcesi
- Arduino Uno (ATmega328P): 32 KB Flash, 2 KB SRAM. Model %1.73 Flash kullanim.
- MSP430G2553: 16 KB Flash, 512 B SRAM. Model %3.45 Flash kullanim.

### PTQ KEŞFI — kayipsiz agirlik PTQ
- Float32 -> Q15 (ağırlıklar) sıfır accuracy kayıp:
  delta_f1 = +0.0001 ± 0.0001
- Sebep: Q15 cozunurlugu (3e-5) FastGRNN agirlik araligi icin fazlasiyla
  yeterli; sparse modelin az parametresi gurultu birikimini engelliyor.
- PTQ yeterli, QAT gereksiz (paper QAT yapıyor ama bizim gibi tiny model icin
  PTQ zaten kayipsiz).

### NAIVE AKTIVASYON Q15 ÇOKER — KRITIK BULGU
- Sabit max_abs=1.0 ile aktivasyon Q15 uygulayinca F1 0.85 → 0.16 (felaket)
- STANDING sinifi tamamen kayboldu (F1 = 0)
- Sebep: `h_t` formul gereği [-1, 1] sinirini asar:
  `h_t = (zeta·(1-z) + nu) · h~ + z · h_prev`
  trained zeta≈0.55, nu≈0.5 ile katsayilar toplami 1+0.5·z, **1'i aşar**
  → h_prev geometrik buyur, 128 step boyunca birikir.

### KALIBRASYON KESF
Train setinden 5 batch ile gercek max'lari olctuk:
- z (sigmoid output): max ~1.00 (beklenen)
- h_tilde (tanh output): max ~1.00 (beklenen)
- **h_t: max ~60-68 (BEKLENMEDIK!)** — paper bunu yazılı belirtmiyor

Q15 [-1, 1) `h_t` icin uygun değil. Gerçek deploy'da `h_t` için ya:
- Q9 (1 sign + 6 integer + 9 fractional, range [-64, 64)) kullan
- Veya per-tensor calibrated scale (bizim simulasyon yaptigi gibi)

Kalibrasyon (headroom=1.1) ile aktivasyon Q15 uyguladigimizda:
- delta_f1 vs float = +0.0001 ± 0.0005 (gurultu icinde)
- **Tam quantization simülasyonu artık kayipsiz**

### REPRODUCTION RAPORUNA EKLENECEK BULGULAR (Hafta 7) — KESINLIKLE
1. **Sparse FastGRNN agirlik PTQ tamamen kayipsiz.** QAT gerektirmez.
2. **`h_t` gerçek dinamik araligi >> [-1, 1].** Paper'in yazili olmayan tasarim
   sonucu: katsayi toplami (ζ·(1-z) + ν + z) trained modelde 1'i asiyor, yani
   `h_t` zamanla buyumeye meyilli. Uygulamada h_t magnitudes 60+ gozlemlendi.
   Bu, Q15 uniform deploy'in mumkun OLMADIGINI kanitlar.
3. **Naive (kalibrasyonsuz) aktivasyon Q15 = catastrophic failure** (-70 puan F1).
   STANDING sinifi tamamen yok olur. Reproduction raporunda "deploy gotcha" olarak.
4. **Kalibrasyon ile aktivasyon Q15 kayipsiz** (+0.0001 ± 0.0005 delta).
   5 train batch yeterli, n_batches >> bu fark etmiyor.
5. **Mixed-precision deploy zorunlulugu:**
   - Agirliklar: Q15 (per-tensor calibrated scale)
   - z, h_tilde: Q15 ([-1, 1) cikis garantili)
   - h_t: Q9 (~[-64, 64)) veya per-tensor calibrated scale (~[-75, 75) bizim icin)
   - pre (intermediate): int32 accumulator
6. **Headroom=1.1 yeterli:** test setinde calibration'i %10 asma riski yok.

### Sicak baslangic noktasi (Hafta 8 icin)
- Production checkpoint: `sparse_h16_rw2_ru8_sp50_s0_e100_best.pt`
- Bu checkpoint'in Q15 ağırlıkları + kalibre scale'leri C kodunda kullanılacak
- Kalibrasyon scale'leri (seed=0 icin):
  - z_max ≈ 1.00 → Q15 scale = 1/32767
  - h_tilde_max ≈ 1.00 → Q15 scale = 1/32767
  - h_t_max ≈ 65 → fixed-point scale ≈ 75/32767 ≈ 2.3e-3
  - W1/W2/U1/U2/biases: her tensor icin ayri scale, quantize.py ureti

### Sonraki: Hafta 8 — Arduino C deploy
- C inference kodu (sabit-nokta, no float)
- AVR Q15 multiply: `int32_t prod = (int32_t)a * (int32_t)b; int16_t r = prod >> 15;`
- Ağırlıkları PROGMEM'e göm (Flash kullan, SRAM koruyalim)
- MPU6050 surucu (I2C)
- 50 Hz örnekleme, streaming inference (sadece h_t SRAM'de tutulur)
- Canli demo: aktivite degisince LED/UART output
- Olcum: SRAM kullanimi, Flash kullanimi, inference gecikmesi, opsiyonel guc tuketimi

### Eklenenler
- quantize.py: q15_round, q15_activation, quantize_weights, calibrate_activations,
  wrap_cell_with_calibrated_quantization, model_size_bytes
- ptq_eval.py: tek-seed weights only PTQ
- ptq_full_eval.py: multi-seed + 3 mode (float/q15_w/q15_w+acts)
- experiments/ptq_q15.json
- experiments/ptq_full_multiseed.json

### Production deploy hazırlığı — checkpoint export listesi
Hafta 8'de C koduna gidecek tensorlar:
| Tensor       | Shape   | Type   | Scale (Q-format)        |
|--------------|---------|--------|--------------------------|
| W1           | (16, 2) | int16  | per-tensor (~4e-5)       |
| W2           | (3,  2) | int16  | per-tensor (~3e-5)       |
| U1           | (16, 8) | int16  | per-tensor (~3e-5)       |
| U2           | (16, 8) | int16  | per-tensor (~3e-5)       |
| mask_W1..U2  | (...)   | bit    | binary, opsiyonel        |
| b_z          | (16,)   | int16  | per-tensor               |
| b_h          | (16,)   | int16  | per-tensor               |
| zeta_raw     | scalar  | int16  | tek deger                |
| nu_raw       | scalar  | int16  | tek deger                |
| classifier.W | (6, 16) | int16  | per-tensor               |
| classifier.b | (6,)    | int16  | per-tensor               |
| input mean   | (3,)    | int16  | normalizasyon (Q-format) |
| input std    | (3,)    | int16  | normalizasyon            |

Boyut: ~283 sıfır olmayan ağırlık × 2 byte = 566 byte ağırlık. + ~50 byte scale lookup
+ ~1-2 KB inference code = toplam <2 KB Flash. Arduino'da bol bol yer var.
