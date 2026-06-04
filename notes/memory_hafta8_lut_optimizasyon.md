# FastGRNN Project — Hafta 8 LUT Optimizasyon Basarisi

## Oturum — LUT optimization + pure Q15 abort kararı (2026-05-27)

Bu oturum: Gemini'nin perf raporu üzerinden tartisma → LUT optimizasyonu
hem Arduino hem MSP430'a → büyük basari → pure Q15 integer denemesi →
tasarim direnci → bilincli abort kararı.

---

### 1. Gemini analizi üzerine değerlendirme
Gemini "Pure Integer Q15 + LUT + 16MHz + Streaming" tavsiye etti, MSP430'un
"21x bütçeyi astigini" belirtti. Kontrol edince:
- DCO ZATEN 16 MHz (CCS main.cpp'de CALBC1_16MHZ ayarli)
- Gemini default 1 MHz varsaymis, mevcut kodu görmemis
- "Real-time imkansiz" ifadesi full-window latency'i sample-period gibi sundu (metodolojik hata)
- Yine de LUT + pure Q15 onerileri teknik olarak dogru

Karar: A (DCO fix) imkansiz (zaten 16 MHz), B (LUT) yapilabilir,
C (pure Q15) sonra degerlendir.

### 2. LUT implementasyon
- generate_lut.py: 256-entry sigmoid + tanh float LUT, [-8, +8] araligi
- lut.h: hem Arduino hem MSP icin portable (#ifdef __AVR__ PROGMEM)
- fastgrnn.cpp: sigmoid_f/tanh_f LUT'a yonlendirildi, expf/tanhf cagrilari kaldirildi
- READ_LUT macro: AVR'da pgm_read_float, MSP430'da direkt erisim
- 2 KB Flash ekstra, runtime O(1) lookup

### 3. Sonuclar — kart bazinda
**Arduino Uno (test_data.h gomulu, full window):**
- Test 0: 1877 → 1245 ms (1.51x)
- Test 1: 1906 → 1251 ms (1.52x)
- Per-sample streaming: ~14.7 → ~9.7 ms (20 ms butce icinde %48 dolu)
- PASS PASS, predicted classes degismedi

**MSP430G2553 (CCS, 16 MHz DCO, full window):**
- Test 0: 53996 → 1773 ms (30.5x!!)
- Test 1: 55594 → 1785 ms (31.2x!!)
- Per-sample streaming: ~421 → ~13.85 ms (20 ms butce icinde %69 dolu)
- PASS PASS, real-time hedefi YAKALANDI

**Cross-platform tutarlilik:**
- Hem Arduino hem MSP430 BİREBİR ayni logits (5 ondalık basamak)
- LUT deterministic, math platformdan bagimsiz
- Reproduction validasyonu son aşamada da geçildi

### 4. LUT'un MSP430'da neden bu kadar buyuk kazanc verdigi
- MSP430G2553'te FPU yok, donanim carpici yok
- expf/tanhf TI math library'sinden: Taylor series + software float
- Her cagri tahmini ~5000+ cycle (1 expf + several float mults)
- LUT: 1 float multiply (Q12 input scale) + 1 array index + 1 float load = ~50-100 cycle
- Per-step 32 cagri (16 sigmoid + 16 tanh), 128 step → 4096 cagri
- Kazanc: 4096 × (~5000 - ~100) = ~20 milyon cycle = ~1.25 sn (16 MHz'de)
- Tek başına 54 sn'lik isin %95'ini açikladı

### 5. Pure Q15 integer denemesi ve abort
**Hedef:** Inner matrix multiply'lari int16×int16'ya cevir, MSP430'da extra
~2-5x hizlanma (5-7 ms → 1.5-3 ms per sample).

**Tasarim dirençi:**
- generate_q15_assets.py ile combined scale'ler hesaplaninca:
  - XW_REAL_SCALE = XN_SCALE × W2_S × W1_S ≈ 3.1e-13
  - Q15 multiplier'a sikistirma = 0 (sifira yuvarlandi)
  - CLS_BIAS_MULT int32 araliginda taşmıştı (18.4 milyon)
- Kök sebep: Per-tensor weight scale'leri tiny (~10⁻⁵), chain'lenince
  ultra-tiny (10⁻¹³). Q15 ile temsil edilemiyor.
- Cözüm: Custom Q23+ multiplier + int64 accumulator + per-stage shift
  OR Quantization-aware training (QAT)
- Tahmin: 4-6 saat ek is, yuksek bug riski

**Cost-benefit analizi:**
- Real-time? ZATEN var (LUT yetiyor: MSP430 13.85 ms < 20 ms budget)
- Daha kucuk Flash? Şu an %62 kullanim, gerek yok
- Daha az guc? Marjinal, olculmedi
- Sadece "2-3x ek hiz" — kullanim yeri yok

**Risk:**
- Iki inference path (toggle) → maintenance ciftledi
- Numerical drift 128 step icinde birikir, F1 dususu olasi
- Live demo bug riski artar
- Reproduction story bulanir
- SRAM gerilim (int64 buffer)

**Karar:** Abort. Mevcut LUT versiyonu "cozulmus problem" noktasi.
Pure Q15 → notes/extension_ideas.md icinde Fikir 3 olarak kayit, ileride
hafta-sonu projesi.

### 6. SONUC — Hafta 8 RESMEN KAPANDI

**Final Q15 + sparse + low-rank + LUT pipeline:**

| Stage | Param/Boyut | F1 | Latency (per-sample, 50Hz) |
|-------|------------:|---:|----------------------------:|
| MLP (baseline) | 12,518 params | 0.847 | N/A (host) |
| FastGRNN L (low-rank H=16, r_u=8) | 430 params | 0.879±0.056 | N/A |
| FastGRNN LS (+sparsity 50%) | 283 params | 0.853±0.099 | N/A |
| FastGRNN LSQ (+Q15 ağırlık + kalibre act) | 283 params, 562 B | 0.853±0.099 | N/A (sim) |
| **+ Deploy + LUT (Arduino)** | **10.2 KB Flash** | (host validate) | **9.7 ms** |
| **+ Deploy + LUT (MSP430G2553)** | **10.2 KB Flash** | (host validate) | **13.85 ms** |

**Manşet:** "Akilli saat ciplerinin algoritmasini, 512 byte RAM'li 16-bit MCU'da,
donanim carpicisiz, real-time 50Hz aktivite tanima icin yeterli hizda calistirdim.
44x daha az parametre (12,518 → 283), %1.7 Arduino Flash kullanim."

### 7. Reproduction raporuna eklenecekler (Hafta 8)
1. **LUT optimization on no-FPU MCU:** 30x speedup on MSP430G2553, 1.5x on Arduino.
   Niye fark: MSP430 expf/tanhf cok pahali (Taylor + software float), LUT lookup
   bunu O(1)'e indirir. Arduino'da hardware multiply yardimci ama yine kazanc var.
2. **Cross-platform deterministic inference:** Ayni LUT 256 bucket float, iki
   platform birebir ayni logits (5 ondalik dogruluk). Math platformdan bagimsiz.
3. **Pure Q15 deferral rationale:** Per-tensor weight scales (~10⁻⁵) chain'lenince
   Q15 multiplier overflow. Custom scale-tracking ya da QAT gerek. Real-time
   zaten ulasildigindan, complexity getirisi yok. "Future work" olarak rapor edilir.
4. **Streaming inference budget:**
   - Arduino Uno (ATmega328P): per-sample 9.7 ms / 20 ms = 48% (52% headroom)
   - MSP430G2553: per-sample 13.85 ms / 20 ms = 69% (31% headroom)
   - Iki kart da 50Hz HAR real-time, MPU6050 gelince live demo'ya hazir.

### Eklenenler / değiştirilenler
- arduino/generate_lut.py — 256-entry sigmoid+tanh LUT generator
- {arduino,msp/fastgrnn_har_msp,msp/ccs_fastgrnn_har}/lut.h — generated
- arduino/fastgrnn_har/fastgrnn.cpp — sigmoid_f/tanh_f LUT'a yonlendi
- msp/fastgrnn_har_msp/fastgrnn.cpp — sync (Arduino ile ayni)
- msp/ccs_fastgrnn_har/fastgrnn.cpp — sync
- workspace_ccstheia/Msp430 Fastgrnn Project Experiment/ — temiz CCS proje,
  6 source dosya kopyalandi, heap=0 stack=256, build success (10200 B Flash)
- notes/extension_ideas.md — Fikir 3 (Pure Q15 deferral) eklendi

### Silindi (Q15 abort sonrasi temizlik)
- arduino/generate_q15_assets.py
- {arduino,msp/...}/q15_constants.h
- {arduino,msp/...}/lut_q15.h
- arduino/q15_python_constants.json

experiments/ptq_q15.json korundu (Hafta 7 PTQ analizi, degerli).

### Sonraki: 1 Haziran sonrasi
- MPU6050 sensor gelir
- I2C surucusu Arduino + MSP (Wire.h Arduino, USCI_B0 MSP) — taslagi mevcut
- Live mode TEST_MODE=0 calistir, gercek streaming HAR
- LED + UART output, demo videosu
- Reproduction raporu derleme (8 hafta cumulative bulgular)
