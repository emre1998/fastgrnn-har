# FastGRNN Project — Hafta 8 (C Deploy Hazirlik)

## Oturum — Sensorsuz C deploy hazirligi (2026-05-25/26)

MPU6050 1 Haziran'da geliyor. O zamana kadar tum C tarafini hazirladik:
Arduino Uno + MSP430G2553 icin paralel paket, sensorsuz test modu, Python
ile birebir matematiksel dogrulama.

---

### 1. Q15 weights export (export_to_c.py)
- sparse_h16_rw2_ru8_sp50_s0_e100_best.pt checkpoint'inden Q15 ağırlıkları
  C header dosyasına aktarir
- Per-tensor scale her tensor icin: W1=4.24e-5, W2=3e-5, U1=2.87e-5, ...
- Kalibrasyon: train setinden 5 batch ile gercek aktivasyon max'lari olculur
  - z_max = 1.0, h_tilde_max = 1.0, h_t_max = ~62 (gercek!)
- Headroom %10 ile final scale'ler
- 281 sifir olmayan agirlik × 2 byte = **562 byte ağırlık** Flash'ta
- model_weights.h hem AVR (PROGMEM) hem MSP430 (const) icin portable yazildi
- Export her iki klasore de yazar: arduino/ ve msp/

### 2. C inference engine (fastgrnn.cpp, fastgrnn.h)
- **Strateji:** Weight-Q15 Flash + float compute. En kolay, en hatasiz,
  Python ile birebir.
- API:
  - `fastgrnn_reset()`
  - `fastgrnn_step(float x[3])`  — streaming, bir ornek
  - `fastgrnn_predict()` — sinif index (0-5)
  - `fastgrnn_classify_window(float X[128][3])` — convenience
- Forward: low-rank distribute carpim (`xn @ W2 @ W1.T`, `h_prev @ U2 @ U1.T`)
- Aktivasyon: sigmoid_f + tanh_f (extreme saturate erken cikis)
- Tek source: hem Arduino hem MSP430'da derlenir (`#ifdef __AVR__` ile pgmspace)
- SRAM kullanim: h_state[16] = 64B, last_logits[6] = 24B, lokal scratch ~168B
  → toplam ~300B working set

### 3. Python validation (test_inference_python.py)
- C kodun algoritmasini Python'da AYNISINI yaz (Q15 weight × scale → float matematik)
- Test setinde 3399 pencere uzerinde PyTorch reference ile karsilastir
- **SONUC: %100 prediction match, F1 0.9176 (her ikisi de)**
- Logit mutlak fark: max=0.15, mean=0.0036 (Q15 dequant gurultusu, argmax'i degistirmiyor)
- **Mathematical validation: C kodu mathematically dogru.**

### 4. Test verisi gomulu (test_data.h, generate_test_data.py)
- test_vectors.json'dan ilk 2 pencere PROGMEM'e gomulur
- N_TEST_SAMPLES = 2, TEST_EXPECTED[] ve TEST_TRUE[] etiketler
- Flash kullanim: 2 × 128 × 3 × 4 = 3 KB
- Hem arduino/ hem msp/ klasorune yazilir (portable, #ifdef __AVR__)

### 5. Arduino sketch (fastgrnn_har.ino)
- Arduino Uno R3 (ATmega328P) hedef
- TEST_MODE 1: gomulu test, sensor gerekmez
  - 2 pencere icin: pred, expected, true, latency, PASS/FAIL, logits
- TEST_MODE 0 (1 Haziran sonrasi): MPU6050 streaming, 50Hz, her saniye predict
- Serial: 115200 baud
- LED: pin 13 (test modunda yanip soner)
- Beklenen Flash kullanim: ~6500 byte (20% / 32KB)
- Beklenen SRAM: ~200 byte (10% / 2KB)

#### Arduino Uno gercek kart testi (2026-05-26)
- Arduino IDE 2.3.9 ile derlendi ve Arduino Uno'ya yuklendi.
- Compile sonucu:
  - Flash: 11302 / 32256 byte = 35%
  - Global SRAM: 356 / 2048 byte = 17%
  - Kalan SRAM: 1692 byte
- Serial Monitor: 115200 baud, TEST_MODE=1.
- Gomulu 2 test penceresi calisti:
  - Test 0: pred=4 (STANDING), expected=4 (STANDING), true=4 -> PASS
    - Latency: 1877 ms
    - Logits: -14.981 3.071 -50.579 7.721 9.671 -16.556
  - Test 1: pred=2 (DOWNSTAIRS), expected=2 (DOWNSTAIRS), true=1 (UPSTAIRS) -> PASS
    - Latency: 1906 ms
    - Logits: 9.485 9.900 10.465 1.848 -2.332 -2.534
- Sonuc: Arduino Uno uzerinde C inference, gomulu test verilerinde Python referansi ile ayni tahminleri uretti.
- Not: Full-window TEST_MODE latency ~1.9 s olculdu. Bu, rapora gercek kart olcumu olarak eklenmeli; sensorlu streaming modda ayrica sample-level latency olculecek.

### 6. MSP430 sketch (msp/fastgrnn_har_msp/fastgrnn_har_msp.ino)
- MSP-EXP430G2 LaunchPad + MSP430G2553 hedef
- Energia 1.8 ile derlenir (Arduino-like syntax)
- LED: RED_LED (P1.0) — GREEN_LED (P1.6) I2C kullaninca devre disi (J5 jumperi cikar)
- Serial: 9600 baud (MSP430 default)
- I2C: USCI_B0, P1.6=SCL, P1.7=SDA
- **SRAM butce uyarisi: 512 byte sinir, ~310 byte tahmin (60% kullanim)**
- Flash: ~9 KB tahmin (56% / 16 KB)
- TEST_MODE icin aynı sketch yapisi, sensor gerekmez

#### CCS alternatifi (2026-05-27)
- Energia indirme linki S3 AccessDenied verdigi icin MSP430 testi Code Composer Studio'ya tasindi.
- Yeni klasor: `msp/ccs_fastgrnn_har/`
- Energia/Arduino API kullanmayan bare-metal `main.cpp` eklendi:
  - DCO: calibrated 1 MHz
  - UART: USCI_A0, 9600 baud, P1.1 RX / P1.2 TX
  - Timer_A: 1 ms tick ile latency olcumu
  - LED: P1.0 heartbeat
- Ayni inference dosyalari ve ayni gomulu test verileri kullaniliyor:
  - `fastgrnn.cpp`, `fastgrnn.h`
  - `model_weights.h`
  - `test_data.h`
- CCS'te hedef: MSP430G2553, Empty MSP430 Project.
- Beklenen test sonucu yine:
  - Test 0: pred=4, expected=4 -> PASS
  - Test 1: pred=2, expected=2 -> PASS

#### MSP430G2553 gercek kart testi - CCS (2026-05-27)
- Code Composer Studio ile build ve debug/yukleme basarili.
- Linker ayarlari:
  - heap_size=0
  - stack_size=256
- Serial Monitor: COM6, 9600 baud.
- Gomulu 2 test penceresi calisti:
  - Test 0: pred=4 (STANDING), expected=4 (STANDING), true=4 -> PASS
    - Latency: 53996 ms
    - Logits: -14.981 3.071 -50.579 7.721 9.671 -16.556
  - Test 1: pred=2 (DOWNSTAIRS), expected=2 (DOWNSTAIRS), true=1 (UPSTAIRS) -> PASS
    - Latency: 55594 ms
    - Logits: 9.485 9.900 10.465 1.848 -2.332 -2.534
- Sonuc: MSP430G2553 uzerinde C inference, gomulu test verilerinde Python referansi ile ayni tahminleri uretti.
- Onemli performans notu: full-window inference ~54-56 s/pencere. Bu, 1 MHz DCO + software float + hardware multiplier yok ayariyla beklenenden cok yavas; correctness kanitlandi, live demo icin streaming latency ve/veya integer Q15 optimizasyonu gerekecek.

#### Python vs MCU hiz karsilastirmasi (full-window, 128x3)
- Python benchmark ayni 2 gomulu test penceresi uzerinde olculdu.
- Python C-equivalent NumPy:
  - Test 0: 2.7266 ms
  - Test 1: 2.7506 ms
- Python PyTorch reference, tek pencere:
  - Test 0: 16.4868 ms
  - Test 1: 16.7361 ms
- Arduino Uno, Python C-equivalent NumPy'ye gore:
  - Test 0: 1877 ms / 2.7266 ms = 688.4x yavas, hiz Python'un %0.145'i, hiz azalmasi %99.855
  - Test 1: 1906 ms / 2.7506 ms = 692.9x yavas, hiz Python'un %0.144'i, hiz azalmasi %99.856
- MSP430G2553, Python C-equivalent NumPy'ye gore:
  - Test 0: 53996 ms / 2.7266 ms = 19803x yavas, hiz Python'un %0.00505'i, hiz azalmasi %99.99495
  - Test 1: 55594 ms / 2.7506 ms = 20212x yavas, hiz Python'un %0.00495'i, hiz azalmasi %99.99505
- Arduino Uno, Python PyTorch reference'a gore ~114x yavas.
- MSP430G2553, Python PyTorch reference'a gore ~3275-3322x yavas.
- Yorum: Bu sayilar correctness testi icin kabul edilebilir ama live demo icin MSP430 tarafinda software float cok pahali. Sonraki optimizasyon hedefi: 16 MHz clock dogrulama, hardware multiplier varsa acma, scratch buffer azaltma ve/veya pure integer Q15 inference.

### 7. msp/README.md
- Energia kurulum talimatlari
- TI MSP430 USB driver
- Board ve port secimi
- J5 jumper cikartma (I2C icin)
- MPU6050 baglanti pinleri
- SRAM bütçe hesabi (tighter than Arduino)
- Sorun giderme tablosu
- Performans beklentisi: MSP430 ~5-10× yavas (carpici yok, software multiply)
  - Streaming 1 sample: ~5-10 ms (Arduino: ~1-2 ms)
  - Full window: ~700-1200 ms (Arduino: ~150-250 ms)
  - 50Hz period 20 ms → streaming rahat yetisir

---

## Klasor yapisi

```
fastgrnn-har/
├── arduino/
│   ├── export_to_c.py            ← weights → C header (her iki klasore yazar)
│   ├── generate_test_data.py     ← test vektoru gomme (her iki klasore)
│   ├── test_inference_python.py  ← C math doğrulama (100% match)
│   ├── test_vectors.json
│   └── fastgrnn_har/             ← Arduino IDE sketch klasoru
│       ├── fastgrnn_har.ino
│       ├── fastgrnn.h
│       ├── fastgrnn.cpp
│       ├── model_weights.h
│       ├── model_info.json
│       └── test_data.h
└── msp/
    ├── README.md                 ← Energia kurulum + MSP430 ozel notlar
    └── fastgrnn_har_msp/         ← Energia sketch klasoru
        ├── fastgrnn_har_msp.ino  ← MSP-spesifik
        ├── fastgrnn.h            ← (Arduino ile ayni)
        ├── fastgrnn.cpp          ← (Arduino ile ayni, portable)
        ├── model_weights.h       ← (auto-synced from arduino/)
        └── test_data.h           ← (auto-synced)
```

## Bugun yapilabilir olanlar (sensor olmadan)

1. **Arduino IDE'de Arduino sketch'i derle ve yukle** — sensor gerekmez
   - 2 test penceresi calistirilir, Python ile karsilastirilir
   - "PASS" beklenir (Python validation %100 match veriyor)
2. **Energia ile MSP430 sketch'i derle ve yukle** — sensor gerekmez
   - Ayni 2 test penceresi MSP430'da calisir
   - Eger PASS → Q15 ağırlıkları her iki donanim da dogru calistiriyor
3. **Inference latency olcum** — her iki kart icin

## 1 Haziran sonrasi yapilacaklar

1. MPU6050 I2C surucusu (Arduino: Wire.h, MSP430: Wire.h via Energia)
   - Wake-up, ±2g range, 50Hz sample rate
2. read_mpu6050() implementasyonu (stub'dan gercek)
3. TEST_MODE 0 (live) test
4. Canli demo: aktivite degisince LED + UART output
5. Olcum: SRAM, Flash, inference time, opsiyonel guc tuketimi

## ÖNEMLİ — Reproduction raporuna eklenecek
- **Mathematical validation:** C inference algoritmasi PyTorch reference ile
  3399 test penceresinde %100 prediction match. Q15 weight dequantization
  gurultusu (logit fark ~0.004 mean) argmax'i etkilemiyor.
- **Iki kart paralel deploy:** Arduino Uno (manşet) + MSP430G2553 (stretch).
  Aynı C source, platform farkı sadece `#ifdef __AVR__` ile PROGMEM macros.
  Tek seferlik C inference kodu, iki farklı bare-metal MCU'da çalışır.
- **Mixed-precision deploy:** Weight-Q15 + float compute. PTQ activation Q15
  bulgumuza göre (gercek h_t ~60'a kadar çıkıyor, kalibrasyon gerek) bu
  hibrit yaklaşım production'da en pragmatik. Pure-integer Q15 ileride
  optimizasyon için kapı açık.

## Sonraki oturum (1 Haziran sonrasi)
- MPU6050 I2C surucu
- Canli streaming inference
- Olcum (latency, accuracy on live data)
- Demo videosu / GIF
- Final reproduction raporu derleme

## Devam notu

MPU6050 ile MSP430G2553 canli streaming testi 2 Haziran 2026 tarihinde
tamamlandi. USCI_B0 kilitlenmesi, CCS stale `.out` problemi, GPIO I2C fallback
surucusu, sensor isinmasi kontrolu ve canli log ornekleri icin bkz:
[`memory_2026-06-02_msp_live_test.md`](./memory_2026-06-02_msp_live_test.md).
