# B2 — Donanım Ölçüm Sonuçları
*Latency + enerji + bellek, GRU/LSTM × MSP430/Arduino. FastGRNN referansı için: docs/energy_measurement.md.*

## Latency (TEST_MODE 0, µs)
| Hücre | Platform | LUT | per-step µs | window µs | %util @50Hz | real-time? |
|-------|----------|-----|-------------|-----------|-------------|------------|
| FastGRNN | MSP430 | 1 | 13900 | 1778000 | 69.5 | ✅ (ref) |
| GRU  | MSP430 | 1 | 12116 | 1550900 | 60.6 | ✅ OK |
| GRU  | MSP430 | 0 | 19273 | 2467000 | 96.4 | ✅ OK (marginal) |
| LSTM | MSP430 | 1 | 12370 | 1583400 | 61.9 | ✅ OK |
| LSTM | MSP430 | 0 | 22097 | 2828400 | 110.5 | ❌ FAIL |
| GRU  | Arduino| 1 | ? | ? | ? | ? |
| LSTM | Arduino| 1 | ? | ? | ? | ? |

## Enerji (TEST_MODE 3, INA226, steady-state)
*NOT: Önceki tur bir Debug/Flash karışıklığı yüzünden şüpheli bulundu ve TAMAMEN SIFIRLANDI (6 Tem 2026).
Yeni prosedür: her #define değişikliğinde Save → Project Clean → Build → Debug ile çalıştır (hep aynı yöntem, tutarlılık için).
Sıra: önce LSTM (baştan), sonra GRU (baştan).*
| Hücre | Platform | BENCH | LUT | I (mA) | V (mV) | P (mW) |
|-------|----------|-------|-----|--------|--------|--------|
| (idle platform-bağımsız — MSP430 idle <0.025 mA / <0.09 mW, referans FastGRNN'den) |
| LSTM | MSP430 | 1 (50Hz) | 1 | 5.058 | 3479.2 | 17.70 |
| LSTM | MSP430 | 2 (cont) | 1 | 5.077 | 3478.9 | 17.70 |
| LSTM | MSP430 | 2 (cont) | 0 | 5.078 | 3478.9 | 17.70 |
| GRU  | MSP430 | 1 (50Hz) | 1 | 5.066 | 3478.9 | 17.65 |
| GRU  | MSP430 | 2 (cont) | 1 | 5.054 | 3478.9 | 17.70 |
| GRU  | MSP430 | 2 (cont) | 0 | 5.054 | 3478.8 | 17.70 |

## Enerji/çıkarım — TÜRETİLMİŞ (E = ölçülen güç × ölçülen latency)
*Güç (INA226) doğrudan ÖLÇÜLDÜ; latency doğrudan ÖLÇÜLDÜ; enerji bu ikisinden TÜRETİLDİ (hesaplama, ölçüm değil).
Bu ayrım hakem için kritik — tabloda "measured" ve "derived" net ayrılıyor. 9 Tem 2026 revizyonu.*

Aktif güç platform-baskın ve tüm hücrelerde ~17.7 mW'de sabit (§Enerji). Bu YANLIŞLIKLA "enerji de eşit"
gibi okunmamalı: güç sabit + latency farklı ⟹ **enerji/çıkarım latency'yi izler.** LUT gücü değiştirmez ama
çıkarımı erken bitirdiği için **türetilen enerji/pencereyi düşürür.**

| Hücre | LUT | Ölçülen P (mW) | Ölçülen t_step (ms) | Ölçülen t_window (ms) | Türetilen E/step (mJ) | Türetilen E/window (mJ) |
|-------|-----|----------------|---------------------|-----------------------|-----------------------|--------------------------|
| GRU  | 1 | 17.70 | 12.116 | 1550.9 | 0.214 | 27.5 |
| GRU  | 0 | 17.70 | 19.273 | 2467.0 | 0.341 | 43.7 |
| LSTM | 1 | 17.70 | 12.370 | 1583.4 | 0.219 | 28.0 |
| LSTM | 0 | 17.70 | 22.097 | 2828.4 | 0.391 | 50.1 |
| FastGRNN | 1 | 17.72 | 13.900 | 1778.0 | 0.246 | 31.5 |
| FastGRNN | 0 | 17.66 | 26.100 | 3338.0 | 0.461 | 58.9 |

*Tüm no-LUT satırları AYNI rejim: derleyici-optimize (aynı cl430 -O ayarı). FastGRNN güç değerleri kendi
ölçümünden (17.72/17.66 mW, docs/energy_measurement.md); GRU/LSTM 17.70 mW — üçü de 1 INA226 LSB içinde eşit.*

BULGU (doğru çerçeve): **Ölçülen güç LUT'tan ve hücreden bağımsız (~17.7 mW), ama türetilen enerji/pencere
LUT ile GRU'da −%37 (43.7→27.5 mJ), LSTM'de −%44 (50.1→28.0 mJ), FastGRNN'de −%46 (58.9→31.5 mJ) düşüyor.**
Üç hücre aynı rejimde tutarlı (LUT tasarrufu %37–46 bandı). Hücreler arası: GRU LUT en verimli (27.5 mJ),
LSTM no-LUT en kötü (50.1 mJ). Makale üç katmanlı: latency (ölçülen) → güç (ölçülen) → enerji (türetilen).

### Derleyici -O duyarlılığı — TAM SWEEP (ÖLÇÜLEN, 9 Tem akşam)
Kontrollü tam matris: AYNI kod, sadece CCS Optimization level değişti. 3 hücre × USE_LUT{0,1} × -O{off,0,1,2,3,4} = 36 ölçüm.
GRU/LSTM: TEST_MODE=1 (Windows timed, zero input). FastGRNN: run_embedded_tests ("Inference time", gerçek test verisi, per-step=window/128).
Not: MSP430G2553'te `--use_hw_mpy = none` (donanım çarpıcı yok) — "çarpıcısız MCU" tezini teyit eder. Enerji GEREKMEDİ (E=P×t, P sabit ~17.7mW; her -O için türetilir). Flash de sweep'lenmedi (ayrı soru, deploy Flash = production @ -O3).

**no-LUT (per-step ms):**
| Hücre | off | 0 | 1 | 2 | 3 | 4 | Δ(off→plato) |
|-------|-----|-----|-----|-----|-----|-----|-----|
| GRU  | 20.202 | 19.663 | 19.510 | 19.273 | 19.273 | 19.273 | −%4.6 |
| LSTM | 22.801 | 22.800 | 22.800 | 22.800 | 22.800 | 22.800 | ~%0 |
| FastGRNN | 27.2 | 26.2 | 26.8 | 26.1 | 26.1 | 26.6 | ~%4 |

**LUT (per-step ms):**
| Hücre | off | 0 | 1 | 2 | 3 | 4 | Δ(off→plato) |
|-------|-----|-----|-----|-----|-----|-----|-----|
| GRU  | 13.041 | 12.504 | 12.353 | 12.116 | 12.116 | 12.116 | −%7.1 |
| LSTM | 13.091 | 12.609 | 12.442 | 12.371 | 12.371 | 12.371 | −%5.5 |
| FastGRNN | 15.3 | 14.2 | 14.2 | 13.9 | 13.9 | 13.9 | −%8.9 |

BULGULAR:
1. **no-LUT: -O neredeyse ETKİSİZ** (%0–4.6). Kök sebep: **MSP430G2553'te FPU YOK** → her float çarpma/toplama DA
   (`expf`/`tanhf` gibi) önceden-derlenmiş **RTS soft-float** çağrısı. Proje -O'su bu kütüphane rutinlerinin içini
   değiştirmez; sadece etraflarındaki döngü/indeksleme glue'yu (ihmal edilir pay) etkiler. Yani MCU'nun neden yavaş
   olduğu (donanım çarpıcı/FPU yok) ile -O'nun neden işe yaramadığı **aynı kök** — tezle birebir örtüşür.
2. **LUT: -O ılımlı kazanç** (%5.5–8.9), **her hücrede -O2'de doyuyor** (-O3/-O4 aynı). Transandantaller tablolaşınca
   derlenebilir glue oranı artar → -O biraz iş görür; ama float MAC'ler hâlâ soft-float olduğu için kazanç sınırlı ve erken plato.
3. **Deploy -O3 = plato değeri** (her yerde); -O2 zaten yakalıyor, ötesi fayda yok — mevcut deploy ayarı yeterli.
4. Enerji/pencere bu değerlerden türetilir (P≈17.7mW sabit); -O'nun enerjiye etkisi = latency'ye etkisi (çok küçük).

⚠️ **54s figürü KESİN GÖMÜLDÜ (iki bağımsız kanıt):**
(a) **FastGRNN'in KENDİ güncel kodunda** -O off = ~27ms/step (3.4s/window) ölçüldü — 54s DEĞİL. Yani 54s bir "-O off"
    ölçümü olamaz; -O'nun tavanı zaten ~%9. (b) Kullanıcı doğruladı: **54s, Week 8'de MPU6050 SENSÖR döngüdeyken
    yapılan bir live-mode deneyiydi** (I2C read + örnekleme + Q15-öncesi/eski kod dahil), saf-hesaplama latency'si değil.
Her iki sebeple 54s bir tablo SÜTUNU YAPILAMAZ → "erken, sensörlü live-mode deneyi (Week 8), non-comparable" diye
etiketli bir dipnot olur. 30.5×/−%96.7 anlatısı ana tablolardan çıkarılır; yerine bu -O-duyarsızlık bulgusu geçer.
−%46 FastGRNN enerji figürü sağlam kalır (no-LUT ~26ms, -O-bağımsız plato).

### Future Work — sensörlü (MPU6050) gerçek-dünya latency'si
Bu geceki tüm sayılar SAF INFERENCE (gömülü veri, sensörsüz). Gerçek dağıtımda per-örnek = MPU6050 I2C read + normalize +
inference. Yapılacak: GRU + LSTM için, LUT{0,1} × sensör-döngüde (TEST_MODE=0 LIVE), uçtan-uca 50Hz latency ölç.
Harness boşluğu: şu an sadece FastGRNN'de live harness var (run_live_mode + USCI_B0/MPU6050); GRU/LSTM harness'lerine
I2C sürücüsü PORT edilmeli. Nüans: sensör maliyeti hücre-bağımsız/toplamsal (kıyas ekseni bench inference kalır) ama
real-time marjını sıkar (örn. no-LUT ~19-23ms + sensör → 20ms bütçesini kesin aşar; LUT ~12-14ms + sensör → sınıra yaklaşır).

### ⚠️ Metodolojik sınır — busy-wait / aktif-rejim üst-sınırı (dürüstlük notu)
Firmware BENCH1/BENCH2'de örnekler arasında **LPM (uyku) kullanmıyor, busy-wait yapıyor**
(`while ((millis_ccs() - t0) < 20) {}`, kodda doğrulandı — yalnızca BENCH0/IDLE LPM3'e giriyor). Bu yüzden:
- BENCH1 (50Hz) ile BENCH2 (continuous) **aynı gücü** verdi (bekleme sırasında CPU aktif kalıyor) — anomali değil, beklenen.
- Raporlanan enerji değerleri firmware'in **aktif çalışma rejimini** yansıtır ⟹ **sistem enerjisinin ÜST SINIRI**, ulaşılabilir minimum değil.
- Gerçek dağıtımda örnekler arası boşluk (20ms − t_compute) LPM3'e verilirse ortalama güç yaklaşık **aktif duty-cycle** (= latency %util: GRU LUT %60.6, LSTM no-LUT %110→uyuyamaz) oranında düşer. Yani LUT'un asıl enerji faydası LPM'li sistemde daha da büyür — LSTM no-LUT ise deadline'ı aştığı için hiç uyuyamaz.

**Future Work (makaleye):** MSP430 LPM0/LPM3 modlarını örnekleme aralıklarına entegre ederek latency azalmasının
*toplam sistem enerjisine* etkisini gerçekçi ölçmek. Mevcut değerler "aktif enerji üst sınırı" olarak yorumlanmalı.

## Bellek — TEST HARNESS build (referans, test kapsamı dahil: UART banner + sprint_* + 10x timing tekrarı)
*Tutarlı ölçüm: üçü de AYNI ayarda — TEST_MODE=1, BENCH_MODE=1, USE_LUT=1 (9 Tem 2026).
Bu sayılar GERÇEK deployment'ı DEĞİL, test/ölçüm kodunu da içeren build'i yansıtır — aşağıdaki
"PRODUCTION" tablosu asıl deployment footprint'i, makalede o kullanılacak.*
| Hücre | Platform | Flash (B) | SRAM (B) |
|-------|----------|-----------|----------|
| FastGRNN | MSP430 | 10214 | 348 |
| GRU  | MSP430 | 6546 | 308 |
| LSTM | MSP430 | 6908 | 324 |

## Bellek — PRODUCTION build (ASIL deployment footprint, UART/debug/tekrar YOK)
*msp/ccs_{cell}_production/ — sadece init + gerçek streaming döngü + LED aksiyonu. 9 Tem 2026.*
| Hücre | Platform | Flash (B) | SRAM (B) |
|-------|----------|-----------|----------|
| FastGRNN | MSP430 | 5544 | 348 |
| GRU  | MSP430 | 5392 | 308 |
| LSTM | MSP430 | 5742 | 324 |

### Production testinin metodolojisi (neye göre ölçüldü)

**Amaç:** Yukarıdaki "test harness" tablosu (satır 29-37), bir hakem tarafından haklı olarak
itiraz edilebilecek şişirilmiş sayılardı — çünkü UART banner metni, `sprint_u`/`sprint_f3`
debug yazdırma fonksiyonları ve latency ortalaması almak için 10× tekrarlanan zamanlama
döngüsü, gerçek dağıtılmış bir cihazda **hiç bulunmaz.** "Production" build bu farkı kapatır.

**Her hücre için ölçüm koşulları (üçü de birebir aynı):**
- **Donanım:** MSP-EXP430G2ET (MSP430G2553), 16 MHz kalibre DCO clock
- **Derleyici:** TI cl430 (ti-cgt-msp430_21.6.1.LTS), `-O3`, aynı proje ayarları (heap=0, stack=256B)
- **Firmware kapsamı — DAHİL OLANLAR (gerçek dağıtımda da gerekli):**
  - `clock_init()` + `timer_init()` (Timer_A, 1ms tik) — gerçek 50Hz örnekleme temposu için
  - Hücrenin çıkarım motoru (`gru.cpp`/`lstm.cpp`/`fastgrnn.cpp`, değişmemiş, B1'de bit-tam doğrulanan aynı kod)
  - Q15 ağırlıklar (`model_weights.h`) + sigmoid/tanh LUT tabloları (`lut.h`, 256-giriş, USE_LUT=1)
  - Sonsuz döngü: pencere boyunca (WINDOW_T adım) 20ms'de bir örnek besle (`{cell}_step`) → pencere bitince sınıflandır (`{cell}_predict`) → sınıfa göre LED (P1.0) aç/kapat (gerçek bir "aşağı akış aksiyonu" temsili — gerçekte bir aktüatör/radyo paketi olurdu)
- **Firmware kapsamı — HARİÇ TUTULANLAR (test-only, gerçek dağıtımda gereksiz):**
  - UART başlatma (`uart_init`) ve TÜM `sprint_*`/`sputc` yazdırma kodu — hiç UART çıktısı yok
  - Banner metinleri ("GRU HAR - MSP430G2553...") ve ilişkili string sabitleri
  - Latency ölçümü için 10× tekrar döngüsü (N_TIMING_WINDOWS) — sadece bir kez, gerçek zamanlı çalışır
  - Girdi: sabit sıfır (`{0,0,0}`) — çünkü Flash/RAM boyutu koddan gelir, veriden değil (bkz. latency bölümündeki aynı gerekçe); MPU6050/I2C sürücüsü bilerek yok (GRU/LSTM firmware'i sensöre bağlanmadı, sadece hesaplama+bellek izole edildi)
- **Derleme doğrulaması:** Her üç `main.cpp` de host'ta `cl430 --compile_only` ile önce sözdizimi hatasız derlendi (exit 0), sonra CCS'te gerçek MSP430'a flash edilip Debug Output'taki linker özeti (`Flash/FRAM usage is X bytes. RAM usage is Y bytes.`) okundu.
- **Proje izolasyonu:** Karışıklığı önlemek için her hücre **ayrı, yepyeni bir CCS projesinde** (eski FastGRNN/GRU/LSTM projelerinin yeniden kullanılması değil) derlendi — B1/B2'de birkaç kez yaşanan "main bulunamıyor" / stale-build hatalarından ders alınarak.

**Sonuç yorumu:** Test-harness'e göre düşüş (FastGRNN -%46, GRU -%18, LSTM -%17) beklenen yöndeydi
ve FastGRNN'in en büyük düşüşü, orijinal `ccs_fastgrnn_har` harness'inin (MPU6050+I2C+live-mode
kodu içeren, GRU/LSTM'in minimal harness'inden daha zengin) diğerlerinden daha fazla test-only
kod taşımasıyla açıklanıyor. Production sayıları (5392-5742B Flash) makaledeki **tek yetkili**
deployment footprint iddiası olacak; test-harness sayıları sadece şeffaflık için referans olarak duruyor.
