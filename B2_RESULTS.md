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
