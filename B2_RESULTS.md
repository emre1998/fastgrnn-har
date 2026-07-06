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
| Hücre | Platform | BENCH | LUT | I (mA) | V (mV) | P (mW) |
|-------|----------|-------|-----|--------|--------|--------|
| (idle platform-bağımsız — MSP430 idle <0.025 mA / <0.09 mW) |
| GRU  | MSP430 | 1 (50Hz) | 1 | ? | ? | ? |
| GRU  | MSP430 | 2 (cont) | 1 | ? | ? | ? |
| GRU  | MSP430 | 2 (cont) | 0 | ? | ? | ? |
| LSTM | MSP430 | 1 (50Hz) | 1 | ? | ? | ? |
| LSTM | MSP430 | 2 (cont) | 1 | ? | ? | ? |
| LSTM | MSP430 | 2 (cont) | 0 | ? | ? | ? |

## Bellek (derleyici çıktısı)
| Hücre | Platform | Flash (B) | SRAM (B) |
|-------|----------|-----------|----------|
| GRU  | MSP430 | ? | ? |
| LSTM | MSP430 | ? | ? |
