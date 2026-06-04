# Arduino Uno Test Kaydi

## Hedef kart
- Arduino Uno R3
- MCU: ATmega328P
- Serial: 115200 baud
- Test modu: `TEST_MODE=1`
- Sensor: MPU6050 bagli degil
- Test verisi: `test_data.h` icine gomulu 2 adet HAPT penceresi
- Pencere boyutu: 128 sample x 3 eksen

Bu dosya Arduino tarafinda yapilan kart-ustu testleri tek yerde tutar.
Canli sensor testleri basladiginda yeni sonuclar ayni dosyaya eklenecek.

---

## Test 1 - Ilk C inference dogrulamasi (2026-05-26)

Arduino IDE 2.3.9 ile derleme ve Arduino Uno'ya yukleme basarili.

### Derleme sonucu
| Metrik | Deger |
|---|---:|
| Flash | 11302 / 32256 byte (%35) |
| Global SRAM | 356 / 2048 byte (%17) |
| Kalan SRAM | 1692 byte |

### Kart-ustu test sonucu
| Test | Prediction | Expected | True label | Full-window latency | Sonuc |
|---|---|---|---|---:|---|
| 0 | 4 - STANDING | 4 - STANDING | 4 - STANDING | 1877 ms | PASS |
| 1 | 2 - DOWNSTAIRS | 2 - DOWNSTAIRS | 1 - UPSTAIRS | 1906 ms | PASS |

`Expected`, Python/C referans inference tahminidir. Ikinci pencerenin gercek
etiketi farkli olsa da kart referans implementasyonla ayni tahmini urettigi
icin deploy correctness testi PASS kabul edilir.

### Logits
```text
Test 0: -14.981  3.071 -50.579  7.721  9.671 -16.556
Test 1:   9.485  9.900  10.465  1.848 -2.332  -2.534
```

### Sonuc
- Q15 weight + float compute C inference Arduino Uno uzerinde calisiyor.
- Kart tahminleri Python referansiyla ayni.
- Full-window latency yaklasik 1.9 saniye.

---

## Test 2 - LUT aktivasyon optimizasyonu sonrasi (2026-05-27)

`expf()` ve `tanhf()` cagrilari 256-entry sigmoid/tanh LUT ile degistirildi.
Ayni iki gomulu pencere yeniden calistirildi.

### Kart-ustu test sonucu
| Test | Onceki latency | LUT sonrasi latency | Hizlanma | Sonuc |
|---|---:|---:|---:|---|
| 0 | 1877 ms | 1245 ms | 1.51x | PASS |
| 1 | 1906 ms | 1251 ms | 1.52x | PASS |

### Dogrulama
- Prediction class degerleri degismedi.
- Logits MSP430 sonucu ile 5 ondalik basamak seviyesinde ayni kaldi.
- LUT, matematik kutuphanesine bagimliligi azaltti ve cross-platform
  deterministic inference sagladi.

### Streaming latency yorumu
Full-window olcumu 128 sample icindir. Bir sample icin turetilmis hesap:

```text
1245 ms / 128 = 9.73 ms/sample
1251 ms / 128 = 9.77 ms/sample
```

50 Hz sampling periodu `20 ms/sample` oldugu icin Arduino Uno real-time
streaming butcesinin icinde kalir. Bu deger kart uzerinde sample-level timer
ile ayri olculmus degil, full-window latency'den turetilmistir.

### Sonuc
- LUT optimizasyonu Arduino Uno'da yaklasik 1.5x hizlanma sagladi.
- Turetilmis compute suresi yaklasik 9.7 ms/sample.
- 50 Hz live HAR icin yaklasik %52 compute headroom var.

---

## Henuz yapilmayan Arduino testleri
- `TEST_MODE=0` ile MPU6050 live streaming
- Gercek sensor verisiyle sample-level latency olcumu
- Live prediction UART loglari
- Aktivite degisimlerinde LED davranisi
- Sleep acik/kapali duty-cycle karsilastirmasi
- INA226 ile MCU-only ve end-to-end enerji olcumu
- Canli test verilerinden grafik ve demo kaydi

