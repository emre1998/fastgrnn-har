# v2 Yürütme Planı — sıralı
*28 Haz 2026 kilitlendi.*

**BAŞLIK/KAPSAM (deliverable):** "A Systematic Evaluation of Lightweight Recurrent Neural Networks
for Human Activity Recognition on Ultra-Constrained [Multiplier-less] Microcontrollers: Accuracy,
Memory, Latency, and Energy" (GPT framing, kullanıcı onaylı).
**MOTİVASYON (intro kancası):** "Çarpıcısız sub-kilobayt MCU'lar gerçek-zamanlı HAR yapabilir mi?"
**DURUŞ:** objektif, advokasi yok. Üç hücre (GRU/LSTM/FastGRNN) = aday mimariler, nötr kıyas. Her şey 3 dataset.
**Kapsam ≠ motivasyon:** fizibilite = neden önemli (stakes); sistematik 3-mimari değerlendirme = ne yaptık (scope).

Kararlar: **K1**=GRU/LSTM tam deployment+ölçüm · **K2**=pruned rota 3 dataset · **K3**=iki-parçalı bütçe (Flash ağırlık + SRAM çalışma seti).

---

## FAZ A — Yazılım deneyleri (donanım gerekmez, Claude sürer)

- [x] **A1 (K2)** `run_baseline_tier2_pruned.py` parametrik + ağırlık-Q15. ✅ commit 9f15844
- [x] **A2 (K2)** GRU/LSTM pruned-to-budget, 3 dataset × 5 tohum, 200ep. ✅ commit 6da41e5. analyze_best_route.py + best_route_summary.json. SONUÇ: en iyi rota tablosu → HAPT gru0.902(pruned)/FastG0.869 kazanan GRU · WISDM FastG0.800/gru0.767(pruned) · PAMAP2 FastG0.444/gru0.354. Saldırı#2 kapandı (baseline'a 2 rota, FastG yine 2/3 kazanır). Nüans: pruning de H16 kapasitesini korur → GRU-pruned WISDM'de sıçradı (0.683→0.767), FastG avantajı +0.033'e daraldı. Rafine mekanizma: kapasite-koruyan sıkıştırma > shrink; FastGRNN yapısal low-rank+sparse en etkili.
- [x] **A3** Sıkıştırma ablation (dense→low-rank→+sparse→+Q15), 3 dataset. ✅ commit c92118e (run_lowrank_stage.py + analyze_ablation.py). BULGU: instabilite LOW-RANK adımında doğuyor (HAPT std 0.018→0.098), IHT'de büyümüyor (worst 0.666→0.708 toparlıyor) — dünkü "IHT amplifies" alt-iddiası DÜZELDİ. HAPT'a özgü low-rank×dataset etkileşimi; WISDM'de low-rank doğruluğu ARTIRIP varyansı DÜŞÜRÜYOR (0.748→0.797, std 0.033→0.018). Sparse/Q15 near-neutral, Q15 kayıpsız. Sıkıştırma çoğu zaman regularizing.
- [x] **A4 (K3)** SRAM+Flash iki-parçalı muhasebe. ✅ commit 1db1c65 (analyze_footprint.py). SRAM streaming: FastGRNN ~114-126B, GRU ~146-158B, LSTM ~210-222B — hepsi 512B'ye sığar. Pencere-sakla 768B (MSP OVER) → streaming şart. Flash 472-770B << 16KB. (SRAM analitik, Faz B .map ile teyit edilecek.)
- [x] **A4b** Eşit-beyin (H=16) footprint dense→sıkıştırılmış: FastGRNN 1760→566B (3.1×), GRU 4440→2220B, LSTM 5784→2892B. ✅ Eşit beyinde FastGRNN 3.9-5.1× küçük → bütçede GRU/LSTM neden H küçültür.

**FAZ A TAMAM (A1-A4b) — 1 Tem 2026.** Sıradaki: Faz B (donanım, kullanıcı cihazı gerekir).

## FAZ B — Donanım deployment + ölçüm (senin Arduino+MSP430+INA226 kurulumun gerekir)

- [x] **B1 (K1)** GRU H6 + LSTM H5 bit-tam C çıkarımı yazıldı + host'ta g++ ile DOĞRULANDI (5/5 %100, commit f03436b). Arduino .ino + MSP430 Energia harness (latency+enerji, commit 727a6f3). GRU 480B/F1 0.915, LSTM 472B/F1 0.818. build_deploy_firmware.py + verify_firmware.py.
- [x] **B2 (K1)** TAMAM (8-9 Tem 2026, B2_RESULTS.md). CCS (Energia değil) kullanıldı — msp/ccs_gru_har + msp/ccs_lstm_har + msp/ccs_uart_test yazıldı. Debug/Flash staleness + proje karışıklığı debug edildi (banner-metni "fingerprint" yöntemiyle çözüldü — TEST_MODE 1'in kendi UART çıktısı "GRU HAR"/"LSTM HAR" + H değeri kesin kanıt verdi). Enerji tamamen baştan ölçüldü (temiz prosedür: Save→Clean→Build→Debug, hep aynı yöntem).
  LATENCY (LUT var): FastGRNN 13.9ms(69.5%) · GRU 12.116ms(60.6%) · LSTM 12.370ms(61.9%) — üçü REAL-TIME OK.
  LATENCY (LUT yok): GRU 19.273ms(96.4%, sınırda OK) · LSTM 22.097ms(110.5%, FAIL) · FastGRNN ~421ms(FAIL). BULGU: LUT hücre-bağımsız ön-koşul, LSTM'de kesin gerekli.
  ENERJİ (INA226, ÖLÇÜLEN güç, steady-state): GRU BENCH1 17.65mW/BENCH2 17.70mW(LUT)/17.70mW(no-LUT) · LSTM BENCH1 17.70mW/BENCH2 17.70mW(LUT)/17.70mW(no-LUT). Ölçülen güç LUT'tan ve hücreden bağımsız (~17.7mW, GRU≈LSTM≈FastGRNN) — platform-baskın.
  ENERJİ/ÇIKARIM (TÜRETİLEN, E=ölçülen P × ölçülen t): güç sabit ama latency farklı ⟹ enerji latency'yi izler. E/window (hepsi derleyici-optimize no-LUT rejimi, adil): GRU LUT 27.5mJ vs no-LUT 43.7mJ (−%37); LSTM LUT 28.0mJ vs no-LUT 50.1mJ (−%44); FastGRNN LUT 31.5mJ vs no-LUT 58.9mJ (−%46). DOĞRU ÇERÇEVE (ChatGPT+ajan konsensüsü, 9 Tem revizyon): "LUT ölçülen gücü DEĞİL, türetilen enerji/çıkarımı düşürür" — "measured" vs "derived" ayrımı korunur.
  DERLEYİCİ -O DUYARLILIĞI (ÖLÇÜLEN, 9 Tem akşam): no-LUT latency -O'ya neredeyse duyarsız — GRU 19.273ms(-O3)→20.202ms(off) +%4.8; LSTM 22.097ms(-O3)→22.801ms(off) +%3.2 (ikisi de her -O'da FAIL). SEBEP: expf/tanhf TI RTS kütüphanesinde önceden-derlenmiş, proje -O'su değiştirmez. use_hw_mpy=none (çarpıcısız MCU teyidi).
  ⚠️ 54s DÜZELTİLDİ: Bu ölçüm, FastGRNN 54s/421ms "optimize-edilmemiş" figürünün -O off OLMADIĞINI kanıtladı (mevcut kod -O off = ~20-23ms). 54s = Week 8 farklı implementasyon (kaynak-içi Taylor/full-float). 54s SÜTUN YAPILMAZ, 30.5×/96.7% ana tablodan çıkar, -O-duyarsızlık bulgusu yerine geçer. −%46 (FastGRNN ~26ms, -O-bağımsız) sağlam kalır.
  ⚠️ SINIR: firmware busy-wait (BENCH1/2'de LPM yok, kodda doğrulandı) ⟹ enerji = AKTİF REJİM ÜST SINIRI, min değil. Future work: LPM0/3'ü örnekleme aralıklarına entegre et (fayda ~duty-cycle=%util oranında; LSTM no-LUT deadline aştığı için uyuyamaz).
  BELLEK (CCS build, adil-kapsam minimal harness): GRU Flash 5936B/RAM 310B · LSTM Flash 6908B/RAM 324B (LSTM +972B Flash [4 kapı], +14B RAM [ekstra c_state], analitik tahminle tutarlı). FastGRNN'in orijinal tam-özellikli build'i (8778B/348B, MPU6050+I2C dahil) FARKLI KAPSAMLI, doğrudan kıyaslanmadı — makalede ANALİTİK ağırlık-footprint (566/480/472B, analyze_footprint.py) adil kıyas olarak kullanılacak.
  Commit'ler: 1b5e6e7, 13d8769, 231ebc8, 7fe3bc7, 2c9ce31, 15f9ad6, 3d239b8.
- [ ] **B3 (K1+K3)** Per-hücre deployment tablosu derle: doğruluk · Flash · SRAM · latency · enerji · **measured/estimated etiketi**. "Gerçek-zamanlı" iddiasını latency ile kanıtla (çıkarım << pencere süresi). B2_RESULTS.md'deki ölçülü verileri + analyze_footprint.py'deki analitik SRAM'i birleştir.

**FAZ B DONANIM ÖLÇÜMÜ TAMAM (B1+B2) — 9 Tem 2026.**

### B2 RİGOR DÜZELTMESİ (9 Tem 2026) — kullanıcının kritik yakalaması
Kullanıcı sordu: "bu bellek verileri test senaryosu için, gerçek kullanım için değil, sorun olur mu?" HAKLI ÇIKTI. İlk ölçülen sayılar (TEST_MODE=1 harness: UART banner+sprint_*+10x tekrar döngüsü) gerçek deployment'ı ŞİŞİRİYORDU. Çözüm: msp/ccs_{gru,lstm,fastgrnn}_production/ — sıfır UART/debug/tekrar, sadece init+gerçek 50Hz streaming döngü+LED aksiyonu (commit 05e3257). Üçü de cl430 ile derleme-doğrulandı (exit 0).
NİHAİ PRODUCTION BELLEK (asıl deployment iddiası, commit 64ac170):
  FastGRNN: Flash 5544B, RAM 348B (test-harness'e göre 10214→5544, -%46 — en büyük düşüş, orijinal harness'i MPU6050+I2C+live-mode içeriyordu)
  GRU:      Flash 5392B, RAM 308B (6546→5392, -%18)
  LSTM:     Flash 5742B, RAM 324B (6908→5742, -%17)
Sıralama: GRU < FastGRNN < LSTM, hepsi 16KB Flash/512B SRAM bütçesinin rahat altında. Bu tablo makalenin YETKİLİ Flash/RAM iddiası olacak (test-harness sayıları B2_RESULTS.md'de "referans" olarak ayrıca duruyor, şeffaflık için).
B2 KESİN TAMAM — latency+LUT ablasyonu+enerji+GERÇEK production bellek, 3 hücre, MSP430.

Sıradaki: B3 (tablo derleme, koşu/ölçüm gerekmez) → Faz C (yazım).

## FAZ C — Makale yazımı (kanıt derlendikten sonra)

- [ ] **C1** Başlık + Abstract — fizibilite/reversal ile başla, objektif ton. ("Real-Time HAR on Multiplier-less MCUs...")
- [ ] **C2** Intro — düzeltilmiş premise; tez = çarpıcısız MCU fizibilitesi; eşit-H'de GRU önde negatif sonucunu erken söyle; 5 katkı.
- [ ] **C3** Related Work — *değerlendirme rejimleri* etrafında yeniden organize (eşit-H / eşit-param / eşit-bayt / ölçülü deployment).
- [ ] **C4** Methods — Regime A (eşit kapasite) + Regime B (eşit bayt); sıkıştırma reçeteleri; iki-parçalı bütçe (K3); MCU deployment.
- [ ] **C5** Experiments — E1 eşit-kapasite → E2 eşit-bayt → E3 mekanizma → E4 Pareto → E5 quantization → E6 deployment (latency/enerji/SRAM).
- [ ] **C6** Failure Analysis (YENİ bölüm) — L-S-Q çöküşü, seed varyansı, IHT geri-dönülmezliği, per-class, dataset-özgüllüğü (HAPT'a özgü).
- [ ] **C7** Discussion — objektif; eşit-kapasite ≠ eşit-bütçe = farklı sorular; FastGRNN avantajı koşullu; sıkıştırma bedava değil.
- [ ] **C8** Conclusion (tek paragraf) + **"Relation to Prior Version"** notu (underspecified→refined, geri çekme değil).

## FAZ D — Yayın

- [ ] **D1** arXiv v2 yükle (şeffaf changelog).
- [ ] **D2** ACM TECS'e gönder (arXiv ID cover letter'da).

---

### Deployability gerçeği (B-fazı + footprint tablosu için kritik)
GRU/LSTM MSP430'a SIĞAR ve deploy edilir. Footprint baytları (GRU 2220B, LSTM 2892B) = FLASH (16KB) ağırlıkları, SRAM değil. 512B SRAM sadece streaming çalışma setini tutar (gizli durum+girdi+kapı scratch ≈ 150-250B, hepsi sığar). Tabloda Flash vs SRAM AYRIMI net gösterilmeli (karıştırma = yanlış "sığmaz" sonucu). Asıl MSP430 maliyeti: donanım çarpıcısı YOK → GRU/LSTM 3-4 kapı = daha çok MAC = daha kötü latency/enerji. Bu ölçülecek = "deploy edilebilir ama bedeli var" bulgusu, "real-time + çarpıcısız" tezini destekler (LSTM failure-mode girdisi).

### Nötr-ton kuralı (her bölümde)
Hiçbir hücreyi savunma. "GRU eşit kapasitede önde; FastGRNN eşit baytta önde; işte sebebi; işte dürüst varyans (HAPT çöküşü dahil)." Katkılar (Q15+LUT, bit-tam, enerji, warm-up) = fizibiliteyi mümkün kılan, hücreden bağımsız araçlar.
