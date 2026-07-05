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

- [ ] **A1 (K2)** `run_baseline_tier2_pruned.py`'yi `--data`/`--tag`/`--val_holdout` parametrik yap + ağırlık-Q15 ekle.
- [ ] **A2 (K2)** GRU/LSTM pruned-to-budget koş: HAPT + WISDM + PAMAP2, 5 tohum. → her baseline'a "ikinci rota", tabloda en iyisini al.
- [ ] **A3** Sıkıştırma ablation tablosu (FastGRNN): dense → low-rank → +sparse → +Q15, 3 dataset. Eldeki sparse/low-rank verisini derle, eksikse hedefli koşu. → Failure Analysis bölümünü besler.
- [ ] **A4 (K3)** SRAM çalışma-seti muhasebesi: her hücre için Flash (nonzero×2) + SRAM (gizli durum H×2 + girdi D×2 + scratch), streaming altında. 512B MSP430'a sığdığını analitik göster (~onlarca bayt; pencereyi saklamamak kritik).
- [ ] **A4b** Eşit-beyin (H=16) footprint tablosu, ÖNCESİ→SONRASI (dense FP32 → deployed Q15+sıkıştırma): FastGRNN 1760B→566B, GRU 4440B→2220B, LSTM 5784B→2892B. → sıkıştırılabilirlik farkını ölçülü göster (reversal'ın görsel ispatı). İsteğe bağlı 2. eksen: eşit-beyin footprint → eşit-bayt (FastGRNN H16 korur / GRU H6'ya düşer).

## FAZ B — Donanım deployment + ölçüm (senin Arduino+MSP430+INA226 kurulumun gerekir)

- [ ] **B1 (K1)** Claude: GRU H? + LSTM H? (dataset başına deploy config) için sabit-nokta **bit-tam C çıkarımı** yaz — Arduino Uno + MSP430. FastGRNN zaten deploy edilmiş, onu şablon al.
- [ ] **B2 (K1)** Sen: flash'la + ölç → her hücre için **latency/çıkarım** + **enerji/çıkarım (INA226)** + bit-tam doğrulama. (FastGRNN mevcut; GRU/LSTM yeni.)
- [ ] **B3 (K1+K3)** Per-hücre deployment tablosu derle: doğruluk · Flash · SRAM · latency · enerji · **measured/estimated etiketi**. "Gerçek-zamanlı" iddiasını latency ile kanıtla (çıkarım << pencere süresi).

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
