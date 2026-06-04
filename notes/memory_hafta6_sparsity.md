# FastGRNN Project — Hafta 6 Sparsity

## Oturum — Hafta 6 (L-S-Q'nun "S" fazi) — 2026-05-25/26

### Yapilanlar
1. **fastgrnn_model.py'a `SparseLowRankFastGRNNCell` eklendi:**
   - LowRank ile ayni parametreler (W1, W2, U1, U2 + biases + zeta/nu)
   - Maskeler `register_buffer` olarak (parametre degil, gradient yok)
   - Forward'da `W1 * mask_W1` (elementwise)
   - `apply_pruning(target)` metodu: per-tensor magnitude pruning
   - `current_sparsity()`, `effective_params()` yardimcilari
   - `FastGRNNClassifier(sparse=True)` ile cagrilir
2. **train_sparse.py yazildi:**
   - IHT (Iterative Hard Thresholding) cubic schedule
   - `s_t = s_final * (1 - (1 - t/ramp)^3)`
   - Sicak baslangic: r_u=8 checkpoint'ten warm-start
   - Mask uygulamasi her epoch basinda (sparsity ramp up)
   - Gradient mask: forward sonrasi grad ile mask carpildi, optimizer step sonrasi
     mask tekrar uygulandi (numeric drift)
   - `--best_after_ramp` flag eklendi: best_val_f1 sadece ramp sonrasi takip
     (yoksa erken faz mild sparsity'den lucky best yakaliyor)

### Single-seed sparsity sweep (target ∈ {0.3, 0.5, 0.7, 0.9}, seed=0)
| Target | Actual sp | Eff. params | Test Acc | Test F1 |
|-------:|----------:|------------:|---------:|--------:|
| Dense  |    0%     |    430      |  0.918   |  0.918  |
| 30%    |   26%     |    344      |  0.897   |  0.899  |
| **50%** | **50%** | **283**     | **0.924** | **0.921 ⭐** |
| 70%    |   69%     |    226      |  0.840   |  0.830  |
| 90%    |   88%     |    167      |  0.741   |  0.716  |

U-egrisi: %50 zirvede, klasik bias-variance trade-off.
Hedef vs actual fark sebebi: per-tensor pruning, kucuk matrislerde tam yuzde tutturmak zor.

### MULTI-SEED dogrulama (sp50 x 5 seed)
Dense r_u=8 ile birebir karsilastirma (ayni seed'ler):

| Metric    | Dense r_u=8     | Sparse sp50     | Delta     |
|-----------|----------------:|----------------:|----------:|
| F1 mean   | 0.879           | 0.856           | -0.023    |
| F1 std    | ±0.056          | ±0.099          | +0.043    |
| Sigma     | -               | -               | ~0.29σ    |
| Params    | 430             | 283 (-35%)      | -         |

**Single-seed iddiamiz duzeltildi:**
- ESKI: "Sparse sp50 dense'i gecti (0.921 vs 0.918)" — seed=0'in iyi sansi
- YENI: "Sparse sp50 dense ile esit performans, varyans daha yuksek"

### Ham F1 (her seed)
- Sparse sp50: 0.921, 0.680, 0.893, 0.895, 0.890
- Dense r_u=8: 0.918, 0.781, 0.901, 0.890, 0.904

Seed=1 her iki modelde de patlatti — kotu init, mimari bagimsiz.
Sparse seed=1'de daha derin patlatti (0.680 vs 0.781) → sparsity instability'i artiriyor.

### Per-class varyans karsilastirma
| Class       | Dense std | Sparse std | Yorum          |
|-------------|----------:|-----------:|----------------|
| WALKING     | ±0.089    | ±0.195     | Daha kararsiz  |
| UPSTAIRS    | ±0.137    | ±0.210     | Daha kararsiz  |
| DOWNSTAIRS  | ±0.057    | ±0.109     | Daha kararsiz  |
| SITTING     | ±0.036    | ±0.047     | Benzer         |
| STANDING    | ±0.033    | ±0.058     | Hafif daha    |
| LAYING      | ±0.012    | ±0.007     | Sabit          |

Sparsity dinamik siniflarda varyansi belirgin artirdi.

### REPRODUCTION RAPORUNA EKLENECEK BULGULAR (Hafta 6)
1. **U-shape sparsity trade-off:** %50 noktasi optimal; %30'da regulariser zayif,
   %70+'ta kapasite kritik bottleneck. Bias-variance dengesinin gorsel ispati.
2. **Compression accuracy-preserving:** %50 sparsity ile %35 daha az efektif param,
   F1 degisimi istatistiksel olarak anlamsiz (~0.3σ).
3. **Sparsity varyansi artiriyor:** Multi-seed std 0.056 → 0.099. Pratik sonuc:
   sparse model daha az kararli; production icin "best of N seed" stratejisi gerekir.
4. **best_after_ramp metodolojik onlemi:** Sparsity ablasyonlarinda best_val_f1
   takibini ramp sonrasi yapmak SART, yoksa erken mild-sparsity epoch'tan lucky
   checkpoint kacirilir. Reproduction defensiveness.
5. **Per-tensor pruning kucuk matrislerde target'i tam tutturamiyor:** target 0.3 →
   actual 0.26 (16 elemanli W1 icin tam-sayi bolunmesi). Reproduction uyarisi.

### Resmi Hafta 6 modeli
- **Deploy:** sparse_h16_rw2_ru8_sp50_s0_e100_best.pt (single best, F1 0.921, 283 params)
- **Reported:** sp50 multi-seed mean 0.856 ± 0.099
- Compression: 12,518 (MLP) → 283 → **44× az parametre, ~ayni accuracy**

### Sonraki: Hafta 7 — Quantization (L-S-Q'nun "Q" fazi)
- Q15 sabit-nokta: float32 → int16
- Aşama 1: Post-training quantization (PTQ) — agirliklar dogrudan quantize
- Aşama 2: Quantization-aware training (QAT) — fine-tune ile kayip telafi
- Hedef: F1 ~0.91 civarinda kalsin, model boyutu Arduino (32KB) ve MSP430 (16KB)
  Flash'a sigsin
- Sicak baslangic: sparse sp50_s0 checkpoint

### Eklenenler / guncellemeler
- fastgrnn_model.py: SparseLowRankFastGRNNCell + FastGRNNClassifier(sparse=True)
- train_sparse.py: IHT + best_after_ramp
- run_sparsity_sweep.py: 4 target paralel sweep
- run_sparse_multiseed.py: sp50 x 5 seed
- aggregate_sparse_seeds.py: multi-seed ozet
- experiments/sparse_h16_rw2_ru8_sp{30,50,70,90}_s0_e100.json (target sweep)
- experiments/sparse_h16_rw2_ru8_sp50_s{0..4}_e100.json (multi-seed)
- logs/sparsity_sweep/, logs/sparse_multiseed/
- sparse_h16_rw2_ru8_sp{...}_best.pt checkpoint'ler
