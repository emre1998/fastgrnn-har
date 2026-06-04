# FastGRNN Project — Hafta 5 Multi-Seed Bulgulari

## Oturum — Hafta 5 ablasyon + multi-seed dogrulama (2026-05-25)

### Yapilanlar
1. fastgrnn_model.py: LowRankFastGRNNCell eklendi (W = W1@W2.T, U = U1@U2.T).
   Distribute carpim: x_t @ W2 @ W1.T (CPU/Flash tasarrufu icin).
2. train_fastgrnn.py: --r_w, --r_u, --patience, --seed, --tag_seed eklendi.
3. r_u sweep (single seed=0): r_u in {4, 6, 8, 12}
4. Multi-seed grid: r_u x seed = 4 x 5 = 20 kosu, paralel CPU (4 worker x 2 thread).
   78 dakika sirec.
5. aggregate_seeds.py: mean +- std + per-class std + sigma anlamilik.

### Single-seed sweep sonuclari (s=0, ilk gun)
| r_u | Params | Acc    | Macro-F1 | Yorum |
|----:|-------:|-------:|---------:|-------|
|   4 |    302 | 0.891  | 0.893    | Dinamik siniflarda en iyi |
|   6 |    366 | 0.876  | 0.872    | Vadi (her iki tarafa da yaramayan) |
|   8 |    430 | 0.920  | 0.918    | Mutlak en iyi, vanilla'yi gecti |
|  12 |    558 | 0.845  | 0.838    | Beklenmedik kotu (outlier hipotezi) |

### MULTI-SEED bulgular (5 seed her config)
| r_u | Params | F1 mean | F1 std  | Yorum |
|----:|-------:|--------:|--------:|-------|
|   4 |    302 |  0.847  | ±0.074  | Yuksek varyans |
|   6 |    366 |  0.837  | ±0.077  | Yine yuksek |
| **8** |  **430** | **0.879** | **±0.056** | **Kazanan, en dusuk varyans** |
|  12 |    558 |  0.836  | ±0.076  | Yuksek varyans |

### Ham F1 her seed icin (KRITIK BULGU)
- r_u= 4: 0.893, 0.721, 0.898, 0.885, 0.841
- r_u= 6: 0.872, 0.713, 0.892, 0.896, 0.814
- r_u= 8: 0.918, 0.781, 0.901, 0.890, 0.904
- r_u=12: 0.838, 0.709, 0.901, 0.847, 0.885

**Seed=1 her config icin anormal kotu (0.71-0.78).** Mimari secimden bagimsiz,
genel egitim hassasiyeti — belirli baslangic noktasi kotu vadi uretiyor.

### Istatistiksel anlamlilik
- r_u=8 vs r_u=4 : delta=+0.031, ~0.5 sigma
- r_u=8 vs r_u=6 : delta=+0.041, ~0.6 sigma
- r_u=8 vs r_u=12: delta=+0.043, ~0.6 sigma
- HEPSI <1 SIGMA. Standartta >2 sigma aranir.
- "r_u=8 vanilla'yi 1.3 puan gecti" iddiasi tek seed sonucuydu, multi-seed seed
  gurultusunun icinde.

### Per-class std raporu
| Class       | r_u=4 std | r_u=8 std | Daha kararli |
|-------------|----------:|----------:|--------------|
| WALKING     | ±0.239    | ±0.089    | r_u=8        |
| UPSTAIRS    | ±0.077    | ±0.137    | r_u=4        |
| DOWNSTAIRS  | ±0.152    | ±0.057    | r_u=8        |
| SITTING     | ±0.016    | ±0.036    | r_u=4        |
| STANDING    | ±0.048    | ±0.033    | r_u=8        |
| LAYING      | ±0.019    | ±0.012    | r_u=8        |

r_u=8 dinamik siniflarda (WALKING, DOWNSTAIRS) cok daha az saviniyor — kritik.

### DUZELTILMIS IDDIA (raporda yer alacak)
ESKI (single-seed): "r_u=8 vanilla'dan 1.3 puan iyi"
YENI (multi-seed): "r_u=8, 5 seed ortalamasinda 0.879 F1 verdi (±0.056).
   Tum konfigler arasinda hem en yuksek mean hem en dusuk varyans.
   Konfigler arasi fark istatistiksel olarak anlamli degil (~0.5-0.6 sigma)
   ancak r_u=8 pratik olarak en guvenilir nokta."

### REPRODUCTION RAPORUNA EKLENECEK BULGULAR (yeni)
1. FastGRNN single-seed sonuclari yaniltici — varyans buyuk (~5-10 puan F1).
2. Architecture (r_u) farki seed gurultusunun icinde kaybolabilir.
3. Belirli "kotu seed"ler (seed=1) tum configlerde patlatti — model baslangic
   hassasiyeti yuksek. Gelecek is: better init / LR warmup / Xavier-orthogonal
   init for U.
4. Paper, multi-seed ablasyon raporlamiyor — bizim 5-seed analizimiz hem
   modelin kararliligini hem de paper iddialarinin pratik anlami konusunda
   ek bilgi veriyor.

### Resmi Hafta 5 model: r_u=8
- 5-seed kanit: en yuksek mean F1 + en dusuk varyans
- Dinamik siniflarda en kararli
- Hafta 6 (sparsity) baseline olarak kullanilacak

### Sonraki: Hafta 6 — Sparsity
- L-S-Q pipeline'in "S" fazi
- r_u=8 model uzerinde magnitude pruning + iterative hard thresholding
- W1, W2, U1, U2'nin cogu agirligini sifira zorla
- Hedef: efektif parametre 430 -> ~200, F1 dususu <2 puan

### Eklenenler / guncellemeler
- experiments/fastgrnn_h16_rw2_ru{4,6,8,12}_s{0..4}_e100.json (20 dosya)
- experiments/multiseed_summary.json (aggregator ozeti)
- logs/multiseed/ru{R}_s{S}.log (20 detayli log)
- run_multiseed_sweep.py
- aggregate_seeds.py
- notes/extension_ideas.md (dual-rank fikri Hafta 8 sonrasi icin)
