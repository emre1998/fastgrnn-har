# FastGRNN Project — Konusma Kaydi / Hafiza

## Oturum — Hafta 2, 3, 4 tamamlandi (2026-05-24)

### Hafta 2-3: PyTorch FastGRNN
- fastgrnn_model.py yazildi:
  * FastGRNNCell (nn.Module) — paylasimli W, U + kapi + h_tilde + zeta/nu birlesimi
  * FastGRNNClassifier (nn.Module) — T=128 adim donguss + Linear(H, num_classes) sinif kafasi
- zeta/nu artik OGRENILEBILIR ve KISITLI: ham parametre + sigmoid(0,1).
  Boylece 1. oturumdaki "h[0]=27 patlamasi" yapisal olarak imkansiz.
- Sanity testler gectik:
  * Parametre listesi (W, U, b_z, b_h, zeta_raw, nu_raw + classifier)
  * Forward (B, T, D) -> (B, num_classes)
  * Backward zincir: tum parametreler gradyan aliyor (zeta_raw/nu_raw dahil)
  * Random loss ~log(6) = 1.79 (init makul)
  * Kararlilik testi: 128 adim, |h| (0.77-1.66) arasinda sikismis

### Hafta 4: Ablasyon (capraz sweep)
4 config'i karsilastirdik:

| Model               | Params | Acc    | Macro-F1 | vs MLP   |
|---------------------|-------:|-------:|---------:|---------:|
| MLP (baseline)      | 12518  | 0.8547 | 0.8473   | baseline |
| FastGRNN H=16 e=30  |   440  | 0.8544 | 0.8465   | ~0       |
| FastGRNN H=16 e=50  |   440  | 0.8691 | 0.8605   | +1.4/+1.3|
| FastGRNN H=32 e=30  |  1384  | 0.8450 | 0.8332   | -1.0/-1.4|
| FastGRNN H=32 e=50  |  1384  | 0.8335 | 0.8297   | -2.1/-1.8|

Capraz efekt analizi:
- Epoch artisi @ H=16: +1.4 F1 (yardim)
- Epoch artisi @ H=32: -0.4 F1 (zarar — overfit)
- H artisi @ e=30: -1.3 F1 (zarar)
- H artisi @ e=50: -3.1 F1 (daha cok zarar)
- Bulgu: H=16 dogru kapasite. Daha buyuk model bu gorev icin asiri.

### Hafta 4 ek: Epoch doyum egrisi (H=16, e=120)
- En iyi val_f1: 0.9135 @ epoch 119
- En iyi test_f1 (izleme): 0.9119 @ epoch 113
- Doyum noktasi: epoch 106 (peak %99'una ilk ulaşma)
- Resmi (val-secili) test_f1: 0.9047
- Plato YOK — model salinarak yukari gidiyor, varyans yuksek (tek epoch'ta 0.25 F1 dususu mumkun)
- Cikari: H=16'da pratik tatli nokta e100-120 + val_f1 secimi

### Final tablo

| Model                          | test F1 | vs MLP |
|--------------------------------|--------:|-------:|
| MLP                            | 0.847   | -      |
| FastGRNN e30                   | 0.847   | ~0     |
| FastGRNN e50                   | 0.861   | +1.4   |
| FastGRNN e120 (val-secili)     | **0.905** | **+5.8** |

Sonuc: FastGRNN MLP'yi 28x daha az parametreyle hem accuracy hem macro-F1'de gecti.

### Per-class kazanim
- UPSTAIRS F1: 0.731 (MLP) -> 0.789 (FastGRNN H=16 e=50). +5.8 puan.
- WALKING F1: 0.843 -> 0.872. +2.9 puan.
- DOWNSTAIRS: MLP hala lider (0.877 vs 0.809). Yuksek frekansli darbe imzasini
  flatten-MLP yakaliyor; bu paper'da tartisilmayan bir nuans, raporda yer alacak.
- SITTING/STANDING: marjinal iyilesme.
- LAYING: %100 idi, hafif geriledi (~0.99); onemsiz.

### REPRODUCTION RAPORUNA EKLENECEK BULGULAR
1. zeta=nu=0.5 elle koyuldugunda ve egitilmemis modelde gercek ivme dizisi (gravity bias)
   beslendiginde h hızla patlıyor — paper'in zeta/nu kısıtınının NIYE gerekli oldugunu
   deneysel olarak gosterir. (Oturum 1'de tespit edildi.)
2. Egitilen modelde zeta ve nu baslangic noktasindan (0.5/0.5) cok az kaydi
   (final ~0.51/0.50). Baslangic noktasinin yakin-optimal oldugu gozlemi.
3. Vanilla FastGRNN icin H=16 yeterli kapasite; H=32 her durumda zarar verdi.
   "Daha buyuk model her zaman daha iyi" mitinin tersi.
4. 120 epoch sonrasi bile model plato yapmadi, sadece dalga genligi azalmadi —
   val_f1 tabanli model secimi varyansi tolere ediyor.
5. DOWNSTAIRS sinifinda MLP hala lider; RNN'in yuksek frekansli dikey impulse
   yakalamada flatten-MLP'den geri olmasi paper'da tartisilmayan bulgu.

### Gemini'den onemli katki (Hafta 5'e tasimak icin)
Paper'in tam egitim pipeline'i 3 asamali (L-S-Q):
- L: Low-rank fazi    — ~100 epoch
- S: Sparsity fazi    — ~100 epoch + early stopping
- Q: Quantization fazi — ~100 epoch + early stopping
Toplam 300 epoch, ama early stopping ile pratik daha kisa.
Hafta 5-6-7 bu uc faza karsilik gelecek.

### Sonraki adim: Hafta 5 — Low-rank faktorizasyon
- W'yi W1 @ W2.T olarak yeniden yaz (H x r_w, D x r_w)
- U'yu U1 @ U2.T olarak yeniden yaz (H x r_u, H x r_u)
- Baslangic: r_w=2 veya 3, r_u=4 veya 6
- Egitim: 100 epoch + early stopping
- Hedef: parametre sayisini daha da dusur, accuracy 0.90 civarinda kalsin

### Proje dosyalari (oturum sonu)
fastgrnn-har/
├── fastgrnn_numpy.py            # Saf NumPy referans
├── fastgrnn_model.py            # PyTorch FastGRNNCell + FastGRNNClassifier
├── train_mlp_baseline.py
├── train_fastgrnn.py            # argparse --hidden --epochs --lr
├── epoch_saturation.py          # 120 epoch doyum deneyi
├── compare_results.py           # Capraz karsilastirma tablosu
├── build_dataset.py             # HAPT pencereleme
├── download_hapt.py             # HAPT indirici
├── explore_hapt.py              # Veri gorsellestirme
├── experiments/
│   ├── mlp_baseline.json
│   ├── fastgrnn_h16_e30.json
│   ├── fastgrnn_h16_e50.json
│   ├── fastgrnn_h32_e30.json
│   ├── fastgrnn_h32_e50.json
│   └── saturation_h16.json
├── saturation_curve.png
├── *_best.pt                    # Egitilmis model checkpoints
└── notes/
    ├── memory_oturum3.md
    ├── memory_hafta2_3_4.md     # bu dosya
    └── gunluk_rapor_2026-05-23.md
