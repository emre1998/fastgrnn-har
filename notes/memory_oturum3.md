# FastGRNN Project — Konusma Kaydi / Hafiza

## Oturum 3 — 2026-05-23

### Yapilanlar (Hafta 1, Blok 1-3 tamamlandi)
- Ortam kuruldu: Python 3.13.1, venv, PyTorch 2.12 (CPU), numpy/pandas/sklearn/jupyter.
- contract.md ve README.md yazildi.
- fastgrnn_numpy.py TAM: sigmoid, fastgrnn_step (pre + z_t + h_tilde + zeta/nu),
  run_sequence.
- scratch_sigmoid.py, scratch_step.py, scratch_sequence.py — dogrulama testleri.

### REPRODUCTION RAPORUNA EKLENECEK BULGU
- Egitilmemis modelde, zeta=nu=0.5 sectigimizde 128-ornekli gercek ivme dizisi
  uzerinde hafiza buyumesi gozlemledik (h[0] ~27).
- Sebep matematiksel: katsayilar toplami `(zeta*(1-z)+nu) + z = 1 + (zeta+nu-1) + ... `
  zeta=nu=0.5 ile her z>0 icin toplam 1'i asar.
- Ek sebep: gercek ivmeolcer sabit DC bilesen iceriyor (yercekimi ~9.8 g), tek
  yone surekli bias.
- Paper'in zeta/nu kisitini (sigmoid arkasinda saklayip ogrenilen) NIYE koydugu
  boylece deneysel olarak dogrulandi:
  "Kararlilik, zeta/nu'nun ogrenilmis ve kisitlanmis olmasina bagli."
- Kucuk rastgele girdiyle (scratch_step.py T=500) kararlilik gozlemlemistik —
  yani: kararlilik kosulu girdi olcegine ve zeta/nu degerine birlikte bagli.
- Rapor bolumu: "FastGRNN'in Stable iddiasinin altinda yatan tasarim kararlari".

### Sonraki adim
- B: UCI HAPT verisini indir ve yukle. X: [windows, 128, 3], y: [windows].
- Subject-based train/test split (HAPT 21 train / 9 test subject ile gelir).
- Sonra Hafta 1 son adim: minik MLP baseline.
