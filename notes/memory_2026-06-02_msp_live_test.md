# FastGRNN HAR - 2 Haziran 2026 MSP430 Canli Sensor Testi

## Oturum Ozeti

Bu oturumda MSP-EXP430G2ET LaunchPad uzerindeki MSP430G2553 ile GY-521
MPU6050 sensorunu canli olarak calistirdik. Hedef, daha once gomulu test
vektorleriyle dogruladigimiz FastGRNN C inference kodunu gercek ivme verisiyle
50 Hz streaming modunda calistirmakti.

Oturum kolay ilerlemedi: once donanimsal I2C birimi (USCI_B0) hatti kilitledi,
sonra CCS eski firmware'i karta yuklemeye devam etti. Son olarak sensorun
isindigi fark edildi. Kablolama ve guc baglantilari kontrol edildikten sonra
isinma ortadan kalkti. Oturum sonunda MPU6050 canli veri uretir ve FastGRNN
aktivite tahmini yazdirir durumdaydi.

---

## 1. Kablolama

Kullanilan baglantilar:

| MSP-EXP430G2ET | MPU6050 GY-521 |
|---|---|
| `3V3` | `VCC` |
| `GND` | `GND` |
| `P1.6` | `SCL` |
| `P1.7` | `SDA` |
| `GND` | `AD0` |

`AD0 -> GND` baglantisi nedeniyle MPU6050 I2C adresi `0x68`.

P1.6 ayni zamanda LaunchPad varyantina gore kart ustundeki LED hattina bagli
olabilir. I2C saat hattini yuklememesi icin `P1.6/LED` jumper baglantisi kontrol
edilmeli ve gerekiyorsa ayrilmali. GY-521 uzerindeki pull-up direncleri yetersizse
SCL ve SDA hatlarindan 3.3V'a birer `4.7 kOhm` pull-up direncten yararlanilabilir.

---

## 2. Ilk Sorun: USCI_B0 I2C Kilitlenmesi

Ilk canli denemelerde sensor yazilimsal GPIO I2C ile cevap verdi:

```text
BUS HEALTH: SCL=1 SDA=1 (ikisi de 1 olmali)
SW Ping 0x68: ACK! (software I2C calisti - USCI'de hata var)
```

Bu, sensorun beslendigini, kablonun en azindan temel seviyede calistigini ve
`0x68` adresinin dogru oldugunu gosterdi.

Fakat donanimsal USCI_B0 yolu START asamasinda kilitlendi:

```text
Ping 0x68: START timeout, UCB0STAT=0x50
(UCBBUSY=1 UCSCLLOW=1 UCALIFG=0)
Initializing MPU6050... FAIL
```

`UCBBUSY=1` ve `UCSCLLOW=1`, donanimsal I2C yolunun bus'i mesgul ve SCL hattini
dusuk durumda gordugunu gosteriyordu. Reset denemelerinden birinde
`BUS HEALTH: SCL=1 SDA=0` da goruldu. Eski yazilimsal ping kodu SDA sabit
dusukken bunu yanlislikla ACK olarak yorumlayabildigi icin bu durum ayrica
duzeltildi.

### Uygulanan cozum

Canli demo icin calistigi logla kanitlanan GPIO tabanli I2C yolu kullanildi.
Yalnizca ping degil, tum MPU6050 islemleri GPIO I2C'ye tasindi:

- register yazma
- repeated START
- coklu byte okuma
- ACK/NACK uretimi
- idle-high bus kontrolu
- stuck-low durumda islemi reddetme

Firmware'e ayirt edici bir surum etiketi eklendi:

```text
Firmware: GPIO-I2C v2
```

---

## 3. Ikinci Sorun: CCS Eski Firmware'i Yukluyordu

Kod guncellendigi halde seri terminalde eski USCI hata mesajlari gorunmeye
devam etti. Bunun kablolama sorunu olmadigi, eski firmware'in calistigi
anlasildi. Guncel firmware'de bulunmayan su satirlar hala cikiyordu:

```text
software I2C calisti - USCI'de hata var
Ping 0x68: START timeout
Donanim kontrolu: J5 jumper
```

Gercek CCS workspace yolu bulundu:

```text
C:\Users\EMRE CAN\workspace_ccstheia\Msp430 Fastgrnn Project Experiment
```

Bu klasorde `main.cpp` eksikti. Buna karsilik `Debug/main.obj` ve eski `.out`
dosyasi duruyordu. CCS eski derleme ciktisini tekrar karta yukluyordu.

### Uygulanan cozum

Guncel `main.cpp` gercek CCS proje klasorune eklendi. Eski nesne dosyalari
silinerek temiz derleme yapildi ve firmware `DSLite` ile karta yuklendi.

Derleme ve yukleme sonucu:

```text
Flash/FRAM usage is 8362 bytes.
RAM usage is 348 bytes.
Running...
Success
```

Portlar:

| Port | Islev |
|---|---|
| `COM6` | MSP Application UART1, seri log |
| `COM7` | MSP Debug Interface |

---

## 4. Ucuncu Sorun: MPU6050 Isinmasi

Test sirasinda MPU6050 modulunun belirgin sekilde isindigi fark edildi.
Bu normal kabul edilmedi. MPU6050 dusuk guc tuketen bir sensordur; belirgin
isinma halinde enerji hemen kesilmeli ve baglantilar kontrol edilmelidir.

Kontrol edilen noktalar:

- `3V3 -> VCC`
- `GND -> GND`
- VCC ve GND'nin ters baglanmamis olmasi
- breadboard guc raylarinin yonu
- modül uzerindeki gercek pin etiketleri
- ayni breadboard satirinda kisa devre olusturan kablo bulunmamasi

Kablolama ve guc baglantilari kontrol edildikten sonra isinma ortadan kalkti.
Kesin fiziksel kok neden olcumle izole edilmedi; bu nedenle "su kablo kesin
hataliydi" sonucu cikarilmamali. Isinma tekrar ederse USB hemen cikarilmali ve
sensor enerjisiz birakilmali.

---

## 5. Canli Test Sonucu

Oturum sonunda MPU6050 canli ivme verisi uretmeye basladi. Paylasilan seri
logda en az `t=695s` ile `t=773s` arasinda kesintisiz veri goruldu.

Durağan konumlardan bir ornek:

```text
[t=703s] Activity: LAYING
Raw: 0.066 0.041 -1.032
Raw: 0.074 0.043 -1.032
Raw: 0.084 0.047 -1.029
```

Hareketli bir bolumden ornek:

```text
[t=767s] Activity: WALKING
Raw: -2.000 -1.218 0.838
Raw: -0.718 0.171 0.565
Raw: -0.766 -0.011 0.719
Raw: -1.206 -0.640 0.614
Raw: 0.880 1.157 0.409
```

Logda gorulen tahmin siniflari:

- `LAYING`
- `STANDING`
- `SITTING`
- `DOWNSTAIRS`
- `WALKING`
- `UPSTAIRS`

Durağan verilerde ivme vektorunun buyuklugu yaklasik `1g` civarinda. Hareketli
bolumlerde eksen degerleri belirgin sekilde degisiyor. Bazi degerler `-2.000`
ve `2.000` sinirina ulasiyor; bu, sensorun `+/-2g` araliginda hareket sirasinda
doyuma ulasabildigini gosteriyor.

Bu log, canli veri hattinin ve inference dongusunun calistigini kanitlar.
Ancak kontrollu bir ground-truth protokolu uygulanmadigi icin bu oturumdan
canli siniflandirma accuracy degeri cikarmiyoruz. Ozellikle aktivite gecislerinde
tahminlerin bir pencere gecikmesiyle degisebilecegi unutulmamali.

---

## 6. Dogrulama

Kaynak seviyesinde eklenen regresyon kontrolleri:

- canli firmware `GPIO-I2C v2` etiketi yazdiriyor
- MPU6050 register islemleri GPIO I2C yolunu kullaniyor
- scan fonksiyonu eski USCI START islemini baslatmiyor
- SDA veya SCL stuck-low ise yazilimsal I2C islemi reddediliyor
- eski USCI transaction surucusu firmware icinde tutulmuyor

Test sonucu:

```text
Ran 5 tests
OK
```

TI MSP430 derleyicisiyle temiz CCS build basarili. Yeni `.out` dosyasi
MSP430G2553 kartina basariyla yuklendi.

---

## 7. Sonuc ve Sonraki Adimlar

### Bu oturumda tamamlananlar

- MSP430G2553 ile MPU6050 canli I2C ile calisti.
- Gercek ivme verisi 50 Hz streaming dongusune girdi.
- FastGRNN canli aktivite tahminleri yazdirdi.
- USCI_B0 kilitlenmesine karsi GPIO I2C fallback tam surucuye donusturuldu.
- CCS stale `.out` problemi bulundu ve temiz derlemeyle giderildi.
- Sensor isinmasi kontrol sonrasinda ortadan kalkti.

### Sonraki deney icin onerilen kontrollu protokol

Her aktiviteyi ayri ayri en az 30-60 saniye uygula:

1. `LAYING`
2. `SITTING`
3. `STANDING`
4. `WALKING`
5. `UPSTAIRS`
6. `DOWNSTAIRS`

Her bolumde baslangic ve bitis zamanini not et. Boylece canli tahminler gercek
etiketlerle eslestirilip aktivite bazinda accuracy ve confusion matrix
hesaplanabilir. Hareketli testlerde `+/-2g` doyumu sik gorulurse MPU6050
araliginin `+/-4g` yapilmasi ayrica degerlendirilmeli.

