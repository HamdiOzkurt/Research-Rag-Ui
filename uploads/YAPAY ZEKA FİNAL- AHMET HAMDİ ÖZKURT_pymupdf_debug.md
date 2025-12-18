![](uploads/YAPAY ZEKA FİNAL- AHMET HAMDİ ÖZKURT_images/YAPAY-ZEKA-FİNAL--AHMET-HAMDİ-ÖZKURT.pdf-0-0.png)
##### **6 ŞUBAT DEPREMLERİ SONRASI YORUMLARDAN KAMUOYU** **TEPKİLERİNİN** **TEMATİK VE ZAMANSAL DUYGU ANALİZİ**

Ahmet Hamdi ÖZKURT

B211306053


Özkurt, A. H., Aydemir, E., & Sönmez, Y. (2025). 6 Şubat depremleri sonrası yorumlardan kamuoyu tepkilerinin tematik ve zamansal duygu analizi. IV.
[Bilişim Festivali (IFEST 2025) Uluslararası Bilişim Kongresi (IIC 2025), 16–17 Mayıs, Batman, Türkiye. https://earsiv.batman.edu.tr/items/9b87ffdc-48cd-](https://earsiv.batman.edu.tr/items/9b87ffdc-48cd-455e-93c4-0f523b2eef72)

[455e-93c4-0f523b2eef72](https://earsiv.batman.edu.tr/items/9b87ffdc-48cd-455e-93c4-0f523b2eef72)


![](uploads/YAPAY ZEKA FİNAL- AHMET HAMDİ ÖZKURT_images/YAPAY-ZEKA-FİNAL--AHMET-HAMDİ-ÖZKURT.pdf-1-0.png)
# **AMAÇ**

###### Bu çalışmanın amacı, 6 Şubat depremleri sonrasında Ekşi Sözlük’ te oluşan kullanıcı yorumları üzerinden kamuoyu tepkilerinin tematik ve duygusal özelliklerini makine öğrenmesi ve doğal dil işleme yöntemleriyle analiz ederek zamansal değişimini incelemektir.


## **YÖNTEM**



![](uploads/YAPAY ZEKA FİNAL- AHMET HAMDİ ÖZKURT_images/YAPAY-ZEKA-FİNAL--AHMET-HAMDİ-ÖZKURT.pdf-3-0.png)
#### **KULLANILAN MAKİNE ÖĞRENMESİ** **ALGORİTMALARI- 1**

![](uploads/YAPAY ZEKA FİNAL- AHMET HAMDİ ÖZKURT_images/YAPAY-ZEKA-FİNAL--AHMET-HAMDİ-ÖZKURT.pdf-4-0.png)

##### Naive Bayes Bayes teoremine dayanan olasılıksal bir

sınıflandırma algoritmasıdır. Özelliklerin birbirinden
bağımsız olduğu varsayımıyla çalışır. Metin
madenciliği ve belge sınıflandırma problemlerinde
hızlı ve etkili sonuçlar vermesi nedeniyle yaygın
olarak tercih edilir.


#### **KULLANILAN MAKİNE ÖĞRENMESİ** **ALGORİTMALARI- 2**

![](uploads/YAPAY ZEKA FİNAL- AHMET HAMDİ ÖZKURT_images/YAPAY-ZEKA-FİNAL--AHMET-HAMDİ-ÖZKURT.pdf-5-0.png)

##### **Logistic Regression**

Denetimli öğrenmeye dayalı bir sınıflandırma
algoritmasıdır. Girdi özelliklerinin doğrusal
kombinasyonunu kullanarak bir örneğin belirli bir
sınıfa ait olma olasılığını hesaplar. Genellikle ikili
sınıflandırma problemlerinde kullanılır.


#### **KULLANILAN MAKİNE ÖĞRENMESİ** **ALGORİTMALARI- 3**

![](uploads/YAPAY ZEKA FİNAL- AHMET HAMDİ ÖZKURT_images/YAPAY-ZEKA-FİNAL--AHMET-HAMDİ-ÖZKURT.pdf-6-0.png)

##### **Ridge Classifier**

Doğrusal sınıflandırma problemleri için kullanılan,
L2 düzenlileştirme içeren bir algoritmadır.
Ağırlıkların büyüklüğünü sınırlandırarak aşırı
öğrenmeyi azaltır ve çoklu doğrusal bağlantının
bulunduğu veri setlerinde daha kararlı sonuçlar
üretir.


**L2 düzenlileştirmesi:** Modelin öğrendiği ağırlıkların çok büyümesini engelleyen bir tekniktir. Böylece model veriyi ezberlemek yerine genelleme yapar ve aşırı
öğrenmenin önüne geçilir.


### **BULGULAR**

_**Karmaşıklık Matrisi**_


Bu karmaşıklık matrisleri, üç farklı duygu analizi
modelinin ( **Logistic Regression, Naive Bayes,**
**Ridge Classifier** ) dört farklı metin kategorisindeki
performansını göstermektedir.


Matrisler, her bir modelin doğru ve yanlış
tahminlerini göstererek, hangi kategorilerde daha
başarılı olduğunu ve hangi tür hatalar yaptığını
anlamamızı sağlar.



![](uploads/YAPAY ZEKA FİNAL- AHMET HAMDİ ÖZKURT_images/YAPAY-ZEKA-FİNAL--AHMET-HAMDİ-ÖZKURT.pdf-7-0.png)
### **BULGULAR**

_**Kategori Bazlı Duygu Analizi**_


Bu pasta grafikleri, dört farklı metin
kategorisindeki duygu dağılımını göstermektedir.
Her bir daire, ilgili kategorideki pozitif ve negatif
duygu oranlarını temsil eder.


Grafiklerden genel olarak, **negatif** duyguların tüm
kategorilerde baskın olduğu görülmektedir.



![](uploads/YAPAY ZEKA FİNAL- AHMET HAMDİ ÖZKURT_images/YAPAY-ZEKA-FİNAL--AHMET-HAMDİ-ÖZKURT.pdf-8-0.png)
### **BULGULAR**

_**Kategori Bazında Zamansal Değişim**_


Bu çizgi grafikleri, dört farklı metin kategorisindeki
aylık pozitif ve negatif duygu sayılarının zaman
içindeki değişimini göstermektedir.


Logaritmik ölçek kullanılarak, büyük sayı farklılıkları
daha kolay görülebilmektedir.


Genel olarak, negatif duygu sayılarının pozitif
duygu sayılarından daha yüksek olduğu ve zaman
içinde dalgalanmalar gösterdiği görülmektedir.



![](uploads/YAPAY ZEKA FİNAL- AHMET HAMDİ ÖZKURT_images/YAPAY-ZEKA-FİNAL--AHMET-HAMDİ-ÖZKURT.pdf-9-0.png)
## **SONUÇ**



![](uploads/YAPAY ZEKA FİNAL- AHMET HAMDİ ÖZKURT_images/YAPAY-ZEKA-FİNAL--AHMET-HAMDİ-ÖZKURT.pdf-10-0.png)
![](uploads/YAPAY ZEKA FİNAL- AHMET HAMDİ ÖZKURT_images/YAPAY-ZEKA-FİNAL--AHMET-HAMDİ-ÖZKURT.pdf-11-full.png)
