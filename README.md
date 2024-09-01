# Vector Search Study Case

Bu proje, hem görsel hem de metin tabanlı sorgularla resim araması yapmayı amaçlamaktadır. Proje, önceden etiketlenmemiş resimlerin vektörlerini CLIP modeli kullanarak oluşturur ve bunları bir vektör veri tabanına kaydeder daha sonra da bu vektörler üzerinden benzer resimleri veya betimlenen içeriği aratmanıza olanak tanır. Sonuçlar bir Gradio ile web arayüzü üzerinden gösterilir ve kullanılır.

## 📑 İçindekiler
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Proje Yapısı](#proje-yapısı)
- [Gereksinimler](#gereksinimler)
- [Sequence Diyagramı](#sequence-diyagramı)
- [Milvus ve Gradio Hakkında](#milvus-ve-gradio-hakkında)

## ⚙️ Kurulum

Proje için gerekli bağımlılıkları yüklemek ve ortamı kurmak için aşağıdaki adımları izleyin.

### Adım 1: Gerekli Kütüphaneleri Yükleyin

Proje Python ile geliştirilmiştir. Gerekli Python kütüphanelerini aşağıdaki komutla yükleyebilirsiniz:

```bash
pip install sentence-transformers pymilvus gradio pillow numpy matplotlib
```
## 🚀 Kullanım
Proje, hem metin tabanlı hem de görsel tabanlı arama yapmanıza olanak tanır.

### Görsel Tabanlı Arama:

1. Görselin dosya yolunu metin kutusuna girin.
2. `Top K` değerini seçin.
3. "Search" düğmesine tıklayın.

### Metin Tabanlı Arama:

1. Aramak istediğiniz metni metin kutusuna girin.
2. `Top K` değerini seçin.
3. "Search" düğmesine tıklayın.

Sonuçlar, sorgunuza en yakın olan görsellerin bir galerisi olarak gösterilecektir.

## 🗂️ Proje Yapısı

- **program.py**: Ana kod dosyasıdır.
- **sequenceDiagram.jpg**: projenin akışını gösterir.

## 🛠️ Gereksinimler

Bu proje aşağıdaki gereksinimlere cevap vermektedir:

1. **Veri Toplama:** Bu adımda önceden hazırlanmış veriler üzerine ekleme yapıldı ve o veriler kullanıldı.
2. **Özellik Çıkarımı:** Görüntüler üzerinden CLIP modeli kullanılarak özellik çıkarımı yapılmıştır. (https://openai.com/index/clip/)
3. **Vektör Veri Tabanlarına Aktarım:** Elde edilen vektörler, Milvus veritabanına aktarılmıştır.
4. **Resimlere Sorgu:** Vektör veri tabanında olan görselleri sorgulamak için metin ve görsel tabanlı sorgular yapılabilmektedir.

## 📊 Sequence Diyagramı

Aşağıdaki diyagram, bu projenin adımlarını ve işleyişini gösterir:

![Sequence Diagram](https://github.com/mrkoglu/vectorSearch/blob/main/sequenceDiagram.jpg)


## 🧰 Milvus ve Gradio Hakkında

### Milvus

Milvus, vektör benzerliği arama için optimize edilmiş açık kaynaklı bir vektör veritabanıdır. Yüksek performanslı ve hızlı olduğu için tercih edildi.

### Gradio

Gradio,makine öğrenimi modellerini ve verilerini web arayüzü ile gösterimini ve kullanımını sağlayan bir Python kütüphanesidir.

## Aşağıda örnek çıktılar gösterilmiştir:

![Outputs](https://github.com/mrkoglu/vectorSearch/blob/main/samples.png)
