# 🧠 CIFAR-10 Görüntü Sınıflandırma

Yapay zeka destekli görüntü sınıflandırma uygulaması. ResNet-18 transfer learning modeli ile CIFAR-10 veri setindeki 10 farklı sınıfı tanıyabilir.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-green.svg)

## 📋 İçindekiler

- [Özellikler](#-özellikler)
- [Kurulum](#-kurulum)
- [Kullanım](#-kullanım)
- [Model Performansı](#-model-performansı)
- [Teknik Detaylar](#-teknik-detaylar)
- [Proje Yapısı](#-proje-yapısı)

## ✨ Özellikler

- ✅ **10 Sınıf Tanıma**: Uçak, Otomobil, Kuş, Kedi, Geyik, Köpek, Kurbağa, At, Gemi, Kamyon
- ✅ **Modern Web Arayüzü**: Gradient tasarım, glassmorphism efektleri
- ✅ **Güven Skorları**: Her tahmin için detaylı olasılık dağılımı
- ✅ **Interaktif Grafikler**: Plotly ile görselleştirme
- ✅ **Transfer Learning**: ImageNet üzerinde önceden eğitilmiş ResNet-18
- ✅ **Data Augmentation**: RandomFlip, Rotation, ColorJitter

## 🚀 Kurulum

### Gereksinimler

```bash
pip install torch torchvision streamlit plotly pillow scikit-learn
```

### Projeyi Çalıştırma

1. Depoyu klonlayın veya indirin
2. Bağımlılıkları yükleyin
3. Uygulamayı başlatın:

```bash
cd image_classifier
streamlit run app.py
```

Tarayıcınızda `http://localhost:8501` adresine gidin.

## 📖 Kullanım

### Web Uygulaması

1. **Görsel Yükle**: Sol panelden bir görsel seçin (JPG, PNG, JPEG)
2. **Tahmin Et**: "🔮 Tahmin Et" butonuna tıklayın
3. **Sonuçları İncele**: Sağ panelde tahmin sonucu ve olasılık grafiğini görün

### Model Eğitimi (Opsiyonel)

Modeli yeniden eğitmek için:

```bash
python train.py
```

### Model Değerlendirme

Model performansını test etmek için:

```bash
python evaluate.py
```

## 📊 Model Performansı

Model, CIFAR-10 test seti üzerinde değerlendirilmiştir.

### Genel Metrikler

| Metrik | Değer |
|--------|-------|
| Accuracy | ~85-90% |
| Precision (Macro) | ~85% |
| Recall (Macro) | ~85% |
| F1-Score (Macro) | ~85% |

> Not: Kesin değerler için `python evaluate.py` komutunu çalıştırın.

### Değerlendirme Metrikleri

- **Accuracy**: Doğru tahmin oranı
- **Precision**: Pozitif tahminlerin ne kadarı gerçekten pozitif
- **Recall**: Gerçek pozitiflerin ne kadarı yakalandı
- **F1-Score**: Precision ve Recall'un harmonik ortalaması

## 🔧 Teknik Detaylar

### Model Mimarisi

- **Backbone**: ResNet-18 (ImageNet pre-trained)
- **Son Katman**: Linear(512, 10)
- **Aktivasyon**: Softmax (çıkış)

### Veri Ön İşleme

```python
transforms.Compose([
    transforms.Resize((224, 224)),      # Boyutlandırma
    transforms.RandomHorizontalFlip(),  # Yatay çevirme
    transforms.RandomRotation(15),      # Döndürme
    transforms.ColorJitter(),           # Renk değişimi
    transforms.ToTensor(),              # Tensor'a çevirme
    transforms.Normalize(...)           # Normalizasyon
])
```

### Eğitim Parametreleri

- **Optimizer**: Adam
- **Learning Rate**: 0.001
- **Batch Size**: 32
- **Epochs**: 25
- **Loss Function**: CrossEntropyLoss

## 📁 Proje Yapısı

```
image_classifier/
├── app.py           # Streamlit web uygulaması
├── train.py         # Model eğitim scripti
├── evaluate.py      # Model değerlendirme scripti
├── model.pth        # Eğitilmiş model ağırlıkları
├── metrics.json     # Değerlendirme sonuçları (evaluate.py sonrası)
├── README.md        # Bu dosya
└── data/            # CIFAR-10 veri seti
    └── cifar-10-batches-py/
```

## 🛠️ Kullanılan Teknolojiler

| Teknoloji | Kullanım Alanı |
|-----------|----------------|
| PyTorch | Derin öğrenme framework |
| Streamlit | Web arayüzü |
| Plotly | İnteraktif grafikler |
| PIL | Görsel işleme |
| scikit-learn | Metrik hesaplama |

## 📚 Kaynaklar

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)

## 📄 Lisans

Bu proje eğitim amaçlı geliştirilmiştir.

---

<div align="center">
  <p>🎓 Yapay Zeka Görüntü Sınıflandırma Projesi</p>
  <p>Made with using PyTorch & Streamlit</p>
</div>
