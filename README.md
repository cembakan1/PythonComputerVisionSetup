# 🍰 Desert Image Classification with PyTorch

Bu proje, PyTorch kullanılarak geliştirilmiş bir bilgisayarlı görü (Computer Vision) ve görüntü sınıflandırma modelidir. Model, dört farklı tatlı kategorisini (Baklava, Cannoli, Cupcakes, Donuts) ayırt etmek üzere eğitilmiştir.

## 📁 Proje Yapısı

Proje, kodun modülerliğini ve okunabilirliğini artırmak için endüstri standartlarına uygun olarak parçalara ayrılmıştır:

* `setup_data.py`: Klasör hiyerarşisinin kurulması ve PyTorch `DataLoader` objelerinin oluşturulması işlemlerini yönetir.
* `model_creation.py`: Sinir ağı (Neural Network) mimarisinin tanımlandığı dosyadır.
* `engine.py`: Modelin eğitim (train) ve test (test/evaluation) döngülerini içeren temel motor fonksiyonlarını barındırır.
* `utils.py`: Model kaydetme, görselleştirme gibi yardımcı fonksiyonları içerir.
* `main.py`: Eğitim sürecini başlatan ve yöneten ana çalıştırılabilir dosyadır.
* `load_model_make_prediction.py`: Eğitilmiş model ağırlıklarını (örn. `.pth` dosyaları) yükleyerek yeni ve görülmemiş görseller üzerinde tahmin (inference) yapmak için kullanılır.

## 📊 Veri Seti (Desert101)

Modelin eğitimi için `desert101` veri seti kullanılmıştır. Veri seti toplamda 4 sınıftan oluşmaktadır:
1. `baklava`
2. `cannoli`
3. `cup_cakes`
4. `donuts`

# 🚀 Kurulum ve Çalıştırma


## Gereksinimler

Projeyi yerel bilgisayarınızda çalıştırmak için öncelikle gerekli kütüphaneleri yükleyin.
Python sanal ortamı (virtual environment) kullanmanız tavsiye edilir.

## Bash
pip install -r requirements.txt

## Modeli Eğitmek
Kendi modelinizi sıfırdan eğitmek isterseniz ana scripti çalıştırabilirsiniz:

## Bash
python main.py

## Tahmin Yapmak (Inference)
Eğitilmiş bir modeli kullanarak dışarıdan verilen bir görselin (örneğin internetten indirilmiş bir baklava fotoğrafı) hangi sınıfa ait olduğunu tahmin etmek için:

## Bash
python load_model_make_prediction.py
(Not: Tahmin edilecek görselin dosya yolunu script içerisinde güncellemeyi unutmayın.)

## 🧠 Modeller
Eğitim sonucunda elde edilen ağırlıklar models/ klasörü altına kaydedilmektedir.

desert_classifier.pth
enhanced_desert_classifier.pth

# 👨‍💻 Geliştirici
Cem BAKAN
