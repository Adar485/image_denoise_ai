# Image Denoise AI

Bu proje, **derin öğrenme tabanlı bir görüntü gürültü giderme (image denoising)** uygulamasıdır.  
Amaç, gürültülü (noisy) RGB görüntüleri alıp daha temiz (denoised) versiyonlarını üretmektir.

- Framework: **PyTorch**
- Arayüz: **Gradio**
- Model: Basit CNN tabanlı denoiser (DenoiseCNN)
- Çalışma şekli:
  - Gürültülü/temiz görüntü çiftlerinden oluşan bir veri seti ile model eğitilir
  - En iyi model `checkpoints/best_model.pth` olarak kaydedilir
  - Gradio arayüzü ile kullanıcı bir görüntü yükler, model temiz halini üretir

---

## Proje Yapısı

```text
image_denoise_ai/
├─ src/
│  ├─ dataset.py      # PyTorch Dataset / DataLoader
│  ├─ model.py        # DenoiseCNN model tanımı
│  ├─ train.py        # Eğitim döngüsü (train + val + checkpoint)
│  ├─ infer.py        # Tek görüntü üzerinde inference (dosyaya kaydetme)
│  └─ ui.py           # Gradio arayüzü
├─ data/              # (Git'e dahil değil) train/val clean/noisy görüntüler
├─ checkpoints/       # (Git'e dahil değil) eğitilmiş modeller (.pth)
├─ outputs/           # (Git'e dahil değil) inference çıktıları
└─ .gitignore
