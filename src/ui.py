"""
ui.py

Amaç:
- checkpoints/best_model.pth modelini yüklemek
- Kullanıcının yüklediği gürültülü (noisy) görüntüyü modele verip
  temizlenmiş (denoised) çıktıyı ekranda göstermek.

Teknoloji: Gradio (basit web arayüzü)
"""

import os

import cv2
import numpy as np
import torch
import gradio as gr

from model import DenoiseCNN
from dataset import BASE_DIR, IMG_HEIGHT, IMG_WIDTH

# Checkpoint yolu
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "best_model.pth")


def load_model(device: torch.device) -> torch.nn.Module:
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint bulunamadı: {CHECKPOINT_PATH}")

    print("Checkpoint yükleniyor:", CHECKPOINT_PATH)

    model = DenoiseCNN(num_features=64, num_layers=8).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Checkpoint epoch: {ckpt.get('epoch')}, val_loss: {ckpt.get('val_loss')}")
    return model


def preprocess_rgb(img_rgb: np.ndarray) -> torch.Tensor:
    """
    Gradio'dan gelen RGB görüntü (HxWx3, [0-255]) -> tensor (1, 3, H, W), [0,1]
    """
    # Emin olalım tip float32 olsun
    if img_rgb.dtype != np.float32:
        img_rgb = img_rgb.astype(np.float32)

    # 0-255 -> 0-1
    img_rgb = img_rgb / 255.0

    # Eğitimde kullandığımız boyuta resize (IMG_HEIGHT x IMG_WIDTH)
    img_rgb = cv2.resize(
        img_rgb, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA
    )

    # H x W x C -> C x H x W
    img_chw = np.transpose(img_rgb, (2, 0, 1))

    # numpy -> torch, batch dimension ekle
    tensor = torch.from_numpy(img_chw).unsqueeze(0)  # (1, 3, H, W)
    return tensor


def postprocess_rgb(tensor: torch.Tensor) -> np.ndarray:
    """
    tensor (1, 3, H, W), [0,1] -> RGB uint8 (H, W, 3), [0-255]
    """
    img_chw = tensor.squeeze(0).detach().cpu().numpy()  # (3, H, W)
    img_hwc = np.transpose(img_chw, (1, 2, 0))          # (H, W, 3)

    img_hwc = np.clip(img_hwc, 0.0, 1.0)
    img_hwc = (img_hwc * 255.0).astype(np.uint8)

    return img_hwc  # RGB, Gradio bunu direkt gösterebilir


# Cihaz ve model global yükleyelim (her request’te tekrar yüklemeyelim)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("UI için kullanılan device:", DEVICE)
MODEL = load_model(DEVICE)


def denoise_image(noisy_img: np.ndarray) -> np.ndarray:
    """
    Gradio arayüzünden gelen noisy görüntüyü alır,
    modeli kullanarak temizlenmiş çıktıyı döndürür.
    """
    if noisy_img is None:
        return None

    # Preprocess
    noisy_tensor = preprocess_rgb(noisy_img).to(DEVICE)

    # İnfer
    with torch.no_grad():
        denoised_tensor = MODEL(noisy_tensor)

    # Postprocess
    denoised_img = postprocess_rgb(denoised_tensor)

    return denoised_img


def main():
    title = "Image Denoise AI"
    description = (
        "Gürültülü bir görüntü yükleyin, model temiz versiyonunu üretmeye çalışsın. "
        "Girdi: noisy RGB görüntü, Çıktı: denoised RGB görüntü."
    )

    demo = gr.Interface(
        fn=denoise_image,
        inputs=gr.Image(type="numpy", label="Noisy görüntü yükle"),
        outputs=gr.Image(type="numpy", label="Temizlenmiş görüntü"),
        title=title,
        description=description,
    )

    demo.launch()


if __name__ == "__main__":
    main()

