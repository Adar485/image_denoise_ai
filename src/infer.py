"""
infer.py

Amaç:
- checkpoints/best_model.pth dosyasını yüklemek
- data/val/noisy içindeki bir görüntüyü alıp modele vermek
- Temizlenmiş çıktıyı outputs/ klasörüne kaydetmek
"""

import os
from glob import glob

import cv2
import numpy as np
import torch

from model import DenoiseCNN
from dataset import BASE_DIR, VAL_NOISY_DIR, IMG_HEIGHT, IMG_WIDTH


OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "best_model.pth")


def load_model(device):
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint bulunamadı: {CHECKPOINT_PATH}")

    print("Checkpoint yükleniyor:", CHECKPOINT_PATH)

    model = DenoiseCNN(num_features=64, num_layers=8).to(device)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(f"Checkpoint epoch: {ckpt.get('epoch')}, val_loss: {ckpt.get('val_loss')}")
    return model


def load_noisy_image():
    # val/noisy içinden ilk uygun resmi seç
    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        image_paths.extend(glob(os.path.join(VAL_NOISY_DIR, ext)))

    image_paths = sorted(image_paths)

    if len(image_paths) == 0:
        raise RuntimeError(f"{VAL_NOISY_DIR} içinde noisy görüntü bulunamadı.")

    img_path = image_paths[0]
    print("Kullanılacak noisy görüntü:", img_path)

    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        raise RuntimeError(f"Görüntü okunamadı: {img_path}")

    return img_bgr, img_path


def preprocess(img_bgr):
    """
    BGR (uint8, HxWx3) -> tensor (1, 3, IMG_HEIGHT, IMG_WIDTH), [0,1] aralığı
    """
    # BGR -> RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Eğitimde kullandığımız boyuta resize
    img_rgb = cv2.resize(
        img_rgb, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA
    )

    img = img_rgb.astype(np.float32) / 255.0  # [0,1]
    img = np.transpose(img, (2, 0, 1))        # C,H,W
    tensor = torch.from_numpy(img).unsqueeze(0)  # 1,C,H,W

    return tensor  # shape: (1, 3, H, W)


def postprocess(tensor):
    """
    tensor (1, 3, H, W), [0,1] -> BGR uint8 (HxWx3)
    """
    img = tensor.squeeze(0).cpu().numpy()    # 3,H,W
    img = np.transpose(img, (1, 2, 0))       # H,W,3
    img = np.clip(img, 0.0, 1.0)
    img = (img * 255.0).astype(np.uint8)

    # RGB -> BGR (cv2.imwrite için)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img_bgr


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Kullanılan device:", device)

    model = load_model(device)

    # Noisy resmi yükle
    noisy_bgr, noisy_path = load_noisy_image()
    noisy_tensor = preprocess(noisy_bgr).to(device)

    with torch.no_grad():
        denoised_tensor = model(noisy_tensor)

    denoised_bgr = postprocess(denoised_tensor)

    # Çıktı dosya adı
    base_name = os.path.splitext(os.path.basename(noisy_path))[0]
    out_path = os.path.join(OUTPUT_DIR, f"denoised_{base_name}.png")

    cv2.imwrite(out_path, denoised_bgr)
    print("Temizlenmiş görüntü kaydedildi:", out_path)


if __name__ == "__main__":
    main()
