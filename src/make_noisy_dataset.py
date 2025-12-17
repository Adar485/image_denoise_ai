

import os
from glob import glob

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Proje kök dizini (image_denoise_ai)
"""
make_noisy_dataset.py

Görev (ileride):
- data/clean_raw klasöründeki temiz görüntüleri okuyacak
- Bunlara gürültü ekleyip:
    - data/train/clean
    - data/train/noisy
    - data/val/clean
    - data/val/noisy
  klasörlerine kaydedecek.

Bu adımda sadece:
- Gerekli paketleri import ediyoruz
- Klasör yollarını tanımlıyoruz
- clean_raw içindeki resimleri bulup sayıyoruz
- Bir örnek resmi okuyup boyutunu yazdırıyoruz
"""

import os
from glob import glob

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Proje kök dizini (image_denoise_ai)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# data klasörü
DATA_DIR = os.path.join(BASE_DIR, "data")

# Alt klasörler
CLEAN_RAW_DIR = os.path.join(DATA_DIR, "clean_raw")
TRAIN_CLEAN_DIR = os.path.join(DATA_DIR, "train", "clean")
TRAIN_NOISY_DIR = os.path.join(DATA_DIR, "train", "noisy")
VAL_CLEAN_DIR = os.path.join(DATA_DIR, "val", "clean")
VAL_NOISY_DIR = os.path.join(DATA_DIR, "val", "noisy")

def add_gaussian_noise(img, sigma=600):
    """
    img: uint8 (0-255) BGR görüntü (cv2.imread çıktısı)
    sigma: gürültü şiddeti (standart sapma)
    """
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def main():
    print("BASE_DIR      :", BASE_DIR)
    print("DATA_DIR      :", DATA_DIR)
    print("CLEAN_RAW_DIR :", CLEAN_RAW_DIR)
    print("-" * 50)

    # clean_raw içindeki tüm resimleri topla
    image_paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
        image_paths.extend(glob(os.path.join(CLEAN_RAW_DIR, ext)))

    image_paths = sorted(image_paths)
    print(f"clean_raw içindeki toplam resim sayısı: {len(image_paths)}")

    if len(image_paths) == 0:
        print("Uyarı: clean_raw klasöründe hiç resim yok.")
        return

    # Train/Val'e böl (örnek: %90 train, %10 val)
    train_paths, val_paths = train_test_split(
        image_paths, test_size=0.1, random_state=42
    )

    print(f"Train resim sayısı: {len(train_paths)}")
    print(f"Val   resim sayısı: {len(val_paths)}")

    # Klasörleri oluştur (yoksa)
    os.makedirs(TRAIN_CLEAN_DIR, exist_ok=True)
    os.makedirs(TRAIN_NOISY_DIR, exist_ok=True)
    os.makedirs(VAL_CLEAN_DIR, exist_ok=True)
    os.makedirs(VAL_NOISY_DIR, exist_ok=True)

    import shutil

    print("-" * 50)
    print("Train temiz ve noisy görüntüler üretiliyor...")

    for i, src_path in enumerate(train_paths, start=1):
        filename = os.path.basename(src_path)

        # Temiz görüntüyü oku
        img = cv2.imread(src_path)
        if img is None:
            print("❌ Okunamayan resim atlandı (train):", src_path)
            continue

        # Gürültülü versiyonunu üret
        noisy = add_gaussian_noise(img, sigma=50)

        # Çıktı yolları
        clean_out = os.path.join(TRAIN_CLEAN_DIR, filename)
        noisy_out = os.path.join(TRAIN_NOISY_DIR, filename)

        # Temizi ve noisy'yi kaydet
        cv2.imwrite(clean_out, img)
        cv2.imwrite(noisy_out, noisy)

        if i % 100 == 0 or i == len(train_paths):
            print(f"Train: {i}/{len(train_paths)} görüntü işlendi")

    print("Val temiz ve noisy görüntüler üretiliyor...")

    for i, src_path in enumerate(val_paths, start=1):
        filename = os.path.basename(src_path)

        img = cv2.imread(src_path)
        if img is None:
            print("❌ Okunamayan resim atlandı (val):", src_path)
            continue

        noisy = add_gaussian_noise(img, sigma=600)

        clean_out = os.path.join(VAL_CLEAN_DIR, filename)
        noisy_out = os.path.join(VAL_NOISY_DIR, filename)

        cv2.imwrite(clean_out, img)
        cv2.imwrite(noisy_out, noisy)

        if i % 50 == 0 or i == len(val_paths):
            print(f"Val: {i}/{len(val_paths)} görüntü işlendi")

    print("-" * 50)
    print("İşlem bitti. clean/noisy train ve val setleri hazır.")


if __name__ == "__main__":
    main()
