"""
dataset.py

Görev:
- data/train/clean, data/train/noisy
- data/val/clean,   data/val/noisy
  klasörlerindeki görüntüleri okuyup
  PyTorch Dataset formatında (noisy, clean) çifti olarak döndürmek.

Bu dosyada:
- DenoiseImageDataset sınıfını tanımlıyoruz
- Test amaçlı: dosya direkt çalıştırıldığında birkaç örneğin shape'ini yazdırıyoruz
"""

import os
from glob import glob

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# Proje kök dizini (image_denoise_ai)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

TRAIN_CLEAN_DIR = os.path.join(DATA_DIR, "train", "clean")
TRAIN_NOISY_DIR = os.path.join(DATA_DIR, "train", "noisy")
VAL_CLEAN_DIR = os.path.join(DATA_DIR, "val", "clean")
VAL_NOISY_DIR = os.path.join(DATA_DIR, "val", "noisy")

IMG_HEIGHT = 256
IMG_WIDTH = 256


class DenoiseImageDataset(Dataset):
    def __init__(self, split: str = "train"):
        """
        split: 'train' veya 'val'
        """
        assert split in ("train", "val"), "split 'train' veya 'val' olmalı"

        if split == "train":
            clean_dir = TRAIN_CLEAN_DIR
            noisy_dir = TRAIN_NOISY_DIR
        else:
            clean_dir = VAL_CLEAN_DIR
            noisy_dir = VAL_NOISY_DIR

        # Temiz ve noisy yol listelerini al
        clean_paths = []
        noisy_paths = []

        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            clean_paths.extend(glob(os.path.join(clean_dir, ext)))
            noisy_paths.extend(glob(os.path.join(noisy_dir, ext)))

        # Sıralayalım ki isimler eşleşsin
        clean_paths = sorted(clean_paths)
        noisy_paths = sorted(noisy_paths)

        if len(clean_paths) == 0:
            raise RuntimeError(f"{clean_dir} içinde temiz görüntü bulunamadı.")
        if len(clean_paths) != len(noisy_paths):
            raise RuntimeError(
                f"clean ({len(clean_paths)}) ve noisy ({len(noisy_paths)}) sayıları eşit değil!"
            )

        self.clean_paths = clean_paths
        self.noisy_paths = noisy_paths

        print(f"[DenoiseImageDataset] split={split}, toplam örnek: {len(self.clean_paths)}")

    def __len__(self):
        return len(self.clean_paths)

    def __getitem__(self, idx):
        clean_path = self.clean_paths[idx]
        noisy_path = self.noisy_paths[idx]

        # Görüntüleri BGR olarak okur (cv2)
        clean_img = cv2.imread(clean_path)
        noisy_img = cv2.imread(noisy_path)

        if clean_img is None:
            raise RuntimeError(f"Temiz görüntü okunamadı: {clean_path}")
        if noisy_img is None:
            raise RuntimeError(f"Noisy görüntü okunamadı: {noisy_path}")

        # BGR -> RGB
        clean_img = cv2.cvtColor(clean_img, cv2.COLOR_BGR2RGB)
        noisy_img = cv2.cvtColor(noisy_img, cv2.COLOR_BGR2RGB)

        # TÜM görüntüleri sabit boyuta getir (256x256)
        clean_img = cv2.resize(clean_img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)
        noisy_img = cv2.resize(noisy_img, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_AREA)

        # uint8 [0,255] -> float32 [0,1]
        clean_img = clean_img.astype(np.float32) / 255.0
        noisy_img = noisy_img.astype(np.float32) / 255.0

        # H x W x C -> C x H x W (PyTorch formatı)
        clean_img = np.transpose(clean_img, (2, 0, 1))
        noisy_img = np.transpose(noisy_img, (2, 0, 1))

        # numpy -> torch tensor
        clean_tensor = torch.from_numpy(clean_img)  # shape: (3, 256, 256)
        noisy_tensor = torch.from_numpy(noisy_img)  # shape: (3, 256, 256)

        return noisy_tensor, clean_tensor



def _test_dataset():
    """
    Hızlı test:
    - Train dataset'ini oluşturur
    - Birkaç örneğin shape'ini ve min/max değerlerini yazar
    - DataLoader ile bir batch çeker
    """
    print("=== Train dataset testi ===")
    train_dataset = DenoiseImageDataset(split="train")
    print("Train örnek sayısı:", len(train_dataset))

    noisy, clean = train_dataset[0]
    print("Tek örnek noisy shape:", noisy.shape)
    print("Tek örnek clean shape:", clean.shape)
    print("noisy min/max:", noisy.min().item(), noisy.max().item())
    print("clean min/max:", clean.min().item(), clean.max().item())

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    batch_noisy, batch_clean = next(iter(train_loader))
    print("Batch noisy shape:", batch_noisy.shape)
    print("Batch clean shape:", batch_clean.shape)

    print("=== Val dataset testi ===")
    val_dataset = DenoiseImageDataset(split="val")
    print("Val örnek sayısı:", len(val_dataset))


if __name__ == "__main__":
    _test_dataset()
