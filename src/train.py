"""
train.py

Amaç:
- DenoiseImageDataset (train/val) üzerinden veri yüklemek
- DenoiseCNN modelini bu verilerle eğitmek
- Temel train/val loss çıktısını ekrana yazdırmak

Bu ilk versiyon:
- Basit MSE loss kullanıyor
- Birkaç epoch çalışacak şekilde ayarlı
- Checkpoint kaydetme ve gelişmiş özellikler sonraki adımlarda eklenecek
"""

import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import DenoiseImageDataset
from model import DenoiseCNN


# Proje kök dizini ve checkpoint klasörü
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        print("CUDA bulundu, GPU kullanılacak.")
        return torch.device("cuda")
    else:
        print("CUDA yok, CPU kullanılacak.")
        return torch.device("cpu")


def create_dataloaders(batch_size=4):
    print("Train ve Val dataset'leri oluşturuluyor...")

    train_dataset = DenoiseImageDataset(split="train")
    val_dataset = DenoiseImageDataset(split="val")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Windows'ta ilk etapta 0 güvenli
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch_idx):
    model.train()
    running_loss = 0.0
    num_batches = 0

    start_time = time.time()

    for step, (noisy, clean) in enumerate(train_loader, start=1):
        noisy = noisy.to(device)   # (B, 3, 256, 256)
        clean = clean.to(device)   # (B, 3, 256, 256)

        optimizer.zero_grad()

        output = model(noisy)      # tahmini temiz görüntü
        loss = criterion(output, clean)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

        if step % 50 == 0 or step == len(train_loader):
            avg_loss = running_loss / num_batches
            print(
                f"[Epoch {epoch_idx}] Step {step}/{len(train_loader)} "
                f"- Avg Train Loss: {avg_loss:.6f}"
            )

    epoch_time = time.time() - start_time
    avg_epoch_loss = running_loss / max(num_batches, 1)
    print(
        f"[Epoch {epoch_idx}] Train bitti - Ortalama Loss: {avg_epoch_loss:.6f} "
        f"- Süre: {epoch_time:.1f} sn"
    )

    return avg_epoch_loss


def validate(model, val_loader, criterion, device, epoch_idx):
    model.eval()
    running_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for noisy, clean in val_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            output = model(noisy)
            loss = criterion(output, clean)

            running_loss += loss.item()
            num_batches += 1

    avg_loss = running_loss / max(num_batches, 1)
    print(f"[Epoch {epoch_idx}] Val Loss: {avg_loss:.6f}")
    return avg_loss

def main():
    # Hiperparametreler
    batch_size = 4
    num_epochs = 5        # İlk test için küçük tutuyoruz
    learning_rate = 1e-3

    device = get_device()

    # Model, loss, optimizer
    model = DenoiseCNN(num_features=64, num_layers=8).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # DataLoader'lar
    train_loader, val_loader = create_dataloaders(batch_size=batch_size)

    print("Eğitim başlıyor...\n")

    best_val_loss = float("inf")
    best_epoch = -1
    best_ckpt_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")

    for epoch in range(1, num_epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        val_loss = validate(model, val_loader, criterion, device, epoch)

        # En iyi (en düşük) val loss için modeli kaydet
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                },
                best_ckpt_path,
            )
            print(
                f"🟢 Yeni en iyi model kaydedildi! Epoch {epoch}, Val Loss: {val_loss:.6f}"
            )

        print("-" * 60)

    print("Eğitim tamamlandı.")
    if best_epoch != -1:
        print(
            f"En iyi epoch: {best_epoch}, En iyi Val Loss: {best_val_loss:.6f}, "
            f"checkpoint: {best_ckpt_path}"
        )


if __name__ == "__main__":
    main()
