"""
model.py

Basit bir CNN tabanlı görüntü gürültü giderme (denoising) modeli.

Girdi:  (N, 3, H, W)  - noisy görüntü
Çıktı:  (N, 3, H, W)  - tahmini temiz görüntü

Mimari:
- 3 -> 64 konvolüsyon + ReLU
- Birkaç adet 64 -> 64 Conv + BatchNorm + ReLU bloğu
- Son katman 64 -> 3 Conv (çıktı yine 3 kanal RGB)

Not:
- Kernel boyutu 3, padding=1 kullanıyoruz ki H, W sabit kalsın.
"""

import sys
import torch
import torch.nn as nn


class DenoiseCNN(nn.Module):
    def __init__(self, num_features: int = 64, num_layers: int = 8):
        """
        num_features: ara katmanlardaki kanal sayısı (ör: 64)
        num_layers:   toplam konvolüsyon katman sayısı (başlangıç + ara + çıkış)
                      burada: 1 giriş + (num_layers-2) ara + 1 çıkış
        """
        super().__init__()

        layers = []

        # 1) Giriş katmanı: 3 -> num_features
        layers.append(
            nn.Conv2d(
                in_channels=3,
                out_channels=num_features,
                kernel_size=3,
                padding=1,
                bias=True,
            )
        )
        layers.append(nn.ReLU(inplace=True))

        # 2) Ara katmanlar: num_features -> num_features
        # Her birinde Conv + BatchNorm + ReLU kullanıyoruz
        for _ in range(num_layers - 2):
            layers.append(
                nn.Conv2d(
                    in_channels=num_features,
                    out_channels=num_features,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                )
            )
            layers.append(nn.BatchNorm2d(num_features))
            layers.append(nn.ReLU(inplace=True))

        # 3) Çıkış katmanı: num_features -> 3
        layers.append(
            nn.Conv2d(
                in_channels=num_features,
                out_channels=3,
                kernel_size=3,
                padding=1,
                bias=True,
            )
        )

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (N, 3, H, W) noisy görüntü
        return: (N, 3, H, W) tahmini temiz görüntü
        """
        out = self.net(x)
        # İleride istersen burayı residual yapıya çevirebiliriz:
        #   noise_pred = self.net(x)
        #   clean = x - noise_pred
        # Şimdilik direkt temiz görüntüyü üretmeye çalışıyoruz.
        return out


def _test_model():
    """
    Hızlı model testi:
    - Hangi Python exe kullanılıyor gösterir
    - torch versiyonu ve CUDA durumunu yazar
    - DenoiseCNN örneği oluşturur
    - Sahte bir girdi ile forward geçirir
    - Çıktı shape'ini ve device bilgisini yazar
    """
    print("Python EXE:", sys.executable)
    print("torch version:", torch.__version__)
    print("cuda is_available:", torch.cuda.is_available())
    print("cuda device count:", torch.cuda.device_count())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Seçilen device:", device)

    model = DenoiseCNN(num_features=64, num_layers=8).to(device)
    print("\n=== Model Özeti ===")
    print(model)

    # Sahte bir batch: 4 adet 3x256x256 noisy görüntü
    dummy_input = torch.randn(4, 3, 256, 256, device=device)
    with torch.no_grad():
        output = model(dummy_input)

    print("\n=== Tensor Bilgisi ===")
    print("Girdi shape :", dummy_input.shape)
    print("Çıktı shape :", output.shape)


if __name__ == "__main__":
    _test_model()
