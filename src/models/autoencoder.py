"""경량 컨볼루션 오토인코더 — 재구성 오차 기반 이상탐지.

patch_knn(메인)과의 비교 실험용. 패치가 아니라 이미지 전체를 입력으로 받아
정상 이미지만으로 재구성을 학습하고, 재구성 오차(상위 k% 픽셀 평균)를 이상 스코어로 쓴다.
정렬 없이 자연스러운 handheld 포즈 변동을 그대로 학습 데이터로 흡수하도록 의도했다.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ConvAutoEncoder(nn.Module):
    """소형 Conv AE. 입력 (H,W)는 4의 배수 4번(=16배수)이어야 한다."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, latent_dim, 4, stride=2, padding=1), nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, in_channels, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def image_to_tensor(image_bgr: np.ndarray) -> torch.Tensor:
    """BGR uint8 HWC → RGB float32 CHW [0,1] 텐서."""
    rgb = image_bgr[..., ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1)


def train_autoencoder(
    images: list[np.ndarray], epochs: int = 30, lr: float = 1e-3, batch_size: int = 16
) -> ConvAutoEncoder:
    """정상 이미지 목록으로 AE를 학습한다 (CPU)."""
    model = ConvAutoEncoder()
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    X = torch.stack([image_to_tensor(img) for img in images])
    n = len(X)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            batch = X[idx]
            opt.zero_grad()
            recon = model(batch)
            loss = crit(recon, batch)
            loss.backward()
            opt.step()

    model.eval()
    return model


def ae_score(model: ConvAutoEncoder, image_bgr: np.ndarray, topk_ratio: float = 0.05) -> float:
    """재구성 오차 상위 topk_ratio 픽셀의 평균을 이상 스코어로 반환한다."""
    x = image_to_tensor(image_bgr).unsqueeze(0)
    with torch.no_grad():
        recon = model(x)
        err = ((recon - x) ** 2).mean(dim=1).flatten()  # 채널 평균 → 픽셀별 오차
    k = max(1, int(topk_ratio * err.numel()))
    return torch.topk(err, k).values.mean().item()
