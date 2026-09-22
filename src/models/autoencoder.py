"""경량 컨볼루션 오토인코더 — 재구성 오차 기반 이상탐지 (CPU 학습 가능한 소형 구조)."""
from __future__ import annotations

import numpy as np

from src.models.base import AnomalyModel

# TODO: import torch, torch.nn as nn


class ConvAutoEncoder:  # TODO: nn.Module 상속
    """소형 Conv AE (파라미터 수를 작게 유지)."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 64) -> None:
        # TODO: encoder / decoder 정의
        raise NotImplementedError

    def forward(self, x):
        # TODO
        raise NotImplementedError


class AutoEncoderModel(AnomalyModel):
    """AE 학습 + 재구성 오차 스코어 래퍼."""

    def __init__(self, epochs: int = 30, lr: float = 1e-3, batch_size: int = 16) -> None:
        # TODO
        raise NotImplementedError

    def fit(self, features: np.ndarray) -> "AutoEncoderModel":
        # TODO: 정상 이미지로만 MSE 학습 (CPU, torch.set_num_threads)
        raise NotImplementedError

    def score(self, features: np.ndarray) -> np.ndarray:
        # TODO: 재구성 오차(픽셀/패치 단위) → 스코어
        raise NotImplementedError
