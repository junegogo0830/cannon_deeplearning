"""가우시안(마할라노비스) 기반 이상탐지 — 가장 가벼운 CPU 베이스라인.

정상 특징 분포를 다변량 가우시안으로 근사하고, 마할라노비스 거리를 이상 스코어로 쓴다.
"""
from __future__ import annotations

import numpy as np

from src.models.base import AnomalyModel


class GaussianModel(AnomalyModel):
    """정상 분포 평균/공분산을 저장하고 마할라노비스 거리를 스코어로 반환한다."""

    def __init__(self, reg_eps: float = 1e-3) -> None:
        """reg_eps: 공분산 대각 정규화 계수 (역행렬 안정화)."""
        # TODO: mean_, cov_inv_ 초기화
        raise NotImplementedError

    def fit(self, features: np.ndarray) -> "GaussianModel":
        # TODO: mean, cov(+reg_eps*I) 계산 → 역행렬 (또는 sklearn.covariance 사용)
        raise NotImplementedError

    def score(self, features: np.ndarray) -> np.ndarray:
        # TODO: (x-mean)^T Σ^-1 (x-mean) 의 제곱근
        raise NotImplementedError
