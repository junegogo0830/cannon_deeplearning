"""가우시안(마할라노비스) 기반 이상탐지 — 비교용 베이스라인.

정상 특징 분포를 다변량 가우시안으로 근사하고, 마할라노비스 거리를 이상 스코어로 쓴다.
patch_knn(메인 모델)이 이것보다 왜 나은지 정량 비교하는 근거 자료로 사용한다.
"""
from __future__ import annotations

import numpy as np

from src.models.base import AnomalyModel


class GaussianModel(AnomalyModel):
    """정상 분포 평균/공분산을 저장하고 마할라노비스 거리를 스코어로 반환한다."""

    def __init__(self, reg_eps: float = 1e-3) -> None:
        self.reg_eps = reg_eps
        self.mean_: np.ndarray | None = None
        self.cov_inv_: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "GaussianModel":
        self.mean_ = features.mean(axis=0)
        cov = np.atleast_2d(np.cov(features, rowvar=False))
        cov = cov + self.reg_eps * np.eye(cov.shape[0])
        self.cov_inv_ = np.linalg.inv(cov)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        if self.mean_ is None or self.cov_inv_ is None:
            raise RuntimeError("fit() 을 먼저 호출해야 합니다.")
        diff = features - self.mean_
        m = np.einsum("ij,jk,ik->i", diff, self.cov_inv_, diff)
        return np.sqrt(np.maximum(m, 0.0))
