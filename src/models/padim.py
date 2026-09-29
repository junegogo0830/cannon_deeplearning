"""PaDiM (Patch Distribution Modeling) — 위치별 가우시안 기반 이상탐지.

patch_knn(뱅크 전체에서 최근접 탐색, 위치 무관)과 정반대되는 가정을 깐다:
"패치 그리드의 같은 위치(position)는 이미지마다 같은 물리적 부위를 나타낸다"고 가정하고,
위치마다 독립적인 다변량 가우시안(평균/공분산)을 학습해 마할라노비스 거리를 스코어로 쓴다.

이 가정은 이미지들이 잘 정렬돼 있을 때만 성립한다 — 핸디형 카메라로 찍은 우리 데이터에서
정렬을 안 쓰면(혹은 정렬이 불안정한 스텝이면) 이 가정 자체가 깨진다. 그 효과를 patch_knn과
나란히 비교해서 확인하는 것이 이 모델을 넣은 목적이다 (scripts/compare_ad_methods.py).
"""
from __future__ import annotations

import numpy as np


class PaDiMModel:
    """패치 그리드 위치별로 (평균, 공분산역행렬)을 저장한다."""

    def __init__(self, reg_eps: float = 1e-2) -> None:
        self.reg_eps = reg_eps
        self.means_: np.ndarray | None = None  # (P, D)
        self.cov_invs_: np.ndarray | None = None  # (P, D, D)

    def fit(self, features: np.ndarray) -> "PaDiMModel":
        """features: (N, P, D) — N장의 이미지, 각 P개 위치, D차원 특징."""
        n, p, d = features.shape
        self.means_ = features.mean(axis=0)  # (P, D)
        self.cov_invs_ = np.empty((p, d, d), dtype=np.float32)
        eye = np.eye(d, dtype=np.float32)
        for pi in range(p):
            x = features[:, pi, :]
            cov = np.cov(x, rowvar=False).astype(np.float32)
            cov = np.atleast_2d(cov) + self.reg_eps * eye
            self.cov_invs_[pi] = np.linalg.inv(cov)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """features: (N, P, D) → (N, P) 위치별 마할라노비스 거리."""
        if self.means_ is None:
            raise RuntimeError("fit() 을 먼저 호출해야 합니다.")
        n, p, d = features.shape
        out = np.empty((n, p), dtype=np.float32)
        for pi in range(p):
            diff = features[:, pi, :] - self.means_[pi]
            m = np.einsum("ij,jk,ik->i", diff, self.cov_invs_[pi], diff)
            out[:, pi] = np.sqrt(np.maximum(m, 0.0))
        return out
