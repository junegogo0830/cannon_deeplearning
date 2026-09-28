"""패치 메모리뱅크 + k-NN 이상탐지 (PatchCore류 아이디어, 직접 구현) — 메인 모델.

정상 패치 특징을 메모리뱅크에 저장하고, 입력 패치와 가장 가까운 정상 패치와의 거리를 스코어로 쓴다.
뱅크 검색은 "같은 위치"가 아니라 뱅크 전체를 대상으로 하므로, 핸디형 카메라의 위치 편차에
어느 정도 자체 내성을 가진다 (reports/eda_summary.md §5, §8 참고).
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors

from src.models.base import AnomalyModel


class PatchKNNModel(AnomalyModel):
    """정상 패치 메모리뱅크 기반 모델."""

    def __init__(self, k: int = 3, coreset_ratio: float = 0.1, max_bank_size: int = 20000) -> None:
        """
        Args:
            k: 최근접 이웃 수.
            coreset_ratio: 메모리뱅크 서브샘플 비율 (CPU 속도/메모리 절감).
            max_bank_size: coreset_ratio를 적용해도 이 값을 넘지 않도록 추가로 캡을 건다.
        """
        self.k = k
        self.coreset_ratio = coreset_ratio
        self.max_bank_size = max_bank_size
        self.bank_: np.ndarray | None = None
        self._nn: NearestNeighbors | None = None

    def fit(self, features: np.ndarray) -> "PatchKNNModel":
        ratio = self.coreset_ratio
        if len(features) * ratio > self.max_bank_size:
            ratio = self.max_bank_size / len(features)
        self.bank_ = greedy_coreset(features, ratio)
        k = min(self.k, len(self.bank_))
        self._nn = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(self.bank_)
        return self

    def score(self, features: np.ndarray) -> np.ndarray:
        """features: (N, D) → (N,), 또는 (N, P, D) → (N, P)."""
        if self._nn is None:
            raise RuntimeError("fit() 을 먼저 호출해야 합니다.")
        orig_shape = features.shape
        flat = features.reshape(-1, orig_shape[-1])
        dist, _ = self._nn.kneighbors(flat)
        d = dist.mean(axis=1)
        if len(orig_shape) == 3:
            return d.reshape(orig_shape[0], orig_shape[1])
        return d


def greedy_coreset(features: np.ndarray, ratio: float, seed: int = 0) -> np.ndarray:
    """k-center greedy 로 대표 패치만 추려 메모리뱅크를 줄인다.

    무작위 서브샘플링과 달리 "특징 공간을 고르게 덮는" 점을 우선 선택하므로,
    소수 정상 서브그룹(예: 색상 변형 부품)도 다수에 묻혀 사라지지 않고 뱅크에 남을 확률이 높다.
    """
    n = len(features)
    m = max(1, int(round(n * ratio)))
    if m >= n:
        return features.copy()

    rng = np.random.RandomState(seed)
    selected = [int(rng.randint(n))]
    min_dist = np.linalg.norm(features - features[selected[0]], axis=1)

    for _ in range(m - 1):
        next_idx = int(np.argmax(min_dist))
        selected.append(next_idx)
        d = np.linalg.norm(features - features[next_idx], axis=1)
        min_dist = np.minimum(min_dist, d)

    return features[selected]
