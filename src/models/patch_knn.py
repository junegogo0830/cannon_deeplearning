"""패치 메모리뱅크 + k-NN 이상탐지 (PatchCore 계열 아이디어, 직접 구현 대상).

정상 패치 특징을 메모리뱅크에 저장하고, 입력 패치와 가장 가까운 정상 패치와의 거리를 스코어로 쓴다.
"""
from __future__ import annotations

import numpy as np

from src.models.base import AnomalyModel


class PatchKNNModel(AnomalyModel):
    """정상 패치 메모리뱅크 기반 모델."""

    def __init__(self, k: int = 3, coreset_ratio: float = 0.1) -> None:
        """
        Args:
            k: 최근접 이웃 수.
            coreset_ratio: 메모리뱅크 서브샘플 비율 (CPU 속도/메모리 절감).
        """
        # TODO
        raise NotImplementedError

    def fit(self, features: np.ndarray) -> "PatchKNNModel":
        # TODO: 패치 특징 모으기 → coreset 서브샘플링 → 메모리뱅크 저장
        raise NotImplementedError

    def score(self, features: np.ndarray) -> np.ndarray:
        # TODO: sklearn NearestNeighbors 등으로 k-NN 거리 → (N, P) 패치 스코어
        raise NotImplementedError


def greedy_coreset(features: np.ndarray, ratio: float, seed: int = 0) -> np.ndarray:
    """k-center greedy 로 대표 패치만 추려 메모리뱅크를 줄인다."""
    # TODO
    raise NotImplementedError
