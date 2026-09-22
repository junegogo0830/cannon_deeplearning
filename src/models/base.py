"""이상탐지 모델 공통 인터페이스.

모든 모델은 '정상(PASS) 데이터만으로 fit' 하고, 입력에 대해 연속적인 이상 스코어를 낸다.
PASS/FAIL 이진 판정은 모델이 아니라 scoring/ 의 threshold 로직이 담당한다.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class AnomalyModel(ABC):
    """이상탐지 모델 추상 클래스."""

    @abstractmethod
    def fit(self, features: np.ndarray) -> "AnomalyModel":
        """정상 데이터의 특징으로 학습한다.

        Args:
            features: (N, D) 또는 (N, P, D) — 이미지 단위/패치 단위.
        """
        # TODO
        raise NotImplementedError

    @abstractmethod
    def score(self, features: np.ndarray) -> np.ndarray:
        """이상 스코어를 반환한다 (클수록 이상).

        Returns:
            (N,) 이미지 스코어 또는 (N, P) 패치 스코어.
        """
        # TODO
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        """학습된 파라미터를 저장한다."""
        # TODO
        raise NotImplementedError

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyModel":
        """저장된 모델을 로드한다."""
        # TODO
        raise NotImplementedError


def build_model(model_cfg: dict) -> AnomalyModel:
    """config.model.type 에 맞는 모델 인스턴스를 만든다 (gaussian | patch_knn | autoencoder)."""
    # TODO: 레지스트리/딕셔너리 매핑
    raise NotImplementedError
