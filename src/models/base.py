"""이상탐지 모델 공통 인터페이스.

모든 모델은 '정상(PASS) 데이터만으로 fit' 하고, 입력에 대해 연속적인 이상 스코어를 낸다.
PASS/FAIL 이진 판정은 모델이 아니라 scoring/ 의 threshold 로직이 담당한다.
"""
from __future__ import annotations

import pickle
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class AnomalyModel(ABC):
    """이상탐지 모델 추상 클래스."""

    #: fit()에 (N, P, D)(이미지별 패치 그리드 보존)를 줘야 하는 모델은 True로 오버라이드한다.
    #: False(기본)면 패치 축을 이미지 경계 없이 펼친 (전체패치수, D)를 받는다 — 뱅크류(patch_knn 등).
    #: 위치별 통계가 필요한 PaDiM처럼 "이미지 i의 p번째 패치"라는 대응이 의미 있는 모델만 True로 둔다.
    expects_patch_grid: bool = False

    @abstractmethod
    def fit(self, features: np.ndarray) -> "AnomalyModel":
        """정상 데이터의 특징으로 학습한다. features: (N, D) — 패치 모델은 전체 패치 풀을 펼쳐서 전달."""
        raise NotImplementedError

    @abstractmethod
    def score(self, features: np.ndarray) -> np.ndarray:
        """이상 스코어를 반환한다 (클수록 이상). (N, D) → (N,), 패치 모델은 (N, P, D) → (N, P) 도 지원."""
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "AnomalyModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def build_model(model_cfg: dict[str, Any]) -> AnomalyModel:
    """config.model 에 맞는 모델 인스턴스를 만든다 (gaussian | patch_knn | padim)."""
    from src.models.gaussian import GaussianModel
    from src.models.padim import PaDiMModel
    from src.models.patch_knn import PatchKNNModel

    model_type = model_cfg["type"]
    params = model_cfg.get("params", {})
    if model_type == "gaussian":
        return GaussianModel(reg_eps=params.get("reg_eps", 1e-3))
    if model_type == "patch_knn":
        return PatchKNNModel(k=params.get("knn_k", 3), coreset_ratio=params.get("coreset_ratio", 0.1))
    if model_type == "padim":
        return PaDiMModel(reg_eps=params.get("reg_eps", 1e-2))
    raise ValueError(f"알 수 없는 model.type: {model_type} (사용 가능: gaussian, patch_knn, padim)")
