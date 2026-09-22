"""패치 단위 특징 추출 (국소 이상 위치 탐지용)."""
from __future__ import annotations

import numpy as np


def extract_patches(image: np.ndarray, patch_size: int, stride: int) -> np.ndarray:
    """이미지를 슬라이딩 윈도우로 잘라 (N, patch_size, patch_size, C) 로 반환한다."""
    # TODO
    raise NotImplementedError


def patch_features(patches: np.ndarray) -> np.ndarray:
    """패치별 특징 벡터 (N, D) 를 계산한다."""
    # TODO: handcrafted 또는 CNN 임베딩 사용
    raise NotImplementedError


def patch_grid_shape(image_size: tuple[int, int], patch_size: int, stride: int) -> tuple[int, int]:
    """패치 스코어를 히트맵으로 복원할 때 쓰는 (rows, cols)."""
    # TODO
    raise NotImplementedError
