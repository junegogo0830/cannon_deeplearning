"""패치 단위 특징 추출 (국소 이상 위치 탐지용, 멀티스케일 지원)."""
from __future__ import annotations

import numpy as np

from src.features.handcrafted import extract_handcrafted


def extract_patches(image: np.ndarray, patch_size: int, stride: int) -> np.ndarray:
    """이미지를 오버랩 슬라이딩 윈도우로 잘라 (N, patch_size, patch_size, C) 로 반환한다.

    이미지 가장자리가 patch_size로 나누어떨어지지 않으면 마지막 패치를 안쪽으로 당겨 맞춘다
    (패치가 잘려서 작아지는 것을 방지 — 잘린 패치는 특징이 왜곡되기 쉬움).
    """
    h, w = image.shape[:2]
    ys = list(range(0, max(h - patch_size, 0) + 1, stride))
    xs = list(range(0, max(w - patch_size, 0) + 1, stride))
    if not ys or ys[-1] != h - patch_size:
        ys.append(max(h - patch_size, 0))
    if not xs or xs[-1] != w - patch_size:
        xs.append(max(w - patch_size, 0))

    patches = [image[y : y + patch_size, x : x + patch_size] for y in ys for x in xs]
    return np.stack(patches, axis=0)


def patch_grid_shape(image_size: tuple[int, int], patch_size: int, stride: int) -> tuple[int, int]:
    """extract_patches와 동일한 규칙으로 (rows, cols)를 계산한다 (히트맵 복원용)."""
    h, w = image_size
    ys = list(range(0, max(h - patch_size, 0) + 1, stride))
    xs = list(range(0, max(w - patch_size, 0) + 1, stride))
    if not ys or ys[-1] != h - patch_size:
        ys.append(max(h - patch_size, 0))
    if not xs or xs[-1] != w - patch_size:
        xs.append(max(w - patch_size, 0))
    return len(ys), len(xs)


def patch_features(patches: np.ndarray) -> np.ndarray:
    """패치별 특징 벡터 (N, D) 를 계산한다."""
    return np.stack([extract_handcrafted(p) for p in patches], axis=0)
