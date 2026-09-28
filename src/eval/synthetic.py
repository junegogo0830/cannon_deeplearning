"""합성 결함 주입 — FAIL 라벨이 없는 기종(13개 중 10개)에서도 정량 평가를 가능하게 한다.

주의: 여기서 나오는 recall/precision은 "실제 불량 검출력"이 아니라 "스코어링 로직이
국소적 이상에 얼마나 민감한가"를 비교하는 용도다 (reports/eda_summary.md §8 참고).
"""
from __future__ import annotations

import numpy as np


def inject_cutout_patch(image: np.ndarray, patch_size: int = 40, seed: int | None = None) -> np.ndarray:
    """무작위 위치를 단색으로 덮어 "부품/라벨 누락"류 결함을 흉내낸다."""
    rng = np.random.RandomState(seed)
    img = image.copy()
    h, w = img.shape[:2]
    ph, pw = min(patch_size, h), min(patch_size, w)
    y = rng.randint(0, h - ph + 1)
    x = rng.randint(0, w - pw + 1)
    fill = int(rng.randint(0, 256))
    img[y : y + ph, x : x + pw] = fill
    return img


def inject_brightness_drop(image: np.ndarray, factor: float = 0.5, seed: int | None = None) -> np.ndarray:
    """전체 밝기를 낮춰 조명/노출 이상을 흉내낸다."""
    img = image.astype(np.float32) * factor
    return np.clip(img, 0, 255).astype(np.uint8)


def inject_patch_copy_shift(
    image: np.ndarray, patch_size: int = 40, shift: int = 60, seed: int | None = None
) -> np.ndarray:
    """한 영역을 복사해 옆으로 옮겨 붙여 "부품 위치 어긋남"류 결함을 흉내낸다."""
    rng = np.random.RandomState(seed)
    img = image.copy()
    h, w = img.shape[:2]
    ph, pw = min(patch_size, h), min(patch_size, w)
    y = rng.randint(0, h - ph + 1)
    x = rng.randint(0, w - pw + 1)
    patch = img[y : y + ph, x : x + pw].copy()
    y2 = min(max(0, y + shift), h - ph)
    x2 = min(max(0, x + shift), w - pw)
    img[y2 : y2 + ph, x2 : x2 + pw] = patch
    return img


METHODS = {
    "cutout_patch": inject_cutout_patch,
    "brightness_drop": inject_brightness_drop,
    "patch_copy_shift": inject_patch_copy_shift,
}


def generate_synthetic_fail(image: np.ndarray, method: str, seed: int | None = None) -> np.ndarray:
    """config.evaluation.synthetic_anomaly.methods 중 하나를 적용한다."""
    if method not in METHODS:
        raise ValueError(f"알 수 없는 합성 결함 method: {method} (사용 가능: {list(METHODS)})")
    return METHODS[method](image, seed=seed)
