"""이미지 전처리 (ROI 크롭, 리사이즈, 정규화, 정렬)."""
from __future__ import annotations

import numpy as np


def crop_roi(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    """ROI [x, y, w, h] 로 이미지를 잘라낸다."""
    # TODO: 경계 벗어남 처리
    raise NotImplementedError


def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """(H, W) 크기로 리사이즈한다."""
    # TODO: cv2.resize, 종횡비 처리 정책 결정
    raise NotImplementedError


def normalize_image(image: np.ndarray, mean: float | None = None, std: float | None = None) -> np.ndarray:
    """픽셀값을 float32 로 정규화한다 (0~1 스케일 후 선택적 표준화)."""
    # TODO
    raise NotImplementedError


def align_to_reference(image: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """촬영 위치 편차 보정: 기준 이미지에 맞춰 정렬한다 (선택 사항).

    이상탐지는 정상 이미지끼리의 위치가 맞을수록 성능이 좋다.
    """
    # TODO: 특징점 매칭/ECC 등으로 정렬 (판정에 쓰지 않고 '정렬'에만 사용)
    raise NotImplementedError


def preprocess(image: np.ndarray, roi: tuple[int, int, int, int], size: tuple[int, int]) -> np.ndarray:
    """crop_roi → resize → normalize 를 순서대로 적용하는 표준 파이프라인."""
    # TODO
    raise NotImplementedError
