"""수작업(handcrafted) 특징 추출 — CPU 경량 접근의 기본 후보."""
from __future__ import annotations

import numpy as np


def color_histogram(image: np.ndarray, bins: int = 16) -> np.ndarray:
    """채널별 색상 히스토그램 특징 벡터."""
    # TODO
    raise NotImplementedError


def texture_stats(image: np.ndarray) -> np.ndarray:
    """텍스처 통계 (예: LBP/GLCM/gradient 통계) 특징 벡터."""
    # TODO
    raise NotImplementedError


def edge_stats(image: np.ndarray) -> np.ndarray:
    """엣지 밀도·방향 분포 등 형태 관련 통계 특징."""
    # TODO: cv2.Sobel/Canny 는 '특징 계산'에만 사용, 판정은 scoring 에서
    raise NotImplementedError


def extract_handcrafted(image: np.ndarray) -> np.ndarray:
    """위 특징들을 이어붙여 (D,) 벡터로 반환한다."""
    # TODO
    raise NotImplementedError
