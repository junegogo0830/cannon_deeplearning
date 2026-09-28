"""수작업(handcrafted) 특징 추출 — CPU 경량 접근의 기본 후보.

EDA(reports/eda_summary.md §6,§7)에서 확인한 두 가지 잡음원에 대응하도록 설계했다:
  - 조명/반사 변화: 원본 밝기(intensity) 대신 그래디언트 방향(형태)을 특징으로 삼는다.
  - 저채도(흰색/회색) 표면의 Hue 불안정: Hue 히스토그램을 채도(Saturation)로 가중해,
    채도가 낮은 픽셀은 자동으로 영향력이 작아지게 한다.
"""
from __future__ import annotations

import cv2
import numpy as np


def edge_orientation_hist(image: np.ndarray, bins: int = 9) -> np.ndarray:
    """그래디언트 방향 히스토그램 (크기로 가중) — 밝기 절대값이 아니라 형태/구조를 포착."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    ang = ang % 180.0  # 방향은 0~180도로 접어서 사용 (극성 무시)
    hist, _ = np.histogram(ang, bins=bins, range=(0, 180), weights=mag)
    total = hist.sum()
    return hist / total if total > 0 else hist


def hue_hist_sat_weighted(image: np.ndarray, bins: int = 12) -> np.ndarray:
    """채도로 가중한 Hue 히스토그램. 저채도(흰/회색) 영역은 자동으로 가중치가 작아진다."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.float32)
    s = hsv[..., 1].astype(np.float32)
    hist, _ = np.histogram(h, bins=bins, range=(0, 180), weights=s)
    total = hist.sum()
    return hist / total if total > 0 else hist


def texture_stats(image: np.ndarray) -> np.ndarray:
    """국소 대비/텍스처 통계 (그레이스케일 평균·표준편차, 라플라시안 평균절대값·표준편차)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    gray = gray.astype(np.float32)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    return np.array([gray.mean(), gray.std(), np.abs(lap).mean(), lap.std()], dtype=np.float32)


def extract_handcrafted(image: np.ndarray) -> np.ndarray:
    """edge_orientation_hist + hue_hist_sat_weighted + texture_stats 를 이어붙인 벡터."""
    return np.concatenate(
        [edge_orientation_hist(image), hue_hist_sat_weighted(image), texture_stats(image)]
    ).astype(np.float32)
