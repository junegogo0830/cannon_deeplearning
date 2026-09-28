"""이미지 전처리 (ROI 크롭, 리사이즈, 정규화, 정렬).

정렬(align_to_reference)의 이중 게이트 파라미터는 EDA(reports/eda_summary.md §5)에서
실측한 핸디형 카메라 편차를 근거로 configs/config.yaml 의 `alignment` 섹션에 정의돼 있다.
"""
from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def crop_roi(image: np.ndarray, roi: tuple[int, int, int, int] | None) -> np.ndarray:
    """ROI [x, y, w, h] 로 이미지를 잘라낸다. roi가 None이면 원본 그대로 반환."""
    if roi is None:
        return image
    x, y, w, h = roi
    H, W = image.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    return image[y0:y1, x0:x1]


def resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """size=(H, W) 로 리사이즈한다."""
    h, w = size
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


def normalize_image(
    image: np.ndarray, mean: float | None = None, std: float | None = None
) -> np.ndarray:
    """픽셀값을 [0,1] float32 로 스케일하고, mean/std가 주어지면 표준화한다."""
    img = image.astype(np.float32) / 255.0
    if mean is not None and std is not None:
        img = (img - mean) / (std + 1e-8)
    return img


def align_to_reference(
    image: np.ndarray, reference: np.ndarray, align_cfg: dict[str, Any]
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """기준 이미지에 image를 affine(4DOF) 정렬한다. 이중 게이트를 통과 못하면 None을 반환한다.

    이중 게이트 (EDA 근거):
      1) 매칭 신뢰도: inlier 개수/비율이 충분해야 함
      2) 물리적 타당성: 추정된 이동/회전이 핸디형 카메라가 낼 수 있는 범위 안이어야 함
    homography(8DOF)는 매칭이 부실할 때 소수 노이즈 포인트에 과적합해 더 나빠지는 것을
    실험으로 확인했기 때문에 쓰지 않는다 (reports/eda_summary.md §5).

    Returns:
        (정렬된 image 또는 None, 진단 정보 dict)
    """
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if reference.ndim == 3 else reference
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    diag: dict[str, Any] = {
        "ok": False, "reason": None, "inliers": 0, "inlier_ratio": 0.0,
        "dx": None, "dy": None, "rot_deg": None,
    }

    orb = cv2.ORB_create(1500)
    kp1, des1 = orb.detectAndCompute(gray_ref, None)
    kp2, des2 = orb.detectAndCompute(gray_img, None)
    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        diag["reason"] = "insufficient_keypoints"
        return None, diag

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda m: m.distance)[:200]
    if len(matches) < 8:
        diag["reason"] = "insufficient_matches"
        return None, diag

    src = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    M, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC,
        ransacReprojThreshold=align_cfg.get("ransac_reproj_threshold", 5.0),
    )
    if M is None:
        diag["reason"] = "estimation_failed"
        return None, diag

    n_inliers = int(inliers.sum())
    inlier_ratio = n_inliers / len(inliers)
    dx, dy = float(M[0, 2]), float(M[1, 2])
    rot_deg = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
    diag.update(inliers=n_inliers, inlier_ratio=inlier_ratio, dx=dx, dy=dy, rot_deg=rot_deg)

    if n_inliers < align_cfg.get("min_inliers", 15) or inlier_ratio < align_cfg.get("min_inlier_ratio", 0.25):
        diag["reason"] = "low_inlier_confidence"
        return None, diag
    if (
        abs(dx) > align_cfg.get("max_abs_dx_px", 200)
        or abs(dy) > align_cfg.get("max_abs_dy_px", 150)
        or abs(rot_deg) > align_cfg.get("max_abs_rotation_deg", 20)
    ):
        diag["reason"] = "implausible_transform"
        return None, diag

    h, w = reference.shape[:2]
    warped = cv2.warpAffine(image, M, (w, h), flags=cv2.WARP_INVERSE_MAP)
    diag["ok"] = True
    return warped, diag


def preprocess(
    image: np.ndarray, roi: tuple[int, int, int, int] | None, size: tuple[int, int]
) -> np.ndarray:
    """crop_roi → resize 를 순서대로 적용하는 표준 파이프라인 (정렬 이전 단계)."""
    img = crop_roi(image, roi)
    img = resize_image(img, size)
    return img
