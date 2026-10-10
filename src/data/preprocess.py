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


def _check_plausible(dx: float, dy: float, rot_deg: float, align_cfg: dict[str, Any]) -> bool:
    return (
        abs(dx) <= align_cfg.get("max_abs_dx_px", 200)
        and abs(dy) <= align_cfg.get("max_abs_dy_px", 150)
        and abs(rot_deg) <= align_cfg.get("max_abs_rotation_deg", 20)
    )


def _align_orb(
    gray_ref: np.ndarray, gray_img: np.ndarray, align_cfg: dict[str, Any]
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """특징점 기반(ORB+RANSAC) 1차 시도. 질감이 충분한 정상 케이스에서 빠르고 정확하다."""
    diag: dict[str, Any] = {"inliers": 0, "inlier_ratio": 0.0, "dx": None, "dy": None, "rot_deg": None}

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
    diag["reason"] = None
    return M, diag


def _align_ecc(
    gray_ref: np.ndarray, gray_img: np.ndarray, align_cfg: dict[str, Any]
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """명암 기반(ECC) 2차 시도 — ORB가 특징점 부족으로 실패했을 때만 쓴다.

    결함이 바로 그 특징점(라벨·마킹)을 지워버린 경우, ORB는 "매칭할 거리가 없다"는
    이유로 포기하지만 ECC는 전체 영역의 명암 그라디언트로 정렬을 시도할 수 있다.
    단, 큰 각도 변화(진짜 촬영 자세 차이)에는 수렴하지 않거나 엉뚱한 지역해로 빠지므로
    물리적 타당성 게이트는 ORB와 동일하게 엄격히 적용한다.
    """
    diag: dict[str, Any] = {"inliers": None, "inlier_ratio": None, "dx": None, "dy": None, "rot_deg": None}
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    try:
        # 다운스케일(1/4)로 수렴 안정성·속도 확보 후, 평행이동 성분만 원본 스케일로 복원.
        scale = 0.25
        small_ref = cv2.resize(gray_ref, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        small_img = cv2.resize(gray_img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        _, warp = cv2.findTransformECC(
            small_ref, small_img, warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5
        )
    except cv2.error:
        diag["reason"] = "ecc_not_converged"
        return None, diag

    dx, dy = float(warp[0, 2]) / scale, float(warp[1, 2]) / scale
    rot_deg = float(np.degrees(np.arctan2(warp[1, 0], warp[0, 0])))
    diag.update(dx=dx, dy=dy, rot_deg=rot_deg)
    M = np.array([[warp[0, 0], warp[0, 1], dx], [warp[1, 0], warp[1, 1], dy]], dtype=np.float32)
    diag["reason"] = None
    return M, diag


def align_to_reference(
    image: np.ndarray, reference: np.ndarray, align_cfg: dict[str, Any]
) -> tuple[np.ndarray | None, dict[str, Any]]:
    """기준 이미지에 image를 affine(4DOF) 정렬한다. 두 단계 모두 실패하면 None을 반환한다.

    1단계 ORB+RANSAC(특징점) → 특징점 부족/매칭 부족/inlier 부족으로 실패하면
    2단계 ECC(명암 기반)로 재시도 → 어느 쪽으로 얻은 변환이든 마지막은 동일한 물리적
    타당성 게이트(이동·회전 범위)를 통과해야 한다 (EDA 근거: reports/eda_summary.md §5).
    homography(8DOF)는 매칭이 부실할 때 소수 노이즈 포인트에 과적합해 더 나빠지는 것을
    실험으로 확인했기 때문에 쓰지 않는다.

    Returns:
        (정렬된 image 또는 None, 진단 정보 dict. diag["method"] = "orb" | "ecc" | None)
    """
    gray_ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if reference.ndim == 3 else reference
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    M, diag = _align_orb(gray_ref, gray_img, align_cfg)
    method = "orb"
    if M is None and diag["reason"] in ("insufficient_keypoints", "insufficient_matches", "low_inlier_confidence"):
        M, ecc_diag = _align_ecc(gray_ref, gray_img, align_cfg)
        if M is not None:
            diag = ecc_diag
            method = "ecc"

    diag["ok"] = False
    diag["method"] = None
    if M is None:
        return None, diag

    dx, dy, rot_deg = diag["dx"], diag["dy"], diag["rot_deg"]
    if not _check_plausible(dx, dy, rot_deg, align_cfg):
        diag["reason"] = "implausible_transform"
        return None, diag

    h, w = reference.shape[:2]
    warped = cv2.warpAffine(image, M, (w, h), flags=cv2.WARP_INVERSE_MAP)
    diag["ok"] = True
    diag["method"] = method
    diag["reason"] = None
    return warped, diag


def preprocess(
    image: np.ndarray, roi: tuple[int, int, int, int] | None, size: tuple[int, int]
) -> np.ndarray:
    """crop_roi → resize 를 순서대로 적용하는 표준 파이프라인 (정렬 이전 단계)."""
    img = crop_roi(image, roi)
    img = resize_image(img, size)
    return img
