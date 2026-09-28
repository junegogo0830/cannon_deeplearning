"""모델 출력(패치 스코어) → 스텝 단위 최종 이상 스코어 변환."""
from __future__ import annotations

import cv2
import numpy as np


def aggregate_patch_scores(patch_scores: np.ndarray, method: str = "max", topk: int = 10) -> np.ndarray:
    """패치 스코어 (N, P) → 이미지 스코어 (N,)."""
    if method == "max":
        return patch_scores.max(axis=1)
    if method == "mean":
        return patch_scores.mean(axis=1)
    if method == "topk_mean":
        k = min(topk, patch_scores.shape[1])
        part = np.partition(patch_scores, -k, axis=1)[:, -k:]
        return part.mean(axis=1)
    raise ValueError(f"알 수 없는 method: {method} (max | mean | topk_mean)")


def normalize_scores(scores: np.ndarray, ref_scores: np.ndarray, method: str = "minmax") -> np.ndarray:
    """스텝별 스케일 차이를 맞추기 위해 정상(학습) 스코어 분포 기준으로 정규화한다."""
    if method == "none":
        return scores
    if method == "minmax":
        lo, hi = ref_scores.min(), ref_scores.max()
        return (scores - lo) / (hi - lo + 1e-8)
    if method == "zscore":
        mu, sd = ref_scores.mean(), ref_scores.std()
        return (scores - mu) / (sd + 1e-8)
    raise ValueError(f"알 수 없는 method: {method} (none | minmax | zscore)")


def combine_scales(scale_scores: dict[int, np.ndarray], method: str = "max") -> np.ndarray:
    """스케일별 정규화된 이미지 스코어 {scale: (N,)} 를 하나로 결합한다.

    기본은 max(OR 논리) — 결함 스펙트럼이 넓어(라벨 누락 ~ 미세 이음새) 어느 스케일이든
    하나가 크게 반응하면 이상으로 보는 게 미검출(false negative) 위험을 줄인다.
    """
    stacked = np.stack(list(scale_scores.values()), axis=0)  # (S, N)
    if method == "max":
        return stacked.max(axis=0)
    if method == "mean":
        return stacked.mean(axis=0)
    raise ValueError(f"알 수 없는 method: {method} (max | mean)")


def anomaly_heatmap(patch_scores: np.ndarray, grid_shape: tuple[int, int], out_size: tuple[int, int]) -> np.ndarray:
    """패치 스코어를 원본 크기 히트맵으로 복원한다 (시각화/원인 분석용)."""
    rows, cols = grid_shape
    grid = patch_scores.reshape(rows, cols).astype(np.float32)
    return cv2.resize(grid, (out_size[1], out_size[0]), interpolation=cv2.INTER_CUBIC)
