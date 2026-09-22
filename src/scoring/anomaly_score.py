"""모델 출력(패치/이미지 스코어) → 스텝 단위 최종 이상 스코어 변환.

판정 로직의 핵심 파트: 기성 알고리즘의 결과를 그대로 쓰지 않고 스코어를 직접 정의한다.
"""
from __future__ import annotations

import numpy as np


def aggregate_patch_scores(patch_scores: np.ndarray, method: str = "max", topk: int = 10) -> np.ndarray:
    """패치 스코어 (N, P) → 이미지 스코어 (N,).

    Args:
        method: "max" | "mean" | "topk_mean".
        topk: topk_mean 에서 사용할 상위 개수.
    """
    # TODO
    raise NotImplementedError


def normalize_scores(scores: np.ndarray, ref_scores: np.ndarray, method: str = "minmax") -> np.ndarray:
    """스텝별 스케일 차이를 맞추기 위해 정상(학습) 스코어 분포 기준으로 정규화한다.

    Args:
        scores: 정규화할 스코어.
        ref_scores: 기준이 되는 정상 스코어.
        method: "none" | "minmax" | "zscore".
    """
    # TODO
    raise NotImplementedError


def anomaly_heatmap(patch_scores: np.ndarray, grid_shape: tuple[int, int], out_size: tuple[int, int]) -> np.ndarray:
    """패치 스코어를 원본 크기 히트맵으로 복원한다 (시각화/원인 분석용)."""
    # TODO: reshape → 업샘플 → (선택) 가우시안 스무딩
    raise NotImplementedError
