"""스텝별 임계값(threshold) 산출.

FAIL 이 극히 적으므로 기본은 '정상 검증 스코어 분포' 기반이고,
FAIL 이 조금이라도 있으면 보정(calibration) 용도로만 활용한다.
"""
from __future__ import annotations

import numpy as np


def threshold_percentile(normal_scores: np.ndarray, percentile: float = 99.0) -> float:
    """정상 스코어의 상위 percentile 값을 임계값으로 사용한다."""
    # TODO
    raise NotImplementedError


def threshold_mean_std(normal_scores: np.ndarray, k: float = 3.0) -> float:
    """mean + k * std 를 임계값으로 사용한다."""
    # TODO
    raise NotImplementedError


def threshold_fail_calibrated(
    normal_scores: np.ndarray, fail_scores: np.ndarray, target_recall: float = 1.0
) -> float:
    """소수의 FAIL 스코어를 참고해 임계값을 보정한다.

    예: 목표 recall(FAIL 검출률)을 만족하면서 정상 오검출(FP)을 최소화하는 지점.
    """
    # TODO: 과적합 주의 (FAIL 샘플 수가 매우 적음 → 여유 마진 고려)
    raise NotImplementedError


def fit_thresholds(
    scores_by_step: dict[int, dict[str, np.ndarray]], cfg: dict
) -> dict[int, float]:
    """스텝별로 config.threshold.method 에 맞는 임계값을 계산한다.

    Args:
        scores_by_step: {step_id: {"normal": ndarray, "fail": ndarray(optional)}}.
        cfg: config 의 threshold 섹션.

    Returns:
        {step_id: threshold}. config 에 스텝별 threshold 가 지정돼 있으면 그 값을 우선한다.
    """
    # TODO
    raise NotImplementedError
