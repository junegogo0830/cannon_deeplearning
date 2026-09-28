"""스텝별 임계값(threshold) 산출.

FAIL 이 극히 적으므로 기본은 '정상 검증 스코어 분포' 기반이고,
FAIL 이 조금이라도 있으면 보정(calibration) 용도로만 활용한다.
"""
from __future__ import annotations

from typing import Any

import numpy as np


def threshold_percentile(normal_scores: np.ndarray, percentile: float = 99.0) -> float:
    """정상 스코어의 상위 percentile 값을 임계값으로 사용한다."""
    return float(np.percentile(normal_scores, percentile))


def threshold_mean_std(normal_scores: np.ndarray, k: float = 3.0) -> float:
    """mean + k * std 를 임계값으로 사용한다."""
    return float(normal_scores.mean() + k * normal_scores.std())


def threshold_fail_calibrated(
    normal_scores: np.ndarray, fail_scores: np.ndarray, target_recall: float = 1.0
) -> float:
    """소수의 FAIL 스코어를 참고해 임계값을 보정한다 (목표 recall을 만족하는 최소 threshold).

    FAIL 샘플 수가 매우 적으므로(전체 12건) 과적합 위험이 크다 — 참고용으로만 쓸 것.
    """
    if len(fail_scores) == 0:
        return threshold_percentile(normal_scores)
    sorted_fail = np.sort(fail_scores)
    idx = min(int(np.ceil((1 - target_recall) * len(sorted_fail))), len(sorted_fail) - 1)
    return float(sorted_fail[idx])


def fit_thresholds(
    scores_by_step: dict[int, dict[str, np.ndarray]], cfg: dict[str, Any]
) -> dict[int, float]:
    """스텝별로 config.threshold.method 에 맞는 임계값을 계산한다.

    Args:
        scores_by_step: {step_id: {"normal": ndarray, "fail": ndarray(optional)}}.
        cfg: config 의 threshold 섹션.
    """
    method = cfg.get("method", "percentile")
    out: dict[int, float] = {}
    for step, d in scores_by_step.items():
        normal = d["normal"]
        if method == "percentile":
            out[step] = threshold_percentile(normal, cfg.get("percentile", 99.0))
        elif method == "mean_std":
            out[step] = threshold_mean_std(normal, cfg.get("k_std", 3.0))
        elif method == "fail_calibrated":
            out[step] = threshold_fail_calibrated(normal, d.get("fail", np.array([])))
        else:
            raise ValueError(f"알 수 없는 threshold.method: {method}")
    return out
