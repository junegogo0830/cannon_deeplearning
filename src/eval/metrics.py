"""평가 지표 — 극심한 클래스 불균형을 고려해 accuracy 는 주 지표로 쓰지 않는다."""
from __future__ import annotations

import numpy as np


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC (임계값 독립). y_true: 0=PASS, 1=FAIL."""
    # TODO: FAIL 이 1건 미만이면 계산 불가 처리
    raise NotImplementedError


def aupr(y_true: np.ndarray, scores: np.ndarray) -> float:
    """PR-AUC (불균형 데이터에서 더 민감)."""
    # TODO
    raise NotImplementedError


def confusion_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, int]:
    """임계값 적용 시 TP/FP/TN/FN."""
    # TODO
    raise NotImplementedError


def precision_recall_f1(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """임계값 적용 시 precision / recall / f1."""
    # TODO
    raise NotImplementedError


def evaluate_step(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """한 스텝의 모든 지표를 dict 로 반환한다."""
    # TODO: 위 함수들 조합
    raise NotImplementedError


def evaluate_all_steps(results: dict[int, dict]) -> "pd.DataFrame":  # noqa: F821
    """스텝별 지표를 표(DataFrame)로 정리한다 (15행)."""
    # TODO
    raise NotImplementedError
