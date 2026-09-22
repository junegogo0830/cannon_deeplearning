"""스코어 → PASS/FAIL 최종 판정 규칙."""
from __future__ import annotations

import numpy as np

PASS = "PASS"
FAIL = "FAIL"


def judge_step(score: float, threshold: float) -> str:
    """단일 스텝 판정: score > threshold 이면 FAIL, 아니면 PASS."""
    # TODO: 경계값(==) 처리 정책 명시
    raise NotImplementedError


def judge_steps(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """벡터화 판정. (N,) 스코어 → (N,) bool (True=FAIL)."""
    # TODO
    raise NotImplementedError


def judge_product(step_results: dict[int, str], rule: str = "all_steps_pass") -> str:
    """15개 스텝 결과를 종합해 제품 단위 PASS/FAIL 을 낸다.

    Args:
        step_results: {step_id: "PASS"|"FAIL"}.
        rule: "all_steps_pass" (하나라도 FAIL 이면 제품 FAIL) 등.
    """
    # TODO: 스텝별 가중치/필수 스텝 규칙이 필요한지 검토
    raise NotImplementedError


def decision_margin(score: float, threshold: float) -> float:
    """임계값까지의 여유(margin). 경계 사례(재검 대상) 표시에 사용."""
    # TODO
    raise NotImplementedError
