"""스코어 → PASS/FAIL 최종 판정 규칙."""
from __future__ import annotations

import numpy as np

PASS = "PASS"
FAIL = "FAIL"


def judge_step(score: float, threshold: float) -> str:
    """단일 스텝 판정: score > threshold 이면 FAIL, 같거나 작으면 PASS."""
    return FAIL if score > threshold else PASS


def judge_steps(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """벡터화 판정. (N,) 스코어 → (N,) bool (True=FAIL)."""
    return scores > thresholds


def judge_product(step_results: dict[int, str], rule: str = "all_steps_pass") -> str:
    """스텝별 결과를 종합해 제품 단위 PASS/FAIL 을 낸다."""
    if rule == "all_steps_pass":
        return FAIL if any(v == FAIL for v in step_results.values()) else PASS
    raise ValueError(f"알 수 없는 rule: {rule}")


def decision_margin(score: float, threshold: float) -> float:
    """임계값까지의 여유(margin). 양수면 PASS 쪽 여유, 음수면 FAIL 쪽으로 넘어간 정도."""
    return threshold - score
