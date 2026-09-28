"""평가 지표 — 극심한 클래스 불균형을 고려해 accuracy 는 주 지표로 쓰지 않는다."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import metrics as skm


def auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC-AUC (임계값 독립). y_true: 0=PASS, 1=FAIL. 클래스가 하나뿐이면 계산 불가(NaN)."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(skm.roc_auc_score(y_true, scores))


def aupr(y_true: np.ndarray, scores: np.ndarray) -> float:
    """PR-AUC (불균형 데이터에서 더 민감)."""
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(skm.average_precision_score(y_true, scores))


def confusion_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, int]:
    """임계값 적용 시 TP/FP/TN/FN."""
    pred = (scores > threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)
    return {
        "tp": int(((pred == 1) & (y_true == 1)).sum()),
        "fp": int(((pred == 1) & (y_true == 0)).sum()),
        "tn": int(((pred == 0) & (y_true == 0)).sum()),
        "fn": int(((pred == 0) & (y_true == 1)).sum()),
    }


def precision_recall_f1(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """임계값 적용 시 precision / recall / f1."""
    c = confusion_at_threshold(y_true, scores, threshold)
    p = c["tp"] / (c["tp"] + c["fp"]) if (c["tp"] + c["fp"]) > 0 else 0.0
    r = c["tp"] / (c["tp"] + c["fn"]) if (c["tp"] + c["fn"]) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def evaluate_step(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    """한 스텝의 모든 지표를 dict 로 반환한다."""
    out: dict[str, float] = {"auroc": auroc(y_true, scores), "aupr": aupr(y_true, scores), "threshold": threshold}
    out.update(precision_recall_f1(y_true, scores, threshold))
    out.update(confusion_at_threshold(y_true, scores, threshold))
    return out


def evaluate_all_steps(results: dict[int, dict]) -> pd.DataFrame:
    """스텝별 지표를 표(DataFrame)로 정리한다."""
    rows = [{"step": k, **v} for k, v in results.items()]
    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
