"""시각화 (matplotlib)."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_score_distribution(
    normal_scores: np.ndarray,
    fail_scores: np.ndarray | None,
    threshold: float,
    save_path: str | Path,
) -> None:
    """정상/FAIL 스코어 히스토그램 + threshold 선."""
    # TODO
    raise NotImplementedError


def plot_roc_pr(y_true: np.ndarray, scores: np.ndarray, save_path: str | Path) -> None:
    """ROC / PR 곡선."""
    # TODO
    raise NotImplementedError


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, save_path: str | Path) -> None:
    """원본 위에 이상 히트맵을 덧씌워 저장한다."""
    # TODO
    raise NotImplementedError
