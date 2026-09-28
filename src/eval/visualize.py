"""시각화 (matplotlib)."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 서버/스크립트 환경에서 창 없이 파일로만 저장
import matplotlib.pyplot as plt
import numpy as np


def plot_score_distribution(
    normal_scores: np.ndarray,
    fail_scores: np.ndarray | None,
    threshold: float,
    save_path: str | Path,
) -> None:
    """정상/FAIL 스코어 히스토그램 + threshold 선."""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(normal_scores, bins=30, alpha=0.6, label=f"PASS (n={len(normal_scores)})", color="#4C72B0")
    if fail_scores is not None and len(fail_scores) > 0:
        ax.hist(fail_scores, bins=min(30, max(1, len(fail_scores))), alpha=0.7,
                 label=f"FAIL (n={len(fail_scores)})", color="#C44E52")
    ax.axvline(threshold, color="black", linestyle="--", label=f"threshold={threshold:.3f}")
    ax.set_xlabel("anomaly score")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_roc_pr(y_true: np.ndarray, scores: np.ndarray, save_path: str | Path) -> None:
    """ROC / PR 곡선. 클래스가 하나뿐이면(FAIL 없음) 건너뛴다."""
    if len(np.unique(y_true)) < 2:
        return
    from sklearn.metrics import precision_recall_curve, roc_curve

    fpr, tpr, _ = roc_curve(y_true, scores)
    prec, rec, _ = precision_recall_curve(y_true, scores)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(fpr, tpr)
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set_xlabel("FPR")
    axes[0].set_ylabel("TPR")
    axes[0].set_title("ROC")

    axes[1].plot(rec, prec)
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("PR")

    fig.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, save_path: str | Path) -> None:
    """원본 위에 이상 히트맵을 덧씌워 저장한다."""
    import cv2

    hm = heatmap.astype(np.float32)
    hm = (hm - hm.min()) / (hm.max() - hm.min() + 1e-8)
    hm_color = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    base = image if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base, 0.6, hm_color, 0.4, 0)
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imencode(".jpg", overlay)[1].tofile(str(save_path))
