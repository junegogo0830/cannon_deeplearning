"""학습/검증/테스트 분할.

FAIL 이 극히 적으므로 학습에는 PASS 만 사용하고,
FAIL 은 검증·테스트(임계값 보정 및 평가)에만 배치하는 것이 기본 전략이다.
"""
from __future__ import annotations

import pandas as pd


def split_normal_only_train(
    df: pd.DataFrame, val_ratio: float, seed: int
) -> dict[str, pd.DataFrame]:
    """PASS → train/val 로 분할하고, FAIL 은 val/test 로 배치한다.

    Args:
        df: load_labels 결과 (특정 기종·스텝으로 필터된 상태).
        val_ratio: PASS 중 검증에 쓸 비율.
        seed: 난수 시드.

    Returns:
        {"train": ..., "val": ..., "test": ...}
    """
    # TODO: FAIL 이 너무 적을 때 val/test 배분 정책 (예: leave-one-out, k-fold)
    raise NotImplementedError


def kfold_splits(df: pd.DataFrame, n_splits: int, seed: int) -> list[dict[str, pd.DataFrame]]:
    """FAIL 이 극소수일 때를 위한 stratified k-fold 분할 목록."""
    # TODO: sklearn StratifiedKFold
    raise NotImplementedError
