"""학습/검증 분할.

FAIL 이 극히 적으므로 학습에는 PASS 만 사용한다 (get_step_paths가 이미 PASS/FAIL을 분리해 줌).
여기서는 PASS 경로 목록을 train/val로만 나눈다.
"""
from __future__ import annotations

import random
from pathlib import Path


def split_pass_paths(
    pass_paths: list[Path], val_ratio: float, seed: int
) -> tuple[list[Path], list[Path]]:
    """PASS 경로 목록을 (train, val) 로 섞어서 나눈다.

    제품 수가 극소(2~9개)인 기종은 val이 0이 될 수 있다 — 이 경우 threshold는
    train 스코어 자체로 산출하게 되므로 신뢰도가 낮음을 결과에 표시해야 한다.
    """
    paths = list(pass_paths)
    rng = random.Random(seed)
    rng.shuffle(paths)

    if len(paths) <= 1:
        return paths, []

    n_val = int(round(len(paths) * val_ratio))
    n_val = min(max(n_val, 0), len(paths) - 1)  # 최소 1개는 train에 남긴다
    val = paths[:n_val]
    train = paths[n_val:]
    return train, val
