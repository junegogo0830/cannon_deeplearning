"""재현성을 위한 시드 고정."""
from __future__ import annotations

import random

import numpy as np


def set_seed(seed: int) -> None:
    """random / numpy / (설치돼 있으면) torch 시드를 고정한다."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass  # torch 미설치 또는 로드 실패(DLL 등) — 이 파이프라인은 torch를 쓰지 않으므로 무시
