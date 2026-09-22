"""이미지/라벨 로딩."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def load_labels(labels_csv: str | Path) -> pd.DataFrame:
    """라벨 CSV를 읽는다.

    기대 컬럼: image_id, machine_type, step, label (PASS/FAIL).
    """
    # TODO: 컬럼 검증, label 을 0(PASS)/1(FAIL) 로 정규화한 컬럼 추가
    raise NotImplementedError


def list_images(raw_dir: str | Path, machine_type: str, extensions: list[str]) -> list[Path]:
    """기종별 원본 이미지 경로 목록을 반환한다."""
    # TODO: raw_dir/<machine_type> 하위 탐색, 정렬해서 반환
    raise NotImplementedError


def read_image(path: str | Path, grayscale: bool = False) -> np.ndarray:
    """이미지를 ndarray(H, W, C) 로 읽는다."""
    # TODO: cv2.imread (한글 경로 주의: np.fromfile + cv2.imdecode)
    raise NotImplementedError


class StepDataset:
    """(기종, 스텝) 하나에 대한 이미지 + 라벨 묶음.

    torch.utils.data.Dataset 호환(__len__, __getitem__)을 목표로 한다.
    """

    def __init__(self, cfg: dict[str, Any], machine_type: str, step_id: int, split: str) -> None:
        """
        Args:
            cfg: load_config 결과.
            machine_type: 기종 이름.
            step_id: 스텝 번호 (1~15).
            split: "train" | "val" | "test".
        """
        # TODO: 라벨 로드 → 해당 기종/스텝 필터 → split 적용
        raise NotImplementedError

    def __len__(self) -> int:
        # TODO
        raise NotImplementedError

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        """(전처리된 ROI 이미지, label) 반환. label: 0=PASS, 1=FAIL."""
        # TODO
        raise NotImplementedError
