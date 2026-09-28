"""이미지/라벨 로딩.

라벨 CSV는 없다. FAIL 라벨은 data/raw/<기종>/<제품>/NG/ 폴더의 존재·파일명으로만 확인 가능하며,
scripts/build_inventory.py 가 이를 스캔해 ng_events.csv 로 만들어 둔다 (data/processed/eda/).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd


def read_image(path: str | Path, grayscale: bool = False) -> np.ndarray:
    """이미지를 ndarray 로 읽는다 (한글/유니코드 경로 안전)."""
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flag)
    if img is None:
        raise ValueError(f"이미지를 읽을 수 없습니다: {path}")
    return img


def load_ng_events(cfg: dict[str, Any]) -> pd.DataFrame:
    """FAIL 라벨의 유일한 소스. scripts/build_inventory.py 로 미리 생성해 둔 CSV를 읽는다."""
    df = pd.read_csv(cfg["paths"]["ng_events_csv"])
    df["step"] = df["step"].astype(int)
    return df


def list_images(raw_dir: str | Path, machine_type: str, extensions: list[str]) -> list[Path]:
    """기종 폴더 아래 모든 step_*.jpg 경로 목록 (정렬됨)."""
    root = Path(raw_dir) / machine_type
    paths: list[Path] = []
    for ext in extensions:
        paths += list(root.glob(f"*/step_*{ext}"))
    return sorted(paths)


def get_step_paths(cfg: dict[str, Any], machine_type: str, step: int) -> tuple[list[Path], list[Path]]:
    """(기종, 스텝) 하나의 PASS 이미지 경로 목록과 FAIL(NG) 이미지 경로 목록을 반환한다.

    PASS: data/raw/<기종>/<제품>/step_{step}.jpg 전부 (재검사 후 최종본이므로 PASS로 취급).
    FAIL: ng_events.csv 에 그 (기종, 제품, 스텝)이 기록돼 있으면 NG/ 폴더 안의 원본 불량 샷.
    """
    raw_dir = Path(cfg["paths"]["raw_dir"])
    root = raw_dir / machine_type
    pass_paths = sorted(root.glob(f"*/step_{step}.jpg"))

    ng = load_ng_events(cfg)
    ng_sub = ng[(ng.machine_type == machine_type) & (ng.step == step)]
    fail_paths = [root / row.product_id / "NG" / row.filename for row in ng_sub.itertuples()]
    return pass_paths, fail_paths


class StepDataset:
    """(기종, 스텝) 하나에 대한 이미지 + 라벨 묶음 (torch Dataset과 유사한 인터페이스)."""

    def __init__(self, cfg: dict[str, Any], machine_type: str, step_id: int, split: str) -> None:
        """
        Args:
            split: "train" | "val" | "fail" — train/val은 PASS만, fail은 실제 FAIL(NG) 이미지.
        """
        from src.data.split import split_pass_paths

        pass_paths, fail_paths = get_step_paths(cfg, machine_type, step_id)
        train_paths, val_paths = split_pass_paths(
            pass_paths, cfg["data"]["val_ratio"], cfg["project"]["seed"]
        )
        self.paths = {"train": train_paths, "val": val_paths, "fail": fail_paths}[split]
        self.label = 0 if split in ("train", "val") else 1

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, int]:
        return read_image(self.paths[idx]), self.label
