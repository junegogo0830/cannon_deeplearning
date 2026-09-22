"""설정(config.yaml) 로딩 유틸."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """YAML 설정 파일을 읽어 dict로 반환한다.

    Args:
        path: config.yaml 경로.

    Returns:
        설정 dict.
    """
    # TODO: yaml.safe_load 로 읽기, 파일 없을 때 명확한 에러
    raise NotImplementedError


def get_step_config(cfg: dict[str, Any], machine_type: str, step_id: int) -> dict[str, Any]:
    """특정 기종·스텝의 설정(roi, threshold 등)을 반환한다.

    Args:
        cfg: load_config 결과.
        machine_type: 기종 이름 (예: "MODEL_A").
        step_id: 스텝 번호 (1~15).
    """
    # TODO: cfg["machine_types"][machine_type]["steps"] 에서 id 매칭
    raise NotImplementedError
