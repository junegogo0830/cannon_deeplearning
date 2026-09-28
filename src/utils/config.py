"""설정(config.yaml) 로딩 유틸."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """YAML 설정 파일을 읽어 dict로 반환한다."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"config 파일을 찾을 수 없습니다: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_machine_config(cfg: dict[str, Any], machine_type: str) -> dict[str, Any]:
    """config.machine_types[machine_type] (n_steps, image_glob, has_real_fail)."""
    try:
        return cfg["machine_types"][machine_type]
    except KeyError as e:
        raise KeyError(f"config.machine_types 에 '{machine_type}' 기종이 없습니다.") from e


def get_step_ids(cfg: dict[str, Any], machine_type: str) -> list[int]:
    """0-indexed 스텝 id 목록 (n_steps 만큼, step_0.jpg ~ step_{n-1}.jpg 에 대응)."""
    n_steps = get_machine_config(cfg, machine_type)["n_steps"]
    return list(range(n_steps))
