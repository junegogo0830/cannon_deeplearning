"""파일 입출력 유틸 (JSON 저장/로드, 실행 결과 폴더 관리)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def save_json(obj: Any, path: str | Path) -> None:
    """dict/list(JSON 직렬화 가능한 값들) 를 JSON 으로 저장한다 (부모 폴더 자동 생성)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    """JSON 파일을 읽어 반환한다."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_run_dir(results_dir: str | Path, run_name: str) -> Path:
    """results/<run_name>/ 실행 결과 폴더를 만들고 경로를 반환한다."""
    run_dir = Path(results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir
