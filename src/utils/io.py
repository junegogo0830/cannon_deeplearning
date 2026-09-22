"""파일 입출력 유틸 (모델/임계값/스코어 저장·로드)."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def save_json(obj: Any, path: str | Path) -> None:
    """dict/list 를 JSON 으로 저장한다 (부모 폴더 자동 생성)."""
    # TODO: json.dump(ensure_ascii=False, indent=2)
    raise NotImplementedError


def load_json(path: str | Path) -> Any:
    """JSON 파일을 읽어 반환한다."""
    # TODO
    raise NotImplementedError


def get_run_dir(results_dir: str | Path, run_name: str) -> Path:
    """results/<run_name>/ 실행 결과 폴더를 만들고 경로를 반환한다."""
    # TODO: mkdir(parents=True, exist_ok=True)
    raise NotImplementedError
