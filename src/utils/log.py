"""로거 설정."""
from __future__ import annotations

import logging


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """콘솔(+선택적 파일) 로거를 반환한다."""
    # TODO: 포맷/레벨 설정, 중복 핸들러 방지
    raise NotImplementedError
