"""로거 설정."""
from __future__ import annotations

import logging
from pathlib import Path


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """콘솔(+선택적 파일) 로거를 반환한다. 같은 name으로 다시 불러도 핸들러가 중복되지 않는다."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger
