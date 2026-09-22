"""평가 결과를 reports/ · results/ 에 저장."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_score_table(df: pd.DataFrame, path: str | Path) -> None:
    """이미지별 스텝 스코어/판정 결과를 CSV 로 저장한다."""
    # TODO
    raise NotImplementedError


def save_metrics_report(metrics_df: pd.DataFrame, out_dir: str | Path) -> None:
    """스텝별 지표 표를 CSV/Markdown 으로 저장한다."""
    # TODO
    raise NotImplementedError
