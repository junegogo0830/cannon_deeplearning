"""평가 결과를 reports/ · results/ 에 저장."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def save_score_table(df: pd.DataFrame, path: str | Path) -> None:
    """이미지별 스텝 스코어/판정 결과를 CSV 로 저장한다."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_metrics_report(metrics_df: pd.DataFrame, out_dir: str | Path) -> None:
    """스텝별 지표 표를 CSV(+텍스트 요약)로 저장한다."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(out_dir / "metrics.csv", index=False)
    with open(out_dir / "metrics.txt", "w", encoding="utf-8") as f:
        f.write(metrics_df.to_string(index=False))
