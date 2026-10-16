"""모델 아티팩트·CNN 백본 캐시 — 같은 프로세스 안에서 (기종,스텝) 아티팩트와 백본을
반복해서 디스크/메모리에서 다시 읽지 않도록 한다 (통합 파이프라인 CPU 효율화 요구사항).

제품 하나(15~20스텝)를 검사할 때도, 홀드아웃 수백 장을 평가할 때도 같은 (기종,스텝)
아티팩트가 여러 번 쓰이므로 캐시 효과가 크다. CNN 백본(mobilenet_v3_small)은 범용
PaDiM(D)과 존재검사(A) 둘 다 같은 걸 쓰므로 인스턴스 하나만 만들어 공유한다
(configs/config_cnn.yaml의 cnn_backbone과 scripts/infer_presence.py가 둘 다
"mobilenet_v3_small"을 쓰는 것을 src/pipeline.py, scripts/infer_presence.py에서 확인함).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.pipeline import build_backbone_if_needed, load_step_artifacts
from src.utils.config import load_config


class ArtifactCache:
    def __init__(self, config_path: str = "configs/config_cnn.yaml") -> None:
        self.cfg = load_config(config_path)
        self.backbone = build_backbone_if_needed(self.cfg)  # 프로세스당 한 번만 생성
        self._generic: dict[tuple[str, int, str], Any] = {}
        self._presence: dict[tuple[str, int], Any] = {}
        self._text: dict[tuple[str, int], Any] = {}

    def generic(self, machine: str, step: int, run_dir: str) -> dict | None:
        """범용 PaDiM(D) 아티팩트. 없으면 None(router가 NOT_APPLICABLE로 처리)."""
        key = (machine, step, run_dir)
        if key not in self._generic:
            try:
                self._generic[key] = load_step_artifacts(Path("results") / run_dir, machine, step)
            except FileNotFoundError:
                self._generic[key] = None
        return self._generic[key]

    def presence(self, machine: str, step: int) -> dict | None:
        from scripts.infer_presence import load_presence_artifacts

        key = (machine, step)
        if key not in self._presence:
            try:
                self._presence[key] = load_presence_artifacts(machine, step)
            except FileNotFoundError:
                self._presence[key] = None
        return self._presence[key]

    def text(self, machine: str, step: int) -> dict | None:
        from scripts.infer_text import load_text_artifacts

        key = (machine, step)
        if key not in self._text:
            try:
                self._text[key] = load_text_artifacts(machine, step)
            except FileNotFoundError:
                self._text[key] = None
        return self._text[key]
