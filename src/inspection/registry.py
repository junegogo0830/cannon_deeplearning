"""configs/inspection_rules.yaml 로더 — 기종×스텝별 검사 규칙 조회.

이 규칙표가 "규칙 선택" 단계의 전부다: 어떤 모듈(A/B/C/D)을 쓸지, 그 모듈의 판정이
실제 NG로 검증된 것인지(module_validated), 체크포인트 번호 대응 자체가 검증된 것인지
(mapping_verified)를 기종×스텝마다 조회한다. 파일은 scripts/build_inspection_rules.py로
생성한 것이고, 손으로 고치지 않는다(재생성하면 덮어써진다).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

RULES_PATH = Path("configs/inspection_rules.yaml")


class InspectionRegistry:
    def __init__(self, path: Path | str = RULES_PATH) -> None:
        with open(path, encoding="utf-8") as f:
            self._data = yaml.safe_load(f)
        self._machines = self._data["machines"]

    def machines(self) -> list[str]:
        return list(self._machines.keys())

    def n_steps(self, machine: str) -> int:
        return self._machines[machine]["n_steps"]

    def run_dir(self, machine: str) -> str:
        """이 기종의 범용(D) 아티팩트가 어느 results/<run>/ 에 있는지."""
        return self._machines[machine]["run_dir"]

    def has_real_fail(self, machine: str) -> bool:
        return self._machines[machine].get("has_real_fail", False)

    def get_rule(self, machine: str, step: int) -> dict[str, Any]:
        if machine not in self._machines:
            raise KeyError(f"등록되지 않은 기종: {machine}")
        steps = self._machines[machine]["steps"]
        if step not in steps:
            raise KeyError(f"{machine}에 스텝 {step}이 없습니다 (n_steps={self.n_steps(machine)})")
        return steps[step]

    def steps(self, machine: str) -> list[int]:
        return sorted(self._machines[machine]["steps"].keys())
