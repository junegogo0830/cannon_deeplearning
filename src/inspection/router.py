"""검사 목적별 모듈 선택 — registry의 규칙을 보고 A/B/C/D 중 무엇을 실행할지 정한다.

정책(섹션 5 요구사항 그대로):
- rule.module_validated가 True면 "검증된 모듈"의 판정을 최종 판정으로 쓴다. 범용 PaDiM(D)은
  참고용으로 같이 기록하되 OR 앙상블로 묶지 않는다 — §3.3에서 patch-kNN∨PaDiM OR 앙상블이
  FPR만 올리고 검출은 못 늘렸던 경험(report.md) 때문에, "여러 모델 중 하나라도 FAIL이면 FAIL"을
  기본 정책으로 쓰지 않는다.
- rule.module_validated가 False면 범용 PaDiM(D)만 돌리고, "전용 검사기 없음"이라는 사실을
  결과에 남긴다.
- A와 B가 모두 지정된 스텝(존재+내용)은 A→B 순서로 실행한다. A가 명백히 FAIL(누락)이면
  B는 건너뛴다(없는 명판에서 글자를 분할하려는 건 의미가 없고, 실제로 분할 실패로 REVIEW만
  쌓인다 — infer_text.py 자체는 이 순서를 모르므로 여기서 라우터가 강제한다).
- C(형상·위치)는 rule.inspection_types에 "shape_position"이 있어도 실행 가능한 전용 판정기가
  없으므로 항상 NOT_APPLICABLE로 보고한다 — 범용 PaDiM(D) 점수가 있어도 그것으로 C를
  "통과했다"고 주장하지 않는다(섹션 5 마지막 문단 요구사항).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from src.inspection import adapters
from src.inspection.artifacts import ArtifactCache
from src.inspection.registry import InspectionRegistry


def run_step(
    image_path: str | Path, machine: str, step: int, registry: InspectionRegistry, cache: ArtifactCache,
) -> dict[str, Any]:
    """(기종,스텝) 하나에 대해 규칙에 맞는 모듈을 실행하고, 실행된 모든 모듈의 결과와
    라우팅에 쓰인 규칙을 함께 반환한다. 최종 PASS/FAIL/REVIEW 결정은 하지 않는다
    (src/inspection/decision.py가 한다) — 라우터는 "무엇을 실행했는가"까지만 책임진다.
    """
    rule = registry.get_rule(machine, step)
    run_dir = registry.run_dir(machine)
    modules = rule.get("modules", ["D"])
    validated = rule.get("module_validated", False)

    module_results: dict[str, dict] = {}

    # 범용 PaDiM은 항상 돌려서 기록한다(참고용) — 단, modules==["D"]이고 validated=True인
    # "검증된 범용" 셀(예: 003AA step9)에서는 이것 자체가 최종 판정의 근거가 된다.
    module_results["D"] = adapters.run_generic(image_path, machine, step, run_dir, cache)

    if "A" in modules:
        module_results["A"] = adapters.run_presence(image_path, machine, step, cache)

    if "B" in modules:
        # A가 명백한 FAIL(누락)이면 B는 건너뛴다 — 없는 명판에서 글자를 찾으려 하지 않는다.
        a_res = module_results.get("A")
        if a_res is not None and a_res["status"] == "FAIL":
            module_results["B"] = {
                "module": "B", "status": "NOT_APPLICABLE", "raw_score": None, "threshold": None,
                "roi": None, "reason": "A(존재검사)에서 이미 FAIL, 문자 검사 생략", "aligned": None,
                "artifact_used": None, "elapsed_s": 0.0,
            }
        else:
            module_results["B"] = adapters.run_text(image_path, machine, step, cache)

    if "C" in modules or "shape_position" in rule.get("inspection_types", []):
        module_results["C"] = {
            "module": "C", "status": "NOT_APPLICABLE", "raw_score": None, "threshold": None,
            "roi": None,
            "reason": "형상·위치 전용 판정기 미구현(report.md §3.3.5, §3.3.7, 두 가지 방법 모두 "
                      "검증 실패). 범용 PaDiM(D) 점수로 C 통과를 대신 주장하지 않음",
            "aligned": None, "artifact_used": None, "elapsed_s": 0.0,
        }

    return {
        "machine": machine, "step": step, "rule": rule, "run_dir": run_dir,
        "module_validated": validated, "modules_executed": list(module_results.keys()),
        "results": module_results,
    }
