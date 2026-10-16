"""모듈별 결과(router.run_step의 출력) → 스텝 최종 판정, 스텝별 판정 → 제품 최종 판정.

최종 판정 우선순위(섹션 6 그대로):
  1. 신뢰할 수 있는 전용 검사에서 명백한 FAIL이 확인되면 FAIL
  2. 검사 자체가 불가능하거나 필수 모듈이 불확실하면 REVIEW
  3. 해당 스텝에 필요한 검사가 모두 유효하게 통과하면 PASS
  4. 전용 검사기가 없어 범용 모델만 사용한 경우 그 사실을 기록
서로 다른 모듈의 raw_score는 절대 섞어서 비교/평균하지 않는다 — 최종 판정은 status 체인만 본다.
"""
from __future__ import annotations

from typing import Any

PASS, FAIL, REVIEW = "PASS", "FAIL", "REVIEW"


def aggregate_step(router_result: dict[str, Any]) -> dict[str, Any]:
    rule = router_result["rule"]
    results = router_result["results"]
    validated = router_result["module_validated"]
    modules = [m for m in rule.get("modules", ["D"]) if m in results]

    deciding: dict | None = None
    reason: str | None = None

    if validated and modules:
        status = PASS
        for m in modules:
            r = results[m]
            if r["status"] == FAIL:
                status, deciding, reason = FAIL, r, r["reason"]
                break
            if r["status"] == REVIEW:
                status, deciding, reason = REVIEW, r, r["reason"]
                break
            if r["status"] == "NOT_APPLICABLE":
                status, deciding = REVIEW, r
                reason = f"{m} 아티팩트를 찾을 수 없음(검증된 모듈로 등록돼 있으나 실행 불가, 사람 확인 필요)"
                break
            deciding = r  # PASS면 체인을 계속 진행하되, 대표값 후보로 남겨둔다
        source = f"전용 모듈({'+'.join(modules)}), 실제 NG 사례로 검증됨"
    else:
        d = results.get("D")
        if d is None or d["status"] == "NOT_APPLICABLE":
            status, deciding, reason = REVIEW, d, (d or {}).get("reason", "범용 모델 아티팩트 없음")
        elif d["status"] == REVIEW:
            status, deciding, reason = REVIEW, d, d["reason"]
        else:
            status, deciding, reason = d["status"], d, d["reason"]
        source = "범용 PaDiM(D), 전용 검사기 미검증(참고: report.md §3.8/§6.2 전역 FPR 범위 적용)"

    return {
        "machine": router_result["machine"], "step": router_result["step"],
        "status": status, "reason": reason, "decision_source": source,
        "representative_module": deciding["module"] if deciding else None,
        "representative_score": deciding["raw_score"] if deciding else None,
        "representative_threshold": deciding["threshold"] if deciding else None,
        "modules_executed": router_result["modules_executed"],
        "module_results": results,
        "mapping_verified": rule.get("mapping_verified", False),
        "module_validated": validated,
        "checkpoint": rule.get("checkpoint"), "part_name": rule.get("part_name"),
    }


def aggregate_product(step_decisions: dict[int, dict[str, Any]]) -> dict[str, Any]:
    """§7: 하나라도 FAIL이면 제품 FAIL, FAIL 없이 REVIEW가 있으면 제품 REVIEW,
    모두 유효하게 PASS일 때만 제품 PASS. 스텝을 조용히 누락하지 않는다."""
    statuses = [d["status"] for d in step_decisions.values()]
    if not statuses:
        return {"product_status": REVIEW, "n_steps": 0, "n_fail": 0, "n_review": 0, "n_pass": 0,
                "reason": "검사된 스텝이 없음"}
    if any(s == FAIL for s in statuses):
        final = FAIL
    elif any(s == REVIEW for s in statuses):
        final = REVIEW
    elif all(s == PASS for s in statuses):
        final = PASS
    else:
        final = REVIEW  # 알 수 없는 status가 섞인 경우 안전 쪽(REVIEW)으로
    return {
        "product_status": final, "n_steps": len(statuses),
        "n_fail": statuses.count(FAIL), "n_review": statuses.count(REVIEW), "n_pass": statuses.count(PASS),
    }
