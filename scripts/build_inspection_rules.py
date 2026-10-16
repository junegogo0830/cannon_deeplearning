"""configs/inspection_rules.yaml 생성기.

195개 (기종,스텝) 셀을 손으로 쳐서 YAML을 만들면 오타·누락 위험이 크다. 대신 여기서
"실제로 검증한 것"만 하드코딩 테이블로 관리하고(3029C003AA 15스텝 전체의 체크포인트
매핑 — report.md §3.2.1, 3029C009AA step3/8 — §3.2.2), 나머지 11개 기종·180개 스텝은
전부 "검증 안 됨 + 범용 PaDiM만 적용"이라는 정직한 기본값으로 채운다. 체크포인트를
모르는 스텝에 그럴듯한 부품명을 지어내지 않는다.

실행: python -m scripts.build_inspection_rules
"""
from __future__ import annotations

import yaml

from src.utils.config import load_config

CONFIG_PATH = "configs/config_cnn.yaml"
OUT_PATH = "configs/inspection_rules.yaml"

# 기종 -> 어느 run(results/<run>/)의 아티팩트를 쓸지. §3.8/§3.9 참고:
# 큰 6개 기종은 all13(독립 홀드아웃)이 유일하게 검증된 것. 작은 7개 기종은 all13의
# "val=홀드아웃 재사용"보다 all13_kfold(out-of-fold)가 덜 낙관적이라고 report.md §3.9에서
# 결론 낸 바를 그대로 따른다 — 둘 다 있는 기종에서는 kfold를 우선 사용한다.
SMALL_MACHINES_USE_KFOLD = {
    "3029C004AA", "3029C005AA", "3029C006AA",
    "3029C010AA", "3030C002AA", "3030C004AA", "3031C001AA",
}

# 3029C003AA 15스텝 체크포인트 매핑 — report.md §3.2.1 표 그대로.
MACHINE_003AA_STEPS = {
    0: {"checkpoint": None, "part_name": "바코드 명판(체크포인트 없음)",
        "reference_image": None, "inspection_types": ["generic"], "mapping_verified": True},
    1: {"checkpoint": "①", "part_name": "C551 토너라벨", "reference_image": "data/processed/xlsx_refs/row11_05.jpg",
        "inspection_types": ["content"], "mapping_verified": True},
    2: {"checkpoint": "②", "part_name": "B793 Canon각인", "reference_image": "data/processed/xlsx_refs/row34_30.png",
        "inspection_types": ["existence"], "mapping_verified": True,
        "modules": ["A"], "module_validated": True,
        "model_artifact_a": "results/presence_artifacts/3029C003AA/step_2",
        "validated_cases": ["2EQ17039(Canon 각인 누락, 검출 성공)"]},
    3: {"checkpoint": "③", "part_name": "6416 2206N명판", "reference_image": "data/processed/xlsx_refs/row26_28.png",
        "inspection_types": ["existence", "content"], "mapping_verified": True,
        "detail": "체크포인트는 확인됐으나 이 기종(003AA) step3에는 실제 NG가 없어 모듈 검증은 못 했다 (009AA step3에서 검증, 아래 참고)"},
    4: {"checkpoint": "④", "part_name": "3102 현상가압라벨 / 3699 레버피스",
        "reference_image": "data/processed/xlsx_refs/row13_14.jpg / data/processed/xlsx_refs/row20_17.jpg",
        "inspection_types": ["existence"], "mapping_verified": True},
    5: {"checkpoint": "⑤", "part_name": "1525 라벨", "reference_image": "data/processed/xlsx_refs/row12_06.jpg",
        "inspection_types": ["existence"], "mapping_verified": True},
    6: {"checkpoint": "⑥", "part_name": "2917 코마", "reference_image": "data/processed/xlsx_refs/row21_16.jpg",
        "inspection_types": ["shape_position"], "mapping_verified": True,
        "detail": "C(형상·위치) 대상 — PaDiM·엣지밀도 둘 다 시도, 실패(report.md §3.3.5, §3.3.7). "
                  "ROI는 정상 12장 중 10장에서 검증됐으나 신호 자체가 없음(원인 불명으로 남김)."},
    7: {"checkpoint": "⑦", "part_name": "1525 라벨(2번째 검사)", "reference_image": "data/processed/xlsx_refs/row12_07.jpg",
        "inspection_types": ["existence"], "mapping_verified": True,
        "detail": "실제 NG 1건(2EQ17023)은 장면 자체가 다른 사진으로 판단돼(§3.2.5) 비교 대상에서 제외했다."},
    8: {"checkpoint": "⑧", "part_name": "0000 모듈러카바", "reference_image": "data/processed/xlsx_refs/row36_31.jpg",
        "inspection_types": ["existence"], "mapping_verified": True},
    9: {"checkpoint": "⑨", "part_name": "3780 전원스위치", "reference_image": "data/processed/xlsx_refs/row24_21.jpg",
        "inspection_types": ["existence"], "mapping_verified": True,
        "modules": ["D"], "module_validated": True,
        "detail": "전용 모듈(A/B) 없이 범용 PaDiM(D)만으로 실제 NG 2/2 검출 성공(report.md §3.2.1). "
                  "검증된 범용 판정 사례로 별도 표시."},
    10: {"checkpoint": "⑩", "part_name": "V242 카바라이트", "reference_image": "data/processed/xlsx_refs/row18_12.jpg",
         "inspection_types": ["existence"], "mapping_verified": True},
    11: {"checkpoint": "⑪", "part_name": "V242 카바라이트(2번째)", "reference_image": "data/processed/xlsx_refs/row18_12.jpg",
         "inspection_types": ["existence"], "mapping_verified": True},
    12: {"checkpoint": "⑫", "part_name": "3779 압판스토퍼", "reference_image": "data/processed/xlsx_refs/row23_19.jpg",
         "inspection_types": ["existence"], "mapping_verified": True},
    13: {"checkpoint": None, "part_name": "일람표 대응 불명", "reference_image": None,
         "inspection_types": ["generic"], "mapping_verified": False},
    14: {"checkpoint": "⑭", "part_name": "3779 압판스토퍼(2번째 검사)", "reference_image": "data/processed/xlsx_refs/row23_18.jpg",
         "inspection_types": ["shape_position"], "mapping_verified": True,
         "detail": "C(형상·위치) 대상 — 촬영 자세 변화로 ROI 자체가 불안정(report.md §3.3.2, §3.3.7). "
                   "자세 군집화도 각도 측정 도구 불안정으로 착수 단계에서 중단."},
}

# 3029C009AA — N=step_N 매핑 자체는 미검증(§3.2.2)이지만, step3은 실제 NG 3건으로
# A+B 조합이 직접 검증됐다. step8은 원인불명 NG 1건, 미검출.
MACHINE_009AA_STEPS = {
    3: {"checkpoint": None, "part_name": "2206N 계열 명판(추정, 003AA ③과 유사)",
        "reference_image": None, "inspection_types": ["existence", "content"], "mapping_verified": False,
        "modules": ["A", "B"], "module_validated": True,
        "model_artifact_a": "results/presence_artifacts/3029C009AA/step_3",
        "model_artifact_b": "results/text_artifacts/3029C009AA/step_3",
        "expected_text": "2206N",
        "validated_cases": [
            "2EX36613(명판 누락, A로 검출)",
            "2EX36661(2206N→2006N 오부착, B로 검출)",
            "2EX36524(N 탈락, B의 문자누락 규칙으로 검출)",
        ]},
    8: {"checkpoint": None, "part_name": "불명", "reference_image": None,
        "inspection_types": ["generic"], "mapping_verified": False,
        "detail": "실제 NG 1건(2EX36527) 존재하나 육안으로도 식별 불가, 미검출(report.md §3.2.2)"},
}


def _default_step_rule() -> dict:
    return {
        "checkpoint": None, "part_name": None, "reference_image": None,
        "inspection_types": ["generic"], "mapping_verified": False,
        "modules": ["D"], "module_validated": False,
    }


def _status_note(rule: dict) -> str:
    """mapping_verified/module_validated/modules 최종값을 보고 상태 요약을 만든다.

    개별 셀에 하드코딩한 설명(위 MACHINE_*_STEPS의 'note')과 섞이면 merge 순서에 따라
    내용이 서로 모순될 수 있어(예: module_validated=True인데 '전용 모듈 없음' 문구가 남는
    버그가 실제로 한 번 있었다), 최종 플래그에서 매번 다시 계산한다.
    """
    modules = rule.get("modules", ["D"])
    validated = rule.get("module_validated", False)
    mapped = rule.get("mapping_verified", False)
    if validated and modules != ["D"]:
        return f"전용 모듈({'+'.join(modules)}) 판정이 실제 NG 사례로 검증됨"
    if validated and modules == ["D"]:
        return "전용 모듈 없음 — 범용 PaDiM(D)이 실제 NG 사례로 검증된 예외적 경우"
    if mapped:
        return "체크포인트 매핑은 확인됐으나 전용 모듈 미검증 — 범용 PaDiM(D)만 적용"
    return "일람표 체크포인트 대응 미검증, 전용 모듈 없음 — 범용 PaDiM(D)만 적용"


def build() -> dict:
    cfg = load_config(CONFIG_PATH)
    machines_cfg = cfg["machine_types"]
    out = {"machines": {}}

    for machine, info in machines_cfg.items():
        n_steps = info["n_steps"]
        run_dir = "all13_kfold" if machine in SMALL_MACHINES_USE_KFOLD else "all13"
        steps_out = {}
        for step in range(n_steps):
            rule = _default_step_rule()
            if machine == "3029C003AA" and step in MACHINE_003AA_STEPS:
                rule.update(MACHINE_003AA_STEPS[step])
            elif machine == "3029C009AA" and step in MACHINE_009AA_STEPS:
                rule.update(MACHINE_009AA_STEPS[step])

            rule.setdefault("modules", ["D"])
            rule.setdefault("module_validated", False)
            rule["status_note"] = _status_note(rule)
            rule["model_artifact_generic"] = f"results/{run_dir}/{machine}/step_{step}"
            steps_out[step] = rule

        out["machines"][machine] = {
            "run_dir": run_dir, "n_steps": n_steps, "has_real_fail": info.get("has_real_fail", False),
            "steps": steps_out,
        }
    return out


def main() -> None:
    data = build()
    header = (
        "# 자동 생성 파일 — scripts/build_inspection_rules.py 로 재생성한다. 직접 손으로 고치지 말 것.\n"
        "# mapping_verified: 체크포인트 번호와 step_N 대응 자체가 검증됐는가 (§1.2 참고 — 2개 기종만 일부 검증)\n"
        "# module_validated: 전용 모듈(A/B)의 판정이 실제 NG 사례로 검증됐는가\n"
        "# status_note: 위 두 플래그로부터 자동 계산한 요약(항상 최신 플래그와 일치)\n"
        "# detail: 셀별로 수기 작성한 추가 설명(선택, status_note를 대체하지 않음)\n"
        "# module_validated=False인 셀은 전부 범용 PaDiM(D)만 적용하고, 그 사실을 판정 결과에 그대로 남긴다.\n\n"
    )
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(header)
        yaml.dump(data, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
    print(f"{OUT_PATH} 생성 완료 (기종 {len(data['machines'])}개)")


if __name__ == "__main__":
    main()
