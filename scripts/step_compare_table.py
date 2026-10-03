"""기종 × 스텝 비교표: (1) 사진 판독 (2) 일람표 항목 (3) 기종 간 유사도 (4) 판정.

판독(visual)은 scripts/step_audit.py 가 만든 기종별 contact sheet 를 육안으로 읽은 결과다 (추정).
판정 규칙은 아래 RULES 에 명시한다. 규칙은 "근거가 되는 문서/사진이 있는가"만 본다.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path("data/processed/step_audit")

# 판독 코드: BC 바코드명판 / TL 토너라벨(C-EXV|NPG) / CN Canon 인쇄 / MD 모델명 명판 /
#           DR 파란 레버+드럼라벨 / BP 파란 부품+▲▲▲ 라벨 / PN 사각 패널 / PW 전원 스위치 /
#           PO 포트+나사 / CV 포트 커버 패널 / NC 패널 노치(이음새) / CR 캐리지·내부 /
#           CA 경고(CAUTION·손끼임) 라벨 / DK 어두운 면(구멍·A4/A5 라벨) / ETC 기타
V = {
 "3029C003AA": [("BC","바코드명판 2206N EU"),("TL","C-EXV 토너라벨"),("CN","Canon 인쇄"),("MD","2206N 명판"),
   ("DR","파란 레버+C-EXV42 드럼라벨"),("BP","파란 부품+▲▲▲"),("BP","파란 부품(하단)"),("BP","파란 부품+▲▲▲+빗살"),
   ("PN","사각 패널"),("PW","전원 스위치"),("PO","USB/LAN 포트+나사"),("CV","포트 커버 패널+USB"),
   ("NC","패널 모서리+나사"),("CR","캐리지·기판 근처"),("NC","패널 노치(이음새)+나사")],
 "3029C004AA": [("BC","바코드명판 2206iF EU"),("TL","C-EXV 토너라벨"),("CN","Canon 인쇄"),("MD","2206iF 명판"),
   ("DR","파란 레버+C-EXV42 드럼라벨"),("BP","파란 부품+▲▲▲"),("BP","파란 부품(하단)"),("BP","파란 부품+▲▲▲"),
   ("PN","사각 패널"),("PW","전원 스위치"),("PO","포트+나사"),("CV","포트 커버 패널+USB"),
   ("CA","다국어 안내 스티커(손 그림)"),("CR","내부 기구·조명"),("DK","어두운 면(구멍/슬롯)"),
   ("CA","CAUTION 라벨(손끼임)"),("CR","내부 캐리지"),("CA","CAUTION 라벨(손끼임)")],
 "3029C005AA": [("BC","바코드명판 2206N"),("TL","NPG 토너라벨"),("CN","Canon 인쇄"),("MD","2206N 명판"),
   ("DR","파란 레버+드럼라벨(NPG계열)"),("BP","파란 부품+▲▲▲"),("BP","파란 부품(하단)"),("BP","파란 부품+▲▲▲"),
   ("PN","사각 패널"),("PW","전원 스위치"),("PO","포트+나사"),("CV","커버 패널+USB"),("CR","내부 캐리지")],
 "3029C006AA": [("BC","바코드명판 2206N IND"),("TL","NPG 토너라벨"),("CN","Canon 인쇄"),("MD","2206N 명판"),
   ("DR","파란 레버+NPG-59 드럼라벨"),("BP","파란 부품+▲▲▲"),("BP","파란 부품(하단)"),("BP","파란 부품+▲▲▲"),
   ("PN","사각 패널"),("PW","전원 스위치"),("PO","포트+나사"),("CV","커버 패널"),("CR","캐리지")],
 "3029C009AA": [("BC","바코드명판 2206N CN"),("TL","NPG 토너라벨"),("CN","Canon 인쇄"),("MD","2206N 명판"),
   ("DR","파란 레버+NPG-59 드럼라벨"),("BP","파란 부품+▲▲▲"),("BP","파란 부품(하단)"),("BP","파란 부품+▲▲▲"),
   ("PN","사각 패널"),("PW","전원 스위치"),("PO","포트+나사"),("CV","커버 패널+USB"),("NC","패널 노치+나사"),
   ("CR","캐리지"),("NC","패널 노치+나사")],
 "3029C010AA": [("BC","바코드명판 2206i CN"),("TL","NPG 토너라벨"),("CN","Canon 인쇄"),("MD","2206i 명판"),
   ("DR","파란 레버+NPG-59 드럼라벨"),("BP","파란 부품+▲▲▲"),("BP","파란 부품(하단)"),("BP","파란 부품+▲▲▲"),
   ("ETC","파란 레버 트레이"),("BP","내부 ▲▲▲ 라벨"),("PN","사각 패널"),("PW","전원 스위치"),("PO","포트+나사"),
   ("CV","USB 포트 패널"),("CA","트레이+손끼임 아이콘"),("DK","어두운 면(▲▲▲ 일부)"),("DK","어두운 면+A4/A5 라벨"),
   ("CA","CAUTION 라벨(손끼임)"),("CR","캐리지"),("CA","CAUTION 라벨(손끼임)")],
 "3030C001AA": [("BC","바코드명판 3030C001"),("TL","C-EXV 토너라벨"),("CN","Canon 인쇄"),("MD","2206 명판"),
   ("DR","파란 레버+드럼(글자 잘림)"),("BP","파란 부품+▲▲▲"),("BP","파란 부품(하단)"),("BP","파란 부품+▲▲▲"),
   ("PN","사각 패널"),("PW","전원 스위치"),("PO","USB+나사"),("CV","USB 포트 커버"),("NC","패널 노치 이음새"),
   ("CR","캐리지"),("NC","패널 노치 이음새")],
 "3030C002AA": [("BC","바코드명판 2206 230V"),("TL","NPG 토너라벨"),("CN","Canon 인쇄"),("MD","2206 명판"),
   ("DR","파란 레버+NPG-59 드럼라벨"),("BP","파란 부품+▲▲▲"),("BP","파란 부품(하단)"),("BP","파란 부품+▲▲▲"),
   ("PN","사각 패널"),("PW","전원 스위치"),("PO","USB+나사"),("CV","USB 포트"),("NC","패널 노치 이음새"),
   ("CR","캐리지"),("NC","패널 노치 이음새")],
 "3030C003AA": [("BC","바코드명판 2206 IND"),("TL","NPG 토너라벨"),("CN","Canon 인쇄"),("MD","2206 명판"),
   ("DR","파란 레버+NPG-59 드럼라벨"),("BP","파란 부품+▲▲▲"),("BP","파란 부품(하단)"),("BP","파란 부품+▲▲▲"),
   ("PN","사각 패널"),("PW","전원 스위치"),("PO","포트+나사"),("CV","USB 포트"),("NC","패널 노치 이음새"),
   ("CR","캐리지"),("NC","패널 노치 이음새")],
 "3030C004AA": [("BC","바코드명판 2206L CN"),("TL","NPG 토너라벨"),("CN","Canon 인쇄"),("MD","2206L 명판"),
   ("DR","파란 레버+NPG-59 드럼라벨"),("BP","파란 부품+▲▲▲"),("BP","파란 부품(하단)"),("BP","파란 부품+▲▲▲"),
   ("PN","사각 패널"),("PW","전원 스위치"),("PO","포트+나사"),("CV","USB 포트"),("NC","패널 노치 이음새"),
   ("CR","캐리지"),("NC","패널 노치 이음새")],
 "3031C001AA": [("BC","바코드명판 2006N TW"),("TL","NPG 토너라벨"),("CN","Canon 인쇄"),("MD","2006N 명판"),
   ("DR","파란 레버+NPG-59 드럼라벨"),("BP","파란 부품+▲▲▲"),("BP","파란 부품(하단)"),("BP","파란 부품+▲▲▲"),
   ("PN","사각 패널"),("PW","전원 스위치"),("PO","포트+나사"),("CV","USB 커버 패널"),("NC","패널 노치+나사"),
   ("CR","캐리지"),("NC","패널 노치+나사")],
 "3031C002AA": [("BC","바코드명판 2006N"),("TL","NPG 토너라벨"),("CN","Canon 인쇄"),("MD","2006N 명판"),
   ("DR","파란 레버+NPG-59 드럼라벨"),("BP","파란 부품+▲▲▲"),("BP","파란 부품(하단)"),("BP","파란 부품+▲▲▲"),
   ("PN","사각 패널"),("PW","전원 스위치"),("PO","포트+나사"),("CV","포트+USB"),("CR","캐리지")],
 "3031C003AA": [("BC","바코드명판 2006N IND"),("TL","NPG 토너라벨"),("CN","Canon 인쇄"),("MD","2006N 명판"),
   ("DR","파란 레버+NPG-59 드럼라벨"),("BP","파란 부품+▲▲▲"),("BP","파란 부품(하단)"),("BP","파란 부품+▲▲▲"),
   ("PN","사각 패널"),("PW","전원 스위치"),("PO","포트+나사"),("CV","USB 커버 패널"),("CR","캐리지")],
}

# 판정 규칙: (판독 코드 집합, 일람표 항목 키워드) → 대응 가능
RULES = [
    ({"CN"}, ["인쇄", "각인"]),
    ({"MD"}, ["명칭"]),
    ({"TL"}, ["토너"]),
    ({"PW"}, ["스위치"]),
    ({"BP"}, ["용지 적재"]),
    ({"DR"}, ["레버", "현상 가압"]),
]


def judge(code: str, item: str | None, visual_text: str) -> str:
    if not item:
        return "문서 근거 없음" if code not in {"BC"} else "문서 근거 없음(바코드는 별도 시트)"
    # 모델명 불일치 점검: 일람표 항목의 모델명 vs 판독 모델명
    if code == "MD":
        for model in ["2206N", "2206iF", "2206L", "2206i", "2206F", "2206", "2006N"]:
            if model in item:
                return "대응 가능" if model in visual_text else f"불일치 의심 (일람표 {model})"
    for codes, kws in RULES:
        if code in codes and any(k in item for k in kws):
            return "대응 가능"
    return "대응 불명"


def main() -> None:
    cp = pd.read_csv("data/processed/eda/xlsx_checkpoint_map.csv")
    cp["item"] = cp.part_name.astype(str).str.replace("\n", " ").str.strip() + " (" + cp.guarantee_item.astype(str).str.replace("\n", " ").str.strip() + ")"
    items = cp.groupby(["machine_type", "checkpoint"])["item"].apply(lambda s: " / ".join(sorted(set(s)))).to_dict()
    align = pd.read_csv(OUT / "step_alignment_test.csv")
    align_map = {(r.machine, int(r.step)): r for r in align.itertuples()}

    rows = []
    for m, steps in V.items():
        for s, (code, text) in enumerate(steps):
            item = items.get((m, s + 1))  # 가설: 체크포인트 = step+1 (반례 있음, 참고용)
            a = align_map.get((m, s))
            rows.append({
                "machine": m, "step": s, "visual_code": code, "visual_desc": text,
                "xlsx_item_at_step_plus_1": item if item else "(항목 없음)",
                "best_ref_step_3029C003AA": int(a.best_ref_step) if a is not None else None,
                "same_index_as_ref": bool(a.same_index_as_ref) if a is not None else None,
                "judgment": judge(code, item, text),
            })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "step_compare_table.csv", index=False, encoding="utf-8-sig")
    print(df.judgment.value_counts().to_string())
    print(df.groupby("machine").judgment.value_counts().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
