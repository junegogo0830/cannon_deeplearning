"""기종 × 스텝 감사 보고서(reports/step_audit.md) 생성. 표는 step_compare_table.csv 에서 옴."""
from pathlib import Path

import pandas as pd

df = pd.read_csv("data/processed/step_audit/step_compare_table.csv")
al = pd.read_csv("data/processed/step_audit/step_alignment_test.csv")

lines = []
lines += [
    "# 기종 × 스텝 감사 (일람표 · 사진 · 기종 간 유사도)",
    "",
    "> 재현: `python scripts/step_audit.py` → `python scripts/step_compare_table.py` → `python scripts/_write_step_audit_md.py`",
    "> 사진 판독(visual)은 기종별 contact sheet(`data/processed/step_audit/machine_sheet_<기종>.jpg`)를 육안으로 읽은 **추정**이다.",
    "> 일람표 항목은 xlsx 원본이며, 표의 항목은 **체크포인트 = step+1 가설**로 붙였다. 이 가설은 반례가 있어서(아래 1절) 확정 대응이 아니다.",
    "",
    "## 1. 방법과 한계",
    "",
    "- 판독 코드로 사진을 분류하고, 일람표 항목과 키워드 규칙으로 대응 여부를 판정했다.",
    "  - **대응 가능**: 판독 코드와 항목 키워드가 맞음 (인쇄/각인, 명칭, 토너, 스위치, 용지 적재, 레버·현상 가압, 압판) — 모델명 불일치도 검사",
    "  - **대응 불명**: 항목은 있으나 판독 코드와 키워드가 맞지 않음 → 사람이 확인해야 함",
    "  - **문서 근거 없음**: 그 번호에 일람표 항목이 없음",
    "- 반례(가설 깨짐): step_0은 바코드 명판인데 체크포인트①은 토너 라벨(C551)이다. 3029C003AA의 전원스위치는 xlsx ⑨이고 사진은 step_9다.",
    "- 자동 판정은 키워드 규칙이라 세밀한 대응은 잡지 못한다. 최종 대응은 사람이 확인해야 한다.",
    "",
    "## 2. 전체 판정 요약",
    "",
]
counts = df.judgment.value_counts()
lines += ["| 판정 | 행 수 |", "|---|---|"]
lines += [f"| {k} | {v} |" for k, v in counts.items()]
lines += ["", "기종별:", "", "| 기종 | 스텝 수 | 대응 가능 | 대응 불명 | 문서 근거 없음 |", "|---|---|---|---|---|"]
for m, g in df.groupby("machine"):
    c = g.judgment.value_counts()
    lines.append(f"| {m} | {len(g)} | {c.get('대응 가능', 0)} | {c.get('대응 불명', 0)} | {c.get('문서 근거 없음', 0)} |")

lines += ["", "## 3. 기종 간 유사도: 같은 번호 스텝이 기준 기종(3029C003AA)과 가장 비슷한가", "",
          "| 기종 | 스텝 수 | 같은 번호가 최고 유사 | 비율 |", "|---|---|---|---|"]
for m, g in al.groupby("machine"):
    same = int(g.same_index_as_ref.sum())
    lines.append(f"| {m} | {len(g)} | {same} | {same/len(g):.0%} |")
lines.append(f"| 전체 | {len(al)} | {int(al.same_index_as_ref.sum())} | {al.same_index_as_ref.mean():.0%} |")
lines += ["",
          "> 해석: 같은 번호가 가장 비슷하다는 것은 같은 대상을 찍는다는 뜻으로 볼 수 있다. 그러나 유사도는 제품 12장 평균 상관이라 거칠다. 라벨 문구가 다르면 유사도가 떨어지므로, 같은 대상이라도 낮게 나올 수 있다.",
          ""]

lines += ["## 4. 기종별 전체 표", ""]
for m, g in df.groupby("machine", sort=True):
    lines += [f"### {m}", "",
              "| step | 사진 판독 (추정) | 일람표 항목 (step+1 가설) | 기준기종 같은 번호 유사 | 판정 |",
              "|---|---|---|---|---|"]
    for r in g.itertuples():
        same = "예" if r.same_index_as_ref else ("아니오" if r.same_index_as_ref is False else "-")
        lines.append(f"| {r.step} | {r.visual_desc} | {r.xlsx_item_at_step_plus_1} | {same} | {r.judgment} |")
    lines.append("")

lines += ["## 5. 기종마다 달라지는 것 (사진으로 확인)", "",
          "- **토너 라벨**: C-EXV(3029C003AA, 3029C004AA, 3030C001AA) vs NPG(나머지 10개 기종). 같은 스텝 1이라도 라벨 문구가 다르다.",
          "- **모델명 명판**: 2206N / 2206iF / 2206 / 2206i / 2206L / 2006N 등 기종마다 다르다. 일람표의 '제품 명칭 라벨' 항목도 기종별로 다르다.",
          "- **드럼 라벨**: C-EXV42(3029C003AA, 3029C004AA) vs NPG-59(나머지) 문구가 다르다.",
          "- **스텝 순서**: 3029C010AA는 전원스위치가 s11, 포트가 s12이다. 기준 기종은 s9, s10이다. 같은 번호가 다른 대상이 될 수 있다.",
          "- **후반 스텝 내용**: 3029C004AA와 3029C010AA는 s12~s19에 경고 스티커(CAUTION, 손끼임 그림)와 A4/A5 라벨 면이 있다. 3029C003AA에는 없다.",
          "- **기구 촬영 위치**: 캐리지·내부 촬영은 기종마다 s12 또는 s13에 있다.",
          ""]

lines += ["## 6. 이 표로 확정할 수 있는 것과 없는 것", "",
          "- **확정 가능 (사진으로 확인)**: 스텝 0은 13개 기종 모두 바코드 명판이다. 같은 번호가 다른 대상이 되는 기종(3029C010AA 등)이 있다.",
          "- **아직 확정 안 됨**: 체크포인트 번호와 스텝 번호의 대응 규칙. 반례가 있으므로 규칙으로 정할 수 없다.",
          "- **문서 근거 없음 (스텝 기준)**: 3029C004AA는 스텝 11, 13, 14, 15, 16, 17에, 3029C010AA는 스텝 13, 15, 16, 17, 18, 19에 일람표 항목이 없다. 이 스텝들의 검사 항목은 문서에서 확인할 수 없다.",
          ""]

Path("reports/step_audit.md").write_text("\n".join(lines), encoding="utf-8")
print("written", len(lines), "lines")
