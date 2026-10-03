"""3029C003AA 스텝 카탈로그 마크다운 생성 (사진 기술은 육안 추정, xlsx 항목은 원본)."""
import pandas as pd

cp = pd.read_csv("data/processed/eda/xlsx_checkpoint_map.csv")
cp = cp[cp.machine_type == "3029C003AA"][["checkpoint", "part_name", "guarantee_item"]]
cp_text = {
    n: " / ".join(f"{r.part_name.strip()} ({str(r.guarantee_item).strip()})" for r in g.itertuples())
    for n, g in cp.groupby("checkpoint")
}

desc = {
    0: "바코드 라벨 (EAN/JAN, GS1-128 시리얼)",
    1: "C-EXV 토너 라벨 (작은 스티커)",
    2: '"Canon" 인쇄 글자 (커버)',
    3: '"2206N" 명판 + "imageRUNNER" 인쇄',
    4: '파란 레버 부품 + "C-EXV42 Drum Unit" 라벨',
    5: "파란 부품 + 흰 라벨(▲▲▲ 표시)",
    6: "파란 부품 (아래 흰 부품 일부)",
    7: "파란 부품 + 흰 라벨(▲▲▲ 표시) + 빗살 모양 부품",
    8: "흰 사각 패널 (슬롯 있음)",
    9: '전원 인렛 + 전원 스위치 + "NO" 각인',
    10: "USB / 네트워크 포트 + 나사",
    11: "사각 패널 + 하단 USB 포트",
    12: "패널 노치(이음새) + 나사",
    13: "내부 구조 (녹색 기판 근처, 케이블) — 확신 낮음",
    14: "패널 노치(이음새) + 나사 — 실제 FAIL 사례들과 같은 부위",
}

samples = pd.read_csv("reports/eda/step_catalog_samples.csv")
lines = [
    "# 3029C003AA 스텝 카탈로그 (스텝별 정상 사진 1장 육안 기술)",
    "",
    "> 그림: `step_catalog_3029C003AA.jpg` (샘플: `step_catalog_samples.csv`, 실제 FAIL 제품 제외 무작위 1장)",
    "> 사진 내용은 육안 기술(추정). 체크포인트 항목은 xlsx 원본 그대로이며, **스텝과의 대응은 확정하지 않음** (step_0 바코드 등 반례 있음).",
    "",
    "| step | 샘플 제품 | 사진에서 보이는 것 (추정) | xlsx 체크포인트 항목 (원본, 번호 = step+1 참고) |",
    "|---|---|---|---|",
]
for _, r in samples.iterrows():
    s = int(r.step)
    prod = r.sample_path.replace("\\", "/").split("/")[-2]
    lines.append(f"| {s} | {prod} | {desc.get(s, '')} | {cp_text.get(s + 1, '(해당 번호 항목 없음)')} |")

with open("reports/eda/step_catalog.md", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("written", len(lines), "lines")
