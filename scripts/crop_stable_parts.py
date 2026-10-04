"""엄격 기준 안정 부품 crop: 제품별 매칭 위치·스케일로 원본 사진(1024×768)에서 부품 영역을 잘라 저장.

기준: reports/product_part_map_summary.csv 의 std ≤ 0.015 항목.
crop 박스: 참고 이미지 원본 크기 × 매칭 스케일, 좌상단 = 매칭 위치 (원본 좌표).
출력: data/processed/crops_3029C003AA/<부품>/<제품>.jpg   (대용량이라 gitignore)
      reports/crop_qa_sheet.jpg                              (부품별 무작위 8장 + 최저 점수 3장)
      reports/crop_summary.csv                               (부품별 crop 수, 크기 분포)
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MACHINE = "3029C003AA"
RAW = Path("data/raw") / MACHINE
REF = Path("data/processed/xlsx_refs")
OUT = Path("data/processed/crops_3029C003AA")
STABLE = {  # 부품 라벨: (참고 파일, 스텝)
    "1525_▲▲▲라벨_5": ("row12_06.jpg", 5),
    "3780_전원스위치_9": ("row24_21.jpg", 9),
    "2917_코마_6": ("row21_16.jpg", 6),
    "3779_압판스토퍼_14": ("row23_18.jpg", 14),
    "V242_카바라이트_10": ("row18_12.jpg", 10),
}
STD_MAX = 0.015
SLUG = {  # 파일 경로는 ASCII로 (OpenCV imwrite 가 한글/기호 경로에서 실패함)
    "1525_▲▲▲라벨_5": "p1525_s5",
    "3780_전원스위치_9": "p3780_s9",
    "2917_코마_6": "p2917_s6",
    "3779_압판스토퍼_14": "p3779_s14",
    "V242_카바라이트_10": "pv242_s10",
}


def main() -> None:
    summ = pd.read_csv("reports/product_part_map_summary.csv")
    stable_parts = set(summ.loc[summ["std"] <= STD_MAX, "part"])
    targets = {k: v for k, v in STABLE.items() if k in stable_parts}
    print("엄격 기준 통과 부품:", list(targets))

    df = pd.read_csv("reports/product_part_map_3029C003AA.csv")
    rows, qa = [], {}
    for part, (fname, step) in targets.items():
        ref = cv2.imdecode(np.fromfile(str(REF / fname), np.uint8), cv2.IMREAD_COLOR)
        rh, rw = ref.shape[:2]
        sub = df[df.part == part].dropna(subset=["x", "y", "scale"])
        slug = SLUG[part]
        outdir = OUT / slug
        outdir.mkdir(parents=True, exist_ok=True)
        ok = 0
        for r in sub.itertuples():
            img = cv2.imdecode(np.fromfile(str(RAW / r.product / f"step_{step}.jpg"), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            w, h = int(round(rw * r.scale)), int(round(rh * r.scale))
            x0, y0 = int(r.x), int(r.y)
            x1, y1 = min(x0 + w, img.shape[1]), min(y0 + h, img.shape[0])
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            crop = img[y0:y1, x0:x1]
            ok_write, buf = cv2.imencode(".jpg", crop)
            (outdir / f"{r.product}.jpg").write_bytes(buf.tobytes())
            ok += 1
        rows.append({"part": part, "ref": fname, "step": step, "products": len(sub), "crops_saved": ok,
                     "std_match_score": float(summ.loc[summ.part == part, "std"].iloc[0])})
        # QA: 무작위 8장 + 최저 점수 3장
        rng = random.Random(0)
        low = sub.nsmallest(3, "score")["product"].tolist()
        pool = [p for p in sub["product"].tolist() if p not in low]
        pick = low + rng.sample(pool, min(8, len(pool)))
        qa[part] = [(p, outdir / f"{p}.jpg") for p in pick if (outdir / f"{p}.jpg").exists()]
        print(part, "->", slug, "saved", ok)

    pd.DataFrame(rows).to_csv("reports/crop_summary.csv", index=False, encoding="utf-8-sig")

    # QA 시트: 부품마다 한 행, 11장(최저 3 + 무작위 8)
    cell_h, cell_w = 120, 120
    row_imgs = []
    for part, items in qa.items():
        tiles = []
        for prod, path in items:
            im = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
            t = cv2.resize(im, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
            cv2.rectangle(t, (0, 0), (cell_w, 14), (0, 0, 0), -1)
            cv2.putText(t, prod[-5:], (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            tiles.append(t)
        while len(tiles) < 11:
            tiles.append(np.zeros((cell_h, cell_w, 3), np.uint8))
        label = np.zeros((cell_h, 180, 3), np.uint8)
        cv2.putText(label, SLUG[part], (4, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        row_imgs.append(np.hstack([label] + tiles))
    if row_imgs:
        sheet = np.vstack(row_imgs)
        Path("reports").mkdir(exist_ok=True)
        cv2.imwrite("reports/crop_qa_sheet.jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        print("QA 시트:", sheet.shape)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
