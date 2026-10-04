"""제품별 부품 매핑 (3029C003AA): 체크포인트 N = step_N 가정 하에, 부품 참고 사진을 제품마다 해당 스텝 사진에서 찾는다.

출력: reports/product_part_map_3029C003AA.csv  (제품 × 부품: 최고 점수, 위치, 스케일)
      reports/product_part_map_summary.csv     (부품별 점수 분포)

주의: 점수는 "그 부품처럼 보이는 정도"이지 불량 판정이 아니다. 낮은 점수 = 찾지 못함 (미부착, 위치 변동, 촬영 차이 등 원인 미확정).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

MACHINE = "3029C003AA"
RAW = Path("data/raw") / MACHINE
REF = Path("data/processed/xlsx_refs")
SCALE_PHOTO = 0.5  # 사진·템플릿 축소 비율 (속도)
SCALES = [0.6, 0.75, 0.9, 1.05, 1.2]

# (부품 라벨, 참고 파일, 체크포인트 번호 = 사진 step 번호 가정)
PAIRS = [
    ("C551_토너라벨_1", "row11_05.jpg", 1),
    ("B793_Canon각인_2", "row34_30.png", 2),
    ("6416_2206N명판_3", "row26_28.png", 3),
    ("3102_현상가압라벨_4", "row13_14.jpg", 4),
    ("3699_레버피스_4", "row20_17.jpg", 4),
    ("1525_▲▲▲라벨_5", "row12_06.jpg", 5),
    ("2917_코마_6", "row21_16.jpg", 6),
    ("1525_▲▲▲라벨_7", "row12_07.jpg", 7),
    ("0000_모듈러카바_8", "row36_31.jpg", 8),
    ("3780_전원스위치_9", "row24_21.jpg", 9),
    ("V242_카바라이트_10", "row18_12.jpg", 10),
    ("V242_카바라이트_11", "row18_12.jpg", 11),
    ("3779_압판스토퍼_12", "row23_19.jpg", 12),
    ("3779_압판스토퍼_14", "row23_18.jpg", 14),
]


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, None, fx=SCALE_PHOTO, fy=SCALE_PHOTO, interpolation=cv2.INTER_AREA)


def load_template(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"참고 이미지 읽기 실패: {path}")
    return cv2.resize(img, None, fx=SCALE_PHOTO, fy=SCALE_PHOTO, interpolation=cv2.INTER_AREA)


def best_match(photo: np.ndarray, tmpl: np.ndarray) -> tuple[float, int, int, float]:
    best = (-1.0, 0, 0, 1.0)
    for s in SCALES:
        t = cv2.resize(tmpl, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        if t.shape[0] >= photo.shape[0] or t.shape[1] >= photo.shape[1] or t.shape[0] < 8:
            continue
        res = cv2.matchTemplate(photo, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx > best[0]:
            best = (float(mx), int(loc[0] / SCALE_PHOTO), int(loc[1] / SCALE_PHOTO), s)
    return best


def main() -> None:
    templates = {label: load_template(REF / fname) for label, fname, _ in PAIRS}
    products = sorted(p.name for p in RAW.iterdir() if p.is_dir())
    print(f"제품 {len(products)}개, 부품 {len(PAIRS)}개")

    def run_product(prod: str) -> list[dict]:
        out = []
        for label, _fname, step in PAIRS:
            photo_path = RAW / prod / f"step_{step}.jpg"
            if not photo_path.exists():
                out.append({"product": prod, "part": label, "step": step, "score": np.nan,
                            "x": np.nan, "y": np.nan, "scale": np.nan, "note": "사진 없음"})
                continue
            photo = load_gray(photo_path)
            score, x, y, s = best_match(photo, templates[label])
            out.append({"product": prod, "part": label, "step": step, "score": score,
                        "x": x, "y": y, "scale": s, "note": ""})
        return out

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for i, res in enumerate(ex.map(run_product, products), 1):
            rows.extend(res)
            if i % 100 == 0:
                print(f"  {i}/{len(products)} 제품 완료")

    df = pd.DataFrame(rows)
    Path("reports").mkdir(exist_ok=True)
    df.to_csv("reports/product_part_map_3029C003AA.csv", index=False, encoding="utf-8-sig")

    summ = df.groupby("part").score.describe(percentiles=[0.05, 0.25, 0.5]).reset_index()
    summ.to_csv("reports/product_part_map_summary.csv", index=False, encoding="utf-8-sig")
    print(summ.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
