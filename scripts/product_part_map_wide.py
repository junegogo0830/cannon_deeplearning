"""넓은 탐색 매칭: 엄격 기준 안정 부품 5개를 스케일 0.5~2.0, 회전 ±8° 까지 찾는다.

참고 이미지 전체를 템플릿으로 쓰고, 사진·템플릿을 절반으로 줄여서 계산한다.
출력: reports/product_part_map_wide.csv, reports/product_part_map_wide_summary.csv,
      reports/box_check_wide_3products.jpg
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RAW = Path("data/raw/3029C003AA")
REF = Path("data/processed/xlsx_refs")
HALF = 0.5
SCALES = np.linspace(0.5, 2.0, 7)
ANGLES = [-8.0, 0.0, 8.0]
PARTS = {  # 라벨: (참고 파일, 스텝, 박스 색)
    "1525_▲▲▲라벨_5": ("row12_06.jpg", 5, (0, 0, 255)),
    "3780_전원스위치_9": ("row24_21.jpg", 9, (0, 255, 0)),
    "2917_코마_6": ("row21_16.jpg", 6, (255, 0, 0)),
    "3779_압판스토퍼_14": ("row23_18.jpg", 14, (0, 255, 255)),
    "V242_카바라이트_10": ("row18_12.jpg", 10, (255, 0, 255)),
}


def gray_half(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(img, None, fx=HALF, fy=HALF, interpolation=cv2.INTER_AREA)


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), borderValue=int(img.mean()))


def best_wide(photo: np.ndarray, tmpl: np.ndarray) -> tuple[float, int, int, float, float]:
    best = (-1.0, 0, 0, 1.0, 0.0)
    for a in ANGLES:
        rt = rotate(tmpl, a) if a else tmpl
        for s in SCALES:
            t = cv2.resize(rt, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            if t.shape[0] >= photo.shape[0] or t.shape[1] >= photo.shape[1] or t.shape[0] < 8:
                continue
            res = cv2.matchTemplate(photo, t, cv2.TM_CCOEFF_NORMED)
            _, mx, _, loc = cv2.minMaxLoc(res)
            if mx > best[0]:
                best = (float(mx), int(loc[0] / HALF), int(loc[1] / HALF), float(s), float(a))
    return best


def main() -> None:
    products = sorted(p.name for p in RAW.iterdir() if p.is_dir())
    tmpl_cache = {}
    for part, (fname, step, _c) in PARTS.items():
        tmpl_cache[part] = gray_half(REF / fname)

    def run(prod: str) -> list[dict]:
        out = []
        for part, (fname, step, _c) in PARTS.items():
            p = RAW / prod / f"step_{step}.jpg"
            photo = gray_half(p)
            score, x, y, s, a = best_wide(photo, tmpl_cache[part])
            out.append({"product": prod, "part": part, "step": step, "score": score,
                        "x": x, "y": y, "scale": s, "angle": a})
        return out

    rows = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for res in ex.map(run, products):
            rows.extend(res)
    df = pd.DataFrame(rows)
    Path("reports").mkdir(exist_ok=True)
    df.to_csv("reports/product_part_map_wide.csv", index=False, encoding="utf-8-sig")
    summ = df.groupby("part").agg(
        median_score=("score", "median"), std=("score", "std"), min=("score", "min"),
        scale_median=("scale", "median"), angle_mode=("angle", lambda s: s.mode().iloc[0]),
    ).reset_index()
    summ.to_csv("reports/product_part_map_wide_summary.csv", index=False, encoding="utf-8-sig")
    print(summ.round(3).to_string(index=False))

    # 박스 확인: 같은 3개 제품
    sample = ["2EQ16144", "2EQ16420", "2EQ16580"]
    panels = []
    for prod in sample:
        tiles = []
        for part, (fname, step, color) in PARTS.items():
            im = cv2.imdecode(np.fromfile(str(RAW / prod / f"step_{step}.jpg"), np.uint8), cv2.IMREAD_COLOR).copy()
            r = df[(df["product"] == prod) & (df.part == part)].iloc[0]
            ref = cv2.imdecode(np.fromfile(str(REF / fname), np.uint8), cv2.IMREAD_COLOR)
            w = int(ref.shape[1] * r.scale)
            h = int(ref.shape[0] * r.scale)
            cv2.rectangle(im, (int(r.x * 2), int(r.y * 2)), (int(r.x * 2) + 2 * w, int(r.y * 2) + 2 * h), color, 4)
            cv2.putText(im, f"{part[:6]} s{step} {r.score:.2f}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
            tiles.append(cv2.resize(im, (512, 384), interpolation=cv2.INTER_AREA))
        tiles.append(np.zeros_like(tiles[0]))
        grid = np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:6])])
        panels.append(grid)
    cv2.imwrite("reports/box_check_wide_3products.jpg", np.vstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    main()
