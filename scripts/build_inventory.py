"""data/raw 전체를 스캔해 인벤토리 CSV를 만든다 (이미지 디코딩 없이 파일 목록만).

출력:
    data/processed/eda/inventory.csv  — machine_type, product_id, step, filename, size_bytes
    data/processed/eda/ng_events.csv  — NG 폴더 안의 불량 스텝 이미지 목록 (유일한 FAIL 라벨 소스)
"""
from __future__ import annotations

import csv
import os
import re

RAW_DIR = "data/raw"
OUT_DIR = "data/processed/eda"

STEP_RE = re.compile(r"^step_(\d+)\.jpg$", re.IGNORECASE)
NG_RE = re.compile(
    r"^(?P<ts>[\d-]+_[\d_]+)_cell(?P<cell>\d+)_process(?P<process>\d+)_step_(?P<step>\d+)\.jpg$",
    re.IGNORECASE,
)


def scan() -> tuple[list[list], list[list]]:
    rows: list[list] = []
    ng_rows: list[list] = []

    for machine in sorted(os.listdir(RAW_DIR)):
        mpath = os.path.join(RAW_DIR, machine)
        if not os.path.isdir(mpath):
            continue
        for product in sorted(os.listdir(mpath)):
            ppath = os.path.join(mpath, product)
            if not os.path.isdir(ppath):
                continue
            for fname in os.listdir(ppath):
                fpath = os.path.join(ppath, fname)
                if os.path.isdir(fpath):
                    if fname.upper() == "NG":
                        for ngf in os.listdir(fpath):
                            ngp = os.path.join(fpath, ngf)
                            if not os.path.isfile(ngp):
                                continue
                            m = NG_RE.match(ngf)
                            size = os.path.getsize(ngp)
                            if m:
                                ng_rows.append([
                                    machine, product, m.group("step"), ngf, size,
                                    m.group("cell"), m.group("process"), m.group("ts"),
                                ])
                            else:
                                ng_rows.append([machine, product, "", ngf, size, "", "", ""])
                    continue
                m = STEP_RE.match(fname)
                if m:
                    rows.append([machine, product, m.group(1), fname, os.path.getsize(fpath)])

    return rows, ng_rows


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    rows, ng_rows = scan()

    with open(os.path.join(OUT_DIR, "inventory.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["machine_type", "product_id", "step", "filename", "size_bytes"])
        w.writerows(rows)

    with open(os.path.join(OUT_DIR, "ng_events.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["machine_type", "product_id", "step", "filename", "size_bytes", "cell", "process", "timestamp"])
        w.writerows(ng_rows)

    print(f"root-level step images: {len(rows)}")
    print(f"NG-flagged images: {len(ng_rows)}")


if __name__ == "__main__":
    main()
