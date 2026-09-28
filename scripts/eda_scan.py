"""EDA: (기종, 스텝)별로 표본 이미지를 읽어 해상도/손상/밝기/색상/정렬난이도를 스캔한다.

결과: data/processed/eda/step_scan.csv
"""
from __future__ import annotations

import random
import time

import cv2
import numpy as np
import pandas as pd

SAMPLES_PER_STEP = 8
SEED = 42


def imread(path: str):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def main() -> None:
    random.seed(SEED)
    inv = pd.read_csv("data/processed/eda/inventory.csv")
    rows = []
    t0 = time.time()

    for machine in sorted(inv.machine_type.unique()):
        sub = inv[inv.machine_type == machine]
        steps = sorted(sub.step.unique())
        for step in steps:
            prods = sub[sub.step == step]["product_id"].unique().tolist()
            sample = random.sample(prods, min(SAMPLES_PER_STEP, len(prods)))

            orb = cv2.ORB_create(1500)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            ref_gray = ref_kp = ref_des = None

            for i, pid in enumerate(sample):
                path = f"data/raw/{machine}/{pid}/step_{step}.jpg"
                img = imread(path)
                row = {
                    "machine_type": machine, "step": step, "product_id": pid,
                    "decode_ok": img is not None,
                    "width": None, "height": None,
                    "mean_brightness": None, "mean_hue": None, "mean_sat": None,
                    "is_ref": i == 0,
                    "orb_matches": None, "orb_inliers": None, "orb_inlier_ratio": None,
                    "dx": None, "dy": None, "rot_deg": None, "scale": None,
                }
                if img is None:
                    rows.append(row)
                    continue

                h, w = img.shape[:2]
                row["width"], row["height"] = w, h
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                row["mean_brightness"] = float(gray.mean())
                row["mean_hue"] = float(hsv[..., 0].mean())
                row["mean_sat"] = float(hsv[..., 1].mean())

                if i == 0:
                    ref_gray = gray
                    ref_kp, ref_des = orb.detectAndCompute(ref_gray, None)
                    rows.append(row)
                    continue

                kp, des = orb.detectAndCompute(gray, None)
                if des is None or ref_des is None or len(kp) < 8:
                    rows.append(row)
                    continue

                matches = sorted(bf.match(ref_des, des), key=lambda m: m.distance)[:200]
                row["orb_matches"] = len(matches)
                if len(matches) >= 8:
                    src = np.float32([ref_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                    dst = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                    M, inliers = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
                    if M is not None:
                        row["orb_inliers"] = int(inliers.sum())
                        row["orb_inlier_ratio"] = float(inliers.sum() / len(inliers))
                        row["dx"], row["dy"] = float(M[0, 2]), float(M[1, 2])
                        row["scale"] = float(np.sqrt(M[0, 0] ** 2 + M[1, 0] ** 2))
                        row["rot_deg"] = float(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
                rows.append(row)

        print(f"  done {machine} ({len(steps)} steps) - elapsed {time.time()-t0:.1f}s")

    out = pd.DataFrame(rows)
    out.to_csv("data/processed/eda/step_scan.csv", index=False)
    print("saved data/processed/eda/step_scan.csv -", len(out), "rows, total", round(time.time() - t0, 1), "s")


if __name__ == "__main__":
    main()
