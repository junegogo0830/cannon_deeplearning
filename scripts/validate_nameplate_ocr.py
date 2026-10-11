"""명판 문자 검증 모듈의 안정성 검증 (3029C009AA step3, 6416 2206N명판).

목적: §3.3.4(report.md)의 "connected component 기반 문자 대조"가 특정 NG 사례
1건에서만 통한 게 아니라, 독립된 정상 holdout에서도 안정적인지 확인한다
(외부 검토 지적 반영 — 이전 실험은 holdout이 val_paths와 겹쳤고, 문자 0개
검출을 "명판 누락"으로 바로 판정하는 등 두 가지 결함이 있었다).

절차:
  1) train_paths로 문자 템플릿(가장 분할이 깨끗한 정상 명판 1장) 구성.
  2) val_paths에서만 임계값(문자별 최저 NCC의 하위 1%)을 정한다.
  3) holdout_paths(= train/val 어디에도 안 쓰인 완전히 별개의 정상 집합)로
     분할 성공률 / FPR / REVIEW율을 측정한다 — 이 값들이 그대로 "운영 성능" 추정치다.
  4) 마지막으로 실제 NG 3건에 같은 고정 임계값을 적용한다.

판정 로직(문자 검사 단계만. 명판 "존재" 자체는 별도 ROI-PaDiM이 맡는다 — report.md §3.4 결정 트리):
  - 문자 5개 분할 성공 + 모든 칸 NCC >= 임계값 → PASS
  - 문자 5개 분할 성공 + 어느 한 칸이라도 임계값 미만 → FAIL(내용 불일치)
  - 문자 수가 5개가 아님(분할 실패) → REVIEW (FAIL로 자동 처리하지 않음)

실행: python -m scripts.validate_nameplate_ocr
"""
from __future__ import annotations

import torch  # noqa: F401,E402  (Windows DLL 순서)

import cv2
import numpy as np
import pandas as pd

from src.data.dataset import get_step_paths, read_image
from src.data.preprocess import align_to_reference, crop_roi
from src.data.split import reserve_val_then_train
from src.utils.config import load_config
from src.utils.log import get_logger

MACHINE, STEP = "3029C009AA", 3
# scripts/product_part_map_wide.py 류의 자동 매칭이 아니라, report.md §3.3.3에서
# 일람표 참고사진(row26_28.png)을 정상 기준사진에 NCC 매칭 + 육안 확인까지 마친 ROI.
ROI = (124, 279, 504, 125)
N_CHARS_EXPECTED = 5
CHAR_SIZE = (40, 60)


def get_roi_crop(raw: np.ndarray, ref_native: np.ndarray, align_cfg: dict) -> np.ndarray:
    aligned, _diag = align_to_reference(raw, ref_native, align_cfg)
    used = aligned if aligned is not None else raw
    crop = crop_roi(used, ROI)
    h, w = crop.shape[:2]
    scale = 900 / w
    return cv2.resize(crop, (900, int(h * scale)), interpolation=cv2.INTER_CUBIC)


def segment_chars(crop: np.ndarray) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # 전역 Otsu 대신 지역 적응형 이진화 -> 대각선 그림자(국소 밝기 편차)에 덜 흔들림.
    bw = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8
    )
    # 끊어진 획을 살짝 이어붙임(그림자로 갈라진 글자 보정), 너무 키우면 옆 글자와 붙으므로 최소한만.
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _labels, stats, _centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)
    h_img, w_img = gray.shape
    boxes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 30 or area > 3000:
            continue
        if h < h_img * 0.15 or h > h_img * 0.7:
            continue
        if w > w_img * 0.25:
            continue
        # 명판 오른쪽 끝의 회색 인디케이터 박스(문자가 아님, 고정 위치)를 제외.
        if x > w_img * 0.85:
            continue
        # 글자 영역의 평균 밝기(원본 그레이) 확인 - 인디케이터 박스는 글자보다 밝은 회색조임.
        mean_val = gray[y : y + h, x : x + w].mean()
        if mean_val > 150:
            continue
        boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[0])
    return boxes, bw


def char_images(bw: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> list[np.ndarray]:
    out = []
    for x, y, w, h in boxes:
        ch = bw[y : y + h, x : x + w]
        out.append(cv2.resize(ch, CHAR_SIZE, interpolation=cv2.INTER_NEAREST))
    return out


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float32), b.astype(np.float32)
    a0, b0 = a - a.mean(), b - b.mean()
    denom = np.sqrt((a0**2).sum()) * np.sqrt((b0**2).sum()) + 1e-6
    return float((a0 * b0).sum() / denom)


def min_char_ncc(crop: np.ndarray, ref_chars: list[np.ndarray]) -> float | None:
    """문자 5개가 정확히 분할됐을 때만 값을 반환, 아니면 None(=REVIEW 대상)."""
    boxes, bw = segment_chars(crop)
    if len(boxes) != N_CHARS_EXPECTED:
        return None
    chars = char_images(bw, boxes)
    return min(ncc(c, r) for c, r in zip(chars, ref_chars))


def main() -> None:
    cfg = load_config("configs/config.yaml")
    align_cfg = cfg["alignment"]
    logger = get_logger("validate_nameplate_ocr")

    pass_paths, fail_paths = get_step_paths(cfg, MACHINE, STEP)
    train_paths, val_paths = reserve_val_then_train(pass_paths, 0.2, 150, 60, cfg["project"]["seed"])
    used = {str(p) for p in train_paths} | {str(p) for p in val_paths}
    holdout_paths = [p for p in pass_paths if str(p) not in used]  # train/val과 완전히 분리된 집합
    logger.info(f"train={len(train_paths)} val={len(val_paths)} holdout={len(holdout_paths)}")

    ref_native = read_image(train_paths[0])

    # 1) 템플릿: train에서 가장 먼저 "문자 5개"로 깨끗하게 분할되는 샘플을 사용.
    ref_chars = None
    ref_source = None
    for p in train_paths:
        crop = get_roi_crop(read_image(p), ref_native, align_cfg)
        boxes, bw = segment_chars(crop)
        if len(boxes) == N_CHARS_EXPECTED:
            ref_chars = char_images(bw, boxes)
            ref_source = p
            break
    if ref_chars is None:
        raise RuntimeError("train_paths 안에서 문자 5개 분할에 성공한 템플릿을 못 찾았습니다.")
    logger.info(f"문자 템플릿: {ref_source}")

    # 2) 임계값: val_paths에서만 결정 (분할 성공한 것 중 최저 NCC의 하위 1%).
    val_scores = []
    val_review = 0
    for p in val_paths:
        crop = get_roi_crop(read_image(p), ref_native, align_cfg)
        s = min_char_ncc(crop, ref_chars)
        if s is None:
            val_review += 1
        else:
            val_scores.append(s)
    val_scores = np.array(val_scores)
    threshold = float(np.percentile(val_scores, 1.0)) if len(val_scores) else float("nan")
    logger.info(
        f"val: 분할성공={len(val_scores)}/{len(val_paths)} "
        f"(REVIEW {val_review}, {val_review/len(val_paths):.1%}) | "
        f"NCC 분포 min={val_scores.min():.3f} 5%={np.percentile(val_scores,5):.3f} "
        f"50%={np.percentile(val_scores,50):.3f} | 임계값(1%ile)={threshold:.3f}"
    )

    # 3) holdout: 완전히 독립된 정상 집합으로 운영 성능(FPR·REVIEW율) 측정.
    holdout_scores, holdout_review = [], 0
    for p in holdout_paths:
        crop = get_roi_crop(read_image(p), ref_native, align_cfg)
        s = min_char_ncc(crop, ref_chars)
        if s is None:
            holdout_review += 1
        else:
            holdout_scores.append(s)
    holdout_scores = np.array(holdout_scores)
    fpr = float((holdout_scores < threshold).mean()) if len(holdout_scores) else float("nan")
    logger.info(
        f"holdout(독립): 분할성공={len(holdout_scores)}/{len(holdout_paths)} "
        f"(REVIEW {holdout_review}, {holdout_review/len(holdout_paths):.1%}) | "
        f"고정임계값 기준 FPR={fpr:.1%}"
    )

    # 4) 실제 NG 3건에 고정 임계값 적용.
    rows = []
    for p in fail_paths:
        name = p.parent.parent.name
        crop = get_roi_crop(read_image(p), ref_native, align_cfg)
        boxes, bw = segment_chars(crop)
        if len(boxes) != N_CHARS_EXPECTED:
            verdict, score = "REVIEW(분할실패 - 별도 ROI-PaDiM 존재검사로 누락 여부 판단)", None
        else:
            chars = char_images(bw, boxes)
            s = min(ncc(c, r) for c, r in zip(chars, ref_chars))
            verdict = "FAIL(내용 불일치)" if s < threshold else "PASS"
            score = round(s, 3)
        rows.append({"product": name, "n_chars": len(boxes), "min_ncc": score, "verdict": verdict})
        logger.info(f"{name}: 문자수={len(boxes)} min_ncc={score} -> {verdict}")

    summary = pd.DataFrame(
        [
            {
                "val_review_rate": round(val_review / len(val_paths), 3),
                "holdout_split_success_rate": round(len(holdout_scores) / len(holdout_paths), 3),
                "holdout_review_rate": round(holdout_review / len(holdout_paths), 3),
                "threshold": round(threshold, 3),
                "holdout_fpr": round(fpr, 3),
            }
        ]
    )
    summary.to_csv("results/compare_full/nameplate_ocr_stability.csv", index=False)
    pd.DataFrame(rows).to_csv("results/compare_full/nameplate_ocr_ng_verdicts.csv", index=False)
    print(summary.to_string(index=False))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
