"""부품 존재·누락 검사(모듈 A) — 정식 재현 스크립트.

지금까지 /tmp에서 1회성으로 돌렸던 실험(report.md §3.3 11번, §3.3.3)을 정식 스크립트로
옮긴 것. 절차:
  1) 일람표 참고사진(xlsx_refs)을 정상 기준사진(train_paths[0])에 멀티스케일 NCC로 매칭
     → ROI 위치 확정 (NG 사진은 전혀 안 씀 — 순환논리 없음, §3.3.3).
  2) 원본 해상도 비율을 유지한 채 저배율로만 리사이즈(찌그러뜨리지 않음 — §3.3.4에서 확인한
     "종횡비 왜곡이 검출을 망친다"는 교훈 반영).
  3) 그 ROI로 PaDiM 학습(train) → 임계값(val, percentile-99) → 독립 holdout으로 FPR 확인
     → 실제 NG로 검출 여부 확인.

대상 두 사례(각각 §3.3.3에서 검증됨):
  - 3029C003AA step2 (B793 Canon각인, 누락형)
  - 3029C009AA step3 (6416 2206N명판, 누락형 — 2EX36613만 해당. 오부착형 2EX36661은
    모듈 B가 담당하므로 여기서는 참고용으로만 같이 출력한다.)

실행: python -m scripts.validate_presence_roi
"""
from __future__ import annotations

import torch  # noqa: F401,E402  (Windows DLL 순서)

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.data.dataset import get_step_paths, read_image
from src.data.preprocess import align_to_reference, crop_roi, preprocess
from src.data.split import reserve_val_then_train
from src.eval.metrics import auroc
from src.utils.io import save_json

ARTIFACT_ROOT = Path("results/presence_artifacts")
from src.features.embeddings import build_backbone, extract_embeddings
from src.models.padim import PaDiMModel
from src.scoring.threshold import threshold_percentile
from src.utils.config import load_config
from src.utils.log import get_logger

LAYER = 4
REF_DIR = "data/processed/xlsx_refs"

CASES = [
    # (기종, 스텝, 참고사진, margin, 해상도배율, 라벨)
    ("3029C003AA", 2, "row34_30.png", 30, 1.0, "step2_Canon각인"),
    ("3029C009AA", 3, "row26_28.png", 30, 2.0, "step3_2206N명판"),
]


def locate_roi(ref_native: np.ndarray, template_path: str, margin: int, scales=None) -> tuple[tuple, float]:
    """참고사진을 정상 기준사진에 멀티스케일 NCC로 매칭해 ROI를 찾는다(정상 데이터만 사용)."""
    scales = scales if scales is not None else np.linspace(1.0, 6.0, 26)
    gray_full = cv2.cvtColor(ref_native, cv2.COLOR_BGR2GRAY)
    tmpl = cv2.imdecode(np.fromfile(template_path, np.uint8), cv2.IMREAD_GRAYSCALE)
    best = (-1.0, None, None)
    for s in scales:
        t = cv2.resize(tmpl, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
        if t.shape[0] >= gray_full.shape[0] or t.shape[1] >= gray_full.shape[1]:
            continue
        res = cv2.matchTemplate(gray_full, t, cv2.TM_CCOEFF_NORMED)
        _, mx, _, loc = cv2.minMaxLoc(res)
        if mx > best[0]:
            best = (float(mx), loc, (t.shape[1], t.shape[0]))
    score, loc, size = best
    x, y = loc
    w, h = size
    roi = (max(0, x - margin), max(0, y - margin), w + 2 * margin, h + 2 * margin)
    return roi, score


def run_case(cfg, machine: str, step: int, template_name: str, margin: int, scale: float, label: str, backbone, logger) -> dict:
    pass_paths, fail_paths = get_step_paths(cfg, machine, step)
    train_paths, val_paths = reserve_val_then_train(pass_paths, 0.2, 150, 60, cfg["project"]["seed"])
    used = {str(p) for p in train_paths} | {str(p) for p in val_paths}
    holdout_paths = [p for p in pass_paths if str(p) not in used][:300]
    align_cfg = cfg["alignment"]

    ref_native = read_image(train_paths[0])
    roi, ncc = locate_roi(ref_native, f"{REF_DIR}/{template_name}", margin)
    rh, rw = roi[3], roi[2]
    roi_size = (max(16, int(rh * scale) // 16 * 16), max(16, int(rw * scale) // 16 * 16))  # 종횡비 유지
    logger.info(f"{label}: ROI={roi} NCC매칭신뢰도={ncc:.3f} ROI_SIZE={roi_size}")

    def load_roi(paths: list) -> list:
        out = []
        for p in paths:
            raw = read_image(p)
            aligned, _diag = align_to_reference(raw, ref_native, align_cfg)
            used_img = aligned if aligned is not None else raw
            out.append(preprocess(crop_roi(used_img, roi), None, roi_size))
        return out

    def embed(imgs: list) -> np.ndarray:
        return np.stack([extract_embeddings(backbone, im, [LAYER])[LAYER] for im in imgs])

    train_feats = embed(load_roi(train_paths))
    val_feats = embed(load_roi(val_paths))
    model = PaDiMModel(reg_eps=1e-2)
    model.fit(train_feats)
    thr = threshold_percentile(model.score(val_feats).max(axis=1), 99.0)

    # 실제 추론(scripts/infer_presence.py)에서 재사용할 수 있도록 아티팩트를 저장한다
    # (이전까지는 학습-채점이 한 스크립트 안에서 1회성으로 끝나, 새 이미지에 바로 쓸 방법이 없었다).
    artifact_dir = ARTIFACT_ROOT / machine / f"step_{step}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    model.save(artifact_dir / "model.pkl")
    ok, buf = cv2.imencode(".jpg", ref_native)
    (artifact_dir / "reference.jpg").write_bytes(buf.tobytes())
    save_json({
        "machine": machine, "step": step, "label": label,
        "roi": list(roi), "roi_size": list(roi_size), "threshold": float(thr),
        "layer": LAYER, "align_cfg": align_cfg,
    }, artifact_dir / "meta.json")
    logger.info(f"{label}: 아티팩트 저장 완료 -> {artifact_dir}")

    s_holdout = model.score(embed(load_roi(holdout_paths))).max(axis=1)
    fpr = float((s_holdout > thr).mean())

    rows = []
    if fail_paths:
        s_fail = model.score(embed(load_roi(fail_paths))).max(axis=1)
        y = np.array([0] * len(s_holdout) + [1] * len(s_fail))
        s = np.concatenate([s_holdout, s_fail])
        auc = auroc(y, s)
        for p, score in zip(fail_paths, s_fail):
            rows.append({
                "product": p.parent.parent.name, "score": round(float(score), 2),
                "threshold": round(float(thr), 2), "verdict": "FAIL(검출)" if score > thr else "PASS(미검출)",
            })
    else:
        auc = float("nan")

    logger.info(f"{label}: threshold={thr:.2f} FPR={fpr:.4f} AUROC={auc:.3f}")
    for r in rows:
        logger.info(f"  {r}")

    return {"label": label, "roi": roi, "ncc": ncc, "threshold": thr, "fpr": fpr, "auroc": auc, "cases": rows}


def main() -> None:
    cfg = load_config("configs/config.yaml")
    logger = get_logger("validate_presence_roi")
    backbone = build_backbone("mobilenet_v3_small")

    all_results = []
    for machine, step, template, margin, scale, label in CASES:
        result = run_case(cfg, machine, step, template, margin, scale, label, backbone, logger)
        all_results.append(result)

    rows = []
    for r in all_results:
        for c in r["cases"]:
            rows.append({"label": r["label"], "fpr": round(r["fpr"], 4), "auroc": round(r["auroc"], 3), **c})
    df = pd.DataFrame(rows)
    df.to_csv("results/compare_full/presence_roi_result.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
