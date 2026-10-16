"""명판 문자 내용 검사(모듈 B) — 실제 추론 엔트리포인트.

scripts/validate_nameplate_ocr.py가 저장한 아티팩트(문자 템플릿·ROI·임계값)를 불러와,
새 이미지 한 장을 PASS/FAIL/REVIEW로 판정한다 (모듈 A의 scripts/infer_presence.py와 동일한
역할). segment_chars/min_char_ncc 로직은 validate_nameplate_ocr.py 것을 그대로 재사용한다
(같은 알고리즘을 두 군데서 따로 구현하지 않기 위함).

실행:
  python -m scripts.infer_text --image <새 이미지 경로>
"""
from __future__ import annotations

import torch  # noqa: F401,E402  (Windows DLL 순서)

import argparse
from pathlib import Path

import cv2
import numpy as np

from scripts.validate_nameplate_ocr import classify_nameplate
from src.data.dataset import read_image
from src.data.preprocess import align_to_reference, crop_roi
from src.utils.io import load_json

ARTIFACT_ROOT = Path("results/text_artifacts")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="모듈 B(명판 문자 내용 검사) 추론")
    p.add_argument("--machine", default="3029C009AA")
    p.add_argument("--step", type=int, default=3)
    p.add_argument("--image", required=True, help="채점할 새 이미지 경로")
    return p.parse_args()


def load_text_artifacts(machine: str, step: int) -> dict:
    """모듈 B 아티팩트를 한 번만 읽어 반환한다 — src/inspection/artifacts.py 캐시용
    (infer_presence.load_presence_artifacts와 같은 이유)."""
    artifact_dir = ARTIFACT_ROOT / machine / f"step_{step}"
    if not artifact_dir.exists():
        raise FileNotFoundError(
            f"아티팩트가 없습니다: {artifact_dir} — 먼저 python -m scripts.validate_nameplate_ocr 로 학습하세요."
        )
    meta = load_json(artifact_dir / "meta.json")
    reference = read_image(artifact_dir / "reference.jpg")
    ref_chars = list(np.load(artifact_dir / "ref_chars.npy"))
    return {"meta": meta, "reference": reference, "ref_chars": ref_chars}


def score_image(image_path: str, machine: str, step: int, artifacts: dict | None = None) -> dict:
    if artifacts is None:
        artifacts = load_text_artifacts(machine, step)
    meta, reference, ref_chars = artifacts["meta"], artifacts["reference"], artifacts["ref_chars"]

    raw = read_image(image_path)
    aligned, diag = align_to_reference(raw, reference, meta["align_cfg"])
    if aligned is None:
        used = raw  # 모듈 B는 정렬 실패해도 원본으로 시도(존재 자체는 모듈 A가 따로 판단)
    else:
        used = aligned
    crop = crop_roi(used, tuple(meta["roi"]))
    h, w = crop.shape[:2]
    scale = 900 / w
    crop = cv2.resize(crop, (900, int(h * scale)), interpolation=cv2.INTER_CUBIC)

    r = classify_nameplate(crop, ref_chars)
    if r["mode"] == "exact":
        verdict = "FAIL(내용 불일치)" if r["score"] < meta["threshold"] else "PASS"
        score = round(r["score"], 3)
    elif r["mode"] == "missing_char":
        verdict, score = "FAIL(문자 누락)", None
    else:
        verdict, score = "REVIEW(문자 분할 실패 — 사람 검토 필요)", None
    return {
        "verdict": verdict, "n_chars": r["n_chars"], "score": score, "threshold": meta["threshold"],
        "aligned": aligned is not None,
    }


def main() -> None:
    args = parse_args()
    result = score_image(args.image, args.machine, args.step)
    print(f"{args.machine} step{args.step} [{args.image}]")
    print(f"  문자수={result['n_chars']} score={result['score']} -> {result['verdict']}")


if __name__ == "__main__":
    main()
