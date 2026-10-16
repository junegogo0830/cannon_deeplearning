"""부품 존재·누락 검사(모듈 A) — 실제 추론 엔트리포인트.

scripts/validate_presence_roi.py가 저장한 아티팩트(모델·ROI·임계값·기준사진)를 불러와,
학습/검증에 전혀 쓰이지 않은 새 이미지 한 장을 채점한다. src/pipeline.py + scripts/infer.py
(모듈 D)와 같은 역할을 모듈 A에 대해 하는 것 — 이게 없으면 "분석은 됐지만 실제 모델로
못 쓰는" 상태다.

실행:
  python -m scripts.infer_presence --machine 3029C003AA --step 2 --image <새 이미지 경로>
"""
from __future__ import annotations

import torch  # noqa: F401,E402  (Windows DLL 순서)

import argparse
from pathlib import Path

import numpy as np

from src.data.dataset import read_image
from src.data.preprocess import align_to_reference, crop_roi, preprocess
from src.features.embeddings import build_backbone, extract_embeddings
from src.models.padim import PaDiMModel
from src.utils.io import load_json

ARTIFACT_ROOT = Path("results/presence_artifacts")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="모듈 A(존재·누락 검사) 추론")
    p.add_argument("--machine", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--image", required=True, help="채점할 새 이미지 경로")
    return p.parse_args()


def load_presence_artifacts(machine: str, step: int) -> dict:
    """모델·기준사진·메타를 디스크에서 한 번만 읽어 반환한다 (호출부가 캐시해서 재사용하도록).

    src.pipeline.load_step_artifacts와 같은 역할을 모듈 A에 대해 한다 — score_image가
    매번 직접 디스크를 읽던 것을 분리해, src/inspection/artifacts.py의 캐시가 여러 이미지를
    채점할 때 반복 로딩을 피할 수 있게 한다(통합 파이프라인 CPU 효율화 요구사항).
    """
    artifact_dir = ARTIFACT_ROOT / machine / f"step_{step}"
    if not artifact_dir.exists():
        raise FileNotFoundError(
            f"아티팩트가 없습니다: {artifact_dir} — 먼저 python -m scripts.validate_presence_roi 로 학습하세요."
        )
    meta = load_json(artifact_dir / "meta.json")
    reference = read_image(artifact_dir / "reference.jpg")
    model = PaDiMModel.load(artifact_dir / "model.pkl")
    return {"meta": meta, "reference": reference, "model": model}


def score_image(
    image_path: str, machine: str, step: int, artifacts: dict | None = None, backbone=None
) -> dict:
    """artifacts/backbone을 넘기면 재사용(캐시)하고, 없으면 기존처럼 직접 로드/생성한다
    (CLI 단독 실행 시 동작은 바뀌지 않는다 — 통합 파이프라인에서만 캐시를 넘긴다)."""
    if artifacts is None:
        artifacts = load_presence_artifacts(machine, step)
    meta, reference, model = artifacts["meta"], artifacts["reference"], artifacts["model"]
    if backbone is None:
        backbone = build_backbone("mobilenet_v3_small")

    raw = read_image(image_path)
    aligned, diag = align_to_reference(raw, reference, meta["align_cfg"])
    if aligned is None:
        return {"ok": False, "reason": diag["reason"], "align_diag": diag, "threshold": meta["threshold"]}
    used_img = aligned
    roi, roi_size = tuple(meta["roi"]), tuple(meta["roi_size"])
    crop = preprocess(crop_roi(used_img, roi), None, roi_size)

    feat = extract_embeddings(backbone, crop, [meta["layer"]])[meta["layer"]]
    score = float(model.score(feat[None, ...]).max(axis=1)[0])
    threshold = meta["threshold"]
    verdict = "FAIL(결함 검출)" if score > threshold else "PASS"
    return {"ok": True, "score": score, "threshold": threshold, "verdict": verdict, "aligned": True}


def main() -> None:
    args = parse_args()
    result = score_image(args.image, args.machine, args.step)
    if not result["ok"]:
        print(f"정렬 실패({result['reason']}) → 판정 보류, 재검 권장")
        return
    print(f"{args.machine} step{args.step} [{args.image}]")
    print(f"  score={result['score']:.2f} threshold={result['threshold']:.2f} -> {result['verdict']}")


if __name__ == "__main__":
    main()
