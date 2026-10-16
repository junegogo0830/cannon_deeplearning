"""기존 범용 PaDiM(D)·존재검사(A)·내용검사(B) 추론 로직을 공통 결과 스키마로 감싼다.

여기서 알고리즘을 다시 구현하지 않는다 — src/pipeline.py::score_image, scripts/infer_presence.py,
scripts/infer_text.py를 그대로 호출만 한다("기존 A/B 모듈을 새로 개발하지 말고 재사용 가능하도록
분리한다"는 요구사항). 공통 스키마로 감싸는 이유는, 서로 다른 모듈의 원시 점수(마할라노비스
거리 vs NCC 유사도)를 섞어서 비교하거나 평균 내는 실수를 막기 위함이다 — raw_score는 모듈마다
의미와 방향(클수록 이상 vs 작을수록 이상)이 다르므로, 최종 판정은 반드시 status(PASS/FAIL/
REVIEW/NOT_APPLICABLE) 쪽을 기준으로 한다.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from src.data.dataset import read_image
from src.inspection.artifacts import ArtifactCache
from src.pipeline import score_image as padim_score_image
from src.scoring.decision import judge_step

PASS, FAIL, REVIEW, NOT_APPLICABLE = "PASS", "FAIL", "REVIEW", "NOT_APPLICABLE"


def _result(
    module: str, status: str, *, raw_score: float | None = None, threshold: float | None = None,
    roi: Any = None, reason: str | None = None, aligned: bool | None = None,
    artifact_used: str | None = None, elapsed_s: float = 0.0,
) -> dict[str, Any]:
    return {
        "module": module, "status": status, "raw_score": raw_score, "threshold": threshold,
        "roi": roi, "reason": reason, "aligned": aligned, "artifact_used": artifact_used,
        "elapsed_s": round(elapsed_s, 4),
    }


def run_generic(image_path: str | Path, machine: str, step: int, run_dir: str, cache: ArtifactCache) -> dict:
    """D(범용 PaDiM). score > threshold 이면 FAIL (src/scoring/decision.py 기준, 큰 점수=이상)."""
    t0 = time.perf_counter()
    artifacts = cache.generic(machine, step, run_dir)
    if artifacts is None:
        return _result("D", NOT_APPLICABLE, reason="아티팩트 없음(학습 실패 또는 미실행)",
                        elapsed_s=time.perf_counter() - t0)
    img = read_image(image_path)
    res = padim_score_image(img, artifacts, backbone=cache.backbone)
    elapsed = time.perf_counter() - t0
    artifact_path = f"results/{run_dir}/{machine}/step_{step}"
    if not res["ok"]:
        # keep_unaligned 정책이라 score_image 자체는 거의 항상 ok=True를 내지만,
        # on_failure=exclude로 바뀌는 경우를 대비해 REVIEW로 처리한다(FAIL로 단정하지 않음).
        return _result("D", REVIEW, threshold=res.get("threshold"), reason=f"정렬 실패({res['reason']})",
                        aligned=False, artifact_used=artifact_path, elapsed_s=elapsed)
    verdict = judge_step(res["score"], res["threshold"])
    return _result("D", verdict, raw_score=res["score"], threshold=res["threshold"],
                    aligned=res["aligned"], artifact_used=artifact_path, elapsed_s=elapsed)


def run_presence(image_path: str | Path, machine: str, step: int, cache: ArtifactCache) -> dict:
    """A(존재검사). scripts/infer_presence.py::score_image 재사용."""
    from scripts.infer_presence import score_image as presence_score_image

    t0 = time.perf_counter()
    artifacts = cache.presence(machine, step)
    if artifacts is None:
        return _result("A", NOT_APPLICABLE, reason="존재검사 아티팩트 없음",
                        elapsed_s=time.perf_counter() - t0)
    res = presence_score_image(str(image_path), machine, step, artifacts=artifacts, backbone=cache.backbone)
    elapsed = time.perf_counter() - t0
    artifact_path = f"results/presence_artifacts/{machine}/step_{step}"
    if not res["ok"]:
        return _result("A", REVIEW, threshold=res.get("threshold"), reason=f"정렬 실패({res['reason']})",
                        aligned=False, artifact_used=artifact_path, elapsed_s=elapsed)
    status = FAIL if res["verdict"].startswith("FAIL") else PASS
    roi = tuple(artifacts["meta"]["roi"]) if artifacts["meta"].get("roi") else None
    return _result("A", status, raw_score=res["score"], threshold=res["threshold"], roi=roi,
                    reason=res["verdict"] if status == FAIL else None,
                    aligned=res["aligned"], artifact_used=artifact_path, elapsed_s=elapsed)


def run_text(image_path: str | Path, machine: str, step: int, cache: ArtifactCache) -> dict:
    """B(내용검사). scripts/infer_text.py::score_image 재사용.

    주의: B의 raw_score(NCC 유사도, 클수록 정상)는 A/D의 raw_score(거리, 클수록 이상)와
    방향이 반대다 — 그래서 adapters는 raw_score를 그대로 노출만 하고, 판정은 status로만 한다.
    """
    from scripts.infer_text import score_image as text_score_image

    t0 = time.perf_counter()
    artifacts = cache.text(machine, step)
    if artifacts is None:
        return _result("B", NOT_APPLICABLE, reason="내용검사 아티팩트 없음",
                        elapsed_s=time.perf_counter() - t0)
    res = text_score_image(str(image_path), machine, step, artifacts=artifacts)
    elapsed = time.perf_counter() - t0
    artifact_path = f"results/text_artifacts/{machine}/step_{step}"
    roi = tuple(artifacts["meta"]["roi"]) if artifacts["meta"].get("roi") else None
    if res["verdict"] == "PASS":
        status = PASS
    elif res["verdict"].startswith("FAIL"):
        status = FAIL
    else:
        status = REVIEW
    return _result(
        "B", status, raw_score=res["score"], threshold=res["threshold"], roi=roi,
        reason=res["verdict"] if status != PASS else None,
        aligned=res.get("aligned"), artifact_used=artifact_path, elapsed_s=elapsed,
    )
