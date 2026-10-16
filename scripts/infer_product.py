"""제품 전체 통합 추론 CLI — 기종·제품을 받아 등록된 전체 스텝을 규칙에 따라 자동 검사한다.

기존에 흩어져 있던 범용 PaDiM(scripts/infer.py), 존재검사(scripts/infer_presence.py),
내용검사(scripts/infer_text.py)를 configs/inspection_rules.yaml 규칙에 따라 자동으로
연결한다 — 어떤 모듈을 쓸지 사람이 매번 골라야 했던 문제를 없앤다.

실행:
  python -m scripts.infer_product --machine 3029C003AA --product 2EQ17039
  python -m scripts.infer_product --machine 3029C009AA --product 2EX36613 --include-ng --verbose
"""
from __future__ import annotations

# Windows DLL 로드 순서 (src/pipeline.py 상단 설명과 동일한 이유)
import torch  # noqa: F401,E402

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

# Windows 콘솔(cp949)이 "—" 같은 유니코드 문자를 못 써서 print()가 죽는 걸 막는다.
# 어떤 문자열이든(데이터에서 온 것 포함) 콘솔에 못 쓰는 글자는 "?"로 대체하고 계속 진행한다 —
# JSON/CSV 저장은 항상 UTF-8로 직접 쓰므로 원문은 결과 파일에 그대로 남는다.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from src.data.dataset import load_ng_events
from src.inspection.artifacts import ArtifactCache
from src.inspection.decision import aggregate_product, aggregate_step
from src.inspection.registry import InspectionRegistry
from src.inspection.router import run_step

RAW_DIR = Path("data/raw")

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover
    _HAS_PSUTIL = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="제품 전체 통합 검사(A/B/C/D 자동 라우팅)")
    p.add_argument("--config", default="configs/config_cnn.yaml")
    p.add_argument("--rules", default="configs/inspection_rules.yaml")
    p.add_argument("--machine", required=True)
    p.add_argument("--product", required=True, help="제품 시리얼 폴더명")
    p.add_argument("--include-ng", action="store_true", help="NG/ 폴더 사진도 전부 별도 평가")
    p.add_argument("--verbose", action="store_true", help="스텝별 상세 결과를 줄바꿈하여 출력")
    p.add_argument("--out-dir", default=None, help="JSON/CSV 저장 위치 (기본: results/infer_product/<machine>/<product>)")
    return p.parse_args()


def _score_str(score: float | None) -> str:
    return f"{score:.3f}" if score is not None else "NA"


def one_line_summary(step_decisions: dict[int, dict]) -> str:
    parts = []
    for step in sorted(step_decisions):
        d = step_decisions[step]
        parts.append(f"STEP {step} {d['status']}(score={_score_str(d['representative_score'])}, module={d['representative_module'] or '-'})")
    return " | ".join(parts)


def inspect_product(
    machine: str, product: str, registry: InspectionRegistry, cache: ArtifactCache, include_ng: bool = False,
) -> tuple[dict[int, dict], list[dict], dict]:
    step_decisions: dict[int, dict] = {}
    skipped_steps: list[int] = []

    for step in registry.steps(machine):
        root_path = RAW_DIR / machine / product / f"step_{step}.jpg"
        if not root_path.is_file():
            skipped_steps.append(step)  # 이 제품엔 해당 스텝 사진 자체가 없음 — 조용히 누락시키지 않고 기록
            continue
        rr = run_step(root_path, machine, step, registry, cache)
        dec = aggregate_step(rr)
        dec["image_path"] = str(root_path)
        dec["source"] = "root"
        step_decisions[step] = dec

    ng_decisions: list[dict] = []
    if include_ng:
        ng_df = load_ng_events(cache.cfg)
        ng_sub = ng_df[(ng_df.machine_type == machine) & (ng_df.product_id == product)]
        for row in ng_sub.itertuples():
            step = int(row.step)
            ng_path = RAW_DIR / machine / product / "NG" / row.filename
            if not ng_path.is_file():
                continue
            rr = run_step(ng_path, machine, step, registry, cache)
            dec = aggregate_step(rr)
            dec["image_path"] = str(ng_path)
            dec["source"] = "NG"
            dec["product"] = product
            ng_decisions.append(dec)

    product_summary = aggregate_product(step_decisions)
    product_summary["skipped_steps_no_image"] = skipped_steps
    return step_decisions, ng_decisions, product_summary


def _serializable(d: dict[str, Any]) -> dict[str, Any]:
    """module_results 안에 있는 rule dict 등 중첩 구조까지 그대로 JSON 직렬화 가능하게 복사."""
    return json.loads(json.dumps(d, default=str, ensure_ascii=False))


def save_results(
    out_dir: Path, machine: str, product: str,
    step_decisions: dict[int, dict], ng_decisions: list[dict], product_summary: dict,
    timing: dict,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "machine": machine, "product": product,
        "steps": {str(k): _serializable(v) for k, v in step_decisions.items()},
        "ng_images": [_serializable(d) for d in ng_decisions],
        "product_summary": product_summary, "timing": timing,
    }
    with open(out_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(out_dir / "result.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["machine", "product", "source", "step", "status", "module", "score", "threshold",
                    "reason", "mapping_verified", "module_validated", "image_path", "elapsed_s_total"])
        for step, d in sorted(step_decisions.items()):
            elapsed = sum(r["elapsed_s"] for r in d["module_results"].values())
            w.writerow([machine, product, "root", step, d["status"], d["representative_module"],
                        d["representative_score"], d["representative_threshold"], d["reason"],
                        d["mapping_verified"], d["module_validated"], d["image_path"], round(elapsed, 4)])
        for d in ng_decisions:
            elapsed = sum(r["elapsed_s"] for r in d["module_results"].values())
            w.writerow([machine, product, "NG", d["step"], d["status"], d["representative_module"],
                        d["representative_score"], d["representative_threshold"], d["reason"],
                        d["mapping_verified"], d["module_validated"], d["image_path"], round(elapsed, 4)])


def main() -> None:
    args = parse_args()

    t_cache0 = time.perf_counter()
    mem_before = psutil.Process().memory_info().rss / 1e6 if _HAS_PSUTIL else None
    registry = InspectionRegistry(args.rules)
    cache = ArtifactCache(args.config)
    cache_build_s = time.perf_counter() - t_cache0

    t_infer0 = time.perf_counter()
    step_decisions, ng_decisions, product_summary = inspect_product(
        args.machine, args.product, registry, cache, include_ng=args.include_ng
    )
    infer_wall_s = time.perf_counter() - t_infer0
    mem_after = psutil.Process().memory_info().rss / 1e6 if _HAS_PSUTIL else None

    all_elapsed = [
        r["elapsed_s"] for d in list(step_decisions.values()) + ng_decisions for r in d["module_results"].values()
    ]
    all_elapsed_sorted = sorted(all_elapsed)
    p95_idx = max(0, int(len(all_elapsed_sorted) * 0.95) - 1)
    timing = {
        "cache_build_s": round(cache_build_s, 4),
        "infer_wall_s": round(infer_wall_s, 4),
        "per_module_call_mean_s": round(sum(all_elapsed) / len(all_elapsed), 4) if all_elapsed else None,
        "per_module_call_p95_s": round(all_elapsed_sorted[p95_idx], 4) if all_elapsed_sorted else None,
        "n_module_calls": len(all_elapsed),
        "rss_mb_before": round(mem_before, 1) if mem_before is not None else None,
        "rss_mb_after": round(mem_after, 1) if mem_after is not None else None,
        "memory_note": None if _HAS_PSUTIL else "psutil 미설치로 메모리 측정 생략",
    }

    print(one_line_summary(step_decisions))
    print(f"\n제품 {args.product} 최종 판정: {product_summary['product_status']} "
          f"(PASS {product_summary['n_pass']} / FAIL {product_summary['n_fail']} / REVIEW {product_summary['n_review']})")
    if product_summary["skipped_steps_no_image"]:
        print(f"사진 없어 건너뛴 스텝(조용히 누락 아님, 명시): {product_summary['skipped_steps_no_image']}")

    if args.verbose:
        print("\n--- 스텝별 상세 ---")
        for step in sorted(step_decisions):
            d = step_decisions[step]
            print(f"step {step} [{d['part_name'] or '체크포인트 미확인'}] -> {d['status']}")
            print(f"    대표: module={d['representative_module']} score={_score_str(d['representative_score'])} "
                  f"threshold={_score_str(d['representative_threshold'])}")
            print(f"    판정근거: {d['decision_source']}" + (f" | 사유: {d['reason']}" if d["reason"] else ""))
            print(f"    실행모듈: {d['modules_executed']} | mapping_verified={d['mapping_verified']} "
                  f"module_validated={d['module_validated']}")
        if ng_decisions:
            print("\n--- NG 폴더 사진 ---")
            for d in ng_decisions:
                print(f"[{d['image_path']}] step{d['step']} -> {d['status']} "
                      f"(module={d['representative_module']}, score={_score_str(d['representative_score'])})")

    print(f"\n[타이밍] 아티팩트 캐시 구축 {timing['cache_build_s']}s, 전체 추론 wall {timing['infer_wall_s']}s, "
          f"모듈호출 평균 {timing['per_module_call_mean_s']}s / p95 {timing['per_module_call_p95_s']}s "
          f"({timing['n_module_calls']}회)")
    if timing["rss_mb_before"] is not None:
        print(f"[메모리] 시작 {timing['rss_mb_before']}MB -> 종료 {timing['rss_mb_after']}MB")
    elif timing["memory_note"]:
        print(f"[메모리] {timing['memory_note']}")

    out_dir = Path(args.out_dir) if args.out_dir else Path("results/infer_product") / args.machine / args.product
    save_results(out_dir, args.machine, args.product, step_decisions, ng_decisions, product_summary, timing)
    print(f"\n결과 저장: {out_dir}/result.json, {out_dir}/result.csv")


if __name__ == "__main__":
    main()
