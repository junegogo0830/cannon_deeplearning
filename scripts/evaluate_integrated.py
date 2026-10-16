"""통합 파이프라인(src/inspection/*) 재검증 — 섹션 10 요구사항.

1. 실제 NG 12장 전부(3개 기종, data/processed/eda/ng_events.csv가 유일한 소스)를 통합
   추론기에 넣어 "기존 범용(D) 판정"과 "새 통합 판정"을 나란히 기록한다.
2. 전용 모듈이 검증된 두 셀(3029C003AA step2, 3029C009AA step3)에서 모듈별/통합 정상
   오탐율(FPR)을 측정한다. 홀드아웃 이미지는 scripts/evaluate.py가 이미 계산해 저장해 둔
   results/all13/<기종>/step_<n>/eval_scores.csv 의 pass_holdout 행(=train/val과 겹치지
   않는 독립 홀드아웃, D 기준으로 이미 검증됨)을 그대로 재사용한다 — val로 정한 임계값을
   같은 val에 다시 적용한 수치를 "독립 holdout FPR"이라 부르지 않기 위함이다.

주의: 모듈 A/B(validate_presence_roi.py, validate_nameplate_ocr.py)는 자체 학습에 쓴
정확한 이미지 목록을 아티팩트에 저장해 두지 않는다(meta.json에 train_paths 없음). 그래서
아래 A/B 홀드아웃 FPR은 "D 기준 독립 홀드아웃"일 뿐, A/B 자신의 학습 이미지와 100% 안
겹친다는 보장은 못 한다 — 이 한계를 결과에 그대로 표시한다. A/B 자체의 진짜 독립 holdout
수치는 validate_presence_roi.py/validate_nameplate_ocr.py 실행 로그(report.md §3.3.3,
§3.3.4 — A 약 1%, B 0%/209장)가 더 신뢰도 높다.

실행: python -m scripts.evaluate_integrated [--max-holdout 300]
"""
from __future__ import annotations

import torch  # noqa: F401,E402

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import pandas as pd

from src.data.dataset import load_ng_events
from src.inspection.artifacts import ArtifactCache
from src.inspection.decision import aggregate_step
from src.inspection.registry import InspectionRegistry
from src.inspection.router import run_step
from src.utils.config import load_config

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

RAW_DIR = Path("data/raw")
OUT_DIR = Path("results/evaluate_integrated")

# 전용 모듈이 실제 NG로 검증된 두 셀만 홀드아웃 FPR을 잰다(§3.9의 나머지 190여 셀은
# 전부 범용 D뿐이라 이미 results/all13/*/eval_scores.csv 수치 그대로임 — 다시 돌릴 필요 없음).
VALIDATED_CELLS = [("3029C003AA", 2), ("3029C009AA", 3)]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="통합 파이프라인 NG 전수 + 정상 홀드아웃 재검증")
    p.add_argument("--rules", default="configs/inspection_rules.yaml")
    p.add_argument("--config", default="configs/config_cnn.yaml")
    p.add_argument("--max-holdout", type=int, default=None, help="셀당 평가할 홀드아웃 최대 수(속도 조절용, 기본 전부)")
    return p.parse_args()


def revalidate_all_ng(registry: InspectionRegistry, cache: ArtifactCache) -> list[dict]:
    ng_df = load_ng_events(cache.cfg)
    rows = []
    for row in ng_df.itertuples():
        machine, product, step, filename = row.machine_type, row.product_id, int(row.step), row.filename
        img_path = RAW_DIR / machine / product / "NG" / filename
        if not img_path.is_file():
            rows.append({"machine": machine, "product": product, "step": step, "file_missing": True})
            continue
        rr = run_step(img_path, machine, step, registry, cache)
        dec = aggregate_step(rr)
        d_only = rr["results"].get("D", {})
        rows.append({
            "machine": machine, "product": product, "step": step, "file_missing": False,
            "modules_executed": ",".join(dec["modules_executed"]),
            "generic_d_status": d_only.get("status"), "generic_d_score": d_only.get("raw_score"),
            "integrated_status": dec["status"], "integrated_module": dec["representative_module"],
            "integrated_score": dec["representative_score"], "reason": dec["reason"],
            "caught_by_generic": d_only.get("status") == "FAIL",
            "caught_by_integrated": dec["status"] == "FAIL",
            "newly_caught": dec["status"] == "FAIL" and d_only.get("status") != "FAIL",
            "still_missed": dec["status"] != "FAIL",
            "review": dec["status"] == "REVIEW",
        })
    return rows


def holdout_paths_for(machine: str, step: int, max_n: int | None) -> list[Path]:
    csv_path = Path("results/all13") / machine / f"step_{step}" / "eval_scores.csv"
    df = pd.read_csv(csv_path)
    holdout = df[df.kind == "pass_holdout"]["path"].tolist()
    if max_n is not None:
        holdout = holdout[:max_n]
    return [Path(p) for p in holdout]


def revalidate_holdout_fpr(
    machine: str, step: int, registry: InspectionRegistry, cache: ArtifactCache, max_n: int | None,
) -> dict:
    paths = holdout_paths_for(machine, step, max_n)
    n = len(paths)
    counts = {"D": 0, "A": 0, "B": 0, "integrated_FAIL": 0, "integrated_REVIEW": 0, "integrated_PASS": 0}
    t0 = time.perf_counter()
    for p in paths:
        rr = run_step(p, machine, step, registry, cache)
        dec = aggregate_step(rr)
        for mod in ("D", "A", "B"):
            r = rr["results"].get(mod)
            if r is not None and r["status"] == "FAIL":
                counts[mod] += 1
        counts[f"integrated_{dec['status']}"] += 1
    elapsed = time.perf_counter() - t0
    return {
        "machine": machine, "step": step, "n": n,
        "fpr_D": counts["D"] / n if n else None,
        "fpr_A": counts["A"] / n if n else None,
        "fpr_B": counts["B"] / n if n else None,
        "fpr_integrated": counts["integrated_FAIL"] / n if n else None,
        "review_rate_integrated": counts["integrated_REVIEW"] / n if n else None,
        "elapsed_s": round(elapsed, 2),
        "caveat": "A/B는 자체 학습이미지 목록이 기록돼 있지 않아 D 기준 독립 홀드아웃과 100% 분리된다는 "
                  "보장이 없음 - 참고용. A/B 자체 독립 holdout 수치는 report.md §3.3.3/§3.3.4 참고(A 약 1%, B 0%).",
    }


def main() -> None:
    args = parse_args()
    registry = InspectionRegistry(args.rules)
    cache = ArtifactCache(args.config)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 1) 실제 NG 12장 전수 재검증 ===")
    ng_rows = revalidate_all_ng(registry, cache)
    with open(OUT_DIR / "ng_revalidation.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(ng_rows[0].keys()))
        w.writeheader()
        w.writerows(ng_rows)
    n_caught_generic = sum(r.get("caught_by_generic", False) for r in ng_rows)
    n_caught_integrated = sum(r.get("caught_by_integrated", False) for r in ng_rows)
    n_review = sum(r.get("review", False) for r in ng_rows)
    n_newly = sum(r.get("newly_caught", False) for r in ng_rows)
    print(f"기존 범용(D)만 썼을 때 검출: {n_caught_generic}/{len(ng_rows)}")
    print(f"통합 파이프라인(A/B 우선) 검출: {n_caught_integrated}/{len(ng_rows)} (그 중 통합이 새로 잡은 것: {n_newly})")
    print(f"REVIEW로 넘어간 건수: {n_review}")
    for r in ng_rows:
        if r.get("file_missing"):
            continue
        tag = "NEW" if r["newly_caught"] else ("OK" if r["caught_by_integrated"] else ("REVIEW" if r["review"] else "MISS"))
        print(f"  [{tag:6s}] {r['machine']} {r['product']} step{r['step']}: D={r['generic_d_status']} -> 통합={r['integrated_status']}"
              f"(module={r['integrated_module']})")

    print("\n=== 2) 전용 모듈 검증 셀의 정상 홀드아웃 FPR (D vs A/B vs 통합) ===")
    fpr_rows = []
    for machine, step in VALIDATED_CELLS:
        res = revalidate_holdout_fpr(machine, step, registry, cache, args.max_holdout)
        fpr_rows.append(res)
        print(f"{machine} step{step} (n={res['n']}, {res['elapsed_s']}s): "
              f"D={res['fpr_D']:.4f} A={res['fpr_A']} B={res['fpr_B']} "
              f"integrated={res['fpr_integrated']:.4f} review={res['review_rate_integrated']:.4f}")

    with open(OUT_DIR / "holdout_fpr.json", "w", encoding="utf-8") as f:
        json.dump(fpr_rows, f, ensure_ascii=False, indent=2)

    summary = {
        "ng_total": len(ng_rows), "ng_caught_generic_only": n_caught_generic,
        "ng_caught_integrated": n_caught_integrated, "ng_newly_caught_by_integration": n_newly,
        "ng_review": n_review, "holdout_fpr": fpr_rows,
    }
    with open(OUT_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n결과 저장: {OUT_DIR}/ng_revalidation.csv, {OUT_DIR}/holdout_fpr.json, {OUT_DIR}/summary.json")


if __name__ == "__main__":
    main()
