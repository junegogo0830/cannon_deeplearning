"""평가 엔트리포인트: 독립 홀드아웃 PASS(FPR) + 실제 FAIL(있으면) + 합성 결함으로 스텝별 성능을 확인한다.

실행: python -m scripts.evaluate --config configs/config.yaml --machine 3029C003AA --steps 2 7 14 --run demo1

주의: FAIL 라벨이 있는 기종은 13개 중 3개뿐이고 그마저도 표본이 매우 적다(전체 12건).
여기서 나오는 AUROC/AUPR은 "실제 FAIL을 제대로 잡는가"의 통계적으로 신뢰할 수 있는 추정치가
아니라 사례별(case-by-case) 확인 자료에 가깝다 — reports/eda_summary.md §3, §8 참고.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import get_step_paths, read_image
from src.eval.metrics import evaluate_all_steps, evaluate_step
from src.eval.report import save_metrics_report, save_score_table
from src.eval.synthetic import generate_synthetic_fail
from src.eval.visualize import plot_score_distribution
from src.pipeline import load_step_artifacts, score_image
from src.utils.config import get_step_ids, load_config
from src.utils.io import get_run_dir
from src.utils.log import get_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained per-step models.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--machine", required=True)
    parser.add_argument("--steps", type=int, nargs="*", default=None)
    parser.add_argument("--run", default="dev")
    parser.add_argument("--n-synthetic", type=int, default=10, help="합성 결함 케이스 수 (스텝당)")
    return parser.parse_args()


def _score_row(path: Path, image, artifacts, label: int, kind: str) -> dict:
    res = score_image(image, artifacts)
    row = {"path": str(path), "label": label, "kind": kind, "ok": res["ok"]}
    if res["ok"]:
        row["score"] = res["score"]
    else:
        row["score"] = np.nan
        row["align_fail_reason"] = res["reason"]
    return row


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger("evaluate")
    steps = args.steps if args.steps is not None else get_step_ids(cfg, args.machine)
    run_dir = get_run_dir(cfg["paths"]["results_dir"], args.run)

    per_step_metrics = {}

    for step in steps:
        artifacts = load_step_artifacts(run_dir, args.machine, step)
        meta = artifacts["meta"]
        threshold = meta["threshold"]

        pass_paths, fail_paths = get_step_paths(cfg, args.machine, step)
        used = set(meta.get("train_paths", [])) | set(meta.get("val_paths", []))
        holdout_paths = [p for p in pass_paths if str(p) not in used]
        used_fallback_holdout = len(holdout_paths) == 0
        if used_fallback_holdout:  # 극소 표본 기종: 독립 홀드아웃이 없으면 val을 재사용 (신뢰도 낮음)
            holdout_paths = [Path(p) for p in meta.get("val_paths", [])]

        rows = []
        for p in holdout_paths:
            rows.append(_score_row(p, read_image(p), artifacts, 0, "pass_holdout"))
        for p in fail_paths:
            rows.append(_score_row(p, read_image(p), artifacts, 1, "real_fail"))

        synth_methods = cfg["evaluation"]["synthetic_anomaly"]["methods"]
        synth_src = (holdout_paths or pass_paths)[: args.n_synthetic]
        for i, p in enumerate(synth_src):
            method = synth_methods[i % len(synth_methods)]
            synth_img = generate_synthetic_fail(read_image(p), method, seed=i)
            rows.append(_score_row(f"{p}::{method}", synth_img, artifacts, 1, f"synthetic_{method}"))

        df = pd.DataFrame(rows)
        save_score_table(df, run_dir / args.machine / f"step_{step}" / "eval_scores.csv")

        valid = df[df.ok]
        if len(valid) > 0 and valid.label.nunique() > 1:
            m = evaluate_step(valid.label.values, valid.score.values, threshold)
        else:
            m = {"auroc": float("nan"), "aupr": float("nan"), "threshold": threshold}

        holdout_valid = valid[valid.kind == "pass_holdout"]
        m["fpr_on_holdout"] = float((holdout_valid.score > threshold).mean()) if len(holdout_valid) else float("nan")
        m["n_align_fail"] = int((~df.ok).sum())
        m["n_pass_holdout"] = int(len(holdout_valid))
        m["n_real_fail"] = int((df.kind == "real_fail").sum())
        m["n_synthetic"] = int(df.kind.str.startswith("synthetic").sum())
        m["holdout_is_val_fallback"] = used_fallback_holdout

        # 실제 FAIL 사례는 정렬 자체가 실패하는 경우가 잦다(결함이 곧 특징점을 없애버리는 경우, 예:
        # 라벨 누락). scripts/infer.py 와 같은 정책으로 "정렬 실패 = FAIL(재검)"까지 caught로 센다.
        real_fail_all = df[df.kind == "real_fail"]
        n_caught = 0
        for r in real_fail_all.itertuples():
            if not r.ok:
                logger.info(f"  실제 FAIL 사례 {r.path}: 정렬 실패({r.align_fail_reason}) → FAIL(재검) 처리, 캐치됨")
                n_caught += 1
            else:
                caught = r.score > threshold
                verdict = "FAIL(잡음)" if caught else "PASS(놓침!)"
                logger.info(f"  실제 FAIL 사례 {r.path}: score={r.score:.4f} threshold={threshold:.4f} → {verdict}")
                n_caught += int(caught)
        m["n_real_fail_caught"] = n_caught
        per_step_metrics[step] = m

        pv = valid[valid.label == 0].score.values
        fv = valid[valid.label == 1].score.values if (valid.label == 1).any() else None
        plot_score_distribution(pv, fv, threshold, run_dir / args.machine / f"step_{step}" / "score_distribution.png")

        logger.info(f"{args.machine} step{step}: {m}")

    metrics_df = evaluate_all_steps(per_step_metrics)
    save_metrics_report(metrics_df, run_dir / args.machine / "_summary")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
