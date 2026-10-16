"""극소표본 기종(PASS<60장/스텝) 전용 — out-of-fold(k-fold) 학습/임계값 산출.

배경: `scripts/run_all_machines.sh`(§3.8)에서 n<60인 6개 기종은 `reserve_val_then_train`이
val을 못 만들어 val=train 폴백을 썼다 — 즉 "채점하는 그 이미지로 이미 학습한 모델"로 다시
채점하는 것이라 FPR이 구조적으로 낙관적인데도 11~42%까지 나왔다(report.md §3.8). 3030C002AA
(n=2)는 폴백 자체가 제품 단위 누수 검사(assert_product_disjoint)와 충돌해 전부 실행 불가였다.

다른 팀원(junsm030627-sudo/canon-project) 저장소의 §A5.9 아이디어를 가져와 적용한다:
"정상 전부로 최종 모델을 만들고, 임계값은 각 이미지를 '그 이미지가 빠진 폴드로 학습한 모델'로
채점한 out-of-fold(OOF) 점수의 percentile로 정한다." 이러면 임계값 산출에 쓰이는 모든 점수는
그 점수를 낸 이미지를 한 번도 보지 않은 모델에서 나온 것이라, val=train 폴백보다 훨씬 덜
낙관적이다(완전한 독립 홀드아웃은 여전히 아니다 — 최종 배포 모델은 전체 데이터로 다시 학습하므로
OOF 점수와 완벽히 같은 분포는 아니다. 이 근사의 한계는 팀원 보고서도 §A5.9에서 동일하게 지적함).

출력 아티팩트(`results/all13_kfold/<기종>/step_<n>/`)는 `src/pipeline.py`의
`load_step_artifacts`/`score_image`가 그대로 읽을 수 있는 형식이라, `scripts/infer.py`로
실제 추론에도 바로 쓸 수 있다.

실행: python -m scripts.kfold_small_machines --config configs/config_cnn.yaml --run all13_kfold
"""
from __future__ import annotations

import torch  # noqa: F401,E402  (Windows DLL 순서)

import argparse
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.data.dataset import get_step_paths, read_image
from src.data.preprocess import crop_roi
from src.models.base import AnomalyModel, build_model
from src.pipeline import (
    _compute_patch_pool,
    _fit_normalize_stats,
    _normalize_with_stats,
    _prepare_image,
    _scale_keys,
    build_backbone_if_needed,
    step_artifact_dir,
)
from src.scoring.anomaly_score import aggregate_patch_scores, combine_scales
from src.scoring.threshold import fit_thresholds
from src.utils.config import get_step_ids, load_config
from src.utils.io import get_run_dir, save_json
from src.utils.log import get_logger
from src.utils.seed import set_seed

# n<60(§3.8 min_val 폴백 대상) 6개 기종 + 구조적으로 전부 실패했던 3030C002AA(n=2).
TARGET_MACHINES = [
    "3029C004AA", "3029C005AA", "3029C006AA",
    "3029C010AA", "3030C002AA", "3030C004AA", "3031C001AA",
]


def _make_folds(n: int, k: int, seed: int) -> list[list[int]]:
    k = max(2, min(k, n))  # 폴드마다 최소 1장, 최대 K=5(또는 n이 더 작으면 n) — n=2는 LOO(K=2)가 됨
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    folds: list[list[int]] = [[] for _ in range(k)]
    for i, v in enumerate(idx):
        folds[i % k].append(v)
    return folds


def _fit_scale_models(images: list[np.ndarray], cfg: dict[str, Any], backbone: Any) -> dict[int, AnomalyModel]:
    scales = _scale_keys(cfg)
    pools_by_scale: dict[int, list[np.ndarray]] = {s: [] for s in scales}
    for img in images:
        pool = _compute_patch_pool(img, cfg, backbone)
        for s in scales:
            pools_by_scale[s].append(pool[s])
    models: dict[int, AnomalyModel] = {}
    for s in scales:
        stacked = np.stack(pools_by_scale[s], axis=0)
        model = build_model(cfg["model"])
        feats = stacked if model.expects_patch_grid else stacked.reshape(-1, stacked.shape[-1])
        model.fit(feats)
        models[s] = model
    return models


def _score_images_raw(
    images: list[np.ndarray], models: dict[int, AnomalyModel], cfg: dict[str, Any], backbone: Any
) -> dict[int, list[float]]:
    scales = _scale_keys(cfg)
    out: dict[int, list[float]] = {s: [] for s in scales}
    for img in images:
        pool = _compute_patch_pool(img, cfg, backbone)
        for s in scales:
            patch_scores = models[s].score(pool[s][None, ...])[0]
            img_score = aggregate_patch_scores(
                patch_scores[None, :], cfg["scoring"]["patch_aggregation"], cfg["scoring"]["topk"]
            )[0]
            out[s].append(float(img_score))
    return out


def kfold_train_step(
    cfg: dict[str, Any], machine_type: str, step: int, run_dir: Path, k: int, backbone: Any, logger=None,
) -> dict[str, Any] | None:
    roi = None
    size = tuple(cfg["data"]["image_size"])
    align_cfg = cfg["alignment"]
    scales = _scale_keys(cfg)
    seed = cfg["project"]["seed"]

    pass_paths, fail_paths = get_step_paths(cfg, machine_type, step)
    n = len(pass_paths)
    if n < 2:
        if logger:
            logger.warning(f"{machine_type} step{step}: PASS {n}장 — 스킵")
        return None

    reference_native = crop_roi(read_image(pass_paths[0]), roi)

    # 전수 정렬/전처리 한 번만 해서 재사용 (폴드마다 다시 정렬하지 않음)
    prepared: list[np.ndarray] = []
    kept_paths: list[Path] = []
    for p in pass_paths:
        used_img, _diag, _was_aligned = _prepare_image(read_image(p), reference_native, roi, align_cfg, size)
        if used_img is not None:
            prepared.append(used_img)
            kept_paths.append(p)
    n_eff = len(prepared)
    if n_eff < 2:
        if logger:
            logger.warning(f"{machine_type} step{step}: 정렬 성공 이미지 {n_eff}장 — 스킵")
        return None

    folds = _make_folds(n_eff, k, seed)
    k_eff = len(folds)

    # PaDiM(위치별 공분산)은 폴드당 학습 이미지가 최소 2장은 있어야 공분산을 추정할 수 있다.
    # n_eff=2(3030C002AA)처럼 폴드당 학습 이미지가 1장뿐이면 공분산이 정의되지 않는다(np.cov가
    # NaN을 낸다) — 이 경우에만 patch-kNN(뱅크 기반, 학습 이미지 1장으로도 계산 가능)으로
    # 바꾼다. 다른 기종(n_eff>=5)은 전부 폴드당 학습 2장 이상이라 PaDiM 그대로 쓴다.
    min_train_size = min(n_eff - len(folds[i]) for i in range(k_eff))
    cfg_fold = cfg
    model_type_used = cfg["model"]["type"]
    if min_train_size < 2:
        cfg_fold = {**cfg, "model": {**cfg["model"], "type": "patch_knn"}}
        model_type_used = "patch_knn"
        if logger:
            logger.info(
                f"{machine_type} step{step}: 폴드당 학습 {min_train_size}장 — PaDiM 공분산 추정 불가, "
                f"patch-kNN으로 대체"
            )

    oof_raw: dict[int, list[float]] = {s: [] for s in scales}
    oof_order: list[int] = []  # prepared 인덱스, 점수와 1:1 대응
    for i in range(k_eff):
        held_idx = folds[i]
        train_idx = [j for j in range(n_eff) if j not in set(held_idx)]
        if len(train_idx) < 1 or len(held_idx) == 0:
            continue
        fold_models = _fit_scale_models([prepared[j] for j in train_idx], cfg_fold, backbone)
        fold_raw = _score_images_raw([prepared[j] for j in held_idx], fold_models, cfg_fold, backbone)
        for s in scales:
            oof_raw[s].extend(fold_raw[s])
        oof_order.extend(held_idx)

    if len(oof_order) < 2:
        if logger:
            logger.warning(f"{machine_type} step{step}: OOF 점수 부족 — 스킵")
        return None

    raw_oof_by_scale = {s: np.array(v) for s, v in oof_raw.items()}
    norm_method = cfg["scoring"]["normalize"]
    norm_stats = {str(s): _fit_normalize_stats(arr, norm_method) for s, arr in raw_oof_by_scale.items()}
    norm_oof_by_scale = {s: _normalize_with_stats(arr, norm_stats[str(s)]) for s, arr in raw_oof_by_scale.items()}
    combined_oof = combine_scales(norm_oof_by_scale, method="max")
    threshold = fit_thresholds({step: {"normal": combined_oof}}, cfg["threshold"])[step]
    oof_fpr = float((combined_oof > threshold).mean())

    # 최종 배포 모델: 정렬 성공한 전체 데이터로 다시 학습 (OOF 폴드 모델보다 뱅크가 큼 —
    # 팀원 보고서 §A5.9와 동일한 근사: 임계값은 OOF로, 점수는 전체뱅크 모델로 낸다)
    final_models = _fit_scale_models(prepared, cfg_fold, backbone)

    out_dir = step_artifact_dir(run_dir, machine_type, step)
    cv2.imencode(".jpg", reference_native)[1].tofile(str(out_dir / "reference.jpg"))
    for s, model in final_models.items():
        model.save(out_dir / f"model_scale{s}.pkl")

    # 실제 NG(있으면) 최종 모델 + OOF 임계값으로 채점 — val=train 폴백과 달리 임계값 쪽은
    # 적어도 그 NG 이미지를 학습에 쓰지 않은 모델들의 점수로 정해진 것이다.
    n_real_fail_caught = 0
    for fp in fail_paths:
        used_img, _diag, _was_aligned = _prepare_image(read_image(fp), reference_native, roi, align_cfg, size)
        if used_img is None:
            n_real_fail_caught += 1  # 정렬 실패=FAIL(재검) 처리 정책과 동일 (scripts/evaluate.py)
            continue
        raw = _score_images_raw([used_img], final_models, cfg_fold, backbone)
        per_scale_norm = [
            _normalize_with_stats(np.array([raw[s][0]]), norm_stats[str(s)])[0] for s in scales
        ]
        score = max(per_scale_norm)
        if score > threshold:
            n_real_fail_caught += 1
        if logger:
            logger.info(f"  실제 FAIL {fp}: score={score:.4f} threshold={threshold:.4f}")

    meta = {
        "machine_type": machine_type, "step": step,
        "n_pass_total": n, "n_pass_aligned": n_eff,
        "kfold_mode": True, "kfold_k": k_eff, "oof_n": len(oof_order), "model_type_used": model_type_used,
        "oof_fpr": oof_fpr, "n_real_fail": len(fail_paths), "n_real_fail_caught": n_real_fail_caught,
        "roi": roi, "image_size": list(size), "scales": list(scales),
        "features_cfg": cfg["features"], "align_cfg": align_cfg,
        "scoring_cfg": cfg["scoring"], "norm_stats": norm_stats,
        "threshold": threshold,
        "train_paths": [str(p) for p in kept_paths],  # 최종 모델 학습에 쓴 전체 — "val" 없음을 명시
        "val_paths": [],
    }
    save_json(meta, out_dir / "meta.json")
    if logger:
        logger.info(
            f"{machine_type} step{step}: n={n}(정렬성공 {n_eff}) k={k_eff} oof_fpr={oof_fpr:.4f} "
            f"threshold={threshold:.4f} real_fail={len(fail_paths)}(검출 {n_real_fail_caught})"
        )
    return meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="극소표본 기종 k-fold OOF 학습")
    p.add_argument("--config", default="configs/config_cnn.yaml")
    p.add_argument("--run", default="all13_kfold")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--machines", nargs="*", default=None, help="기본: TARGET_MACHINES 7개")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    logger = get_logger("kfold_small_machines")
    run_dir = get_run_dir(cfg["paths"]["results_dir"], args.run)
    backbone = build_backbone_if_needed(cfg)

    machines = args.machines if args.machines else TARGET_MACHINES
    rows = []
    for machine in machines:
        for step in get_step_ids(cfg, machine):
            meta = kfold_train_step(cfg, machine, step, run_dir, args.k, backbone, logger=logger)
            if meta:
                rows.append(meta)

    print(f"{'기종':12s} {'스텝':4s} {'n':>5s} {'k':>3s} {'oof_fpr':>8s} {'real_fail':>9s} {'검출':>4s}")
    for m in rows:
        print(
            f"{m['machine_type']:12s} {m['step']:>4d} {m['n_pass_aligned']:>5d} {m['kfold_k']:>3d} "
            f"{m['oof_fpr']*100:7.2f}% {m['n_real_fail']:>9d} {m['n_real_fail_caught']:>4d}"
        )


if __name__ == "__main__":
    main()
