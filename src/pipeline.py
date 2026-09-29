"""(기종, 스텝) 단위 학습/추론 오케스트레이션 — scripts/train.py, evaluate.py, infer.py 공용.

여기 있는 함수들이 지금까지 설계한 전체 흐름을 실제로 연결한다:
정렬(이중 게이트, 실패 시 config.alignment.on_failure 정책에 따라 제외/원본유지)
→ 멀티스케일 패치 특징(handcrafted 그리드 패치 또는 frozen CNN 임베딩)
→ 스케일별 patch_knn 뱅크 → 스케일 내부 top-k 집계 → 정상 검증 분포로 정규화
→ 스케일 간 max 결합 → percentile threshold.
"""
from __future__ import annotations

# Windows(이 개발 환경)에서 재현되는 DLL 로드 순서 문제 회피: cv2/numpy가 먼저 로드된 프로세스에서
# 나중에 torch를 import하면 c10.dll 초기화가 실패한다(torch를 먼저 import하면 문제없이 로드됨).
# cnn_embedding 을 안 쓰더라도 torch는 requirements.txt 의 필수 의존성이라 여기서 먼저 import해도 안전함.
import torch  # noqa: F401  (import 순서 고정이 목적)

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.data.dataset import get_step_paths, read_image
from src.data.preprocess import align_to_reference, preprocess
from src.data.split import split_pass_paths
from src.features.patches import extract_patches, patch_features
from src.models.base import AnomalyModel, build_model
from src.scoring.anomaly_score import aggregate_patch_scores, combine_scales
from src.scoring.threshold import fit_thresholds
from src.utils.io import load_json, save_json


def step_artifact_dir(run_dir: Path, machine_type: str, step: int) -> Path:
    d = Path(run_dir) / machine_type / f"step_{step}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def build_backbone_if_needed(cfg: dict[str, Any]):
    """features.type이 cnn_embedding이면 backbone을 한 번만 만들어 반환한다 (여러 스텝에서 재사용)."""
    if cfg["features"]["type"] != "cnn_embedding":
        return None
    from src.features.embeddings import build_backbone

    return build_backbone(cfg["features"].get("cnn_backbone", "mobilenet_v3_small"))


def _scale_keys(cfg: dict[str, Any]) -> list[int]:
    if cfg["features"]["type"] == "handcrafted":
        return list(cfg["features"]["multi_scale"])
    return list(cfg["features"]["cnn_layers"])


def _compute_patch_pool(image: np.ndarray, cfg: dict[str, Any], backbone: Any) -> dict[int, np.ndarray]:
    """이미지 한 장 → {scale_key: (P, D) 패치 특징}. handcrafted / cnn_embedding 을 여기서 분기한다."""
    ftype = cfg["features"]["type"]
    if ftype == "handcrafted":
        stride = cfg["features"]["stride"]
        return {
            scale: patch_features(extract_patches(image, scale, stride))
            for scale in cfg["features"]["multi_scale"]
        }
    if ftype == "cnn_embedding":
        from src.features.embeddings import extract_embeddings

        return extract_embeddings(backbone, image, cfg["features"]["cnn_layers"])
    raise ValueError(f"알 수 없는 features.type: {ftype}")


def _align_or_fallback(
    image: np.ndarray, reference: np.ndarray, align_cfg: dict[str, Any]
) -> tuple[np.ndarray | None, dict[str, Any], bool]:
    """정렬을 시도하고, 실패 시 align_cfg.on_failure 정책을 적용한다.

    Returns:
        (사용할 이미지 또는 None, 진단정보, 실제로 정렬이 적용됐는지)
        on_failure="exclude" 인데 실패하면 이미지는 None (호출부가 건너뛰어야 함).
        on_failure="keep_unaligned" 이면 실패해도 원본(전처리된) 이미지를 그대로 반환한다.
    """
    aligned, diag = align_to_reference(image, reference, align_cfg)
    if aligned is not None:
        return aligned, diag, True
    if align_cfg.get("on_failure", "exclude") == "keep_unaligned":
        return image, diag, False
    return None, diag, False


def _normalize_with_stats(arr: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    if stats["method"] == "minmax":
        return (arr - stats["lo"]) / (stats["hi"] - stats["lo"] + 1e-8)
    if stats["method"] == "zscore":
        return (arr - stats["mu"]) / (stats["sd"] + 1e-8)
    return arr


def _fit_normalize_stats(arr: np.ndarray, method: str) -> dict[str, Any]:
    if method == "minmax":
        return {"method": "minmax", "lo": float(arr.min()), "hi": float(arr.max())}
    if method == "zscore":
        return {"method": "zscore", "mu": float(arr.mean()), "sd": float(arr.std())}
    return {"method": "none"}


def train_step(
    cfg: dict[str, Any],
    machine_type: str,
    step: int,
    run_dir: Path,
    max_train: int | None = None,
    logger=None,
    backbone: Any = None,
) -> dict[str, Any]:
    """(기종, 스텝) 하나를 학습하고 결과(모델·threshold·메타데이터)를 run_dir 에 저장한다."""
    roi = None  # 아직 미지정 (config.machine_types 참고 — 전체 프레임 사용)
    size = tuple(cfg["data"]["image_size"])
    align_cfg = cfg["alignment"]
    scales = _scale_keys(cfg)
    seed = cfg["project"]["seed"]
    if backbone is None:
        backbone = build_backbone_if_needed(cfg)

    pass_paths, _ = get_step_paths(cfg, machine_type, step)
    if len(pass_paths) < 2:
        raise ValueError(f"{machine_type} step{step}: PASS 이미지가 너무 적습니다 ({len(pass_paths)}장)")

    if max_train is not None and len(pass_paths) > max_train:
        rng = np.random.RandomState(seed)
        idx = sorted(rng.choice(len(pass_paths), size=max_train, replace=False))
        pass_paths = [pass_paths[i] for i in idx]

    train_paths, val_paths = split_pass_paths(pass_paths, cfg["data"]["val_ratio"], seed)
    val_is_fallback = len(val_paths) == 0
    if val_is_fallback:
        val_paths = train_paths  # 극소 표본 기종: val 없으면 train으로 대체 (신뢰도 낮음, meta에 표시)

    reference = preprocess(read_image(train_paths[0]), roi, size)

    aligned_train: list[np.ndarray] = []
    n_excluded = 0
    n_used_unaligned = 0
    for p in train_paths:
        img = preprocess(read_image(p), roi, size)
        used_img, _diag, was_aligned = _align_or_fallback(img, reference, align_cfg)
        if used_img is None:
            n_excluded += 1
            continue
        if not was_aligned:
            n_used_unaligned += 1
        aligned_train.append(used_img)

    if len(aligned_train) < 2:
        raise ValueError(f"{machine_type} step{step}: 정렬 성공한 학습 이미지가 너무 적습니다 ({len(aligned_train)}장)")

    pools_by_scale: dict[int, list[np.ndarray]] = {scale: [] for scale in scales}
    for img in aligned_train:
        pool = _compute_patch_pool(img, cfg, backbone)
        for scale in scales:
            pools_by_scale[scale].append(pool[scale])

    models: dict[int, AnomalyModel] = {}
    for scale in scales:
        all_feats = np.concatenate(pools_by_scale[scale], axis=0)
        model = build_model(cfg["model"])
        model.fit(all_feats)
        models[scale] = model

    val_scores_by_scale: dict[int, list[float]] = {scale: [] for scale in scales}
    val_used: list[Path] = []
    for p in val_paths:
        img = preprocess(read_image(p), roi, size)
        used_img, _diag, _was_aligned = _align_or_fallback(img, reference, align_cfg)
        if used_img is None:
            continue
        val_used.append(p)
        pool = _compute_patch_pool(used_img, cfg, backbone)
        for scale in scales:
            patch_scores = models[scale].score(pool[scale][None, ...])[0]
            img_score = aggregate_patch_scores(
                patch_scores[None, :], cfg["scoring"]["patch_aggregation"], cfg["scoring"]["topk"]
            )[0]
            val_scores_by_scale[scale].append(float(img_score))

    raw_val_by_scale = {s: np.array(v) for s, v in val_scores_by_scale.items()}
    norm_method = cfg["scoring"]["normalize"]
    norm_stats = {str(s): _fit_normalize_stats(arr, norm_method) for s, arr in raw_val_by_scale.items()}
    norm_val_by_scale = {s: _normalize_with_stats(arr, norm_stats[str(s)]) for s, arr in raw_val_by_scale.items()}
    combined_val = combine_scales(norm_val_by_scale, method="max")

    threshold = fit_thresholds({step: {"normal": combined_val}}, cfg["threshold"])[step]

    out_dir = step_artifact_dir(run_dir, machine_type, step)
    cv2.imencode(".jpg", reference)[1].tofile(str(out_dir / "reference.jpg"))
    for scale, model in models.items():
        model.save(out_dir / f"model_scale{scale}.pkl")

    meta = {
        "machine_type": machine_type, "step": step,
        "n_pass_total": len(pass_paths), "n_train": len(train_paths),
        "n_train_aligned": len(aligned_train), "n_train_excluded": n_excluded,
        "n_train_unaligned_fallback": n_used_unaligned,
        "n_val": len(val_used), "val_is_fallback": val_is_fallback,
        "roi": roi, "image_size": list(size), "scales": list(scales),
        "features_cfg": cfg["features"], "align_cfg": align_cfg,
        "scoring_cfg": cfg["scoring"], "norm_stats": norm_stats,
        "threshold": threshold,
        "train_paths": [str(p) for p in train_paths],
        "val_paths": [str(p) for p in val_used],
    }
    save_json(meta, out_dir / "meta.json")

    if logger:
        logger.info(
            f"{machine_type} step{step}: pass={len(pass_paths)} train={len(train_paths)}"
            f"(사용 {len(aligned_train)} [정렬대체 {n_used_unaligned}], 제외 {n_excluded}) val={len(val_used)}"
            f"{' [val=train 대체]' if val_is_fallback else ''} threshold={threshold:.4f}"
        )
    return meta


def load_step_artifacts(run_dir: Path, machine_type: str, step: int) -> dict[str, Any]:
    """train_step 이 저장한 결과(모델·기준이미지·메타데이터)를 불러온다."""
    out_dir = step_artifact_dir(run_dir, machine_type, step)
    meta = load_json(out_dir / "meta.json")
    reference = read_image(out_dir / "reference.jpg")
    models = {scale: AnomalyModel.load(out_dir / f"model_scale{scale}.pkl") for scale in meta["scales"]}
    return {"meta": meta, "reference": reference, "models": models}


def score_image(image: np.ndarray, artifacts: dict[str, Any], backbone: Any = None) -> dict[str, Any]:
    """원본 이미지 한 장의 스텝 스코어를 계산한다 (전처리/정렬 포함).

    on_failure="exclude" 이고 정렬이 실패하면 ok=False 를 반환한다.
    on_failure="keep_unaligned" 이면 정렬 실패해도 원본으로 점수를 계산하고 ok=True, aligned=False 로 표시한다.
    """
    meta = artifacts["meta"]
    roi = meta["roi"]
    size = tuple(meta["image_size"])
    cfg_like = {"features": meta["features_cfg"]}

    img = preprocess(image, roi, size)
    used_img, diag, was_aligned = _align_or_fallback(img, artifacts["reference"], meta["align_cfg"])
    if used_img is None:
        return {"ok": False, "reason": diag["reason"], "align_diag": diag, "threshold": meta["threshold"]}

    pool = _compute_patch_pool(used_img, cfg_like, backbone)
    per_scale = {}
    for scale in meta["scales"]:
        model = artifacts["models"][scale]
        patch_scores = model.score(pool[scale][None, ...])[0]
        img_score = aggregate_patch_scores(
            patch_scores[None, :], meta["scoring_cfg"]["patch_aggregation"], meta["scoring_cfg"]["topk"]
        )[0]
        norm_score = _normalize_with_stats(np.array([img_score]), meta["norm_stats"][str(scale)])[0]
        per_scale[scale] = {"raw": float(img_score), "norm": float(norm_score)}

    combined = max(v["norm"] for v in per_scale.values())
    return {
        "ok": True, "aligned": was_aligned, "score": float(combined), "per_scale": per_scale,
        "align_diag": diag, "threshold": meta["threshold"],
    }
