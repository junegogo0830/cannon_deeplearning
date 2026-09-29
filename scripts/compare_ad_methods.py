"""세 가지 이상탐지 방법 비교 실험: patch_knn(메인) vs padim vs autoencoder.

프로덕션 파이프라인(src/pipeline.py)과 별개의 비교실험 스크립트. 최신 결론(정렬을 hard gate로
쓰지 않기로 함)에 맞춰 셋 다 "정렬 없이" 같은 프로토콜로 평가한다:
  - patch_knn, padim: frozen CNN(MobileNetV3-Small) 패치 임베딩 사용 (layer 4, 위치 그리드 공유)
  - autoencoder: 원본 픽셀을 직접 재구성 (별도 소형 입력 크기)

patch_knn은 위치 무관(뱅크 전체 탐색), padim은 "그리드 위치 = 같은 물리적 부위"를 가정한다.
정렬을 안 쓰는 상황에서 이 가정이 얼마나 깨지는지 확인하는 게 이 비교의 핵심 포인트다.

실행: python -m scripts.compare_ad_methods --machine 3029C003AA --steps 2 14
"""
from __future__ import annotations

# Windows(이 개발 환경) DLL 로드 순서 문제 회피 — cv2/numpy/pandas보다 먼저 import 해야 함.
import torch  # noqa: F401,E402

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import get_step_paths, read_image
from src.data.preprocess import preprocess
from src.data.split import reserve_val_then_train
from src.eval.metrics import auroc, aupr
from src.eval.synthetic import generate_synthetic_fail
from src.features.embeddings import build_backbone, extract_embeddings
from src.models.autoencoder import ae_score, train_autoencoder
from src.models.padim import PaDiMModel
from src.models.patch_knn import PatchKNNModel
from src.scoring.threshold import threshold_percentile
from src.utils.config import load_config
from src.utils.log import get_logger

ROI = None
SIZE = (384, 512)    # patch_knn / padim (CNN 임베딩 입력)
AE_SIZE = (96, 128)  # autoencoder 입력 (CPU 속도용으로 더 작게, 4:3 비율 유지, 16의 배수)
LAYER = 4            # patch_knn / padim 공용 CNN 레이어 (mobilenet_v3_small.features 인덱스)


def load_images(paths: list[Path], size: tuple[int, int]) -> list[np.ndarray]:
    return [preprocess(read_image(p), ROI, size) for p in paths]


def run_step(cfg, machine: str, step: int, max_train: int, min_val: int, n_synthetic: int, backbone, out_dir: Path, logger):
    pass_paths, fail_paths = get_step_paths(cfg, machine, step)
    train_paths, val_paths = reserve_val_then_train(
        pass_paths, 0.2, min_val, max_train, cfg["project"]["seed"]
    )
    used = {str(p) for p in train_paths} | {str(p) for p in val_paths}
    holdout_paths = [p for p in pass_paths if str(p) not in used][:300]  # 속도용 상한

    logger.info(
        f"[{machine} step{step}] train={len(train_paths)} val={len(val_paths)} "
        f"holdout={len(holdout_paths)} real_fail={len(fail_paths)}"
    )

    def embed(imgs: list[np.ndarray]) -> np.ndarray:
        return np.stack([extract_embeddings(backbone, im, [LAYER])[LAYER] for im in imgs])

    t0 = time.time()
    train_feats = embed(load_images(train_paths, SIZE))
    val_feats = embed(load_images(val_paths, SIZE))
    logger.info(f"  임베딩 추출(train+val): {time.time()-t0:.1f}s, feat shape={train_feats.shape}")

    pk = PatchKNNModel(k=3, coreset_ratio=0.05)
    pk.fit(train_feats.reshape(-1, train_feats.shape[-1]))
    thr_pk = threshold_percentile(pk.score(val_feats).max(axis=1), 99.0)

    pdm = PaDiMModel(reg_eps=1e-2)
    pdm.fit(train_feats)
    thr_pd = threshold_percentile(pdm.score(val_feats).max(axis=1), 99.0)

    t0 = time.time()
    ae_model = train_autoencoder(load_images(train_paths, AE_SIZE), epochs=25, batch_size=16)
    logger.info(f"  AE 학습: {time.time()-t0:.1f}s")
    thr_ae = threshold_percentile(
        np.array([ae_score(ae_model, im) for im in load_images(val_paths, AE_SIZE)]), 99.0
    )

    rows = []

    def add(kind: str, label: int, path, img: np.ndarray):
        e = extract_embeddings(backbone, preprocess(img, ROI, SIZE), [LAYER])[LAYER][None, ...]
        s_pk = pk.score(e).max(axis=1)[0]
        s_pd = pdm.score(e).max(axis=1)[0]
        s_ae = ae_score(ae_model, preprocess(img, ROI, AE_SIZE))
        rows.append({
            "kind": kind, "label": label, "path": str(path),
            "patch_knn": float(s_pk), "padim": float(s_pd), "autoencoder": float(s_ae),
        })

    for p in holdout_paths:
        add("pass_holdout", 0, p, read_image(p))
    for p in fail_paths:
        add("real_fail", 1, p, read_image(p))
    synth_methods = cfg["evaluation"]["synthetic_anomaly"]["methods"]
    synth_src = (holdout_paths or pass_paths)[:n_synthetic]
    for i, p in enumerate(synth_src):
        method = synth_methods[i % len(synth_methods)]
        add(f"synthetic_{method}", 1, p, generate_synthetic_fail(read_image(p), method, seed=i))

    df = pd.DataFrame(rows)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"step{step}_scores.csv", index=False)

    thresholds = {"patch_knn": thr_pk, "padim": thr_pd, "autoencoder": thr_ae}
    summary = {}
    for method, thr in thresholds.items():
        y, s = df.label.values, df[method].values
        holdout = df[df.kind == "pass_holdout"]
        real = df[df.kind == "real_fail"]
        summary[method] = {
            "threshold": thr,
            "auroc": auroc(y, s),
            "aupr": aupr(y, s),
            "fpr_on_holdout": float((holdout[method] > thr).mean()) if len(holdout) else float("nan"),
            "n_real_fail": len(real),
            "n_real_fail_caught": int((real[method] > thr).sum()) if len(real) else 0,
        }
    return summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare patch_knn vs padim vs autoencoder.")
    p.add_argument("--config", default="configs/config_cnn.yaml")
    p.add_argument("--machine", required=True)
    p.add_argument("--steps", type=int, nargs="+", required=True)
    p.add_argument("--max-train", type=int, default=60)
    p.add_argument("--min-val", type=int, default=150)
    p.add_argument("--n-synthetic", type=int, default=15)
    p.add_argument("--out", default="results/compare")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    logger = get_logger("compare")
    backbone = build_backbone("mobilenet_v3_small")
    out_dir = Path(args.out) / args.machine

    all_rows = []
    for step in args.steps:
        summary = run_step(cfg, args.machine, step, args.max_train, args.min_val, args.n_synthetic, backbone, out_dir, logger)
        for method, m in summary.items():
            all_rows.append({"step": step, "method": method, **m})
            logger.info(f"  step{step} {method}: {m}")

    out = pd.DataFrame(all_rows)
    print(out.to_string(index=False))
    out.to_csv(out_dir / "summary.csv", index=False)


if __name__ == "__main__":
    main()
