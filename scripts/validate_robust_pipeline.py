"""최종 설계 검증: ORB+ECC 2단계 정렬 + patch-kNN/PaDiM 앙상블(OR), 신뢰 가능한 2개 기종 전체 스텝.

배경(reports/model_results.md §6): 실제 놓친 NG 3건을 육안 확인한 결과,
  - 결함이 특징점을 지워버려 ORB가 포기한 경우 → ECC(명암 기반)로 구제 가능
  - 진짜 촬영 각도 차이인 경우 → 어떤 정렬도 못 함 (물리적 타당성 게이트가 정상적으로 거름)
  - patch-kNN(위치 무관)과 PaDiM(위치 인식)은 서로 다른 결함 유형에 강함 → OR 앙상블
이 세 가지를 src/data/preprocess.py(ECC 추가)에 정식 반영하고, 게이트 숫자는 원래 엄격한
값 그대로 유지한 채(완화 없음) 여기서 일괄 검증한다.

실행: python -m scripts.validate_robust_pipeline
"""
from __future__ import annotations

import torch  # noqa: F401,E402  (Windows DLL 순서)

from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import get_step_paths, read_image
from src.data.preprocess import align_to_reference, preprocess
from src.data.split import reserve_val_then_train
from src.eval.metrics import auroc
from src.features.embeddings import build_backbone, extract_embeddings
from src.models.padim import PaDiMModel
from src.models.patch_knn import PatchKNNModel
from src.scoring.threshold import threshold_percentile
from src.utils.config import load_config
from src.utils.log import get_logger

SIZE = (384, 512)
LAYER = 4
TARGETS = [("3029C003AA", s) for s in range(1, 15)] + [("3029C009AA", s) for s in range(1, 15)]


def load_aligned(paths: list, ref_bgr: np.ndarray, align_cfg: dict) -> tuple[list, dict]:
    imgs, methods = [], {"orb": 0, "ecc": 0, "none": 0}
    for p in paths:
        img = read_image(p)
        aligned, diag = align_to_reference(img, ref_bgr, align_cfg)
        imgs.append(preprocess(aligned if aligned is not None else img, None, SIZE))
        methods[diag["method"] or "none"] += 1
    return imgs, methods


def main() -> None:
    cfg = load_config("configs/config.yaml")
    align_cfg = cfg["alignment"]  # 원래 엄격한 게이트 그대로
    logger = get_logger("validate_robust")
    backbone = build_backbone("mobilenet_v3_small")

    rows = []
    for machine, step in TARGETS:
        pass_paths, fail_paths = get_step_paths(cfg, machine, step)
        train_paths, val_paths = reserve_val_then_train(pass_paths, 0.2, 150, 60, cfg["project"]["seed"])
        used = {str(p) for p in train_paths} | {str(p) for p in val_paths}
        holdout_paths = [p for p in pass_paths if str(p) not in used][:300]
        ref_bgr = read_image(train_paths[0])

        def embed(imgs: list) -> np.ndarray:
            return np.stack([extract_embeddings(backbone, im, [LAYER])[LAYER] for im in imgs])

        train_imgs, _ = load_aligned(train_paths, ref_bgr, align_cfg)
        val_imgs, _ = load_aligned(val_paths, ref_bgr, align_cfg)
        train_feats, val_feats = embed(train_imgs), embed(val_imgs)

        pk = PatchKNNModel(k=3, coreset_ratio=0.05)
        pk.fit(train_feats.reshape(-1, train_feats.shape[-1]))
        thr_pk = threshold_percentile(pk.score(val_feats).max(axis=1), 99.0)

        pdm = PaDiMModel(reg_eps=1e-2)
        pdm.fit(train_feats)
        thr_pd = threshold_percentile(pdm.score(val_feats).max(axis=1), 99.0)

        holdout_imgs, methods = load_aligned(holdout_paths, ref_bgr, align_cfg)
        holdout_feats = embed(holdout_imgs)
        s_pk_h = pk.score(holdout_feats).max(axis=1)
        s_pd_h = pdm.score(holdout_feats).max(axis=1)
        fail_pk = fail_pd = (s_pk_h > thr_pk) | (s_pd_h > thr_pd)

        n_fail = n_caught = 0
        auc = float("nan")
        if fail_paths:
            fail_imgs, _ = load_aligned(fail_paths, ref_bgr, align_cfg)
            fail_feats = embed(fail_imgs)
            s_pk_f = pk.score(fail_feats).max(axis=1)
            s_pd_f = pdm.score(fail_feats).max(axis=1)
            caught = (s_pk_f > thr_pk) | (s_pd_f > thr_pd)
            n_fail, n_caught = len(caught), int(caught.sum())
            y = np.array([0] * len(s_pk_h) + [1] * len(s_pk_f))
            s_ens = np.concatenate([
                np.maximum(s_pk_h / thr_pk, s_pd_h / thr_pd),
                np.maximum(s_pk_f / thr_pk, s_pd_f / thr_pd),
            ])
            auc = auroc(y, s_ens)

        row = {
            "machine": machine, "step": step,
            "align_orb": methods["orb"], "align_ecc": methods["ecc"], "align_none": methods["none"],
            "fpr_ensemble": round(float(fail_pk.mean()), 4),
            "auroc_ensemble": round(auc, 3) if n_fail else float("nan"),
            "n_real_fail": n_fail, "n_caught": n_caught,
        }
        rows.append(row)
        logger.info(row)

    df = pd.DataFrame(rows)
    Path("results/compare_full").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/compare_full/robust_pipeline_validation.csv", index=False)
    print(df.to_string(index=False))
    print(f"\n평균 FPR(앙상블): {df.fpr_ensemble.mean():.4f}")
    print(f"평균 AUROC(앙상블, NG 있는 스텝만): {df[df.n_real_fail>0].auroc_ensemble.mean():.3f}")
    print(f"실제 NG 검출: {df.n_caught.sum()} / {df.n_real_fail.sum()}")
    print(f"정렬 방법 비율: orb={df.align_orb.sum()} ecc={df.align_ecc.sum()} none={df.align_none.sum()}")


if __name__ == "__main__":
    main()
