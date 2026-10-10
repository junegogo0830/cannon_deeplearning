"""최종 설계 검증: ORB+ECC 2단계 정렬(게이트 완화 없음) + PaDiM 단독, 신뢰 가능한 2개 기종 전체 28개 스텝.

배경(reports/model_results.md §6~7): 놓친 실제 NG 3건을 육안 확인한 결과
  - 결함이 특징점을 지워버려 ORB가 포기한 경우 → ECC(명암 기반, src/data/preprocess.py)로 구제 가능
  - 진짜 촬영 각도 차이인 경우 → 물리적 타당성 게이트가 (의도대로) 거름, 못 고침
  - patch-kNN/PaDiM 앙상블은 시도해봤지만 검출은 그대로에 FPR만 올라가 폐기(scripts/validate_robust_pipeline.py)
→ 최종안은 "ECC 보강 정렬 + PaDiM 단독"이며, 이 스크립트가 그 수치를 재현한다.
이 설계는 src/pipeline.py(build_model("padim"), _prepare_image의 원본 해상도 정렬)에도 반영했다.

실행: python -m scripts.validate_padim_pipeline
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
from src.scoring.threshold import threshold_percentile
from src.utils.config import load_config
from src.utils.log import get_logger

SIZE = (384, 512)
LAYER = 4
TARGETS = [("3029C003AA", s) for s in range(1, 15)] + [("3029C009AA", s) for s in range(1, 15)]


def load_aligned(paths: list, ref_native: np.ndarray, align_cfg: dict) -> list:
    """원본 해상도에서 정렬 후 리사이즈 (게이트가 원본 해상도 기준으로 EDA 실측됐기 때문)."""
    out = []
    for p in paths:
        raw = read_image(p)
        aligned, _diag = align_to_reference(raw, ref_native, align_cfg)
        out.append(preprocess(aligned if aligned is not None else raw, None, SIZE))
    return out


def main() -> None:
    cfg = load_config("configs/config.yaml")
    align_cfg = cfg["alignment"]  # 원래 엄격한 게이트 그대로 (완화하지 않음)
    logger = get_logger("validate_padim")
    backbone = build_backbone("mobilenet_v3_small")

    rows = []
    for machine, step in TARGETS:
        pass_paths, fail_paths = get_step_paths(cfg, machine, step)
        train_paths, val_paths = reserve_val_then_train(pass_paths, 0.2, 150, 60, cfg["project"]["seed"])
        used = {str(p) for p in train_paths} | {str(p) for p in val_paths}
        holdout_paths = [p for p in pass_paths if str(p) not in used][:300]
        ref_native = read_image(train_paths[0])

        def embed(imgs: list) -> np.ndarray:
            return np.stack([extract_embeddings(backbone, im, [LAYER])[LAYER] for im in imgs])

        train_feats = embed(load_aligned(train_paths, ref_native, align_cfg))
        val_feats = embed(load_aligned(val_paths, ref_native, align_cfg))
        pdm = PaDiMModel(reg_eps=1e-2)
        pdm.fit(train_feats)
        thr = threshold_percentile(pdm.score(val_feats).max(axis=1), 99.0)

        holdout_feats = embed(load_aligned(holdout_paths, ref_native, align_cfg))
        s_holdout = pdm.score(holdout_feats).max(axis=1)
        y, s = [0] * len(s_holdout), list(s_holdout)
        n_fail = n_caught = 0
        if fail_paths:
            fail_feats = embed(load_aligned(fail_paths, ref_native, align_cfg))
            s_fail = pdm.score(fail_feats).max(axis=1)
            y += [1] * len(s_fail)
            s += list(s_fail)
            n_fail, n_caught = len(s_fail), int((s_fail > thr).sum())

        row = {
            "machine": machine, "step": step, "threshold": round(float(thr), 2),
            "fpr": round(float((s_holdout > thr).mean()), 4),
            "auroc": round(auroc(np.array(y), np.array(s)), 3) if n_fail else float("nan"),
            "n_real_fail": n_fail, "n_caught": n_caught,
        }
        rows.append(row)
        logger.info(row)

    df = pd.DataFrame(rows)
    Path("results/compare_full").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/compare_full/final_padim_ecc_strict.csv", index=False)
    print(df.to_string(index=False))
    print(
        f"\n평균 FPR: {df.fpr.mean():.4f} | "
        f"평균 AUROC(NG있는 스텝): {df[df.n_real_fail > 0].auroc.mean():.3f} | "
        f"검출: {df.n_caught.sum()}/{df.n_real_fail.sum()}"
    )


if __name__ == "__main__":
    main()
