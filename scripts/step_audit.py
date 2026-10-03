"""기종 × 스텝 전수 감사: 일람표 항목 vs 실제 사진 vs 기종 간 유사도.

출력 (data/processed/step_audit/):
  xlsx_items_by_machine_checkpoint.csv : 기종 × 체크포인트(①~⑮) → 일람표 항목명
  machine_sheet_<기종>.jpg            : 기종별 전 스텝 샘플 contact sheet (사진 판독용)
  machine_step_samples.csv            : 기종·스텝별 샘플 경로 (판독 기준)
  cross_machine_similarity.csv        : 기준 기종 스텝 k 와 다른 기종 스텝 j 의 유사도 (평균 상관)
  step_alignment_test.csv             : 기종별 스텝 k 가 기준 기종 중 어느 스텝과 가장 비슷한지
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed/step_audit")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
N_SIM = 12        # 기종·스텝당 유사도 계산에 쓰는 샘플 수
REF_MACHINE = "3029C003AA"
EXCLUDE = {"2EQ17039", "2EQ16165", "2EQ16580", "2EQ17023", "2EX36524", "2EX36527", "2EX36613", "2EX36661",
           "2ES02926", "2EQ16160", "2EQ16973"}  # NG 제품·중첩 폴더 제품 제외 (정상 샘플만)


def norm_small(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(path.read_bytes(), np.uint8), cv2.IMREAD_GRAYSCALE)
    small = cv2.resize(img, (128, 96), interpolation=cv2.INTER_AREA).astype(np.float32)
    small -= small.mean()
    return small / (small.std() + 1e-6)


def main() -> None:
    rng = random.Random(SEED)
    machines = sorted(p.name for p in RAW.iterdir() if p.is_dir())

    # 1) 일람표 피벗
    cp = pd.read_csv("data/processed/eda/xlsx_checkpoint_map.csv")
    cp["item"] = cp.part_name.astype(str).str.replace("\n", " ").str.strip()
    pivot = cp.groupby(["machine_type", "checkpoint"])["item"].apply(lambda s: " / ".join(s)).reset_index()
    pivot.to_csv(OUT / "xlsx_items_by_machine_checkpoint.csv", index=False, encoding="utf-8-sig")

    # 2) 기종·스텝별 샘플 + contact sheet
    samples = []
    pools: dict[tuple[str, int], list[Path]] = {}
    for m in machines:
        steps = sorted({int(p.name.split("_")[1].split(".")[0]) for p in (RAW / m).glob("*/step_*.jpg")})
        for s in steps:
            ps = [p for p in (RAW / m).glob(f"*/step_{s}.jpg") if p.parent.name not in EXCLUDE]
            ps.sort()
            pools[(m, s)] = ps
            samples.append({"machine": m, "step": s, "n_pool": len(ps), "sample": str(rng.choice(ps)).replace("\\", "/")})
    pd.DataFrame(samples).to_csv(OUT / "machine_step_samples.csv", index=False, encoding="utf-8-sig")

    for m in machines:
        rows = [r for r in samples if r["machine"] == m]
        tiles = []
        for r in rows:
            t = cv2.resize(cv2.imdecode(np.frombuffer(Path(r["sample"]).read_bytes(), np.uint8), cv2.IMREAD_COLOR),
                           (256, 192), interpolation=cv2.INTER_AREA)
            cv2.rectangle(t, (0, 0), (256, 28), (0, 0, 0), -1)
            cv2.putText(t, f"s{r['step']}", (6, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            tiles.append(t)
        while len(tiles) % 5:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 5]) for i in range(0, len(tiles), 5)])
        cv2.imwrite(str(OUT / f"machine_sheet_{m}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])

    # 3) 기종 간 유사도: 각 (기종,스텝)의 샘플 N장 평균 벡터를 만들고 상관 비교
    feats: dict[tuple[str, int], np.ndarray] = {}
    for key, ps in pools.items():
        if not ps:
            continue
        pick = rng.sample(ps, min(N_SIM, len(ps)))
        feats[key] = np.mean([norm_small(p) for p in pick], axis=0)
    flat = {k: (v - v.mean()) / (v.std() + 1e-6) for k, v in feats.items()}

    ref_steps = sorted(s for (m, s) in flat if m == REF_MACHINE)
    sim_rows = []
    align_rows = []
    for m in machines:
        for s in sorted(s for (mm, s) in flat if mm == m):
            best = None
            for rs in ref_steps:
                c = float((flat[(m, s)] * flat[(REF_MACHINE, rs)]).mean())
                sim_rows.append({"machine": m, "step": s, "ref_step": rs, "corr": c})
                if best is None or c > best[1]:
                    best = (rs, c)
            align_rows.append({"machine": m, "step": s, "best_ref_step": best[0] if best else None,
                               "best_corr": best[1] if best else None,
                               "same_index_as_ref": (best[0] == s) if best else None,
                               "corr_at_same_index": float((flat[(m, s)] * flat[(REF_MACHINE, s)]).mean())
                               if (REF_MACHINE, s) in flat else None})
    pd.DataFrame(sim_rows).to_csv(OUT / "cross_machine_similarity.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(align_rows).to_csv(OUT / "step_alignment_test.csv", index=False, encoding="utf-8-sig")
    print(f"done: 기종 {len(machines)}, 기종·스텝 {len(pools)}")


if __name__ == "__main__":
    main()
