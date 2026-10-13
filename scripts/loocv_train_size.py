"""PaDiM train 크기(max_train) 선택을 leave-one-out으로 검증 — 실제 NG가 7건뿐이라
일반적인 train/val/test 3분할이 불가능할 때 쓰는 방법 (configs/config.yaml의
`real_fail_eval_method: leave_one_out` 설정을 실제로 적용한 것).

전제: scripts/validate_padim_pipeline.py 류로 train 크기별(60/150/300/500/700) 스코어를
이미 뽑아 `results/compare_full/train_size_wide_sweep_full.csv`에 저장해 뒀다는 것.
이 스크립트는 새로 스코어링하지 않고, 그 결과로 "하이퍼파라미터 선택 → 미검증 사례 적용"을
leave-one-out으로 재구성해 누수 없는 검출률을 계산한다.

실행: python -m scripts.loocv_train_size
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd

LOG_PATH = Path("results/train_size_wide_sweep.log")
CONFIGS = [60, 150, 300, 500, 700]


def load_examples(log_path: Path) -> pd.DataFrame:
    """로그에서 (train 크기, 스텝, 실제 NG 개별 사진)별 검출 여부를 펼친다."""
    rows = []
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"train_size_wide: (\{.*\})", line)
            if not m:
                continue
            d = ast.literal_eval(m.group(1))
            if "machine" not in d:
                continue
            for i, score in enumerate(d["fail_scores"]):
                rows.append({
                    "max_train": d["max_train"], "step": d["step"], "idx": i,
                    "example_id": f"{d['step']}_{i}", "score": score, "thr": d["thr"],
                    "caught": int(score > d["thr"]),
                })
    df = pd.DataFrame(rows).drop_duplicates(subset=["max_train", "example_id"])
    return df.pivot_table(index="example_id", columns="max_train", values="caught")


def loocv(matrix: pd.DataFrame) -> pd.DataFrame:
    """예제마다: 나머지로 최선의 max_train을 고르고(동점이면 더 작은 값 우선), 그 설정을
    held-out 예제에 적용한 결과를 기록한다. 정답은 설정 선택에 절대 쓰이지 않는다."""
    examples = list(matrix.index)
    results = []
    for held_out in examples:
        others = [e for e in examples if e != held_out]
        counts = {c: int(matrix.loc[others, c].sum()) for c in CONFIGS}
        best = max(counts.values())
        chosen = min(c for c in CONFIGS if counts[c] == best)
        caught = int(matrix.loc[held_out, chosen])
        results.append({
            "held_out": held_out, "chosen_max_train": chosen,
            "counts_on_remaining6": counts, "result": "CAUGHT" if caught else "MISS",
        })
    return pd.DataFrame(results)


def main() -> None:
    matrix = load_examples(LOG_PATH)
    print(f"실제 NG 예제 {len(matrix)}개:", list(matrix.index))
    print(matrix, "\n")

    result = loocv(matrix)
    print(result[["held_out", "chosen_max_train", "result"]].to_string(index=False))

    n_caught = (result["result"] == "CAUGHT").sum()
    n_total = len(result)
    print(f"\nLOOCV 검출률(누수 없음): {n_caught}/{n_total} = {n_caught/n_total*100:.1f}%")

    Path("results/compare_full").mkdir(parents=True, exist_ok=True)
    result.to_csv("results/compare_full/loocv_train_size_result.csv", index=False)


if __name__ == "__main__":
    main()
