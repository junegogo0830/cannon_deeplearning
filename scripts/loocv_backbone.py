"""백본/레이어 선택(§3.3 5~7번: mobilenet layer4·layer9, resnet18 layer2·layer3)을
leave-one-out으로 검증 — scripts/loocv_train_size.py와 같은 목적, 다른 하이퍼파라미터.

전제: results/backbone_layer_sweep.log 에 4개 조합 × 5개 스텝의 결과가 이미 있다는 것
(scripts/ 실험 스크립트 자체는 /tmp에서 1회성으로 돌렸던 것 — 이 스크립트는 그 결과만 재사용).

실행: python -m scripts.loocv_backbone
"""
from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd

LOG_PATH = Path("results/backbone_layer_sweep.log")
CONFIGS = [("mobilenet_v3_small", 4), ("mobilenet_v3_small", 9), ("resnet18", 2), ("resnet18", 3)]


def load_examples(log_path: Path) -> pd.DataFrame:
    rows = []
    with open(log_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = re.search(r"backbone_sweep: (\{.*\})", line)
            if not m:
                continue
            d = ast.literal_eval(m.group(1))
            config = (d["backbone"], d["layer"])
            n_fail = d["n_real_fail"]
            n_caught = d["n_caught"]
            if n_fail == 0:
                continue
            # 이 실험 범위에서는 n_caught이 전부 0 또는 n_fail 이었음(중간값 없음) -> 예제별로 동일하게 배정.
            # 혹시 중간값이 나오면 (더 큰 점수 쪽부터 caught으로) fail_scores 순서로 추정한다.
            scores = d["fail_scores"]
            if n_caught in (0, n_fail):
                caught_flags = [n_caught == n_fail] * n_fail
            else:
                order = sorted(range(n_fail), key=lambda i: -scores[i])
                caught_flags = [False] * n_fail
                for i in order[:n_caught]:
                    caught_flags[i] = True
            for i in range(n_fail):
                rows.append({
                    "config": config, "step": d["step"], "idx": i,
                    "example_id": f"{d['step']}_{i}", "caught": int(caught_flags[i]),
                })
    df = pd.DataFrame(rows)
    mat = df.pivot_table(index="example_id", columns="config", values="caught")
    mat.columns = list(mat.columns)  # MultiIndex -> 평범한 튜플 컬럼으로
    return mat


def loocv(matrix: pd.DataFrame) -> pd.DataFrame:
    examples = list(matrix.index)
    results = []
    for held_out in examples:
        others = [e for e in examples if e != held_out]
        counts = {c: int(matrix[c].loc[others].sum()) for c in matrix.columns}
        best = max(counts.values())
        # 동점이면 config 목록 순서(= 먼저 시도한 mobilenet layer4)를 우선
        chosen = next(c for c in CONFIGS if c in counts and counts[c] == best)
        caught = int(matrix[chosen].loc[held_out])
        results.append({
            "held_out": held_out, "chosen_config": chosen,
            "counts_on_remaining": counts, "result": "CAUGHT" if caught else "MISS",
        })
    return pd.DataFrame(results)


def main() -> None:
    matrix = load_examples(LOG_PATH)
    print(f"실제 NG 예제 {len(matrix)}개:", list(matrix.index))
    print(matrix, "\n")

    result = loocv(matrix)
    print(result[["held_out", "chosen_config", "result"]].to_string(index=False))

    n_caught = (result["result"] == "CAUGHT").sum()
    n_total = len(result)
    print(f"\nLOOCV 검출률(누수 없음): {n_caught}/{n_total} = {n_caught/n_total*100:.1f}%")

    Path("results/compare_full").mkdir(parents=True, exist_ok=True)
    result.to_csv("results/compare_full/loocv_backbone_result.csv", index=False)


if __name__ == "__main__":
    main()
