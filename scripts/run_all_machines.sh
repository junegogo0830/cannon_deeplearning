#!/usr/bin/env bash
# 13개 기종 전체(195개 (기종,스텝) 셀)에 PaDiM+ECC 파이프라인(configs/config_cnn.yaml)을 돌린다.
#
# 배경: 지금까지 깊게 검증한 건 3029C003AA/3029C009AA 두 기종뿐이었다 — 나머지 11개 기종은
# 전혀 돌려본 적이 없었다는 지적에 따라 전체 커버리지를 만든다.
#
# min_val/max_train을 기종마다 다르게 주는 이유: train.py의 min_val 기본값 150은
# 3029C003AA(953장)/3029C009AA(272장) 같은 대량 기종 전용으로 검증된 값이다. 표본이
# 작은 기종(최소 2장!)에 그대로 쓰면 src/data/split.py의 reserve_val_then_train이
# val을 min_val만큼 먼저 떼어가 버려서 train에 1장도 안 남고 ValueError로 전부 스킵된다.
# 그래서 아래처럼 기종별 실제 PASS 이미지 수(n)에 맞춰 규칙적으로 줄인다:
#   n>=200      : min_val=150, max_train=60  (003AA/009AA에서 실측 검증된 설정, LOOCV로도
#                 max_train=60이 150~700보다 나쁘지 않음을 확인함 — scripts/loocv_train_size.py)
#   60<=n<200   : min_val=max(20, n/4), max_train=60
#   20<=n<60    : min_val=max(5, n/5),  max_train 없음(남는 전부 사용)
#   n<20        : min_val=max(2, n/5),  max_train 없음
# n<=2인 기종(3030C002AA)은 src/data/split.py가 별도로 처리(train=val=전체, 신뢰도 낮음으로 표시).
#
# 표본이 극히 작은 기종의 결과는 신뢰도가 낮다 — "데이터가 없어서 못한다"가 아니라 "이 설정으로
# 돌렸을 때 뭐가 나오는지, 어디서 깨지는지"를 전부 기록하는 것이 목적이다 (report.md §3.x 참고).

set -uo pipefail
cd "$(dirname "$0")/.."

CONFIG=configs/config_cnn.yaml
RUN=all13
LOG=results/all13_run.log
: > "$LOG"

# machine min_val max_train(0=없음)
declare -a ROWS=(
  "3029C003AA 150 60"
  "3029C004AA 5 0"
  "3029C005AA 2 0"
  "3029C006AA 2 0"
  "3029C009AA 150 60"
  "3029C010AA 5 0"
  "3030C001AA 150 60"
  "3030C002AA 2 0"
  "3030C003AA 29 60"
  "3030C004AA 2 0"
  "3031C001AA 2 0"
  "3031C002AA 150 60"
  "3031C003AA 150 60"
)

for row in "${ROWS[@]}"; do
  read -r machine min_val max_train <<< "$row"
  echo "=== $machine (min_val=$min_val max_train=$max_train) TRAIN ===" | tee -a "$LOG"
  if [ "$max_train" = "0" ]; then
    python -m scripts.train --config "$CONFIG" --machine "$machine" --run "$RUN" --min-val "$min_val" 2>&1 | tee -a "$LOG"
  else
    python -m scripts.train --config "$CONFIG" --machine "$machine" --run "$RUN" --min-val "$min_val" --max-train "$max_train" 2>&1 | tee -a "$LOG"
  fi
  echo "=== $machine EVALUATE ===" | tee -a "$LOG"
  python -m scripts.evaluate --config "$CONFIG" --machine "$machine" --run "$RUN" --n-synthetic 20 2>&1 | tee -a "$LOG"
done

echo "=== ALL DONE ===" | tee -a "$LOG"
