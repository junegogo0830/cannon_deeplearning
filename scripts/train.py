"""학습 엔트리포인트: 정상(PASS) 데이터로 (기종, 스텝)별 모델을 학습하고 threshold 를 산출한다.

실행: python -m scripts.train --config configs/config.yaml --machine MODEL_A
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train anomaly models per step.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--machine", required=True, help="기종 이름 (config.machine_types 키)")
    parser.add_argument("--steps", type=int, nargs="*", default=None, help="지정 시 해당 스텝만 (기본: 전체 15개)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: load_config → set_seed
    # TODO: 스텝 루프: StepDataset(train) → 특징 추출 → build_model().fit()
    # TODO: 정상 검증 스코어로 fit_thresholds → results/<run>/ 에 모델·threshold 저장
    raise NotImplementedError


if __name__ == "__main__":
    main()
