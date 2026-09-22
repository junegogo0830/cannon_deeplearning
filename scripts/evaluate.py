"""평가 엔트리포인트: 검증/테스트 세트에서 스텝별 지표와 시각화를 만든다.

실행: python -m scripts.evaluate --config configs/config.yaml --machine MODEL_A --run <run_name>
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained models.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--machine", required=True)
    parser.add_argument("--run", required=True, help="results/ 하위 학습 결과 폴더명")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: 저장된 모델·threshold 로드 → 스텝별 스코어 계산 → evaluate_step
    # TODO: 스코어 분포/ROC/히트맵 저장 → reports/ 에 요약 리포트
    raise NotImplementedError


if __name__ == "__main__":
    main()
