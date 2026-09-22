"""추론 엔트리포인트: 새 이미지 → 15개 스텝 PASS/FAIL + 제품 최종 판정.

실행: python -m scripts.infer --config configs/config.yaml --machine MODEL_A --run <run_name> --image path/to/img.png
"""
from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference on an image.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--machine", required=True)
    parser.add_argument("--run", required=True)
    parser.add_argument("--image", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # TODO: 이미지 로드 → 스텝별 ROI 전처리 → 특징 → 모델 스코어 → judge_step → judge_product
    # TODO: 스텝별 (score, threshold, PASS/FAIL) 출력
    raise NotImplementedError


if __name__ == "__main__":
    main()
