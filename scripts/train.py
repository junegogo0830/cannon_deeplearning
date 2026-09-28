"""학습 엔트리포인트: 정상(PASS) 데이터로 (기종, 스텝)별 이상탐지 모델을 학습하고 threshold 를 산출한다.

실행: python -m scripts.train --config configs/config.yaml --machine 3029C003AA --steps 2 7 14 --run demo1
"""
from __future__ import annotations

import argparse

from src.pipeline import train_step
from src.utils.config import get_step_ids, load_config
from src.utils.io import get_run_dir
from src.utils.log import get_logger
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train per-step anomaly models.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--machine", required=True, help="기종 이름 (config.machine_types 키)")
    parser.add_argument("--steps", type=int, nargs="*", default=None, help="지정 시 해당 스텝만 (기본: 기종 전체)")
    parser.add_argument("--run", default="dev", help="results/<run>/ 에 저장")
    parser.add_argument("--max-train", type=int, default=None, help="학습 PASS 이미지 수 상한 (속도용 샘플링)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["project"]["seed"])
    logger = get_logger("train")

    steps = args.steps if args.steps is not None else get_step_ids(cfg, args.machine)
    run_dir = get_run_dir(cfg["paths"]["results_dir"], args.run)

    for step in steps:
        try:
            train_step(cfg, args.machine, step, run_dir, max_train=args.max_train, logger=logger)
        except ValueError as e:
            logger.warning(str(e))


if __name__ == "__main__":
    main()
