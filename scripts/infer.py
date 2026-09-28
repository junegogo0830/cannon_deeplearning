"""추론 엔트리포인트: 학습된 (기종,스텝) 아티팩트로 판정한다.

사용법 1) 이미지 한 장 + 스텝 지정:
  python -m scripts.infer --config configs/config.yaml --machine 3029C003AA --run demo1 --image path/to/step_2.jpg --step 2

사용법 2) 제품 시리얼 지정 → 학습된 스텝 전체 판정 + 제품 최종 PASS/FAIL:
  python -m scripts.infer --config configs/config.yaml --machine 3029C003AA --run demo1 --product 2EQ16144
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.data.dataset import read_image
from src.pipeline import load_step_artifacts, score_image
from src.scoring.decision import FAIL, judge_product, judge_step
from src.utils.config import get_step_ids, load_config
from src.utils.io import get_run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--machine", required=True)
    parser.add_argument("--run", default="dev")
    parser.add_argument("--steps", type=int, nargs="*", default=None, help="판정할 스텝 (기본: 학습된 스텝 전체)")
    parser.add_argument("--product", default=None, help="제품 시리얼 폴더명")
    parser.add_argument("--image", default=None, help="단일 이미지 경로 (--step 과 함께 사용)")
    parser.add_argument("--step", type=int, default=None)
    return parser.parse_args()


def _print_step_result(step: int, res: dict) -> str:
    if not res["ok"]:
        print(f"step {step}: 정렬 실패 ({res['reason']}) → 판정 보류, 재검 권장")
        return FAIL  # 정렬 자체가 안 되면 보수적으로 FAIL(재검) 처리
    verdict = judge_step(res["score"], res["threshold"])
    print(f"step {step}: score={res['score']:.4f} threshold={res['threshold']:.4f} -> {verdict}")
    return verdict


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_dir = get_run_dir(cfg["paths"]["results_dir"], args.run)

    if args.image is not None:
        if args.step is None:
            raise SystemExit("--image 사용 시 --step 도 지정해야 합니다.")
        artifacts = load_step_artifacts(run_dir, args.machine, args.step)
        res = score_image(read_image(args.image), artifacts)
        _print_step_result(args.step, res)
        return

    if args.product is None:
        raise SystemExit("--image 또는 --product 중 하나를 지정해야 합니다.")

    raw_dir = Path(cfg["paths"]["raw_dir"])
    steps = args.steps if args.steps is not None else get_step_ids(cfg, args.machine)
    step_results = {}
    for step in steps:
        img_path = raw_dir / args.machine / args.product / f"step_{step}.jpg"
        artifact_dir = run_dir / args.machine / f"step_{step}"
        if not img_path.is_file() or not (artifact_dir / "meta.json").is_file():
            continue
        artifacts = load_step_artifacts(run_dir, args.machine, step)
        res = score_image(read_image(img_path), artifacts)
        step_results[step] = _print_step_result(step, res)

    if step_results:
        product_result = judge_product(step_results, cfg["decision"]["product_rule"])
        print(f"\n제품 {args.product} 최종 판정: {product_result}")
    else:
        print("판정할 수 있는 (학습된) 스텝이 없습니다.")


if __name__ == "__main__":
    main()
