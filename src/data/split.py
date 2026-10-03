"""학습/검증 분할.

FAIL 이 극히 적으므로 학습에는 PASS 만 사용한다 (get_step_paths가 이미 PASS/FAIL을 분리해 줌).
여기서는 PASS 경로 목록을 train/val로만 나눈다.
"""
from __future__ import annotations

import random
from pathlib import Path


def split_pass_paths(
    pass_paths: list[Path], val_ratio: float, seed: int
) -> tuple[list[Path], list[Path]]:
    """PASS 경로 목록을 (train, val) 로 섞어서 나눈다.

    제품 수가 극소(2~9개)인 기종은 val이 0이 될 수 있다 — 이 경우 threshold는
    train 스코어 자체로 산출하게 되므로 신뢰도가 낮음을 결과에 표시해야 한다.
    """
    paths = list(pass_paths)
    rng = random.Random(seed)
    rng.shuffle(paths)

    if len(paths) <= 1:
        return paths, []

    n_val = int(round(len(paths) * val_ratio))
    n_val = min(max(n_val, 0), len(paths) - 1)  # 최소 1개는 train에 남긴다
    val = paths[:n_val]
    train = paths[n_val:]
    return train, val


def reserve_val_then_train(
    pass_paths: list[Path], val_ratio: float, min_val: int, max_train: int | None, seed: int
) -> tuple[list[Path], list[Path]]:
    """threshold 산출용 val을 먼저 넉넉히 확보하고, 남는 것 중 최대 max_train개만 학습(뱅크)에 쓴다.

    percentile(예: 99%) threshold를 val 12장 정도로 잡으면 사실상 "12개 중 제일 큰 값 근처"밖에
    안 돼서, 훨씬 큰 정상 모집단의 실제 꼬리 분포를 전혀 대변하지 못한다(실측으로 확인됨 —
    val 기준 threshold가 정상 홀드아웃 p99보다 한참 낮게 나옴). max_train으로 뱅크 크기를
    줄이더라도 val 크기는 별개로 min_val 이상을 확보해서 threshold 신뢰도를 지킨다.

    Args:
        min_val: val에 최소 이만큼은 확보한다 (표본이 충분할 때).
        max_train: 남은 것 중 뱅크에 쓸 상한. None이면 남는 것 전부 사용.
    """
    paths = list(pass_paths)
    rng = random.Random(seed)
    rng.shuffle(paths)

    if len(paths) <= 2:
        return paths, []  # 극소 표본 기종: train만, val 없음 (호출부가 폴백 처리)

    val_target = max(min_val, int(round(len(paths) * val_ratio)))
    val_target = min(val_target, len(paths) - 1)  # 최소 1장은 train에 남긴다
    val_paths = paths[:val_target]
    remaining = paths[val_target:]

    if max_train is not None and len(remaining) > max_train:
        remaining = remaining[:max_train]
    return remaining, val_paths


def product_key(path: str | Path) -> str:
    """이미지가 속한 제품 식별자 '기종/제품시리얼'. NG 폴더 안 사진도 같은 제품으로 본다."""
    parent = Path(path).parent
    if parent.name.upper() == "NG":
        parent = parent.parent
    return f"{parent.parent.name}/{parent.name}"


def assert_product_disjoint(**groups: list) -> None:
    """서로 다른 분할(train/val/holdout 등)에 같은 제품의 사진이 섞였는지 검사한다.

    같은 제품의 일반·NG·재촬영 사진이 여러 분할에 걸치면 평가가 부풀려진다(누수).
    현재 스텝별 모델은 제품당 루트 사진이 스텝마다 1장뿐이라 자연스럽게 분리되지만,
    나중에 스텝을 공유하는 모델을 만들 때를 대비해 검사를 남겨 둔다.
    """
    owner: dict[str, str] = {}
    for name, paths in groups.items():
        for p in paths:
            k = product_key(p)
            if k in owner and owner[k] != name:
                raise ValueError(f"제품 {k} 가 {owner[k]} 와 {name} 에 동시에 있음 (누수)")
            owner[k] = name
