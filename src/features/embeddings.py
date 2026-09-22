"""CNN 임베딩 특징 추출 (CPU, 경량 백본).

사전학습 CNN 은 '특징 추출기'로만 사용한다.
PASS/FAIL 판정은 scoring/ 의 자체 스코어링·threshold 로직이 담당한다.
"""
from __future__ import annotations

import numpy as np


def build_backbone(name: str | None):
    """경량 CNN 백본을 로드해 eval 모드(CPU)로 반환한다.

    Args:
        name: 백본 이름 (config.features.cnn_backbone).
    """
    # TODO: 가중치 다운로드 가능 여부 확인, torch.set_num_threads 설정
    raise NotImplementedError


def extract_embeddings(backbone, images: np.ndarray, batch_size: int = 16) -> np.ndarray:
    """이미지 배치 → 임베딩 (N, D) 또는 패치 임베딩 (N, h, w, D).

    Args:
        backbone: build_backbone 결과.
        images: (N, H, W, C) float32.
        batch_size: CPU 메모리 고려한 배치 크기.
    """
    # TODO: torch.no_grad(), 중간 레이어 feature 사용 여부 결정
    raise NotImplementedError
