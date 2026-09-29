"""CNN 임베딩 특징 추출 (CPU, 경량 사전학습 백본, frozen).

사전학습 CNN은 '특징 추출기'로만 쓴다 — PASS/FAIL 판정은 여전히 scoring/ 의 자체
patch_knn + threshold 로직이 담당한다 (PatchCore 등 산업 이상탐지 연구와 같은 사용 방식이며,
CNN의 분류 결과를 그대로 판정에 쓰는 게 아니므로 "기성 알고리즘 결과를 그대로 쓰지 않는다"는
조건에 위배되지 않는다).

정렬(ORB) 문제의 근본 원인은 우리가 쓰던 수작업 특징(그래디언트 방향 히스토그램 등)이 미세한
이동/회전에 취약해서 정렬 없이는 못 쓴다는 점이었다. CNN의 conv+pooling 구조는 그 자체로 약간의
이동에 어느 정도 강건하므로, 정렬을 hard gate로 걸지 않고도(설정에 따라) 쓸 수 있는지 실험한다.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch

_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_SUPPORTED = {"mobilenet_v3_small"}


def build_backbone(name: str = "mobilenet_v3_small") -> torch.nn.Module:
    """경량 사전학습 CNN을 로드해 eval 모드(CPU)로 반환한다. 파라미터는 전부 freeze."""
    if name not in _SUPPORTED:
        raise ValueError(f"지원하지 않는 backbone: {name} (사용 가능: {_SUPPORTED})")
    import torchvision.models as tvm

    model = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _to_input_tensor(image: np.ndarray) -> torch.Tensor:
    """BGR uint8 HWC → ImageNet 정규화된 RGB CHW 텐서 (배치 차원 포함)."""
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - _IMAGENET_MEAN) / _IMAGENET_STD
    return torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()


def extract_embeddings(
    backbone: torch.nn.Module, image: np.ndarray, layers: list[int]
) -> dict[int, np.ndarray]:
    """이미지 한 장 → 지정한 레이어들의 패치 임베딩 {layer_idx: (P, D)}.

    한 번의 forward pass로 여러 레이어(=스케일)를 동시에 뽑는다. mobilenet_v3_small.features 는
    순차적 블록들이라, index가 작을수록 해상도 높은(=세밀한) 패치, 클수록 수용영역이 넓은(=거시적) 패치다.
    """
    x = _to_input_tensor(image)
    captured: dict[int, torch.Tensor] = {}
    max_layer = max(layers)
    with torch.no_grad():
        h = x
        for i, block in enumerate(backbone.features):
            h = block(h)
            if i in layers:
                captured[i] = h
            if i == max_layer:
                break

    out: dict[int, np.ndarray] = {}
    for layer, feat in captured.items():
        _, c, hh, ww = feat.shape
        # (1, C, H, W) -> (H*W, C): feature map 의 각 칸이 곧 하나의 "패치"
        out[layer] = feat.squeeze(0).permute(1, 2, 0).reshape(hh * ww, c).numpy()
    return out
