"""可复用的神经网络子模块和 DeepSets 运算。"""

from __future__ import annotations

import torch
import torch.nn as nn


class LocalFeatureNetwork(nn.Sequential):
    """逐微球方向共享的局部特征提取器：1-32-32-6。"""

    def __init__(self) -> None:
        super().__init__(
            nn.Linear(1, 32),
            nn.Softplus(),
            nn.Linear(32, 32),
            nn.Softplus(),
            nn.Linear(32, 6),
        )


class FreeEnergyPredictor(nn.Sequential):
    """自由能预测器：12-64-64-1，输出非负。"""

    def __init__(self) -> None:
        super().__init__(
            nn.Linear(12, 64),
            nn.Softplus(),
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )


class DamagePredictor(nn.Sequential):
    """损伤预测器：12-64-64-1，输出范围为 [0, 1]。"""

    def __init__(self) -> None:
        super().__init__(
            nn.Linear(12, 64),
            nn.Softplus(),
            nn.Linear(64, 64),
            nn.Softplus(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )


class VolumetricEnergyNetwork(nn.Sequential):
    """体积自由能网络：J_hat -> U_vol，结构为 1-32-32-1。"""

    def __init__(self) -> None:
        super().__init__(
            nn.Linear(1, 32),
            nn.Softplus(),
            nn.Linear(32, 32),
            nn.Softplus(),
            nn.Linear(32, 1),
        )


class FreeEnergyModule(nn.Module):
    """自由能之路：局部提取、加权池化与方向自由能预测。"""

    def __init__(self) -> None:
        super().__init__()
        self.extractor = LocalFeatureNetwork()
        self.predictor = FreeEnergyPredictor()

    def forward(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return deepsets_directional_prediction(
            values, self.extractor, self.predictor, weights
        )


class DamageModule(nn.Module):
    """损伤之路：历史最大伸长驱动的局部/全局损伤预测。"""

    def __init__(self) -> None:
        super().__init__()
        self.extractor = LocalFeatureNetwork()
        self.predictor = DamagePredictor()

    def forward(self, values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return deepsets_directional_prediction(
            values, self.extractor, self.predictor, weights
        )


def deepsets_directional_prediction(
    scalar_input: torch.Tensor,
    local_network: nn.Module,
    predictor: nn.Module,
    normalized_weights: torch.Tensor,
) -> torch.Tensor:
    """执行局部特征提取、球面加权池化和局部/全局耦合预测。"""
    if scalar_input.ndim != 2:
        raise ValueError(
            f"DeepSets 输入应为 [batch, directions]，当前为 {tuple(scalar_input.shape)}。"
        )

    batch_size, direction_count = scalar_input.shape
    local_features = local_network(
        scalar_input.reshape(batch_size * direction_count, 1)
    ).reshape(batch_size, direction_count, 6)

    weight_tensor = normalized_weights.reshape(1, direction_count, 1)
    global_features = torch.sum(
        local_features * weight_tensor, dim=1, keepdim=True
    ).expand(batch_size, direction_count, 6)

    coupled_features = torch.cat([local_features, global_features], dim=-1)
    return predictor(
        coupled_features.reshape(batch_size * direction_count, 12)
    ).reshape(batch_size, direction_count)
