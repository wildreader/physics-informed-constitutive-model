"""从微球自由能到 Cauchy 应力的解析物理层。"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


def stress_vec_to_mat(vector: torch.Tensor) -> torch.Tensor:
    """[S11,S22,S33,S12,S13,S23] -> 对称 3x3 张量。"""
    if vector.ndim != 2 or vector.shape[1] != 6:
        raise ValueError(
            f"应力向量形状应为 [batch, 6]，当前为 {tuple(vector.shape)}。"
        )
    row1 = torch.stack([vector[:, 0], vector[:, 3], vector[:, 4]], dim=1)
    row2 = torch.stack([vector[:, 3], vector[:, 1], vector[:, 5]], dim=1)
    row3 = torch.stack([vector[:, 4], vector[:, 5], vector[:, 2]], dim=1)
    return torch.stack([row1, row2, row3], dim=1)


def stress_mat_to_vec(matrix: torch.Tensor) -> torch.Tensor:
    """对称 3x3 张量 -> [S11,S22,S33,S12,S13,S23]。"""
    if matrix.ndim != 3 or matrix.shape[1:] != (3, 3):
        raise ValueError(
            f"应力张量形状应为 [batch, 3, 3]，当前为 {tuple(matrix.shape)}。"
        )
    return torch.stack(
        [
            matrix[:, 0, 0],
            matrix[:, 1, 1],
            matrix[:, 2, 2],
            matrix[:, 0, 1],
            matrix[:, 0, 2],
            matrix[:, 1, 2],
        ],
        dim=1,
    )


def deformation_vec_to_mat(vector: torch.Tensor) -> torch.Tensor:
    """[F11,F12,...,F33]（按行展开）-> 3x3 张量。"""
    if vector.ndim != 2 or vector.shape[1] != 9:
        raise ValueError(
            f"F 向量形状应为 [batch, 9]，当前为 {tuple(vector.shape)}。"
        )
    return vector.reshape(-1, 3, 3)


class PhysicsIntegrationLayer(nn.Module):
    """将各方向加权链力积分为宏观未投影应力 S_bar。"""

    def __init__(self, directions: np.ndarray) -> None:
        super().__init__()
        direction_tensor = torch.as_tensor(directions, dtype=torch.float32)
        if direction_tensor.ndim != 2 or direction_tensor.shape[1] != 3:
            raise ValueError(
                "方向数组 Dir 的形状应为 [direction_count, 3]，"
                f"当前为 {tuple(direction_tensor.shape)}。"
            )
        dyadic = direction_tensor.unsqueeze(2) * direction_tensor.unsqueeze(1)
        self.register_buffer("dyadic", dyadic)

    def forward(
        self, weighted_chain_force: torch.Tensor, lambda_current: torch.Tensor
    ) -> torch.Tensor:
        term = weighted_chain_force / (lambda_current + 1.0e-8)
        micro_stress = (
            term.unsqueeze(-1).unsqueeze(-1) * self.dyadic.unsqueeze(0)
        )
        macro_stress = torch.sum(micro_stress, dim=1)
        return stress_mat_to_vec(macro_stress)


class FloryDecouplingProjection(nn.Module):
    """计算 S_iso、S_vol，并组装第二类 Piola-Kirchhoff 应力。"""

    def forward(
        self,
        s_bar_vector: torch.Tensor,
        j_scalar: torch.Tensor,
        c_vector: torch.Tensor,
        pressure: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        c_matrix = stress_vec_to_mat(c_vector)
        s_bar_matrix = stress_vec_to_mat(s_bar_vector)
        c_inverse = torch.linalg.inv(c_matrix)

        c_contract_s_bar = torch.sum(
            c_matrix * s_bar_matrix, dim=(1, 2), keepdim=True
        )
        deviatoric_s_bar = (
            s_bar_matrix - (1.0 / 3.0) * c_contract_s_bar * c_inverse
        )
        j_power = torch.pow(j_scalar.reshape(-1), -2.0 / 3.0).view(-1, 1, 1)
        s_iso_matrix = j_power * deviatoric_s_bar

        if pressure is None:
            # 关闭时跳过体积本构公式，零张量仅用于统一输出接口。
            s_vol_matrix = torch.zeros_like(s_iso_matrix)
            s_total_matrix = s_iso_matrix
        else:
            jp = j_scalar.reshape(-1) * pressure.reshape(-1)
            s_vol_matrix = jp.view(-1, 1, 1) * c_inverse
            s_total_matrix = s_iso_matrix + s_vol_matrix

        return (
            stress_mat_to_vec(s_total_matrix),
            stress_mat_to_vec(s_iso_matrix),
            stress_mat_to_vec(s_vol_matrix),
        )


class PK2ToCauchyLayer(nn.Module):
    """sigma=(1/J) F S F^T。"""

    def forward(
        self, s_vector: torch.Tensor, f_vector: torch.Tensor, j_scalar: torch.Tensor
    ) -> torch.Tensor:
        s_matrix = stress_vec_to_mat(s_vector)
        f_matrix = deformation_vec_to_mat(f_vector)
        j_column = j_scalar.reshape(-1, 1, 1)
        sigma_matrix = (
            torch.bmm(torch.bmm(f_matrix, s_matrix), f_matrix.transpose(1, 2))
            / j_column
        )
        return stress_mat_to_vec(sigma_matrix)
