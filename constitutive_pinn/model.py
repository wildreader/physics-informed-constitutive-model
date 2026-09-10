"""始终包含自由能、损伤、体积三个模块的单个本构 PINN。"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn

from .config import ModelArchitecture
from .networks import (
    DamageModule,
    FreeEnergyModule,
    VolumetricEnergyNetwork,
)
from .physics import (
    FloryDecouplingProjection,
    PK2ToCauchyLayer,
    PhysicsIntegrationLayer,
)


DEFAULT_DELTA_J_REF = 1.0e-4


class ModularConstitutivePINN(nn.Module):
    """模块只初始化一次；切换阶段不创建新模型、不重置已有参数。"""

    def __init__(
        self,
        directions: np.ndarray,
        weights: np.ndarray,
        mean_value: float,
        std_value: float,
        architecture: ModelArchitecture | str = ModelArchitecture.FULL,
        delta_j_ref: float = DEFAULT_DELTA_J_REF,
    ) -> None:
        super().__init__()
        self.architecture = ModelArchitecture.parse(architecture)
        self.active_architecture = self.architecture
        self.delta_J_ref = float(delta_j_ref)
        if self.delta_J_ref <= 0.0:
            raise ValueError(
                f"delta_J_ref 必须为正数，当前为 {self.delta_J_ref:.8g}。"
            )

        self.register_buffer("x_mean", torch.tensor(mean_value).float())
        self.register_buffer("x_std", torch.tensor(std_value).float())

        normalized_weights = torch.as_tensor(weights, dtype=torch.float32).reshape(-1)
        if normalized_weights.numel() != int(np.asarray(directions).shape[0]):
            raise ValueError(
                "Dir 与 Wt 的方向数不一致："
                f"Dir={np.asarray(directions).shape[0]}, Wt={normalized_weights.numel()}。"
            )
        if not torch.isclose(
            normalized_weights.sum(), torch.tensor(1.0), atol=1.0e-5
        ):
            raise ValueError(
                "Lebedev 权重之和应为 1，当前为 "
                f"{normalized_weights.sum().item():.8f}。"
                "请检查 MATLAB 端是否已除以 4*pi。"
            )
        self.register_buffer("norm_weights", normalized_weights.view(1, -1))

        # 三种训练情况的模型结构和 state_dict 键完全相同。
        self.free_energy_module = FreeEnergyModule()
        self.damage_module = DamageModule()
        self.volumetric_module = VolumetricEnergyNetwork()
        self.physics_layer = PhysicsIntegrationLayer(directions)
        self.projection_layer = FloryDecouplingProjection()
        self.cauchy_layer = PK2ToCauchyLayer()
        self._initialize_parameters()
        self.activate_modules(self.architecture)

    @property
    def use_damage(self) -> bool:
        return self.active_architecture.use_damage

    @property
    def use_volumetric(self) -> bool:
        return self.active_architecture.use_volumetric

    def _initialize_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0.0)

        # 保留原初始化：损伤初始 d=0.5；体积初始 U_vol=0、p=0。
        # 未开启时不执行这些模块，故自由能阶段不受 d=0.5 影响。
        for final_linear in (self.damage_module.predictor[4], self.volumetric_module[4]):
            nn.init.zeros_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)

    def stage_module(self, stage: ModelArchitecture | str) -> nn.Module:
        return getattr(self, ModelArchitecture.parse(stage).module_name + "_module")

    def activate_modules(self, architecture: ModelArchitecture | str) -> None:
        """切换前向开关（不改变权重）；未开启模块也不参与优化。"""
        active = ModelArchitecture.parse(architecture)
        if active not in self.architecture.training_stages:
            raise ValueError(f"{self.architecture.value} 模型不能开启 {active.value}。")
        self.active_architecture = active
        for stage in ModelArchitecture:
            module = self.stage_module(stage)
            module.requires_grad_(stage in active.training_stages)
            for parameter in module.parameters():
                parameter.grad = None

    def set_training_stage(self, stage: ModelArchitecture | str) -> None:
        """开启本阶段所需模块，仅训练新加入模块，冻结此前的最佳权重。"""
        selected = ModelArchitecture.parse(stage)
        self.activate_modules(selected)
        self.requires_grad_(False)
        self.stage_module(selected).requires_grad_(True)

    def stage_parameters(self, stage: ModelArchitecture | str) -> Iterable[nn.Parameter]:
        yield from self.stage_module(stage).parameters()

    def parameter_summary(self) -> dict[str, int]:
        """返回各分支以及模型总参数量，便于记录实验配置。"""
        summary = {
            stage.module_name: sum(p.numel() for p in self.stage_parameters(stage))
            for stage in ModelArchitecture
        }
        summary["total"] = sum(parameter.numel() for parameter in self.parameters())
        summary["trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return summary

    def forward(
        self,
        lambda_current: torch.Tensor,
        lambda_maximum: torch.Tensor | None,
        j_scalar: torch.Tensor,
        c_vector: torch.Tensor,
        f_vector: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """预测总 Cauchy 应力，并可返回等体积/体积分量与压力。

        即使调用方处于 ``torch.no_grad()`` 中，模型内部仍必须为自由能关于
        拉伸量的导数建立自动微分图。因此这里局部开启梯度计算。
        """
        with torch.enable_grad():
            if lambda_current.ndim != 2:
                raise ValueError(
                    "lambda_current 应为 [batch, directions]，"
                    f"当前为 {tuple(lambda_current.shape)}。"
                )
            if not lambda_current.requires_grad:
                lambda_current = (
                    lambda_current.detach().clone().requires_grad_(True)
                )

            current_normalized = (
                lambda_current - self.x_mean
            ) / (self.x_std + 1.0e-8)
            phi = self.free_energy_module(current_normalized, self.norm_weights)

            if self.use_damage:
                if lambda_maximum is None:
                    raise ValueError(
                        f"{self.architecture.value} 架构需要 lambda_maximum 输入。"
                    )
                maximum_normalized = (
                    lambda_maximum - self.x_mean
                ) / (self.x_std + 1.0e-8)
                damage = self.damage_module(maximum_normalized, self.norm_weights)
                direction_energy = (1.0 - damage) * phi
            else:
                # 自由能阶段直接使用 phi，完全跳过损伤计算及乘法。
                direction_energy = phi
            macro_energy = torch.sum(
                direction_energy * self.norm_weights, dim=1, keepdim=True
            )
            weighted_chain_force = torch.autograd.grad(
                outputs=macro_energy,
                inputs=lambda_current,
                grad_outputs=torch.ones_like(macro_energy),
                create_graph=self.training,
            )[0]
            s_bar = self.physics_layer(weighted_chain_force, lambda_current)

            if self.use_volumetric:
                j_energy = j_scalar.reshape(-1, 1)
                if not j_energy.requires_grad:
                    j_energy = j_energy.detach().clone().requires_grad_(True)
                j_hat = ((j_energy - 1.0) / self.delta_J_ref).to(
                    dtype=lambda_current.dtype
                )
                volumetric_energy = self.volumetric_module(j_hat)
                pressure = torch.autograd.grad(
                    outputs=volumetric_energy,
                    inputs=j_energy,
                    grad_outputs=torch.ones_like(volumetric_energy),
                    create_graph=self.training,
                )[0]
                j_projection = j_energy.to(dtype=c_vector.dtype)
                pressure = pressure.to(dtype=c_vector.dtype)
            else:
                j_projection = j_scalar.reshape(-1, 1).to(dtype=c_vector.dtype)
                # 架构一/二：p=0，从而 S_vol 在解析投影层中严格为零。
                pressure = torch.zeros_like(j_projection)

            s_total, s_iso, s_vol = self.projection_layer(
                s_bar, j_projection, c_vector, pressure if self.use_volumetric else None
            )
            sigma_total = self.cauchy_layer(s_total, f_vector, j_projection)
            sigma_iso = self.cauchy_layer(s_iso, f_vector, j_projection)
            sigma_vol = self.cauchy_layer(s_vol, f_vector, j_projection)

        if return_components:
            return sigma_total, sigma_iso, sigma_vol, pressure
        return sigma_total


# 向后兼容旧脚本中的类名。
FullyDecoupledPINN = ModularConstitutivePINN
