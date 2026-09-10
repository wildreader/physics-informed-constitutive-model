"""带架构元数据的检查点保存与旧检查点兼容加载。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .config import ModelArchitecture


CHECKPOINT_FORMAT_VERSION = 2


@dataclass(frozen=True)
class LoadedCheckpoint:
    state_dict: dict[str, torch.Tensor]
    architecture: ModelArchitecture
    mean_value: float | None
    std_value: float | None
    delta_j_ref: float | None
    legacy: bool
    format_version: int = 0


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    architecture: ModelArchitecture,
    mean_value: float,
    std_value: float,
    delta_j_ref: float,
    sigma_scale: list[float],
    training: dict | None = None,
) -> None:
    if model.architecture is not architecture or model.active_architecture is not architecture:
        raise ValueError("只保存最终模块组合已开启的模型，检查点开关必须与实际模型一致。")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "architecture": architecture.value,
        "enabled_modules": {
            "free_energy": True,
            "damage": architecture.use_damage,
            "volumetric": architecture.use_volumetric,
        },
        "model_state_dict": model.state_dict(),
        "normalization": {
            "mean_value": float(mean_value),
            "std_value": float(std_value),
            "delta_j_ref": float(delta_j_ref),
            "sigma_scale": [float(value) for value in sigma_scale],
        },
        "training": training or {},
    }
    # 覆盖唯一的最佳模型；中断写入不会破坏之前的有效检查点。
    temporary = target.with_name(target.name + ".tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(
    path: str | Path,
    map_location: torch.device | str,
    expected_architecture: ModelArchitecture | str | None = None,
) -> LoadedCheckpoint:
    source = Path(path)
    raw: Any = torch.load(source, map_location=map_location, weights_only=True)
    expected = (
        ModelArchitecture.parse(expected_architecture)
        if expected_architecture is not None
        else None
    )

    if isinstance(raw, dict) and "model_state_dict" in raw:
        version = int(raw.get("format_version", 1))
        if version not in {1, CHECKPOINT_FORMAT_VERSION}:
            raise ValueError(f"不支持的检查点版本：{version}。")
        architecture = ModelArchitecture.parse(raw.get("architecture", "full"))
        if version == CHECKPOINT_FORMAT_VERSION:
            expected_flags = {"free_energy": True, "damage": architecture.use_damage,
                              "volumetric": architecture.use_volumetric}
            if raw.get("enabled_modules") != expected_flags:
                raise ValueError("检查点 enabled_modules 与 architecture 不一致。")
        if expected is not None and architecture is not expected:
            raise ValueError(
                f"检查点架构为 {architecture.value}，但命令行指定为 {expected.value}。"
            )
        normalization = raw.get("normalization", {})
        if not isinstance(normalization, dict):
            normalization = {}
        return LoadedCheckpoint(
            state_dict=raw["model_state_dict"],
            architecture=architecture,
            mean_value=_optional_float(normalization.get("mean_value")),
            std_value=_optional_float(normalization.get("std_value")),
            delta_j_ref=_optional_float(normalization.get("delta_j_ref")),
            legacy=False,
            format_version=version,
        )

    if not isinstance(raw, dict) or not all(
        isinstance(key, str) for key in raw.keys()
    ):
        raise ValueError(f"无法识别检查点格式：{source}。")

    # 旧版 decoupled_model.pth 只保存完整架构的 state_dict。
    if expected is not None and expected is not ModelArchitecture.FULL:
        raise ValueError("旧版纯 state_dict 检查点只能按 full 架构载入。")
    return LoadedCheckpoint(
        state_dict=raw,
        architecture=ModelArchitecture.FULL,
        mean_value=None,
        std_value=None,
        delta_j_ref=None,
        legacy=True,
    )


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def restore_checkpoint(model: torch.nn.Module, checkpoint: LoadedCheckpoint) -> None:
    """新版严格加载；旧版只允许补齐当时不存在的关闭模块。"""
    if model.architecture is not checkpoint.architecture:
        raise ValueError("模型与检查点的模块组合不匹配。")
    if checkpoint.format_version == CHECKPOINT_FORMAT_VERSION:
        model.load_state_dict(checkpoint.state_dict, strict=True)
        return
    prefix_map = {
        "elastic_extractor.": "free_energy_module.extractor.",
        "elastic_predictor.": "free_energy_module.predictor.",
        "damage_extractor.": "damage_module.extractor.",
        "damage_predictor.": "damage_module.predictor.",
        "volumetric_net.": "volumetric_module.",
    }
    migrated = {}
    for key, value in checkpoint.state_dict.items():
        new_key = key
        for old, new in prefix_map.items():
            if key.startswith(old):
                new_key = new + key[len(old):]
                break
        migrated[new_key] = value
    for key, value in model.state_dict().items():
        disabled = (
            key.startswith("damage_module.") and not checkpoint.architecture.use_damage
        ) or (
            key.startswith("volumetric_module.") and not checkpoint.architecture.use_volumetric
        )
        if disabled and key not in migrated:
            migrated[key] = value
    model.load_state_dict(migrated, strict=True)
