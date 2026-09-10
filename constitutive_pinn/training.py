"""同一个模型按自由能、损伤、体积顺序逐模块训练。"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import math

import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .data import TrainingData, component_rms_scale
from .config import ModelArchitecture
from .model import ModularConstitutivePINN


@dataclass(frozen=True)
class TrainingOptions:
    batch_size: int = 256
    validation_fraction: float = 0.2
    seed: int = 42
    free_energy_epochs: int = 500
    damage_epochs: int = 500
    volumetric_epochs: int = 500
    # None 保留原学习率对应关系：单自由能 0.001，多模块首阶段 0.002。
    free_energy_learning_rate: float | None = None
    damage_learning_rate: float = 0.001
    volumetric_learning_rate: float = 0.001
    scheduler_step_size: int = 200
    scheduler_gamma: float = 0.5
    progress_interval: int = 50
    selection_loss: str = "validation"

    def validate(self, architecture: ModelArchitecture | str | None = None) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size 必须为正整数。")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction 必须位于 (0, 1) 内。")
        counts = (self.free_energy_epochs, self.damage_epochs, self.volumetric_epochs)
        if any(value < 0 for value in counts):
            raise ValueError("训练 epoch 数不能为负。")
        if architecture is not None:
            for stage in ModelArchitecture.parse(architecture).training_stages:
                if self.epochs_for(stage) <= 0:
                    raise ValueError(f"已启用的 {stage.module_name} 阶段必须训练至少 1 epoch。")
        rates = (self.free_energy_learning_rate, self.damage_learning_rate,
                 self.volumetric_learning_rate)
        if any(value is not None and (not math.isfinite(value) or value <= 0) for value in rates):
            raise ValueError("学习率必须为正数。")
        if self.scheduler_step_size <= 0 or not 0.0 < self.scheduler_gamma <= 1.0:
            raise ValueError("学习率调度参数无效。")
        if self.selection_loss not in {"train", "validation"}:
            raise ValueError("selection_loss 必须是 train 或 validation。")

    def epochs_for(self, stage: ModelArchitecture) -> int:
        return getattr(self, stage.module_name + "_epochs")

    def learning_rate_for(self, stage: ModelArchitecture, target: ModelArchitecture) -> float:
        if stage is ModelArchitecture.FREE_ENERGY and self.free_energy_learning_rate is None:
            return 0.001 if target is ModelArchitecture.FREE_ENERGY else 0.002
        return float(getattr(self, stage.module_name + "_learning_rate"))


@dataclass(frozen=True)
class TrainingLoaders:
    train: DataLoader
    validation: DataLoader
    sigma_scale: np.ndarray
    architecture: ModelArchitecture


@dataclass
class TrainingHistory:
    train_loss: list[float] = field(default_factory=list)
    validation_loss: list[float] = field(default_factory=list)
    learning_rate: list[float] = field(default_factory=list)
    stage_names: list[str] = field(default_factory=list)
    stages: list[dict] = field(default_factory=list)


def set_reproducible_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def select_device(requested: str = "auto") -> torch.device:
    normalized = requested.strip().lower()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("命令行要求 CUDA，但当前 PyTorch 未检测到可用 GPU。")
    if normalized not in {"cpu", "cuda"}:
        raise ValueError("device 只能是 auto、cpu 或 cuda。")
    return torch.device(normalized)


def build_training_loaders(
    data: TrainingData, options: TrainingOptions
) -> TrainingLoaders:
    options.validate()
    split = train_test_split(
        data.lambda_current,
        data.lambda_maximum,
        data.j_ratio,
        data.c_vector,
        data.f_vector,
        data.sigma_total,
        test_size=options.validation_fraction,
        random_state=options.seed,
    )
    (
        current_train,
        current_validation,
        maximum_train,
        maximum_validation,
        j_train,
        j_validation,
        c_train,
        c_validation,
        f_train,
        f_validation,
        sigma_train,
        sigma_validation,
    ) = split

    sigma_scale = component_rms_scale(sigma_train)
    train_dataset = TensorDataset(
        torch.from_numpy(current_train),
        torch.from_numpy(maximum_train),
        torch.from_numpy(j_train),
        torch.from_numpy(c_train),
        torch.from_numpy(f_train),
        torch.from_numpy(sigma_train),
    )
    validation_dataset = TensorDataset(
        torch.from_numpy(current_validation),
        torch.from_numpy(maximum_validation),
        torch.from_numpy(j_validation),
        torch.from_numpy(c_validation),
        torch.from_numpy(f_validation),
        torch.from_numpy(sigma_validation),
    )

    generator = torch.Generator()
    generator.manual_seed(options.seed)
    return TrainingLoaders(
        train=DataLoader(
            train_dataset,
            batch_size=options.batch_size,
            shuffle=True,
            generator=generator,
        ),
        validation=DataLoader(
            validation_dataset,
            batch_size=options.batch_size,
            shuffle=False,
        ),
        sigma_scale=sigma_scale,
        architecture=ModelArchitecture.parse(data.model_architecture or "full"),
    )


def normalized_stress_loss(
    predicted: torch.Tensor, expected: torch.Tensor, scale: torch.Tensor
) -> torch.Tensor:
    """总 Cauchy 应力的逐分量 RMS 归一化均方误差。"""
    return torch.mean(torch.square((predicted - expected) / scale))


def fit_model(
    model: ModularConstitutivePINN,
    stage_loaders: Mapping[ModelArchitecture, TrainingLoaders],
    options: TrainingOptions,
    device: torch.device,
    on_final_best: Callable[[dict], None] | None = None,
) -> TrainingHistory:
    """各阶段恢复自身最优权重；仅最后阶段改善时可覆盖唯一检查点。"""
    options.validate(model.architecture)
    stages = model.architecture.training_stages
    for stage in stages:
        if stage not in stage_loaders:
            raise ValueError(f"缺少 {stage.value} 阶段数据；不能用最终阶段标签代替。")
        if stage_loaders[stage].architecture is not stage:
            raise ValueError(f"{stage.value} 阶段的数据标签不匹配。")
    history = TrainingHistory()

    for stage in stages:
        model.set_training_stage(stage)
        loaders = stage_loaders[stage]
        scale = torch.from_numpy(loaders.sigma_scale).view(1, 6).to(device)
        optimizer = optim.Adam(
            model.stage_parameters(stage),
            lr=options.learning_rate_for(stage, model.architecture),
        )
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=options.scheduler_step_size,
            gamma=options.scheduler_gamma,
        )
        epochs = options.epochs_for(stage)
        start_epoch = len(history.train_loss)
        best_loss = float("inf")
        best_state = None
        best_info = None
        print(f"\n>>> [{stage.module_name}] 开启 {stage.value}；"
              f"仅训练 {stage.module_name} 模块，共 {epochs} epochs。")
        for epoch in range(1, epochs + 1):
            learning_rate = float(optimizer.param_groups[0]["lr"])
            train_loss = _run_epoch(model, loaders.train, scale, device, optimizer)
            validation_loss = _run_epoch(model, loaders.validation, scale, device, None)
            # train 模式也必须比较本 epoch 结束时的同一份权重，而非混合批次 loss。
            if options.selection_loss == "train":
                selection_loader = DataLoader(
                    loaders.train.dataset, batch_size=loaders.train.batch_size,
                    shuffle=False,
                )
                train_loss = _run_epoch(model, selection_loader, scale, device, None)
            if not math.isfinite(train_loss) or not math.isfinite(validation_loss):
                raise RuntimeError(f"{stage.module_name} epoch {epoch} 出现非有限 loss。")
            history.train_loss.append(train_loss)
            history.validation_loss.append(validation_loss)
            history.learning_rate.append(learning_rate)
            history.stage_names.append(stage.module_name)
            selected_loss = validation_loss if options.selection_loss == "validation" else train_loss
            if selected_loss < best_loss:
                best_loss = selected_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                best_info = {
                    "stage": stage.module_name,
                    "architecture": stage.value,
                    "start_epoch": start_epoch + 1,
                    "epochs": epochs,
                    "best_epoch": epoch,
                    "best_global_epoch": start_epoch + epoch,
                    "selection_loss": options.selection_loss,
                    "best_loss": selected_loss,
                    "train_loss": train_loss,
                    "validation_loss": validation_loss,
                    "learning_rate": learning_rate,
                }
                if stage is stages[-1] and on_final_best is not None:
                    on_final_best({
                        "strategy": "sequential_modules",
                        "selection_loss": options.selection_loss,
                        "stages": [*history.stages, best_info],
                    })
            scheduler.step()
            if epoch == 1 or epoch % max(options.progress_interval, 1) == 0 or epoch == epochs:
                print(f"{stage.module_name} Epoch [{epoch}/{epochs}], lr={learning_rate:.6g}, "
                      f"Train Loss: {train_loss:.6f}, Val Loss: {validation_loss:.6f}, "
                      f"Best {options.selection_loss}: {best_loss:.6f}")

        if best_state is None or best_info is None:
            raise RuntimeError(f"{stage.module_name} 阶段未产生有效最佳权重。")
        model.load_state_dict(best_state, strict=True)
        history.stages.append(best_info)
        print(f"已恢复 {stage.module_name} 最优 epoch {best_info['best_epoch']}；"
              f"loss={best_loss:.6g}，供后续阶段继承。")

    model.eval()
    return history


def _run_epoch(
    model: ModularConstitutivePINN,
    loader: DataLoader,
    scale: torch.Tensor,
    device: torch.device,
    optimizer: optim.Optimizer | None,
) -> float:
    training = optimizer is not None
    model.train(mode=training)
    total_loss = 0.0
    sample_count = 0

    for batch in loader:
        current, maximum, j_ratio, c_vector, f_vector, sigma_total = (
            tensor.to(device) for tensor in batch
        )
        if optimizer is not None:
            optimizer.zero_grad()

        maximum_input = maximum if model.use_damage else None
        prediction = model(
            current, maximum_input, j_ratio, c_vector, f_vector
        )
        loss = normalized_stress_loss(prediction, sigma_total, scale)
        if optimizer is not None:
            loss.backward()
            optimizer.step()

        batch_size = current.shape[0]
        total_loss += float(loss.detach().item()) * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise RuntimeError("DataLoader 为空，无法计算损失。")
    return total_loss / sample_count
