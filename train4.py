"""在一个三模块本构 PINN 内进行顺序训练。

示例：
    python train4.py --architecture free_energy
    python train4.py --architecture free_energy_damage
    python train4.py --architecture full
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import scipy.io

from constitutive_pinn import (
    ARCHITECTURE_CHOICES,
    FullyDecoupledPINN,
    ModelArchitecture,
    ModularConstitutivePINN,
)
from constitutive_pinn.checkpoint import save_checkpoint
from constitutive_pinn.data import (
    assert_compatible_stages, default_training_data_path, load_training_data,
)
from constitutive_pinn.inference import predict_components
from constitutive_pinn.plotting import plot_loss_history, plot_training_cases
from constitutive_pinn.training import (
    TrainingOptions,
    build_training_loaders,
    fit_model,
    select_device,
    set_reproducible_seed,
)


# 保留旧版 ``from train4 import FullyDecoupledPINN, device`` 的导入习惯。
device = select_device("auto")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="训练自由能/损伤/体积分支可配置的本构 PINN。"
    )
    parser.add_argument(
        "--architecture",
        choices=ARCHITECTURE_CHOICES,
        default=ModelArchitecture.FULL.value,
        help="模型架构；默认 full。",
    )
    parser.add_argument(
        "--train-data",
        default=None,
        help=(
            "最终阶段的七工况训练 MAT 文件；默认根据 --architecture 从 "
            "matlab_data_generation/generated/ 自动选择。"
        ),
    )
    for name in ("free-energy", "damage", "volumetric"):
        parser.add_argument(f"--{name}-data", default=None,
                            help=f"单独指定 {name} 阶段的训练 MAT 文件。")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录；默认 outputs/<architecture>。",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="检查点输出路径；默认 <output-dir>/model.pth。",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--prediction-batch-size", type=int, default=1024)
    parser.add_argument("--free-energy-epochs", type=int, default=500)
    parser.add_argument("--damage-epochs", type=int, default=500)
    parser.add_argument("--volumetric-epochs", type=int, default=500)
    parser.add_argument("--free-energy-lr", type=float, default=None,
                        help="默认：单自由能 0.001，多模块第一阶段 0.002。")
    parser.add_argument("--damage-lr", type=float, default=0.001)
    parser.add_argument("--volumetric-lr", type=float, default=0.001)
    parser.add_argument("--selection-loss", choices=("validation", "train"),
                        default="validation", help="选择最优 epoch 的 loss，默认 validation。")
    parser.add_argument("--scheduler-step", type=int, default=200)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--progress-interval", type=int, default=50)
    parser.add_argument(
        "--show",
        action="store_true",
        help="生成图片后同时显示 Matplotlib 窗口。",
    )
    return parser


def main(arguments: list[str] | None = None) -> None:
    args = build_parser().parse_args(arguments)
    architecture = ModelArchitecture.parse(args.architecture)
    output_directory = Path(args.output_dir or Path("outputs") / architecture.value)
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path(args.checkpoint or output_directory / "model.pth")

    options = TrainingOptions(
        batch_size=args.batch_size,
        seed=args.seed,
        free_energy_epochs=args.free_energy_epochs,
        damage_epochs=args.damage_epochs,
        volumetric_epochs=args.volumetric_epochs,
        free_energy_learning_rate=args.free_energy_lr,
        damage_learning_rate=args.damage_lr,
        volumetric_learning_rate=args.volumetric_lr,
        scheduler_step_size=args.scheduler_step,
        scheduler_gamma=args.scheduler_gamma,
        progress_interval=args.progress_interval,
        selection_loss=args.selection_loss,
    )
    options.validate(architecture)
    set_reproducible_seed(options.seed)
    selected_device = select_device(args.device)

    print(f"当前使用的计算设备: {selected_device}")
    print(f"模型架构: {architecture.value} — {architecture.description}")
    stage_data = {}
    stage_loaders = {}
    data_paths = {}
    for stage in architecture.training_stages:
        override = getattr(args, stage.module_name + "_data")
        if stage is architecture and args.train_data is not None:
            if override is not None:
                raise ValueError("--train-data 与最终阶段专用数据参数不能同时指定。")
            override = args.train_data
        path = Path(override) if override else default_training_data_path(stage)
        print(f"正在加载 {stage.module_name} 阶段数据: {path}")
        data_paths[stage.value] = str(path.resolve())
        stage_data[stage] = load_training_data(path, expected_architecture=stage)
        reference = stage_data[ModelArchitecture.FREE_ENERGY]
        assert_compatible_stages(reference, stage_data[stage])
        stage_loaders[stage] = build_training_loaders(stage_data[stage], options)
        print(f"  sigma_total 逐分量 RMS 尺度: {stage_loaders[stage].sigma_scale}")
    training_data = stage_data[architecture]
    # 归一化在首阶段确定，后续继承时不更改，否则自由能函数会改变。
    mean_value, std_value = reference.normalization

    model = ModularConstitutivePINN(
        directions=reference.directions,
        weights=reference.weights,
        mean_value=mean_value,
        std_value=std_value,
        architecture=architecture,
        delta_j_ref=training_data.delta_j_ref,
    ).to(selected_device)
    print(f"模型参数量: {model.parameter_summary()}")

    def save_final_best(training_info: dict) -> None:
        save_checkpoint(
            checkpoint_path, model=model, architecture=architecture,
            mean_value=mean_value, std_value=std_value,
            delta_j_ref=reference.delta_j_ref,
            sigma_scale=stage_loaders[architecture].sigma_scale.tolist(),
            training={**training_info, "options": asdict(options), "data_paths": data_paths},
        )

    history = fit_model(model, stage_loaders, options, selected_device,
                        on_final_best=save_final_best)
    (output_directory / "training_history.json").write_text(
        json.dumps({"architecture": architecture.value, "options": asdict(options),
                    "data_paths": data_paths, **asdict(history)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"训练完成，唯一最优检查点已保存至: {checkpoint_path}")

    plot_loss_history(
        history,
        output_directory / "train_loss_curve.png",
        show=args.show,
    )
    training_prediction = predict_components(
        model=model,
        lambda_current=training_data.lambda_current,
        lambda_maximum=training_data.lambda_maximum,
        j_ratio=training_data.j_ratio,
        c_vector=training_data.c_vector,
        f_vector=training_data.f_vector,
        device=selected_device,
        batch_size=args.prediction_batch_size,
    )
    scipy.io.savemat(
        output_directory / "train_components.mat",
        {
            "architecture": architecture.value,
            "sigma_total_pred": training_prediction.sigma_total,
            "sigma_iso_pred": training_prediction.sigma_iso,
            "sigma_vol_pred": training_prediction.sigma_vol,
            "p_pred": training_prediction.pressure,
            "sigma_total_true": training_data.sigma_total,
        },
    )
    plot_training_cases(
        training_data,
        training_prediction.sigma_total,
        output_directory,
        show=args.show,
    )
    print(f"训练结果图与应力分量已生成至: {output_directory}")


if __name__ == "__main__":
    main()
