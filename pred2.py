"""使用指定架构在五工况数据上预测 Cauchy 应力。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import scipy.io

from constitutive_pinn import ARCHITECTURE_CHOICES, ModelArchitecture
from constitutive_pinn.checkpoint import load_checkpoint, restore_checkpoint
from constitutive_pinn.data import (
    assert_compatible_datasets,
    default_prediction_data_path,
    default_training_data_path,
    load_prediction_data,
    load_training_data,
)
from constitutive_pinn.inference import predict_components
from constitutive_pinn.model import ModularConstitutivePINN
from constitutive_pinn.plotting import plot_prediction_cases, prediction_metrics
from constitutive_pinn.training import select_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在五工况数据上评估指定架构的本构 PINN。"
    )
    parser.add_argument(
        "--architecture",
        choices=ARCHITECTURE_CHOICES,
        default=ModelArchitecture.FULL.value,
        help="必须与训练检查点的架构一致；默认 full。",
    )
    parser.add_argument(
        "--train-data",
        default=None,
        help=(
            "训练 MAT 文件；默认根据 --architecture 从 "
            "matlab_data_generation/generated/ 自动选择。"
        ),
    )
    parser.add_argument(
        "--test-data",
        default=None,
        help=(
            "五工况预测 MAT 文件；默认根据 --architecture 从 "
            "matlab_data_generation/generated/ 自动选择。"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="检查点路径；默认 outputs/<architecture>/model.pth。",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录；默认 outputs/<architecture>/prediction。",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument(
        "--show",
        action="store_true",
        help="生成图片后同时显示 Matplotlib 窗口。",
    )
    return parser


def main(arguments: list[str] | None = None) -> None:
    args = build_parser().parse_args(arguments)
    architecture = ModelArchitecture.parse(args.architecture)
    training_data_path = (
        Path(args.train_data)
        if args.train_data is not None
        else default_training_data_path(architecture)
    )
    prediction_data_path = (
        Path(args.test_data)
        if args.test_data is not None
        else default_prediction_data_path(architecture)
    )
    selected_device = select_device(args.device)
    output_directory = Path(
        args.output_dir or Path("outputs") / architecture.value / "prediction"
    )
    output_directory.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(
        args.checkpoint or Path("outputs") / architecture.value / "model.pth"
    )
    legacy_path = Path("decoupled_model.pth")
    if (
        args.checkpoint is None
        and architecture is ModelArchitecture.FULL
        and not checkpoint_path.exists()
        and legacy_path.exists()
    ):
        checkpoint_path = legacy_path

    print(f"当前使用的计算设备: {selected_device}")
    print(f"模型架构: {architecture.value} — {architecture.description}")
    print(f"载入检查点: {checkpoint_path}")
    checkpoint = load_checkpoint(
        checkpoint_path,
        map_location=selected_device,
        expected_architecture=architecture,
    )

    print(f"训练数据: {training_data_path}")
    print(f"预测数据: {prediction_data_path}")
    training_data = load_training_data(
        training_data_path, expected_architecture=architecture
    )
    prediction_data = load_prediction_data(
        prediction_data_path, expected_architecture=architecture
    )
    assert_compatible_datasets(training_data, prediction_data)
    data_mean, data_std = training_data.normalization
    mean_value = (
        checkpoint.mean_value if checkpoint.mean_value is not None else data_mean
    )
    std_value = checkpoint.std_value if checkpoint.std_value is not None else data_std
    delta_j_ref = (
        checkpoint.delta_j_ref
        if checkpoint.delta_j_ref is not None
        else training_data.delta_j_ref
    )
    if not np.isclose(
        delta_j_ref, training_data.delta_j_ref, rtol=0.0, atol=1.0e-14
    ):
        raise ValueError(
            "检查点和训练数据的 delta_J_ref 不一致："
            f"{delta_j_ref:.8g} vs {training_data.delta_j_ref:.8g}。"
        )

    model = ModularConstitutivePINN(
        directions=training_data.directions,
        weights=training_data.weights,
        mean_value=mean_value,
        std_value=std_value,
        architecture=architecture,
        delta_j_ref=delta_j_ref,
    ).to(selected_device)
    restore_checkpoint(model, checkpoint)

    prediction = predict_components(
        model=model,
        lambda_current=prediction_data.lambda_current,
        lambda_maximum=prediction_data.lambda_maximum,
        j_ratio=prediction_data.j_ratio,
        c_vector=prediction_data.c_vector,
        f_vector=prediction_data.f_vector,
        device=selected_device,
        batch_size=args.batch_size,
    )
    if not architecture.use_volumetric:
        max_volume = float(np.max(np.abs(prediction.sigma_vol)))
        max_pressure = float(np.max(np.abs(prediction.pressure)))
        if max_volume != 0.0 or max_pressure != 0.0:
            raise RuntimeError(
                "体积分支已关闭，但 sigma_vol 或 p 没有严格归零："
                f"max|sigma_vol|={max_volume:.3e}, max|p|={max_pressure:.3e}。"
            )

    mse, rmse_components = prediction_metrics(
        prediction.sigma_total, prediction_data.sigma_total
    )
    print(f"测试集 Cauchy 应力 MSE: {mse:.6e}")
    print(f"各 Cauchy 应力分量 RMSE: {rmse_components}")

    scipy.io.savemat(
        output_directory / "prediction_components.mat",
        {
            "architecture": architecture.value,
            "sigma_total_pred": prediction.sigma_total,
            "sigma_iso_pred": prediction.sigma_iso,
            "sigma_vol_pred": prediction.sigma_vol,
            "p_pred": prediction.pressure,
            "sigma_total_true": prediction_data.sigma_total,
            "mse": mse,
            "rmse_components": rmse_components,
        },
    )
    plot_prediction_cases(
        prediction_data,
        prediction.sigma_total,
        output_directory,
        show=args.show,
    )
    print(f"五工况预测图与应力分量已生成至: {output_directory}")


if __name__ == "__main__":
    main()
