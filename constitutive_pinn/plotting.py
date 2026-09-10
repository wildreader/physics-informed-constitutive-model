"""训练和五工况预测结果的统一绘图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data import PredictionData, TrainingData
from .training import TrainingHistory


COMPONENT_NAMES = [
    "sigma11",
    "sigma22",
    "sigma33",
    "sigma12",
    "sigma13",
    "sigma23",
]


def plot_loss_history(
    history: TrainingHistory, output_path: str | Path, show: bool = False
) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 5))
    epochs = np.arange(1, len(history.train_loss) + 1)
    axis.plot(epochs, history.train_loss, "b-", lw=2, label="Train Loss")
    axis.plot(epochs, history.validation_loss, "r--", lw=2, label="Validation Loss")
    for stage in history.stages:
        if stage["start_epoch"] > 1:
            axis.axvline(stage["start_epoch"] - 0.5, color="0.45", ls=":", lw=1.2)
        axis.scatter(stage["best_global_epoch"], stage["best_loss"], marker="*",
                     s=90, zorder=3, label=f"Best {stage['stage']}")
    axis.set_yscale("log")
    axis.set_title("Training and Validation Loss", fontweight="bold")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Normalized MSE Loss")
    axis.legend()
    axis.grid(ls="--", alpha=0.5)
    figure.tight_layout()
    figure.savefig(target, dpi=300)
    _finish_figure(figure, show)


def plot_training_cases(
    data: TrainingData,
    sigma_prediction: np.ndarray,
    output_directory: str | Path,
    show: bool = False,
) -> None:
    if sigma_prediction.shape != data.sigma_total.shape:
        raise ValueError(
            "训练预测值与真实值形状不一致："
            f"{sigma_prediction.shape} vs {data.sigma_total.shape}。"
        )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    load_names = [
        "Uniaxial X",
        "Pure Shear X",
        "Equibiaxial XY",
        "Uniaxial Y",
        "Proportional Biaxial",
        "Rotated Uniaxial",
        "Z Pre-stretch then X",
    ]

    start = 0
    for index, case_length in enumerate(data.case_lengths):
        end = start + int(case_length)
        true_case = data.sigma_total[start:end]
        predicted_case = sigma_prediction[start:end]
        if index < 6:
            x_axis = data.stretch_path
        else:
            x_axis = data.stretch_path_x
            slice_start = data.pre_stretch_length - 1
            true_case = true_case[slice_start:]
            predicted_case = predicted_case[slice_start:]

        _plot_six_components(
            x_axis=x_axis,
            expected=true_case,
            predicted=predicted_case,
            title=f"Train Case {index + 1}: {load_names[index]}",
            output_path=output / f"train_case_{index + 1}.png",
            show=show,
        )
        start = end


def plot_prediction_cases(
    data: PredictionData,
    sigma_prediction: np.ndarray,
    output_directory: str | Path,
    show: bool = False,
) -> None:
    if sigma_prediction.shape != data.sigma_total.shape:
        raise ValueError(
            "预测值与真实值形状不一致："
            f"{sigma_prediction.shape} vs {data.sigma_total.shape}。"
        )
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    load_names = [
        "Test 1: Uniaxial Z",
        "Test 2: Equibiaxial YZ",
        "Test 3: Pure Shear Y",
        "Test 4: Y Pre-stretch then X",
        "Test 5: Proportional Biaxial YZ (1:0.5)",
    ]

    start = 0
    for index, case_length in enumerate(data.case_lengths):
        end = start + int(case_length)
        true_case = data.sigma_total[start:end]
        predicted_case = sigma_prediction[start:end]
        if index == 3:
            x_axis = data.stretch_path_x
            slice_start = data.pre_stretch_length - 1
            true_case = true_case[slice_start:]
            predicted_case = predicted_case[slice_start:]
        else:
            x_axis = data.stretch_path

        _plot_six_components(
            x_axis=x_axis,
            expected=true_case,
            predicted=predicted_case,
            title=load_names[index],
            output_path=output / f"test_case_{index + 1}_pred.png",
            show=show,
        )
        start = end


def prediction_metrics(
    prediction: np.ndarray, expected: np.ndarray
) -> tuple[float, np.ndarray]:
    residual = prediction.astype(np.float64) - expected.astype(np.float64)
    mse = float(np.mean(np.square(residual)))
    rmse_components = np.sqrt(np.mean(np.square(residual), axis=0))
    return mse, rmse_components


def _plot_six_components(
    x_axis: np.ndarray,
    expected: np.ndarray,
    predicted: np.ndarray,
    title: str,
    output_path: Path,
    show: bool,
) -> None:
    if expected.shape[0] != x_axis.size:
        raise ValueError(
            f"绘图横轴长度 {x_axis.size} 与应力行数 {expected.shape[0]} 不一致。"
        )
    figure, axes = plt.subplots(2, 3, figsize=(16, 9))
    for component_index, axis in enumerate(axes.flatten()):
        axis.plot(
            x_axis,
            expected[:, component_index],
            "k-",
            lw=1,
            label="True Cauchy stress",
        )
        axis.plot(
            x_axis,
            predicted[:, component_index],
            "r--",
            lw=1,
            label="Predicted Cauchy stress",
        )
        axis.set_title(COMPONENT_NAMES[component_index], fontweight="bold")
        axis.set_xlabel("Stretch ratio lambda")
        axis.set_ylabel("Cauchy stress")
        axis.grid(True, ls="--", alpha=0.5)
        if component_index == 0:
            axis.legend()
    figure.suptitle(title, fontsize=16, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_path, dpi=300)
    _finish_figure(figure, show)


def _finish_figure(figure: plt.Figure, show: bool) -> None:
    if show:
        figure.show()
    else:
        plt.close(figure)
