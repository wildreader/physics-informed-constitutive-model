"""MATLAB 数据读取、校验和训练/预测张量组织。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import scipy.io

from .config import ModelArchitecture


SCALE_FLOOR_RATIO = 1.0e-4
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GENERATED_DATA_ROOT = PROJECT_ROOT / "matlab_data_generation" / "generated"


def default_training_data_path(
    architecture: ModelArchitecture | str,
) -> Path:
    """返回指定架构的默认七工况数据路径，并兼容旧 full 数据。"""
    selected = ModelArchitecture.parse(architecture)
    organized = GENERATED_DATA_ROOT / selected.value / "train_7cases.mat"
    legacy = PROJECT_ROOT / "train_decoupled_7cases.mat"
    if selected is ModelArchitecture.FULL and not organized.exists() and legacy.exists():
        return legacy
    return organized


def default_prediction_data_path(
    architecture: ModelArchitecture | str,
) -> Path:
    """返回指定架构的默认五工况数据路径，并兼容旧 full 数据。"""
    selected = ModelArchitecture.parse(architecture)
    organized = GENERATED_DATA_ROOT / selected.value / "test_5cases.mat"
    legacy = PROJECT_ROOT / "test_decoupled_5cases.mat"
    if selected is ModelArchitecture.FULL and not organized.exists() and legacy.exists():
        return legacy
    return organized


def component_rms_scale(
    values: np.ndarray, floor_ratio: float = SCALE_FLOOR_RATIO
) -> np.ndarray:
    """计算总 Cauchy 应力各分量的 RMS 尺度。"""
    rms = np.sqrt(np.mean(np.square(values, dtype=np.float64), axis=0))
    reference = max(float(np.max(rms)), 1.0e-8)
    floor = max(floor_ratio * reference, 1.0e-8)
    return np.maximum(rms, floor).astype(np.float32)


def _required(data: dict[str, Any], key: str, path: Path) -> np.ndarray:
    if key not in data:
        raise KeyError(f"MAT 文件 {path} 缺少字段 {key!r}。")
    return np.asarray(data[key])


def _scalar(
    data: dict[str, Any], key: str, path: Path, default: float | None = None
) -> float:
    if key not in data:
        if default is None:
            raise KeyError(f"MAT 文件 {path} 缺少标量字段 {key!r}。")
        return float(default)
    values = np.asarray(data[key]).reshape(-1)
    if values.size != 1:
        raise ValueError(f"字段 {key!r} 应为标量，当前包含 {values.size} 个值。")
    return float(values[0])


def _optional_text(data: dict[str, Any], key: str) -> str | None:
    if key not in data:
        return None
    values = np.asarray(data[key])
    if values.size == 0:
        return None
    if values.dtype.kind in {"U", "S"}:
        return "".join(values.astype(str).reshape(-1)).strip()
    if values.dtype == object:
        return "".join(str(item) for item in values.reshape(-1)).strip()
    return str(values.reshape(-1)[0]).strip()


def _validate_architecture_metadata(
    stored_architecture: str | None,
    expected_architecture: ModelArchitecture | str | None,
    j_ratio: np.ndarray,
    path: Path,
    raw: dict[str, Any],
    pressure_key: str,
) -> None:
    if expected_architecture is None:
        return
    expected = ModelArchitecture.parse(expected_architecture)
    if stored_architecture:
        stored = ModelArchitecture.parse(stored_architecture)
        if stored is not expected:
            raise ValueError(
                f"数据集 {path} 属于 {stored.value}，但当前模型为 {expected.value}。"
            )
    elif expected is not ModelArchitecture.FULL:
        raise ValueError(
            f"数据集 {path} 不含 model_architecture 元数据。"
            "前两种架构必须使用 matlab_data_generation 生成的新数据集。"
        )

    for key, enabled in (("use_damage", expected.use_damage),
                         ("use_volumetric", expected.use_volumetric)):
        if key in raw and _scalar(raw, key, path) != float(enabled):
            raise ValueError(f"数据集 {path} 的 {key} 与 {expected.value} 不匹配。")

    if np.any(j_ratio <= 0.0):
        raise ValueError(f"数据集 {path} 中存在非正体积比 J。")
    if not expected.use_volumetric and not np.allclose(j_ratio, 1.0, rtol=0, atol=1e-10):
        raise ValueError(
            f"{expected.value} 数据必须为等体积路径 J=1；{path} 含体积变形。"
            "请重新运行 matlab_data_generation/generate_all_datasets.m。"
        )
    if not expected.use_volumetric and pressure_key in raw:
        pressure = np.asarray(raw[pressure_key])
        if not np.all(np.isfinite(pressure)) or np.any(pressure != 0):
            raise ValueError(f"数据集 {path} 关闭体积模块，但包含非零或无效压力标签。")


def _validate_rows(reference_name: str, reference: np.ndarray, **arrays: np.ndarray) -> None:
    row_count = reference.shape[0]
    mismatched = {
        name: value.shape[0]
        for name, value in arrays.items()
        if value.shape[0] != row_count
    }
    if mismatched:
        raise ValueError(
            f"样本数不一致：{reference_name}={row_count}，其他字段={mismatched}。"
        )


def _validate_core_shapes(
    lambda_current: np.ndarray,
    lambda_maximum: np.ndarray,
    j_ratio: np.ndarray,
    c_vector: np.ndarray,
    f_vector: np.ndarray,
    sigma_total: np.ndarray,
) -> None:
    if lambda_current.ndim != 2:
        raise ValueError("当前方向伸长 X_La_curr 必须是二维数组。")
    if lambda_maximum.shape != lambda_current.shape:
        raise ValueError(
            "当前方向伸长与历史最大方向伸长形状不一致："
            f"{lambda_current.shape} vs {lambda_maximum.shape}。"
        )
    expected_columns = {
        "J": (j_ratio, 1),
        "C": (c_vector, 6),
        "F": (f_vector, 9),
        "sigma_total": (sigma_total, 6),
    }
    for name, (array, width) in expected_columns.items():
        if array.ndim != 2 or array.shape[1] != width:
            raise ValueError(
                f"{name} 形状应为 [samples, {width}]，当前为 {array.shape}。"
            )
    for array in (lambda_current, lambda_maximum, j_ratio, c_vector, f_vector, sigma_total):
        if not np.all(np.isfinite(array)):
            raise ValueError("数据含 NaN/Inf，无法训练。")
    if np.any(j_ratio <= 0.0):
        raise ValueError("体积比 J 必须全部大于 0。")


@dataclass(frozen=True)
class TrainingData:
    lambda_current: np.ndarray
    lambda_maximum: np.ndarray
    j_ratio: np.ndarray
    c_vector: np.ndarray
    f_vector: np.ndarray
    sigma_total: np.ndarray
    directions: np.ndarray
    weights: np.ndarray
    case_lengths: np.ndarray
    stretch_path: np.ndarray
    stretch_path_x: np.ndarray
    pre_stretch_length: int
    delta_j_ref: float
    model_architecture: str | None

    @property
    def normalization(self) -> tuple[float, float]:
        mean_value = float(np.mean(self.lambda_current))
        std_value = float(np.std(self.lambda_current))
        if std_value == 0.0:
            std_value = 1.0
        return mean_value, std_value


@dataclass(frozen=True)
class PredictionData:
    lambda_current: np.ndarray
    lambda_maximum: np.ndarray
    j_ratio: np.ndarray
    c_vector: np.ndarray
    f_vector: np.ndarray
    sigma_total: np.ndarray
    case_lengths: np.ndarray
    stretch_path: np.ndarray
    stretch_path_x: np.ndarray
    pre_stretch_length: int
    delta_j_ref: float
    model_architecture: str | None


def load_training_data(
    path: str | Path,
    expected_architecture: ModelArchitecture | str | None = None,
) -> TrainingData:
    source = Path(path)
    data = scipy.io.loadmat(source)

    lambda_current = _required(data, "X_La_curr_export", source).astype(np.float32)
    lambda_maximum = _required(data, "X_La_max_export", source).astype(np.float32)
    # J≈1 且变化量仅约 1e-4；保留 float64 后再在模型内部中心化。
    j_ratio = _required(data, "X_J_export", source).astype(np.float64)
    c_vector = _required(data, "X_C_export", source).astype(np.float32)
    f_vector = _required(data, "X_F_export", source).astype(np.float32)
    sigma_total = _required(data, "Y_sigma_total_export", source).astype(np.float32)

    _validate_rows(
        "X_La_curr_export",
        lambda_current,
        X_La_max_export=lambda_maximum,
        X_J_export=j_ratio,
        X_C_export=c_vector,
        X_F_export=f_vector,
        Y_sigma_total_export=sigma_total,
    )
    _validate_core_shapes(
        lambda_current,
        lambda_maximum,
        j_ratio,
        c_vector,
        f_vector,
        sigma_total,
    )
    stored_architecture = _optional_text(data, "model_architecture")
    _validate_architecture_metadata(
        stored_architecture, expected_architecture, j_ratio, source, data, "Y_p_export"
    )

    directions = _required(data, "Dir", source).astype(np.float32)
    weights = _required(data, "Wt", source).reshape(-1).astype(np.float32)
    if directions.shape != (weights.size, 3):
        raise ValueError(
            f"Dir/Wt 形状不一致：Dir={directions.shape}, Wt={weights.shape}。"
        )
    if lambda_current.shape[1] != weights.size:
        raise ValueError("方向伸长列数与 Dir/Wt 的方向数不一致。")

    return TrainingData(
        lambda_current=lambda_current,
        lambda_maximum=lambda_maximum,
        j_ratio=j_ratio,
        c_vector=c_vector,
        f_vector=f_vector,
        sigma_total=sigma_total,
        directions=directions,
        weights=weights,
        case_lengths=_required(data, "Train_Case_Lengths", source)
        .reshape(-1)
        .astype(int),
        stretch_path=_required(data, "landa", source).reshape(-1),
        stretch_path_x=_required(data, "landa_trainX", source).reshape(-1),
        pre_stretch_length=int(_scalar(data, "len_Z", source)),
        delta_j_ref=_scalar(data, "delta_J_ref", source, default=1.0e-4),
        model_architecture=stored_architecture,
    )


def load_prediction_data(
    path: str | Path,
    expected_architecture: ModelArchitecture | str | None = None,
) -> PredictionData:
    source = Path(path)
    data = scipy.io.loadmat(source)

    lambda_current = _required(data, "X_La_curr_test", source).astype(np.float32)
    lambda_maximum = _required(data, "X_La_max_test", source).astype(np.float32)
    j_ratio = _required(data, "X_J_test", source).astype(np.float64)
    c_vector = _required(data, "X_C_test", source).astype(np.float32)
    f_vector = _required(data, "X_F_test", source).astype(np.float32)
    sigma_total = _required(data, "Y_sigma_total_test", source).astype(np.float32)

    _validate_rows(
        "X_La_curr_test",
        lambda_current,
        X_La_max_test=lambda_maximum,
        X_J_test=j_ratio,
        X_C_test=c_vector,
        X_F_test=f_vector,
        Y_sigma_total_test=sigma_total,
    )
    _validate_core_shapes(
        lambda_current,
        lambda_maximum,
        j_ratio,
        c_vector,
        f_vector,
        sigma_total,
    )
    stored_architecture = _optional_text(data, "model_architecture")
    _validate_architecture_metadata(
        stored_architecture, expected_architecture, j_ratio, source, data, "Y_p_test"
    )

    return PredictionData(
        lambda_current=lambda_current,
        lambda_maximum=lambda_maximum,
        j_ratio=j_ratio,
        c_vector=c_vector,
        f_vector=f_vector,
        sigma_total=sigma_total,
        case_lengths=_required(data, "Case_Lengths", source).reshape(-1).astype(int),
        stretch_path=_required(data, "landa_test1", source).reshape(-1),
        stretch_path_x=_required(data, "landa_testX", source).reshape(-1),
        pre_stretch_length=int(_scalar(data, "len_Y", source)),
        delta_j_ref=_scalar(data, "delta_J_ref", source, default=1.0e-4),
        model_architecture=stored_architecture,
    )


def assert_compatible_datasets(
    training_data: TrainingData, prediction_data: PredictionData
) -> None:
    """检查训练/预测数据是否共享相同的体积缩放和方向数量。"""
    if not np.isclose(
        training_data.delta_j_ref,
        prediction_data.delta_j_ref,
        rtol=0.0,
        atol=1.0e-14,
    ):
        raise ValueError(
            "训练和预测数据的 delta_J_ref 不一致："
            f"{training_data.delta_j_ref:.8g} vs "
            f"{prediction_data.delta_j_ref:.8g}。"
        )
    direction_count = training_data.directions.shape[0]
    if prediction_data.lambda_current.shape[1] != direction_count:
        raise ValueError(
            "训练球面方向数与预测输入列数不一致："
            f"{direction_count} vs {prediction_data.lambda_current.shape[1]}。"
        )


def assert_compatible_stages(reference: TrainingData, other: TrainingData) -> None:
    """继承参数时固定首阶段归一化及方向顺序，防止悄然改变模型物理含义。"""
    for name in ("directions", "weights"):
        left, right = getattr(reference, name), getattr(other, name)
        if left.shape != right.shape or not np.allclose(left, right, rtol=0, atol=1e-7):
            raise ValueError(f"顺序训练各阶段的 {name} 必须一致（含方向顺序）。")
    if not np.isclose(reference.delta_j_ref, other.delta_j_ref, rtol=0, atol=1e-14):
        raise ValueError("顺序训练各阶段的 delta_J_ref 必须一致。")
