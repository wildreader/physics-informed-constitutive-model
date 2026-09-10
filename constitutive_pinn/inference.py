"""分批预测总应力及各物理分量。"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .model import ModularConstitutivePINN


@dataclass(frozen=True)
class PredictionComponents:
    sigma_total: np.ndarray
    sigma_iso: np.ndarray
    sigma_vol: np.ndarray
    pressure: np.ndarray


def predict_components(
    model: ModularConstitutivePINN,
    lambda_current: np.ndarray,
    lambda_maximum: np.ndarray,
    j_ratio: np.ndarray,
    c_vector: np.ndarray,
    f_vector: np.ndarray,
    device: torch.device,
    batch_size: int = 1024,
) -> PredictionComponents:
    if batch_size <= 0:
        raise ValueError("预测 batch_size 必须为正整数。")
    model.eval()
    outputs: list[list[np.ndarray]] = [[], [], [], []]

    for start in range(0, lambda_current.shape[0], batch_size):
        end = min(start + batch_size, lambda_current.shape[0])
        current_tensor = torch.from_numpy(lambda_current[start:end]).to(device)
        maximum_tensor = (
            torch.from_numpy(lambda_maximum[start:end]).to(device)
            if model.use_damage
            else None
        )
        j_tensor = torch.from_numpy(j_ratio[start:end]).to(device)
        c_tensor = torch.from_numpy(c_vector[start:end]).to(device)
        f_tensor = torch.from_numpy(f_vector[start:end]).to(device)

        # forward 内部会局部开启能量求导所需的 autograd。
        with torch.no_grad():
            batch_outputs = model(
                current_tensor,
                maximum_tensor,
                j_tensor,
                c_tensor,
                f_tensor,
                return_components=True,
            )
        for collection, tensor in zip(outputs, batch_outputs):
            collection.append(tensor.detach().cpu().numpy())

    sigma_total, sigma_iso, sigma_vol, pressure = (
        np.concatenate(parts, axis=0) for parts in outputs
    )
    return PredictionComponents(
        sigma_total=sigma_total,
        sigma_iso=sigma_iso,
        sigma_vol=sigma_vol,
        pressure=pressure,
    )

