"""模块化高分子链本构 PINN。"""

from .config import ARCHITECTURE_CHOICES, ModelArchitecture
from .model import FullyDecoupledPINN, ModularConstitutivePINN

__all__ = [
    "ARCHITECTURE_CHOICES",
    "FullyDecoupledPINN",
    "ModelArchitecture",
    "ModularConstitutivePINN",
]

