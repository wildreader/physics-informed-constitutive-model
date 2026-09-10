"""单个模型的模块组合及顺序训练配置。

用户只需要选择一个架构名称，不需要在各处手工同步多个布尔开关。
自由能支路始终启用；损伤和体积分支按三种受支持的组合逐级加入。
"""

from __future__ import annotations

from enum import Enum


class ModelArchitecture(str, Enum):
    """同一个本构模型支持的三种模块组合，而非三个模型类。"""

    FREE_ENERGY = "free_energy"
    FREE_ENERGY_DAMAGE = "free_energy_damage"
    FULL = "full"

    @property
    def training_stages(self) -> tuple["ModelArchitecture", ...]:
        stages = (self.FREE_ENERGY, self.FREE_ENERGY_DAMAGE, self.FULL)
        return stages[: stages.index(self) + 1]

    @property
    def module_name(self) -> str:
        return {
            self.FREE_ENERGY: "free_energy",
            self.FREE_ENERGY_DAMAGE: "damage",
            self.FULL: "volumetric",
        }[self]

    @property
    def use_damage(self) -> bool:
        return self in {self.FREE_ENERGY_DAMAGE, self.FULL}

    @property
    def use_volumetric(self) -> bool:
        return self is self.FULL

    @property
    def description(self) -> str:
        descriptions = {
            self.FREE_ENERGY: "自由能支路（d=0，p=0，S_vol=0）",
            self.FREE_ENERGY_DAMAGE: "自由能+损伤支路（p=0，S_vol=0）",
            self.FULL: "自由能+损伤+体积分支",
        }
        return descriptions[self]

    @classmethod
    def parse(cls, value: "ModelArchitecture | str") -> "ModelArchitecture":
        if isinstance(value, cls):
            return value

        normalized = str(value).strip().lower().replace("-", "_")
        aliases = {
            "energy": cls.FREE_ENERGY,
            "elastic": cls.FREE_ENERGY,
            "free_energy": cls.FREE_ENERGY,
            "energy_damage": cls.FREE_ENERGY_DAMAGE,
            "elastic_damage": cls.FREE_ENERGY_DAMAGE,
            "free_energy_damage": cls.FREE_ENERGY_DAMAGE,
            "all": cls.FULL,
            "fully_decoupled": cls.FULL,
            "full": cls.FULL,
        }
        try:
            return aliases[normalized]
        except KeyError as exc:
            choices = ", ".join(item.value for item in cls)
            raise ValueError(
                f"未知模型架构 {value!r}；可选值为：{choices}。"
            ) from exc


ARCHITECTURE_CHOICES = tuple(item.value for item in ModelArchitecture)
