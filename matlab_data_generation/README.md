# 与单模型顺序训练匹配的 MATLAB 数据

## 一次生成全部数据

从项目根目录在 MATLAB 中运行：

```matlab
run('matlab_data_generation/generate_all_datasets.m')
```

公共实现为 `common/generate_polymer_dataset.m`，Lebedev 方向数 110、步长 0.01、损伤耦合系数 alpha=0.5。保留原七工况训练路径、五工况测试路径和预拉伸顺序，输出目录为 `generated/<architecture>/`，各目录包含 `train_7cases.mat` 和 `test_5cases.mat`。

## 三个数据模式

| 模式 | Fc | F / J | 压力、体积应力 |
|---|---|---|---|
| free_energy | force_chain | F=F_bar / J=1 | 不计算体积本构公式；压力占位为 0 |
| free_energy_damage | Dmg_chain .* force_chain | F=F_bar / J=1 | 不计算体积本构公式；压力占位为 0 |
| full | Dmg_chain .* force_chain | F=F_bar*F_vol / 原 J 路径 | p=k_vol*(J-1)，S_vol=J*p*Cinv |

公共链力 `force_chain=(max(LS,1)-1).^3`。关闭损伤时不计算损伤历史与 D_local、D_avg、Dmg_chain，直接 Fc=force_chain。开启时保留图片中的局部/全局剩余刚度公式：`Dmg_chain=(1-alpha)*D_local+alpha*D_avg`，对应网络 `(1-d)`。

前两模式不计算 F_vol、静水压或 S_vol 公式，总应力直接用 S_iso。只有 full 使用 `delta_J_ref=1e-4` 的 J 变化以及 `k_vol=1e6`。delta_J_ref 仍保留在每份 MAT 中作为模型统一参考尺度。

三个模式共用 MAT 张量字段。free_energy 的 X_La_max 为全 1 的占位字段，模型不读取；前两模式 Y_p 为全 0。训练 MAT 额外包含 Dir、Wt。每份数据均记录 model_architecture、use_damage、use_volumetric、dataset_format_version=2；k_vol 只在 full 中保存。

旧版本前两模式使用了变化的 J；请重新生成，Python 会拒绝将它们用于当前等体积阶段。

## 单独生成与绘图

```matlab
addpath('matlab_data_generation/common')
generate_polymer_dataset('train', 'free_energy', true);
generate_polymer_dataset('test', 'free_energy', true);
```

将模式改为 `free_energy_damage` 或 `full` 即可生成其他模式。第三个参数控制是否绘图，批量入口默认 false。

也可使用保留的快捷入口：

```matlab
run('matlab_data_generation/architecture_1_free_energy/generate_train_7cases.m')
run('matlab_data_generation/architecture_2_free_energy_damage/generate_train_7cases.m')
run('matlab_data_generation/architecture_3_full/generate_train_7cases.m')
```

各目录中的 `generate_test_5cases.m` 生成对应测试集。这些目录名称是数据模式快捷入口，不代表三个独立神经网络。

## Python 自动读取顺序

- `python train4.py --architecture free_energy`：仅读取自由能训练数据。
- `python train4.py --architecture free_energy_damage`：依次读取自由能、自由能+损伤数据。
- `python train4.py --architecture full`：依次读取上述两套及完整数据。

每阶段使用自己的标签；前一阶段最佳权重保留在同一个模型中。预测只读取最终模块组合对应的测试数据。

完整命令、初始化、学习率及最优保存规则见 [单模型顺序训练使用说明](../README_单模型顺序训练.md)。
