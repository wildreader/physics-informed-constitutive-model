# 双轴 S_bar：无损伤与损伤分阶段训练

本文对应最新功能提交 `632cd56`（新增 S_bar 无损伤/损伤拟合），按数据 → 物理接口 → 训练 → 预测 → 结果组织。通用网络结构与层数见 [主 README 第 2 节](README.md#model)，初始化见 [第 4 节](README.md#hyperparameters)，MSE/RMSE 定义见 [第 6 节](README.md#metrics)。

本流程使用同一个三模块模型，只启用自由能和可选损伤；通过 `forward_sbar()` 输出未投影 S_bar。环境安装按 [快速开始](README.md#quick-start) 完成后，可直接运行本文命令，现有两份 MAT 无需重新生成。

## 已核实的数据

- `proportional_biaxial_Sbar_NoDamage.mat`：无损伤标签。
- `proportional_biaxial_Sbar.mat`：带损伤标签。
- 两文件输入完全相同；直接读取 `Xtrain`、`Ytrain`，不重新生成标签。
- MATLAB `Xtrain` 形状为 `[2,110,24005]`，`Ytrain` 为 `[6,24005]`。
- `Xtrain(1,:,:)` 是当前**方向伸长比**，`Xtrain(2,:,:)` 是历史最大方向伸长比；这里不是工程应变或 Green 应变。
- Python 输入分别为 `Xtrain[0].T` 和 `Xtrain[1].T`，形状 `[24005,110]`；标签为 `Ytrain.T`，形状 `[24005,6]`。转置不改变样本、方向、应力分量之间的对应关系。
- 按各工况逐方向累计最大值检查，第二通道与 `maximum.accumulate(max(current,1))` 的误差约 `1.1e-16`。由 loading 和 quadrature 重建当前方向伸长的误差约 `1.8e-15`。
- **五个工况比例为 `0,0.25,0.5,0.75,1`，各有 4801 点，总计 24005 点。**
- **每个工况的 lambda_x 路径为 `1→4→1→6→1→8→1→10→1`，步长绝对值 0.01。**
- lambda_y = 1 + ratio*(lambda_x-1)，lambda_z = 1/(lambda_x*lambda_y)。各工况历史独立重置。
- 方向与权重直接来自 `quadrature`，110 个方向，权重和为 1。
- 应力分量沿用项目的 `11,22,33,12,13,23` 顺序；本数据三个剪切分量接近机器零，仅凭这组双轴标签无法单独辨认剪切分量之间的排列。

## 物理接口与监督目标

本任务在“能量 → 自动微分 → 微球积分 → S_bar”处输出，直接与 `Ytrain` 比较。Cauchy 入口还会做 Flory 投影及应力转换，因此仅关闭体积分支不足以拟合本数据。`forward_sbar()` 拒绝启用体积分支，训练脚本只构造 `free_energy` / `free_energy_damage` 两种组合。

完整数据流和固定历史变量求导的说明见 [主 README 第 2 节](README.md#model)；网络层数、激活与初始化均沿用主模型，无须为 S_bar 更换网络。

## 两种模式

1. `no_damage`：只开启并训练自由能模块，**直接拟合无损伤 Ytrain；**损伤和体积模块不参与计算。
2. `damage`：先获得无损伤最佳自由能，随后冻结自由能，仅训练损伤模块，**直接拟合带损伤 Ytrain。**可以指定已训练的无损伤 S_bar 检查点，也可以自动先训练无损伤阶段。

**`both` 是便利命令，按顺序运行上述两种模式；使用同一个模型对象，在两个阶段之间继承权重，不重置自由能。无损伤和带损伤各保存一个独立的最佳检查点及全量预测结果，分别对应各自的数据目标。**

这沿用项目既有的逐模块训练策略，没有增加所有分支联合训练。带损伤阶段效果取决于无损伤自由能的拟合质量。

## 默认训练配置

与 Cauchy 分阶段训练相比，需注意：S_bar 两阶段学习率均为 0.001，按工况分层划分数据，输入统计量仅来自无损伤训练子集，并且两个阶段各保存一个检查点。

- Adam，学习率 0.001，batch size 256，各阶段 500 epoch。
- 每 200 epoch 学习率乘 0.5，损伤阶段重新初始化优化器。
- 随机种子 42，按工况分层划分 80% 训练/20% 验证，分别为 19204/4801 点；两阶段使用相同样本索引。
- 仅由无损伤训练子集计算当前方向伸长的均值和标准差，后续固定。历史最大伸长使用同一归一化尺度。
- 每阶段用自己的训练标签计算逐分量 RMS 尺度，训练六分量归一化 MSE。接近零的剪切分量使用尺度下限，避免除零和过度放大浮点噪声。
- 每阶段按最低验证损失选择最优 epoch；结束时恢复该 epoch，再预测全部 24005 点。
- 全量预测不会重新训练模型，也不会通过输入 Ytrain 帮助预测；Ytrain 只用于训练监督和事后计算误差。

随机验证点来自相同五条加载路径，误差衡量的是当前路径上的拟合/插值能力。若要评估新工况泛化，需要另做整条工况或加载循环留出实验。

### 可调参数

| 参数 | 默认值 / 用法 |
| --- | --- |
| `--mode` | `both`；可选 `no_damage`、`damage` |
| `--no-damage-data` / `--damage-data` | 根目录两份对应 MAT |
| `--elastic-checkpoint` | 仅用于 `damage`，继承无损伤 S_bar 最佳模型 |
| `--free-energy-epochs` / `--damage-epochs` | 各 500 |
| `--learning-rate` | 两阶段均为 0.001 |
| `--batch-size` | 256，同时用于训练和训练后预测 |
| `--validation-fraction` / `--seed` | 0.2 / 42 |
| `--output-dir` | `outputs/sbar` |
| `--device` | `auto`；可选 `cpu`、`cuda` |
| `--progress-interval` | 10 |

StepLR 的 200 / 0.5 在脚本中固定，没有对应命令行选项；最佳选择固定使用验证集。归一化 MSE 和 RMSE 的公式与换算关系见 [主 README 第 6 节](README.md#metrics)。

使用 `--elastic-checkpoint` 时仍需提供原无损伤 MAT（默认从根目录读取），路径必须与检查点记录一致；程序检查积分方向、权重和样本划分，并继承检查点中的输入统计量与划分。

## 运行命令

以下均从项目根目录、已激活的 Python 环境运行。默认自动选择设备；需要 GPU 时可显式添加 `--device cuda`。

推荐一次完成两种模式：

```powershell
python train_sbar.py --mode both
```

分别运行并继承已训练自由能：

```powershell
python train_sbar.py --mode no_damage
python train_sbar.py --mode damage --elastic-checkpoint outputs/sbar/no_damage/model.pth
```

自定义轮数/输出目录：

```powershell
python train_sbar.py --mode both --free-energy-epochs 500 --damage-epochs 500 --output-dir outputs/sbar_run2
```

重复使用同一个输出目录会更新其中的训练结果；不同实验使用不同目录。

## 输出与误差阅读

每阶段恢复最佳权重后，按原始 MAT 顺序预测全部 24005 点，包含训练与验证点。标签仅用于监督和计算误差，预测不依赖前一步应力递推；共用说明见 [主 README 第 7 节](README.md#outputs)。

`outputs/sbar/no_damage` 和 `outputs/sbar/damage` 各包含：

- `model.pth`：该阶段最佳模型，记录 `stress_measure=S_bar`、架构、数据路径、划分、缩放及最佳 epoch；不能用旧 Cauchy 检查点代替。
- `history.json`、`loss.png`：训练/验证损失。
- `full_prediction.mat`：`Sbar_pred` 与 `Sbar_true` 均为 `[6,24005]`，与原始 Ytrain 顺序一致；含 MATLAB 1-based 样本编号和训练/验证索引。
- `case_1.png` 至 `case_5.png`：各工况六个应力分量，目标与预测按原始加载/卸载顺序连线。
- `metrics.json`：全数据、训练、验证、各工况误差；主要观察三个正应力分量的相对 L2 和各分量 RMSE。剪切分量接近零，不用其相对误差判断拟合质量。

## 训练后再次预测（不重新训练）

预测最佳检查点中记录的完整原始 MAT，仍保持原样本顺序：

```powershell
python pred_sbar.py --checkpoint outputs/sbar/no_damage/model.pth
python pred_sbar.py --checkpoint outputs/sbar/damage/model.pth
```

默认输出到对应检查点目录下的 `prediction` 子目录。这个入口用于复现原数据集的完整曲线；另一个未见工况的数据集应另作独立评估，不沿用原训练/验证索引。

## 已有实验结果（2026-09-15）

以下保留原文实验记录；本次文档整合未重新训练，表中相对 L2 已与现有 `outputs/sbar/*/metrics.json` 核对。

RTX 4060 Laptop GPU；两阶段各 500 epoch；使用本说明的默认超参数。**误差为三个正应力分量合并的相对 L2。**

| 模式 | 最优 epoch | 全数据误差 | 验证误差 |
|---|---:|---:|---:|
| no_damage | 494 | 0.854% | 0.889% |
| damage | 495 | 1.849% | 1.883% |

各工况完整路径误差：

| 双轴比例 | 无损伤 | 带损伤 |
|---|---:|---:|
| 0 | 2.071% | 2.239% |
| 0.25 | 1.814% | 3.363% |
| 0.5 | 0.828% | 2.028% |
| 0.75 | 0.615% | 1.662% |
| 1 | 0.372% | 1.349% |

主要曲线和损伤加载/卸载响应能够拟合，但不是逐点完全重合。无损伤第一工况高伸长处 S33 仍有偏差；带损伤第二工况合并误差约 3.36%。剪切分量的预测约为浮点数值噪声量级。

原实验验证记录（本次未重跑）：12 项单元/回归测试通过；两个最佳检查点独立重预测与训练后自动预测的最大差均为 0；两文件目标顺序均与原始 Ytrain 一致；损伤前后自由能权重逐张量完全相同。

`outputs/sbar_smoke` 与 `outputs/sbar_checkpoint_smoke` 是流程试跑，正式结果位于 `outputs/sbar`。
