# 两个双轴 S_bar 数据集的训练方式

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

## 原来输出曲线是怎样得到的

`train4.py` 在最终训练阶段结束后，用恢复的最佳模型，对该阶段 MAT 中的完整原始**输入按顺序分批预测。所以曲线包含内部划分后的训练点和验证点。**

训练默认打乱训练批次，验证不打乱。**曲线的绘制顺序使用原始 MAT 样本顺序，**不是训练 DataLoader 的随机顺序，也不是把随机训练集和验证集简单拼接起来。

它是逐样本本构预测，不是递归预测：最大方向伸长由输入直接提供，不使用上一个预测应力更新下一个样本。

`pred2.py` 则加载检查点，预测另外指定的测试 MAT 文件；这和训练结束自动导出的全数据曲线不同。

## 为什么需要调整架构

原物理路径为：自由能（可乘损伤系数）→自动微分→微球积分 S_bar→Flory 投影 S_iso→可选体积应力→Cauchy 应力。

新路径为：自由能（可乘损伤系数）→自动微分→微球积分 S_bar→直接与 Ytrain 比较。

仅关闭体积分支仍会执行 Flory 投影和 Cauchy 转换，不能用于直接拟合 S_bar。因此增加了 `model.forward_sbar(current, maximum)`，并让原 forward 和它共用内部 S_bar 计算。S_bar 接口拒绝 full 架构，避免体积分支无监督地启用。旧 `train4.py` / `pred2.py` 的 Cauchy 流程保留。

求当前方向伸长的导数时，将已给定的历史最大伸长视为独立历史变量。训练时仍保留自由能导数对网络参数的梯度；验证/预测时也局部启用自动微分，不能直接用 inference_mode 禁掉能量求导。

## 两种模式

1. `no_damage`：只开启并训练自由能模块，**直接拟合无损伤 Ytrain；**损伤和体积模块不参与计算。
2. `damage`：先获得无损伤最佳自由能，随后冻结自由能，仅训练损伤模块，**直接拟合带损伤 Ytrain。**可以指定已训练的无损伤 S_bar 检查点，也可以自动先训练无损伤阶段。

**`both` 是便利命令，按顺序运行上述两种模式；使用同一个模型对象，在两个阶段之间继承权重，不重置自由能。无损伤和带损伤各保存一个独立的最佳检查点及全量预测结果，分别对应各自的数据目标。**

这沿用项目既有的逐模块训练策略，没有增加所有分支联合训练。带损伤阶段效果取决于无损伤自由能的拟合质量。

## 默认训练配置

- Adam，学习率 0.001，batch size 256，各阶段 500 epoch。
- 每 200 epoch 学习率乘 0.5，损伤阶段重新初始化优化器。
- 随机种子 42，按工况分层划分 80% 训练/20% 验证，分别为 19204/4801 点；两阶段使用相同样本索引。
- 仅由无损伤训练子集计算当前方向伸长的均值和标准差，后续固定。历史最大伸长使用同一归一化尺度。
- 每阶段用自己的训练标签计算逐分量 RMS 尺度，训练六分量归一化 MSE。接近零的剪切分量使用尺度下限，避免除零和过度放大浮点噪声。
- 每阶段按最低验证损失选择最优 epoch；结束时恢复该 epoch，再预测全部 24005 点。
- 全量预测不会重新训练模型，也不会通过输入 Ytrain 帮助预测；Ytrain 只用于训练监督和事后计算误差。

随机验证点来自相同五条加载路径，误差衡量的是当前路径上的拟合/插值能力。若要评估新工况泛化，需要另做整条工况或加载循环留出实验。

## 运行命令（在项目根目录 PowerShell 中）

推荐一次完成两种模式：

```powershell
& 'C:\programming\anaconda3\envs\torch\python.exe' train_sbar.py --mode both --device cuda
```

分别运行并继承已训练自由能：

```powershell
& 'C:\programming\anaconda3\envs\torch\python.exe' train_sbar.py --mode no_damage --device cuda
& 'C:\programming\anaconda3\envs\torch\python.exe' train_sbar.py --mode damage --elastic-checkpoint outputs/sbar/no_damage/model.pth --device cuda
```

自定义轮数/输出目录：

```powershell
& 'C:\programming\anaconda3\envs\torch\python.exe' train_sbar.py --mode both --free-energy-epochs 500 --damage-epochs 500 --output-dir outputs/sbar_run2 --device cuda
```

重复使用同一个输出目录会更新其中的训练结果；不同实验使用不同目录。

## 输出

`outputs/sbar/no_damage` 和 `outputs/sbar/damage` 各包含：

- `model.pth`：该阶段最佳模型，记录 `stress_measure=S_bar`、架构、数据路径、划分、缩放及最佳 epoch；不能用旧 Cauchy 检查点代替。
- `history.json`、`loss.png`：训练/验证损失。
- `full_prediction.mat`：`Sbar_pred` 与 `Sbar_true` 均为 `[6,24005]`，与原始 Ytrain 顺序一致；含 MATLAB 1-based 样本编号和训练/验证索引。
- `case_1.png` 至 `case_5.png`：各工况六个应力分量，目标与预测按原始加载/卸载顺序连线。
- `metrics.json`：全数据、训练、验证、各工况误差；主要观察三个正应力分量的相对 L2 和各分量 RMSE。剪切分量接近零，不用其相对误差判断拟合质量。

## 训练后再次预测（不重新训练）

预测最佳检查点中记录的完整原始 MAT，仍保持原样本顺序：

```powershell
& 'C:\programming\anaconda3\envs\torch\python.exe' pred_sbar.py --checkpoint outputs/sbar/no_damage/model.pth --device cuda
& 'C:\programming\anaconda3\envs\torch\python.exe' pred_sbar.py --checkpoint outputs/sbar/damage/model.pth --device cuda
```

默认输出到对应检查点目录下的 `prediction` 子目录。这个入口用于复现原数据集的完整曲线；另一个未见工况的数据集应另作独立评估，不沿用原训练/验证索引。

## 本次实际训练结果（2026-09-15）

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

验证：12 项单元/回归测试通过；两个最佳检查点独立重预测与训练后自动预测的最大差均为 0；两文件目标顺序均与原始 Ytrain 一致；损伤前后自由能权重逐张量完全相同。

`outputs/sbar_smoke` 与 `outputs/sbar_checkpoint_smoke` 是流程试跑，正式结果位于 `outputs/sbar`。
