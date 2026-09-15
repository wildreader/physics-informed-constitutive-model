# 聚合物三模块本构 PINN：分阶段训练完整指南

基于 PyTorch 的单模型本构建模项目：以自由能、损伤和体积三个神经网络模块，结合微球积分、自动微分与 Flory 等体积–体积分解，预测有限变形下的应力。

根目录文档整合为两份：

| 文档 | 内容 |
| --- | --- |
| **README.md（本文）** | 分阶段训练完整流程，以及共用的网络层数、数据流、初始化、超参数、损失和评估指标 |
| [README_Sbar训练.md](README_Sbar训练.md) | 最新提交 `632cd56` 的双轴 S_bar 数据、训练、预测与已有实验结果；共用原理按章节引用本文 |

| 任务 | 训练 / 预测入口 | 监督目标 |
| --- | --- | --- |
| 三模块分阶段训练 | `train4.py` / `pred2.py` | 投影与应力转换后的总 Cauchy 应力 |
| 双轴 S_bar 拟合 | `train_sbar.py` / `pred_sbar.py` | 未经 Flory 投影的 S_bar；只启用自由能与可选损伤 |

以下命令及默认参数除明确标注外，均针对 `train4.py` / `pred2.py`。两条流程共享网络结构，但数据接口、部分参数和检查点格式不同。

## 目录

1. [环境、数据与快速运行](#quick-start)
2. [模型架构、网络层数与力学数据流](#model)
3. [分阶段训练与最佳模型继承](#stages)
4. [初始化、输入归一化与超参数](#hyperparameters)
5. [阶段数据与合成标签](#data)
6. [归一化 MSE、RMSE 与相对误差](#metrics)
7. [检查点、输出曲线与预测评估](#outputs)
8. [项目结构、验证与常见问题](#reference)

<a id="quick-start"></a>

## 1. 环境、数据与快速运行

以下命令均从项目根目录执行。

### 1.1 准备 Python 环境

项目依赖 PyTorch、NumPy、SciPy、scikit-learn 和 Matplotlib。可使用 Python 3.10 或更高版本建立虚拟环境；当前仓库未提供依赖版本锁定文件。

```bash
python -m venv .venv
```

激活环境：

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

```bash
# Linux / macOS
source .venv/bin/activate
```

安装依赖：

```bash
python -m pip install torch numpy scipy scikit-learn matplotlib
```

使用 GPU 时，环境需要安装与本机驱动兼容的 CUDA 版 PyTorch。默认 `--device auto` 会在 CUDA 可用时使用 GPU，否则使用 CPU；也可显式指定 `--device cpu` 或 `--device cuda`。

### 1.2 准备数据

数据由 MATLAB 生成。如果已有与当前代码匹配的六份 MAT 文件，可直接进入训练步骤；重新生成数据时，在 MATLAB 中将工作目录切换到项目根目录并运行：

```matlab
run('matlab_data_generation/generate_all_datasets.m')
```

生成目录：

```text
matlab_data_generation/generated/
├── free_energy/
│   ├── train_7cases.mat
│   └── test_5cases.mat
├── free_energy_damage/
│   ├── train_7cases.mat
│   └── test_5cases.mat
└── full/
    ├── train_7cases.mat
    └── test_5cases.mat
```

默认采用 110 个 Lebedev 方向、加载步长 `0.01` 和损伤耦合系数 `alpha=0.5`。前两种数据为严格等体积路径 `J=1`；`full` 数据加入体积变化与压力响应。

训练阶段必须与数据匹配：自由能阶段读取 `free_energy`，损伤阶段读取 `free_energy_damage`，体积阶段读取 `full`。程序会检查架构元数据、张量形状、体积比，以及阶段之间的积分方向、权重和参考尺度。

### 1.3 训练模型

运行完整三模块顺序训练：

```bash
python train4.py --architecture full
```

仅训练自由能，或训练自由能与损伤：

```bash
python train4.py --architecture free_energy
python train4.py --architecture free_energy_damage
```

默认结果保存到 `outputs/<architecture>/`。

### 1.4 预测与评估

```bash
python pred2.py --architecture full
```

预测配置必须与检查点匹配。评估过程会读取该配置对应的训练 MAT 文件、测试 MAT 文件和检查点；训练文件用于提供积分方向、权重等信息。程序输出测试集总应力 MSE、六个分量的 RMSE，并保存预测数据与五工况对比图。

其他配置使用对应命令：

```bash
python pred2.py --architecture free_energy
python pred2.py --architecture free_energy_damage
```

### 1.5 快速检查完整流程

每阶段仅运行 1 个 epoch，并保存到独立目录：

```bash
python train4.py --architecture full --free-energy-epochs 1 --damage-epochs 1 --volumetric-epochs 1 --output-dir outputs/smoke_full
python pred2.py --architecture full --checkpoint outputs/smoke_full/model.pth --output-dir outputs/smoke_full/prediction
```

该示例用于检查数据、训练与预测流程是否连通，不用于判断模型精度。

<a id="model"></a>

## 2. 模型架构、网络层数与力学数据流

仅有一个 `ModularConstitutivePINN`，内部始终包含三个模块。`--architecture` 控制最终启用组合；三种配置的参数结构和参数名相同，关闭的模块仍保留在模型及检查点中。每次训练只创建一个模型，各阶段切换数据、模块开关和优化器。

### 2.1 网络结构与层数

**局部特征网络为 `1-32-32-6`，自由能/损伤预测器为 `12-64-64-1`，体积网络为 `1-32-32-1`。**

下表按 `nn.Linear` 计层，不把输入、激活函数、池化或解析物理运算计为线性层。

| 模块 / 子网络 | 结构 | 线性层数与隐藏层 | 激活与输出 |
| --- | --- | --- | --- |
| 自由能局部提取器 `free_energy_module.extractor` | `1 → 32 → 32 → 6` | 3 层；2 个 32 宽隐藏层 | 前两层后为 Softplus；输出 6 维局部特征 |
| 自由能预测器 `free_energy_module.predictor` | `12 → 64 → 64 → 1` | 3 层；2 个 64 宽隐藏层 | 每层后为 Softplus；输出非负方向自由能 φ |
| 损伤局部提取器 `damage_module.extractor` | `1 → 32 → 32 → 6` | 3 层；2 个 32 宽隐藏层 | 前两层后为 Softplus；输出 6 维局部特征 |
| 损伤预测器 `damage_module.predictor` | `12 → 64 → 64 → 1` | 3 层；2 个 64 宽隐藏层 | 前两层后为 Softplus，末端 Sigmoid；输出损伤 d |
| 体积自由能 `volumetric_module` | `1 → 32 → 32 → 1` | 3 层；2 个 32 宽隐藏层 | 前两层后为 Softplus；末端线性输出 U_vol |

自由能、损伤模块各含 6 个线性层，体积模块含 3 个，完整模型合计 15 个。自由能与损伤结构相似但参数独立；同一模块的局部提取器在所有积分方向上共享参数。

### 2.2 局部与全局特征如何组合

设批大小为 B、积分方向数为 D（默认 110）。自由能输入是当前方向伸长，损伤输入是已给定的历史最大方向伸长，两者均为 `[B,D]`。

```text
方向标量 [B,D]
  → 输入归一化
  → 共享局部网络 1→32→32→6
  → 局部特征 [B,D,6]
  → 按 Lebedev 权重对方向求和，得到全局特征 [B,1,6]
  → 将全局特征广播到每个方向，与局部特征拼接 [B,D,12]
  → 预测器 12→64→64→1
  → 方向自由能 φ 或损伤 d [B,D]
```

预测器的 12 维输入来自“6 维局部 + 6 维全局”，不是 12 个应变分量。

### 2.3 从能量到应力

```text
当前方向伸长 → 自由能模块 → φ ──────────┐
历史最大伸长 → 损伤模块 → d（可选）──────┤
                                      ↓
                     方向能量 ψ = φ 或 (1-d)φ
                                      ↓
                   加权宏观能量 W = Σ w_a ψ_a
                                      ↓ 自动微分 ∂W/∂λ_a
                           加权链力 → 微球积分
                                      ↓
                                    S_bar
                          ┌───────────┴───────────┐
                 forward_sbar()                forward()
                          ↓                       ↓
                   直接输出 S_bar          Flory 投影 → S_iso
                                                  ↑
J → J_hat → 体积网络 → U_vol → p=dU_vol/dJ → S_vol（可选）
                                                  ↓
                                         S = S_iso + S_vol
                                                  ↓
                                         σ = F S Fᵀ / J
```

微球积分使用 `S_bar = Σ_a [(∂W/∂λ_a)/(λ_a+1e-8)] (n_a⊗n_a)`。导数已经包含能量中的积分权重，物理层不重复乘权重。投影与体积项为：

```text
S_iso = J^(-2/3) [S_bar - (C:S_bar) C^(-1)/3]
S_vol = J p C^(-1)    （体积模块关闭时为零）
```

输出六分量排列统一为 `11,22,33,12,13,23`。关闭体积模块时，Cauchy 流程仍执行投影与转换；直接拟合 S_bar 必须使用 `forward_sbar()`。

求当前伸长导数时，历史最大伸长作为独立历史变量固定。训练保留能量导数对网络参数的梯度；验证/预测也需要局部自动微分，不能用 `torch.inference_mode()` 禁用能量求导。

<a id="stages"></a>

## 3. 分阶段训练与最佳模型继承

| 命令参数 | 第 1 阶段 | 第 2 阶段 | 第 3 阶段 | 默认总 epoch |
|---|---|---|---|---:|
| `free_energy` | 只开启、训练自由能 500 epoch | — | — | 500 |
| `free_energy_damage` | 只开启、训练自由能 500 epoch | 继承自由能**最佳权重**，开启损伤，仅训练损伤 500 epoch | — | 1000 |
| `full` | 只开启、训练自由能 500 epoch | 继承自由能**最佳权重**，仅训练损伤 500 epoch | 继承自由能和损伤最佳权重，仅训练体积 500 epoch | 1500 |

**每阶段结束先恢复该阶段最低 loss 对应的权重，再进入下一阶段。已完成模块继续参与前向计算，但参数冻结；后续阶段不能修改其权重和偏置。没有原来的 150 epoch 损伤冻结预热，也没有所有模块联合训练。**

损伤、体积阶段默认也设为 500 epoch，沿用原主训练轮数。可分别通过 `--free-energy-epochs`、`--damage-epochs`、`--volumetric-epochs` 调整。已开启阶段必须至少为 1 epoch，不能跳过尚未训练的前置模块。

<a id="hyperparameters"></a>

## 4. 初始化、输入归一化与超参数

### 4.1 参数初始化

- 普通线性层权重使用 Xavier normal，偏置为 0。
- 损伤预测器最后一个线性层的权重与偏置再置零，首次启用时 `d=sigmoid(0)=0.5`。自由能阶段完全绕过损伤，不会乘以 0.5。
- 体积网络最后一个线性层权重与偏置再置零，首次启用时 `U_vol=0`、`p=0`。
- 三个模块在模型构造时仅初始化一次。阶段切换恢复最佳权重，冻结已完成模块，不重新初始化。

### 4.2 输入归一化

当前和历史最大方向伸长共用 `x_hat=(x-mean)/(std+1e-8)`；统计量在自由能阶段确定，后续固定，避免继承权重后自由能函数发生变化。

| 流程 | mean / std 来源 |
| --- | --- |
| `train4.py` | 首阶段 MAT 的完整当前方向伸长数据；标准差为零时设为 1 |
| `train_sbar.py` | 无损伤数据划分后的训练子集；继承检查点时恢复已保存的统计量和划分 |

体积输入单独使用 `J_hat=(J-1)/delta_J_ref`，默认 `delta_J_ref=1e-4`。压力是对原始 J 求导，自动微分包含归一化的链式法则。

输入统计量与应力损失尺度是两件事：前者固定，后者在每阶段根据该阶段训练子集重新计算。

### 4.3 训练参数与学习率

| 训练参数 | 默认值 / 行为 |
| --- | --- |
| `--architecture` | `full` |
| `--free-energy-epochs` / `--damage-epochs` / `--volumetric-epochs` | 每个启用阶段 500 epoch |
| `--batch-size` | `256` |
| `--free-energy-lr` | 仅自由能时 `0.001`；多模块首阶段 `0.002` |
| `--damage-lr` / `--volumetric-lr` | `0.001` |
| `--scheduler-step` / `--scheduler-gamma` | 每 200 epoch 将学习率乘以 `0.5`；各阶段独立计数 |
| `--selection-loss` | `validation`；可选 `train` |
| `--seed` | `42` |
| `--device` | `auto`；可选 `cpu`、`cuda` |
| `--output-dir` | `outputs/<architecture>` |
| `--checkpoint` | `<output-dir>/model.pth`，指定检查点输出路径 |
| `--show` | 显示生成的 Matplotlib 图窗，默认仅保存图片 |

训练使用 Adam 优化器，验证比例为 `0.2`，损失为按训练子集各应力分量 RMS 归一化的总应力 MSE。每阶段重新计算损失尺度；方向伸长的输入归一化统计量来源见本节上表。

自定义各阶段的数据文件：

```bash
python train4.py --architecture full --free-energy-data path/to/energy.mat --damage-data path/to/damage.mat --volumetric-data path/to/full.mat
```

训练参数 `--train-data` 只覆盖最终阶段的数据，不能与最终阶段对应的专用数据参数同时使用。预测脚本通过 `--train-data`、`--test-data` 和 `--checkpoint` 指定输入；预测的 `--batch-size` 默认为 `1024`。

查看完整参数：

```bash
python train4.py --help
python pred2.py --help
```

每阶段重新建立 Adam 和独立 StepLR。学习率按 `lr(epoch)=lr0 × gamma^floor((epoch-1)/step)` 变化（epoch 从 1 开始）：默认第 1–200、201–400、401–500 轮分别使用 `lr0`、`lr0/2`、`lr0/4`。新阶段从自身初始学习率重新计数。

补充参数：`--prediction-batch-size` 默认 1024，控制训练结束后的全量预测；`--progress-interval` 默认 50。`train4.py` 验证比例 0.2 固定在 `TrainingOptions`，没有对应 CLI 参数；划分为随机样本划分，未按工况分层。S_bar 的参数差异见其专属文档。

<a id="data"></a>

## 5. 阶段数据与合成标签

在 MATLAB 中先运行一次（项目根目录为当前目录）：

```matlab
run('matlab_data_generation/generate_all_datasets.m')
```

这会生成三套数据，每套含七工况训练集与五工况测试集：

```text
matlab_data_generation/generated/
  free_energy/train_7cases.mat
  free_energy/test_5cases.mat
  free_energy_damage/train_7cases.mat
  free_energy_damage/test_5cases.mat
  full/train_7cases.mat
  full/test_5cases.mat
```

阶段数据选择固定为：自由能阶段读取 `free_energy`；损伤阶段读取 `free_energy_damage`；体积阶段读取 `full`。例如 `--architecture full` 会依次读取三套训练数据，不能三个阶段都用含损伤与体积项的 full 标签。

| 数据模式 | 链力 Fc | 运动学 | 压力与体积应力 |
|---|---|---|---|
| `free_energy` | `Fc = force_chain` | `F=F_bar, J=1` | 不计算 p、S_vol 的本构公式 |
| `free_energy_damage` | `Fc = Dmg_chain .* force_chain` | `F=F_bar, J=1` | 不计算 p、S_vol 的本构公式 |
| `full` | `Fc = Dmg_chain .* force_chain` | `F=F_bar*F_vol`，J 按原路径变化 | `p=k_vol*(J-1)`，`S_vol=J*p*Cinv` |

公共链力公式是 `force_chain=(max(LS,1)-1).^3`。仅开启损伤时计算：

```matlab
D_local = max((12.0 - max(LS_max, 1.0))/12.0, 0.0);
D_avg = sum(Wt .* D_local);
Dmg_chain = (1.0-alpha)*D_local + alpha*D_avg;
Fc = Dmg_chain .* force_chain;
```

公式中的 `Dmg_chain` 实际表示剩余刚度/能量比例，对应网络的 `(1-d)`，并非网络输出的损伤 d。`alpha=0.5` 保持不变。纯自由能分支不计算这些损伤公式，直接 `Fc=force_chain`。

**前两种数据也不调用体积变形函数，**不计算 `k_vol*(J-1)` 或 `J*p*Cinv`，总应力直接取等体积分量。为保持 MAT 接口统一，仍导出值为 0 的压力字段；纯自由能的历史最大伸长字段为全 1 占位，不参与模型计算。`delta_J_ref=1e-4` 作为固定参考尺度保留，只有 full 保存和使用 `k_vol=1e6`。

旧版前两种数据虽然 p=0，**但 J 仍随加载变化；不符合当前严格等体积路径**，必须重新生成。Python 会检查模块组合、J、字段形状与阶段间方向/权重/参考尺度，避免静默误用。

更多 MATLAB 入口见 [数据生成说明](matlab_data_generation/README.md)。

<a id="metrics"></a>

## 6. 归一化 MSE、RMSE 与相对误差

以下令 `y_ij` 为真实应力、`ŷ_ij` 为预测应力，i 是样本、j 是六个应力分量。Cauchy 流程使用总应力 σ，S_bar 流程使用 S_bar 标签，公式相同。

### 6.1 训练损失：逐分量 RMS 归一化 MSE

每阶段只用自己的训练子集计算标签尺度；验证沿用相同尺度：

```text
r_j = sqrt(mean_train(y_ij²))                         # 标签 RMS
floor = max(1e-4 × max(max_j(r_j), 1e-8), 1e-8)
s_j = max(r_j, floor)
L = (1/(6N)) Σ_i Σ_j [(ŷ_ij-y_ij)/s_j]²              # 归一化 MSE
```

尺度下限防止零应力或接近机器零的剪切分量被过度放大。模型直接输出应力，归一化仅用于计算损失，不需要对预测结果再乘尺度“反归一化”。

默认以验证集 L 选择最佳模型，**并未把训练目标改成 RMSE**。不同阶段的标签、尺度不同，loss 不跨阶段比较。

### 6.2 评估：原始应力 MSE 与逐分量 RMSE

```text
MSE_j = (1/N) Σ_i (ŷ_ij-y_ij)²
RMSE_j = sqrt(MSE_j)
MSE_all = (1/6) Σ_j MSE_j
RMSE_all = sqrt(MSE_all) = sqrt((1/6) Σ_j RMSE_j²)
```

`pred2.py` 报告六分量合并的 `MSE_all` 与六个 `RMSE_j`；`RMSE_all` 是由 MSE 可推导的指标，当前并不单独导出。MSE 的单位为应力单位的平方，RMSE 与 MAT 标签的应力单位相同；不能直接把 `sqrt(L)` 当作原始应力 RMSE。

对同一固定模型、同一批评估样本和相同尺度，有：

```text
L = (1/6) Σ_j (RMSE_j/s_j)²
```

只有掌握逐分量归一化 MSE，才能分别通过 `RMSE_j=s_j×sqrt(L_j)` 还原；一个合并 L 无法反推出六个 RMSE。训练图中的批次平均 loss 混合了更新过程中的多份权重，也不能直接套用此关系换算最终模型 RMSE。

### 6.3 S_bar 的相对 L2

S_bar 还报告三个正应力分量合并的相对 L2：

```text
normal_relative_l2 = ||ŷ[:,0:3]-y[:,0:3]||_F / max(||y[:,0:3]||_F, 1e-12)
百分比误差 = 100 × normal_relative_l2
```

它是三个正应力合并后的误差范数比例，并非三个分量相对误差的算术平均。`normal_component_relative_l2` 则逐正应力分量计算 `RMSE_j/max(RMS_j,1e-12)`，这里 RMS 取自当前评估样本。近零剪切分量主要观察 RMSE 和最大绝对误差。

<a id="outputs"></a>

## 7. 检查点、输出曲线与预测评估

### 7.1 最佳模型与保存规则

默认以每个 epoch 完成后的验证集归一化 MSE 为 loss 选择标准。需要按训练集 loss 选择时，添加：

```powershell
python train4.py --architecture full --selection-loss train
```

`train` 模式会在每个 epoch 更新结束后重新评估整个训练子集，保证比较的 loss 和保存的权重来自同一时刻，而非不同 batch 更新过程中混合的 loss。

**每阶段单独选择最佳 epoch，不跨阶段比较 loss**（标签、归一化尺度不同）。最优规则采用严格小于；相同 loss 保留最早达到的 epoch。

- 前置阶段**最佳状态仅暂存在内存，用于恢复与继承，不另存阶段模型**。
- 最终阶段每当 loss 改善，原子覆盖唯一的 `model.pth`。最后一个 epoch 不是最优时，不覆盖最佳模型。
- **训练结束恢复最终阶段最佳权重，再生成训练应力曲线和预测结果，确保图与检查点对应**。
- 不保存逐 epoch 模型，也不保存 last 模型。检查点中包含三个模块的参数、最终开关、归一化、每阶段最佳 epoch/loss、训练配置及数据路径。
- 关闭的模块在检查点中仍为未训练的初始权重。最终阶段尚未开始前中断，不会生成本次训练的新模型文件。

默认输出：

```text
outputs/<architecture>/
  model.pth                  # 唯一最终组合最佳模型
  training_history.json      # 各 epoch train/validation loss、学习率和阶段最佳记录
  train_loss_curve.png        # 阶段边界和最佳 epoch 标记
  train_components.mat
  train_case_1.png ... train_case_7.png
  prediction/
    prediction_components.mat
    test_case_1_pred.png ... test_case_5_pred.png
```

损失曲线的 Train Loss 在默认 validation 模式下是该 epoch 的训练批次平均值；验证集 loss 用于最佳选择。切换 `train` 选择时，Train Loss 是更新结束后重新评估的值。

旧版检查点可经 `restore_checkpoint` 映射旧参数名用于预测，但这不表示它们已按新流程训练。已有 `outputs/` 的旧模型、图像不会因修改源码自动更新；需要运行新训练命令。使用其他 `--output-dir` 可以保留旧实验。

训练参数 `--checkpoint` 指定检查点保存位置，不是断点续训入口。最终阶段开始前中断不会生成本次的新检查点；复用输出目录时已有旧文件可能仍在，因此每次实验宜使用独立 `--output-dir`。

### 7.2 曲线中的样本与预测顺序

`train4.py` 训练结束恢复最终阶段最佳模型，按原始 MAT 顺序分批预测完整阶段数据，曲线同时包含内部训练点和验证点。训练批次会打乱，导出曲线不会按训练 DataLoader 顺序拼接。第七训练工况及对应预拉伸测试工况的图会截取后续加载段，完整预测数组仍保留原始样本。

模型做逐样本本构预测：最大方向伸长由输入提供，不使用上一步预测应力递推下一步。全量曲线用于观察拟合；`pred2.py` 加载检查点并预测独立指定的测试 MAT，用于测试评估。标签只用于监督、对照和误差计算，不作为应力预测输入。

<a id="reference"></a>

## 8. 项目结构、验证与常见问题

```text
.
├── README.md                         # 分阶段训练完整指南与共用原理
├── README_Sbar训练.md                # 双轴 S_bar 训练与已有结果
├── train4.py                         # 训练入口
├── pred2.py                          # 预测与评估入口
├── train_sbar.py                     # S_bar 两阶段训练
├── pred_sbar.py                      # S_bar 原数据完整重预测
├── constitutive_pinn/
│   ├── config.py                     # 模块组合配置
│   ├── model.py                      # 单模型前向计算与模块冻结
│   ├── networks.py                   # 自由能、损伤和体积网络
│   ├── physics.py                    # 微球积分、Flory 投影和应力转换
│   ├── sbar_data.py                  # S_bar 数据读取和成对一致性检查
│   ├── data.py                       # MAT 数据读取与一致性检查
│   ├── training.py                   # 顺序训练、损失和学习率调度
│   ├── checkpoint.py                 # 检查点保存、恢复与旧格式兼容
│   ├── inference.py                  # 分批推理与应力分量输出
│   └── plotting.py                   # 损失曲线、工况图与误差指标
├── matlab_data_generation/
│   ├── README.md                     # 数据生成说明
│   ├── generate_all_datasets.m        # 批量生成三套训练 / 测试数据
│   ├── common/                       # 本构数据生成与 Lebedev 积分工具
│   └── generated/                    # 按模块组合组织的 MAT 数据
├── tests/
│   └── test_sequential_training.py    # 回归测试
└── outputs/                          # 模型、训练记录与预测结果
```

### 8.1 回归测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖模块结构、初始化与开关、冻结和最佳权重继承、检查点保存与恢复，以及数据与解析公式的一致性。数据相关测试依赖生成的 MAT 文件。测试用于验证实现行为，不替代完整训练后的收敛与泛化评估。

### 8.2 常见问题

- **找不到 MAT 文件**：先运行 MATLAB 批量生成脚本，或通过数据路径参数指定文件。
- **架构或 `J` 校验失败**：确认使用与阶段匹配的数据。旧版前两种数据可能含体积变化，需要重新生成。
- **预测时找不到模型**：先完成对应配置的训练，或通过 `--checkpoint` 指定已保存模型。
- **CUDA 不可用**：使用 `--device cpu` 运行；若需要 GPU，检查本地 PyTorch 的 CUDA 支持。

合成数据的更多 MATLAB 入口与字段说明见 [MATLAB 数据生成说明](matlab_data_generation/README.md)。双轴 S_bar 数据请使用 [S_bar 训练指南](README_Sbar训练.md) 中的专属入口。
