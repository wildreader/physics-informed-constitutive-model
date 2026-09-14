# 聚合物三模块本构 PINN

基于 **PyTorch** 的聚合物本构建模项目，将自由能、损伤与体积响应组织在一个神经网络模型中，结合微球积分、自动微分和 Flory 等体积–体积分解，预测三维有限变形下的 Cauchy 应力。

项目提供 MATLAB 合成数据生成、Python 顺序训练、五工况预测评估和结果可视化的完整流程，适用于模块化本构建模与数值实验。

## 项目特点

- **单模型、三模块**：使用 `ModularConstitutivePINN`，通过配置选择参与计算的模块。
- **顺序训练**：按自由能 → 损伤 → 体积的顺序学习；每阶段继承前一阶段最佳权重，仅优化新加入的模块。
- **物理计算嵌入前向过程**：由能量自动微分得到链力和压力，再通过微球积分与应力转换得到宏观应力。
- **阶段匹配数据**：三种模块组合分别使用对应的合成标签，提供七工况训练集和五工况测试集。
- **分量输出与评估**：导出总应力、等体积应力、体积应力、压力，以及测试集 MSE 和逐分量 RMSE。
- **CPU / CUDA 支持**：通过命令行选择计算设备。

## 模型与训练方式

| 模块 | 输入与作用 |
| --- | --- |
| 自由能模块 `free_energy_module` | 根据当前方向伸长计算方向自由能 |
| 损伤模块 `damage_module` | 根据历史最大方向伸长计算损伤变量 `d`，以 `(1-d)` 调制方向自由能 |
| 体积模块 `volumetric_module` | 根据体积比 `J` 计算体积自由能，经自动微分得到压力 `p = dU_vol/dJ` |

方向能量经过加权微球积分、Flory 投影和第二类 Piola–Kirchhoff 应力到 Cauchy 应力的转换，得到六个应力分量，排列为：

```text
11, 22, 33, 12, 13, 23
```

`--architecture` 指定最终启用的模块组合：

| 配置 | 训练顺序 | 默认总 epoch |
| --- | --- | ---: |
| `free_energy` | 自由能 | 500 |
| `free_energy_damage` | 自由能 → 损伤 | 1000 |
| `full`（默认） | 自由能 → 损伤 → 体积 | 1500 |

三种配置共享相同的模型参数结构。每个训练命令都会完成所需的全部前置阶段，无需先运行另一种配置。阶段切换时恢复该阶段最佳权重并冻结已完成模块，后续阶段仍使用这些模块进行前向计算。

## 项目结构

```text
.
├── README.md                         # 项目概览与快速开始
├── README_单模型顺序训练.md            # 训练机制与参数详细说明
├── train4.py                         # 训练入口
├── pred2.py                          # 预测与评估入口
├── constitutive_pinn/
│   ├── config.py                     # 模块组合配置
│   ├── model.py                      # 单模型前向计算与模块冻结
│   ├── networks.py                   # 自由能、损伤和体积网络
│   ├── physics.py                    # 微球积分、Flory 投影和应力转换
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

## 快速开始

以下命令均从项目根目录执行。

### 1. 准备 Python 环境

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

### 2. 准备数据

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

### 3. 训练模型

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

### 4. 预测与评估

```bash
python pred2.py --architecture full
```

预测配置必须与检查点匹配。评估过程会读取该配置对应的训练 MAT 文件、测试 MAT 文件和检查点；训练文件用于提供积分方向、权重等信息。程序输出测试集总应力 MSE、六个分量的 RMSE，并保存预测数据与五工况对比图。

其他配置使用对应命令：

```bash
python pred2.py --architecture free_energy
python pred2.py --architecture free_energy_damage
```

### 5. 快速检查完整流程

每阶段仅运行 1 个 epoch，并保存到独立目录：

```bash
python train4.py --architecture full --free-energy-epochs 1 --damage-epochs 1 --volumetric-epochs 1 --output-dir outputs/smoke_full
python pred2.py --architecture full --checkpoint outputs/smoke_full/model.pth --output-dir outputs/smoke_full/prediction
```

该示例用于检查数据、训练与预测流程是否连通，不用于判断模型精度。

## 常用参数

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

训练使用 Adam 优化器，验证比例为 `0.2`，损失为按训练子集各应力分量 RMS 归一化的总应力 MSE。每阶段重新计算损失尺度；方向伸长的输入归一化统计量取自首阶段数据，并在后续阶段固定。

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

## 输出文件与最佳模型

```text
outputs/<architecture>/
├── model.pth                         # 最终阶段最佳模型
├── training_history.json             # 各阶段 loss、学习率与最佳 epoch
├── train_loss_curve.png              # 训练 / 验证损失曲线
├── train_components.mat              # 训练集应力分量
├── train_case_1.png ... train_case_7.png
└── prediction/                       # 运行 pred2.py 后生成
    ├── prediction_components.mat     # 预测应力、压力、真实应力与误差指标
    └── test_case_1_pred.png ... test_case_5_pred.png
```

默认按每阶段验证集最低损失选择最佳权重，各阶段之间不比较损失。前置阶段最佳状态保留在内存中用于后续训练；最终阶段损失改善时覆盖 `model.pth`。训练结束会恢复最终阶段最佳权重，再生成训练结果图。

检查点包含三个模块参数、模块开关、归一化信息和训练元数据。未启用模块仍保留初始化参数。`--checkpoint` 在训练脚本中用于指定保存位置，不是断点续训入口。

如需保留多次实验，请为每次运行指定不同的 `--output-dir`。仓库中已有的模型或图片不会随源码更新自动重新生成。

## 回归测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖模块结构、初始化与开关、冻结和最佳权重继承、检查点保存与恢复，以及数据与解析公式的一致性。数据相关测试依赖生成的 MAT 文件。测试用于验证实现行为，不替代完整训练后的收敛与泛化评估。

## 常见问题

- **找不到 MAT 文件**：先运行 MATLAB 批量生成脚本，或通过数据路径参数指定文件。
- **架构或 `J` 校验失败**：确认使用与阶段匹配的数据。旧版前两种数据可能含体积变化，需要重新生成。
- **预测时找不到模型**：先完成对应配置的训练，或通过 `--checkpoint` 指定已保存模型。
- **CUDA 不可用**：使用 `--device cpu` 运行；若需要 GPU，检查本地 PyTorch 的 CUDA 支持。

## 更多说明

- [单模型顺序训练详细说明](README_单模型顺序训练.md)：初始化、阶段继承、学习率与最佳模型保存规则。
- [MATLAB 数据生成说明](matlab_data_generation/README.md)：合成数据模式、公式与生成接口。

## 直接拟合双轴 S_bar 数据

本目录的 `proportional_biaxial_Sbar*.mat` 使用 `Xtrain/Ytrain`，应通过新增的 `train_sbar.py` 训练、`pred_sbar.py` 复现完整预测。物理层停在未投影 S_bar。两种模式、已核实的数据维度和运行命令见 [S_bar 训练说明](README_Sbar训练.md)。原 `train4.py` / `pred2.py` 继续用于 Cauchy 应力数据。
