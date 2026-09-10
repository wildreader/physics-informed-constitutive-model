# 单个三模块本构模型：顺序训练使用说明

## 1. 现在的模型是什么

仅有一个 `ModularConstitutivePINN` 模型，内部始终包含三个模块：

| 模块 | 属性名 | 作用 |
|---|---|---|
| 自由能之路（等体积） | `free_energy_module` | 当前方向伸长 → 方向自由能 |
| 损伤之路（等体积） | `damage_module` | 历史最大方向伸长 → 损伤 d |
| 体积之路 | `volumetric_module` | J → 体积自由能 U_vol → p=dU_vol/dJ |

`--architecture` 指定同一个模型最终开启哪些模块。三种情况的模型参数结构、参数名完全相同；关闭模块仍在模型及检查点内，但不会执行，也不会更新。每次训练命令仅创建一个模型对象，各阶段切换开关、优化器和数据，不重新构造模型。

自由能经微球积分和 Flory 投影得到等体积应力；开启损伤时能量变成 `(1-d)*phi`；开启体积时加入体积应力。最终输出六个 Cauchy 应力分量，顺序为 `11,22,33,12,13,23`。

## 2. 三种训练情况

| 命令参数 | 第 1 阶段 | 第 2 阶段 | 第 3 阶段 | 默认总 epoch |
|---|---|---|---|---:|
| `free_energy` | 只开启、训练自由能 500 epoch | — | — | 500 |
| `free_energy_damage` | 只开启、训练自由能 500 epoch | 继承自由能**最佳权重**，开启损伤，仅训练损伤 500 epoch | — | 1000 |
| `full` | 只开启、训练自由能 500 epoch | 继承自由能**最佳权重**，仅训练损伤 500 epoch | 继承自由能和损伤最佳权重，仅训练体积 500 epoch | 1500 |

**每阶段结束先恢复该阶段最低 loss 对应的权重，再进入下一阶段。已完成模块继续参与前向计算，但参数冻结；后续阶段不能修改其权重和偏置。没有原来的 150 epoch 损伤冻结预热，也没有所有模块联合训练。**

损伤、体积阶段默认也设为 500 epoch，沿用原主训练轮数。可分别通过 `--free-energy-epochs`、`--damage-epochs`、`--volumetric-epochs` 调整。已开启阶段必须至少为 1 epoch，不能跳过尚未训练的前置模块。

## 3. 初始化、超参数及学习率

- 网络结构保持原配置：局部网络 `1-32-32-6`，自由能/损伤预测器 `12-64-64-1`，体积网络 `1-32-32-1`。
- 普通线性层权重采用 Xavier normal，偏置为 0。
- 损伤输出线性层权重、偏置为 0，因此首次开启时 `d=sigmoid(0)=0.5`。自由能阶段完全绕过损伤，不会乘以 0.5。
- 体积输出线性层权重、偏置为 0，因此首次开启时 `U_vol=0`、`p=0`。
- 所有模块仅在构造时初始化一次。阶段切换不重新初始化，自由能、损伤最佳参数直接保留。
- Adam；batch size 256；验证比例 0.2；随机种子 42；按训练集六个应力分量 RMS 归一化的总应力 MSE。每阶段用自身标签重新计算损失尺度。
- **方向伸长输入的均值和标准差取自首阶段数据，**并在后续所有阶段固定，避免改变已学到的自由能函数。

为了保留原来的学习率数值及使用场景，默认对应关系为：

| 阶段 | 初始学习率 |
|---|---:|
| 仅训练自由能的情况 | 0.001（原单阶段主训练） |
| 多模块情况下的自由能第一阶段 | 0.002（原首阶段学习率；不再执行旧预热流程） |
| 损伤阶段 | 0.001 |
| 体积阶段 | 0.001 |

每阶段都有独立 StepLR：`lr(epoch)=lr0 * gamma^floor((epoch-1)/step)`，epoch 从 1 开始。默认 `step=200`、`gamma=0.5`，因此第 1–200、201–400、401–500 轮分别使用 `lr0`、`lr0/2`、`lr0/4`。进入新阶段从该阶段初始学习率开始计数。

可用 `--free-energy-lr`、`--damage-lr`、`--volumetric-lr`、`--scheduler-step` 和 `--scheduler-gamma` 覆盖。例如希望三种情况的自由能阶段都用 0.001，添加 `--free-energy-lr 0.001`。

## 4. 数据必须逐阶段匹配

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

图中的 `Dmg_chain` 实际表示剩余刚度/能量比例，对应网络的 `(1-d)`，并非网络输出的损伤 d。`alpha=0.5` 保持不变。纯自由能分支不计算这些损伤公式，直接 `Fc=force_chain`。

**前两种数据也不调用体积变形函数，**不计算 `k_vol*(J-1)` 或 `J*p*Cinv`，总应力直接取等体积分量。为保持 MAT 接口统一，仍导出值为 0 的压力字段；纯自由能的历史最大伸长字段为全 1 占位，不参与模型计算。`delta_J_ref=1e-4` 作为固定参考尺度保留，只有 full 保存和使用 `k_vol=1e6`。

旧版前两种数据虽然 p=0，**但 J 仍随加载变化；不符合当前严格等体积路径**，必须重新生成。**Python 会检查模块组合、J、字段形状与阶段间方向/权重/参考尺度，避免静默误用。

更多 MATLAB 入口见 [数据生成说明](matlab_data_generation/README.md)。

## 5. 正式训练与预测命令

在装有 PyTorch、NumPy、SciPy、scikit-learn、Matplotlib 的 Python 环境中，从项目根目录运行。每条训练命令都自动完成对应的全部前置阶段，无需先手动训练另一个模型。

```powershell
python train4.py --architecture free_energy
python pred2.py --architecture free_energy

python train4.py --architecture free_energy_damage
python pred2.py --architecture free_energy_damage

python train4.py --architecture full
python pred2.py --architecture full
```

快速检查连接是否正常（仅 3 epoch，不代表模型已训练好）：

```powershell
python train4.py --architecture full --free-energy-epochs 1 --damage-epochs 1 --volumetric-epochs 1 --output-dir outputs/smoke_full
python pred2.py --architecture full --checkpoint outputs/smoke_full/model.pth --output-dir outputs/smoke_full/prediction
```

自定义三个阶段的数据：

```powershell
python train4.py --architecture full --free-energy-data path/to/energy.mat --damage-data path/to/damage.mat --volumetric-data path/to/full.mat
```

`--train-data` 只覆盖最终阶段的数据，不覆盖前置阶段；不能同时与最终阶段对应的专用数据参数使用。自定义数据仍必须具备匹配的元数据。预测的 `--train-data`、`--test-data` 分别指定最终组合对应的训练和测试文件。`--device cpu/cuda/auto` 选择计算设备，`--show` 显示图窗。

## 6. 最优模型保存规则

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

## 7. 代码位置与回归验证

- `constitutive_pinn/model.py`：单模型、三个常驻模块、开启/冻结及前向计算。
- `constitutive_pinn/networks.py`：模块内部网络。
- `constitutive_pinn/training.py`：顺序训练、独立学习率调度、最佳权重恢复。
- `constitutive_pinn/checkpoint.py`：唯一最佳检查点与旧参数名迁移。
- `constitutive_pinn/data.py`：阶段数据匹配与兼容性验证。
- `train4.py` / `pred2.py`：训练 / 预测入口。
- `matlab_data_generation/common/generate_polymer_dataset.m`：三种标签与加载路径。

运行针对性回归测试：

```powershell
python -m unittest discover -s tests -v
```

这些测试用于验证结构、梯度、冻结继承和最佳保存规则；不代替完整 500/1000/1500 epoch 的收敛与精度评估。

#   8.基于力学机理的神经网络架构数据流

![image-20260910184116901](C:\Users\YizhenZhang\AppData\Roaming\Typora\typora-user-images\image-20260910184116901.png)
