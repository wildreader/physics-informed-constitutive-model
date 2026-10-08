# 9种不同的变形和损伤耦合系数下的多工况预拉伸和比例双轴工况的损伤Sbar

## 2026-10-08：自由能预热后联合训练

当前默认流程改为：每种材料自由能预热 100 epoch → 恢复最佳预热权重 → 自由能与损伤联合训练 500 epoch → 恢复最佳联合模型 → 完整比例双轴拟合及预拉伸测试。两阶段使用同一比例双轴训练/验证划分和归一化统计量；切换时重建 Adam 和 StepLR，不重新初始化模型，不冻结自由能。体积分支始终关闭。

**没有无损伤标签，因此预热仍使用损伤比例双轴标签，暂时关闭损伤支路，只学习自由能的初始拟合。它不是无损伤自由能标定。相同当前伸长在不同历史下可以有不同应力，纯自由能阶段无法区分这些历史，因此预热曲线可能折中、加载卸载重合；需要第二阶段学习历史效应。预热能否改善最终误差要由对比实验判断，不能保证。**

默认输出改为 `outputs/9_sbar_warmup/<编号>/`，保留旧实验 `outputs/9_sbar`。新增 `warmup/` 子目录保存预热最佳模型、损失、完整曲线及误差；最终 `proportional/` 和 `prestretch_test/` 含义不变。`--predict-only` 只加载最终联合检查点，不重复预热。

以下为当前 CMD 可直接运行的命令（PowerShell 需在完整解释器路径前加 `&`）：

```cmd
C:\programming\anaconda3\envs\torch\python.exe train_9_sbar.py
C:\programming\anaconda3\envs\torch\python.exe train_9_sbar.py --materials 2
C:\programming\anaconda3\envs\torch\python.exe train_9_sbar.py --predict-only
```

### 可调参数与修改位置

优先在命令行传参，无需修改 Python。直接点 VS Code 的 Python 运行按钮且不传参时，使用 `train_9_sbar.py` 中 `parser()` 的 `default=` 值。

| 参数 | 当前默认 | 作用 |
| --- | --- | --- |
| `--warmup-epochs` | 100 | 自由能预热轮数；0 表示旧的直接联合训练 |
| `--damage-epochs` | 500 | 联合训练轮数，虽名为 damage，实际同时更新两支路 |
| `--learning-rate` | 0.001 | 联合学习率，也是未单独指定时的预热学习率 |
| `--warmup-learning-rate` | 跟随 learning-rate | 独立指定预热学习率 |
| `--batch-size` | 256 | 每批样本数 |
| `--lr-step-size` | 200 | 每个阶段各自每隔多少轮衰减学习率 |
| `--lr-gamma` | 0.5 | 每次衰减乘数；1 表示不衰减 |
| `--seed` | 42 | 初始化、数据划分和随机批次种子 |
| `--validation-fraction` | 0.2 | 比例双轴内部验证比例；对比实验宜固定 |
| `--output-dir` | outputs/9_sbar_warmup | 本次实验目录，比较不同设置应使用不同目录 |

例如只针对材料 2 尝试预热 100 轮、联合 1000 轮，预热学习率 0.001、联合学习率 0.0003：

```cmd
C:\programming\anaconda3\envs\torch\python.exe train_9_sbar.py --materials 2 --warmup-epochs 100 --damage-epochs 1000 --warmup-learning-rate 0.001 --learning-rate 0.0003 --output-dir outputs/9_sbar_w100_j1000_lr3e4
```

这是待验证的实验配置，不是已确定更优的参数。可以先保持其余参数相同，仅比较预热 0、50、100 轮，再单独比较联合学习率及轮数。保持验证划分一致，以比例双轴验证损失选配置；反复根据预拉伸测试误差调参后，它就不能再被视为完全独立的最终测试集。

重绘旧直接联合训练结果需明确指定旧目录：

```cmd
C:\programming\anaconda3\envs\torch\python.exe train_9_sbar.py --predict-only --output-dir outputs/9_sbar
```

### 网络层数和宽度

网络也是可调参数，但本次没有修改，且当前没有对应命令行选项。定义位于 `constitutive_pinn/networks.py`：

- `LocalFeatureNetwork`：`1→32→32→6`，自由能/损伤各有独立的一套，两个隐藏层宽度 32。
- `FreeEnergyPredictor` 和 `DamagePredictor`：`12→64→64→1`，两个隐藏层宽度 64；输出分别使用 Softplus 和 Sigmoid。
- `6` 是局部特征维数，`12` 是局部与全局特征拼接后的维数，不宜当作独立隐藏宽度随意修改；池化中的 reshape 也依赖这些维数。

改隐藏宽度需要在这些类的 `nn.Linear` 中同步修改相邻层的输入/输出维度。改隐藏层数还涉及 `model.py` 内损伤末层的初始化索引 `predictor[4]`，必须同步处理。改变结构后旧检查点通常不能直接加载；应使用新输出目录重新训练，并在预测时保持相同结构。这些类也被旧训练入口使用，直接修改会影响所有入口。因此当前先比较训练策略和优化参数，再决定是否需要结构调整。

旧结果检查：材料 2、3、5、6、9 的完整比例双轴相对 L2 约为 10.8%、23.6%、19.3%、23.6%、23.6%，训练拟合本身已存在较大误差。曲线单调或重合不能单独证明某一种原因；本次改动仅检验预热初始化是否有帮助。

本次验证：15 项测试通过，覆盖最佳预热权重继承、预热时损伤/体积分支不变、联合更新和零预热兼容路径。材料 2 的真实数据已完成 1 轮预热 + 1 轮联合训练的流程验证，输出在 `outputs/9_sbar_warmup_smoke`；未执行九种材料的正式预热实验，不能据此判断新策略的最终拟合质量。

## 2026-10-07：原直接联合训练记录（默认值以以上更新为准）

运行入口为 `train_9_sbar.py`。旧 `train_sbar.py --mode damage` 仍表示先获得无损伤自由能再冻结它，不能用于本次仅有损伤数据的流程。

### 已核验的数据

已逐一读取 `data/1` 至 `data/9` 的全部 18 个 MAT 文件，以下结论对九种材料均成立。

| 编号 | 变形耦合系数 | 损伤耦合系数 |
| --- | --- | --- |
| 1 | 0 | 0 |
| 2 | 0.5 | 0 |
| 3 | 1.0 | 0 |
| 4 | 0 | 0.5 |
| 5 | 0.5 | 0.5 |
| 6 | 1.0 | 0.5 |
| 7 | 0 | 1.0 |
| 8 | 0.5 | 1.0 |
| 9 | 1.0 | 1.0 |

系数对应关系沿用本文件原始记录和 `data/info.txt`，系数不作为额外网络输入。每个编号独立初始化、训练、选最佳模型和预测，不跨材料继承参数或归一化统计量。

- **训练数据：`proportional_biaxial_Sbar.mat`，比例为 0、0.25、0.5、0.75、1，每工况 4801 点，共 24005 点。**
- **测试数据：`orthogonal_prestretch_x_cyclic_Sbar.mat`，y 预拉伸峰值为 1.01、4、6、8，每工况保存正式 x 循环的 4801 点，共 19204 点。**
- **两类数据保存的 x 循环均为 `1→4→1→6→1→8→1→10→1`。**
- 预拉伸文件元数据说明 y 先从 1 拉至峰值再回到 1；保存的正式循环中 y=1。预拉伸过程本身的应力样本不在文件中，因此不能从本文件画出该过程的应力曲线。
- **最大方向伸长包含预拉伸历史。**已验证它等于 y 预拉伸峰值状态 `(lambda_x,lambda_y,lambda_z)=(1,p,1/p)` 与初始状态形成的历史种子，再与正式 x 循环逐方向累计最大值合并。不会重新从 1 起算或抹去预拉伸历史。
- 输入直接读取 `Xtrain`：第一通道为当前方向伸长比，第二通道为历史最大方向伸长比；严格说是伸长比，不是工程应变。110 个方向，Python 张量为 `[N,110]`。
- 输出直接使用 `Ytrain.T` 的 S_bar 六分量，顺序沿用 `11,22,33,12,13,23`，不转换为 Cauchy 应力。

### 本次必要的代码调整及不变项

**新入口直接启用第二阶段的 `free_energy_damage` 组合，并同时优化自由能与损伤支路；无需无损伤数据和预训练检查点。体积分支关闭。**原有顺序训练入口保持原语义。

**网络层数、宽度、激活、初始化、微球积分、自由能与损伤耦合公式、固定历史变量求导以及 S_bar 输出定义均未修改。没有增加物理约束、正则项或新的输入特征。**

**默认配置继续沿用当前 S_bar 训练**：**Adam、学习率 0.001、batch size 256、500 epoch、StepLR 每 200 epoch 乘 0.5、随机种子 42、六分量 RMS 归一化 MSE。**

比例双轴数据内部按工况分层划分 80% 训练（19204 点）与 20% 验证（4801 点），以最低验证损失选择最佳 epoch。输入均值/标准差与标签 RMS 仅由该材料的训练子集计算，历史最大伸长沿用相同输入尺度。预拉伸数据不参与训练、归一化或最佳 epoch 选择。比例双轴全部 24005 点均输出拟合曲线，包含验证点；预拉伸全部 19204 点用于泛化测试。

### 运行命令

在项目根目录及已配置 PyTorch 的 Python 环境下运行。当前电脑可先激活 `torch` 环境；如 `python` 指向 Windows 商店别名，可将下列 `python` 替换为 PowerShell 的 `& C:/programming/anaconda3/envs/torch/python.exe`。

**一次依次完成九次独立训练、完整训练数据拟合和测试预测：**

```powershell
python train_9_sbar.py
```

只运行指定材料（例如 1 号），或明确使用 GPU：

```powershell
python train_9_sbar.py --materials 1
python train_9_sbar.py --device cuda
```

只校验全部数据，不训练：

```powershell
python train_9_sbar.py --audit-only
```

训练完成后从各自最佳检查点重新生成训练拟合与测试预测，无需重新训练：

```powershell
python train_9_sbar.py --predict-only
python train_9_sbar.py --predict-only --materials 1
```

### 输出与验证

默认输出到 `outputs/9_sbar/<编号>/`：

- `data_audit.json`：工况、点数、输入通道和 x 路径核验结果。
- `proportional/model.pth`：自由能与损伤联合训练的最佳模型，含归一化统计量及训练/验证索引。
- `proportional/history.json`、`loss.png`：训练/验证损失。
- `proportional/case_*.png`：**五个比例双轴工况的完整六分量拟合曲线，保留原始加载/卸载顺序。**
- `prestretch_test/case_*.png`：**四个预拉伸历史下正式 x 循环的完整六分量预测曲线。**
- **两个结果子目录各含 `full_prediction.mat` 与 `metrics.json`，提供逐点预测、真实值、工况及整体/逐工况误差；测试文件的训练/验证索引为空。**
- 根输出目录 `summary_<本次编号列表>.json` 汇总本次材料结果。

验证：18 份真实数据全部通过核验；14 项单元/回归测试通过，包括联合训练两支路参数更新、体积分支不变、预拉伸历史保留及错误历史拒绝。1 号材料已完成 1 epoch 的真实数据端到端试运行，用于验证训练、保存、绘图和预测流程；试运行输出单独位于 `outputs/9_sbar_smoke`，不代表充分训练后的拟合质量。尚未执行九种材料各 500 epoch 的正式训练。

## 原始材料记录

1.多工况比例双轴

![image-20261007105140925](C:\Users\YizhenZhang\AppData\Roaming\Typora\typora-user-images\image-20261007105140925.png)

2.多工况预拉伸



1   0       4    0    4
2   0.5     4    0    4
3   1.0     4    0    4

4   0       4    0.5   4
5   0.5     4    0.5   4
6   1.0     4    0.5   4

7   0       4    1.0   4
8   0.5     4    1.0   4
9   1.0     4    1.0   4

里面有9个文件夹  对应于9种参数组合，变形耦合系数 0 0.5 1.0  和  损伤耦合系数 0  0.5  1.0

