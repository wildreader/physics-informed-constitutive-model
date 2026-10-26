# 自由能 → 损伤串联实验（9 个材料）

新增 `damage_input` 配置，与原有 `free_energy` / `free_energy_damage` / `full` 模块组合独立。旧入口默认仍为 `maximum_stretch`；新入口默认 `free_energy`。每个材料独立初始化、划分和训练。

## 数据流与自动微分

旧模式：归一化的 110 个历史最大伸长 → 损伤网络 → 110 个 d。

新模式：归一化的 110 个当前伸长 → 自由能网络 → 110 个 φ → 损伤网络 → 110 个 d。

φ 原样送入原损伤 DeepSets 网络，不复用伸长的均值/标准差，不做 detach，不添加新网络或历史状态。两个模式随后都计算：

`ψ_a = (1-d_a)φ_a → W = Σ_a w_a ψ_a → ∂W/∂λ_a → 微球积分 → S_bar`。

串联模式保留完整链式求导：

`∂W/∂λ_b = Σ_a w_a [(1-d_a)∂φ_a/∂λ_b - φ_a ∂d_a/∂λ_b]`。

因此不能将串联模式的链力简化为 `(1-d) × 弹性链力`。DeepSets 的全局池化也参与上述导数。若损伤输入采用 `φ.detach()`，会得到不同的应力定义，本次没有采用。

`forward_sbar` 仍停在 S_bar，不启用体积支路或 Flory 投影。通用 `forward` 的后续 Flory 投影、体积能求导和 Cauchy 转换保持不变；没有改成截图右侧的解析体积公式。

## 沿用的配置

已从 `outputs/9_sbar_warmup_d006_plateau_wide/1/proportional/model.pth` 的 options 和权重形状核对：

| 项目 | 新入口默认值 |
| --- | --- |
| 训练 / 测试 | 双轴训练及验证 / 预拉伸独立测试 |
| 材料 | 1–9，每个单独训练 |
| 自由能预热 / 联合训练 | 100 / 1000 epochs |
| 初始损伤 | 0.06 |
| 学习率 / batch | 0.001 / 256 |
| 联合调度器 | ReduceLROnPlateau，监测验证 loss |
| patience / factor / threshold / min_lr | 40 / 0.5 / 1e-4（相对）/ 1e-5 |
| **预热调度器** | StepLR，step_size=200，gamma=0.5 |
| seed / validation_fraction | 42 / 0.2，按 case 分层随机划分 |
| 局部特征网络 | 1 → 64 → 64 → 6 |
| 自由能 / 损伤预测器 | 12 → 128 → 128 → 1 |

输出仍为 `warmup/`、`proportional/`、`prestretch_test/` 和各材料汇总。默认新目录为 `outputs/9_sbar_serial_d006_plateau_wide`。

## 训练与预测入口

在项目根目录运行，以下为 PowerShell 命令：

```powershell
# 审核全部 9 个材料的数据，不训练
& 'C:/programming/anaconda3/envs/torch/python.exe' train_9_sbar_serial.py --audit-only

# 全部 9 个材料：双轴训练，预拉伸测试（训练结束自动导出两者）
& 'C:/programming/anaconda3/envs/torch/python.exe' train_9_sbar_serial.py

# 可先单独训练一个材料
& 'C:/programming/anaconda3/envs/torch/python.exe' train_9_sbar_serial.py --materials 9

# 从各材料最佳 checkpoint 重新导出双轴和预拉伸预测，不训练
& 'C:/programming/anaconda3/envs/torch/python.exe' pred_9_sbar_serial.py

# 只预测已完成的材料
& 'C:/programming/anaconda3/envs/torch/python.exe' pred_9_sbar_serial.py --materials 9

# 旧架构对照，除损伤输入与输出目录外保持同一配置
& 'C:/programming/anaconda3/envs/torch/python.exe' train_9_sbar_serial.py --damage-input maximum_stretch --output-dir outputs/9_sbar_history_control
```

也可使用通用入口 `train_9_sbar.py --damage-input free_energy --training-data proportional --initial-damage 0.06`。单数据集训练入口 `train_sbar.py` 同样支持 `--damage-input`。

如要作混合训练对照，传入 `--training-data combined` 和独立 `--output-dir`；预测时也要传入相同两项。这时预拉伸输出是 `prestretch_fit/`，不应作为独立测试性能。

新 checkpoint 保存顶层 `damage_input`，预测自动使用该元数据；缺少此字段的旧 checkpoint 按 `maximum_stretch` 加载。两模式 state_dict 键及形状相同，但直接搬运旧权重并切换输入不等价于训练串联模型。`pred_sbar.py` 也读取该元数据，但仅支持原始单数据集回放；九材料混合训练的 checkpoint 应通过九材料预测入口回放。

## 初始损伤建议与结果解释

建议第一轮保持 **0.06**，仅改变损伤输入，从而能与已有实验比较。末层权重初始化为零、bias 为 `logit(0.06) ≈ -2.7515`，所以初始 d 恒为 0.06，初始 `∂d/∂φ=0`。从预热切换到联合训练时应力为预热值的约 94%。这是优化初值，并非对材料物理初始损伤的标定；尚无实验依据称它是最佳值。

混合训练能拟合预拉伸，说明已有模型对这些已见路径具有拟合能力，并不能单独证明“必须加入预拉伸训练”或者新串联模式一定提高泛化。新模式只依赖当前变形，没有历史输入；相同当前伸长向量无论经历何种加载历史都会输出相同应力。若循环数据在相同变形处具有不同损伤，应力就不能被这个无记忆结构同时精确表达。它也不强制 d 随历史单调增长。

若后续证实历史信息不可缺少，可另做“历史能量驱动”的实验；那需要定义历史能量如何累计，不能将本次的当前 φ 串联与历史能量模型混为一谈。本次严格实现当前自由能直接连接损伤网络。

## 验证范围

23 项测试全部通过，覆盖 110 个原始自由能传递、完整链式求导、双支路梯度、0.06 初始化、旧参数兼容与 checkpoint 模式恢复。全部 9 个材料通过数据审核。材料 1 的真实数据短流程（预热 1 轮、联合 1 轮）及独立预测回放成功，回放的双轴和预拉伸误差指标与训练结束导出一致；验证输出位于 `.verification/serial_smoke`。完整 100+1000 轮实验尚未执行，因此目前没有新架构泛化改善的结论。
