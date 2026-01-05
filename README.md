# TimeMixer++：用于二分类

一个模块化的 PyTorch 实现，用于时间序列二分类（事故概率预测）的 TimeMixer++ 架构。

## 架构概览

模型实现了 TimeMixer++ 论文中的核心组件：

```
输入 (B, 48) → 多尺度生成 → [MixerBlock × L] → 输出头 → 概率
                      ↓
              {x_0, x_1, ..., x_M}
                      ↓
              ┌─────────────┐
              │ MixerBlock  │
              │ ┌─────────┐ │
              │ │  MRTI   │ │ → 基于 FFT 的周期检测，1D→2D 重塑
              │ ├─────────┤ │
              │ │  TID    │ │ → 双轴注意力（季节性 + 趋势）
              │ ├─────────┤ │
              │ │  MCM    │ │ → 自底向上 + 自顶向下 混合
              │ ├─────────┤ │
              │ │  MRM    │ │ → 幅值加权聚合
              │ └─────────┘ │
              └─────────────┘
```

### 关键组件

1. **MRTI（Multi-Resolution Time Imaging，多分辨率时间成像）**：基于 FFT 检测到的周期，将 1D 时间序列转换为 2D“时间图像”
2. **TID（Time Image Decomposition，时间图像分解）**：通过双轴注意力分离季节性（列）与趋势（行）模式
3. **MCM（Multi-Scale Mixing，多尺度混合）**：跨尺度的自底向上季节性混合 + 自顶向下趋势混合
4. **MRM（Multi-Resolution Mixing，多分辨率混合）**：使用 FFT 幅值权重在不同周期之间进行聚合

## 安装

```bash
# 克隆仓库
cd TimeMixer

# 安装依赖
pip install torch numpy pandas openpyxl
```

**要求**：

- Python 3.8+
- PyTorch 1.12+
- NumPy
- Pandas（用于数据加载）
- openpyxl（用于 Excel 文件支持）

## 快速开始

### 使用随机数据训练（测试运行）

```bash
python scripts/train.py --use_random_data --epochs 2 --batch_size 16
```

### 使用真实数据训练

```bash
# CSV 格式（0-47 列为特征，48 列为标签）
python scripts/train.py --data_path TDdata/TrainData.csv --epochs 50

# Excel 格式（Sheet3，4-51 列为特征，52 列为标签）
python scripts/train.py --data_path TDdata/alldata.xlsx --epochs 50
```

### 推理

```bash
python scripts/infer.py --checkpoint checkpoints/best_model.pt --input data.csv --output predictions.csv
```

### 继续训练（从检查点恢复）

如果训练中断或需要在已有模型基础上继续训练，可以使用 `--resume` 参数：

```bash
# 从保存的检查点继续训练
python scripts/train.py --data_path TDdata/TrainData.csv --resume checkpoints/best_model.pt --epochs 100

# 继续训练并指定新的学习率
python scripts/train.py --data_path TDdata/TrainData.csv --resume checkpoints/final_model.pt --epochs 100 --lr 1e-4
```

**说明**：

- `--resume` 会自动加载模型权重、优化器状态和训练进度
- 模型配置（`d_model`、`n_layers` 等）会从检查点中恢复，无需重新指定
- `--epochs` 为训练的总轮数，会从上次中断的位置继续

### 在测试集上评估模型

使用 `test.py` 脚本在带标签的测试集上评估模型性能：

```bash
# 基本用法：在测试集上评估并打印指标
python scripts/test.py --checkpoint checkpoints/best_model.pt --test_path TDdata/TestData.csv

# 保存预测结果到文件
python scripts/test.py --checkpoint checkpoints/best_model.pt --test_path TDdata/TestData.csv --output test_predictions.csv

# 同时保存多尺度特征
python scripts/test.py --checkpoint checkpoints/best_model.pt --test_path TDdata/TestData.csv --output_features --features_output test_features.npz

# 使用不同的分类阈值
python scripts/test.py --checkpoint checkpoints/best_model.pt --test_path TDdata/TestData.csv --threshold 0.3
```

**输出指标**：

- Accuracy（准确率）
- Precision（精确率）
- Recall（召回率）
- F1 Score（F1 分数）
- AUROC（ROC 曲线下面积）
- **误报率 FPR** = FP / (FP + TN)：实际为负类但被预测为正类的比例
- **漏报率 FNR** = FN / (TP + FN)：实际为正类但被预测为负类的比例（= 1 - Recall）
- 混淆矩阵（TP、FP、TN、FN）

**关于阈值处理**：

- 模型输出为 0-1 之间的概率值
- 标签可以是 0-1 之间的小数（概率值）
- 计算分类指标时，预测和标签都会通过阈值转换为 0/1：
  - `y_pred >= threshold` → 1（预测为正类）
  - `y_true >= label_threshold` → 1（实际为正类）

## 输入/输出格式

### 输入

- **形状**：`(B, 48)` 或 `(B, 48, 1)` - 温度时间序列
- **CSV 格式**：无表头，0-47 列为特征，48 列为标签（0/1）
- **Excel 格式**：Sheet3，4-51 列为特征，52 列为标签

### 输出

- **Logits**：`(B, 1)` - 模型原始输出（用于使用 `BCEWithLogitsLoss` 训练）
- **概率**：`(B, 1)` - `sigmoid(logits)`，取值范围 [0, 1]
- **特征**：用于多尺度表示的 `M+1` 个张量列表

## 关键超参数

| 参数            | 默认值 | 说明                         |
| --------------- | ------ | ---------------------------- |
| `d_model`     | 64     | 隐藏维度                     |
| `n_layers`    | 2      | MixerBlock 数量              |
| `n_heads`     | 4      | 注意力头数                   |
| `top_k`       | 3      | 选取的 FFT 频率数量（Top-K） |
| `min_fft_len` | 8      | FFT 最小长度（决定 M）       |
| `dropout`     | 0.1    | Dropout 比例                 |
| `pos_weight`  | None   | 类别不平衡时的正类权重       |

### 动态 M（尺度）

对于 T=48 这样的短序列，我们动态计算尺度数 M：

- 选择 M 以保证最粗尺度至少有 `min_fft_len` 个点
- 公式：`M = min(max_scales_upper_bound, floor(log2(T / min_fft_len)))`
- 示例：T=48，min_fft_len=8 → M=2，尺度：[48, 24, 12]

## 单元测试

### 测试文件说明

`tests/test_shapes.py` 是模型的单元测试文件，用于验证各模块的正确性。包含以下测试类：

| 测试类             | 测试内容                                                  |
| ------------------ | --------------------------------------------------------- |
| `TestConfig`     | 验证动态 M 计算、尺度长度计算是否正确                     |
| `TestLayers`     | 测试基础层（MHSA、Conv1d、Conv2d、match_shape）的输出形状 |
| `TestMRTI`       | 测试 FFT 周期检测、周期去重、1D↔2D 重塑的数据一致性      |
| `TestTID`        | 验证双轴注意力（季节性/趋势分解）保持输入形状             |
| `TestMCM`        | 测试多尺度混合后序列长度是否正确恢复                      |
| `TestMRM`        | 验证多分辨率混合的全局/逐样本权重模式                     |
| `TestMixerBlock` | 测试 MixerBlock 残差连接后形状一致性                      |
| `TestFullModel`  | 完整模型前向/反向传播、特征提取测试                       |
| `TestEdgeCases`  | 边界条件测试（batch=1、K 截断、不同 top_k 值）            |

### 运行测试

```bash
# 运行全部测试
python -m pytest tests/ -v

# 运行指定测试类
python -m pytest tests/test_shapes.py::TestFullModel -v

# 运行单个测试方法
python -m pytest tests/test_shapes.py::TestMRTI::test_reshape_1d_to_2d_and_back -v

# 查看详细输出
python -m pytest tests/ -v --tb=short
```

### 测试覆盖的关键验证点

1. **形状一致性**：确保各模块输入输出形状正确
2. **数据保持**：1D→2D→1D 变换后数据无损
3. **动态参数**：短序列下 M 和 K 的自动截断
4. **梯度流动**：反向传播梯度正常计算
5. **边界条件**：极端参数下模型仍能正常运行

## 项目结构

```
TimeMixer/
├── src/timemixerpp/
│   ├── __init__.py      # 包导出
│   ├── config.py        # TimeMixerPPConfig 数据类
│   ├── layers.py        # MHSA, ConvDown, ConvUp, match_shape
│   ├── mrti.py          # 多分辨率时间成像（MRTI）
│   ├── tid.py           # 时间图像分解（TID）
│   ├── mcm.py           # 多尺度混合（MCM）
│   ├── mrm.py           # 多分辨率混合（MRM）
│   ├── block.py         # MixerBlock
│   ├── model.py         # 编码器 + 分类头
│   ├── data.py          # Dataset 与 DataLoader 工具
│   └── utils.py         # 随机种子、指标、checkpoint
├── scripts/
│   ├── train.py            # 训练脚本（支持 --resume 继续训练）
│   ├── test.py             # 测试脚本（在测试集上评估）
│   ├── infer.py            # 推理脚本（无标签预测）
│   ├── inspect_shapes.py   # 查看中间张量形状
│   ├── extract_features.py # 提取多尺度特征
│   ├── baseline_comparison.py # 基线模型对比
│   └── ablation_study.py   # 消融实验
├── tests/
│   └── test_shapes.py   # 单元测试
├── checkpoints/         # 保存的模型
└── README.md
```

## 查看中间形状

使用 `inspect_shapes.py` 脚本可以查看模型各阶段的张量形状：

```bash
# 使用默认配置
python scripts/inspect_shapes.py

# 自定义参数
python scripts/inspect_shapes.py --batch_size 4 --d_model 64 --top_k 3

# 从检查点加载配置
python scripts/inspect_shapes.py --checkpoint checkpoints/best_model.pt
```

### 输出示例

```
======================================================================
 TimeMixer++ 中间形状检查
======================================================================

配置参数:
  batch_size (B) = 2
  seq_len (T) = 48
  d_model = 64
  n_layers = 2
  top_k (K) = 3
  动态尺度数 M = 2
  各尺度长度 = [48, 24, 12]

----------------------------------------------------------------------
 3. 多尺度生成 (Multi-Scale Generation)
----------------------------------------------------------------------
  生成 M+1 = 3 个尺度:
    x_0 (尺度 0, L_0=48): (2 × 48 × 64)
    x_1 (尺度 1, L_1=24): (2 × 24 × 64)
    x_2 (尺度 2, L_2=12): (2 × 12 × 64)

----------------------------------------------------------------------
 4. MRTI (多分辨率时间成像)
----------------------------------------------------------------------
  检测到的周期 (K_eff=3): [6, 4, 3]
  
  周期 k=0, period=6:
      z_0^(0): (B=2, d=64, H=6, W=8)
      z_1^(0): (B=2, d=64, H=6, W=4)
      z_2^(0): (B=2, d=64, H=6, W=2)

----------------------------------------------------------------------
 5. TID (时间图像分解)
----------------------------------------------------------------------
  周期 k=0 的分解结果:
      尺度 0:
        季节性 s_0^(0): (2, 64, 6, 8)
        趋势   t_0^(0): (2, 64, 6, 8)

----------------------------------------------------------------------
 7. MRM (多分辨率混合)
----------------------------------------------------------------------
  聚合后各尺度输出:
    x_0^{out}: (2 × 48 × 64)
    x_1^{out}: (2 × 24 × 64)
    x_2^{out}: (2 × 12 × 64)
```

### 形状变化总结表

| 阶段 | 输入形状 | 输出形状 | 说明 |
|------|----------|----------|------|
| 输入 | (B, T) | (B, T, 1) | 增加通道维度 |
| 投影 | (B, T, 1) | (B, T, d) | Linear: 1→d_model |
| 多尺度 | (B, T, d) | [(B, L_m, d)]×(M+1) | Conv1d stride=2 |
| MRTI | (B, L_m, d) | (B, d, H, W) | 1D→2D, H=period |
| TID | (B, d, H, W) | s,t: (B, d, H, W) | 双轴注意力，形状不变 |
| MCM | s,t: (B, d, H, W) | (B, L_m, d) | 2D→1D 还原 |
| MRM | [(B, L_m, d)]×K | (B, L_m, d) | 跨周期加权聚合 |
| 输出头 | [(B, L_m, d)]×(M+1) | (B, 1) | 池化+多尺度集成 |

其中：
- `B` = batch_size
- `T` = seq_len = 48
- `d` = d_model = 64
- `M` = 尺度数（动态计算）
- `L_m` = T / 2^m（第 m 个尺度的长度）
- `H` = period（周期长度）
- `W` = ceil(L_m / period)（时间图像宽度）
- `K` = 周期数量

## 技术细节

### 周期计算与去重

对于短序列，FFT 的频率分辨率有限。我们通过以下方式处理：

1. **在最粗尺度上做 FFT**：对 `x_M` 计算 FFT，选取 Top-K 频率
2. **周期计算**：`p_k = clamp(round(L_M / f_k), min_period, L_0)`
3. **去重**：若多个频率映射到同一周期，则保留幅值更大的那个
4. **结果**：得到 `K_eff` 个唯一周期（`K_eff ≤ K`）

### TID 双轴注意力

关键点在于通过 reshape 将非目标轴合并到 batch 维度中：

```python
# 列注意力（季节性）：沿 W 维做注意力
# (B, d, H, W) → (B*H, W, d) → MHSA → (B, d, H, W)

# 行注意力（趋势）：沿 H 维做注意力
# (B, d, H, W) → (B*W, H, d) → MHSA → (B, d, H, W)
```

这样可以使用标准 MHSA 实现进行高效计算。

### MCM 步幅约定

2D 卷积使用 `stride=(1, 2)`：

- H 维（行 = 周期）保持不变
- W 维（列 = 时间跨度）随尺度变化
- 这与论文对时间步幅（temporal stride）的描述一致

## 训练建议

1. **类别不平衡**：使用 `--pos_weight` 对正样本加权
2. **短序列**：动态 M 使得即使在 T=48 时也能得到有意义的 FFT
3. **可复现性**：设置 `--seed` 以获得一致结果
4. **早停**：默认基于 F1 分数，patience=10 个 epoch
5. **继续训练**：使用 `--resume` 从检查点继续训练，避免从头开始

## 完整命令行参数

### train.py 参数

| 参数                  | 类型  | 默认值      | 说明             |
| --------------------- | ----- | ----------- | ---------------- |
| `--data_path`       | str   | None        | 训练数据路径     |
| `--use_random_data` | flag  | -           | 使用随机数据测试 |
| `--resume`          | str   | None        | 从检查点继续训练 |
| `--epochs`          | int   | 50          | 训练轮数         |
| `--batch_size`      | int   | 32          | 批大小           |
| `--lr`              | float | 1e-3        | 学习率           |
| `--d_model`         | int   | 64          | 隐藏维度         |
| `--n_layers`        | int   | 2           | MixerBlock 层数  |
| `--save_dir`        | str   | checkpoints | 模型保存目录     |

### test.py 参数

| 参数                  | 类型  | 默认值         | 说明                                           |
| --------------------- | ----- | -------------- | ---------------------------------------------- |
| `--checkpoint`      | str   | **必需** | 模型检查点路径                                 |
| `--test_path`       | str   | **必需** | 测试数据路径                                   |
| `--output`          | str   | None           | 预测结果保存路径                               |
| `--threshold`       | float | 0.5            | 预测分类阈值                                   |
| `--label_threshold` | float | None           | 标签分类阈值（默认与 threshold 相同）          |
| `--output_features` | flag  | -              | 是否输出特征                                   |

### infer.py 参数

| 参数             | 类型  | 默认值          | 说明             |
| ---------------- | ----- | --------------- | ---------------- |
| `--checkpoint` | str   | **必需**  | 模型检查点路径   |
| `--input`      | str   | None            | 输入数据路径     |
| `--output`     | str   | predictions.csv | 预测结果保存路径 |
| `--threshold`  | float | 0.5             | 分类阈值         |

## API 用法

```python
from timemixerpp import TimeMixerPPConfig, TimeMixerPPForBinaryCls

# 创建模型
config = TimeMixerPPConfig(
    seq_len=48,
    d_model=64,
    n_layers=2,
    top_k=3
)
model = TimeMixerPPForBinaryCls(config)

# 前向计算
x = torch.randn(32, 48)  # (batch, seq_len)
output = model(x)
# output['logits']: (32, 1)
# output['probs']: (32, 1)

# 获取多尺度特征
features = model.get_multi_scale_features(x)
# features：包含 M+1 个张量的列表
```

## 完整工作流示例

以下是一个完整的训练、继续训练、测试的工作流示例：

```bash
# 1. 首次训练（50 个 epoch）
python scripts/train.py --data_path TDdata/TrainData.csv --epochs 50 --save_dir checkpoints

# 2. 查看训练结果后，继续训练（从 epoch 50 继续到 100）
python scripts/train.py --data_path TDdata/TrainData.csv --resume checkpoints/best_model.pt --epochs 100

# 3. 在测试集上评估最佳模型
python scripts/test.py --checkpoint checkpoints/best_model.pt --test_path TDdata/TestData.csv

# 使用相同阈值（默认 0.5）
python scripts/test.py --checkpoint checkpoints/best_model.pt --test_path TDdata/alldata.xlsx

# 使用不同阈值（预测用 0.3，标签用 0.5）
python scripts/test.py --checkpoint checkpoints/best_model.pt --test_path TDdata/alldata.xlsx --threshold 0.3 --label_threshold 0.5

# 标签阈值默认与预测阈值相同
python scripts/test.py --checkpoint checkpoints/best_model.pt --test_path TDdata/alldata.xlsx --threshold 0.4

# 4. 保存测试集预测结果
python scripts/test.py --checkpoint checkpoints/best_model.pt --test_path TDdata/TestData.csv --output results/test_predictions.csv

# 5. 对新数据进行推理（无标签）
python scripts/infer.py --checkpoint checkpoints/best_model.pt --input new_data.csv --output results/predictions.csv
```

### 检查点文件内容

保存的 `.pt` 文件包含以下内容：

| 键名                     | 说明                                 |
| ------------------------ | ------------------------------------ |
| `model_state_dict`     | 模型权重                             |
| `optimizer_state_dict` | 优化器状态                           |
| `epoch`                | 保存时的 epoch 数                    |
| `metrics`              | 验证集指标（accuracy, f1, auroc 等） |
| `config`               | 模型配置参数                         |
| `normalizer_mean`      | 数据归一化均值                       |
| `normalizer_std`       | 数据归一化标准差                     |

## 提取多尺度特征

使用 `extract_features.py` 提取经过所有 MixerBlock 后、输出头之前的多尺度特征：

```bash
# 提取特征并保存
python scripts/extract_features.py --checkpoint checkpoints/best_model.pt --data_path TDdata/TrainData.csv --output features/train_features.npz --save_labels

# 查看已保存的特征
python scripts/extract_features.py --view features/train_features.npz
```

### 输出示例

```
==============================================================
 Multi-Scale Features Summary
==============================================================

Keys in file: ['scale_0', 'scale_1', 'scale_2', 'labels', 'config']

Number of scales: 3
------------------------------------------------------------

scale_0:
  Shape: (1000, 48, 64)
  Dtype: float32
  Min:   -3.245612
  Max:   4.123456
  Mean:  0.001234
  Std:   0.987654

scale_1:
  Shape: (1000, 24, 64)
  ...

scale_2:
  Shape: (1000, 12, 64)
  ...
```

### 在代码中使用提取的特征

```python
import numpy as np

# 加载特征
data = np.load('features/train_features.npz')

# 获取各尺度特征
scale_0 = data['scale_0']  # (n_samples, 48, d_model)
scale_1 = data['scale_1']  # (n_samples, 24, d_model)
scale_2 = data['scale_2']  # (n_samples, 12, d_model)

# 获取标签（如果保存了）
labels = data['labels']  # (n_samples,)
```

## 基线模型对比

使用 `baseline_comparison.py` 与其他时序分类模型进行对比：

```bash
# 运行所有基线模型对比（默认从训练集分出30%作为测试集）
python scripts/baseline_comparison.py --data_path TDdata/TrainData.csv --epochs 50

# 指定独立测试集
python scripts/baseline_comparison.py --data_path TDdata/TrainData.csv --test_path TDdata/TestData.csv --epochs 50

# 使用全部训练数据作为测试集（test_split=0）
python scripts/baseline_comparison.py --data_path TDdata/TrainData.csv --test_split 0 --epochs 50

# 自定义测试集划分比例（例如20%）
python scripts/baseline_comparison.py --data_path TDdata/TrainData.csv --test_split 0.2 --epochs 50

# 只对比特定模型
python scripts/baseline_comparison.py --data_path TDdata/TrainData.csv --models lstm bilstm transformer

# 包含 TimeMixer++ 一起对比
python scripts/baseline_comparison.py --data_path TDdata/TrainData.csv --include_timemixer
```

### baseline_comparison.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_path` | str | **必需** | 训练数据路径 |
| `--test_path` | str | None | 测试数据路径（可选） |
| `--test_split` | float | 0.3 | 测试集划分比例（若未指定 test_path） |
| `--models` | list | all | 要对比的模型列表 |
| `--include_timemixer` | flag | - | 是否包含 TimeMixer++ |
| `--epochs` | int | 50 | 训练轮数 |
| `--batch_size` | int | 32 | 批大小 |
| `--lr` | float | 1e-3 | 学习率 |
| `--val_split` | float | 0.2 | 验证集划分比例 |

**测试集逻辑**：
- 若指定 `--test_path`：使用独立测试集
- 若 `--test_split > 0`：从训练数据中划分指定比例作为测试集
- 若 `--test_split = 0`：使用全部训练数据作为测试集（训练完后在同一数据上测试）

### 可用的基线模型

| 模型名称 | 描述 |
|----------|------|
| `lstm` | LSTM 分类器 |
| `bilstm` | 双向 LSTM 分类器 |
| `lstm_transformer` | LSTM + Transformer 混合模型 |
| `cnn_bilstm` | CNN + BiLSTM 混合模型 |
| `transformer` | 纯 Transformer 分类器 |
| `mlp` | 多层感知机 |
| `gru` | GRU 分类器 |

### 添加自定义模型

```python
from scripts.baseline_comparison import register_model

class MyModel(nn.Module):
    def __init__(self, seq_len=48, hidden_dim=64, **kwargs):
        super().__init__()
        # ... 定义模型结构
    
    def forward(self, x):
        # ... 前向传播
        return {'logits': logits, 'probs': torch.sigmoid(logits)}

# 注册模型
register_model(
    'my_model',
    MyModel,
    {'hidden_dim': 64, 'dropout': 0.1},
    'My custom model description'
)
```

### 输出示例

```
==========================================================================================
 Test Set Results
==========================================================================================
Model                    Params      Acc     Prec   Recall       F1    AUROC      FPR      FNR
------------------------------------------------------------------------------------------
timemixer++              52,481   0.8520   0.8456   0.8567   0.8511   0.9123   0.1234   0.1433
transformer              45,123   0.8234   0.8123   0.8234   0.8178   0.8901   0.1456   0.1766
lstm_transformer         38,567   0.8156   0.8045   0.8167   0.8106   0.8756   0.1567   0.1833
cnn_bilstm               34,234   0.8089   0.7956   0.8078   0.8017   0.8678   0.1678   0.1922
bilstm                   28,456   0.7945   0.7823   0.7934   0.7878   0.8534   0.1789   0.2066
lstm                     24,123   0.7834   0.7712   0.7823   0.7767   0.8423   0.1890   0.2177
==========================================================================================

指标说明:
  FPR (误报率) = FP / (FP + TN) - 实际为负类但被预测为正类的比例
  FNR (漏报率) = FN / (TP + FN) - 实际为正类但被预测为负类的比例 (= 1 - Recall)
```

## 消融实验

使用 `ablation_study.py` 分析各组件的贡献：

```bash
# 运行所有消融实验（默认从训练集分出30%作为测试集）
python scripts/ablation_study.py --data_path TDdata/TrainData.csv --epochs 50

# 指定独立测试集
python scripts/ablation_study.py --data_path TDdata/TrainData.csv --test_path TDdata/TestData.csv --epochs 50

# 使用全部训练数据作为测试集（test_split=0）
python scripts/ablation_study.py --data_path TDdata/TrainData.csv --test_split 0 --epochs 50

# 自定义测试集划分比例（例如20%）
python scripts/ablation_study.py --data_path TDdata/TrainData.csv --test_split 0.2 --epochs 50

# 只运行特定消融
python scripts/ablation_study.py --data_path TDdata/TrainData.csv --ablations full no_tid no_mcm
```

### ablation_study.py 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--data_path` | str | **必需** | 训练数据路径 |
| `--test_path` | str | None | 测试数据路径（可选） |
| `--test_split` | float | 0.3 | 测试集划分比例（若未指定 test_path） |
| `--ablations` | list | all | 要运行的消融实验列表 |
| `--epochs` | int | 50 | 训练轮数 |
| `--batch_size` | int | 32 | 批大小 |
| `--lr` | float | 1e-3 | 学习率 |
| `--d_model` | int | 64 | 基础隐藏维度 |
| `--n_layers` | int | 2 | 基础层数 |
| `--top_k` | int | 3 | 基础 Top-K |
| `--val_split` | float | 0.2 | 验证集划分比例 |

**测试集逻辑**：
- 若指定 `--test_path`：使用独立测试集
- 若 `--test_split > 0`：从训练数据中划分指定比例作为测试集
- 若 `--test_split = 0`：使用全部训练数据作为测试集（训练完后在同一数据上测试）

### 可用的消融实验

| 消融名称 | 描述 |
|----------|------|
| `full` | 完整模型（基准） |
| `no_fft` | 使用固定周期代替 FFT 检测 |
| `no_tid` | 移除 TID（无季节性/趋势分解） |
| `no_mcm` | 移除 MCM（无跨尺度混合） |
| `no_mrm` | 移除 MRM（使用简单平均代替幅值加权） |
| `single_scale` | 单尺度（无多尺度处理） |
| `top_k_1` | Top-K=1（只用 1 个周期） |
| `top_k_5` | Top-K=5（使用 5 个周期） |
| `layers_1` | 1 层 MixerBlock |
| `layers_4` | 4 层 MixerBlock |
| `d_model_32` | d_model=32（较小隐藏维度） |
| `d_model_128` | d_model=128（较大隐藏维度） |

### 输出示例

```
====================================================================================================
 Ablation Study - Test Set Results
====================================================================================================
Ablation         Description                       Params      Acc       F1    AUROC      FPR      FNR
----------------------------------------------------------------------------------------------------
full             Complete TimeMixer++ model        52,481   0.8520   0.8456   0.9123   0.1234   0.1433
no_mrm           Simple average instead of..       52,481   0.8423   0.8345   0.9012   0.1345   0.1544
no_mcm           No cross-scale mixing             48,234   0.8312   0.8234   0.8901   0.1456   0.1655
no_tid           No seasonal/trend decompos..      45,678   0.8178   0.8089   0.8789   0.1567   0.1766
no_fft           Fixed periods instead of FFT      52,481   0.8045   0.7956   0.8678   0.1678   0.1877
single_scale     No multi-scale processing         35,234   0.7823   0.7712   0.8456   0.1789   0.2066
====================================================================================================

Relative F1 (vs Full Model):
  no_mrm: -0.0111 (-1.3%)
  no_mcm: -0.0222 (-2.6%)
  no_tid: -0.0367 (-4.3%)
  no_fft: -0.0500 (-5.9%)
  single_scale: -0.0744 (-8.8%)

指标说明:
  FPR (误报率) = FP / (FP + TN) - 实际为负类但被预测为正类的比例
  FNR (漏报率) = FN / (TP + FN) - 实际为正类但被预测为负类的比例 (= 1 - Recall)
```

## 许可

MIT 许可证
