# LNO 算法与代码实现说明

本文档说明 Latent Neural Operator (LNO) 在本仓库 PyTorch 版本中的实现方式，重点覆盖 `LNO-PyTorch/ForwardProblem` 和 `LNO-PyTorch/InverseProblem` 的模型、数据流、训练配置与可调参数。内容依据论文 `2406.03923v5.pdf`、`README.md`、`module/model.py`、`module/dataset.py`、`module/utils.py`、`exp.py`、`train.py` 和配置文件整理。

## 1. 论文中的核心思想

论文将 PDE 前向问题和逆问题都表述为算子学习：

```text
F: f -> g
```

其中输入函数 `f` 可以是 PDE 系数、边界/初始条件、几何信息或部分观测值；输出函数 `g` 是待预测的物理场。传统 Transformer 类神经算子常在原始几何空间中对 `N` 个采样点做注意力，复杂度随点数快速上升。LNO 的核心思路是：

1. 用 Physics-Cross-Attention (PhCA) 把几何空间中的输入点序列编码到长度为 `M` 的潜空间。
2. 在潜空间中用 Transformer block 学习 PDE 算子。
3. 再用反向 PhCA 把潜表示解码回指定查询位置。

论文强调 PhCA 的关键性质是“观测位置”和“预测位置”解耦，因此模型可以在不同于输入采样点的位置输出结果，这对不规则网格、插值、外推和逆问题很重要。

## 2. ForwardProblem 的整体数据流

前向代码入口是：

```text
LNO-PyTorch/ForwardProblem/prepare.py
LNO-PyTorch/ForwardProblem/exp.py
LNO-PyTorch/ForwardProblem/module/dataset.py
LNO-PyTorch/ForwardProblem/module/model.py
```

运行顺序：

1. `prepare.py` 把原始 `.mat/.npy` 数据整理为 `{x, y1, y2}` 并保存到 `./datas/{name}_train.npy`、`./datas/{name}_val.npy`。
2. `LNO_dataset` 读取数据，必要时将 `x` 拼接到 `y1`，再做标准化。
3. `exp.py` 将高维网格展平为点序列 `[B, N, C]`。
4. `get_model_data()` 根据 config 构造模型、损失函数、优化器和学习率调度器。
5. 训练时调用 `model(x, y1)`，输出 `res` 与 `y2` 计算损失。

前向问题的统一张量约定：

```text
x  : [B, N, x_dim]   # trunk 输入，表示查询/参考坐标
y1 : [B, N, y1_dim]  # branch 输入，表示输入函数或几何信息
y2 : [B, N, y2_dim]  # 监督目标
```

## 3. Forward LNO 模型

### 3.1 主要类

`ForwardProblem/module/model.py` 中包含三个 LNO 变体：

| 类 | forward 签名 | 用途 |
| --- | --- | --- |
| `LNO` | `forward(x, y)` | 当前五个前向 benchmark 默认使用 |
| `LNO_single` | `forward(y)` | 单输入版本，不显式使用 trunk 坐标 |
| `LNO_triple` | `forward(x1, x2, y)` | 编码位置和解码位置分开的版本，最接近论文中输入/输出位置显式解耦的形式 |

五个前向配置文件都使用 `model.name = "LNO"`。

### 3.2 `LNO` 的构造模块

`LNO.__init__()` 中的关键模块：

```python
trunk_projector     = MLP(x_dim,  n_dim, n_dim, n_layer, act)
branch_projector    = MLP(y1_dim, n_dim, n_dim, n_layer, act)
attention_projector = MLP(n_dim,  n_dim, n_mode, n_layer, act)
attn_blocks         = [AttentionBlock] * n_block
out_mlp             = MLP(n_dim, n_dim, y2_dim, n_layer, act)
```

参数含义：

| 参数 | 含义 |
| --- | --- |
| `n_block` | 潜空间 Transformer block 数量 |
| `n_mode` | 潜空间 token 数，即论文中的 `M` |
| `n_dim` | 每个 token 的隐藏维度 |
| `n_head` | 多头注意力头数 |
| `n_layer` | 各 MLP 内部残差隐藏层数 |
| `attn` | 潜空间 self-attention 类型 |
| `act` | 激活函数 |

### 3.3 前向传播细节

源码：

```python
def forward(self, x, y):
    x = self.trunk_projector(x)
    y = self.branch_projector(y)

    score = self.attention_projector(x)
    score_encode = torch.softmax(score, dim=1)
    score_decode = torch.softmax(score, dim=-1)

    z = torch.einsum("bij,bic->bjc", score_encode, y)

    for block in self.attn_blocks:
        z = block(z)

    r = torch.einsum("bij,bjc->bic", score_decode, z)
    r = self.out_mlp(r)
    return r
```

若 batch 后张量为：

```text
x: [B, N, x_dim]
y: [B, N, y1_dim]
```

则中间形状为：

```text
trunk_projector(x)     -> [B, N, D]
branch_projector(y)    -> [B, N, D]
attention_projector(x) -> [B, N, M]
score_encode           -> [B, N, M]  # 在 N 维 softmax
score_decode           -> [B, N, M]  # 在 M 维 softmax
z                      -> [B, M, D]
attn_blocks(z)         -> [B, M, D]
r before out_mlp       -> [B, N, D]
out_mlp(r)             -> [B, N, y2_dim]
```

这对应论文中的 PhCA 编码、潜空间算子学习、PhCA 解码：

1. 编码：`score_encode` 把原始 `N` 个点的 branch 特征聚合为 `M` 个潜 token。
2. 潜空间建模：`attn_blocks` 在 `M` 个 token 上做 self-attention。
3. 解码：`score_decode` 把 `M` 个潜 token 分发回 `N` 个查询点。

当前 `LNO` 版本使用同一个 `attention_projector` 生成编码和解码权重，这对应论文中 PhCA encoder/decoder 共享投影参数的思想。`LNO_triple` 则拆成 `attention_encoder` 和 `attention_decoder`，并允许编码位置 `x1` 与解码位置 `x2` 不同。

### 3.4 AttentionBlock

`AttentionBlock` 是标准 Transformer 残差块：

```text
y = y + SelfAttention(LayerNorm(y))
y = y + MLP(LayerNorm(y))
```

可选 attention：

| config 值 | 实现 | 说明 |
| --- | --- | --- |
| `Attention_Vanilla` | scaled dot-product attention | 五个默认配置均使用 |
| `Attention_Linear_GNOT` | GNOT 风格线性注意力 | 降低复杂度，表达力不同 |
| `Galerkin` | Galerkin/Fourier linear attention 类中的 `galerkin` | 使用归一化的线性注意力 |
| `Nystrom` | `addition.py` 中的 `NystromAttention` | Nystrom 低秩近似注意力 |

论文消融中比较了多种 attention，最终默认配置保留经典 scaled dot-product attention。

### 3.5 MLP 结构

`MLP` 不是简单串联层，而是带残差的隐藏层结构：

```python
r = act(input(x))
for hidden_layer in hidden:
    r = r + act(hidden_layer(r))
r = output(r)
```

因此 `n_layer` 增大时，MLP 内部残差层变多，表达能力更强，但也增加参数量和优化难度。

## 4. 损失、优化器和训练循环

### 4.1 损失函数

`ForwardProblem/module/loss.py` 提供：

| config | 类 | 含义 |
| --- | --- | --- |
| `L2` | `LpLoss(p=2)` | 绝对 L2 风格误差 |
| `L1` | `LpLoss(p=1)` | 绝对 L1 风格误差 |
| `rL2` | `RelLpLoss(p=2)` | 相对 L2，五个默认配置使用 |
| `rL1` | `RelLpLoss(p=1)` | 相对 L1 |

论文附录说明前向问题使用 relative L2 训练 500 epoch；本仓库五个前向配置与此一致。

### 4.2 优化器和 scheduler

`get_model_data()` 支持：

```text
optimizer: Adam, AdamW, SGD
scheduler: NULL, Step, CosRestart, Cos, OneCycle
```

五个前向配置均为：

```text
optimizer = AdamW
lr = 1e-3
weight_decay = 5e-5
betas = (0.9, 0.99)
scheduler = OneCycle
batch_size = 4
epoch = 500
grad_clip = 1000.0
```

训练使用 `torch.distributed` 和 `DistributedDataParallel`，即便单卡脚本也通过 `torchrun --nproc_per_node 1` 启动。

### 4.3 验证逻辑

`val()` 在无梯度模式下运行模型。若训练集 normalizer 的 `y2_flag=True`，则将 `res` 和 `y2` 反标准化后再计算损失。

需要注意：当前 `val_dataset` 构造时会自己计算并应用验证集 normalizer，但 `val()` 反标准化时使用训练集 normalizer。严格复现实验时，建议让验证集复用训练集 normalizer。

## 5. 五个前向任务的默认配置

以本地 `ForwardProblem/configs/*.jsonc` 为准：

| 配置 | 数据集 | `n_block` | `n_mode` | `n_dim` | `n_head` | `n_layer` | loss |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `LNO_Darcy` | Darcy | 4 | 256 | 128 | 8 | 2 | rL2 |
| `LNO_Airfoil` | Airfoil | 8 | 256 | 128 | 8 | 2 | rL2 |
| `LNO_Elasticity` | Elasticity | 4 | 256 | 192 | 8 | 3 | rL2 |
| `LNO_Plasticity` | Plasticity | 4 | 256 | 128 | 8 | 2 | rL2 |
| `LNO_Pipe` | Pipe | 8 | 256 | 128 | 8 | 2 | rL2 |

论文附录的超参数表与当前代码配置不完全一致；复现实验时应优先记录实际使用的 config。

## 6. 配置字段说明与调参建议

### 6.1 `data`

```json
"data": {
  "name": "Darcy",
  "train_batch_size": 4,
  "val_batch_size": 4
}
```

`name` 必须与 `prepare.py` 和 `LNO_dataset` 中的分支一致。batch size 受显存限制较大，特别是 Darcy 和 Plasticity 点数很多。

### 6.2 `model`

```json
"model": {
  "name": "LNO",
  "n_block": 4,
  "n_mode": 256,
  "n_dim": 128,
  "n_head": 8,
  "n_layer": 2,
  "attn": "Attention_Vanilla",
  "act": "GELU"
}
```

调参建议：

| 参数 | 增大后的影响 | 风险 |
| --- | --- | --- |
| `n_mode` | 潜空间 token 更多，能表示更复杂场 | 显存和计算增加，过大可能收益饱和 |
| `n_block` | 潜空间算子层数更深 | 论文消融显示过深可能优化困难 |
| `n_dim` | token 表达能力增强 | 参数量显著增加 |
| `n_head` | 多头子空间更多 | 要保证 `n_dim % n_head == 0` |
| `n_layer` | projector/out MLP 更强 | 训练更慢，可能更难优化 |
| `attn` | 改变潜空间注意力机制 | 不同任务精度差异明显，需要验证 |

论文中的经验是：深度通常在 4 或 8 附近较优；宽度增大到 256 后收益趋于饱和；`n_mode` 增大通常先提升后饱和，Darcy/NS2d 可能更偏好较大的潜 token 数。

### 6.3 `loss`

前向 benchmark 默认使用 `rL2`，因为不同样本的场幅值可能不同，相对误差更适合比较算子预测质量。

### 6.4 `optimizer` 与 `scheduler`

默认 `AdamW + OneCycle` 是论文和代码共同采用的前向训练方案。若训练不稳定，优先调整：

```text
lr: 1e-3 -> 5e-4 或 2e-4
grad_clip: 保持 1000 或适当降低
batch_size: 显存不足时降低；但 DDP 下过小会增大梯度噪声
```

## 7. 时间序列模式

`exp.py` 通过配置名判断：

```python
if "_time" in arg.config:
    model_attr["time"] = True
```

若为时间模式，模型输出维度被设为 1，并使用 `train_time()` 做自回归式训练：每次预测一个时间步，把真实值或预测值拼回输入，再继续下一步。该路径主要服务 `LNO_time_NS2d.jsonc`，不是本文五个数据集默认路径。

## 8. InverseProblem 实现

逆问题代码位于：

```text
LNO-PyTorch/InverseProblem/prepare.py
LNO-PyTorch/InverseProblem/train.py
LNO-PyTorch/InverseProblem/infer.py
LNO-PyTorch/InverseProblem/module/model.py
LNO-PyTorch/InverseProblem/module/utils.py
```

### 8.1 数据生成

`prepare.py` 生成一维 Burgers 或 Allen-Cahn 数据。以 Burgers 为例，代码中的 PDE 为：

```text
u_t = 0.01 u_xx - u u_x
x in [0, 1], t in [0, 1]
```

空间和时间分辨率默认都是 512。初值来自 Gaussian Random Field；如果数据名含 `Force`，还会生成外力项 `F`，否则 `F=0`。

保存格式：

```python
{
    "x": [num, NT, NX, 2],  # [space, time] 坐标
    "y": [num, NT, NX, 1],  # 解 u
    "f": [num, NT, NX, 1],  # 外力项或 0
}
```

`*_std` 数据规模为：

```text
train = 4096
val   = 128
test  = 128
```

### 8.2 两阶段策略

论文和 README 描述的逆问题流程是：

1. **Completer**：在局部子域内，根据稀疏观测点补全该子域的完整解。
2. **Propagator**：把补全后的子域解作为输入，逐步外推到更大的区域或全域。

这正对应 `train.py` 中的两个训练函数：

```text
train_completer()
train_propagator()
```

推理时 `infer.py` 先调用 completer，再把补全结果拼成 propagator 的初始观测，最后递推到目标区域。

### 8.3 Masker 与 Poser

`utils.py` 中的 Masker 负责产生观测点：

| 类 | 用途 |
| --- | --- |
| `Masker_Completer_Random` | 子域内随机采样观测点 |
| `Masker_Completer_Fix` | 子域内按固定步长采样 |
| `Masker_Propagator_Random` | propagator 的逐阶段区域扩展 |

`Poser_Completer` 和 `Poser_Propagator` 负责决定测试指标统计的位置。

典型 completer 配置：

```json
"observation": {
  "method": "random",
  "initial_region": [0.5, 1.0],
  "initial_ratio": 0.1
}
```

含义是只在时间-空间域的一个中心子区域内观测，其中一个维度覆盖 50%，另一个维度覆盖 100%，并随机保留 10% 的点。

### 8.4 Inverse LNO 与 Forward LNO 的差异

`InverseProblem/module/model.py` 中的 `LNO` 更显式地拆出了 PhCA 的若干线性层：

```python
mode_mlp   = Linear(n_dim, n_mode)
encode_mlp = Linear(n_dim, n_dim)
decode_mlp = Linear(n_dim, n_dim)
Wv         = Linear(n_dim, n_dim)
```

前向传播：

```python
x = trunk_mlp(y[..., :x_dim])   # 观测点坐标
query = trunk_mlp(query)        # 查询点坐标
score_encode = softmax(mode_mlp(x), dim=1)
score_decode = softmax(mode_mlp(query), dim=-1)
y = branch_mlp(y)               # 观测点坐标 + 观测值
v = Wv(LayerNorm(y))
z = einsum(score_encode, v)
z = encode_mlp(z)
z = attn_blocks(z)
z = decode_mlp(z)
r = einsum(score_decode, z)
r = out_mlp(r)
```

与 Forward `LNO` 相比，inverse 版的 `query` 和观测 `y` 天然可以来自不同位置，因此更直接体现“观测位置和预测位置解耦”。

### 8.5 Inverse 默认配置

三个 LNO Burgers 配置：

| 配置 | role | 观测方式 | 优化器 | scheduler |
| --- | --- | --- | --- | --- |
| `LNO_completer_fix_Burgers` | completer | 固定步长 `[4,4]` | Adam | Step |
| `LNO_completer_random_Burgers` | completer | 随机 10% | Adam | Step |
| `LNO_propagator_Burgers` | propagator | 初始区域全观测 | AdamW | OneCycle |

模型参数均为：

```text
n_block = 4
n_mode  = 256
n_dim   = 96
n_head  = 8
n_layer = 3
loss    = MSE
epoch   = 500
```

## 9. 实验目录与日志

`exp.py` 会把实验输出写到：

```text
ForwardProblem/experiments/{exp}/
  checkpoint/
  log/
  para/
  src/
```

其中：

| 子目录 | 内容 |
| --- | --- |
| `checkpoint` | 每隔 `model_save_interval_epoch` 保存的 `.pt` |
| `log` | `log.txt`、TensorBoard event、loss/lr 的 `.npy` 和 `.png` |
| `para` | 本次运行的 `arg.json` 和 `config.json` |
| `src` | 运行时复制的源码快照 |

当前仓库已有五个 forward 实验目录。Airfoil、Elasticity、Pipe、Plasticity 日志跑到 500 epoch；Darcy 当前日志记录到 458 epoch，checkpoint 到 450 epoch。



