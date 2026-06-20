# LNO 数据集说明

本文档面向 `LNO-PyTorch/ForwardProblem` 中的五个前向问题数据集：Darcy、Airfoil、Elasticity、Plasticity、Pipe。内容依据本仓库 `README.md`、论文 `2406.03923v5.pdf`、`ForwardProblem/prepare.py`、`ForwardProblem/module/dataset.py`、五个配置文件和本地数据文件元信息整理。

## 1. 总体数据协议

论文把前向问题统一成算子学习任务：给定输入函数 `f` 的采样值，学习到输出函数 `g` 的映射。代码中所有前向数据最终都会被 `prepare.py` 转成同一种保存格式：

```python
{
    "x":  x,   # 查询/输出位置坐标，或参考计算网格坐标
    "y1": y1,  # 输入函数、几何信息或输入物理场
    "y2": y2,  # 监督目标，即模型要预测的输出物理量
}
```

`LNO_dataset` 读取 `./datas/{data_name}_train.npy` 和 `./datas/{data_name}_val.npy` 后会做两步统一处理：

1. 对 Darcy、Plasticity、Airfoil、Pipe、NS2d，将 `x` 与原始 `y1` 在最后一维拼接：`y1 = torch.cat((x, y1), dim=-1)`。也就是说 branch 输入同时包含参考坐标和输入物理/几何信息。
2. 对 Darcy、Elasticity、Plasticity 做标准化：`x`、`y1`、`y2` 分别按最后一维统计均值和标准差，再做 `(data - mean) / std`。Airfoil 和 Pipe 在当前实现中不做该标准化。

训练循环会把任意网格维度展平为点序列：

```python
x  : [B, ..., x_dim]  -> [B, N, x_dim]
y1 : [B, ..., y1_dim] -> [B, N, y1_dim]
y2 : [B, ..., y2_dim] -> [B, N, y2_dim]
```

这里 `N` 是每个样本的空间点数，或空间-时间点数；最后一维永远被代码视为通道/特征维。

## 2. 数据集总览

| 数据集 | 问题类型 | 原始文件 | 训练/验证 | 模型输入维度 | 输出维度 |
| --- | --- | --- | --- | --- | --- |
| Darcy | 二维达西流，规则网格 | `piececonst_r241_N1024_smooth1/2.mat` | 1024 / 1024 | `x_dim=2`, `y1_dim=3` | `y2_dim=1` |
| Airfoil | NACA 翼型流场，不规则几何映射到参考网格 | `NACA_Cylinder_Q/X/Y.npy` | 1000 / 200 | `x_dim=2`, `y1_dim=4` | `y2_dim=1` |
| Elasticity | 随机单胞弹性，应力预测，不规则点云 | `Random_UnitCell_XY/sigma/rr/theta_10.npy` | 1000 / 1000 | `x_dim=2`, `y1_dim=2` | `y2_dim=1` |
| Plasticity | 塑性成形/锻造，空间-时间网格 | `plas_N987_T20.mat` | 900 / 87 | `x_dim=3`, `y1_dim=4` | `y2_dim=4` |
| Pipe | 参数化管道流，不规则几何映射到参考网格 | `Pipe_Q/X/Y.npy` | 1000 / 200 | `x_dim=2`, `y1_dim=4` | `y2_dim=1` |

说明：表中的 `y1_dim` 是经过 `LNO_dataset` 拼接后的实际 branch 输入维度。

下载地址

| Dataset       | Link                                                         |
| ------------- | ------------------------------------------------------------ |
| Darcy         | [[Google Cloud]](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) |
| NS2d          | [[Google Cloud]](https://drive.google.com/drive/folders/1UnbQh2WWc6knEHbLn-ZaXrKUZhp7pjt-) |
| AirFoil       | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Elasticity    | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Plasticity    | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |
| Pipe          | [[Google Cloud]](https://drive.google.com/drive/folders/1YBuaoTdOSr_qzaow-G-iwvbUI7fiUzu8) |

## 3. Darcy

### 3.1 物理背景

Darcy 数据集对应二维多孔介质稳态渗流问题，典型形式是

```text
-div(a(x) grad u(x)) = f(x), x in [0, 1]^2
u(x) = 0, x on boundary
```

其中 `a(x)` 是空间变化的渗透率/扩散系数，`u(x)` 是压力或势函数。前向学习任务是从系数场 `a` 预测解场 `u`。

论文中 Darcy 与 NS2d 被归为规则网格 benchmark。当前代码默认使用 `241 x 241` 分辨率；本地还包含 `421 x 421` 高分辨率数据，但 `prepare.py` 默认没有使用它。

### 3.2 原始数据

默认使用：

```text
Darcy_241/piececonst_r241_N1024_smooth1.mat
Darcy_241/piececonst_r241_N1024_smooth2.mat
```

MAT 文件变量：

| 变量 | 形状 | 类型 | 用途 |
| --- | --- | --- | --- |
| `coeff` | `[1024, 241, 241]` | single | 输入系数场 `a(x)` |
| `sol` | `[1024, 241, 241]` | single | 输出解场 `u(x)` |
| `Kcoeff`, `Kcoeff_x`, `Kcoeff_y` | `[1024, 241, 241]` | double | 附加系数相关数据，当前 `prepare.py` 未使用 |

本地 `Darcy_421` 文件包含相同变量，分辨率为 `[1024, 421, 421]`，可用于分辨率泛化或高分辨率实验，但当前配置不直接调用。

### 3.3 `prepare.py` 处理流程

`load_Darcy(path, src_res=241, obj_res=241)` 做以下处理：

1. 读取 `coeff` 作为 `y1`，读取 `sol` 作为 `y2`。
2. 按 `(src_res - 1) // (obj_res - 1)` 下采样；默认 `src_res=obj_res=241`，因此不下采样。
3. 生成规则坐标网格：

```python
x = meshgrid(linspace(0, 1, 241), linspace(0, 1, 241))
x.shape = [1024, 241, 241, 2]
```

4. 将 `coeff` 和 `sol` 扩展通道维：

```text
raw y1: [1024, 241, 241, 1]
raw y2: [1024, 241, 241, 1]
```

5. 拼接 smooth1 和 smooth2，共 2048 个样本，前 1024 个作为训练，后 1024 个作为验证。

### 3.4 进入模型的张量含义

`LNO_dataset` 会把 `x` 拼入 `y1`：

```text
x  : [B, 241, 241, 2] -> [B, 58081, 2]
y1 : [B, 241, 241, 3] -> [B, 58081, 3]  # [x_ref, y_ref, coeff]
y2 : [B, 241, 241, 1] -> [B, 58081, 1]  # sol
```

Darcy 会做输入和输出标准化。验证时如果 `normalizer.is_apply_y2()` 为真，`val()` 会对预测和标签做反标准化后再计算相对 L2。

## 4. Airfoil

### 4.1 物理背景

Airfoil 数据集来自 Geo-FNO 系列不规则网格 benchmark，用于学习不同 NACA 翼型/圆柱外流几何下的流场标量。论文将 Airfoil 归为不规则网格问题，因为真实物理坐标随几何变化，而模型还需要在统一参考网格上组织输入。

在当前 LNO 代码中，输入几何由每个样本的物理坐标 `X, Y` 表示，监督目标取 `Q` 的第 5 个通道：`Q[:, 4, :, :]`。该 benchmark 通常把这一通道作为目标标量场，例如 Mach 数；源码没有在变量名层面对通道物理量做进一步注释。

### 4.2 原始数据

当前目录包含两组 Airfoil 数据：

```text
airfoil/naca/NACA_Cylinder_Q.npy      shape=(2490, 5, 221, 51)
airfoil/naca/NACA_Cylinder_X.npy      shape=(2490, 221, 51)
airfoil/naca/NACA_Cylinder_Y.npy      shape=(2490, 221, 51)
airfoil/naca/NACA_Q.npy               shape=(2490, 5, 11220)
airfoil/naca/NACA_X.npy               shape=(2490, 11220)
airfoil/naca/NACA_Y.npy               shape=(2490, 11220)
airfoil/naca_interp/NACA_*_interp.npy shape=(1200, 101, 101)
```

当前 `prepare.py` 使用的是 `NACA_Cylinder_Q/X/Y.npy`，不是 `naca_interp` 目录。

### 4.3 `prepare.py` 处理流程

```python
Q = NACA_Cylinder_Q[:, 4, :, :]  # [2490, 221, 51]
X = NACA_Cylinder_X              # [2490, 221, 51]
Y = NACA_Cylinder_Y              # [2490, 221, 51]
```

随后构造参考坐标网格：

```text
x_ref: 221 x 51 x 2, 坐标范围均为 [0, 1]
x.shape = [1200, 221, 51, 2]
```

`y1 = concat(X, Y)`，表示每个参考网格点对应的真实物理坐标；`y2 = Q`，表示该物理几何下的目标流场标量。

训练/验证划分：

```text
train: 前 1000 个样本
val  : 后 200 个样本
```

注意：`x_ref` 对所有样本相同，代码只重复 1200 份；`X/Y/Q` 本地文件有 2490 份。由于 `x_ref` 是样本无关的参考网格，训练前 1000 和验证后 200 在形状上仍可正常保存和训练。

### 4.4 进入模型的张量含义

```text
x  : [B, 221, 51, 2] -> [B, 11271, 2]  # 参考计算坐标
y1 : [B, 221, 51, 4] -> [B, 11271, 4]  # [x_ref, y_ref, X_phys, Y_phys]
y2 : [B, 221, 51, 1] -> [B, 11271, 1]  # Q 的第 5 个通道
```

Airfoil 在当前 `LNO_dataset` 中不做标准化。

## 5. Elasticity

### 5.1 物理背景

Elasticity 数据集对应随机单胞材料的弹性响应预测。输入是不规则网格上的二维坐标，输出是该坐标点上的应力标量 `sigma`。论文将 Elasticity 归为不规则网格 benchmark。

从算子学习角度看，模型学习的是由单胞几何/材料微结构诱导的坐标到应力场的映射。当前 LNO 代码没有显式把 `rr` 或 `theta` 输入模型，而是直接使用每个样本的点坐标 `XY` 作为输入函数。

### 5.2 原始数据

当前使用 `elasticity/Meshes`：

```text
Random_UnitCell_rr_10.npy     shape=(42, 2000)
Random_UnitCell_sigma_10.npy  shape=(972, 2000)
Random_UnitCell_theta_10.npy  shape=(2000, 10)
Random_UnitCell_XY_10.npy     shape=(972, 2, 2000)
```

其中：

| 变量 | 代码用途 |
| --- | --- |
| `XY` | 使用。转置为 `[2000, 972, 2]`，作为点坐标 |
| `sigma` | 使用。转置为 `[2000, 972]` 后扩展为 `[2000, 972, 1]` |
| `rr` | 读取但当前未进入训练张量 |
| `theta` | 读取但当前未进入训练张量 |

目录中还存在 `Interp`、`Omesh`、`Rmesh` 插值或变形网格版本，当前 `prepare.py` 未使用。

### 5.3 `prepare.py` 处理流程

```python
XY = transpose(Random_UnitCell_XY_10, (2, 0, 1))  # [2000, 972, 2]
sigma = transpose(Random_UnitCell_sigma_10, (1, 0))
sigma = expand_dims(sigma, axis=2)                # [2000, 972, 1]
x = XY
y1 = x
y2 = sigma
```

训练/验证划分：

```text
train: 前 1000 个样本
val  : 后 1000 个样本
```

### 5.4 进入模型的张量含义

Elasticity 是五个数据集中唯一没有在 `LNO_dataset` 中把 `x` 拼接进 `y1` 的任务，因为 `prepare.py` 已经令 `y1=x`：

```text
x  : [B, 972, 2]
y1 : [B, 972, 2]  # 物理点坐标
y2 : [B, 972, 1]  # sigma
```

Elasticity 会做 `x/y1/y2` 标准化。

## 6. Plasticity

### 6.1 物理背景

Plasticity 数据集对应塑性成形/锻造过程中的时空场预测。Geo-FNO 系列 benchmark 中，该任务通常包含二维空间和时间维度，模型需要根据输入形状/工况预测随时间演化的输出场。

当前代码将 `x` 构造成三维坐标 `[x, y, t]`，将 `input` 扩展并广播到所有 `y` 和 `t` 位置，再预测 `output` 的 4 个通道。

### 6.2 原始数据

```text
plasticity/plas_N987_T20.mat
```

MAT 文件变量：

| 变量 | 形状 | 类型 | 用途 |
| --- | --- | --- | --- |
| `input` | `[987, 101]` | double | 输入函数，一维参数/形状描述 |
| `output` | `[987, 101, 31, 20, 4]` | double | 输出时空场，最后一维为 4 个输出通道 |

### 6.3 `prepare.py` 处理流程

网格参数：

```text
SRC_RES1 = 101
SRC_RES2 = 31
SRC_RES3 = 20
```

构造坐标：

```text
x.shape = [987, 101, 31, 20, 3]
x[..., :] = [x_position, y_position, time]
```

处理输入：

```text
input: [987, 101]
y1   : [987, 101, 31, 20, 1]
```

其中 `input` 被沿第二个空间维和时间维广播，使每个时空点都带有对应的一维输入信息。

处理输出：

```text
y2 = output
y2.shape = [987, 101, 31, 20, 4]
```

训练/验证划分：

```text
train: 前 900 个样本
val  : 后 87 个样本
```

### 6.4 进入模型的张量含义

```text
x  : [B, 101, 31, 20, 3] -> [B, 62620, 3]
y1 : [B, 101, 31, 20, 4] -> [B, 62620, 4]  # [x, y, t, input]
y2 : [B, 101, 31, 20, 4] -> [B, 62620, 4]  # 4 通道输出场
```

Plasticity 会做 `x/y1/y2` 标准化。

## 7. Pipe

### 7.1 物理背景

Pipe 数据集是参数化管道几何下的流场预测任务。每个样本有一套真实物理坐标 `X, Y`，表示参考网格被映射到不同管道形状后的实际位置；输出是 `Pipe_Q` 的第 1 个通道。

论文将 Pipe 归为不规则网格 benchmark。LNO 通过把参考坐标和物理坐标同时放进 branch 输入，使模型看到几何变形信息。

### 7.2 原始数据

```text
Pipe_Q.npy          shape=(2310, 3, 129, 129)
Pipe_X.npy          shape=(2310, 129, 129)
Pipe_Y.npy          shape=(2310, 129, 129)
Pipe_Q_interp_x.npy shape=(1200, 129, 129)  # 当前 prepare.py 未使用
Pipe_res.npy        shape=(2310,)           # 当前 prepare.py 未使用
```

当前 `prepare.py` 取：

```python
Q = Pipe_Q[:, 0, :, :]  # 第 1 个物理量通道
X = Pipe_X
Y = Pipe_Y
```

### 7.3 `prepare.py` 处理流程

构造参考坐标网格：

```text
x_ref: 129 x 129 x 2
x.shape = [2310, 129, 129, 2]
```

输入和输出：

```text
y1 = concat(X, Y)  # [2310, 129, 129, 2]
y2 = Q             # [2310, 129, 129, 1]
```

训练/验证划分：

```text
train: 前 1000 个样本
val  : 后 200 个样本
```

### 7.4 进入模型的张量含义

```text
x  : [B, 129, 129, 2] -> [B, 16641, 2]  # 参考计算坐标
y1 : [B, 129, 129, 4] -> [B, 16641, 4]  # [x_ref, y_ref, X_phys, Y_phys]
y2 : [B, 129, 129, 1] -> [B, 16641, 1]  # Pipe_Q 的第 1 个通道
```

Pipe 在当前 `LNO_dataset` 中不做标准化。

## 8. 数据准备与目录约定

`ForwardProblem/module/setting.py` 固定了路径：

```python
CONFIG_PATH = "./configs"
DATA_PATH = "./datas"
EXP_PATH = "./experiments"
```

因此运行脚本时，原始数据应位于 `LNO-PyTorch/ForwardProblem/datas` 下。仓库当前把原始数据按任务放在 `LNO-PyTorch/airfoil`、`LNO-PyTorch/pipe` 等目录；若直接运行 `ForwardProblem/prepare.py`，需要把相应原始文件复制到 `ForwardProblem/datas`，或使用实验目录中保存的 `src/prepare.py` 路径版本作为参考进行调整。

典型流程：

```bash
cd LNO-PyTorch/ForwardProblem
python prepare.py --data_name Darcy
torchrun --nnodes 1 --nproc_per_node 1 exp.py --config LNO_Darcy --device "0" --exp LNO_Darcy --seed 0
```

`prepare.py` 只在 `{data_name}_train.npy` 或 `{data_name}_val.npy` 不存在时重新生成数据。



