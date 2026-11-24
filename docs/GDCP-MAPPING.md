# GDCP：DTensor 到 GDTensor 的映射规格

本文定义 GDCP 如何把 PyTorch DTensor 的布局转换成 GDTensor 的计算视图和存储视图，以及转换必须满足的不变量。实现见 [`gdcp/adapters/fsdp2.py`](../gdcp/adapters/fsdp2.py) 的 `_describe_tensor`、[`gdcp/adapters/optimizer_state.py`](../gdcp/adapters/optimizer_state.py) 和 [`gdcp/policy.py`](../gdcp/policy.py)；持有集合的计算见 [`gdtensor_conn/_runs.py`](../gdtensor_conn/_runs.py) 和 [`gdtensor_conn/_geometry.py`](../gdtensor_conn/_geometry.py)。

映射语义以提交 `27e1f4a` 为基础，本文同时规定公开的数据访问扩展点。项目声明的最低 PyTorch 版本是 2.12；第 4 节的 PyTorch 语义在 2.12、2.14、2.15 上实测一致。

## 1 记号

- **进程组** `G = (g_0, …, g_{N-1})`：本次存取的进程组，`g_i` 是全局 rank。省略 `process_group` 时 `G = (0, …, N-1)`。组内编号 `λ(g_i) = i`。
- **DTensor** `T = (S, d, M, P)`：
  - `S = (s_0, …, s_{m-1})`：全局形状，`m` 是张量维数。
  - `d`：数据类型。
  - `M`：`device_mesh.mesh`，一个形状为 `(n_0, …, n_{k-1})` 的整数数组，元素是全局 rank，`k` 是 mesh 轴数。
  - `P = (p_0, …, p_{k-1})`：每条 mesh 轴一个分片方式。
- **坐标**：rank `r` 在 `M` 中的位置 `c(r) = (c_0, …, c_{k-1})`。`r` 不在 `M` 中时不持有 `T` 的任何元素。
- **全局下标**：元素 `(a_0, …, a_{m-1})` 的全局下标是行优先展开 `Σ a_e · Π_{f>e} s_f`。
- **持有序列** `own(T, r)`：rank `r` 的本地张量（`to_local()`）按行优先逻辑顺序逐个元素对应的全局下标序列。本地张量在内存中是否连续不影响这个定义。
- **chunk(A, n, c)**：把序列 `A` 按 `n` 等分取第 `c` 段。设 `L = |A|`，`w = ⌈L / n⌉`，取 `A` 中位置 `[min(c·w, L), min(c·w + w, L))` 的元素。末尾的段可能更短或为空。

## 2 计算视图的映射 Φ

### 2.1 前置条件

映射只依赖形状、数据类型、mesh 和分片方式，与张量设备及是否有实际存储无关；meta 张量也可建立同一映射。以下任一条件成立时，GDCP 在建立暂存目录、写任何文件之前报错：

| 条件 | 报错位置 |
|-|-|
| `d` 不在支持列表中。支持：bool、uint8/16/32/64、int8/16/32/64、float16、bfloat16、float32、float64 | `_describe_tensor` |
| `k ∉ {1, 2, 3}` | `_describe_tensor`、`policy._stored_tensor` |
| `M` 中有重复 rank，或含 `G` 以外的 rank | `_compute_mesh` |
| 某个 `p_a` 不是 `Shard`、`Replicate`、`_StridedShard` 三者之一（如 `Partial`） | `_torch_placement`、`axis_runs_of` |
| `Shard(e)` 或 `_StridedShard(e, ·)` 的 `e ∉ [0, m)`，或 `split_factor < 1` | `_torch_placement` |

数据访问的前置条件独立于上述映射：省略 `data_accessor` 时，默认 `TorchAccessor` 要求每个张量有可读的实际存储；meta 张量在准备阶段报错，消息包含存档键并说明需要 `data_accessor`。传入自定义工厂时，由工厂返回的 `gdtensor.access.DTensorDataAccessor` 提供或接收字节。各 rank 必须一致选择默认或自定义访问器；自定义访问器与 HF `hf_dtype` 权重转换（含 fp32 master 导出）的组合在写文件前拒绝，HF 完整状态格式不受此限制。

### 2.2 映射规则

`Φ(T) = (meta, mesh, placements)`：

- **meta** = `GlobalTensorMeta(S', d 的编码, 元素字节数)`，其中 `S' = S`；`m = 0`（标量）时 `S' = [1]`。
- **mesh**：令 `R = (λ(x) for x in flatten(M))`，即把 `M` 行优先展开后换成组内编号。
  - 若 `R = (0, 1, …, N-1)`，基础 mesh 为 `ArrangedMesh(N)`；否则为 `LiteralMesh(R)`。
  - 若 `k > 1`，mesh 为 `ReshapedMesh(基础 mesh, [n_0, …, n_{k-1}])`；否则就是基础 mesh。
- **placements**：对每条轴 `a` 逐项转换，不调整顺序，不合并：

| `p_a` | GDTensor |
|-|-|
| `Replicate()` | `Replicate()` |
| `Shard(e)`（类型恰为 `Shard`） | `Shard(e)` |
| `_StridedShard(e, split_factor=q)` | `StridedShard(e, q)` |

判断时先检查是否为 `_StridedShard`，再检查类型是否恰为 `Shard`，不依赖两者的继承关系（PyTorch 2.15 中 `_StridedShard` 不是 `Shard` 的子类）。PyTorch 私有类型的引用只出现在 [`gdcp/adapters/torch_placements.py`](../gdcp/adapters/torch_placements.py)。

### 2.3 非 DTensor 的张量

| 张量 `x` | Φ(x) |
|-|-|
| 模型的普通 buffer，或不是 DTensor 的参数（如 `ignored_params`） | `ArrangedMesh(N)`，`[Replicate()]` |
| 优化器状态，普通张量，`x.shape` 等于对应参数 `p` 的本地分片形状 | 先构造 `DTensor.from_local(x, p.device_mesh, p.placements)`，再按 2.2 转换 |
| 优化器状态，普通张量，维数大于 0，不满足上一行 | `ArrangedMesh(N)`，`[Replicate()]`，且须满足不变量 I7 |
| 优化器状态，标量，对应参数 `p` 是 DTensor 且 `p.device_mesh.size() < N` | mesh 按 2.2 由 `p.device_mesh` 得出，每条轴都是 `Replicate()` |
| 优化器状态，标量，其他情况 | `ArrangedMesh(N)`，`[Replicate()]` |

优化器状态如果本身是 DTensor，按它自己的 `(S, d, M, P)` 套用 2.2，不参照参数的布局。这些规则对实际存储张量和 meta 张量相同；标量 `step` 保持原有的 CPU 标量规则，不因使用自定义访问器改变。

## 3 GDTensor 的持有序列

对 `Φ(T)` 和 rank `r`（组内编号），GDTensor 按以下过程计算持有序列 `own_G(Φ(T), r)`：

1. 对每个张量维 `e`，令 `A_e = (0, 1, …, s_e - 1)`。
2. 按 `a = 0, 1, …, k-1` 依次处理每条 mesh 轴，`n = n_a`，`c = c_a(r)`：
   - `Replicate()`：不变。
   - `Shard(e)`：`A_e ← chunk(A_e, n, c)`。
   - `StridedShard(e, q)`：设 `L = |A_e|`，`w₁ = ⌈L / q⌉`。把 `A_e` 按位置切成连续的块 `B_0, B_1, …`，每块长 `w₁`（最后一块可能更短），共 `⌈L / w₁⌉` 块。`A_e ← chunk(B_0, n, c) ++ chunk(B_1, n, c) ++ …`。
3. `own_G = A_0 × A_1 × … × A_{m-1}` 的行优先展开，每个元组换成全局下标。

`L = 0` 时各步都得到空序列。实现中 `A_e` 以“有序的连续区间列表”表示，相邻区间合并，运算量与区间数成正比，与元素个数无关（`axis_runs_of`）。当所有 `A_e` 都是单个区间时，持有范围是一个矩形块（`owned_box`）。

公开的冻结数据类 `gdcp.DataAccessorRequest` 把这些持有区间转换成全局字节区间，按本地打包顺序提供为 `ownership[key]`。`tensors[key]` 是本 rank 的原始张量（DTensor 或普通张量），`local_offset(key, global_offset, size)` 使用 GDCP 原有的区间翻译，返回请求范围在打包后的本地字节序列中的偏移。访问器不能假定本地持有范围是单个矩形，也不能用全局偏移直接索引本地缓冲区。每个组件、每次保存或加载分别调用工厂，`component` 为 `"model"` 或 `"optimizer"`，`mode` 为 `"save"` 或 `"load"`。

## 4 PyTorch 侧的语义

`own(T, r)` 由 PyTorch 决定。PyTorch 按 mesh 轴从左到右，对当前保留的部分依次切分：

- `Shard(e)`：按 `torch.chunk` 规则切成 `n` 段，段长 `⌈L / n⌉`，取第 `c` 段。
- `_StridedShard(e, split_factor=q)`：先按同样规则切成 `q` 块，每块再切成 `n` 段，把每块的第 `c` 段按块的顺序拼接（`torch/distributed/tensor/placement_types.py` 中 `_StridedShard._split_tensor`）。

PyTorch 在 `flatten`、`view`、`reshape`、`transpose`、`permute` 等操作中自行推导输出的 `P`（`torch/distributed/tensor/_ops/_view_ops.py`），GDCP 只转换操作之后的 `T`。

已知差异：PyTorch 2.15 对嵌套的不均匀 `_StridedShard`（如形状 `[37, 3]`、`(_StridedShard(0, 3), _StridedShard(0, 2))`）调用 `full_tensor()` 得到的结果与 `arange` 不一致，而各 rank 的 `to_local()` 是自洽的。GDCP 以 `to_local()` 为准。

## 5 存储视图 Σ

对 `DTensorTable` 中第 `j` 个张量 `T_j`（`j` 从 0 开始，按表顺序）：

- 若 `Φ(T_j)` 的所有分片方式都是 `Replicate()`：存储视图为 `[Replicate()]`，mesh 只含组内 rank `j mod N`。
- 否则：存储视图为 `[Shard(0)]`，mesh 为 `ArrangedMesh(N)`。

`Σ` 只依赖 `S`、`d`、`j`、`N` 以及“是否有分片轴”，与 `M` 的形状、rank 排列和具体分片方式无关。safetensors 和 DCP 格式按这个逻辑布局每个组内 rank 写一个文件；HF 格式使用同一逻辑布局，再按 `max_shard_size` 组织文件。

同一个存档键出现在多个互不相交的参数子 mesh 上时，GDCP 先通过访问器的 `read_to_buffer` 读取持有区间，比对各子 mesh 对应位置的分片内容（SHA-256），一致后选定其中一个子 mesh 作为写入方（`_select_submesh_writers`）。

## 6 加载时的目标布局

加载时，`Φ` 作用于目标模型和目标优化器当前的张量，`Σ` 由存档决定。目标布局可以与保存时不同。

未训练过的优化器没有状态张量。GDCP 按存档中记录的字段布局新建目标张量，不调用 `step()`。记录的布局关系有五种：

| 记录 | 新建的目标张量 |
|-|-|
| `parameter` | 形状、mesh、分片方式与参数相同 |
| `reshape` | 参数的 `view(shape)`，分片方式由 PyTorch 推导 |
| `project` | 参数某几个维度上的张量，分片方式取参数在这些维度上的分片方式，其他轴为 `Replicate()` |
| `replicate` | 各轴都是 `Replicate()` |
| `independent` | 存档中记录的 mesh 形状、rank 和分片方式 |

`independent` 的 mesh 若与参数的 mesh、根 mesh、子 mesh 或某条轴的 rank 排列和形状相同，直接复用那个 `DeviceMesh` 对象（见 I8）。

## 7 不变量

| 编号 | 不变量 | 验证方式 |
|-|-|-|
| I1 | **持有序列一致**：对通过 2.1 的任意 `T` 和任意 `r ∈ G`，`own_G(Φ(T), λ(r)) = own(T, r)`，元素和顺序都相同。 | 2.12、2.14、2.15 上对真实 DTensor 逐 rank 比对 `to_local()`（[`tests/support/gdcp_strided_probe.py`](../tests/support/gdcp_strided_probe.py)、[`tests/gdcp/test_strided.py`](../tests/gdcp/test_strided.py)）；GDTensor 侧语义见 [`tests/core/test_strided_shard.py`](../tests/core/test_strided_shard.py)。另把第 3 节的定义逐字实现，与三个 PyTorch 版本、4 进程和 8 进程共 432 条 rank 记录逐一比对，全部一致。未做形式化证明。 |
| I2 | **单轴划分**：固定其他轴坐标，`Shard` 或 `StridedShard` 轴上 `c = 0, …, n-1` 的结果两两不相交，并集等于切分前的 `A_e`。`Replicate` 轴上各坐标结果相同。 | `Shard` 部分由 [`formal/lean/ShardAlgebra.lean`](../formal/lean/ShardAlgebra.lean) 证明；`StridedShard` 是对每块 `B_i` 应用 `chunk` 后拼接，由 `chunk` 的划分性质直接得出，尚未形式化。 |
| I3 | **整体覆盖**：由 I2，所有 rank 的持有集合之并等于全部 `Π s_e` 个元素；两个 rank 持有同一元素，当且仅当它们的坐标只在 `Replicate` 轴上不同。 | 由 I2 推出；保存前的全局计划检查（GDTensor `validate_rank_save`）逐 rank 核对覆盖。 |
| I4 | **只存一份**：`Σ` 中每个元素恰好属于一个存储槽，与计算侧有多少副本无关。 | [`tests/gdcp/test_fsdp2.py`](../tests/gdcp/test_fsdp2.py) `test_06_replicated_tensor_dedup`；HSDP、HSDP + TP 的写入方统计。 |
| I5 | **存储与计算无关**：`Σ` 不随 `M` 和 `P` 变化（只区分“全复制”和“有分片”），因此任何满足 2.1 的目标布局都能从同一份存档读取。 | 跨布局、跨进程数恢复的回归（`test_02_reshard_matrix`、`test_parameter_layouts`、`test_dp_tp_same_and_different_dimensions`、`test_hsdp_tp_checkpoint_layouts`、`test_real_fsdp2_strided_cross_world`）。 |
| I6 | **先拒绝后写入**：2.1 中的任一条件，以及各 rank 对是否使用自定义访问器或张量清单（键、全局形状、数据类型、元素宽度、mesh、分片方式、别名）不一致，都在建立暂存目录前报错，已发布的存档不受影响。 | `tests/gdcp/test_strided.py`（`Partial` 拒绝且无文件）；[`tests/gdcp/test_review_fixes.py`](../tests/gdcp/test_review_fixes.py) `test_facade_checks_materialized_tensor_schema`。 |
| I7 | **普通张量的副本**：2.3 第三行的普通张量，保存前各 rank 比对 SHA-256，不一致就报错。模型的普通 buffer 不做比对，取组内 rank `j mod N` 的值。 | [`tests/gdcp/test_optimizer_audit.py`](../tests/gdcp/test_optimizer_audit.py) `test_plain_optimizer_state_requires_collective_byte_agreement`。 |
| I8 | **复用参数的 mesh**：加载后，与参数 mesh 等价的优化器状态 mesh 是同一个 `DeviceMesh` 对象，保证优化器的下一步可以运行。 | `test_restored_optimizer_state_reuses_parameter_meshes`（Gloo、纯 NCCL、混合，4 卡和 8 卡）。 |
| I9 | **组内编号**：存档中的 rank 编号、文件名和每 rank 额外状态都使用组内编号 `λ`，与全局 rank 无关。 | `test_independent_process_subgroups`（非连续子组 `[0,2]`、`[1,3]`）。 |
| I10 | **只经访问器存取数据**：GDCP 对模型及优化器张量的所有数据读取和写入都通过 `DTensorDataAccessor`；普通优化器副本和参数子 mesh 的 SHA-256 比对也通过 `read_to_buffer` 读取持有区间，不直接读取张量存储。布局及元数据检查只访问形状、数据类型和分片信息。 | 自定义访问器回归覆盖保存、加载、普通优化器副本和参数子 mesh 比对；工厂请求的持有区间及偏移覆盖连续、非连续和一般 `StridedShard` 布局。 |
| I11 | **默认数据访问与存档字节不变**：不传 `data_accessor` 时继续使用 `TorchAccessor`，既有输入的发布文件字节、保存 I/O 和载入 payload I/O 不变。载入预检查的例外：本地文件系统由协调 rank 检查全部存储槽，其他 rank 只检查与其已验证计划相交的槽；所有 rank 通过集体预检查后才修改目标。省略非本地 DCP 记录及 safetensors 头读取，因此完整预检查 I/O 轨迹会变化。HDFS 保留原有按需下载和已下载文件检查范围。 | 原有逐文件内容和 I/O 证明见[数据访问扩展验证报告](../../docs/gdcp/review-fixes-evidence/data-accessor/REPORT.md)。本次与 `6da99a2` 比较计划、发布字节、保存及 payload I/O，单独列出预检查读取差异；不把完整 I/O 轨迹宣称为相同。见[本地元数据验证报告](../../docs/gdcp/review-fixes-evidence/local-metadata/REPORT.md)。 |

载入时若各 rank 使用不同的本地存档副本，完整预检查针对协调 rank 的副本；
其他 rank 检查其实际读取的槽，不检查各自副本中未读取的槽。所有实际读取的
槽仍须在任何目标写入前通过集体预检查。

## 8 示例

**DP + TP 切同一维度**：`S = (8, 3)`，`M = [[0,1],[2,3]]`（轴为 dp、tp），`P = (_StridedShard(0, split_factor=2), Shard(0))`，`N = 4`。

`Φ(T)` = `ReshapedMesh(ArrangedMesh(4), [2,2])`，`[StridedShard(0, 2), Shard(0)]`。按第 3 节，维 0 上：

| rank | 坐标 | 轴 0 之后的 `A_0` | 轴 1 之后的 `A_0` |
|-|-|-|-|
| 0 | (0, 0) | 0, 1, 4, 5 | 0, 1 |
| 1 | (0, 1) | 0, 1, 4, 5 | 4, 5 |
| 2 | (1, 0) | 2, 3, 6, 7 | 2, 3 |
| 3 | (1, 1) | 2, 3, 6, 7 | 6, 7 |

**flatten 之后**：`S = (32,)`，`M` 同上，`P = (Shard(0), _StridedShard(0, split_factor=2))`（由形状 `(4, 8)`、`(Shard(0), Shard(1))` 的 DTensor 调用 `flatten()` 得到）。rank 0 的 `A_0`：轴 0 之后为 `0…15`；轴 1 把它切成 `0…7`、`8…15` 两块，各取前一半，得到 `0…3, 8…11`。持有序列由两个不相邻的区间组成，不是一个矩形块。
