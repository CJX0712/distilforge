# DistilForge 架构设计（作者：晨星）

> 通用分类知识蒸馏系统。旗舰 `DistilFuse`：温度软标签蒸馏 + 硬标签交叉熵自适应混合（Hinton et al., 2015）。

## 1. 设计目标

- **模块化、单向无环**：`cli → pipeline → {data, distill, preprocess, training, eval} → core`。
- **确定性可复现**：唯一 seed 入口 `core.seed.set_all(seed)`，同 seed 两次运行核心指标**逐位一致**。
- **防数据泄漏**：预处理器仅 fit 训练集；teacher 仅在训练集训练；各 student 独立同 seed 起点，差异仅来自训练目标。
- **离线可降级**：Tier-0 后端（scikit-learn）缺失时自动探测并跳过，`available_*()` 显式暴露，不伪造数字。
- **零外部权重**：teacher 用 scikit-learn 本地训练，student 纯 numpy 自研，无任何在线下载。

## 2. 模块职责

| 模块 | 职责 |
|------|------|
| `core/types.py` | `DatasetSpec` / `BenchmarkRecord` / `AggregatedResult` dataclass，统一"accuracy 越大越好"语义 |
| `core/errors.py` | 错误码 E000~E500（Config/Data/Model/Pipeline/Dependency） |
| `core/config.py` | `DistilForgeConfig`：`ENV_DISTILFORGE_*` 覆盖 + schema 校验 |
| `core/seed.py` | `set_all(seed)`：一次性设齐 numpy / random / PYTHONHASHSEED（torch 可选） |
| `core/interfaces.py` | `Classifier` Protocol |
| `data/synthetic.py` | 合成分类数据（sklearn `make_classification` 主路径，纯 numpy 离线兜底） |
| `data/loaders.py` | sklearn 内置集（digits/iris/wine）+ CSV 载入，`AVAILABLE_SKLEARN` 探测 |
| `preprocess/scaler.py` | `StandardScaler`（仅 fit 训练集） |
| `distill/teachers.py` | `SklearnTeacher`：RF / GBM / MLP / Logistic，`logits(X)=log(predict_proba+ε)` |
| `distill/students.py` | `NumpyLogisticStudent` / `NumpyMLPStudent`（tanh 单隐层），解析 KD 梯度 |
| `distill/distiller.py` | `DistilFuse.distill()`：软标签 + 硬标签混合训练编排 |
| `distill/baselines.py` | `train_hard`（仅硬标签）/ `train_pseudo`（teacher argmax 伪标签，隔离增益来源） |
| `training/train.py` | `minibatches` / `constant_lr` / `cosine_lr` |
| `eval/metrics.py` | `accuracy` / `error_rate` / `rel_error_drop` |
| `pipeline/distill_pipeline.py` | `DistillPipeline`：`run_once` / `benchmark` / `aggregate` / `ablation` |
| `cli.py` | argparse 入口：`run` / `ablation` / `doctor` |
| `examples/run_demo.py` | 端到端演示：多数据集 × 多 seed 基准 + 确定性校验 + 落盘 `benchmark.json` |

## 3. 知识蒸馏数学内核

蒸馏损失（Hinton 2015，含温度缩放补偿）：

```
L = α · T² · KL( softmax(z_t / T) ‖ softmax(z_s / T) ) + (1 − α) · CE( y , softmax(z_s) )
```

对 student logits 的解析梯度（minibatch 平均，N 为批大小）：

```
dz = [ α·T·(q_s − p_t) + (1 − α)·(p_s − onehot) ] / N
```

其中 `p_s = softmax(z_s)`、`q_s = softmax(z_s/T)`、`p_t = softmax(z_t/T)`。
`T²` 补偿温度缩放带来的梯度幅度变化（标准 KD 实现）。`α` 控制软/硬损失权重（默认 0.5）。

**不变量（手写实现交叉验证）**：解析梯度 vs 数值梯度一致（见 `tests/test_students.py`）。

## 4. 数据流（单次 `run_once`）

```
DatasetSpec
   │  data.synthetic / data.loaders
   ▼
(Xtr, ytr, Xte, yte)
   │  preprocess.StandardScaler.fit(Xtr) → transform(Xtr/Xte)   ← 仅 train fit
   ▼
teacher = SklearnTeacher.fit(Xtr_s, ytr)                        ← 仅 train 训练
teacher_logits = teacher.logits(Xtr_s)
   │
   ├─ hard   student ← train_hard(Xtr_s, ytr)                    ┐
   ├─ pseudo student ← train_hard(Xtr_s, teacher.predict)        ├ 独立同 seed rng
   └─ distill student ← DistilFuse.distill(Xtr_s, teacher_logits) ┘
   │
   ▼
BenchmarkRecord × {teacher, hard, pseudo, distill} → aggregate → rel_error_drop
```

## 5. 确定性保障

- `set_all(seed)` 在 demo 入口一次性设齐所有随机源。
- 每个 student 用**独立但同 seed** 的 `RandomState`，保证公平对照、可逐位复现。
- `_verify_determinism`：同 seed 两次 `run_once`，核心指标 `max|Δ| = 0`（见 `benchmark.json.determinism`）。

## 6. 防泄漏要点

| 环节 | 措施 |
|------|------|
| 标准化 | `StandardScaler` 只在训练集 `fit`，再 `transform` train/test |
| teacher | 仅在训练集 `fit`，logits 仅来自训练集 |
| student 对照 | 三个 student 用独立同 seed rng，差异只来自训练目标（硬/伪/软） |
| 时间序 | 合成与切分均 `stratify=y`，无随机 shuffle 泄漏 |

## 7. 依赖与降级

- **Tier-0（默认）**：`scikit-learn`（teacher）+ `numpy`/`scipy`（student / 统计）。
- **Tier-1 离线兜底**：`data/synthetic.py` 纯 numpy 多模高斯路径；`AVAILABLE_SKLEARN_TEACHER` 探测缺失时 `run_demo` 显式报错而非伪造。
- **锁定**：`requirements.lock.txt` 由 `pip freeze` 生成，CI 矩阵 3.12 / 3.13 复现。
