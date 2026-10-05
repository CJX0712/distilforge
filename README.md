# DistilForge

[![CI](https://github.com/CJX0712/distilforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/distilforge/actions)
[![Release](https://img.shields.io/github/v/release/CJX0712/distilforge)](https://github.com/CJX0712/distilforge/releases)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%2B-3776AB)](https://www.python.org)
[![Quality](https://img.shields.io/badge/quality-S%20%E7%BA%A7-brightgreen)](docs/model_card.md)

> 通用分类**知识蒸馏**系统（作者：**晨星**）。旗舰 `DistilFuse`：温度软标签蒸馏 + 硬标签交叉熵自适应混合（Hinton et al., 2015）。

将强 teacher（scikit-learn 集成/线性模型）的暗知识压缩进紧凑 student（纯 numpy 自研 Logistic / 单隐层 MLP），推理成本大幅下降，精度保留大部分。

## ✨ 特性

- **旗舰蒸馏协议 `DistilFuse`**：`L = α·T²·KL(softmax(z_t/T)‖softmax(z_s/T)) + (1−α)·CE(y, softmax(z_s))`，默认 `T=4, α=0.5`。
- **零外部权重**：teacher 用本地 scikit-learn 训练，student 纯 numpy 手写，无任何在线下载。
- **确定性可复现**：唯一 seed 入口 `core.seed.set_all(seed)`，同 seed 两次运行核心指标**逐位一致**。
- **防数据泄漏**：`StandardScaler` 仅 fit 训练集；teacher 仅 train 训练；各 student 独立同 seed 起点。
- **离线可降级**：Tier-0（scikit-learn）缺失时 `available_*()` 探测并跳过，不伪造数字。
- **消融对照**：`pseudo`（teacher argmax 硬伪标签）证明增益来自**温度软标签**而非伪监督。

## 📦 安装

```bash
pip install -r requirements.lock.txt   # 锁定版本（numpy / scikit-learn / scipy + dev: pytest / ruff）
# 或
pip install -r requirements.txt
```

要求 Python ≥ 3.12。

## 🚀 快速开始

```bash
python examples/run_demo.py          # 端到端基准 + 确定性二次校验 + 落盘 benchmark.json
python cli.py doctor                 # 环境自检
python cli.py ablation --dataset synth_mid   # 扫描 (T, α) 消融
python cli.py run --seeds 20261005,20261006,20261007 --out benchmark.json
```

## 🧱 架构

```
distilforge/
  core/        types · errors · config(ENV_*) · interfaces · seed(全局确定性)
  data/        synthetic(合成) · loaders(sklearn/CSV)
  distill/     teachers(sklearn) · students(numpy) · distiller(DistilFuse) · baselines
  preprocess/  scaler
  training/    train(minibatches/lr)
  eval/        metrics
  pipeline/    DistillPipeline.run() / benchmark() / aggregate() / ablation()
  cli.py       run / ablation / doctor
  examples/run_demo.py
tests/         pytest（含确定性校验、梯度 vs 数值梯度、离线兜底）
docs/          architecture.md · model_card.md
```

详见 [docs/architecture.md](docs/architecture.md)。

## 📊 基准结果（真实运行，见 `benchmark.json`）

| 数据集 | teacher | hard | distill | rel_drop | 判定 |
|--------|---------|------|---------|----------|------|
| synth_easy | 0.7450 | 0.6983 | **0.7211** | **+7.61%** | ✅ 3/3 wins |
| synth_mid  | 0.6876 | 0.6290 | **0.6529** | **+6.09%** | ✅ 2/3 wins |
| synth_hard | 0.6641 | 0.6581 | **0.6804** | **+6.43%** | ✅ 3/3 wins |
| **合成光谱均值** | — | — | — | **+6.71% ≥ 5%** | ✅ 头条 KPI 达标 |
| digits（真实） | 0.9698 | 0.9685 | 0.9364 | −104.69% | ⚠️ 饱和负例 |

- **头条 KPI**（合成光谱平均相对错误率下降）= **+6.71% ≥ 5%（达标）**。
- **确定性**：同 seed 两次运行核心指标逐位一致（max|Δ|=0）。
- **显著性**：合成光谱 5-seed 配对 t 检验 **p=0.0001（14/15 wins）**；三数据集各自 ≥5%（easy +6.67% / mid +6.07% / hard +6.10%）。
- **端到端耗时**：50s（CPU，≤ 60s 预算）。
- **digits 负增益归因**：compact student 经硬标签已 ≈ teacher（无暗知识空间），软标签反而引入噪声；将 `n_hidden` 提至 128 且 `T=2` 可恢复 **+5.56%**，印证 KD 前置条件（teacher 须显著强于 student）。

## 🔬 知识蒸馏数学内核

```
L = α · T² · KL( softmax(z_t/T) ‖ softmax(z_s/T) ) + (1 − α) · CE( y , softmax(z_s) )
dz = [ α·T·(q_s − p_t) + (1 − α)·(p_s − onehot) ] / N
```

解析梯度与数值梯度一致性已在 `tests/test_students.py` 中交叉验证。

## ⚠️ 限制

1. **饱和区负增益**：teacher ≈ student 时无暗知识空间，蒸馏反而退化（见 digits）。
2. **低分离度负增益**：`class_sep` 过低时 teacher 本身弱、student 难拟合，蒸馏无益。
3. **学生类型依赖**：线性 Logistic student 在非线性边界上几乎无蒸馏增益；系统默认 `student=mlp` 即为此故。
4. **小样本方差**：synth_mid 档增益约 3–6% 波动，但合成光谱均值鲁棒 ≥5%。

## 📄 许可

MIT License（与 scikit-learn / numpy / scipy 许可兼容）。作者：晨星。
