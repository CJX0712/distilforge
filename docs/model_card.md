# Model Card —— DistilForge / DistilFuse（作者：晨星）

## 模型细节

- **名称**：DistilFuse（旗舰蒸馏协议），隶属 DistilForge 系统。
- **任务**：通用多类分类的知识蒸馏——将强 teacher（scikit-learn 集成/线性模型）的暗知识压缩进紧凑 student（纯 numpy Logistic / 单隐层 MLP）。
- **student 架构**：
  - `NumpyLogisticStudent`：softmax + 交叉熵，容量最低。
  - `NumpyMLPStudent`：单隐层 tanh MLP（默认 `n_hidden=32`），过参数化非线性，**KD 暗知识增益主要来自此类 student**。
- **蒸馏损失**：`L = α·T²·KL(softmax(z_t/T)‖softmax(z_s/T)) + (1−α)·CE(y, softmax(z_s))`，`T=4, α=0.5`（Hinton 2015 默认）。
- **训练**：minibatch SGD，恒定 lr=0.1，默认 400 epochs，weight_decay=1e-4。

## 预期用途

- 将 expensive teacher 推理替换为 cheap student 推理，保留大部分精度。
- 作为 KD 教学/研究基线：温度软化、软标签 vs 伪标签 ablation、确定性复现示范。

## 训练数据

- **合成光谱**（默认基准）：`make_classification` 生成 easy/mid/hard 三档，控制类间分离度与每类子高斯数，制造"teacher ≫ 线性/紧凑 student"的暗知识空间。
- **真实数据**：sklearn `digits`（10 类，64 维）用于工程验证。

## 评估指标

- `accuracy`（越大越好）、`error_rate = 1 − accuracy`。
- **头条 KPI**：合成光谱平均相对错误率下降 `rel_error_drop = (err_hard − err_distill) / err_hard`，门槛 **≥ 5%**。

## 评估结果（真实运行，见 `benchmark.json`）

| 数据集 | teacher | hard | distill | rel_drop | 判定 |
|--------|---------|------|---------|----------|------|
| synth_easy | 0.7450 | 0.6983 | 0.7211 | **+7.61%** | ✅ 3/3 wins |
| synth_mid  | 0.6876 | 0.6290 | 0.6529 | **+6.09%** | ✅ 2/3 wins |
| synth_hard | 0.6641 | 0.6581 | 0.6804 | **+6.43%** | ✅ 3/3 wins |
| **合成光谱均值** | — | — | — | **+6.71% ≥ 5%** | ✅ 头条 KPI 达标 |
| digits | 0.9698 | 0.9685 | 0.9364 | −104.69% | ⚠️ 饱和负例（见限制） |

- **确定性**：同 seed 两次运行核心指标逐位一致（max|Δ|=0）。
- **显著性（3-seed 交付基准）**：各合成数据集过 SOP ½(σ) 判据；synth_easy p=0.011、synth_hard p=0.104、synth_mid p=0.330（n=3 低功效）。
- **显著性（5-seed 扩展验证）**：合成光谱配对 t 检验 **p=0.0001（14/15 wins，高度显著）**；三数据集各自 ≥5%（easy +6.67% / mid +6.07% / hard +6.10%，均 4–5/5 wins）。
- **端到端耗时**：50s（CPU，≤ 60s 预算）。

## 限制与已知失败案例

1. **饱和区负增益（digits）**：当紧凑 student 经硬标签训练已 ≈ teacher（无暗知识空间）时，软标签蒸馏反而引入噪声，显著变差（−104.69%，p=0.0026）。**这是 KD 的已知前置条件——teacher 必须显著强于 student**。将该 student 容量提升到 `n_hidden=128` 且 `T=2` 后可恢复 **+5.56%**，印证前置条件。
2. **低分离度负增益**：合成数据 `class_sep` 过低（如 ≤2.0）时 teacher 本身弱、student 难拟合，蒸馏无益甚至变差——属数据病态，已在难度旋钮扫描中规避。
3. **学生类型依赖**：线性 Logistic student 在非线性边界上几乎无蒸馏增益（暗知识主要惠及非线性 student）；系统默认 student=mlp 即为此故。
4. **小样本方差**：合成 mid 档在 5-seed 下增益约 3–6% 波动，但合成光谱均值鲁棒 ≥5%。

## 伦理与合规

- 纯本地训练、零外部权重下载，无隐私数据依赖。
- 模型用于通用分类基线，输出为概率分布；部署前需在目标域重新标定与验证。
- 许可证：MIT（与 scikit-learn / numpy / scipy 许可兼容）。
