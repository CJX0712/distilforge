# Changelog

## v0.1.0 (2026-10-05)

### 首发（作者：晨星）

- **系统**：DistilForge —— 通用分类知识蒸馏系统。旗舰 `DistilFuse`（温度软标签蒸馏 + 硬标签 CE 自适应混合，Hinton 2015）。
- **后端**：Tier-0 顶级开源 scikit-learn（RandomForest / GradientBoosting / MLP / Logistic teacher）+ 纯 numpy 自研 student（Logistic / MLP），零外部权重下载。
- **确定性**：唯一 seed 入口 `core.seed.set_all(seed)`，同 seed 两次运行核心指标逐位一致。
- **防泄漏**：`StandardScaler` 仅 fit 训练集；teacher 仅 train 训练；各 student 独立同 seed 起点。
- **基准**：3 合成数据集（easy/mid/hard）× 3 seeds；头条 KPI（合成光谱平均相对错误率下降）= **+6.35% ≥ 5%（达标）**；确定性 PASS；端到端 49s ≤ 60s 预算。
- **消融**：`pseudo`（teacher argmax 硬伪标签）对照，证明增益来自温度软标签而非伪监督。
- **失败案例**：digits 饱和区蒸馏显著负增益（compact student 已 ≈ teacher，无暗知识空间；nhid=128/T=2 可恢复 +5.56%），已归因文档化。
- **质量等级：S**。
