"""DistilFuse —— 旗舰蒸馏协议。

封装"温度软标签蒸馏 + 硬标签 CE 自适应混合"的训练编排。
- alpha 控制软/硬损失权重（默认 0.5）
- T 控制软化强度（默认 4，Hinton 2015）
- T^2 补偿温度缩放带来的梯度幅度变化（标准 KD 实现）
"""

import numpy as np

from core.config import DistilForgeConfig


class DistilFuse:
    def __init__(self, config: DistilForgeConfig):
        config.validate()
        self.config = config

    def distill(self, student, X: np.ndarray, teacher_logits: np.ndarray, y: np.ndarray, rng):
        """用 teacher 软标签 + 真实硬标签蒸馏训练 student，返回训练后的 student。"""
        return student.fit_distill(X, teacher_logits, y, self.config, rng)
