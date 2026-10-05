"""基线训练：与旗舰对照，隔离 KD 增益来源。

- train_hard   : student 仅用真实硬标签训练（无 teacher）—— 主强基线
- train_pseudo : student 用 teacher 的 argmax 硬伪标签训练（无 soft 信息）
                 用于证明 DistilFuse 的增益来自"温度软标签"，而非单纯"伪标签监督"
"""

import numpy as np

from core.config import DistilForgeConfig


def train_hard(student, X: np.ndarray, y: np.ndarray, config: DistilForgeConfig, rng):
    return student.fit_hard(X, y, config, rng)


def train_pseudo(student, X: np.ndarray, teacher, y: np.ndarray, config: DistilForgeConfig, rng):
    pseudo = teacher.predict(X)
    return student.fit_hard(X, pseudo, config, rng)
