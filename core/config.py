"""配置：ENV_DISTILFORGE_* 覆盖 + schema 校验。"""

import os
from dataclasses import dataclass
from typing import Callable

from .errors import ConfigError

_STUDENT_TYPES = ("logistic", "mlp")
_TEACHER_TYPES = ("random_forest", "gradient_boosting", "mlp", "logistic")


@dataclass
class DistilForgeConfig:
    # 蒸馏温度：软化 teacher/student 输出，暴露类间结构信息（Hinton 2015 默认 4）
    temperature: float = 4.0
    # 软标签损失权重：L = alpha * T^2 * KL(soft_t || soft_s) + (1-alpha) * CE(hard, s)
    alpha: float = 0.5
    # student 梯度下降学习率
    lr: float = 0.1
    n_epochs: int = 400  # MLP student 收敛所需轮次（LR 可适当减少，见 docs）
    batch_size: int = 128
    weight_decay: float = 1e-4
    student_type: str = "mlp"  # logistic | mlp（KD 暗知识增益主要来自过参数化非线性 student）
    teacher_type: str = "random_forest"  # random_forest | gradient_boosting | mlp | logistic
    n_hidden: int = 32  # 仅 mlp student 使用
    seed: int = 20261005
    significance: float = 0.05

    @classmethod
    def from_env(cls) -> "DistilForgeConfig":
        def _get(key: str, cast: Callable, default):
            v = os.environ.get(f"DISTILFORGE_{key}")
            if v is None:
                return default
            try:
                return cast(v)
            except ValueError as exc:
                raise ConfigError(f"invalid DISTILFORGE_{key}={v!r}: {exc}") from exc

        return cls(
            temperature=_get("TEMPERATURE", float, cls.temperature),
            alpha=_get("ALPHA", float, cls.alpha),
            lr=_get("LR", float, cls.lr),
            n_epochs=_get("N_EPOCHS", int, cls.n_epochs),
            batch_size=_get("BATCH_SIZE", int, cls.batch_size),
            weight_decay=_get("WEIGHT_DECAY", float, cls.weight_decay),
            student_type=_get("STUDENT_TYPE", str, cls.student_type),
            teacher_type=_get("TEACHER_TYPE", str, cls.teacher_type),
            n_hidden=_get("N_HIDDEN", int, cls.n_hidden),
            seed=_get("SEED", int, cls.seed),
            significance=_get("SIGNIFICANCE", float, cls.significance),
        )

    def validate(self) -> None:
        if not (self.temperature > 0):
            raise ConfigError("temperature must be > 0")
        if not (0.0 <= self.alpha <= 1.0):
            raise ConfigError("alpha must be in [0, 1]")
        if not (self.lr > 0):
            raise ConfigError("lr must be > 0")
        if self.n_epochs <= 0:
            raise ConfigError("n_epochs must be > 0")
        if self.batch_size <= 0:
            raise ConfigError("batch_size must be > 0")
        if self.student_type not in _STUDENT_TYPES:
            raise ConfigError(f"student_type must be one of {_STUDENT_TYPES}")
        if self.teacher_type not in _TEACHER_TYPES:
            raise ConfigError(f"teacher_type must be one of {_TEACHER_TYPES}")
        if self.n_hidden <= 0:
            raise ConfigError("n_hidden must be > 0")
