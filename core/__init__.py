"""DistilForge —— 知识蒸馏通用分类系统（作者：晨星）。

core 层：类型 / 错误码 / 配置 / 接口协议 / 全局确定性种子。
调用单向无环：cli -> pipeline -> {data, preprocess, training, distill, eval} -> core。
"""

from .config import DistilForgeConfig
from .errors import (
    ConfigError,
    DataError,
    DependencyError,
    DistilForgeError,
    ModelError,
    PipelineError,
)
from .seed import get_seed, set_all
from .types import AggregatedResult, BenchmarkRecord, DatasetSpec

__all__ = [
    "DatasetSpec",
    "BenchmarkRecord",
    "AggregatedResult",
    "DistilForgeError",
    "ConfigError",
    "DataError",
    "ModelError",
    "PipelineError",
    "DependencyError",
    "DistilForgeConfig",
    "set_all",
    "get_seed",
]
