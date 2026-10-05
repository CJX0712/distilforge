"""核心数据类型（dataclass）。

统一语义约定：
- accuracy：越大越好
- error_rate = 1 - accuracy
- rel_error_drop_vs_hard = (err_hard - err_distill) / err_hard  （>0 表示蒸馏优于硬标签基线）
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetSpec:
    name: str
    n_classes: int
    n_features: int
    n_train: int
    n_test: int
    difficulty: float = 1.0  # 类间分离度旋钮：值越小边界越模糊、越难
    clusters_per_class: int = 3  # 每类子高斯数：>1 制造非线性决策边界（让线性 student 明显弱）
    class_sep: Optional[float] = (
        None  # sklearn make_classification 的 class_sep；显式给定则覆盖 difficulty 推导
    )
    seed: int = 20261005
    source: str = "synthetic"  # synthetic | sklearn | csv


@dataclass
class BenchmarkRecord:
    dataset: str
    method: str
    seed: int
    accuracy: float
    error_rate: float
    rel_error_drop_vs_hard: Optional[float] = None
    elapsed_sec: float = 0.0


@dataclass
class AggregatedResult:
    dataset: str
    method: str
    mean_accuracy: float
    std_accuracy: float
    mean_error: float
    std_error: float
    mean_rel_drop: Optional[float] = None
    std_rel_drop: Optional[float] = None
